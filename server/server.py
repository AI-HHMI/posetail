#!/usr/bin/env python3
"""
FastAPI inference server for TrackerEncoder.

Start with either a wandb run directory or explicit config/checkpoint paths:

    python server/server.py --wandb /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX
    python server/server.py --config files/config.toml --checkpoint files/checkpoints/checkpoint_00010000.pth
"""

import argparse
import asyncio
import io
import json
import os
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_video import load_model_from_base_folder, resize_camera_group
from posetail.posetail.tracker_encoder import TrackerEncoder
from train_utils import load_checkpoint, load_config

_gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = getattr(app.state, 'cli_args', None)
    if args is None:
        raise RuntimeError(
            'Server must be started via `python server/server.py ...`, not directly via uvicorn.'
        )

    device_arg = args.device  # string or None

    if args.wandb:
        model, config, config_path, checkpoint_path = load_model_from_base_folder(
            args.wandb, checkpoint=args.checkpoint_number, device=device_arg
        )
    else:
        config = load_config(args.config)
        if device_arg is None:
            device_arg = config.devices.device if torch.cuda.is_available() else 'cpu'
        checkpoint_dict = load_checkpoint(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=device_arg,
        )
        model = checkpoint_dict['model']
        if not isinstance(model, TrackerEncoder):
            raise RuntimeError(
                f'Loaded model must be a TrackerEncoder, got {type(model).__name__}'
            )
        model.eval()
        config_path = args.config
        checkpoint_path = args.checkpoint

    device = next(model.parameters()).device

    app.state.model = model
    app.state.device = device
    app.state.config_path = str(config_path)
    app.state.checkpoint_path = str(checkpoint_path)
    app.state.n_frames = model.n_frames
    app.state.image_size = model.image_size
    app.state.mode_3d = config.model.get('mode_3d', 'encoder')

    print(
        f'Model loaded | n_frames={model.n_frames} | image_size={model.image_size} | device={device}'
    )
    yield


app = FastAPI(title='TrackerEncoder Server', lifespan=lifespan)


@app.get('/info')
async def info():
    return {
        'n_frames': app.state.n_frames,
        'image_size': app.state.image_size,
        'device': str(app.state.device),
        'config_path': app.state.config_path,
        'checkpoint_path': app.state.checkpoint_path,
        'mode_3d': app.state.mode_3d,
    }


@app.post('/predict')
async def predict(
    metadata: str = Form(...),
    images: list[UploadFile] = File(...),
):
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(400, detail=f'Invalid metadata JSON: {e}')

    if 'cameras' not in meta:
        raise HTTPException(400, detail='metadata must include "cameras"')
    if 'coords' not in meta:
        raise HTTPException(400, detail='metadata must include "coords"')

    cameras_meta = meta['cameras']
    coords_list = meta['coords']
    query_times_list = meta.get('query_times', None)

    model = app.state.model
    device = app.state.device
    n_frames = app.state.n_frames

    if query_times_list is not None and len(query_times_list) != len(coords_list):
        raise HTTPException(
            400,
            detail=(
                f'query_times length {len(query_times_list)} != '
                f'coords length {len(coords_list)}'
            ),
        )

    # Decode and group images by camera name
    cam_frames: dict[str, dict[int, np.ndarray]] = {}
    for upload in images:
        fname = upload.filename or ''
        stem = os.path.splitext(fname)[0]
        parts = stem.split('__', 1)
        if len(parts) != 2:
            raise HTTPException(
                400,
                detail=f'Image filename must be <cam_name>__<frame_idx>.<ext>, got: {fname}',
            )
        cam_name, frame_str = parts
        try:
            frame_idx = int(frame_str)
        except ValueError:
            raise HTTPException(400, detail=f'Frame index must be an integer, got: {frame_str}')

        data = await upload.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail=f'Failed to decode image: {fname}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cam_frames.setdefault(cam_name, {})[frame_idx] = img

    # Build camera group dicts first (needed to compute resize scales before loading views).
    # Image size is read from the actual decoded frames — client need not send 'size'.
    from train_utils import format_orthographic_camera
    camera_group = []
    for cam_info in cameras_meta:
        cam_name = cam_info['name']
        if cam_name not in cam_frames:
            raise HTTPException(400, detail=f'No images uploaded for camera: {cam_name}')
        first_frame = next(iter(cam_frames[cam_name].values()))
        H, W = first_frame.shape[:2]
        cam_type = cam_info.get('type', 'pinhole')
        offset = cam_info.get('offset', [0.0, 0.0])

        if cam_type == 'orthographic':
            camera_group.append(format_orthographic_camera(
                cam_info['L'], cam_name, [W, H], device=device, offset=offset))
            continue

        size = torch.tensor([W, H], dtype=torch.int32, device=device)
        ext = torch.tensor(cam_info['ext'], dtype=torch.float32, device=device)
        mat = torch.tensor(cam_info['mat'], dtype=torch.float32, device=device)
        dist = torch.tensor(cam_info['dist'], dtype=torch.float32, device=device)
        offset = torch.tensor(offset, dtype=torch.float32, device=device)
        ext_inv = torch.linalg.inv(ext)
        R = ext[:3, :3]
        t = ext[:3, 3]
        center = -R.T @ t
        camera_group.append({
            'name': cam_name,
            'type': cam_type,
            'mat': mat,
            'dist': dist,
            'ext': ext,
            'size': size,
            'offset': offset,
            'ext_inv': ext_inv,
            'center': center,
        })

    # Record per-camera scale factors before resize so we can un-scale 2d_pred later
    image_size = app.state.image_size
    scales = [float(image_size) / max(cam['size'].tolist()) for cam in camera_group]

    # Scale so max(H,W) == image_size, matching PosetailDataset / inference_video
    camera_group = resize_camera_group(camera_group, image_size)

    # Build per-camera view tensors, resizing frames to the scaled camera size
    views = []
    for cam_idx, cam_info in enumerate(cameras_meta):
        cam_name = cam_info['name']
        frame_dict = cam_frames[cam_name]
        if len(frame_dict) != n_frames:
            raise HTTPException(
                400,
                detail=f'Camera {cam_name}: expected {n_frames} frames, got {len(frame_dict)}',
            )
        target_wh = tuple(camera_group[cam_idx]['size'].tolist())  # (W, H) for cv2.resize
        frames = np.stack(
            [cv2.resize(frame_dict[i], target_wh) for i in sorted(frame_dict)], axis=0
        )  # (T, H, W, 3)
        view_tensor = (
            torch.from_numpy(frames).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        )
        views.append(view_tensor)

    coords = torch.tensor(coords_list, dtype=torch.float32, device=device).unsqueeze(0)
    query_times = None
    if query_times_list is not None:
        query_times = torch.tensor(
            query_times_list, dtype=torch.int32, device=device
        ).unsqueeze(0)

    async with _gpu_lock:
        with torch.no_grad():
            outputs = model(
                views=views,
                coords=coords,
                camera_group=camera_group,
                query_times=query_times,
            )

    # Un-scale 2d_pred from resized coords back to the client's sent-image resolution.
    # All other output keys are in 3D world space and are unaffected by image resize.
    if outputs.get('2d_pred') is not None:
        scale_t = torch.tensor(scales, device=device).view(-1, 1, 1, 1, 1)
        outputs['2d_pred'] = outputs['2d_pred'] / scale_t

    result = {}
    for key, val in outputs.items():
        if val is None:
            continue
        result[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)

    buf = io.BytesIO()
    np.savez(buf, **result)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type='application/octet-stream',
        headers={'Content-Disposition': 'attachment; filename="predictions.npz"'},
    )


def parse_args():
    parser = argparse.ArgumentParser(description='TrackerEncoder inference server')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--wandb', type=str,
        help='Path to a wandb run directory (same as --base-folder in inference_video.py)',
    )
    group.add_argument(
        '--config', type=str,
        help='Path to config.toml (requires --checkpoint)',
    )
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint .pth (required with --config)')
    parser.add_argument(
        '--checkpoint-number', type=int, default=None,
        help='Checkpoint number to load (only with --wandb; default: latest)',
    )
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device string, e.g. cuda:0 (default: from config, or cpu if CUDA unavailable)',
    )
    args = parser.parse_args()
    if args.config and not args.checkpoint:
        parser.error('--checkpoint is required when using --config')
    if args.checkpoint and not args.config:
        parser.error('--config is required when using --checkpoint')
    if args.checkpoint_number is not None and not args.wandb:
        parser.error('--checkpoint-number is only valid with --wandb')
    return args


if __name__ == '__main__':
    import uvicorn

    args = parse_args()
    app.state.cli_args = args
    uvicorn.run(app, host=args.host, port=args.port)
