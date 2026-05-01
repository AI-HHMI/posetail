"""
FastAPI server exposing the PyTorch TrackerEncoder model for 3D pose tracking
over a 16-frame clip.

Request: multipart/form-data
  - metadata (JSON string): cameras + query_coords + optional query_times
  - cam_<i>_frames (List[UploadFile]): exactly 16 PNG/JPEG files per camera

Response: JSON with all 13 model outputs as nested arrays plus a `shapes` map.

Run:
    pixi run python server.py [--config-path ...] [--checkpoint-path ...] \\
                              [--host 0.0.0.0] [--port 8000] [--device cuda:0]
"""
import argparse
import json
import os
import re
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from pydantic import BaseModel, ValidationError, field_validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_utils import load_config
from posetail.posetail.tracker_encoder import TrackerEncoder


IMAGE_SIZE = 256
N_FRAMES   = 16

DEFAULT_RUN_DIR = (
    "/groups/karashchuk/home/karashchukl/results/posetail-test-vjepa/"
    "wandb/run-20260410_131633-lwauuvci/files"
)
DEFAULT_CONFIG_PATH     = os.path.join(DEFAULT_RUN_DIR, "config.toml")
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_RUN_DIR, "checkpoints",
                                        "checkpoint_00552960.pth")

CAM_FRAMES_RE = re.compile(r"^cam_(\d+)_frames$")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class CameraSpec(BaseModel):
    name:   str
    ext:    List[List[float]]
    mat:    List[List[float]]
    dist:   List[float]
    offset: List[float]

    @field_validator('ext')
    @classmethod
    def _ext_shape(cls, v):
        if len(v) != 4 or any(len(r) != 4 for r in v):
            raise ValueError('ext must be 4x4')
        return v

    @field_validator('mat')
    @classmethod
    def _mat_shape(cls, v):
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError('mat must be 3x3')
        return v

    @field_validator('dist')
    @classmethod
    def _dist_shape(cls, v):
        if len(v) != 5:
            raise ValueError('dist must have length 5')
        return v

    @field_validator('offset')
    @classmethod
    def _offset_shape(cls, v):
        if len(v) != 2:
            raise ValueError('offset must have length 2')
        return v


class InferRequest(BaseModel):
    cameras:      List[CameraSpec]
    query_coords: List[List[float]]
    query_times:  Optional[List[int]] = None

    @field_validator('query_coords')
    @classmethod
    def _coords_shape(cls, v):
        if len(v) == 0:
            raise ValueError('query_coords must be non-empty')
        if any(len(p) != 3 for p in v):
            raise ValueError('each query coord must have length 3')
        return v


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Mirrors verify_onnx_parity.py:93-98."""
    print(f'loading config: {config_path}', flush=True)
    cfg = load_config(config_path)
    print(f'building TrackerEncoder...', flush=True)
    model = TrackerEncoder(**cfg.model).eval()
    print(f'loading checkpoint: {checkpoint_path}', flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.scene_encoder.encoder.use_activation_checkpointing = False
    model.to(device)
    print(f'model loaded on {device}', flush=True)
    return model


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def decode_image(blob: bytes) -> np.ndarray:
    """Decode bytes (PNG/JPEG) to RGB uint8 [H, W, 3]."""
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='Failed to decode image')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_and_pad(frames: np.ndarray, mat: np.ndarray, offset: np.ndarray):
    """Resize so the longest side is IMAGE_SIZE, then zero-pad bottom/right.
    Adjust intrinsics accordingly. Matches PosetailInferenceDataset.resize_camera_group
    + PadToSize.

    Args:
        frames: [T, H, W, 3] uint8
        mat:    [3, 3] float
        offset: [2]    float

    Returns:
        frames_padded: [T, IMAGE_SIZE, IMAGE_SIZE, 3] uint8
        mat_new:       [3, 3] float32
        offset_new:    [2]    float32
    """
    T, H, W, _ = frames.shape
    scale = float(IMAGE_SIZE) / max(H, W)
    H_new = int(round(H * scale))
    W_new = int(round(W * scale))

    if (H_new, W_new) != (H, W):
        resized = np.empty((T, H_new, W_new, 3), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames[i], (W_new, H_new),
                                    interpolation=cv2.INTER_LINEAR)
    else:
        resized = frames

    pad_h = IMAGE_SIZE - H_new
    pad_w = IMAGE_SIZE - W_new
    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((T, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        padded[:, :H_new, :W_new, :] = resized
    else:
        padded = resized

    mat_new = mat * scale
    mat_new[2, 2] = 1.0
    offset_new = offset * scale

    return padded, mat_new.astype(np.float32), offset_new.astype(np.float32)


def build_camera_dict(spec: CameraSpec, mat_new: np.ndarray, offset_new: np.ndarray,
                      device: torch.device) -> dict:
    """Build a camera_group dict matching format_camera (train_utils.py:211).
    center and ext_inv are derived from ext (verify_onnx_parity.py:144-148)."""
    ext = torch.tensor(spec.ext, dtype=torch.float32, device=device)
    cam = {
        'name':   spec.name,
        'type':   'pinhole',
        'ext':    ext,
        'mat':    torch.tensor(mat_new, dtype=torch.float32, device=device),
        'dist':   torch.tensor(spec.dist, dtype=torch.float32, device=device),
        'offset': torch.tensor(offset_new, dtype=torch.float32, device=device),
        'size':   torch.tensor([IMAGE_SIZE, IMAGE_SIZE], dtype=torch.int32, device=device),
    }
    cam['ext_inv'] = torch.linalg.inv(ext)
    R = ext[:3, :3]
    t = ext[:3, 3]
    cam['center'] = -R.T @ t
    return cam


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------

def serialize_outputs(out: dict) -> dict:
    outputs = {}
    shapes  = {}
    for k, v in out.items():
        if v is None:
            outputs[k] = None
            shapes[k]  = None
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            outputs[k] = arr.tolist()
            shapes[k]  = list(arr.shape)
        else:
            outputs[k] = v
            shapes[k]  = None
    return outputs, shapes


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title='posetail tracker server')


@app.get('/health')
def health():
    return {
        'status':   'ok',
        'device':   str(app.state.device),
        'image_size': IMAGE_SIZE,
        'n_frames':   N_FRAMES,
    }


@app.post('/infer')
async def infer(request: Request, metadata: str = Form(...)):
    # Parse + validate metadata
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f'Invalid metadata JSON: {e}')
    try:
        req = InferRequest(**meta_dict)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f'metadata validation: {e.errors()}')

    n_cams = len(req.cameras)
    n_pts  = len(req.query_coords)

    # Default query_times to zeros, validate range
    if req.query_times is None:
        qtimes_list = [0] * n_pts
    else:
        if len(req.query_times) != n_pts:
            raise HTTPException(status_code=400,
                detail=f'query_times length ({len(req.query_times)}) != n_points ({n_pts})')
        if any((t < 0 or t >= N_FRAMES) for t in req.query_times):
            raise HTTPException(status_code=400,
                detail=f'query_times must be in [0, {N_FRAMES - 1}]')
        qtimes_list = list(req.query_times)

    # Pull files; group by cam_<i>_frames
    form = await request.form()
    cam_files: dict[int, list] = {}
    for key in form.keys():
        m = CAM_FRAMES_RE.match(key)
        if not m:
            continue
        idx = int(m.group(1))
        files = form.getlist(key)
        cam_files[idx] = files

    if set(cam_files.keys()) != set(range(n_cams)):
        raise HTTPException(status_code=400,
            detail=f'expected cam_0_frames..cam_{n_cams-1}_frames, '
                   f'got {sorted(cam_files.keys())}')

    device = app.state.device
    model  = app.state.model

    # Decode + preprocess each camera
    views        = []
    camera_group = []
    for i in range(n_cams):
        files = cam_files[i]
        if len(files) != N_FRAMES:
            raise HTTPException(status_code=400,
                detail=f'cam_{i}_frames has {len(files)} frames, expected {N_FRAMES}')

        frames = []
        for f in files:
            blob = await f.read()
            frames.append(decode_image(blob))

        shapes_orig = {(im.shape[0], im.shape[1]) for im in frames}
        if len(shapes_orig) != 1:
            raise HTTPException(status_code=400,
                detail=f'cam_{i}_frames has inconsistent frame sizes: {shapes_orig}')

        frames_arr = np.stack(frames, axis=0)
        spec = req.cameras[i]
        mat_orig    = np.array(spec.mat,    dtype=np.float32)
        offset_orig = np.array(spec.offset, dtype=np.float32)
        frames_pad, mat_new, offset_new = resize_and_pad(frames_arr, mat_orig, offset_orig)

        # [B=1, T, H, W, C] in [0, 1]
        view_t = torch.from_numpy(frames_pad).to(device, dtype=torch.float32) / 255.0
        view_t = view_t.unsqueeze(0)
        views.append(view_t)

        camera_group.append(build_camera_dict(spec, mat_new, offset_new, device))

    coords = torch.tensor(req.query_coords, dtype=torch.float32, device=device).unsqueeze(0)
    qtimes = torch.tensor(qtimes_list, dtype=torch.int32, device=device).unsqueeze(0)

    # Run model
    try:
        with torch.inference_mode():
            out = model(views=views, coords=coords, query_times=qtimes,
                        camera_group=camera_group)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'model inference failed: {e}')

    outputs, shapes = serialize_outputs(out)
    return {
        'outputs':   outputs,
        'shapes':    shapes,
        'n_cameras': n_cams,
        'n_points':  n_pts,
        'n_frames':  N_FRAMES,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config-path',     type=str, default=DEFAULT_CONFIG_PATH)
    p.add_argument('--checkpoint-path', type=str, default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument('--host',            type=str, default='0.0.0.0')
    p.add_argument('--port',            type=int, default=8000)
    p.add_argument('--device',          type=str, default=None,
                   help='torch device string (e.g. cuda:0). Default: cuda if available else cpu')
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = load_model(args.config_path, args.checkpoint_path, device)

    app.state.model  = model
    app.state.device = device

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
