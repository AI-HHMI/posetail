#!/usr/bin/env python3
"""
Example client for the TrackerEncoder inference server.

Loads a trial directory (same layout as inference_video.py expects), sends a single
window of n_frames to the server, and prints summary stats for each output key.

Usage:
    python server/example_client.py \
        --trial-path /path/to/session/trial/ \
        --server-url http://localhost:8000 \
        --start-frame 0
"""

import argparse
import io
import json
import os
import sys

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_video import (
    build_video_readers, crop_camera_group_to_queries,
    load_camera_group_from_metadata, load_trial,
)


def cam_dict_to_json(cam_dict):
    """Serialize a formatted camera dict (with torch tensors) to plain Python lists."""
    return {
        'name': cam_dict['name'],
        'type': cam_dict['type'],
        'mat':    cam_dict['mat'].tolist(),
        'dist':   cam_dict['dist'].tolist(),
        'ext':    cam_dict['ext'].tolist(),
        'offset': cam_dict['offset'].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Example client for the TrackerEncoder server')
    parser.add_argument(
        '--trial-path', required=True,
        help='Trial directory containing metadata.yaml, pose3d.npz, and img/ or vid/',
    )
    parser.add_argument(
        '--server-url', default='http://localhost:8000',
        help='Base URL of the running server (default: http://localhost:8000)',
    )
    parser.add_argument(
        '--start-frame', type=int, default=0,
        help='First frame index to use for both query points and image window (default: 0)',
    )
    args = parser.parse_args()

    # Discover model parameters from the server
    resp = requests.get(f'{args.server_url}/info')
    resp.raise_for_status()
    info = resp.json()
    print(f'Server info: {info}')
    n_frames = info['n_frames']

    # Load trial: metadata path, per-camera video/image paths, 3D query points
    trial = load_trial(args.trial_path, start_frame=args.start_frame)
    metadata_path = trial['metadata_path']
    video_paths = trial['video_paths']
    query_points_3d = trial['query_points']
    print(f'Query points: {query_points_3d.shape}')

    # Load camera group as formatted dicts
    camera_group = load_camera_group_from_metadata(metadata_path, device='cpu')
    cam_names = [cam['name'] for cam in camera_group]

    image_size = info['image_size']

    # Crop each camera to the bounding box of the projected query points, with
    # padding=20 so points have room to move out of their starting positions.
    # Mirrors inference_video.run_tracker_encoder_on_videos lines 339-344.
    camera_group, crop_boxes = crop_camera_group_to_queries(
        camera_group, query_points_3d.unsqueeze(0),
        min_crop_dim=image_size, padding=20,
    )

    # Serialize the post-crop camera group (offset/size already updated by crop)
    cameras_json = [cam_dict_to_json(cam) for cam in camera_group]

    # Open readers for each camera
    readers, reader_lengths = build_video_readers(video_paths)
    print(f'Cameras: {cam_names}  |  frames available per camera: {reader_lengths}')
    print(f'Reading {n_frames} frames starting at frame {args.start_frame} (cropped per camera)')

    # Build multipart file list: one PNG per (camera, frame), with crop applied
    # Filename convention: <cam_name>__<frame_idx>.png
    files = []
    for cam_idx, (reader, cam_name) in enumerate(zip(readers, cam_names)):
        frames = reader[args.start_frame: args.start_frame + n_frames]
        if hasattr(frames, 'asnumpy'):
            frames = frames.asnumpy()

        x1, y1, x2, y2 = crop_boxes[cam_idx].tolist()
        for frame_idx in range(len(frames)):
            frame_rgb = frames[frame_idx][y1:y2, x1:x2]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode('.png', frame_bgr)
            if not ok:
                raise RuntimeError(f'Failed to encode frame {frame_idx} for camera {cam_name}')
            filename = f'{cam_name}__{str(frame_idx).zfill(6)}.png'
            files.append(('images', (filename, buf.tobytes(), 'image/png')))

    meta = {
        'cameras': cameras_json,
        'coords': query_points_3d.tolist(),
    }

    print(f'Sending {len(files)} images and {len(meta["coords"])} query points to {args.server_url}/predict ...')
    resp = requests.post(
        f'{args.server_url}/predict',
        data={'metadata': json.dumps(meta)},
        files=files,
    )

    if resp.status_code != 200:
        print(f'Error {resp.status_code}: {resp.text}')
        sys.exit(1)

    result = np.load(io.BytesIO(resp.content))

    print('\n--- Prediction results ---')
    for key in result.files:
        arr = result[key]
        nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        print(f'  {key}: shape={arr.shape}  dtype={arr.dtype}  nans={nan_count}')


if __name__ == '__main__':
    main()
