"""
Example invocation:

    python render_video.py \
        --input-npz /path/to/output.npz \
        --output-dir /path/to/rendered/ \
        --conf-threshold 0.5 \
        --marker-size 4 \
        --fps 30

Example with cropping around tracked points:

    python render_video.py \
        --input-npz /path/to/output.npz \
        --output-dir /path/to/rendered/ \
        --crop --crop-padding 80

Reads the output .npz from inference_video.py, loads the corresponding
video/image data, projects predicted 3D keypoints onto each camera view,
and saves rendered videos with overlaid keypoints.
"""

import os
import cv2
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from inference_video import (
    build_video_readers,
    load_camera_group_from_metadata,
)
from posetail.posetail.cube import project_points_torch


def generate_colors(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


def render_keypoints_on_frame(frame, points_2d, conf, conf_threshold, colors, marker_size, vis=None):
    """Draw keypoints on a single frame (BGR, uint8).

    Args:
        frame: (H, W, 3) BGR uint8 image (modified in-place).
        points_2d: (N, 2) projected 2D coordinates.
        conf: (N,) confidence values (probabilities in [0, 1]).
        conf_threshold: minimum confidence/visibility to draw a keypoint.
        colors: list of BGR tuples, one per keypoint.
        marker_size: radius of the circle marker.
        vis: optional (N,) visibility probabilities in [0, 1]; gated like conf.
    """
    n_kpts = points_2d.shape[0]
    h, w = frame.shape[:2]

    for k in range(n_kpts):
        if conf[k] < conf_threshold:
            continue
        if vis is not None and vis[k] < conf_threshold:
            continue
        if not np.all(np.isfinite(points_2d[k])):
            # NaN/inf coords (e.g. behind-camera or undefined projections); skip.
            continue

        x, y = int(round(points_2d[k, 0])), int(round(points_2d[k, 1]))

        if not (0 <= x < w and 0 <= y < h):
            continue

        color = colors[k % len(colors)]
        cv2.circle(frame, (x, y), marker_size, color, -1, lineType=cv2.LINE_AA)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render projected keypoints from inference_video.py output onto videos.')

    parser.add_argument('--input-npz', type=str, required=True,
                        help='Path to the .npz output from inference_video.py')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save rendered videos')
    parser.add_argument('--trial-path', type=str, default=None,
                        help='Optional trial path override; if provided, resolves video paths '
                             'and metadata from this directory instead of the paths stored in the npz')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Minimum confidence to render a keypoint')
    parser.add_argument('--marker-size', type=int, default=4,
                        help='Radius of keypoint markers in pixels')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second for output videos')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='FourCC codec for output videos')
    parser.add_argument('--crop', action='store_true', default=False,
                        help='Crop each camera view around the tracked keypoints')
    parser.add_argument('--crop-padding', type=int, default=50,
                        help='Padding in pixels around the bounding box of projected keypoints')

    return parser.parse_args()


def compute_crop_boxes(p2d, conf_pred, conf_threshold, padding=20):
    """Compute a single crop box per camera across all frames.

    Args:
        p2d: (n_cams, T, N, 2) projected 2D coordinates
        conf_pred: (T, N) confidence values
        conf_threshold: minimum confidence required for a point to influence cropping
        padding: pixels to add around the bounding box

    Returns:
        List of (x1, y1, x2, y2) tuples, one per camera
    """
    n_cams = p2d.shape[0]
    crop_boxes = []

    for cam_idx in range(n_cams):
        pts = p2d[cam_idx]  # (T, N, 2)
        mask = conf_pred >= conf_threshold  # (T, N)
        pts_masked = pts.copy()
        pts_masked[~mask] = np.nan

        pts_flat = pts_masked.reshape(-1, 2)
        valid = np.all(np.isfinite(pts_flat), axis=1)
        pts_valid = pts_flat[valid]

        if len(pts_valid) == 0:
            # No valid points, return None to indicate no cropping
            crop_boxes.append(None)
            continue

        x_min = np.min(pts_valid[:, 0]) - padding
        y_min = np.min(pts_valid[:, 1]) - padding
        x_max = np.max(pts_valid[:, 0]) + padding
        y_max = np.max(pts_valid[:, 1]) + padding

        x1 = int(np.floor(x_min))
        y1 = int(np.floor(y_min))
        x2 = int(np.ceil(x_max))
        y2 = int(np.ceil(y_max))

        width = x2 - x1
        height = y2 - y1
        side = max(width, height)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        x1 = cx - side // 2
        x2 = x1 + side
        y1 = cy - side // 2
        y2 = y1 + side

        crop_boxes.append((x1, y1, x2, y2))

    return crop_boxes


def main():
    args = parse_args()

    # Load inference outputs
    data = np.load(args.input_npz, allow_pickle=True)

    coords_pred = data['coords_pred']       # (1, T, N, R), R=3 (3D) or 2 (2D)
    conf_pred = data['conf_pred']           # (1, T, N) or (1, T, N, 1), logits
    vis_pred = data['vis_pred'] if 'vis_pred' in data else None  # (1, T, N, 1), logits
    frame_numbers = data['frame_numbers']   # (T,)
    video_paths = [str(p) for p in data['video_paths']]
    start_frame = int(data['start_frame']) if 'start_frame' in data else 0
    if 'fps' in data and args.fps == 30.0:
        args.fps = float(data['fps'])

    # 2D vs 3D: 2D predictions are already pixel coords (no projection / metadata).
    is_2d = coords_pred.shape[-1] == 2

    # Override with trial path if provided (recompute video paths from the trial)
    if args.trial_path is not None:
        from inference_video import load_trial
        _mode, metadata_path, _cam_names, video_paths, *_ = load_trial(
            args.trial_path, start_frame=start_frame)
    else:
        metadata_path = data['metadata_path'].item() if 'metadata_path' in data else ''

    # Remove batch dimension
    coords_pred = coords_pred[0]    # (T, N, R)
    conf_pred = conf_pred[0]        # (T, N) or (T, N, 1)
    if conf_pred.ndim == 3:
        conf_pred = conf_pred[..., 0]   # (T, N)

    # conf/vis are stored as logits; map to [0, 1] probabilities so the
    # conf_threshold (default 0.5) is meaningful.
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    conf_pred = _sigmoid(conf_pred)
    if vis_pred is not None:
        vis_pred = vis_pred[0]          # (T, N) or (T, N, 1)
        if vis_pred.ndim == 3:
            vis_pred = vis_pred[..., 0]
        vis_pred = _sigmoid(vis_pred)   # (T, N)

    n_frames = coords_pred.shape[0]
    n_kpts = coords_pred.shape[1]
    n_cams = len(video_paths)

    print(f'Loaded {n_frames} frames, {n_kpts} keypoints, {n_cams} cameras '
          f'({"2D" if is_2d else "3D"})')

    if is_2d:
        # Single (uncalibrated) camera; predictions are pixel coords already.
        cam_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
        p2d = coords_pred[np.newaxis]  # (1, T, N, 2)
    else:
        # Load camera group and project predicted 3D points to each view.
        camera_group = load_camera_group_from_metadata(metadata_path, device='cpu')
        cam_names = [cam['name'] for cam in camera_group]

        if n_cams != len(camera_group):
            raise ValueError(
                f'Number of video paths ({n_cams}) does not match '
                f'number of cameras ({len(camera_group)}) in metadata'
            )

        coords_3d_torch = torch.as_tensor(coords_pred, dtype=torch.float32)
        p2d = project_points_torch(camera_group, coords_3d_torch)  # (n_cams, T, N, 2)
        p2d = p2d.cpu().numpy()

    # Compute crop boxes once for all frames
    if args.crop:
        crop_boxes = compute_crop_boxes(
            p2d,
            conf_pred=conf_pred,
            conf_threshold=args.conf_threshold,
            padding=args.crop_padding,
        )
    else:
        crop_boxes = [None] * n_cams

    # Generate colors for keypoints
    colors = generate_colors(n_kpts)

    # Build readers
    readers, _ = build_video_readers(video_paths)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    render_info = {
        'input_npz': args.input_npz,
        'trial_path': args.trial_path,
        'metadata_path': metadata_path,
        'video_paths': video_paths,
        'conf_threshold': args.conf_threshold,
        'marker_size': args.marker_size,
        'fps': args.fps,
        'codec': args.codec,
        'crop': args.crop,
        'crop_padding': args.crop_padding,
    }
    with open(os.path.join(args.output_dir, 'render_info.json'), 'w') as f:
        json.dump(render_info, f, indent=2)

    # Render each camera view
    for cam_idx in tqdm(range(n_cams), desc='Rendering cameras', unit='cam'):
        cam_name = cam_names[cam_idx]
        reader = readers[cam_idx]

        frame_ids = [int(fn) for fn in frame_numbers]
        if len(frame_ids) == 0:
            print(f'Warning: no frames to render for camera {cam_name}')
            continue
        if min(frame_ids) < 0 or max(frame_ids) >= len(reader):
            raise ValueError(
                f'Frame ids for camera {cam_name} are out of bounds: '
                f'[{min(frame_ids)}, {max(frame_ids)}] vs available [0, {len(reader)-1}]'
            )

        # Read first frame to get dimensions
        first_frame = reader.get_batch([frame_ids[0]])
        if hasattr(first_frame, 'asnumpy'):
            first_frame = first_frame.asnumpy()
        first_frame = first_frame[0]
        h_orig, w_orig = first_frame.shape[:2]

        # Clamp crop box to frame boundaries once
        crop_box = crop_boxes[cam_idx]
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_orig, x2)
            y2 = min(h_orig, y2)
            crop_box = (x1, y1, x2, y2)
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0:
                crop_box = None
                h, w = h_orig, w_orig
        else:
            h, w = h_orig, w_orig

        # Setup video writer
        out_path = os.path.join(args.output_dir, f'{cam_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

        if not writer.isOpened():
            print(f'Warning: could not open video writer for {out_path}')
            continue

        try:
            for t in tqdm(range(n_frames), desc=f'Writing {cam_name}', unit='frames', leave=False):
                frame = reader.get_batch([frame_ids[t]])
                if hasattr(frame, 'asnumpy'):
                    frame = frame.asnumpy()
                frame = frame[0]

                if crop_box is not None:
                    x1, y1, x2, y2 = crop_box
                    frame = frame[y1:y2, x1:x2]

                # Convert RGB to BGR for OpenCV (both readers return RGB)
                frame = frame[:, :, ::-1].copy()

                # Adjust 2D points for crop offset
                pts = p2d[cam_idx, t].copy()  # (N, 2)
                if crop_box is not None:
                    pts[:, 0] -= x1
                    pts[:, 1] -= y1

                conf = conf_pred[t]  # (N,)
                vis = vis_pred[t] if vis_pred is not None else None  # (N,)

                frame = render_keypoints_on_frame(
                    frame, pts, conf,
                    conf_threshold=args.conf_threshold,
                    colors=colors,
                    marker_size=args.marker_size,
                    vis=vis,
                )

                writer.write(frame)
        finally:
            writer.release()
        print(f'Saved rendered video: {out_path}')

    del readers
    print('Done.')


if __name__ == '__main__':
    main()
