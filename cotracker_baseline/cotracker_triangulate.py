"""
Triangulate per-camera CoTracker predictions to 3D.

Reads the per-camera .npz files produced by run_cotracker_inference.py
from each trial's cotracker/ subfolder, triangulates using the calibrated
aniposelib CameraGroup, and saves the result to:

    trial_path/cotracker_3d.npz

Output arrays
-------------
    coords_3d     (T, N, 3)  float32   triangulated world coords (NaN where
                                        fewer than --min-cams cameras visible)
    visibility    (T, N)     bool      True if ≥ min_cams cameras tracked point
    frame_numbers (T,)       int64     absolute frame indices
    valid_mask    (S*K,)     bool      which (subject×kpt) entries are included
                                       (same as in the per-camera files)
    cam_names     (C,)       str       camera names used for triangulation

Example
-------
    python triangulate_cotracker.py \\
        --dataset-path /data/myproject \\
        --split test \\
        --min-cams 2

No GPU required.
"""

import os
import json
import argparse

import yaml
import numpy as np
from tqdm import tqdm

from aniposelib.cameras import CameraGroup, Camera
from posetail.datasets.utils import get_dirs, disassemble_extrinsics


# ── camera loading ────────────────────────────────────────────────────────

def load_anipose_camera_group(metadata_path: str) -> tuple[CameraGroup, list[str]]:
    """
    Load a raw aniposelib CameraGroup from a metadata.yaml file.

    Returns
    -------
    cgroup    : aniposelib CameraGroup
    cam_names : sorted list of camera names (matches the ordering used
                when saving per-camera predictions)
    """
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)

    intrinsics = meta["intrinsic_matrices"]
    extrinsics = meta["extrinsic_matrices"]
    distortions = meta["distortion_matrices"]
    heights = meta["camera_heights"]
    widths = meta["camera_widths"]

    cam_names = list(intrinsics.keys())
    if all(n.isdigit() for n in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)

    cams = []
    for name in cam_names:
        rvec, tvec = disassemble_extrinsics(extrinsics[name])
        cam = Camera(
            matrix=intrinsics[name],
            dist=distortions[name],
            rvec=rvec,
            tvec=tvec,
            name=name,
        )
        cam.set_size((widths[name], heights[name]))
        cams.append(cam)

    return CameraGroup(cams), cam_names


# ── triangulation ─────────────────────────────────────────────────────────

def triangulate_tracks(
    cgroup: CameraGroup,
    tracks_per_cam: list[np.ndarray],    # list of (T, N, 2), one per camera
    vis_per_cam: list[np.ndarray],       # list of (T, N) bool, one per camera
    min_cams: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 2D tracks from multiple cameras into 3D.

    Points are set to NaN where fewer than `min_cams` cameras have a
    visible observation.

    Returns
    -------
    coords_3d  : (T, N, 3) float32
    visibility : (T, N)    bool
    """
    n_cams = len(tracks_per_cam)
    T, N, _ = tracks_per_cam[0].shape

    # Stack and mask invisible detections with NaN
    # tracks_stack : (n_cams, T, N, 2)
    tracks_stack = np.stack(tracks_per_cam, axis=0).astype(np.float32)
    vis_stack    = np.stack(vis_per_cam,    axis=0)  # (n_cams, T, N) bool

    tracks_stack[~vis_stack] = np.nan  # NaN where not visible

    # Count visible cameras per (frame, keypoint)
    n_visible = vis_stack.sum(axis=0)  # (T, N)
    enough    = n_visible >= min_cams  # (T, N) bool

    # Reshape for aniposelib: it expects (n_cams, M, 2) where M = T*N
    # and returns (M, 3).  We triangulate all frames × keypoints at once.
    tracks_2d = tracks_stack.reshape(n_cams, T * N, 2)  # (n_cams, T*N, 2)

    coords_3d_flat = cgroup.triangulate(tracks_2d, undistort=True)  # (T*N, 3)
    coords_3d = coords_3d_flat.reshape(T, N, 3).astype(np.float32)

    # Zero out points that didn't have enough cameras
    coords_3d[~enough] = np.nan

    return coords_3d, enough


# ── per-trial processing ──────────────────────────────────────────────────

def process_trial(
    trial_path: str,
    output_trial_path: str,
    min_cams: int = 2,
    overwrite: bool = False,
):
    os.makedirs(output_trial_path, exist_ok=True)
    out_path = os.path.join(output_trial_path, "cotracker_3d.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"    [skip] already done (--overwrite to redo)")
        return

    cotracker_dir = os.path.join(output_trial_path, "cotracker")
    metadata_path = os.path.join(trial_path, "metadata.yaml")

    if not os.path.isdir(cotracker_dir):
        print(f"    [skip] no cotracker/ folder — run inference first")
        return
    if not os.path.exists(metadata_path):
        print(f"    [skip] missing metadata.yaml")
        return

    # Discover which cameras have prediction files
    pred_files = sorted(
        [f for f in os.listdir(cotracker_dir) if f.endswith(".npz")]
    )
    if len(pred_files) == 0:
        print(f"    [skip] cotracker/ is empty")
        return

    # Load camera group (raw aniposelib, needed for triangulate)
    cgroup, cam_names_all = load_anipose_camera_group(metadata_path)

    # Match prediction files to cameras (order matters)
    cam_names_pred = [os.path.splitext(f)[0] for f in pred_files]

    # Only keep cameras that are in both the metadata and the predictions
    cam_names_used = [n for n in cam_names_all if n in cam_names_pred]
    if len(cam_names_used) < min_cams:
        print(
            f"    [skip] only {len(cam_names_used)} camera(s) with predictions, "
            f"need ≥ {min_cams}"
        )
        return

    # Subset aniposelib CameraGroup to the cameras we have predictions for
    cgroup_sub = cgroup.subset_cameras_names(cam_names_used)

    # Load per-camera predictions and verify consistency
    tracks_per_cam = []
    vis_per_cam    = []
    frame_numbers  = None
    valid_mask     = None

    for cam_name in cam_names_used:
        npz = np.load(os.path.join(cotracker_dir, f"{cam_name}.npz"))

        tracks     = npz["tracks"]      # (T, N, 2)
        visibility = npz["visibility"]  # (T, N) bool

        if frame_numbers is None:
            frame_numbers = npz["frame_numbers"]
            valid_mask    = npz["valid_mask"]
        else:
            # Sanity check: all cameras should cover the same frames
            if not np.array_equal(frame_numbers, npz["frame_numbers"]):
                print(
                    f"    [warn] frame_numbers mismatch for camera {cam_name} — "
                    f"using intersection"
                )
                common = np.intersect1d(frame_numbers, npz["frame_numbers"])
                idx_a  = np.isin(frame_numbers, common)
                idx_b  = np.isin(npz["frame_numbers"], common)
                # Trim already-loaded cameras too
                tracks_per_cam = [t[idx_a] for t in tracks_per_cam]
                vis_per_cam    = [v[idx_a] for v in vis_per_cam]
                tracks     = tracks[idx_b]
                visibility = visibility[idx_b]
                frame_numbers = common

        tracks_per_cam.append(tracks.astype(np.float32))
        vis_per_cam.append(visibility.astype(bool))

    # Triangulate
    coords_3d, visibility_3d = triangulate_tracks(
        cgroup_sub, tracks_per_cam, vis_per_cam, min_cams=min_cams
    )

    np.savez(
        out_path,
        coords_3d=coords_3d,           # (T, N, 3) float32
        visibility=visibility_3d,      # (T, N) bool
        frame_numbers=frame_numbers,   # (T,) int64
        valid_mask=valid_mask,         # (S*K,) bool — same as inference
        cam_names=np.array(cam_names_used, dtype=str),
        min_cams_used=min_cams,
    )

    n_finite = np.isfinite(coords_3d).all(axis=-1).sum()
    print(
        f"    saved: {tuple(coords_3d.shape)}  "
        f"({n_finite}/{coords_3d.shape[0]*coords_3d.shape[1]} finite points)"
    )


# ── main loop ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True,
                   help="Root directory containing all datasets")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more dataset names (sub-folders of dataset-root)")
    p.add_argument("--output-root", required=True,
                   help="Root directory where cotracker/ predictions were saved "
                        "(same value used for inference); cotracker_3d.npz is written here too")
    p.add_argument("--split", default="test",
                   help="Which split to process (default: test)")
    p.add_argument("--min-cams", type=int, default=2,
                   help="Minimum cameras needed to triangulate a point "
                        "(default: 2; set higher for more robust 3D)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-triangulate trials that already have cotracker_3d.npz")
    return p.parse_args()


def main():
    args = parse_args()

    for dataset_name in args.datasets:
        print(f"\nDataset: {dataset_name}")
        split_path = os.path.join(args.dataset_root, dataset_name, args.split)
        sessions   = sorted(get_dirs(split_path))

        for session in sessions:
            session_path = os.path.join(split_path, session)
            trials = sorted(get_dirs(session_path))
            for trial in trials:
                trial_path = os.path.join(session_path, trial)
                output_trial_path = os.path.join(
                    args.output_root, dataset_name, args.split, session, trial
                )
                print(f"  {session}/{trial}")
                process_trial(
                    trial_path=trial_path,
                    output_trial_path=output_trial_path,
                    min_cams=args.min_cams,
                    overwrite=args.overwrite,
                )

    print("Done.")


if __name__ == "__main__":
    main()