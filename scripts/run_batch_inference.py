#!/usr/bin/env python
"""Batch inference + render over every dataset in the finetuning prefix.

For each dataset, pick a test trial and a 256-frame window where the animal is
moving, run inference, then render an overlaid (cropped) video.

The model (and torch) is loaded ONCE and reused for every dataset; inference runs
sequentially on the GPU. Rendering is CPU-bound, so each render is launched as a
background subprocess that overlaps the *next* dataset's inference.

Run via the pixi env:
    pixi run python run_batch_inference.py --base-folder <wandb_run_dir>
"""
import argparse
import glob
import os
import subprocess
import sys

# Reduce CUDA fragmentation on dense point sets (point-odyssey, cmupanoptic) before torch is
# imported anywhere (main() / pick_inference_caps import it lazily). Mirrors eval_testset_metrics.py.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import yaml

from posetail.inference.inference_utils import load_model_from_base_folder, run_inference

DEFAULT_PREFIX = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4'
DEFAULT_OUT_ROOT = '~/ghome/results/posetail-inference/2026-06-11'

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-dataset override of which 3D model output to use as the prediction.
# Datasets not listed use the default 'coords_pred' (== '3d_pred_direct').
# NOTE: johnson-fly used '3d_pred_triangulate', but its cameras are near-orthographic
# (telecentric; DLT L9=L10=L11~=0), so weighted-DLT triangulation is ill-conditioned along
# the low-parallax world axis and injects large anisotropic jitter (~20x GT accel on y). The
# direct head is both smoother (0.7x GT accel) and more accurate, so it uses the default.
PRED_KEY_3D_OVERRIDES = {}

# GPU-memory-aware inference caps (dynamic, per-trial -- not hardcoded per dataset).
MAX_VIEWS = 14
# Peak decoder memory ~ (views used) * (kpt chunk). Calibrated conservatively so an 80GB GPU
# reproduces the known-good cmupanoptic_3dgs setting (14 views * 600 kpts ~= 8400).
KPTVIEW_BUDGET_PER_GB = 100   # -> ~8000 on 80GB, ~4000 on 40GB
MIN_KPT_CHUNK = 256


def count_cameras(trial_dir, is_2d):
    """Cameras available for a trial, without loading video. 2D trials are single-camera."""
    if is_2d:
        return 1
    with open(os.path.join(trial_dir, 'metadata.yaml')) as f:
        meta = yaml.safe_load(f)
    return len(meta.get('intrinsic_matrices', {}))


def pick_inference_caps(n_cams_total, n_kpts, device):
    """Choose (n_views, max_kpts) from GPU memory + this trial's shape.

    - Cameras capped at MAX_VIEWS (the scene-encoder memory lever; the only knob that reduces
      per-camera scene-encoder cost).
    - kpt_chunk (decoder point-axis chunking, numerically identical to a single pass) sized so
      (views_used * kpt_chunk) fits a conservative per-GB budget; returns None (no chunking) when
      the full keypoint set already fits.
    """
    n_views = min(n_cams_total, MAX_VIEWS)
    if device is None or device.type != 'cuda':
        return n_views, None
    import torch
    idx = device.index if device.index is not None else torch.cuda.current_device()
    mem_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    budget = KPTVIEW_BUDGET_PER_GB * mem_gb
    max_kpts = int(budget / max(n_views, 1))
    max_kpts = max(MIN_KPT_CHUNK, min(max_kpts, n_kpts))
    return n_views, (None if max_kpts >= n_kpts else max_kpts)


def find_trials(dataset_dir, split):
    """Return list of (trial_dir, pose_npz_path, is_2d) for a dataset/split."""
    trials = []
    for name, is_2d in (('pose2d.npz', True), ('pose3d.npz', False)):
        pattern = os.path.join(dataset_dir, split, '**', name)
        for pose_path in glob.glob(pattern, recursive=True):
            trials.append((os.path.dirname(pose_path), pose_path, is_2d))
    return trials


def motion_signal(pose):
    """Per-frame inter-frame displacement magnitude, NaN-ignored.

    pose: (subjects, T, kpts, R) -> motion: (T,) with motion[0] = 0.
    """
    if pose.shape[1] < 2:
        return np.zeros(pose.shape[1], dtype=np.float64)
    disp = np.linalg.norm(pose[:, 1:] - pose[:, :-1], axis=-1)  # (S, T-1, K)
    with np.errstate(invalid='ignore'):
        per_frame = np.nanmean(disp, axis=(0, 2))  # (T-1,)
    per_frame = np.nan_to_num(per_frame, nan=0.0)
    return np.concatenate([[0.0], per_frame])  # (T,)


def best_window(motion, n_frames):
    """Return (start_frame, window_score) maximizing windowed sum of motion."""
    T = len(motion)
    if T <= n_frames:
        return 0, float(np.sum(motion))
    csum = np.concatenate([[0.0], np.cumsum(motion)])
    win = csum[n_frames:] - csum[:-n_frames]  # length T - n_frames + 1
    start = int(np.argmax(win))
    return start, float(win[start])


def pick_trial(trials, n_frames):
    """Pick the trial+window with the most movement.

    Returns (trial_dir, is_2d, start_frame, n_frames_effective, score, clamped).
    """
    candidates = []  # (score, trial_dir, is_2d, start, T)
    for trial_dir, pose_path, is_2d in trials:
        try:
            pose = np.load(pose_path)['pose']
        except Exception as exc:
            print(f'    [warn] could not load {pose_path}: {exc}')
            continue
        T = pose.shape[1]
        motion = motion_signal(pose)
        start, score = best_window(motion, n_frames)
        candidates.append((score, trial_dir, is_2d, start, T))

    if not candidates:
        return None

    enough = [c for c in candidates if c[4] >= n_frames]
    if enough:
        score, trial_dir, is_2d, start, T = max(enough, key=lambda c: c[0])
        return trial_dir, is_2d, start, n_frames, score, False

    # Fallback: no trial long enough -> use the longest, full length.
    score, trial_dir, is_2d, start, T = max(candidates, key=lambda c: c[4])
    print(f'    [clamp] no trial has >= {n_frames} frames; using longest '
          f'trial ({T} frames), n_frames={T}')
    return trial_dir, is_2d, 0, T, score, True


def launch_render(tracks_path, vids_path, is_2d, trails=False):
    """Start render_video.py as a background subprocess; return the Popen."""
    conf = '0.2' if is_2d else '0.5'
    cmd = [sys.executable, os.path.join(HERE, 'render_video.py'),
           '--input-npz', tracks_path,
           '--output-dir', vids_path,
           '--crop',
           '--conf-threshold', conf]
    if trails:
        cmd.append('--trails')
    print('    [render bg] $', ' '.join(cmd))
    return subprocess.Popen(cmd)


def has_rendered_video(vids_path):
    """True if vids_path holds at least one rendered .mp4.

    render_info.json is written before rendering starts, so it is not a reliable
    'done' marker; the .mp4 files are.
    """
    return os.path.isdir(vids_path) and any(
        f.endswith('.mp4') for f in os.listdir(vids_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-folder', required=True,
                    help='wandb run dir (config + checkpoints) for inference')
    ap.add_argument('--prefix', default=DEFAULT_PREFIX,
                    help='root folder containing the datasets')
    ap.add_argument('--out-root', default=DEFAULT_OUT_ROOT,
                    help='output root; tracks/ and vids/ are created under it')
    ap.add_argument('--split', default='test')
    ap.add_argument('--n-frames', type=int, default=256)
    ap.add_argument('--checkpoint', type=int, default=None,
                    help='checkpoint step; default = latest in the run')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='restrict to these dataset names (default: all under prefix)')
    ap.add_argument('--no-query-first', dest='query_first', action='store_false', default=True,
                    help='disable query-first (default ON): anchor each point at its first '
                         'valid+visible frame (mvtracker/training convention). With this flag, '
                         'all points are anchored at start_frame instead (legacy 3D behavior).')
    ap.add_argument('--trails', action='store_true', default=False,
                    help='enable motion trails (default OFF): trails draw each '
                         'point\'s recent path so a still frame conveys tracking. '
                         'Uses render_video.py defaults (length 30, thickness 2).')
    ap.add_argument('--view-seed', type=int, default=0,
                    help='seed for camera + point subsampling (default 0, reproducible '
                         'across --force reruns)')
    ap.add_argument('--max-points', type=int, default=600,
                    help='cap tracked query points PER SUBJECT (default 600); subjects with '
                         'more are random-subsampled. Bounds memory/time on dense sets '
                         '(point-odyssey has ~40k-76k pts/subject). 0 disables the cap.')
    ap.add_argument('--force', action='store_true', default=False,
                    help='re-run inference and re-render even when outputs already '
                         'exist. By default a dataset whose tracks npz + video are both '
                         'present is skipped; if only the video is missing, inference is '
                         'reused and only the render runs.')
    args = ap.parse_args()

    out_root = os.path.expanduser(args.out_root)
    tracks_dir = os.path.join(out_root, 'tracks')
    vids_dir = os.path.join(out_root, 'vids')
    os.makedirs(tracks_dir, exist_ok=True)
    os.makedirs(vids_dir, exist_ok=True)
    print(f'Output root: {out_root}')

    import torch
    device = torch.device(args.device) if args.device is not None else None
    print(f'Loading model from {args.base_folder} ...')
    model, config, config_path, checkpoint_path = load_model_from_base_folder(
        args.base_folder, checkpoint=args.checkpoint, device=device)
    print(f'Model loaded (checkpoint: {checkpoint_path})')

    # os.path.isdir follows symlinks, so symlinked datasets (e.g. 3dpop, rat7m,
    # which point into sibling prefixes) are included.
    datasets = sorted(
        d for d in os.listdir(args.prefix)
        if os.path.isdir(os.path.join(args.prefix, d))
    )
    if args.datasets:
        datasets = [d for d in datasets if d in set(args.datasets)]
    print(f'Found {len(datasets)} datasets: {datasets}')

    renders = []     # (dataset, Popen)
    summary = []     # (dataset, status, trial, start, n, score, mode)
    for ds in datasets:
        print(f'\n=== {ds} ===')
        try:
            tracks_path = os.path.join(tracks_dir, f'{ds}.npz')
            vids_path = os.path.join(vids_dir, ds)

            # Skip work whose output already exists (unless --force). Inference and
            # render are gated independently: a present tracks npz is reused, and a
            # present video skips the render.
            need_inference = args.force or not os.path.exists(tracks_path)
            need_render = args.force or not has_rendered_video(vids_path)
            if not need_inference and not need_render:
                print('    [skip] tracks + video already present (use --force to overwrite)')
                summary.append((ds, 'skipped', os.path.basename(tracks_path),
                                '', '', '', ''))
                continue

            if need_inference:
                trials = find_trials(os.path.join(args.prefix, ds), args.split)
                if not trials:
                    print(f'    [warn] no {args.split} trials found, skipping')
                    summary.append((ds, 'no-trials', '', '', '', '', ''))
                    continue

                pick = pick_trial(trials, args.n_frames)
                if pick is None:
                    print('    [warn] no loadable pose data, skipping')
                    summary.append((ds, 'no-pose', '', '', '', '', ''))
                    continue
                trial_dir, is_2d, start, n_eff, score, clamped = pick
                mode = '2d' if is_2d else '3d'
                trial_name = os.path.basename(trial_dir)
                pred_key_3d = PRED_KEY_3D_OVERRIDES.get(ds, 'coords_pred')
                print(f'    trial: {trial_dir}')
                print(f'    mode={mode} start_frame={start} n_frames={n_eff} '
                      f'score={score:.4f} pred_key_3d={pred_key_3d}')

                # Dynamic, GPU-memory-aware caps from this trial's actual shape: cap cameras at
                # MAX_VIEWS and size the keypoint chunk to fit a conservative memory budget.
                pose_name = 'pose2d.npz' if is_2d else 'pose3d.npz'
                n_kpts = int(np.load(os.path.join(trial_dir, pose_name))['pose'].shape[2])
                # Points are subsampled to max_points/subject before tracking, so size the chunk
                # to the count actually tracked (n_kpts is per-subject: pose is (S,T,K,R)).
                if args.max_points:
                    n_kpts = min(n_kpts, args.max_points)
                n_cams = count_cameras(trial_dir, is_2d)
                dev = device if device is not None else next(model.parameters()).device
                n_views, max_kpts = pick_inference_caps(n_cams, n_kpts, dev)
                print(f'    n_cams={n_cams} -> n_views={n_views}  '
                      f'n_kpts={n_kpts} -> max_kpts={max_kpts}')

                # Inference: in-process, sequential on the GPU (model already loaded).
                run_inference(
                    model=model,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    trial_path=trial_dir,
                    start_frame=start,
                    n_frames=n_eff,
                    n_overlap=8,
                    n_views=n_views,
                    seed=args.view_seed,
                    max_kpts=max_kpts,
                    max_points=(args.max_points or None),  # 0 -> None (disable cap)
                    per_subject=True,
                    device=device,
                    outpath=tracks_path,
                    pred_key_3d=pred_key_3d,
                    query_first=args.query_first,
                )
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                score_str = f'{score:.3f}'
            else:
                # Reuse existing tracks; only (re)render. Recover mode/is_2d from the npz.
                print('    [skip inference] tracks present; rendering only')
                mode = str(np.load(tracks_path, allow_pickle=True)['mode'])
                is_2d = (mode == '2d')
                trial_name = '(cached)'
                start = n_eff = score_str = ''

            # Render: background subprocess, overlaps the next dataset's inference.
            if need_render:
                renders.append((ds, launch_render(tracks_path, vids_path, is_2d, args.trails)))
            else:
                print('    [skip render] video already present')

            summary.append((ds, 'ok', trial_name,
                            str(start), str(n_eff), score_str, mode))
        except Exception as exc:
            print(f'    [error] {ds} failed: {exc}')
            summary.append((ds, 'FAILED', '', '', '', '', ''))

    # Wait for all background renders to finish.
    print('\nWaiting for background renders to finish...')
    for ds, proc in renders:
        rc = proc.wait()
        status = 'ok' if rc == 0 else f'render-rc={rc}'
        print(f'    {ds}: render {status}')
        if rc != 0:
            for i, row in enumerate(summary):
                if row[0] == ds and row[1] == 'ok':
                    summary[i] = (ds, 'RENDER-FAIL', *row[2:])

    print('\n========== SUMMARY ==========')
    hdr = ('dataset', 'status', 'trial', 'start', 'n', 'score', 'mode')
    print('{:<16} {:<11} {:<28} {:>6} {:>5} {:>9} {:>4}'.format(*hdr))
    for row in summary:
        print('{:<16} {:<11} {:<28} {:>6} {:>5} {:>9} {:>4}'.format(*row))


if __name__ == '__main__':
    main()
