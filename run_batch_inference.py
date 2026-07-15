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

import numpy as np

from inference_video import load_model_from_base_folder, run_inference

DEFAULT_PREFIX = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v3'
DEFAULT_OUT_ROOT = '~/ghome/results/posetail-inference/2026-06-11'

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-dataset override of which 3D model output to use as the prediction.
# Datasets not listed use the default 'coords_pred'.
PRED_KEY_3D_OVERRIDES = {
    'johnson-fly': '3d_pred_triangulate',
}


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


def launch_render(tracks_path, vids_path, is_2d, trails=True):
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
    ap.add_argument('--no-trails', dest='trails', action='store_false', default=True,
                    help='disable motion trails (default ON): trails draw each '
                         'point\'s recent path so a still frame conveys tracking. '
                         'Uses render_video.py defaults (length 30, thickness 2).')
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
            pred_key_3d = PRED_KEY_3D_OVERRIDES.get(ds, 'coords_pred')
            print(f'    trial: {trial_dir}')
            print(f'    mode={mode} start_frame={start} n_frames={n_eff} '
                  f'score={score:.4f} pred_key_3d={pred_key_3d}')

            tracks_path = os.path.join(tracks_dir, f'{ds}.npz')
            vids_path = os.path.join(vids_dir, ds)

            # Inference: in-process, sequential on the GPU (model already loaded).
            run_inference(
                model=model,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                trial_path=trial_dir,
                start_frame=start,
                n_frames=n_eff,
                n_overlap=8,
                per_subject=True,
                device=device,
                outpath=tracks_path,
                pred_key_3d=pred_key_3d,
                query_first=args.query_first,
            )

            # Render: background subprocess, overlaps the next dataset's inference.
            renders.append((ds, launch_render(tracks_path, vids_path, is_2d, args.trails)))

            summary.append((ds, 'ok', os.path.basename(trial_dir),
                            str(start), str(n_eff), f'{score:.3f}', mode))
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
