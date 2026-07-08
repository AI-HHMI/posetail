"""
Compute eval metrics for CoTracker3 predictions.

Reads cotracker_3d.npz from output_root, loads GT from dataset_root,
computes metrics via get_eval_metrics, saves eval_metrics.npz alongside
each prediction, and aggregates to CSV.

Structure expected:
    output_root/dataset/split/session/trial/cotracker_3d.npz
    dataset_root/dataset/split/session/trial/pose3d.npz

Usage:
    python cotracker_metrics.py \\
        --dataset-root /path/to/datasets \\
        --output-root  /path/to/cotracker-outputs \\
        --datasets kubric-multiview \\
        --split test
"""

import os
import argparse
import traceback

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from posetail.posetail.eval_metrics import get_eval_metrics


# Per-dataset default thresholds (already divided to the correct scale).
# Used when --thresholds is not passed on the command line.
# Explicit --thresholds always overrides these.
DATASET_THRESHOLDS = {
    'dex_ycb':          [x / 100 for x in [1, 2, 5, 10, 20]],
    'kubric-multiview': [x / 100 for x in [0.65, 1.3, 2.6, 5.2, 10.4]],
    'cmupanoptic_3dgs': [x / 13  for x in [5, 10, 20, 40]],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True,
                        help='Root of source datasets (contains pose3d.npz ground truth)')
    parser.add_argument('--output-root', required=True,
                        help='Root of CoTracker outputs (contains cotracker_3d.npz)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names to process (default: all found)')
    parser.add_argument('--split', default='test',
                        help='Split name (default: test)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=None,
                        help='Distance thresholds for delta_x / jaccard')
    parser.add_argument('--survival-threshold', type=float, default=50,
                        help='L2 failure threshold for survival rate (default: 50)')
    parser.add_argument('--prefix', type=str, default='cotracker_',
                        help='Metric key prefix (default: cotracker_)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if eval_metrics.npz already exists')
    parser.add_argument('--csv-out', type=str, default=None,
                        help='Path for per-trial CSV '
                             '(default: output_root/all_metrics.csv)')
    parser.add_argument('--summary-out', type=str, default=None,
                        help='Path for summary CSV '
                             '(default: output_root/summary_metrics.csv)')
    return parser.parse_args()


def find_trials(output_root, split, datasets=None):
    """
    Yield (dataset, session, trial, pred_path) for every cotracker_3d.npz
    found at output_root/dataset/split/session/trial/.
    """
    root = Path(output_root)
    for pred_path in sorted(root.glob(f'*/{split}/*/*/cotracker_3d.npz')):
        parts = pred_path.relative_to(root).parts
        # parts = (dataset, split, session, trial, 'cotracker_3d.npz')
        dataset_name, _, session, trial, _ = parts
        if datasets and dataset_name not in datasets:
            continue
        yield dataset_name, session, trial, pred_path


def load_ground_truth(pose_path, frame_numbers, valid_mask):
    """
    Load GT coords and visibility from pose3d.npz, aligned to frame_numbers
    and filtered to only the keypoints tracked (via valid_mask).

    Returns
    -------
    coords_true : (1, T, N, 3) float32
    vis_true    : (1, T, N, 1) bool
    """
    data = np.load(pose_path)
    coords_gt  = data['pose']                              # (S, T_full, K, 3)
    vis_gt_raw = data['vis'] if 'vis' in data else None

    S, T_full, K, _ = coords_gt.shape
    frame_nums_clipped = np.clip(frame_numbers, 0, T_full - 1)

    # Flatten (subject, kpt) -> (S*K, T_full, 3), apply valid_mask
    coords_flat = coords_gt.transpose(0, 2, 1, 3)          # (S, K, T_full, 3)
    coords_flat = coords_flat.reshape(S * K, T_full, 3)    # (S*K, T_full, 3)
    coords_flat = coords_flat[valid_mask]                   # (N, T_full, 3)

    coords_true = coords_flat[:, frame_nums_clipped, :]     # (N, T, 3)
    coords_true = coords_true.transpose(1, 0, 2)[np.newaxis]  # (1, T, N, 3)

    if vis_gt_raw is not None:
        vis_agg  = vis_gt_raw.any(axis=-1)                  # (S, T_full, K)
        vis_flat = vis_agg.transpose(0, 2, 1)               # (S, K, T_full)
        vis_flat = vis_flat.reshape(S * K, T_full)          # (S*K, T_full)
        vis_flat = vis_flat[valid_mask]                     # (N, T_full)
        vis_true = vis_flat[:, frame_nums_clipped].T        # (T, N)
        vis_true = vis_true[np.newaxis, :, :, np.newaxis]  # (1, T, N, 1)
    else:
        vis_true = np.all(np.isfinite(coords_true), axis=-1, keepdims=True)

    return coords_true.astype(np.float32), vis_true.astype(bool)


def run_trial(pred_path, pose_path, thresholds, survival_threshold, prefix):
    pred = np.load(pred_path)

    coords_3d     = pred['coords_3d']      # (T, N, 3)
    visibility    = pred['visibility']     # (T, N) bool
    frame_numbers = pred['frame_numbers']  # (T,)
    valid_mask    = pred['valid_mask']     # (S*K,) bool

    # Mask out triangulation failures: points with NaN coords (fewer than
    # min_cams cameras visible) should not be counted as visible.
    finite_mask = np.isfinite(coords_3d).all(axis=-1)  # (T, N)
    visibility  = visibility & finite_mask

    # Replace NaN coords with 0 so np.median inside get_eval_metrics never
    # sees a NaN (masked-out points won't contribute to any metric anyway).
    coords_3d = np.where(finite_mask[:, :, np.newaxis], coords_3d, 0.0)

    coords_pred = torch.from_numpy(
        coords_3d[np.newaxis].astype(np.float32)                       # (1, T, N, 3)
    )
    vis_pred = torch.from_numpy(
        visibility[np.newaxis, :, :, np.newaxis].astype(np.float32)    # (1, T, N, 1)
    )

    coords_true_np, vis_true_np = load_ground_truth(pose_path, frame_numbers, valid_mask)
    coords_true = torch.from_numpy(coords_true_np)  # (1, T, N, 3)
    vis_true    = torch.from_numpy(vis_true_np)     # (1, T, N, 1)

    metrics = get_eval_metrics(
        vis_pred=vis_pred,
        vis_true=vis_true,
        coords_pred=coords_pred,
        coords_true=coords_true,
        thresholds=thresholds,
        survival_threshold=survival_threshold,
        prefix=prefix,
    )
    return metrics


def main():
    args = parse_args()

    trials = list(find_trials(args.output_root, args.split, datasets=args.datasets))
    if not trials:
        print(f'No cotracker_3d.npz files found under {args.output_root}')
        return

    print(f'Found {len(trials)} trial(s)')

    # Accumulate rows per dataset so we can write one CSV per dataset,
    # matching the structure combine_metrics.py expects:
    #   output_root/dataset_name/all_metrics.csv
    rows_by_dataset = {}
    n_computed = n_skipped = n_failed = 0

    for dataset_name, session, trial, pred_path in tqdm(trials, desc='Trials'):
        metrics_out = pred_path.parent / 'eval_metrics.npz'

        # resolve thresholds: explicit flag > dataset default > get_eval_metrics default
        thresholds = args.thresholds or DATASET_THRESHOLDS.get(dataset_name)
        if thresholds is None and dataset_name not in DATASET_THRESHOLDS:
            tqdm.write(f'  WARNING: no default thresholds for dataset {dataset_name!r} '
                       f'— using get_eval_metrics defaults. '
                       f'Add it to DATASET_THRESHOLDS or pass --thresholds explicitly.')

        if metrics_out.exists() and not args.force:
            existing = np.load(metrics_out, allow_pickle=True)
            metrics  = {k: float(existing[k]) for k in existing.files}
            n_skipped += 1
        else:
            pose_path = (
                Path(args.dataset_root)
                / dataset_name / args.split / session / trial / 'pose3d.npz'
            )
            if not pose_path.exists():
                tqdm.write(f'  MISSING GT: {dataset_name}/{session}/{trial}')
                n_failed += 1
                continue
            try:
                metrics = run_trial(
                    pred_path=pred_path,
                    pose_path=pose_path,
                    thresholds=thresholds,
                    survival_threshold=args.survival_threshold,
                    prefix=args.prefix,
                )
                np.savez(metrics_out, **{k: np.array(v) for k, v in metrics.items()})
                n_computed += 1
            except Exception as e:
                tqdm.write(f'  FAILED {dataset_name}/{session}/{trial}: {e}')
                traceback.print_exc()
                n_failed += 1
                continue

        row = {'dataset': dataset_name, 'split': args.split,
               'subject': session, 'trial': trial}
        row.update({k: float(v) for k, v in metrics.items()})
        rows_by_dataset.setdefault(dataset_name, []).append(row)

    print(f'\nDone: {n_computed} computed, {n_skipped} skipped, {n_failed} failed')

    if not rows_by_dataset:
        print('No results to aggregate.')
        return

    all_rows = []
    for dataset_name, rows in rows_by_dataset.items():
        df = pd.DataFrame(rows)
        id_cols     = ['dataset', 'split', 'subject', 'trial']
        metric_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + metric_cols]
        all_rows.append(df)

        # Save per-dataset CSVs inside the dataset subdirectory so
        # combine_metrics.py --prefix output_root finds them correctly
        dataset_dir = Path(args.output_root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        csv_out     = args.csv_out     or str(dataset_dir / 'all_metrics.csv')
        summary_out = args.summary_out or str(dataset_dir / 'summary_metrics.csv')

        df.to_csv(csv_out, index=False)
        print(f'Saved per-trial CSV to {csv_out}')

        summary_df = df.groupby('dataset')[metric_cols].mean().reset_index()
        summary_df.to_csv(summary_out, index=False)
        print(f'Saved summary CSV   to {summary_out}')

    print('\n--- Mean per dataset ---')
    combined = pd.concat(all_rows, ignore_index=True)
    metric_cols = [c for c in combined.columns if c not in ['dataset', 'split', 'subject', 'trial']]
    summary = combined.groupby('dataset')[metric_cols].mean()
    max_len = max(len(k) for k in metric_cols)
    for dataset_name, row in summary.iterrows():
        print(f'\n  [{dataset_name}]')
        for k in metric_cols:
            print(f'    {k:<{max_len}}  {row[k]:.4f}')


if __name__ == '__main__':
    main()