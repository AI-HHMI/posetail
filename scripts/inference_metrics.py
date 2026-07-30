"""
Compute eval metrics over a directory of inference outputs with structure:
    dataset_root/dataset_name/subject_name/trial_name/output.npz

For each trial, saves eval_metrics.npz alongside the output.npz.
Aggregates all results into a CSV at dataset_root/all_metrics.csv.

Usage:
    python inference_metrics.py --dataset-root /path/to/dataset_root
    python inference_metrics.py --dataset-root /path/to/dataset_root --force
    python inference_metrics.py --dataset-root /path/to/dataset_root \\
        --input-name output.npz --output-name eval_metrics.npz \\
        --thresholds 1 2 4 8 16 --survival-threshold 50
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
    '''
    parse command line arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root', required=True,
                        help='Root directory containing dataset/subject/trial subdirectories')

    parser.add_argument('--input-name', default='output.npz',
                        help='Filename of inference output inside each trial dir (default: output.npz)')

    parser.add_argument('--output-name', default='eval_metrics.npz',
                        help='Filename to write per-trial metrics (default: eval_metrics.npz)')

    parser.add_argument('--thresholds', nargs='+', type=float, default=None,
                        help='Distance thresholds for delta_x / jaccard. '
                             'If omitted, uses per-dataset defaults from DATASET_THRESHOLDS '
                             '(dex_ycb, kubric-multiview, cmupanoptic_3dgs); '
                             'falls back to get_eval_metrics defaults for unknown datasets.')

    parser.add_argument('--survival-threshold', type=float, default=50,
                        help='L2 failure threshold for survival rate (default: 50)')

    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names to process (default: all)')

    parser.add_argument('--splits', nargs='+', default=['test'],
                        help='Split names to process (default: test)')

    parser.add_argument('--trials', nargs='+', default=None,
                        help='Trial names to process (default: all)')

    parser.add_argument('--prefix', type=str, default='',
                        help='Metric key prefix (default: no prefix)')

    parser.add_argument('--force', action='store_true',
                        help='Recompute metrics even if eval_metrics.npz already exists')

    parser.add_argument('--csv-out', type=str, default=None,
                        help='Path for per-trial CSV (default: dataset_root/all_metrics.csv)')

    parser.add_argument('--summary-out', type=str, default=None,
                        help='Path for per-dataset summary CSV (default: dataset_root/summary_metrics.csv)')

    return parser.parse_args()


def load_trial_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    required = {'coords_pred', 'coords_true', 'vis_pred', 'vis_true'}
    missing = required - set(data.files)
    if missing:
        raise KeyError(
            f'Missing keys {missing} in {npz_path}. '
            f'Re-run inference_video.py with the GT extraction block.'
        )

    coords_pred = torch.from_numpy(data['coords_pred'].astype('float32'))
    coords_true = torch.from_numpy(data['coords_true'].astype('float32'))
    vis_pred    = torch.from_numpy(data['vis_pred'].astype('float32'))
    vis_true    = torch.from_numpy(data['vis_true'].astype(bool))
    # per-point query times (query_first); absent in legacy outputs -> None (anchor frame 0)
    query_times = torch.from_numpy(data['query_times'].astype('int64')) \
        if 'query_times' in data.files else None

    return coords_pred, coords_true, vis_pred, vis_true, query_times


def run_trial(npz_path, thresholds, survival_threshold, prefix):
    """Load one output.npz, compute metrics, return dict."""
    coords_pred, coords_true, vis_pred, vis_true, query_times = load_trial_npz(npz_path)
    metrics = get_eval_metrics(
        vis_pred=vis_pred,
        vis_true=vis_true,
        coords_pred=coords_pred,
        coords_true=coords_true,
        thresholds=thresholds,
        survival_threshold=survival_threshold,
        prefix=prefix,
        query_times=query_times,
    )
    return metrics


def save_trial_metrics(metrics, out_path):
    """Save a metrics dict as a flat .npz."""
    np.savez(out_path, **{k: np.array(v) for k, v in metrics.items()})


def find_trials(dataset_root, input_name, datasets=None, splits=None, trials=None):
    """
    Yield (dataset_name, split_name, subject_name, trial_name, npz_path) for every
    input_name file found at depth dataset_root/dataset/split/subject/trial/.
    """
    root = Path(dataset_root)
    pattern = f'*/*/*/*/{input_name}'
    for npz_path in sorted(root.glob(pattern)):
        parts = npz_path.relative_to(root).parts
        # parts = (dataset_name, split_name, subject_name, trial_name, input_name)
        if len(parts) == 5:
            dataset_name, split_name, subject_name, trial_name, _ = parts
            if datasets and dataset_name not in datasets:
                continue
            if splits and split_name not in splits:
                continue
            if trials and trial_name not in trials:
                continue
            yield dataset_name, split_name, subject_name, trial_name, npz_path


def main():

    args = parse_args()

    csv_out     = args.csv_out     or os.path.join(args.dataset_root, 'all_metrics.csv')
    summary_out = args.summary_out or os.path.join(args.dataset_root, 'summary_metrics.csv')

    # discover trials
    trials = list(find_trials(args.dataset_root, args.input_name,
                              datasets=args.datasets, splits=args.splits, trials=args.trials))
    if len(trials) == 0:
        print(f'No {args.input_name} files found under {args.dataset_root}')
        return

    print(f'Found {len(trials)} trial(s) under {args.dataset_root}')

    rows = []
    n_skipped = n_failed = n_computed = 0

    for dataset_name, split_name, subject_name, trial_name, npz_path in tqdm(trials, desc='Trials'):
        trial_dir   = npz_path.parent
        metrics_out = trial_dir / args.output_name

        # resolve thresholds: explicit flag > dataset default > get_eval_metrics default
        thresholds = args.thresholds or DATASET_THRESHOLDS.get(dataset_name)
        # print(f"  dataset_name={dataset_name!r}  thresholds={thresholds}")

        # optionally skip already-computed trials
        if metrics_out.exists() and not args.force:
            existing = np.load(metrics_out, allow_pickle=True)
            metrics = {k: float(existing[k]) for k in existing.files}
            n_skipped += 1
        else:
            try:
                metrics = run_trial(
                    npz_path,
                    thresholds=thresholds,
                    survival_threshold=args.survival_threshold,
                    prefix=args.prefix,
                )
                save_trial_metrics(metrics, metrics_out)
                n_computed += 1
            except Exception as e:
                tqdm.write(f'  FAILED {dataset_name}/{split_name}/{subject_name}/{trial_name}: {e}')
                traceback.print_exc()
                n_failed += 1
                continue

        row = {
            'dataset':  dataset_name,
            'split':    split_name,
            'subject':  subject_name,
            'trial':    trial_name,
            'npz_path': str(npz_path),
        }
        row.update({k: float(v) for k, v in metrics.items()})
        rows.append(row)

    print(f'\nDone: {n_computed} computed, {n_skipped} skipped, {n_failed} failed')

    if len(rows) == 0:
        print('No results to aggregate.')
        return

    df = pd.DataFrame(rows)

    # move id columns to front
    id_cols     = ['dataset', 'split', 'subject', 'trial', 'npz_path']
    metric_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + metric_cols]

    df.to_csv(csv_out, index=False)
    print(f'Saved per-trial CSV  to {csv_out}')

    # per-dataset summary (averaged across splits and subjects within each dataset)
    summary_df = df.groupby('dataset')[metric_cols].mean().reset_index()
    summary_df.to_csv(summary_out, index=False)
    print(f'Saved summary CSV    to {summary_out}')

    print('\n--- Mean per dataset ---')
    max_len = max(len(k) for k in metric_cols)
    for _, row in summary_df.iterrows():
        print(f"\n  [{row['dataset']}]")
        for k in metric_cols:
            print(f'    {k:<{max_len}}  {row[k]:.4f}')


if __name__ == '__main__':
    main()