"""
Combine per-n_views eval CSVs into a single file.

Expects a prefix directory whose immediate subdirectories are named by the
number of camera views used (e.g. 1/, 2/, ..., 10/), each containing
all_metrics.csv and summary_metrics.csv produced by inference_metrics.py.

Usage:
    python combine_nviews_metrics.py \
        --prefix /home/ruppk2@hhmi.org/dataset_predictions/f8pai8gk_n_cams

Outputs (written to --prefix):
    combined_all_metrics.csv — per-trial rows with n_views column
    combined_summary_metrics.csv — per-dataset mean rows with n_views column
"""

import os
import argparse

import pandas as pd


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', required = True)
    parser.add_argument('--all-name', default = 'all_metrics.csv')
    parser.add_argument('--summary-name', default = 'summary_metrics.csv')

    return parser.parse_args()


def main():

    args = parse_args()
    prefix = args.prefix

    all_out = os.path.join(prefix, 'combined_all_metrics.csv')
    summary_out = os.path.join(prefix, 'combined_summary_metrics.csv')

    # find subdirectories that are integers, sorted numerically
    subdirs = sorted(
        [d for d in os.listdir(prefix) if os.path.isdir(os.path.join(prefix, d)) and d.isdigit()],
        key=lambda d: int(d),
    )

    if not subdirs:
        print(f'No numbered subdirectories found under {prefix}')
        return

    print(f'Found {len(subdirs)} subdirectories: {subdirs}')

    all_dfs = []
    summary_dfs = []

    for subdir in subdirs:
        n_views = int(subdir)
        subdir_path = os.path.join(prefix, subdir)

        all_path = os.path.join(subdir_path, args.all_name)
        if os.path.exists(all_path):
            df = pd.read_csv(all_path)
            df.insert(0, 'n_views', n_views)
            all_dfs.append(df)
        else:
            print(f'WARNING: {all_path} not found, skipping')

        summary_path = os.path.join(subdir_path, args.summary_name)
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df.insert(0, 'n_views', n_views)
            summary_dfs.append(df)
        else:
            print(f'WARNING: {summary_path} not found, skipping')

    if all_dfs:
        combined_all = pd.concat(all_dfs, ignore_index=True)
        combined_all.to_csv(all_out, index=False)
        print(f'Saved combined per-trial CSV -> {all_out} ({len(combined_all)} rows)')
    else:
        print('No per-trial CSVs found.')

    if summary_dfs:
        combined_summary = pd.concat(summary_dfs, ignore_index=True)
        combined_summary.to_csv(summary_out, index=False)
        print(f'Saved combined summary CSV -> {summary_out} ({len(combined_summary)} rows)')
    else:
        print('No summary CSVs found.')


if __name__ == '__main__':
    main()