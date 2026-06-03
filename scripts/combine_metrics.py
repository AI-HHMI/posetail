"""
combine eval metric results across multiple experiments into a single file
for downstream analysis

example:
    python combine_metrics.py --prefix /home/ruppk2@hhmi.org/dataset_predictions

outputs (written to --prefix):
    combined_all_metrics.csv: contains per-trial averages for each experiment
    combined_summary_metrics.csv: contains per-dataset averages for each experiment
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

    experiments = sorted(
        d for d in os.listdir(prefix)
        if os.path.isdir(os.path.join(prefix, d))
    )

    print(f'Found {len(experiments)} experiment(s): {experiments}')

    all_dfs = []
    summary_dfs = []

    for experiment in experiments:

        exp_path = os.path.join(prefix, experiment)

        all_path = os.path.join(exp_path, args.all_name)
        if os.path.exists(all_path):
            df = pd.read_csv(all_path)
            df.insert(0, 'experiment', experiment)
            all_dfs.append(df)
        else:
            print(f'WARNING: {all_path} not found, skipping')

        summary_path = os.path.join(exp_path, args.summary_name)
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df.insert(0, 'experiment', experiment)
            summary_dfs.append(df)
        else:
            print(f'WARNING: {summary_path} not found, skipping')

    if all_dfs:
        combined_all = pd.concat(all_dfs, ignore_index = True)
        combined_all.to_csv(all_out, index = False)
        print(f'Saved combined per-trial CSV  -> {all_out}  ({len(combined_all)} rows)')
    else:
        print('No per-trial CSVs found.')

    if summary_dfs:
        combined_summary = pd.concat(summary_dfs, ignore_index=True)
        combined_summary.to_csv(summary_out, index=False)
        print(f'Saved combined summary CSV    -> {summary_out}  ({len(combined_summary)} rows)')
    else:
        print('No summary CSVs found.')


if __name__ == '__main__':
    main()