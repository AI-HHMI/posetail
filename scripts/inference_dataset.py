"""
run_inference_dataset.py

Run inference_video.py over multiple trials, saving outputs to a separate
output root that mirrors the source directory structure.

Source structure (input):
    dataset_root/dataset_name/split_name/subject_name/trial_name/   (4-level)
    dataset_root/dataset_name/split_name/trial_name/                 (3-level, no subject)

Output structure:
    output_root/dataset_name/split_name/subject_name/trial_name/output.npz
    output_root/dataset_name/split_name/trial_name/output.npz

Usage:
    python run_inference_dataset.py \\
        --dataset-root /path/to/dataset_root \\
        --output-root  /path/to/predictions \\
        --datasets kubric-multiview \\
        --base-folder /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX \\
        --checkpoint 799992 \\
        --n-frames 500 --n-overlap 2 \\
        --device cuda:1
"""

import os
import sys
import argparse
import subprocess
import traceback
from pathlib import Path

from tqdm import tqdm


def parse_args():
    '''
    parse command line arguments
    '''
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--dataset-root', required=True,
                        help='Root directory containing source trial data')
    parser.add_argument('--output-root', required=True,
                        help='Root directory for inference outputs — mirrors dataset-root structure')

    # dataset selection
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names to process (default: all found under dataset-root)')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        help='Split names to process, e.g. train val test (default: test)')
    parser.add_argument('--trials', nargs='+', default=None,
                        help='Trial names to process (default: all)')

    # model
    parser.add_argument('--base-folder', required=True,
                        help='Wandb run folder containing files/config.toml and files/checkpoints/')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Checkpoint step number; if omitted, uses latest')

    # inference settings (passed through to inference_video.py)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=None, help='Max frames per trial')
    parser.add_argument('--n-overlap', type=int, default=2)
    parser.add_argument('--n-views', type=int, default=None, help='Evaluate on a subset of cameras')
    parser.add_argument('--view-seed', type=int, default=None, help='Random seed for subsampling cameras')
    parser.add_argument('--max-kpts', type=int, default=None, help='Max keypoints per forward pass')
    parser.add_argument('--clip-len', type=int, default=None,
                        help='Frames fed to the model per forward (passed to inference_video). '
                             'Defaults to model.n_frames; the whole chunk is encoded in one '
                             'pass.')
    parser.add_argument('--per-subject', action='store_true', default=False)
    parser.add_argument('--device', type=str, default=None)

    # output
    parser.add_argument('--output-name', type=str, default='output.npz',
                        help='Filename to write inside each trial directory (default: output.npz)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run inference even if output already exists')

    # path to inference script
    parser.add_argument('--inference-script', type=str, default=None,
                        help='Path to inference_video.py (default: alongside this script)')

    return parser.parse_args()


def find_trials(dataset_root, datasets=None, splits=None, trials=None):
    """
    Yield (dataset_name, split_name, subject_name, trial_name, trial_path)
    for every valid trial directory under dataset_root.

    Supports:
        4-level: dataset_root/dataset/split/subject/trial/
        3-level: dataset_root/dataset/split/trial/   (subject_name = None)

    A valid trial directory must contain metadata.yaml + pose3d.npz
    and an img/ or vid/ folder.
    """
    root = Path(dataset_root)

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if datasets and dataset_dir.name not in datasets:
            continue

        for split_dir in sorted(dataset_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            if splits and split_dir.name not in splits:
                continue

            for level2_dir in sorted(split_dir.iterdir()):
                if not level2_dir.is_dir():
                    continue

                # Check if level2_dir is itself a trial (3-level structure)
                if _is_trial_dir(level2_dir):
                    if trials and level2_dir.name not in trials:
                        continue
                    yield dataset_dir.name, split_dir.name, None, level2_dir.name, level2_dir

                else:
                    # Treat level2_dir as subject, iterate into trials (4-level structure)
                    for trial_dir in sorted(level2_dir.iterdir()):
                        if not trial_dir.is_dir():
                            continue
                        if trials and trial_dir.name not in trials:
                            continue
                        if _is_trial_dir(trial_dir):
                            yield dataset_dir.name, split_dir.name, level2_dir.name, trial_dir.name, trial_dir


def _is_trial_dir(path):
    """A trial directory contains metadata.yaml + pose3d.npz + img/ or vid/."""
    return (
        (path / 'metadata.yaml').exists()
        and (path / 'pose3d.npz').exists()
        and ((path / 'img').exists() or (path / 'vid').exists())
    )



def main():
    args = parse_args()

    # resolve inference script path
    inference_script = args.inference_script or str(Path(__file__).parent / 'inference_video.py')
    if not os.path.exists(inference_script):
        print(f'ERROR: inference_video.py not found at {inference_script}')
        print('Use --inference-script to specify the path explicitly.')
        sys.exit(1)

    output_root = Path(args.output_root)

    # discover trials
    all_trials = list(find_trials(
        args.dataset_root,
        datasets=args.datasets,
        splits=args.splits,
        trials=args.trials,
    ))

    if len(all_trials) == 0:
        print(f'No valid trial directories found under {args.dataset_root}')
        if args.datasets:
            print(f'  (filtered to datasets : {args.datasets})')
        if args.splits:
            print(f'  (filtered to splits   : {args.splits})')
        if args.trials:
            print(f'  (filtered to trials   : {args.trials})')
        return

    print(f'Found {len(all_trials)} trial(s)')
    if args.datasets:
        print(f'  datasets : {args.datasets}')
    if args.splits:
        print(f'  splits   : {args.splits}')
    if args.trials:
        print(f'  trials   : {args.trials}')
    print(f'  output   : {output_root}')
    print()

    n_skipped = n_failed = n_computed = 0

    for dataset_name, split_name, subject_name, trial_name, trial_path in tqdm(all_trials, desc='Trials'):
        # build output path, omitting subject if None (3-level structure)
        path_parts = [p for p in [dataset_name, split_name, subject_name, trial_name] if p is not None]
        label      = '/'.join(path_parts)
        out_dir    = output_root.joinpath(*path_parts)
        outpath    = out_dir / args.output_name

        if outpath.exists() and not args.force:
            tqdm.write(f'  skip  {label}  (already exists)')
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, inference_script,
            '--base-folder',  args.base_folder,
            '--trial-path',   str(trial_path),
            '--start-frame',  str(args.start_frame),
            '--n-frames',     str(args.n_frames) if args.n_frames is not None else '999999',
            '--n-overlap',    str(args.n_overlap),
            '--outpath',      str(outpath),
        ]
        if args.checkpoint is not None:
            cmd += ['--checkpoint', str(args.checkpoint)]
        if args.device is not None:
            cmd += ['--device', args.device]
        if args.n_views is not None:
            cmd += ['--n-views', str(args.n_views)]
        if args.view_seed is not None:
            cmd += ['--view-seed', str(args.view_seed)]
        if args.max_kpts is not None:
            cmd += ['--max-kpts', str(args.max_kpts)]
        if args.clip_len is not None:
            cmd += ['--clip-len', str(args.clip_len)]
        if args.per_subject:
            cmd += ['--per-subject']

        tqdm.write(f'  run   {label}')

        try:
            subprocess.run(cmd, check=True)
            n_computed += 1
        except subprocess.CalledProcessError as e:
            tqdm.write(f'  FAILED {label}: inference_video.py exited with code {e.returncode}')
            n_failed += 1
        except Exception as e:
            tqdm.write(f'  FAILED {label}: {e}')
            traceback.print_exc()
            n_failed += 1

    print(f'\nDone: {n_computed} computed, {n_skipped} skipped, {n_failed} failed')


if __name__ == '__main__':
    main()