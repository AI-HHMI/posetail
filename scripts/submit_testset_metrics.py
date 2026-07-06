#!/usr/bin/env python
"""Submit the test-set metric eval (scripts/eval_testset_metrics.py) to the LSF cluster.

Give it a run (by --runid, or a full --wandb-folder) and the datasets; it resolves the wandb
folder, names the output folder, and submits a self-contained LSF job that runs the eval under
pixi -- including the LD_LIBRARY_PATH=$CONDA_PREFIX/lib workaround (compute nodes' system
libstdc++ lacks CXXABI_1.3.15 that scipy/highspy needs; the pixi env's libstdc++ has it).

Examples:
    # submit all datasets for run 1d1r24ff (project defaults to posetail-finetuning-v3):
    pixi run python scripts/submit_testset_metrics.py --runid 1d1r24ff

    # just print the job script, don't submit:
    pixi run python scripts/submit_testset_metrics.py --runid 1d1r24ff --dry-run

    # override the wandb folder / datasets / output:
    pixi run python scripts/submit_testset_metrics.py --wandb-folder <run_dir> \\
        --datasets dex_ycb --out <dir>
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

RESULTS_ROOT = '/groups/karashchuk/home/karashchukl/results'
DEFAULT_PROJECT = 'posetail-finetuning-v3'
ALL_DATASETS = ['dex_ycb', 'kubric-multiview', 'cmupanoptic_3dgs']
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_run(args):
    """Return (wandb_folder, runid)."""
    if args.wandb_folder:
        folder = os.path.abspath(os.path.expanduser(args.wandb_folder))
        if not os.path.isdir(folder):
            sys.exit(f'--wandb-folder not found: {folder}')
        return folder, os.path.basename(folder.rstrip('/')).split('-')[-1]
    if not args.runid:
        sys.exit('provide --runid (or --wandb-folder)')
    pattern = os.path.join(RESULTS_ROOT, args.wandb_project, 'wandb', f'run-*-{args.runid}')
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f'no wandb folder matching {pattern}')
    if len(matches) > 1:
        sys.exit('multiple wandb folders match runid {}:\n  '.format(args.runid)
                 + '\n  '.join(matches))
    return matches[0], args.runid


def build_job_script(args, wandb_folder, runid, out_dir):
    log_dir = os.path.expanduser('~/logs/posetail')
    # eval passthrough flags
    ev = ['--datasets', *args.datasets, '--out', out_dir,
          '--wandb-folder', wandb_folder, '--device', args.device]
    if args.checkpoint is not None:
        ev += ['--checkpoint', str(args.checkpoint)]
    if args.force:
        ev += ['--force']
    if args.max_kpts is not None:
        ev += ['--max-kpts', str(args.max_kpts)]
    if args.n_views is not None:
        ev += ['--n-views', str(args.n_views)]
    eval_args = ' '.join(ev)
    # inner command runs inside the pixi env; $CONDA_PREFIX is set there, so the libstdc++ fix
    # must be applied inside `pixi run`. Single-quoted so it expands on the compute node.
    inner = ("export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
             f"exec python scripts/eval_testset_metrics.py {eval_args}")
    return f"""#!/bin/bash
#BSUB -J eval-{runid}
#BSUB -e {log_dir}/eval-{runid}.err
#BSUB -o {log_dir}/eval-{runid}.out
#BSUB -n {args.cores}
#BSUB -q {args.queue}
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={args.gpus}"
#BSUB -W {args.walltime}
cd {PROJECT_DIR}
pixi run bash -c '{inner}'
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runid', help='wandb run id, e.g. 1d1r24ff')
    ap.add_argument('--wandb-project', default=DEFAULT_PROJECT,
                    help=f'results subdir holding wandb/ (default: {DEFAULT_PROJECT})')
    ap.add_argument('--wandb-folder', help='explicit wandb run dir (overrides --runid/--wandb-project)')
    ap.add_argument('--datasets', nargs='+', default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument('--out', help='output/predictions dir '
                    '(default: ~/ghome/results/posetail-inference/testset-eval-<runid>)')
    # eval passthrough
    ap.add_argument('--checkpoint', type=int, default=None, help='default: latest checkpoint')
    ap.add_argument('--force', action='store_true', help='recompute even if predictions exist')
    ap.add_argument('--max-kpts', type=int, default=None)
    ap.add_argument('--n-views', type=int, default=None)
    ap.add_argument('--device', default='cuda:0')
    # LSF knobs
    ap.add_argument('--queue', default='gpu_a100')
    ap.add_argument('--cores', type=int, default=12)
    ap.add_argument('--walltime', default='6:00')
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true', help='print the job script, do not submit')
    args = ap.parse_args()

    wandb_folder, runid = resolve_run(args)
    out_dir = os.path.abspath(os.path.expanduser(
        args.out or f'~/ghome/results/posetail-inference/testset-eval-{runid}'))
    job = build_job_script(args, wandb_folder, runid, out_dir)

    print(f'run id      : {runid}')
    print(f'wandb folder: {wandb_folder}')
    print(f'datasets    : {" ".join(args.datasets)}')
    print(f'output dir  : {out_dir}')
    print('----- LSF job script -----')
    print(job, end='')
    print('--------------------------')

    if args.dry_run:
        print('[dry-run] not submitted -- drop --dry-run to submit via bsub.')
        return

    if shutil.which('bsub') is None:
        sys.exit('bsub not found on PATH -- run this on an LSF submit host (or use --dry-run).')
    os.makedirs(os.path.expanduser('~/logs/posetail'), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    res = subprocess.run(['bsub'], input=job, text=True)
    sys.exit(res.returncode)


if __name__ == '__main__':
    main()
