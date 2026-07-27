#!/usr/bin/env python
"""Does memory help TRACKING? A paired long-video A/B.

`scripts/eval_memory.py` cannot answer this. It is clip-level over 24 frames, and the anchored
arms come back with `survival_rate == 1.0000` at every memory size -- no point is ever lost, so
memory has nothing to recover. Memory's tracking job is holding identity ACROSS chunks, which
only exists over a long video.

Paired by construction: the same trial, the same window, the same `seed` (so the same camera
subset and the same point subsample), differing only in the arm's flag. A per-point paired delta
has far less variance than two independent runs, which is what makes a handful of subjects enough.

Arms:
  memory-off / memory-on     does memory help at all
  per_subject on/off         what per-subject cropping is worth. A rat-city animal spans 256px
                             in its own crop and 13px in a joint one -- smaller than one VJEPA
                             patch (16), so an entry cannot encode which point it is.

The headline is NOT the aggregate: it is error bucketed by elapsed frames since each point's
query frame. Memory should pay increasingly with distance from the anchor, so the on/off gap
should WIDEN with elapsed time. A flat curve means memory is not doing its tracking job whatever
the aggregate says.

  pixi run python scripts/eval_memory_video.py --base-folder <wandb run> --out /tmp/ab
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.inference.inference_utils import (load_model_from_base_folder,  # noqa: E402
                                                run_inference)

PREFIX = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4'

# Trials picked for signal, not coverage. 3dpop is split BY SUBJECT COUNT (Pigeon01/02/05/10 =
# 1/2/5/10 birds on the same rig, same task), so Pigeon01-vs-Pigeon05 is a controlled contrast
# for the crop hypothesis rather than just a second example.
TRIALS = {
    'rat-city': dict(
        path=f'{PREFIX}/rat-city/test/cohort7_20251209_1659/ix71493',
        max_subjects=6,     # of 12; the windowed loop runs once per subject
        has_vis_gt=False,   # 2D trials never load `vis`, so GT visibility does not exist
    ),
    '3dpop-1': dict(
        path=f'{PREFIX}/3dpop/test/Pigeon01/Sequence1_n01_01072022',
        max_subjects=1, has_vis_gt=True,
    ),
    '3dpop-5': dict(
        path=f'{PREFIX}/3dpop/test/Pigeon05/Sequence10_n05_01072022',
        max_subjects=3,     # of 5
        has_vis_gt=True,
    ),
}

# Elapsed-frame buckets. Memory's value should grow left-to-right; the last bucket is where an
# anchor is most stale and identity is most likely to have drifted.
BUCKETS = [(0, 32), (32, 96), (96, 256), (256, 10 ** 9)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-folder', required=True, help='wandb run dir (config + checkpoints)')
    p.add_argument('--out', required=True, help='output dir for predictions + summary.json')
    p.add_argument('--checkpoint', type=int, default=None)
    p.add_argument('--trials', nargs='+', default=list(TRIALS), choices=list(TRIALS))
    p.add_argument('--n-frames', type=int, default=900,
                   help='cap; the trial is used whole when shorter')
    p.add_argument('--n-overlap', type=int, default=8)
    p.add_argument('--memory-context', type=int, default=8)
    p.add_argument('--max-points', type=int, default=600, help='per subject')
    p.add_argument('--seed', type=int, default=0,
                   help='SHARED by every arm -- this is what makes the comparison paired')
    p.add_argument('--crop-arm', action='store_true',
                   help='also run a per_subject=False arm, pricing per-subject cropping')
    p.add_argument('--device', default=None)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def arms(args):
    """(name, use_memory, per_subject, group).

    Pairing only holds WITHIN a group: `per_subject` changes which points are tracked at all
    (per-subject concatenates each subject's valid mask, the flat path does not), so the two
    settings return different point sets and cannot be differenced point-by-point. Each group
    therefore carries its own memory-off baseline, which defines that group's elapsed-time axis
    and occlusion event set.
    """
    out = [('mem_off', False, True, 'per_subject'),
           ('mem_on', True, True, 'per_subject')]
    if args.crop_arm:
        out += [('mem_off_nocrop', False, False, 'joint'),
                ('mem_on_nocrop', True, False, 'joint')]
    return out


def elapsed_frames(out):
    """(T, K) frames since each point's query frame, or None when query times are absent."""
    if 'query_times' not in (out.files if hasattr(out, 'files') else out):
        return None
    qt = np.asarray(out['query_times']).astype(np.int64)              # (K,)
    T = np.asarray(out['coords_pred']).shape[1]
    return np.arange(T)[:, None] - qt[None, :]                        # (T, K)


def per_point_error(out):
    """(T, K) euclidean error, NaN where GT is absent or the point has not appeared."""
    cp = np.asarray(out['coords_pred'])[0].astype(np.float64)         # (T, K, R)
    ct = np.asarray(out['coords_true'])[0].astype(np.float64)
    n = min(cp.shape[0], ct.shape[0])
    return np.linalg.norm(cp[:n] - ct[:n], axis=-1)                   # (T, K)


def occluded_mask(out):
    """Points the BASELINE arm predicts as invisible at some frame -> (K,) bool.

    Uses predicted visibility, not GT: 2D trials have no `vis`, and inference already runs on
    predicted occlusion (`occ_chunk = occ_pred.clone()` for every chunk after the first). The
    event set must come from ONE fixed arm and then be applied to all of them -- deriving it
    per-arm would give the arms different point sets, which breaks the pairing and is circular,
    since memory changes the very visibility predictions it would be judged on.
    """
    vp = np.asarray(out['vis_pred'])[0]                               # (T, K, 1)
    return (1.0 / (1.0 + np.exp(-vp[..., 0])) < 0.5).any(axis=0)      # (K,)


def summarize(err, elapsed, keep=None):
    """Mean error per elapsed-frame bucket, over the kept points."""
    row = {}
    m = np.isfinite(err)
    if keep is not None:
        m = m & keep[None, :]
    for lo, hi in BUCKETS:
        sel = m & (elapsed >= lo) & (elapsed < hi) if elapsed is not None else m
        row[f'{lo}-{hi if hi < 10 ** 9 else "inf"}'] = (
            float(err[sel].mean()) if sel.any() else float('nan'))
        if elapsed is None:
            break
    row['all'] = float(err[m].mean()) if m.any() else float('nan')
    row['n_points'] = int(keep.sum()) if keep is not None else int(err.shape[1])
    return row


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device) if args.device else None
    model, config, config_path, ckpt = load_model_from_base_folder(
        args.base_folder, checkpoint=args.checkpoint, device=device)
    print(f'checkpoint: {os.path.basename(ckpt)}')
    if not getattr(model, 'memory_attention', False):
        sys.exit('model has no memory_attention; the A/B has nothing to compare')

    summary = {}
    for name in args.trials:
        spec = TRIALS[name]
        if not os.path.isdir(spec['path']):
            print(f'[skip] {name}: {spec["path"]} not found')
            continue
        print(f'\n=== {name} ===')
        preds, groups = {}, {}
        for arm, use_memory, per_subject, group in arms(args):
            groups[arm] = group
            cache = os.path.join(args.out, f'{name}.{arm}.npz')
            if os.path.exists(cache) and not args.force:
                print(f'  [{arm}] cached')
                preds[arm] = np.load(cache, allow_pickle=True)
                continue
            t0 = time.time()
            print(f'  [{arm}] use_memory={use_memory} per_subject={per_subject}')
            out = run_inference(
                model=model, config_path=config_path, checkpoint_path=ckpt,
                trial_path=spec['path'], start_frame=0, n_frames=args.n_frames,
                n_overlap=args.n_overlap, seed=args.seed, per_subject=per_subject,
                max_points=args.max_points, max_subjects=spec['max_subjects'],
                device=device, outpath=cache, use_memory=use_memory,
                memory_context=args.memory_context, query_first=True)
            print(f'      {time.time() - t0:.0f}s')
            preds[arm] = out
            if device is None or torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows, n_occ = {}, {}
        for group in sorted(set(groups.values())):
            members = [a for a in preds if groups[a] == group]
            base_name = f'mem_off{"" if group == "per_subject" else "_nocrop"}'
            if base_name not in members:
                continue
            # Within a group the baseline defines the elapsed-time axis and the occlusion event
            # set, so its members are scored on identical points.
            base = preds[base_name]
            elapsed, occ = elapsed_frames(base), occluded_mask(base)
            for arm in members:
                err = per_point_error(preds[arm])
                if err.shape[1] != occ.shape[0]:
                    print(f'  [warn] {arm}: {err.shape[1]} points vs baseline {occ.shape[0]}; '
                          'skipping (arms in a group must share a point set)')
                    continue
                e = elapsed[:err.shape[0], :err.shape[1]] if elapsed is not None else None
                rows[arm] = {'group': group,
                             'all_points': summarize(err, e),
                             'through_occlusion': summarize(err, e, keep=occ)}
            n_occ[group] = int(occ.sum())
        if not rows:
            continue
        summary[name] = {'has_vis_gt': spec['has_vis_gt'], 'arms': rows,
                         'n_occluded_points': n_occ}

        first = next(iter(rows.values()))['all_points']
        print(f'  {"arm":16s} ' + ' '.join(f'{k:>12}' for k in first))
        for arm, r in rows.items():
            print(f'  {arm:16s} ' + ' '.join(
                f'{v:12.4f}' if isinstance(v, float) else f'{v:12d}'
                for v in r['all_points'].values()))

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'checkpoint': ckpt, 'buckets': [list(b) for b in BUCKETS],
                   'summary': summary}, f, indent=2)
    print(f'\nwrote {os.path.join(args.out, "summary.json")}')
    print('Headline: the mem_on/mem_off gap should WIDEN across the elapsed-frame buckets. '
          'A flat gap means memory is not holding identity across chunks.')
    print('NOTE: avg_jaccard/survival_rate are meaningless on trials without vis GT '
          '(vis_true is NaN there and casts to True), so this script reports error only.')


if __name__ == '__main__':
    main()
