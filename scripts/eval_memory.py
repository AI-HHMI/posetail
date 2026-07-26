#!/usr/bin/env python
"""Does the memory actually help? A controlled, clip-level comparison.

Runs the SAME clips and the SAME points through the model several times, varying only how
many memory entries the decoder may read, and reports the standard metric set per amount of
memory and per dataset. The number that matters is the delta between `M=0` (memory bypassed
entirely) and `M>0` -- a single with-memory number cannot tell you the memory path learned
anything.

This is clip-level (straight through the dataloader), not full-video windowed inference, so
it is cheap enough to run against a training checkpoint while a run is in flight.

The memory bank is built ONCE per batch and sliced for each arm, so the memory ViT encode is
not repeated. Each remembered frame contributes one entry PER CAMERA, so k frames is
`bank[:, :, :k*n_cams]` -- nothing in the model is sized by the entry count, it is read from
the tensor shape.

Usage:
    # the two arms that answer "does memory help at all"
    pixi run python scripts/eval_memory.py \\
        --base-folder /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX \\
        --n-per-dataset 5

    # sweep how much memory helps, and include memory-only queries
    pixi run python scripts/eval_memory.py --base-folder <run> \\
        --contexts 0 2 4 8 --memory-only --outpath memory_eval.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate  # noqa: E402
from posetail.inference.inference_utils import (build_chunk_memory,            # noqa: E402
                                                load_model_from_base_folder)
from posetail.posetail.cube import compute_cube_scale                            # noqa: E402
from posetail.posetail.eval_metrics import get_eval_metrics                     # noqa: E402
from posetail.posetail.train_utils import (_eval_cube_scale, dict_to_device,    # noqa: E402
                                           memory_kwargs, memory_only_kwargs)

METRICS = ['mte', 'delta_x_avg', 'avg_jaccard', 'survival_rate', 'mpjpe']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-folder', required=True,
                   help='wandb run folder holding config.toml + checkpoints')
    p.add_argument('--checkpoint', type=int, default=None,
                   help='checkpoint step; default = latest')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--contexts', type=int, nargs='+', default=[0, 8],
                   help='memory entry counts to evaluate. 0 = memory bypassed entirely. '
                        'Each extra value costs one more forward pass.')
    p.add_argument('--n-per-dataset', type=int, default=5,
                   help='clips per dataset -- the main dial for total runtime')
    p.add_argument('--max-points', type=int, default=600,
                   help='cap on tracked points per sample')
    p.add_argument('--memory-only', action='store_true',
                   help='also withhold some query positions, so those points must be found '
                        'from memory alone; reported separately and only for M>0')
    p.add_argument('--memory-source', default='dataset', choices=['dataset', 'chunk'],
                   help="where remembered frames come from. 'dataset' (default): sampled "
                        'from anywhere in the video with their own crops and encoded in '
                        'isolation -- the TRAINING distribution, and the only source that '
                        "tests long-range memory. 'chunk': remembered from the clip itself "
                        'and read out of the clip encode via build_chunk_memory -- the '
                        'literal INFERENCE path. The pair brackets the train/inference gap.')
    p.add_argument('--num-workers', type=int, default=None,
                   help='dataloader workers (default: from config). Memory adds M x n_cams '
                        'image reads per sample, so the loader is often the bottleneck.')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--outpath', default=None, help='optional JSON dump')
    return p.parse_args()


def build_bank(model, batch, device, cube_scale=None):
    """Encode this batch's remembered observations once, so the M sweep can slice the result.

    Training hands the RAW observations to forward() and lets it encode the bank internally,
    because encoding outside would mean calling build_memory_bank on the DDP-wrapped module
    -- which routes through DDP's forward and desynced the per-rank collective count (see
    train_utils.memory_kwargs). This script is single-process and the model is not wrapped,
    so calling it directly is safe here, and it means the frame encode happens once per batch
    instead of once per arm.

    The batch unpacking is `memory_kwargs`, deliberately: training, this script, diag_memory
    and inference must not drift apart on how a memory observation is assembled.
    """
    kw = memory_kwargs(model, batch, device)
    if not kw:
        return None
    return model.build_memory_bank(**kw, device=device, cube_scale=cube_scale)


def build_bank_from_chunk(model, batch, views, cgroup, coords, device, n_ctx,
                          cube_scale=None):
    """The INFERENCE memory path: remember frames from the clip itself and read their tokens
    straight out of the clip encode, exactly as build_chunk_memory does while tracking.

    Frames are spread evenly over the clip. This measures a different thing from the dataset
    source -- memory from within the window, not from across the video -- so the two are
    reported as separate arms, never mixed.
    """
    T = views[0].shape[1]
    K = min(n_ctx, T)
    frame_idx = torch.linspace(0, T - 1, K, device=device).round().long()
    is_2d = coords.shape[-1] == 2
    scene_features = model.encode_scene(model._normalize_views(views, device))
    coords_at = coords[:, frame_idx]                              # (B, K, N, R)
    return build_chunk_memory(model, views, cgroup, frame_idx, coords_at, is_2d,
                              cube_scale=cube_scale, scene_features=scene_features)


def build_dataset(config, args):
    """Apply the eval overrides, then build the dataset. All of these are existing config
    knobs -- n_samples_per_dataset in particular gives even per-dataset coverage AND bounds
    the total work."""
    split = args.split
    ds_cfg = config.dataset[split]
    ds_cfg['kpts_to_sample'] = args.max_points
    ds_cfg['memory_prob'] = 1.0
    ds_cfg['memory_num_context'] = max(args.contexts) if max(args.contexts) > 0 else 2
    ds_cfg['memory_only_prob'] = 1.0 if args.memory_only else 0.0
    ds_cfg['balance_datasets'] = True
    ds_cfg['n_samples_per_dataset'] = args.n_per_dataset
    ds_cfg['aug_prob'] = 0.0
    return PosetailDataset(config, split=split)


def metrics_for(coords_pred, coords_true, vis_pred, vis_true, cube_scale, keep=None):
    """Standard metric set, optionally restricted to a subset of points (used to split
    memory-only points from normally-queried ones). Point axis is dim 2."""
    if keep is not None:
        if not bool(keep.any()):
            return None
        coords_pred = coords_pred[:, :, keep]
        coords_true = coords_true[:, :, keep]
        vis_pred = vis_pred[:, :, keep] if vis_pred is not None else None
        vis_true = vis_true[:, :, keep] if vis_true is not None else None
    md = get_eval_metrics(
        vis_pred=vis_pred, vis_true=vis_true,
        coords_pred=coords_pred, coords_true=coords_true,
        prefix='', cube_scale=cube_scale)
    return {k: float(v) for k, v in md.items() if k in METRICS}


def fmt_table(rows, title):
    """rows: {label: {metric: value}} -> a small aligned table."""
    if not rows:
        return f'{title}\n  (no data)\n'
    out = [title, '  ' + f'{"M":>4}  ' + '  '.join(f'{m:>13}' for m in METRICS)]
    for label, md in rows.items():
        cells = '  '.join(f'{md.get(m, float("nan")):>13.4f}' for m in METRICS)
        out.append('  ' + f'{label:>4}  ' + cells)
    return '\n'.join(out) + '\n'


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, _, ckpt_path = load_model_from_base_folder(
        args.base_folder, checkpoint=args.checkpoint, device=device)
    model.eval()
    if not getattr(model, 'memory_attention', False):
        sys.exit('this model was built without memory_attention; nothing to evaluate')
    print(f'checkpoint: {os.path.basename(ckpt_path)}')
    print(f'contexts:   {args.contexts}   split: {args.split}   '
          f'points/sample: {args.max_points}   clips/dataset: {args.n_per_dataset}')

    dataset = build_dataset(config, args)
    loader = DataLoader(
        dataset, batch_size=config.dataset.batch_size, collate_fn=custom_collate,
        shuffle=False,
        num_workers=(args.num_workers if args.num_workers is not None
                     else config.dataset.num_workers),
        pin_memory=True)

    # acc[arm][dataset] -> list of metric dicts;  arm is 'M=k' or 'M=k/memonly'
    acc = defaultdict(lambda: defaultdict(list))
    n_seen = n_yielded = n_unreadable = 0
    skipped_ds = defaultdict(int)   # datasets with no memory path (native-2D trials)
    t0 = time.time()

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            n_yielded += 1
            # A sample can fail to load (unreadable trial, missing frame); custom_collate
            # yields None and the dataset name is gone with it, so these CANNOT be
            # attributed per-dataset -- counted separately rather than silently merged
            # into the no-memory-path tally, which made the coverage line misleading.
            if batch is None or batch.views is None:
                n_unreadable += 1
                continue
            views = [v.to(device) for v in batch.views]
            coords = batch.coords.to(device)
            vis = batch.vis.to(device) if batch.vis is not None else None
            # Training gets this for free from fabric.setup_dataloaders, which moves batch
            # tensors to the device; a plain DataLoader does not, and query_times feeds a
            # torch.gather inside sample_patches that requires a matching device.
            query_times = batch.query_times.to(device)
            cgroup = ([dict_to_device(c, device) for c in batch.cgroup]
                      if batch.cgroup else batch.cgroup)
            p2d = batch.p2d.to(device) if batch.p2d is not None else None
            ds_name = batch.sample_info.get('dataset', 'unknown')

            if p2d is None:
                query_coords = coords[:, query_times[0], torch.arange(len(query_times[0]))]
            else:
                query_coords = p2d[:, 0, query_times[0], torch.arange(len(query_times[0]))]

            # Build the memory bank ONCE; each arm just slices it.
            n_cams = len(batch.mem_views) if batch.mem_views is not None else 1
            mem_cube_scale = (None if p2d is not None else compute_cube_scale(
                cgroup, query_coords, len(cgroup), device,
                per_camera=getattr(model, 'per_camera_cube_scale', False)))
            if args.memory_source == 'chunk':
                # A batch with no dataset memory also has no memory-only mask, so keep the
                # same skip rule -- otherwise the two sources would run on different clips
                # and the comparison between them would be meaningless.
                if batch.mem_views is None:
                    skipped_ds[ds_name] += 1
                    continue
                bank = build_bank_from_chunk(
                    model, batch, views, cgroup,
                    (coords if p2d is None else p2d[:, 0]), device,
                    max(args.contexts), cube_scale=mem_cube_scale)
                n_cams = len(cgroup)
            else:
                bank = build_bank(model, batch, device, cube_scale=mem_cube_scale)
            if bank is None:                 # dataset sampled no memory (e.g. 2D-only trial)
                skipped_ds[ds_name] += 1
                continue

            qc, mo_kw = memory_only_kwargs(model, batch, query_coords, cgroup, p2d is not None)
            mo_mask = getattr(batch, 'memory_only', None)
            if mo_mask is not None:
                mo_mask = mo_mask.to(device)[0]
            cube_scale = _eval_cube_scale(cgroup, query_coords)

            base_kwargs = dict(views=views, query_times=query_times, camera_group=cgroup)
            if getattr(model, 'occlusion_embedding', False):
                base_kwargs['occlusion'] = batch.query_occlusion.to(device)
            base_kwargs.update(mo_kw)

            for k in args.contexts:
                # A memory-only point has no query at all, so it cannot be run without
                # memory -- fall back to the intact queries for the k=0 arm.
                if k == 0:
                    if mo_mask is not None and bool(mo_mask.any()):
                        coords_in, mem = query_coords, None
                    else:
                        coords_in, mem = qc, None
                else:
                    coords_in, mem = qc, bank[:, :, :k * n_cams]

                out = model(coords=coords_in, memory_bank=mem, **base_kwargs)
                cp, vp = out['coords_pred'], out['vis_pred']

                if mo_mask is None or not bool(mo_mask.any()):
                    m = metrics_for(cp, coords, vp, vis, cube_scale)
                    if m:
                        acc[f'{k}'][ds_name].append(m)
                else:
                    m = metrics_for(cp, coords, vp, vis, cube_scale, keep=~mo_mask)
                    if m:
                        acc[f'{k}'][ds_name].append(m)
                    if k > 0:      # memory-only points are meaningless without memory
                        mm = metrics_for(cp, coords, vp, vis, cube_scale, keep=mo_mask)
                        if mm:
                            acc[f'{k}*'][ds_name].append(mm)

            n_seen += 1
            if n_seen % 10 == 0:
                print(f'  {n_seen} clips  ({time.time() - t0:.0f}s)', flush=True)

    if not n_seen:
        sys.exit('no evaluable clips (every sample was skipped)')

    def agg(dicts):
        return {m: float(np.nanmean([d[m] for d in dicts if m in d])) for m in METRICS}

    overall = {arm: agg([m for ms in per_ds.values() for m in ms])
               for arm, per_ds in acc.items()}
    per_dataset = {
        ds: {arm: agg(acc[arm][ds]) for arm in sorted(acc) if acc[arm][ds]}
        for ds in sorted({d for per_ds in acc.values() for d in per_ds})
    }

    n_nomem = sum(skipped_ds.values())
    skip_note = ('  (no memory path: ' + ', '.join(f'{k} x{v}' for k, v in sorted(skipped_ds.items())) + ')') if skipped_ds else ''
    print(f'\n{n_seen} clips evaluated, {n_nomem} without a memory path, '
          f'{n_unreadable} unreadable, of {n_yielded} sampled, {time.time() - t0:.0f}s'
          f'{skip_note}')
    # The three buckets must exhaust what the loader produced, or the coverage line is
    # lying about which datasets were actually measured.
    assert n_seen + n_nomem + n_unreadable == n_yielded, (
        f'skip accounting does not close: {n_seen}+{n_nomem}+{n_unreadable} != {n_yielded}')
    missing = sorted(set(dataset.metadata['dataset'].unique()) - set(acc['0'].keys())
                     - set(skipped_ds))
    if missing:
        print(f'  WARNING: sampled but never evaluated or attributed: {", ".join(missing)}')
    print()
    print(fmt_table({a: overall[a] for a in sorted(overall)}, 'OVERALL  (M* = memory-only points)'))
    for ds, rows in per_dataset.items():
        print(fmt_table(rows, ds))

    lo, hi = f'{min(args.contexts)}', f'{max(args.contexts)}'
    if lo in overall and hi in overall and lo != hi:
        d = overall[lo]['mte'] - overall[hi]['mte']
        print(f'memory effect: mte {overall[lo]["mte"]:.4f} (M={lo}) -> '
              f'{overall[hi]["mte"]:.4f} (M={hi})   delta {d:+.4f} '
              f'({"memory helps" if d > 0 else "no gain from memory"})')

    if args.outpath:
        with open(args.outpath, 'w') as f:
            json.dump({'checkpoint': ckpt_path, 'contexts': args.contexts,
                       'n_clips': n_seen, 'overall': overall,
                       'per_dataset': per_dataset}, f, indent=2)
        print(f'wrote {args.outpath}')


if __name__ == '__main__':
    main()
