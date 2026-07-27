#!/usr/bin/env python
"""Annotation transfer: label one animal, find the same keypoints on another.

The product this measures: a user labels the keypoints on one rat, boxes a second rat, and
gets its keypoints without re-annotating. That is memory-only inference -- appearance memory
of a point, no position -- and it is CROSS-INSTANCE, where every memory-only number so far
(12.1 mte on wpfb20a7) has been cross-TIME on the same animal.

Per (donor, target) subject pair: build the bank from the DONOR's keypoints in the donor's own
crop, then ask the model to place those points on the TARGET, and score against the target's
ground truth. Keypoint index k means the same thing on both (same skeleton, aligned indices),
so the correspondence is exact and needs no matching step.

The `same` arm is the control: donor == target, i.e. ordinary same-subject memory-only. The
gap between `same` and `cross` is the cross-instance generalization cost -- the thing
memory_cross_subject_prob exists to close.

  pixi run python scripts/eval_memory_transfer.py --base-folder <wandb run>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.inference.inference_utils import (build_chunk_memory,           # noqa: E402
                                                load_model_from_base_folder)
from posetail.posetail.cube import compute_cube_scale                          # noqa: E402
from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate  # noqa: E402
from posetail.posetail.train_utils import (dict_to_device, load_config,        # noqa: E402
                                           memory_raw_from_batch)

PREFIX = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-folder', required=True)
    p.add_argument('--checkpoint', type=int, default=None)
    p.add_argument('--split', default='val')
    p.add_argument('--datasets', nargs='+',
                   default=['rat-city', 'branson-fly', 'ravan-fish-sim', '3dpop'],
                   help='multi-subject datasets; single-subject ones cannot transfer')
    p.add_argument('--n-per-dataset', type=int, default=4)
    p.add_argument('--max-points', type=int, default=128)
    p.add_argument('--cross-prob', type=float, default=0.5,
                   help='fraction of samples whose memory comes from another individual; '
                        'both arms are filled from one pass so they share clips')
    p.add_argument('--device', default=None)
    p.add_argument('--outpath', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, _, ckpt = load_model_from_base_folder(
        args.base_folder, checkpoint=args.checkpoint, device=device)
    model.eval()
    if not getattr(model, 'memory_attention', False):
        sys.exit('model has no memory_attention; nothing to transfer with')
    print(f'checkpoint: {os.path.basename(ckpt)}')

    d = config.dataset[args.split]
    d['kpts_to_sample'] = args.max_points
    d['memory_prob'] = 1.0
    d['memory_only_prob'] = 1.0       # every point is memory-only: that IS the product mode
    d['memory_only_kpt_prob'] = 1.0
    d['balance_datasets'] = True
    d['n_samples_per_dataset'] = args.n_per_dataset
    d['aug_prob'] = 0.0
    # The val config does not carry this (it lives under [dataset.train]), so without an
    # explicit override the cross arm silently comes back empty. 0.5 so BOTH arms fill from
    # one pass and the pair is drawn from the same clips.
    d['memory_cross_subject_prob'] = args.cross_prob
    ds = PosetailDataset(config, split=args.split)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=custom_collate,
                                     shuffle=False, num_workers=6)

    # Both arms come from the dataset's own cross-subject switch, so the donor lookup, the
    # crop and the keypoint alignment are EXACTLY what training uses -- a reimplementation
    # here would be the usual way for an eval to drift from the thing it measures.
    acc = {'same': [], 'cross': []}
    with torch.inference_mode():
        for b in dl:
            if b is None or b.views is None or b.mem_views is None:
                continue
            if b.memory_only is None or not bool(b.memory_only.any()):
                continue
            arm = 'cross' if b.sample_info.get('cross_subject') else 'same'
            views = [v.to(device) for v in b.views]
            qt = b.query_times.to(device)
            cg = [dict_to_device(c, device) for c in b.cgroup]
            coords = b.coords.to(device)
            N = coords.shape[2]
            is2d = b.p2d is not None
            qc = (b.p2d.to(device)[:, 0, qt[0], torch.arange(N)] if is2d
                  else coords[:, qt[0], torch.arange(N)])
            cs = None if is2d else compute_cube_scale(
                cg, qc, len(cg), device,
                per_camera=getattr(model, 'per_camera_cube_scale', False))
            raw = memory_raw_from_batch(model, b)
            # withhold every position: the model has only the memory
            qc_unknown = torch.full_like(qc, float('nan'))
            kw = {} if cs is None else {'cube_scale': cs}
            out = model(views=views, coords=qc_unknown, query_times=qt,
                        camera_group=cg, memory_raw=raw, **kw)
            cp = out['2d_pred'][0] if is2d else out['coords_pred']
            ct = (b.p2d.to(device)[:, 0] if is2d else coords)
            n = min(cp.shape[1], ct.shape[1])
            err = torch.linalg.norm(cp[:, :n] - ct[:, :n], dim=-1)
            err = err[torch.isfinite(err)]
            if err.numel():
                acc[arm].append((b.sample_info.get('dataset', '?'), float(err.mean())))

    print(f'\n{"arm":8s} {"n":>4} {"mean err":>10}')
    summary = {}
    for arm, rows in acc.items():
        if not rows:
            print(f'{arm:8s} {0:4d} {"-":>10}')
            continue
        v = float(np.mean([r[1] for r in rows]))
        summary[arm] = {'n': len(rows), 'mean_err': v,
                        'per_dataset': {k: float(np.mean([r[1] for r in rows if r[0] == k]))
                                        for k in {r[0] for r in rows}}}
        print(f'{arm:8s} {len(rows):4d} {v:10.4f}')
    for arm in ('same', 'cross'):
        if arm in summary:
            print(f'\n{arm}:')
            for k, v in sorted(summary[arm]['per_dataset'].items()):
                print(f'   {k:22s} {v:10.4f}')
    print('\n`cross` is the product metric. The same/cross gap is the cross-instance cost '
          'that\nmemory_cross_subject_prob exists to close; watch it shrink across training.')
    if args.outpath:
        with open(args.outpath, 'w') as f:
            json.dump({'checkpoint': ckpt, 'summary': summary}, f, indent=2)
        print(f'wrote {args.outpath}')


if __name__ == '__main__':
    main()
