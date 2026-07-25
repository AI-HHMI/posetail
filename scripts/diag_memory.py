#!/usr/bin/env python
"""Where does the memory pathway lose per-point information?

Memory-only tracking was found to land in the right region but not on the right point. The
cause was capacity, not mechanism: the per-point codes spanned only ~2-3 effective dimensions
while the ordinary query token (which localizes well) spanned ~5. This script measures that
directly, stage by stage, so a change can be judged BEFORE spending a training run.

Effective rank here is the participation ratio of the singular spectrum -- "how many
dimensions are really being used" by the N points of a sample. If it is far below N, the
model cannot tell those points apart no matter how well the rest is trained.

  pixi run python scripts/diag_memory.py --base-folder <wandb run>    # a trained checkpoint
  pixi run python scripts/diag_memory.py --config configs/config_encoder_memory.toml
        # an UNTRAINED model straight from the config -- measures capacity, which is the
        # cheap pre-flight check after an architecture change
"""
import argparse
import os
import sys

import numpy as np
import toml
import torch
from easydict import EasyDict as edict
from einops import rearrange
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate  # noqa: E402
from posetail.inference.inference_utils import load_model_from_base_folder      # noqa: E402
from posetail.posetail.cube import compute_cube_scale                           # noqa: E402
from posetail.posetail.tracker_encoder import TrackerEncoder                    # noqa: E402
from posetail.posetail.train_utils import dict_to_device, load_config           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--base-folder', help='wandb run folder (trained checkpoint)')
    g.add_argument('--config', help='config .toml -> UNTRAINED model (capacity only)')
    p.add_argument('--checkpoint', type=int, default=None)
    p.add_argument('--split', default='val')
    p.add_argument('--n-samples', type=int, default=6)
    p.add_argument('--min-points', type=int, default=24,
                   help='skip samples with too few points for rank to be meaningful')
    p.add_argument('--encoder-version', default=None,
                   help="override e.g. 'base' to keep an untrained model small")
    p.add_argument('--device', default=None)
    return p.parse_args()


def bank_summary(bank, valid):
    """Per-point view of the bank, for a rank measurement.

    Deliberately does NOT average the entries. Averaging is the very operation that was
    found to destroy per-point structure (it is why the camera pool was removed), so using
    it to measure the bank would build the collapse into the metric. The decoder ATTENDS
    over the entries, so the information available to it is the whole set -- flatten it and
    let the rank speak for the set.

    bank: (N, M*cams, dim)  ->  (N, M*cams*dim)
    """
    return bank.reshape(bank.shape[0], -1)


def eff_rank(x):
    """Participation ratio of the singular spectrum of the per-point codes."""
    x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = s ** 2 / (s ** 2).sum().clamp(min=1e-12)
    return float(1.0 / (p ** 2).sum())


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.base_folder:
        model, config, _, ck = load_model_from_base_folder(
            args.base_folder, checkpoint=args.checkpoint, device=device)
        tag = f'trained: {os.path.basename(ck)}'
    else:
        config = load_config(args.config)
        m = dict(config.model)
        if args.encoder_version:
            m['video_encoder_version'] = args.encoder_version
        model = TrackerEncoder(**m).to(device)
        tag = f'UNTRAINED from {os.path.basename(args.config)} (capacity only)'
    model.eval()
    if not getattr(model, 'memory_attention', False):
        sys.exit('model has no memory_attention; nothing to diagnose')
    print(tag)

    d = config.dataset[args.split]
    d['kpts_to_sample'] = 128
    d['memory_prob'] = 1.0
    d['memory_num_context'] = 8
    d['memory_only_prob'] = 0.0
    d['balance_datasets'] = True
    d['n_samples_per_dataset'] = 2
    d['aug_prob'] = 0.0
    ds = PosetailDataset(config, split=args.split)
    dl = DataLoader(ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=4)

    me = model.memory_encoder
    rows = []
    with torch.inference_mode():
        for b in dl:
            if b.mem_views is None or b.p2d is not None:      # 3D path only
                continue
            N = b.mem_p2d.shape[3]
            if N < args.min_points:
                continue
            mv = model._normalize_views([v.to(device) for v in b.mem_views], device)
            p2d, val = b.mem_p2d.to(device), b.mem_valid.to(device)
            coords, qt = b.coords.to(device), b.query_times.to(device)
            cg = [dict_to_device(c, device) for c in b.cgroup]
            qc = coords[:, qt[0], torch.arange(N)]
            cs = compute_cube_scale(cg, qc, len(cg), device,
                                    per_camera=getattr(model, 'per_camera_cube_scale', False))

            # --- stage by stage, for camera 0 ---
            v = rearrange(mv[0], 'b m c h w -> (b m) c h w')
            p = rearrange(p2d[0], 'b m n r -> (b m) n r')
            ok = rearrange(val[0], 'b m n -> (b m) n')
            dep = itr = None
            if b.mem_depth is not None:
                dep = rearrange(b.mem_depth.to(device)[0], 'b m n -> (b m) n') / cs[0].clamp(min=1e-6)
                itr = rearrange(b.mem_intrinsics.to(device)[0], 'b m r -> (b m) r')
            q_seed = me._query(v, p, ok, depth=dep, intrinsics=itr)
            tok = me.vit(v)
            x = q_seed
            for blk in me.read_blocks:
                x = x + blk['attn'](blk['norm_q'](x), tok)
                x = x + blk['mlp'](blk['norm_m'](x))

            bank = model.build_memory_bank(
                [w.to(device) for w in b.mem_views], p2d, val,
                mem_depth=(b.mem_depth.to(device) if b.mem_depth is not None else None),
                mem_intrinsics=(b.mem_intrinsics.to(device) if b.mem_intrinsics is not None else None),
                device=device, cube_scale=cs)

            # ONE target frame is enough to measure rank, and it keeps the (T*N) flatten
            # unambiguous. `mv` is the MEMORY views, whose axis 1 is M (not the clip
            # length), so reusing it as T silently mis-shaped the token.
            T = 1
            tq = torch.zeros(1, N, dtype=torch.long, device=device)
            qq = qt[0][None].to(torch.long)
            sizes = torch.stack([torch.tensor([w.shape[-1], w.shape[-2]], dtype=torch.float32,
                                              device=device) for w in mv])
            mtok = model.memory_query_encoder(bank, qq, tq, cg, sizes, T)
            mtok = mtok[0].reshape(T, N, len(cg), -1)[0, :, 0]

            qe = model.query_encoder(model._normalize_views(
                [w.to(device) for w in b.views], device), cg, qc, qt, qt, cs)[0, :, 0]

            # Rank only over points the memory could POSSIBLY tell apart. A point that no
            # camera sees in any remembered frame is the null token by construction, and a
            # crowd of identical nulls drags the measurement down for reasons that have
            # nothing to do with capacity. cam0 stages use cam0's own validity.
            seen_any = val[:, 0].any(dim=(0, 1))                       # (N,) over cams, M
            seen_c0 = val[0, 0].any(dim=0)                             # (N,) camera 0
            frac = float(seen_any.float().mean())
            if int(seen_any.sum()) < args.min_points or int(seen_c0.sum()) < args.min_points:
                continue

            rows.append((b.sample_info.get('dataset', '?'), int(seen_any.sum()), len(cg), frac,
                         eff_rank(q_seed[0][seen_c0]), eff_rank(x[0][seen_c0]),
                         eff_rank(bank_summary(bank[0], val)[seen_any]),
                         eff_rank(mtok[seen_any]), eff_rank(qe[seen_any])))
            if len(rows) >= args.n_samples:
                break

    if not rows:
        sys.exit('no usable samples')
    print(f'\n{"dataset":22s} {"Nvis":>5s} {"cam":>4s} {"vis":>5s} | {"query":>7s} {"read":>7s} '
          f'{"bank":>7s} {"memTok":>7s} | {"qryTok":>7s}')
    for r in rows:
        print(f'{r[0]:22s} {r[1]:5d} {r[2]:4d} {r[3]:5.2f} | {r[4]:7.1f} {r[5]:7.1f} '
              f'{r[6]:7.1f} {r[7]:7.1f} | {r[8]:7.1f}')
    a = np.array([r[4:] for r in rows])
    print(f'\nmean effective rank   query {a[:,0].mean():.1f} -> read {a[:,1].mean():.1f} '
          f'-> bank {a[:,2].mean():.1f} -> memory-query token {a[:,3].mean():.1f}')
    print(f'                      ordinary query token (localizes well) {a[:,4].mean():.1f}')
    print(f'-> the memory token uses {a[:,3].mean()/max(a[:,4].mean(),1e-9)*100:.0f}% '
          f'as many dimensions as the query token')
    print('\nbaseline before the capacity rework: query 4.3 -> read 2.5 -> bank 3.1 '
          '-> token 2.4, vs query token 4.9')


if __name__ == '__main__':
    main()
