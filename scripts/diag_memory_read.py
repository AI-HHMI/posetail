#!/usr/bin/env python
"""Where does the decoder's memory cross-attention actually look?

The M-curve is flat: M=2, 4 and 8 give the same memory-only error. Two opposite failures
produce that signature and they need opposite fixes, so measure which one it is.

  entropy -> 0  (one-hot)  only ever one entry is read; extra entries are dead weight and
                           the fix is entry DIVERSITY, not more capacity.
  entropy -> 1  (uniform)  the read is averaging the bank, i.e. the collapse that per-camera
                           entries were meant to prevent, returning at a later stage.

A shuffled-bank control separates "uniform because it is averaging real evidence" from
"uniform because it ignores the bank entirely": if handing a point somebody else's memory
does not move the distribution, the read is not reading.

`DecoupledCrossAttention` goes straight to `F.scaled_dot_product_attention`, which returns no
weights, so this hooks the module to capture (query, kv) and recomputes the softmax. No model
code is touched -- this is a diagnostic, not a feature.

Also reports the null-dilution number (0c): every (frame, camera) pair where a point is not
visible carries `null_entry`, so a bank of M*n_cams entries can hold far fewer than M*n_cams
real observations. If the non-null count barely rises with M, the flat curve is explained by
sampling rather than architecture.

  pixi run python scripts/diag_memory_read.py --base-folder <wandb run> --contexts 2 8
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate  # noqa: E402
from posetail.inference.inference_utils import load_model_from_base_folder      # noqa: E402
from posetail.posetail.cube import compute_cube_scale                           # noqa: E402
from posetail.posetail.train_utils import (dict_to_device, memory_raw_from_batch,  # noqa: E402
                                           memory_only_kwargs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-folder', required=True)
    p.add_argument('--checkpoint', type=int, default=None)
    p.add_argument('--split', default='val')
    p.add_argument('--contexts', type=int, nargs='+', default=[2, 8],
                   help='memory frame counts to compare')
    p.add_argument('--n-per-dataset', type=int, default=2)
    p.add_argument('--max-points', type=int, default=128)
    p.add_argument('--device', default=None)
    return p.parse_args()


def attn_entropy(query, kv, mod):
    """Normalized entropy of the attention distribution over the ENTRY axis.

    query: (BK, T, latent) -- one row per (camera, batch, point)
    kv:    (BK, M, latent) -- that point's own bank
    Returns (BK*T*heads,) entropies in [0, 1], where 1 == uniform over M.
    """
    q = rearrange(mod.q_proj(query), 'b l (h d) -> b h l d', h=mod.num_heads)
    k = rearrange(mod.k_proj(kv), 'b l (h d) -> b h l d', h=mod.num_heads)
    logits = (q @ k.transpose(-2, -1)) / (mod.head_dim ** 0.5)     # (b, h, T, M)
    w = logits.float().softmax(dim=-1)
    M = w.shape[-1]
    if M < 2:
        return None
    ent = -(w * (w + 1e-12).log()).sum(-1) / np.log(M)
    return ent.reshape(-1)


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, _, ckpt = load_model_from_base_folder(
        args.base_folder, checkpoint=args.checkpoint, device=device)
    model.eval()
    if not getattr(model, 'memory_attention', False):
        sys.exit('model has no memory_attention; nothing to diagnose')
    print(f'checkpoint: {os.path.basename(ckpt)}   contexts: {args.contexts}')

    d = config.dataset[args.split]
    d['kpts_to_sample'] = args.max_points
    d['memory_prob'] = 1.0
    d['memory_num_context'] = max(args.contexts)
    d['memory_only_prob'] = 1.0
    d['memory_only_kpt_prob'] = 0.5
    d['balance_datasets'] = True
    d['n_samples_per_dataset'] = args.n_per_dataset
    d['aug_prob'] = 0.0
    ds = PosetailDataset(config, split=args.split)
    dl = DataLoader(ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=4)

    # Capture (query, kv) for every memory cross-attention layer without touching the model.
    captured = []

    def hook(mod, inputs, _out):
        captured.append((mod, inputs[0].detach(), inputs[1].detach()))

    handles = [m.register_forward_hook(hook) for m in model.decoder.memory_cross_attns]

    ent_acc = defaultdict(list)      # (M, arm) -> entropies
    null_acc = defaultdict(list)     # M -> mean non-null entries per point
    vis_acc = []
    n_cams_seen = []

    with torch.inference_mode():
        for batch in dl:
            if batch is None or batch.views is None or batch.mem_views is None:
                continue
            views = [v.to(device) for v in batch.views]
            coords = batch.coords.to(device)
            query_times = batch.query_times.to(device)
            cgroup = [dict_to_device(c, device) for c in batch.cgroup] if batch.cgroup else None
            p2d = batch.p2d.to(device) if batch.p2d is not None else None
            if p2d is not None:
                continue                                    # 3D path only, like diag_memory
            N = batch.mem_p2d.shape[3]
            query_coords = coords[:, query_times[0], torch.arange(N)]
            cube_scale = compute_cube_scale(
                cgroup, query_coords, len(cgroup), device,
                per_camera=getattr(model, 'per_camera_cube_scale', False))

            mem_valid = batch.mem_valid.to(device)          # (cams, B, M, N)
            n_cams = mem_valid.shape[0]
            vis_acc.append(float(mem_valid.float().mean()))
            n_cams_seen.append(n_cams)

            bank = model.build_memory_bank(
                memory_raw_from_batch(model, batch), device=device, cube_scale=cube_scale)

            qc, mo_kw = memory_only_kwargs(model, batch, query_coords, cgroup, False)
            mo = batch.memory_only
            mo = mo.to(device)[0] if mo is not None else None
            if mo is None or not bool(mo.any()):
                continue

            base = dict(views=views, query_times=query_times, camera_group=cgroup)
            if getattr(model, 'occlusion_embedding', False):
                base['occlusion'] = batch.query_occlusion.to(device)
            base.update(mo_kw)

            for M in args.contexts:
                sl = bank[:, :, :M * n_cams]
                # 0c: how many of those M*n_cams entries are real observations?
                # mem_valid is frame-major to match the bank's entry order.
                nn_per_pt = mem_valid[:, :, :M].permute(1, 3, 2, 0).reshape(1, N, -1)
                null_acc[M].append(float(nn_per_pt.float().sum(-1).mean()))

                for arm, b_in in (('real', sl), ('shuffled', sl[:, torch.randperm(N)])):
                    captured.clear()
                    model(coords=qc, memory_bank=b_in, **base)
                    for mod, q, kv in captured:
                        # _memory_read groups as '(cams b k) t d' with k fastest and b == 1,
                        # so tiling the point mask per camera selects the memory-only rows.
                        if q.shape[0] != n_cams * N:
                            continue
                        keep = mo.repeat(n_cams)
                        e = attn_entropy(q[keep], kv[keep], mod)
                        if e is not None:
                            ent_acc[(M, arm)].append(e.cpu().numpy())

    for h in handles:
        h.remove()

    if not ent_acc:
        sys.exit('no memory-only samples captured')

    print(f'\nvisible fraction of (frame, camera, point): {np.mean(vis_acc):.3f}')
    print(f'{"M":>4}  {"entries":>8}  {"non-null":>9}  {"entropy(real)":>14}  {"entropy(shuf)":>14}')
    for M in args.contexts:
        e_r = np.concatenate(ent_acc[(M, 'real')]) if (M, 'real') in ent_acc else np.array([np.nan])
        e_s = np.concatenate(ent_acc[(M, 'shuffled')]) if (M, 'shuffled') in ent_acc else np.array([np.nan])
        # camera count varies per sample, so the bank size is a mean too
        entries = M * float(np.mean(n_cams_seen)) if n_cams_seen else 0.0
        print(f'{M:>4}  {entries:>8.1f}  {np.mean(null_acc[M]):>9.2f}  '
              f'{np.nanmean(e_r):>14.4f}  {np.nanmean(e_s):>14.4f}')
    print('\nentropy ~0 -> one-hot (fix diversity/temperature); ~1 -> averaging (collapse).')
    print('real == shuffled -> the read ignores bank CONTENT entirely.')


if __name__ == '__main__':
    main()
