#!/usr/bin/env python
"""Smoke test for the decoded `learnable_scale` (+ decoder dict refactor).

Builds a FRESH TrackerEncoder from config_encoder_gridresid_learnscale.toml, runs one
real val batch through the full forward + TotalLoss + backward, and checks:
  1. the decoder now returns a DICT (per-head keys), forward runs, coords_pred finite;
  2. grid dict carries learnable_scale + s3d/sdep, both > 0 and ~= scale_init at init
     (head zero-init -> exp(0)=1 -> no-op multiply);
  3. IDENTIFIABILITY: after backward, grads reach scale_3d_head (via the metric regression
     loss) AND heads_3d (via the grid CE) — the detach-in-CE-target split works;
  4. scale_depth_head gets a grad too.

Run:  pixi run python parity_check/smoke_learnable_scale.py
"""
import os
import sys
import math
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from train_utils import load_config, dict_to_device
from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.losses import TotalLoss
from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from torch.utils.data import DataLoader

CONFIG = os.path.join(REPO, 'configs', 'config_encoder_gridresid_learnscale.toml')


def _grad_norm(module):
    g = 0.0
    for p in module.parameters():
        if p.grad is not None:
            g += float(p.grad.detach().float().norm())
    return g


def main():
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(CONFIG)
    assert config.model.get('learnable_scale'), 'config should enable learnable_scale'
    delta = float(config.model.get('scale_delta', 2.0))
    sinit = float(config.model.get('scale_init', 1.0))

    print('building model from config ...', flush=True)
    model = TrackerEncoder(**config.model).to(device)
    model.train()
    assert model.decoder.scale_3d_head is not None and model.decoder.scale_depth_head is not None
    print('  decoder.scale_3d_head / scale_depth_head present', flush=True)

    # one real val batch, prefer a 3D (R==3) one for the grid checks
    val = PosetailDataset(config, split='val')
    loader = DataLoader(val, batch_size=config.dataset.batch_size, collate_fn=custom_collate,
                        shuffle=True, num_workers=2)
    batch = None
    for b in loader:
        if b.coords.shape[-1] == 3:
            batch = b; break
    assert batch is not None, 'no R==3 val batch found'

    views = [v.to(device) for v in batch.views]
    coords = batch.coords.to(device)                                   # (B,T,N,3)
    qt = batch.query_times.to(device)
    cg = [dict_to_device(c, device) for c in batch.cgroup]
    p2d = batch.p2d.to(device) if batch.p2d is not None else None
    vis = batch.vis.to(device) if batch.vis is not None else None
    vis_2d = batch.vis_2d.to(device) if batch.vis_2d is not None else None
    qc = coords[:, qt[0], torch.arange(coords.shape[2])]              # (B,N,3) query coords

    # ---- forward -------------------------------------------------------------------
    out = model(views=list(views), coords=qc, query_times=qt, camera_group=cg)
    assert 'coords_pred' in out and torch.isfinite(out['coords_pred']).all(), 'coords_pred not finite'
    print(f"  forward OK; coords_pred {tuple(out['coords_pred'].shape)}", flush=True)

    g = out['grid']
    assert g.get('learnable_scale') is True, 'grid dict missing learnable_scale'
    s3d, sdep = g['s3d'], g['sdep']
    assert s3d is not None and sdep is not None, 'grid dict missing s3d/sdep'
    lo, hi = sinit * math.exp(-delta), sinit * math.exp(delta)
    assert (s3d > 0).all() and (sdep > 0).all(), 'scales must be positive'
    assert (s3d >= lo - 1e-4).all() and (s3d <= hi + 1e-4).all(), f's3d out of [{lo},{hi}]'
    # no-op at init: head zero-init -> exp(0) = 1 -> s == scale_init
    assert torch.allclose(s3d, torch.full_like(s3d, sinit), atol=1e-4), \
        f's3d should be ~scale_init={sinit} at init, got [{s3d.min():.4f},{s3d.max():.4f}]'
    assert torch.allclose(sdep, torch.full_like(sdep, sinit), atol=1e-4)
    print(f'  s3d/sdep ~= scale_init ({sinit}) at init (no-op) — OK', flush=True)

    # ---- loss + backward: identifiability split ------------------------------------
    pccs = config.model.get('per_camera_cube_scale', False)
    loss_fn = TotalLoss(**config.training.losses, per_camera_cube_scale=pccs).to(device)
    result = loss_fn(model, out, coords, vis, vis_2d, cgroup=cg, p2d=p2d, device=device)
    total = result[0] if isinstance(result, (tuple, list)) else result
    assert torch.isfinite(total), f'loss not finite: {total}'
    model.zero_grad(set_to_none=True)
    total.backward()

    m = 1  # 3D-query head index
    g_scale3d = _grad_norm(model.decoder.scale_3d_head[m])
    g_scaledep = _grad_norm(model.decoder.scale_depth_head[m])
    g_head3d = _grad_norm(model.decoder.heads_3d[m])
    print(f'  grad norms: scale_3d_head={g_scale3d:.3e}  scale_depth_head={g_scaledep:.3e}  '
          f'heads_3d={g_head3d:.3e}', flush=True)
    assert g_scale3d > 0, 'scale_3d_head got NO grad (metric loss should train it)'
    assert g_head3d > 0, 'heads_3d got NO grad (grid CE should train it)'
    assert g_scaledep > 0, 'scale_depth_head got NO grad (depth loss should train it)'

    print('\nSMOKE PASSED: dict return + learnable_scale forward/backward + identifiability split')


if __name__ == '__main__':
    main()
