#!/usr/bin/env python3
"""Smoke tests for the per-point memory cross-attention (MemoryEncoder + Decoder read).

Checks, in order:
  1. no parameter duplication from the shared scene/query encoder references
  2. bank shape, and the query frame anchored at context slot 0
  3. warm-start parity -- memory ON at init is an exact no-op (zero-init out_proj)
  4. memory_bank=None    -- the memory path is skipped entirely
  5. degenerate memory   -- a point never in frame gets the null entry, adds no new NaN
  6. gradients           -- out_proj learns at step 0; the whole encoder learns after
  6b. context selection  -- frames biased to visible; occluded cameras excluded
  6c. memory ViT        -- small dedicated single-frame encoder; memory_prob gating
  7. kpt_chunk parity    -- chunked decode == full-N decode with memory on

Run: pixi run python smoke_memory.py
"""
import toml
import torch
from easydict import EasyDict as edict

from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.train_utils import sample_context_idx, frame_visibility_weight

CFG = "configs/config_encoder_memory.toml"
B, T, H, W, N = 1, 8, 256, 256, 4
M_CTX = 3


def make_cams(n=2):
    cams = []
    for i in range(n):
        K = torch.eye(3)
        K[0, 0] = K[1, 1] = 300.0
        K[0, 2] = K[1, 2] = 128.0
        ext = torch.eye(4)
        ext[0, 3] = i * 0.5
        ext[2, 3] = 3.0
        cams.append({'mat': K, 'ext': ext, 'ext_inv': torch.linalg.inv(ext),
                     'dist': torch.zeros(5), 'size': torch.tensor([W, H]),
                     'offset': torch.zeros(2), 'center': -ext[:3, :3].T @ ext[:3, 3],
                     'name': f'c{i}', 'type': 'p'})
    return cams


def make_batch(seed=1234, coords=None):
    """Fixed synthetic batch. Seeded HERE (not at model build) because constructing a
    memory-enabled model consumes extra RNG and would otherwise shift the inputs."""
    torch.manual_seed(seed)
    cg = make_cams()
    views = [torch.rand(B, T, H, W, 3) for _ in cg]
    if coords is None:
        coords = torch.randn(B, N, 3) * 0.2
    traj = coords[:, None].expand(B, T, N, 3).contiguous() + torch.randn(B, T, N, 3) * 0.01
    qt = torch.zeros(B, N, dtype=torch.int32)
    return views, coords, traj, qt, cg


def build(memory, **over):
    cfg = edict(toml.load(CFG))
    m = dict(cfg.model)
    m['video_encoder_version'] = 'base'
    m['memory_attention'] = memory
    m['memory_num_context'] = M_CTX
    m['memory_prob'] = 1.0
    m.update(over)
    return TrackerEncoder(**m)


def bank_of(model, views, coords, traj, qt, cg):
    """Mirror what train_utils.memory_kwargs does for one batch."""
    ctx = sample_context_idx(model, qt, T)
    return model.build_memory_bank(views, coords, cg, traj, ctx), ctx


def report(name, ok):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return ok


def main():
    results = []
    torch.manual_seed(0)

    # ---- 1. no duplicated parameters -------------------------------------------------
    print('1. parameter accounting')
    base, mem = build(False), build(True)
    n_base = sum(p.numel() for p in base.parameters())
    n_mem = sum(p.numel() for p in mem.parameters())
    listed = list(mem.parameters())
    uniq = len({id(p) for p in listed})
    results.append(report(f'no double registration ({uniq} unique == {len(listed)} listed)',
                          uniq == len(listed)))
    results.append(report(f'memory adds params ({n_mem - n_base:,} new)', n_mem > n_base))

    # ---- 2. bank shape ---------------------------------------------------------------
    print('2. bank shape')
    mem.eval()
    views, coords, traj, qt, cg = make_batch()
    with torch.no_grad():
        bank, ctx = bank_of(mem, views, coords, traj, qt, cg)
    results.append(report(f'bank {tuple(bank.shape)} == (B,N,M,dim)',
                          tuple(bank.shape[:3]) == (B, N, M_CTX)))
    results.append(report('query frame is context slot 0', bool((ctx[:, 0] == qt[:, 0]).all())))

    # ---- 3 & 4. parity ---------------------------------------------------------------
    print('3-4. parity')
    # Identical weights, so any difference is attributable to the memory path alone.
    mem.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    mem.eval()

    views, coords, traj, qt, cg = make_batch()
    with torch.no_grad():
        out_off = base(views=views, coords=coords, camera_group=cg, query_times=qt)
    views, coords, traj, qt, cg = make_batch()
    with torch.no_grad():
        bank, _ = bank_of(mem, views, coords, traj, qt, cg)
        out_on = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                     memory_bank=bank)
    keys = ['coords_pred', '2d_pred', 'vis_pred', 'depth_pred']
    results.append(report('warm-start: memory ON at init == memory OFF (zero-init out_proj)',
                          all(torch.allclose(out_off[k], out_on[k], atol=1e-6) for k in keys)))
    for k in keys:
        d = (out_off[k].double() - out_on[k].double()).abs().max().item()
        print(f'        {k:12s} maxabsdiff={d:.3e}')

    views, coords, traj, qt, cg = make_batch()
    with torch.no_grad():
        out_none = mem(views=views, coords=coords, camera_group=cg, query_times=qt)
    results.append(report('memory_bank=None -> memory skipped, matches baseline',
                          all(torch.allclose(out_off[k], out_none[k], atol=1e-6) for k in keys)))

    # ---- 5. degenerate memory --------------------------------------------------------
    # A point out of frame in every camera already yields non-finite coords_pred/2d_pred
    # in the BASELINE (the geometry has nothing to solve), so assert the memory path adds
    # no NEW non-finite values rather than absolute finiteness.
    print('5. degenerate memory (point never in frame)')
    far = torch.zeros(B, N, 3)
    far[..., 0] = 3.0
    views, coords, traj, qt, cg = make_batch(coords=far)
    with torch.no_grad():
        bank_f, _ = bank_of(mem, views, far, traj, qt, cg)
    results.append(report('bank is finite for an always-invisible point',
                          bool(torch.isfinite(bank_f).all())))
    null = mem.memory_encoder.null_entry
    is_null = torch.allclose(bank_f[0, 0, 1], null + mem.memory_encoder._frame_embed(
        torch.zeros(1, 1, dtype=torch.long))[0, 0] * 0, atol=1e-4) or \
        torch.allclose(bank_f[0, 0, 1], null, atol=1e-4)
    results.append(report('invalid entries carry the learned null token', bool(is_null)))
    with torch.no_grad():
        out_f = mem(views=views, coords=far, camera_group=cg, query_times=qt,
                    memory_bank=bank_f)
        out_f_base = base(views=views, coords=far, camera_group=cg, query_times=qt)
    results.append(report('memory introduces no new non-finite values vs baseline',
                          all((~torch.isfinite(out_f[k])).sum() == (~torch.isfinite(out_f_base[k])).sum()
                              for k in ['coords_pred', '2d_pred', 'depth_pred'])))

    # ---- 6. gradients ----------------------------------------------------------------
    # The memory read's out_proj is zero-init, so at step 0 it is the ONLY memory module
    # with a gradient -- everything upstream is multiplied by a zero weight. That is the
    # intended warm start, so check both regimes.
    print('6. gradients')

    def grads_of(model):
        views, coords, traj, qt, cg = make_batch()
        bank, _ = bank_of(model, views, coords, traj, qt, cg)
        out = model(views=views, coords=coords, camera_group=cg, query_times=qt,
                    memory_bank=bank)
        (out['coords_pred'].square().mean() + out['2d_pred'].square().mean()).backward()
        return {n for n, p in model.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0}

    step0 = build(True)
    step0.train()
    have0 = grads_of(step0)
    results.append(report('step 0: memory_cross_attns.out_proj receives grad',
                          any('memory_cross_attns' in n and 'out_proj' in n for n in have0)))

    warmed = build(True)
    warmed.train()
    with torch.no_grad():                       # emulate one optimizer step off zero-init
        for mca in warmed.decoder.memory_cross_attns:
            mca.out_proj.weight.normal_(0, 0.02)
    have1 = grads_of(warmed)
    for w in ['memory_encoder.read_attn', 'memory_encoder.camera_pool',
              'memory_encoder.temporal_embed', 'decoder.memory_cross_attns']:
        results.append(report(f'trained regime: grad reaches {w}',
                              any(n.startswith(w) for n in have1)))

    # ---- 6b. context selection & occlusion masking -----------------------------------
    print('6b. visibility-aware context + occlusion masking')
    mem.eval()
    views, coords, traj, qt, cg = make_batch()
    # points visible only at frames 2 and 5 -> the sampler must spend its slots there,
    # otherwise the bank fills with null padding.
    vis2d = torch.zeros(B, T, N, len(cg), 1)
    vis2d[:, 2] = 1.0
    vis2d[:, 5] = 1.0
    w = frame_visibility_weight(vis2d)
    picked = set()
    for _ in range(10):
        picked.update(sample_context_idx(mem, qt, T, random=True, weights=w)[0, 1:].tolist())
    results.append(report(f'context frames restricted to visible frames {sorted(picked)}',
                          picked <= {2, 5}))

    ctx = sample_context_idx(mem, qt, T, weights=w)
    with torch.no_grad():
        b_all = mem.build_memory_bank(views, coords, cg, traj, ctx)
        occ_one = torch.ones(B, T, N, len(cg), dtype=torch.long)
        occ_one[..., 0] = 0                                    # camera 0 occluded
        b_occ = mem.build_memory_bank(views, coords, cg, traj, ctx, occlusion_traj=occ_one)
        b_none = mem.build_memory_bank(views, coords, cg, traj, ctx,
                                       occlusion_traj=torch.zeros(B, T, N, len(cg),
                                                                  dtype=torch.long))
    results.append(report('occluded camera is excluded from the camera pool',
                          not torch.allclose(b_all, b_occ, atol=1e-6)))
    results.append(report('all-occluded entry falls back to the null token',
                          torch.allclose(b_none[0, 0, 1], mem.memory_encoder.null_entry,
                                         atol=1e-5)))

    # ---- 6c. dedicated ViT + memory_prob ---------------------------------------------
    print('6c. memory ViT and memory_prob')
    vit = mem.memory_encoder.vit
    n_vit = sum(p.numel() for p in vit.parameters())
    n_scene = sum(p.numel() for p in mem.scene_encoder.parameters())
    results.append(report(f'memory ViT is much smaller than the scene backbone '
                          f'({n_vit/1e6:.1f}M vs {n_scene/1e6:.0f}M)', n_vit < n_scene / 10))
    p = vit.patch_size
    with torch.no_grad():
        tok = vit(torch.randn(2, 3, 224, 320))     # non-native size -> interpolated pos-embed
    results.append(report(f'ViT encodes a single frame at a non-native size (patch {p})',
                          tuple(tok.shape) == (2, (224 // p) * (320 // p), vit.embed_dim)))

    from posetail.posetail.train_utils import memory_kwargs
    vis2d_full = torch.ones(B, T, N, len(cg), 1)
    never = build(True, memory_prob=0.0)
    never.train()
    always = build(True, memory_prob=1.0)
    always.train()
    got_never = memory_kwargs(never, views, coords, cg, traj, None, vis2d_full, qt)
    got_always = memory_kwargs(always, views, coords, cg, traj, None, vis2d_full, qt)
    results.append(report('memory_prob=0 skips the bank while training', got_never == {}))
    results.append(report('memory_prob=1 always builds the bank', 'memory_bank' in got_always))
    never.eval()
    results.append(report('memory_prob is ignored at eval (memory always on)',
                          'memory_bank' in memory_kwargs(never, views, coords, cg, traj,
                                                         None, vis2d_full, qt)))

    # ---- 7. kpt_chunk parity ---------------------------------------------------------
    print('7. kpt_chunk parity')
    mem.eval()
    views, coords, traj, qt, cg = make_batch()
    with torch.no_grad():
        bank, _ = bank_of(mem, views, coords, traj, qt, cg)
        full = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                   memory_bank=bank)
        chunked = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                      memory_bank=bank, kpt_chunk=2)
    d = (full['coords_pred'].double() - chunked['coords_pred'].double()).abs().max().item()
    results.append(report(f'chunked == full-N (maxabsdiff={d:.3e})', d < 1e-5))

    print()
    print(f'RESULT: {sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
