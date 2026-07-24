#!/usr/bin/env python3
"""Smoke tests for the per-point memory cross-attention (MemoryEncoder + Decoder read).

Memory is a SET of remembered observations -- (image, point pixel, visible?) triples from
anywhere in the video -- rather than a slice of the current clip, so these tests drive the
model the same way the dataset / tracking loop does.

  1. parameter accounting -- no double registration; memory adds params
  2. bank shape, and M is not baked in (a different count still works)
  3. warm-start parity    -- memory ON at init is an exact no-op (zero-init out_proj)
  4. memory_bank=None     -- the memory path is skipped entirely
  5. degenerate memory    -- a point no camera can see gets the null token, no new NaN
  6. gradients            -- out_proj learns at step 0; the whole encoder learns after
  7. memory ViT           -- small dedicated single-frame encoder, non-native sizes
  8. kpt_chunk parity     -- chunked decode == full-N decode with memory on

Run: pixi run python smoke_memory.py
"""
import toml
import torch
from easydict import EasyDict as edict

from posetail.posetail.tracker_encoder import TrackerEncoder

CFG = 'configs/config_encoder_memory.toml'
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
    qt = torch.zeros(B, N, dtype=torch.int32)
    return views, coords, qt, cg


def make_memory(n_cams=2, M=M_CTX, valid=True):
    """Remembered observations, as the dataset would emit them."""
    mem_views = [torch.rand(B, M, H, W, 3) for _ in range(n_cams)]
    mem_p2d = torch.rand(n_cams, B, M, N, 2) * (W - 1)
    mem_valid = torch.full((n_cams, B, M, N), bool(valid))
    return mem_views, mem_p2d, mem_valid


def build(memory, **over):
    cfg = edict(toml.load(CFG))
    m = dict(cfg.model)
    m['video_encoder_version'] = 'base'
    m['memory_attention'] = memory
    m.update(over)
    return TrackerEncoder(**m)


def report(name, ok):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return ok


def main():
    results = []
    torch.manual_seed(0)

    # ---- 1. parameter accounting -----------------------------------------------------
    print('1. parameter accounting')
    base, mem = build(False), build(True)
    n_base = sum(p.numel() for p in base.parameters())
    n_mem = sum(p.numel() for p in mem.parameters())
    listed = list(mem.parameters())
    uniq = len({id(p) for p in listed})
    results.append(report(f'no double registration ({uniq} unique == {len(listed)} listed)',
                          uniq == len(listed)))
    results.append(report(f'memory adds params ({n_mem - n_base:,} new)', n_mem > n_base))

    # ---- 2. bank shape, and M is dynamic ---------------------------------------------
    print('2. bank shape')
    mem.eval()
    with torch.no_grad():
        bank = mem.build_memory_bank(*make_memory(M=M_CTX))
        bank8 = mem.build_memory_bank(*make_memory(M=8))
    results.append(report(f'bank {tuple(bank.shape)} == (B,N,M,dim)',
                          tuple(bank.shape[:3]) == (B, N, M_CTX)))
    results.append(report(f'a different memory count works unchanged (M=8 -> '
                          f'{tuple(bank8.shape[:3])})', bank8.shape[2] == 8))

    # ---- 3 & 4. parity ---------------------------------------------------------------
    print('3-4. parity')
    mem.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    mem.eval()

    views, coords, qt, cg = make_batch()
    with torch.no_grad():
        out_off = base(views=views, coords=coords, camera_group=cg, query_times=qt)
        out_on = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                     memory_bank=mem.build_memory_bank(*make_memory()))
        out_none = mem(views=views, coords=coords, camera_group=cg, query_times=qt)
    keys = ['coords_pred', '2d_pred', 'vis_pred', 'depth_pred']
    results.append(report('warm-start: memory ON at init == memory OFF (zero-init out_proj)',
                          all(torch.allclose(out_off[k], out_on[k], atol=1e-6) for k in keys)))
    for k in keys:
        d = (out_off[k].double() - out_on[k].double()).abs().max().item()
        print(f'        {k:12s} maxabsdiff={d:.3e}')
    results.append(report('memory_bank=None -> memory skipped, matches baseline',
                          all(torch.allclose(out_off[k], out_none[k], atol=1e-6) for k in keys)))

    # ---- 5. degenerate memory --------------------------------------------------------
    print('5. degenerate memory (no camera can see the point)')
    with torch.no_grad():
        bank_bad = mem.build_memory_bank(*make_memory(valid=False))
    results.append(report('bank is finite when nothing is visible',
                          bool(torch.isfinite(bank_bad).all())))
    results.append(report('invisible entries carry the learned null token',
                          torch.allclose(bank_bad[0, 0, 0], mem.memory_encoder.null_entry,
                                         atol=1e-5)))
    with torch.no_grad():
        out_bad = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                      memory_bank=bank_bad)
    results.append(report('forward stays finite with an all-null bank',
                          bool(torch.isfinite(out_bad['coords_pred']).all())))

    # ---- 6. gradients ----------------------------------------------------------------
    # The memory read's out_proj is zero-init, so at step 0 it is the ONLY memory module
    # with a gradient -- everything upstream is multiplied by a zero weight. That is the
    # intended warm start, so check both regimes.
    print('6. gradients')

    def grads_of(model):
        v, c, q, g = make_batch()
        out = model(views=v, coords=c, camera_group=g, query_times=q,
                    memory_bank=model.build_memory_bank(*make_memory()))
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
    for w in ['memory_encoder.vit', 'memory_encoder.patch_processor',
              'memory_encoder.read_attn', 'memory_encoder.camera_pool',
              'decoder.memory_cross_attns']:
        results.append(report(f'trained regime: grad reaches {w}',
                              any(n.startswith(w) for n in have1)))

    # ---- 7. the memory ViT -----------------------------------------------------------
    print('7. memory ViT')
    vit = mem.memory_encoder.vit
    n_vit = sum(p.numel() for p in vit.parameters())
    n_scene = sum(p.numel() for p in mem.scene_encoder.parameters())
    results.append(report(f'much smaller than the scene backbone '
                          f'({n_vit/1e6:.1f}M vs {n_scene/1e6:.0f}M)', n_vit < n_scene / 10))
    p = vit.patch_size
    with torch.no_grad():
        tok = vit(torch.randn(2, 3, 224, 320))     # non-native size -> interpolated pos-embed
    results.append(report(f'encodes a single frame at a non-native size (patch {p})',
                          tuple(tok.shape) == (2, (224 // p) * (320 // p), vit.embed_dim)))

    # ---- 8. kpt_chunk parity ---------------------------------------------------------
    print('8. kpt_chunk parity')
    mem.eval()
    with torch.no_grad():
        bank = mem.build_memory_bank(*make_memory())
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
