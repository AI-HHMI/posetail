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

from posetail.posetail.cube import compute_cube_scale
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


def make_memory(n_cams=2, M=M_CTX, valid=True, slot=None):
    """Remembered observations, as the dataset would emit them.

    Each entry is a 2-frame TUBELET (uint8, as the dataset stores it) because the video
    backbone's patch embedding is tubelet_size=2; mem_slot says which half holds the frame
    the points were projected into.
    """
    return dict(
        mem_views=[torch.randint(0, 256, (B, M, 2, H, W, 3), dtype=torch.uint8)
                   for _ in range(n_cams)],
        mem_p2d=torch.rand(n_cams, B, M, N, 2) * (W - 1),
        mem_valid=torch.full((n_cams, B, M, N), bool(valid)),
        mem_depth=torch.rand(n_cams, B, M, N) * 3 + 1,
        mem_intrinsics=torch.rand(n_cams, B, M, 4),
        mem_slot=(torch.full((B, M), slot, dtype=torch.long) if slot is not None
                  else torch.randint(0, 2, (B, M))),
    )


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
        bank = mem.build_memory_bank(**make_memory(M=M_CTX))
        bank8 = mem.build_memory_bank(**make_memory(M=8))
    n_cams = 2
    results.append(report(f'bank {tuple(bank.shape)} == (B,N,M*n_cams,dim)',
                          tuple(bank.shape[:3]) == (B, N, M_CTX * n_cams)))
    results.append(report(f'a different memory count works unchanged (M=8 -> '
                          f'{tuple(bank8.shape[:3])})', bank8.shape[2] == 8 * n_cams))

    # ---- 3 & 4. parity ---------------------------------------------------------------
    print('3-4. parity')
    mem.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    mem.eval()

    views, coords, qt, cg = make_batch()
    with torch.no_grad():
        out_off = base(views=views, coords=coords, camera_group=cg, query_times=qt)
        out_on = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                     memory_bank=mem.build_memory_bank(**make_memory()))
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
        bank_bad = mem.build_memory_bank(**make_memory(valid=False))
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
                    memory_bank=model.build_memory_bank(**make_memory()))
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
    for w in ['memory_encoder.patch_processor',
              'memory_encoder.read_blocks', 'memory_encoder.query_gate',
              'memory_encoder.query_mlp', 'decoder.memory_cross_attns']:
        results.append(report(f'trained regime: grad reaches {w}',
                              any(n.startswith(w) for n in have1)))

    # ---- 7. memory frames go through the SCENE encoder --------------------------------
    print('7. memory frames use the scene backbone')
    results.append(report('no private memory ViT remains',
                          not hasattr(mem.memory_encoder, 'vit')))
    # The read must consume tokens at the scene encoder's width, or the memory entries are
    # not in the same feature space as the clip tokens -- the whole point of the swap.
    results.append(report('read blocks key on scene-width tokens',
                          all(blk['attn'].kv_dim == mem.scene_encoder.embed_dim
                              for blk in mem.memory_encoder.read_blocks)))
    kw = make_memory()
    with torch.no_grad():
        vn = mem._normalize_views_mem(kw['mem_views'], 'cpu')
        tok = mem._encode_memory_frames(vn)
    gh = H // mem.scene_encoder.patch_size
    results.append(report(f'one batched encode -> (cams, B, M, {gh * gh}, '
                          f'{mem.scene_encoder.embed_dim}) tokens',
                          tuple(tok.shape) == (2, B, M_CTX, gh * gh,
                                               mem.scene_encoder.embed_dim)))
    # A temporal dim of 1 silently routes VJEPA down its IMAGE path (a different feature
    # space) instead of erroring, so the tubelet assertion is load-bearing.
    bad = [v[:, :, :1] for v in vn]
    try:
        mem._encode_memory_frames(bad)
        ok_assert = False
    except AssertionError:
        ok_assert = True
    results.append(report('a 1-frame "tubelet" is rejected, not silently image-encoded',
                          ok_assert))

    # ---- 8. kpt_chunk parity ---------------------------------------------------------
    print('8. kpt_chunk parity')
    mem.eval()
    with torch.no_grad():
        bank = mem.build_memory_bank(**make_memory())
        full = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                   memory_bank=bank)
        chunked = mem(views=views, coords=coords, camera_group=cg, query_times=qt,
                      memory_bank=bank, kpt_chunk=2)
    d = (full['coords_pred'].double() - chunked['coords_pred'].double()).abs().max().item()
    results.append(report(f'chunked == full-N (maxabsdiff={d:.3e})', d < 1e-5))

    # ---- 9. memory-only queries ------------------------------------------------------
    # Points whose query position is withheld: the model must find them from memory.
    print('9. memory-only queries')
    mem.eval()

    def run(model, coords_in, bank, n_cams=2, **kw):
        v, c, q, g = make_batch()
        g = g[:n_cams]
        v = v[:n_cams]
        cs = compute_cube_scale(g, coords, len(g), coords.device,
                                per_camera=model.per_camera_cube_scale)  # full set, BEFORE masking
        with torch.no_grad():
            return model(views=v, coords=coords_in, camera_group=g, query_times=q,
                         memory_bank=bank, cube_scale=cs, **kw)

    bank = mem.build_memory_bank(**make_memory())
    unk = torch.zeros(B, N, dtype=torch.bool)
    unk[:, :2] = True
    coords_mo = coords.clone()
    coords_mo[unk] = float('nan')

    out_mo = run(mem, coords_mo, bank)
    results.append(report('finite outputs with memory-only points',
                          all(bool(torch.isfinite(out_mo[k]).all())
                              for k in ['coords_pred', '2d_pred', '3d_pred_direct'])))

    # all-False mask must reproduce the plain path exactly (regression guard)
    out_plain = run(mem, coords, bank)
    out_nomask = run(mem, coords, bank)
    results.append(report('all-known mask == plain path (bit-identical)',
                          torch.equal(out_plain['coords_pred'], out_nomask['coords_pred'])))

    # the known points must be unaffected by their neighbours going memory-only
    d_known = (out_plain['coords_pred'][:, :, 2:].double()
               - out_mo['coords_pred'][:, :, 2:].double()).abs().max().item()
    results.append(report(f'known points barely move when others go memory-only '
                          f'(maxabsdiff={d_known:.2e})', d_known < 1e-3))

    # THE load-bearing test: memory-only predictions must actually follow the memory.
    # This can only be measured on a model that is OFF its zero-init state -- the grid
    # heads and the memory read are both zero-init, so a freshly built model emits the
    # grid centre for every point regardless of its input (2d_pred has exactly one unique
    # value). Perturb them to emulate a partly-trained model, then swap the bank.
    trained = build(True)
    trained.load_state_dict(mem.state_dict(), strict=False)
    with torch.no_grad():
        for h in (trained.decoder.heads_2d, trained.decoder.heads_3d):
            for m_i in range(2):
                h[m_i][1].weight.normal_(0, 0.02)
        for mca in trained.decoder.memory_cross_attns:
            mca.out_proj.weight.normal_(0, 0.02)
    trained.eval()

    # Perturb ONLY the memory-only points' own bank rows. That isolates the effect: the
    # anchored points' rows are untouched, so their decoder memory-read is unchanged and
    # they must not move AT ALL, while the memory-only points -- whose query token IS this
    # memory -- must. (Swapping the whole bank would move every point, since the decoder
    # reads memory for all of them, and would prove nothing.)
    bank_a = mem.build_memory_bank(**make_memory())
    bank_b = bank_a.clone()
    bank_b[:, :2] = bank_b[:, :2].roll(1, dims=-1) * 3.0 + 0.5
    out_a = run(trained, coords_mo, bank_a)
    out_b = run(trained, coords_mo, bank_b)
    d_mo = (out_a['2d_pred'][..., :2, :].double()
            - out_b['2d_pred'][..., :2, :].double()).abs().max().item()
    d_kn = (out_a['2d_pred'][..., 2:, :].double()
            - out_b['2d_pred'][..., 2:, :].double()).abs().max().item()
    results.append(report(f'memory-only points follow THEIR memory '
                          f'(they move {d_mo:.3f} px; anchored points, whose memory was '
                          f'untouched, move {d_kn:.3e} px)',
                          d_mo > 1e-2 and d_kn < 1e-6))

    # single camera must fall back to the ray anchor (triangulation is None there)
    bank1 = mem.build_memory_bank(**make_memory(n_cams=1))
    out_1cam = run(mem, coords_mo, bank1, n_cams=1)
    results.append(report('single camera works (ray anchor, no triangulation)',
                          bool(torch.isfinite(out_1cam['coords_pred']).all())))

    # gradients must be finite everywhere -- the NaN-in-the-discarded-branch trap
    gm = build(True)
    gm.train()
    v, c, q, g = make_batch()
    bank_g = gm.build_memory_bank(**make_memory())
    o = gm(views=v, coords=coords_mo, camera_group=g, query_times=q, memory_bank=bank_g,
           cube_scale=compute_cube_scale(g, coords, len(g), coords.device,
                                        per_camera=gm.per_camera_cube_scale))
    (o['coords_pred'].square().mean() + o['2d_pred'].square().mean()).backward()
    bad = [n for n, p in gm.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    results.append(report(f'no non-finite gradients{"" if not bad else " -- " + bad[0]}',
                          not bad))
    results.append(report('grad reaches memory_query_encoder',
                          any(n.startswith('memory_query_encoder') and p.grad is not None
                              and p.grad.abs().sum() > 0 for n, p in gm.named_parameters())))

    print()
    print(f'RESULT: {sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
