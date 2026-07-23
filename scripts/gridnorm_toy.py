#!/usr/bin/env python
"""Phase 0 toy verification for the `gridnorm` output mode (see the plan).

Pure-geometry checks on real kubric multiview data -- NO network, NO training.
Every claim of `gridnorm` is geometric and is checked here with an "oracle" grid
(we synthesize the grid coordinates the network would ideally predict, then verify
the solve / reconstruction / coverage math).

Run:  pixi run python scripts/gridnorm_toy.py
Writes: reports/gridnorm_toy.md
"""
import os
import sys
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.posetail.cube import (
    points_to_rays, get_camera_scale, _invert_SE3,
    to_homogeneous, from_homogeneous, project_points_torch,
)
from posetail.posetail.losses import normalize_by_mean_depth
from posetail.inference.inference_utils import load_camera_group_from_metadata

DATA = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v3/kubric-multiview'
IMAGE_SIZE = 256          # model input res (only used by the gridresid comparison)
DTYPE = torch.float64     # geometry in double so exactness asserts are meaningful
torch.manual_seed(0)
np.random.seed(0)

_LOG = []
def log(msg=''):
    print(msg)
    _LOG.append(str(msg))


# ----------------------------------------------------------------------------- helpers
def load_trial(trial_path):
    """Return (world (T,N,3), vis (T,N,C) bool, camera_group)."""
    d = np.load(os.path.join(trial_path, 'pose3d.npz'), allow_pickle=True)
    pose = torch.as_tensor(d['pose'][0], dtype=DTYPE)          # (T, N, 3)
    vis = torch.as_tensor(np.asarray(d['vis'][0]), dtype=torch.bool)  # (T, N, C)
    cg = load_camera_group_from_metadata(os.path.join(trial_path, 'metadata.yaml'),
                                         device='cpu')
    for cam in cg:                                             # cast to DTYPE
        for k in ('ext', 'mat', 'center', 'dist'):
            cam[k] = cam[k].to(DTYPE)
    return pose, vis, cg


def build_raylocal(cg):
    """rays_c: (C,4,4) world->ray-local SE3 (built at each cam's principal point)."""
    mats = []
    for cam in cg:
        pp = cam['mat'][:2, 2].reshape(1, 2)                  # principal point (cx,cy)
        mats.append(points_to_rays(cam, pp, normalize_t=False)[0])
    return torch.stack(mats)                                  # (C,4,4)


def world_to_raylocal(rays_c, world):
    """world (T,N,3) -> q (C,T,N,3) ray-local coords per camera."""
    wh = to_homogeneous(world)                                # (T,N,4)
    qh = torch.einsum('cij,tnj->ctni', rays_c, wh)            # (C,T,N,4)
    return from_homogeneous(qh)


def raylocal_to_world(rays_c_inv, q):
    """q (C,T,N,3) -> world (C,T,N,3) per camera (should agree across c for exact q)."""
    qh = to_homogeneous(q)
    wh = torch.einsum('cij,ctnj->ctni', rays_c_inv, qh)
    return from_homogeneous(wh)


def solve_scale_offset(g, q):
    """Fixed-rotation LS: find scalar s and 3-vec t minimizing ||s*g + t - q||^2.
    g, q: (..., K, 3). Returns s (...), t (..., 3)."""
    gbar = g.mean(-2, keepdim=True)
    qbar = q.mean(-2, keepdim=True)
    gc = g - gbar
    qc = q - qbar
    num = (gc * qc).sum((-1, -2))
    den = (gc * gc).sum((-1, -2)).clamp_min(1e-30)
    s = num / den
    t = qbar.squeeze(-2) - s[..., None] * gbar.squeeze(-2)
    return s, t


def quantize_clip(x, radius, G):
    """Snap x to the nearest of G bins on [-radius, radius]; also clip out-of-range.
    Returns (x_quant, clipped_mask)."""
    binw = 2.0 * radius / (G - 1)
    xc = x.clamp(-radius, radius)
    xq = torch.round(xc / binw) * binw
    clipped = x.abs() > radius
    return xq, clipped


def rmse(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1)).mean().item()


def test_trials(split='test', n=1):
    trials = sorted(glob.glob(os.path.join(DATA, split, '*', 'trial')))
    return trials[:n]


# ----------------------------------------------------------------------------- toys
def toy1_frame_roundtrip(pose, cg):
    rays_c = build_raylocal(cg)
    rays_c_inv = _invert_SE3(rays_c)
    q = world_to_raylocal(rays_c, pose)
    w_rec = raylocal_to_world(rays_c_inv, q)                  # (C,T,N,3)
    err = max(rmse(w_rec[c], pose) for c in range(w_rec.shape[0]))
    log(f'[Toy 1] frame round-trip world RMSE = {err:.3e}  (want <1e-6)')
    assert err < 1e-6, err
    return rays_c, rays_c_inv, q


def toy2_solve_recovers(q):
    C, T, N, _ = q.shape
    # synthesize a gauge-free grid g = (q - t*)/s* with random per-cam (s*, t*)
    s_star = torch.rand(C, dtype=DTYPE) * 4 + 0.5             # (C,)
    t_star = torch.randn(C, 3, dtype=DTYPE) * 3
    g = (q - t_star[:, None, None, :]) / s_star[:, None, None, None]
    # solve from all query-frame (t=0) points
    s_hat, t_hat = solve_scale_offset(g[:, 0], q[:, 0])       # (C,), (C,3)
    ds = (s_hat - s_star).abs().max().item()
    dt = (t_hat - t_star).abs().max().item()
    # reconstruct ALL frames/points and compare
    q_hat = s_hat[:, None, None, None] * g + t_hat[:, None, None, :]
    err = rmse(q_hat, q)
    log(f'[Toy 2] recover s err={ds:.3e}  t err={dt:.3e}  full-recon RMSE={err:.3e}')
    assert ds < 1e-6 and dt < 1e-6 and err < 1e-6
    return s_star, t_star, g


def toy3_full_trajectory(rays_c_inv, q, g):
    C = q.shape[0]
    s_hat, t_hat = solve_scale_offset(g[:, 0], q[:, 0])       # solve from queries only
    q_hat = s_hat[:, None, None, None] * g + t_hat[:, None, None, :]
    w_hat = raylocal_to_world(rays_c_inv, q_hat)
    # each camera should reconstruct the true world trajectory
    err = max(rmse(w_hat[c], _POSE) for c in range(C))
    log(f'[Toy 3] solve-from-queries -> all-frames world RMSE = {err:.3e}  (want <1e-5)')
    assert err < 1e-5, err


def toy4_gauge_invariance(rays_c_inv, q, g):
    C = q.shape[0]
    # apply a random global (alpha, beta) to the model's internal grid
    alpha = torch.rand(C, dtype=DTYPE) * 5 + 0.2
    beta = torch.randn(C, 3, dtype=DTYPE) * 10
    g2 = alpha[:, None, None, None] * g + beta[:, None, None, :]
    s2, t2 = solve_scale_offset(g2[:, 0], q[:, 0])
    q_hat2 = s2[:, None, None, None] * g2 + t2[:, None, None, :]
    w_hat2 = raylocal_to_world(rays_c_inv, q_hat2)
    err = max(rmse(w_hat2[c], _POSE) for c in range(C))
    log(f'[Toy 4] reconstruction under random internal gauge: world RMSE = {err:.3e}')
    assert err < 1e-5, err


def toy5_cubescale_failure(q, cg, pose):
    """gridresid bins scale by cube_scale*image_size (focal-sensitive); gridnorm bins
    scale by the solved query spread (focal-invariant). Sweep a focal/cube_scale
    miscalibration factor lambda and compare clip-fraction + world RMSE."""
    C, T, N, _ = q.shape
    coords0 = pose[0:1]                                       # (1,N,3) as (B,N,3)
    cube_scale = get_camera_scale(cg, coords0).to(DTYPE)[:, 0]  # (C,)

    G = 1024
    R_res = 1.8      # gridresid: motion-anchored, small radius
    R_norm = 4.0     # gridnorm: shared per-cam frame, must cover object extent (RMS units)

    # gridnorm target (focal-independent): centre on query centroid, scale by RMS spread
    t_c = q[:, 0].mean(1)                                     # (C,3)
    resid0 = q[:, 0] - t_c[:, None]                           # (C,N,3)
    s_c = torch.sqrt((resid0 ** 2).mean((1, 2))).clamp_min(1e-9)  # (C,) scalar RMS
    g_norm = (q - t_c[:, None, None]) / s_c[:, None, None, None]
    gq, clip_norm = quantize_clip(g_norm, R_norm, G)
    q_norm_hat = gq * s_c[:, None, None, None] + t_c[:, None, None]
    err_norm = rmse(q_norm_hat, q)
    clipf_norm = clip_norm.float().mean().item()

    log('[Toy 5] focal/cube_scale sweep (clip-fraction and world RMSE, metres):')
    log(f'         gridnorm (focal-invariant): clip={clipf_norm:.3f}  RMSE={err_norm:.4f}')
    log('         lambda |  gridresid clip |  gridresid RMSE |  (gridnorm RMSE)')
    rows = []
    for lam in [1.0, 2.0, 4.0, 8.0, 16.0, 64.0]:
        denom = (cube_scale / lam)[:, None, None, None] * IMAGE_SIZE     # (C,1,1,1)
        motion = q - q[:, 0:1]                                # residual from query anchor
        tgt = motion / denom
        tq, clip_res = quantize_clip(tgt, R_res, G)
        q_res_hat = tq * denom + q[:, 0:1]
        err_res = rmse(q_res_hat, q)
        clipf_res = clip_res.float().mean().item()
        rows.append((lam, clipf_res, err_res))
        log(f'         {lam:6.1f} |     {clipf_res:8.3f}   |   {err_res:10.4f}  |   {err_norm:8.4f}')

    # claims: gridnorm clip/err flat in lambda (by construction); gridresid degrades
    err_res_lo = rows[0][2]
    err_res_hi = rows[-1][2]
    clip_res_hi = rows[-1][1]
    log(f'         -> gridresid RMSE x{err_res_hi/max(err_res_lo,1e-9):.1f} from lambda 1->64; '
        f'clip {rows[0][1]:.3f}->{clip_res_hi:.3f}')
    assert clipf_norm < 0.02, f'gridnorm should barely clip, got {clipf_norm}'
    assert err_res_hi > 5 * err_res_lo, 'gridresid should degrade badly at high lambda'
    assert clip_res_hi > 0.2, 'gridresid should clip heavily at high lambda'
    return err_norm, rows[0][2]


def toy6_noise_fusion(rays_c, rays_c_inv, q, pose, vis):
    """Noisy per-camera grid predictions (depth axis noisier); show multi-view fusion
    beats single-camera, and >=2 query points suffice / more stabilise the solve."""
    C, T, N, _ = q.shape
    t_c = q[:, 0].mean(1)
    s_c = torch.sqrt(((q[:, 0] - t_c[:, None]) ** 2).mean((1, 2))).clamp_min(1e-9)
    g_true = (q - t_c[:, None, None]) / s_c[:, None, None, None]

    sig_lat, sig_depth = 0.03, 0.15                           # depth (z) axis noisier
    noise = torch.randn(C, T, N, 3, dtype=DTYPE)
    noise[..., 2] *= (sig_depth / sig_lat)
    g_pred = g_true + noise * sig_lat

    def reconstruct(query_idx):
        gq = g_pred[:, 0][:, query_idx]                      # (C,K,3) noisy query preds
        qq = q[:, 0][:, query_idx]                           # (C,K,3) known ray-local
        s_hat, t_hat = solve_scale_offset(gq, qq)
        q_hat = s_hat[:, None, None, None] * g_pred + t_hat[:, None, None, :]
        return raylocal_to_world(rays_c_inv, q_hat)          # (C,T,N,3)

    all_idx = torch.arange(N)
    w_hat = reconstruct(all_idx)
    visf = vis.permute(2, 0, 1).to(DTYPE)                    # (C,T,N)
    # per-camera error (visible pts), and uniform multi-view fusion
    single = []
    for c in range(C):
        m = visf[c] > 0.5
        single.append(torch.sqrt(((w_hat[c] - pose) ** 2).sum(-1))[m].mean().item())
    w_fused = w_hat.mean(0)                                   # uniform fuse across cams
    m_any = (visf.sum(0) > 0.5)
    fused = torch.sqrt(((w_fused - pose) ** 2).sum(-1))[m_any].mean().item()
    mean_single = float(np.mean(single))
    log(f'[Toy 6] noisy recon world RMSE: mean single-cam={mean_single:.4f}  '
        f'fused(10 cams)={fused:.4f}')
    assert fused < mean_single, (fused, mean_single)

    # #query-points sweep: solve stability
    log('         K (query pts) |  fused RMSE')
    prev = None
    for K in [2, 3, 5, 10, 50, N]:
        idx = all_idx[:K]
        wf = reconstruct(idx).mean(0)
        e = torch.sqrt(((wf - pose) ** 2).sum(-1))[m_any].mean().item()
        log(f'         {K:12d}  |  {e:.4f}')
        prev = e
    log('         -> K=2 finite & stable; more query points reduce solve noise')
    return mean_single, fused


def toy7_2d_special_case(pose, vis, cg):
    """Single camera: gridnorm with the mean-depth gauge (scale=mean depth from camera
    centre, offset=centre) must reproduce the existing 2D-mode normalize_by_mean_depth."""
    cam = cg[0]
    C_center = cam['center']
    x = pose                                                 # (T,N,3) world
    v = vis[..., 0:1].to(DTYPE)                              # (T,N,1) visibility in cam 0
    lib_out, mean_depth = normalize_by_mean_depth(x, v, C_center)
    manual = (x - C_center) / (mean_depth + 1e-6)            # gridnorm N=1 mean-depth gauge
    valid = torch.isfinite(x).all(-1) & (v[..., 0] > 0.5)
    err = (lib_out - manual)[valid].abs().max().item()
    log(f'[Toy 7] gridnorm(N=1, mean-depth gauge) vs normalize_by_mean_depth: '
        f'max|diff|={err:.3e}')
    assert err < 1e-6, err


def toy8_gauge_coverage(split='test', n=8):
    """Across trials: with the RMS-spread gauge, do the query points (and the moving
    trajectory) stay inside a usable grid radius?"""
    q_abs, all_abs = [], []
    for tp in test_trials(split, n):
        pose, vis, cg = load_trial(tp)
        rays_c = build_raylocal(cg)
        q = world_to_raylocal(rays_c, pose)                  # (C,T,N,3)
        t_c = q[:, 0].mean(1)
        s_c = torch.sqrt(((q[:, 0] - t_c[:, None]) ** 2).mean((1, 2))).clamp_min(1e-9)
        g = (q - t_c[:, None, None]) / s_c[:, None, None, None]
        q_abs.append(g[:, 0].abs().reshape(-1))              # query frame
        all_abs.append(g.abs().reshape(-1))                  # all frames (incl motion)
    qa = torch.cat(q_abs)
    aa = torch.cat(all_abs)
    def p(t, x):  # numpy percentile (torch.quantile caps input size)
        a = t.numpy()
        if a.size > 2_000_000:
            a = np.random.choice(a, 2_000_000, replace=False)
        return float(np.percentile(a, x * 100))
    log(f'[Toy 8] gauge coverage over {n} trials (|g| per-axis, RMS-spread gauge):')
    log(f'         query frame : p50={p(qa,.5):.2f}  p99={p(qa,.99):.2f}  '
        f'p99.9={p(qa,.999):.2f}  max={qa.max():.2f}')
    log(f'         all frames  : p50={p(aa,.5):.2f}  p99={p(aa,.99):.2f}  '
        f'p99.9={p(aa,.999):.2f}  max={aa.max():.2f}')
    for R in (3.0, 4.0, 6.0):
        cov = (aa <= R).float().mean().item()
        log(f'         radius {R:.1f}: covers {cov*100:.2f}% of all-frame targets')
    assert p(qa, .99) < 6.0, 'query constellation unexpectedly spread'


# ----------------------------------------------------------------------------- main
def main():
    global _POSE
    trial = test_trials('test', 1)[0]
    log(f'# gridnorm Phase-0 toy verification')
    log(f'trial: {trial}')
    pose, vis, cg = load_trial(trial)
    _POSE = pose
    T, N, _ = pose.shape
    log(f'shapes: pose {tuple(pose.shape)}  vis {tuple(vis.shape)}  cams {len(cg)}')
    log('')

    rays_c, rays_c_inv, q = toy1_frame_roundtrip(pose, cg)
    s_star, t_star, g = toy2_solve_recovers(q)
    toy3_full_trajectory(rays_c_inv, q, g)
    toy4_gauge_invariance(rays_c_inv, q, g)
    log('')
    toy5_cubescale_failure(q, cg, pose)
    log('')
    toy6_noise_fusion(rays_c, rays_c_inv, q, pose, vis)
    log('')
    toy7_2d_special_case(pose, vis, cg)
    log('')
    toy8_gauge_coverage('test', 8)

    log('')
    log('ALL TOYS PASSED')

    os.makedirs('reports', exist_ok=True)
    with open('reports/gridnorm_toy.md', 'w') as f:
        f.write('# gridnorm Phase-0 toy verification results\n\n')
        f.write('Pure-geometry checks on kubric-multiview (no network/training). '
                'Generated by `scripts/gridnorm_toy.py`.\n\n```\n')
        f.write('\n'.join(_LOG))
        f.write('\n```\n')
    print('\nwrote reports/gridnorm_toy.md')


if __name__ == '__main__':
    main()
