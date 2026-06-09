#!/usr/bin/env python
"""Estimate grid representations for TrackerTapNext's 3D + depth heads.

The grid heads bin each coordinate into ``head_grid_size`` bins. This probe
replays the EXACT loss-side target math (``losses.py`` grid CE) on the real
cameras training sees and reports, for BOTH 3D output modes plus the depth head:

  * grid      (output_mode='grid')      3D target = (rays_c·GT) / (cube·f_eff)
  * gridresid (output_mode='gridresid') 3D target = (rays_c·(GT_t−GT_q)) / (cube·f_eff)
  * depth     (both grid modes)         target    = log(‖GT−cam‖ / (cube·f_eff))

For each it answers three questions:
  1. RADIUS  — how wide must the linear grid [-r,r] be? (p99.9 of |target|, per axis).
  2. TRANSFORM — would a signed-log warp use the 256 bins better than a linear grid?
     We measure bin-occupancy entropy -> "effective bins" (2^H, max = grid size);
     a peaked-at-0 distribution wastes a linear grid and a log warp recovers bins.
  3. DEPTH-RANGE — is the depth head's [depth_log_min, depth_log_max] well matched to
     the log-depth distribution, or could it be tightened (wasted end bins)?

``rays_c`` is the world->ray-local SE3 at the image centre (``points_to_rays``,
``normalize_t=False``), identical to ``tracker_tapnext.py``'s direct-lift block.
The residual anchor is the window's first frame; motion is measured forward over
the window (causal tracking). Image-free, bounded file-walk (reuses
``estimate_scale_stats`` helpers) — fast even on kubric.

Usage
-----
    pixi run python scripts/estimate_grid_radius.py \
        --root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v3 \
        --split val --max-clips 8
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from posetail.posetail.cube import (get_camera_scale, points_to_rays,  # noqa: E402
                                    to_homogeneous, from_homogeneous)
from posetail.datasets.utils import get_dirs, load_yaml  # noqa: E402
from scripts.estimate_scale_stats import (candidate_pool, _load_traj, _flat_pts,  # noqa: E402
                                          crop_resize_cams, build_cams)

GRID = 256  # head_grid_size (bins per axis)


def _ray_local(rays_c, pts_world):
    """Transform world points (...,3) into the ray-local frame via SE3 ``rays_c``."""
    return from_homogeneous(
        torch.einsum('xr,...r->...x', rays_c, to_homogeneous(pts_world)))


def analyze(tpath, image_size, n_frames, min_crop_dim, max_pts, rng):
    """Per-clip SIGNED target arrays: grid (M,3), gridresid (M,3), log-depth (M,)."""
    md_path = os.path.join(tpath, "metadata.yaml")
    pose_path = os.path.join(tpath, "pose3d.npz")
    if not (os.path.exists(md_path) and os.path.exists(pose_path)):
        raise ValueError("missing metadata.yaml or pose3d.npz")
    traj = _load_traj(pose_path)                       # (T,P,3), NaNs preserved
    if traj is None:
        raise ValueError("no (S,T,K,3) pose array")
    t_total = traj.shape[0]
    win_len = max(2, min(n_frames, t_total))
    t0 = rng.randint(0, t_total - win_len) if t_total > win_len else 0
    win = traj[t0:t0 + win_len].double()               # (L,P,3)
    pts = _flat_pts(win, max_pts, rng)
    if pts is None:
        raise ValueError("< 8 finite 3D points in window")
    cams = build_cams(load_yaml(md_path))
    if not cams:
        raise ValueError("no usable cameras in metadata")
    fcams = crop_resize_cams(cams, pts, image_size, min_crop_dim)
    if not fcams:
        raise ValueError("no camera sees the points")
    cube = float(torch.nanmedian(get_camera_scale(fcams, pts[None].double())))
    if not np.isfinite(cube) or cube <= 0:
        raise ValueError("cube_scale non-finite")

    centre = torch.tensor([image_size / 2, image_size / 2], dtype=torch.float64).reshape(1, 2)
    grid_c, gridnf_c, resid_c, residnf_c, logd_c = [], [], [], [], []
    for c in fcams:
        f_eff = 0.5 * (float(c["mat"][0, 0]) + float(c["mat"][1, 1]))
        rays_c = points_to_rays(c, centre, normalize_t=False)[0]    # (4,4)
        p_rl = _ray_local(rays_c, win)                              # (L,P,3) ray-local
        motion = p_rl - p_rl[0:1]                                   # forward motion from query
        target_grid = p_rl / (cube * f_eff)                        # absolute /cube/feff, signed
        target_gridnf = p_rl / cube                                # absolute /cube (~f_eff units), signed
        target_resid = motion / (cube * f_eff)                     # f_eff-normalized residual, signed
        target_residnf = motion / cube                             # NO-f_eff residual (~pixels), signed
        cam_center = -(c["ext"][:3, :3].T @ c["ext"][:3, 3])
        depth = torch.linalg.norm(win - cam_center, dim=-1)        # (L,P)
        logd = torch.log(depth / (cube * f_eff) + 1e-6)            # depth grid target

        gm = torch.isfinite(target_grid).all(-1)
        grid_c.append(target_grid[gm])
        gridnf_c.append(target_gridnf[gm])
        rm = torch.isfinite(target_resid).all(-1)
        rm[0] = False                                              # frame-0 residual is 0 by def
        resid_c.append(target_resid[rm])
        residnf_c.append(target_residnf[rm])
        dm = torch.isfinite(logd)
        logd_c.append(logd[dm])
    cat = lambda xs, d: (torch.cat(xs, 0).numpy() if xs else np.zeros((0, d)))  # noqa: E731
    return dict(cube=cube,
                f_eff=float(np.median([0.5 * (float(c["mat"][0, 0]) + float(c["mat"][1, 1]))
                                       for c in fcams])),
                grid=cat(grid_c, 3), gridnf=cat(gridnf_c, 3),
                resid=cat(resid_c, 3), residnf=cat(residnf_c, 3),
                logd=(np.concatenate(logd_c) if logd_c else np.zeros(0)))


# ------------------------------------------------------------- representation analysis
def _bin_entropy(mapped, nbins=GRID):
    """Bin-occupancy entropy (bits) of values pre-mapped to [-1,1]; max = log2(nbins)."""
    h, _ = np.histogram(np.clip(mapped, -1, 1), bins=nbins, range=(-1, 1))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def _signed_log(v, eps):
    return np.sign(v) * np.log1p(np.abs(v) / eps)


def transform_report(vals, nbins=GRID):
    """Compare linear vs best signed-log bin utilization for a SIGNED target array.

    Returns (radius, eff_linear, eff_log, best_eps, p50). radius = p99.9 of |val|
    so both representations cover the same data; eff_* = 2^entropy (effective bins).
    """
    vals = vals.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    a = np.abs(vals)
    radius = float(np.percentile(a, 99.9))
    p50 = float(np.percentile(a, 50))
    eff_lin = 2 ** _bin_entropy(vals / max(radius, 1e-9), nbins)
    best_eff, best_eps = 0.0, float("nan")
    for frac in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        eps = max(frac * radius, 1e-9)
        mapped = _signed_log(vals, eps) / _signed_log(np.array(radius), eps)
        eff = 2 ** _bin_entropy(mapped, nbins)
        if eff > best_eff:
            best_eff, best_eps = eff, eps
    return radius, eff_lin, best_eff, best_eps, p50


def depth_report(logd, log_min, log_max, nbins=GRID):
    """Log-depth distribution + how well [log_min,log_max] fits (effective bins +
    a tighter data-driven range from the 0.5/99.5 percentiles)."""
    logd = logd[np.isfinite(logd)]
    if logd.size == 0:
        return None
    q = {p: float(np.percentile(logd, p)) for p in (0.1, 1, 50, 99, 99.9)}
    within = float(((logd >= log_min) & (logd <= log_max)).mean())
    mapped = (logd - log_min) / (log_max - log_min) * 2 - 1
    eff = 2 ** _bin_entropy(mapped, nbins)
    tight = (float(np.percentile(logd, 0.5)), float(np.percentile(logd, 99.5)))
    mapped_t = (logd - tight[0]) / (tight[1] - tight[0]) * 2 - 1
    eff_t = 2 ** _bin_entropy(mapped_t, nbins)
    return dict(q=q, within=within, eff=eff, tight=tight, eff_tight=eff_t)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/groups/karashchuk/karashchuklab/"
                    "animal-datasets-processed/posetail-finetuning-v3")
    ap.add_argument("--split", default="val")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--min-crop-dim", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--depth-log-min", type=float, default=-3.0)
    ap.add_argument("--depth-log-max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    datasets = args.datasets or get_dirs(args.root)
    print(f"root={args.root}  split={args.split}  image_size={args.image_size}  "
          f"n_frames={args.n_frames}  max_clips={args.max_clips}  grid={GRID} bins")
    print("Per-axis |target| percentiles (each axis binned independently in [-r,r]). "
          "no_feff vs with_feff.\n")

    agg = {k: [] for k in ("grid", "gridnf", "resid", "residnf", "logd")}
    recs = []   # per-dataset stats dict
    for ds in datasets:
        if not os.path.isdir(os.path.join(args.root, ds, args.split)):
            continue
        pool = candidate_pool(args.root, ds, args.split, args.max_clips * 5, rng)
        rows = []
        for c in pool:
            if len(rows) >= args.max_clips:
                break
            try:
                rows.append(analyze(c, args.image_size, args.n_frames,
                                    args.min_crop_dim, args.max_points, rng))
            except Exception as e:
                if args.verbose:
                    print(f"  skip {os.path.relpath(c, args.root)}: {e}")
        if not rows:
            continue
        arrs = {k: np.concatenate([r[k] for r in rows], 0) for k in agg}
        for k in agg:
            agg[k].append(arrs[k])
        pa = lambda k, p: float(np.percentile(np.abs(arrs[k]).reshape(-1), p))  # noqa: E731
        recs.append(dict(
            ds=ds, n=len(rows), f_eff=float(np.median([r["f_eff"] for r in rows])),
            g_nf=(pa("gridnf", 50), pa("gridnf", 99.9)),
            g_wf=(pa("grid", 50), pa("grid", 99.9)),
            r_nf=(pa("residnf", 50), pa("residnf", 99.9)),
            r_wf=(pa("resid", 50), pa("resid", 99.9))))

    if not recs:
        print("No usable 3D data.")
        return

    def _table(title, defn, nf_key, wf_key, nf_fmt, wf_fmt):
        print(f"{title}\n  {defn}")
        hdr = (f"{'dataset':18s} {'n':>2s} {'f_eff':>7s} | "
               f"{'no_feff p50':>12s} {'p99.9':>9s} | {'with_feff p50':>13s} {'p99.9':>9s}")
        print(hdr); print("-" * len(hdr))
        for r in recs:
            print(f"{r['ds']:18s} {r['n']:2d} {r['f_eff']:7.0f} | "
                  f"{r[nf_key][0]:12{nf_fmt}} {r[nf_key][1]:9{nf_fmt}} | "
                  f"{r[wf_key][0]:13{wf_fmt}} {r[wf_key][1]:9{wf_fmt}}")
        # cross-dataset radius spread (per-dataset p99.9), degenerate f_eff excluded
        good = [r for r in recs if r["f_eff"] < 1e6]
        nf = [r[nf_key][1] for r in good]; wf = [r[wf_key][1] for r in good]
        print(f"  radius spread across datasets (p99.9; ortho/degenerate excluded):")
        print(f"    no_feff  : {min(nf):.3f} .. {max(nf):.3f}  ({max(nf)/max(min(nf),1e-9):.1f}x)")
        print(f"    with_feff: {min(wf):.4f} .. {max(wf):.4f}  ({max(wf)/max(min(wf),1e-9):.1f}x)\n")

    # absolute grid: f_eff cancels the depth scaling -> O(1) AND camera-invariant
    # (works for ortho); no_feff ~ f_eff (huge, broken for ortho johnson-fly).
    _table("GRID (absolute) |target| per dataset:",
           "no_feff = rays_c·GT / cube (~f_eff units)   with_feff = rays_c·GT / (cube·f_eff)",
           "g_nf", "g_wf", ".1f", ".4f")
    # residual grid: MIRROR IMAGE. no_feff ~ pixels (camera-invariant, survives ortho);
    # with_feff collapses to ~0 for ortho (f_eff meaningless there).
    _table("GRIDRESID (residual) |target| per dataset:",
           "no_feff = motion / cube (~pixels)           with_feff = motion / (cube·f_eff)",
           "r_nf", "r_wf", ".2f", ".4f")

    print("KEY: f_eff cancels for ABSOLUTE depth (so grid wants /cube/f_eff, ortho-safe),")
    print("     but f_eff is the WRONG normalizer for MOTION (gridresid wants /cube ~pixels,")
    print("     ortho-safe). With f_eff the ortho johnson-fly residual collapses to ~0.\n")

    for k in agg:
        agg[k] = np.concatenate(agg[k], 0) if agg[k] else np.zeros((0, 3))
    agg_grid, agg_resid, agg_residnf, agg_logd = (
        [agg["grid"]], [agg["resid"]], [agg["residnf"]], [agg["logd"]])

    g = np.concatenate(agg_grid, 0)
    r_ = np.concatenate(agg_resid, 0)
    rnf = np.concatenate(agg_residnf, 0)
    ld = np.concatenate(agg_logd, 0)

    # ---- TRANSFORM analysis: linear vs signed-log effective bins ----
    print("\n=== TRANSFORM: linear vs signed-log bin utilization (effective bins / "
          f"{GRID}) ===")
    print("  Higher 'eff' = bins better used. A peaked-at-0 target wastes a linear "
          "grid; signed-log\n  spreads the central bulk across more bins. "
          "'resid_nofeff' = motion/cube ~ PIXELS\n  (camera-type-invariant; the only "
          "residual that survives ortho johnson-fly).\n")
    for name, arr in [("grid       (absolute  /cube/feff)", g),
                      ("gridresid  (residual  /cube/feff)", r_),
                      ("resid_nofeff (residual /cube ~px)", rnf)]:
        radius, eff_lin, eff_log, eps, p50 = transform_report(arr)
        gain = eff_log / max(eff_lin, 1e-9)
        verdict = ("LOG HELPS" if gain >= 1.5 else
                   "marginal" if gain >= 1.15 else "linear OK")
        print(f"  {name}: radius(p99.9)={radius:7.3f}  p50={p50:.4f}  "
              f"ratio={radius/max(p50,1e-9):5.1f}x")
        print(f"       linear eff bins = {eff_lin:6.1f}   signed-log eff bins = "
              f"{eff_log:6.1f} (eps={eps:.4f})  -> {gain:.2f}x  [{verdict}]")
        print()

    # ---- DEPTH range analysis ----
    print("=== DEPTH grid: log(depth/(cube*f_eff)) vs "
          f"[depth_log_min={args.depth_log_min}, depth_log_max={args.depth_log_max}] ===")
    dr = depth_report(ld, args.depth_log_min, args.depth_log_max)
    q = dr["q"]
    print(f"  log-depth percentiles: p0.1={q[0.1]:.2f}  p1={q[1]:.2f}  "
          f"p50={q[50]:.2f}  p99={q[99]:.2f}  p99.9={q[99.9]:.2f}")
    print(f"  fraction inside [{args.depth_log_min},{args.depth_log_max}]: {dr['within']*100:.2f}%   "
          f"effective bins = {dr['eff']:.1f}/{GRID}")
    print(f"  tighter data range [p0.5,p99.5] = [{dr['tight'][0]:.2f}, {dr['tight'][1]:.2f}]"
          f"  -> would give {dr['eff_tight']:.1f}/{GRID} effective bins")

    # ---- final radius suggestion ----
    gr = float(np.percentile(np.abs(g), 99.9))
    rr = float(np.percentile(np.abs(r_), 99.9))
    print("\n=== Suggested head_3d_grid_radius (linear grid) ===")
    print(f"  output_mode='grid'      -> ~{_round_up(gr):.2f}  (absolute; depth axis dominates)")
    print(f"  output_mode='gridresid' -> ~{_round_up(rr):.2f}  (f_eff-normalized residual)")


def _round_up(x):
    if not np.isfinite(x):
        return float("nan")
    for step in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        if x <= step:
            return step
    return float(np.ceil(x))


if __name__ == "__main__":
    main()
