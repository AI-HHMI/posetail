#!/usr/bin/env python
"""Offline geometric SIMULATION of the speed-targeted gridresid crop — NO model, NO training.

Goal: pick a crop SIZE so a spatially-coherent cluster of tracks realizes a chosen per-track
head-speed s*, drawn from a target band [s_lo, s_hi], and verify (across datasets) that the
pooled per-axis |head| the loss would see actually FILLS that band with small out-of-grid %.

Lever (image_size cancels; the ray-local residual is a rigid transform of the world residual):
    head = raylocal_residual / (cube_scale_orig * crop_size)
Set a high percentile (ref_pct) of the cluster's per-axis ray-local residual equal to s*:
    crop_size* = motion_ref / (cube_scale_orig * s*)
then crop a FIXED square of that size centered on the cluster (points may sweep toward/out of
frame — that lateral+depth sweep is what produces the large head residuals the edge bins need),
resize to image_size, recompute cube on the cropped cameras, and measure the per-axis |head|
over the VISIBLE frames/points/axes (visibility-masked, multi-camera) — the real training signal.

Nested cluster radii (tight->loose) give each cluster an achievable band
[lo,hi]=motion_ref/(cube*[frame_px, anchor_bbox_px]); pick the tightest whose band brackets s*.

Usage:
  pixi run python scripts/speed_target_coverage.py --split train --max-clips 20 \
      --s-lo 0.5 --s-hi 2.0 --ref-pct 99 --high-bias 0.5
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from posetail.posetail.cube import (get_camera_scale, points_to_rays, project_cam,   # noqa: E402
                                    is_point_visible, to_homogeneous, from_homogeneous)
from posetail.datasets.utils import get_dirs, load_yaml                              # noqa: E402
from scripts.estimate_scale_stats import (candidate_pool, _load_traj,                # noqa: E402
                                          _flat_pts, build_cams)


def _pad_bbox_side(uv):
    """max projected bbox side (+ the -20/+20=40px crop pad) for finite pixel coords, or None."""
    uv = uv[torch.all(torch.isfinite(uv), dim=1)]
    if uv.shape[0] < 2:
        return None
    return float(max(uv[:, 0].max() - uv[:, 0].min(), uv[:, 1].max() - uv[:, 1].min())) + 40.0


def _axis_resid(cam, cluster, at_pixel):
    """Per-axis |ray-local residual| (world units) of a cluster (L,k,3) about each point's
    first-valid anchor, using one ray frame built at `at_pixel` (1,2) on `cam`. Returns the
    pooled finite values over non-anchor VISIBLE frames, plus the (L,k) validity mask."""
    valid = torch.isfinite(cluster).all(-1)                     # (L,k)
    fv = valid.float().argmax(0)
    k = cluster.shape[1]
    rays = points_to_rays(cam, at_pixel, normalize_t=False)[0]  # (4,4)
    p_rl = from_homogeneous(torch.einsum('xr,...r->...x', rays, to_homogeneous(cluster)))
    anchor = p_rl[fv, torch.arange(k)]
    resid = (p_rl - anchor[None]).abs()                         # (L,k,3)
    keep = valid.clone()
    keep[fv, torch.arange(k)] = False                           # drop each point's anchor frame
    m = keep[..., None].expand_as(resid) & torch.isfinite(resid)
    return resid[m], valid


def forced_fcams(cams, cluster, crop_px, image_size, contain=True):
    """Per-camera square crop resized to image_size, returned as (src_cam, cropped_cam) pairs.
    contain=True (default, mirrors the dataset): side max(crop_px, this camera's cluster
    trajectory bbox), centered on the bbox, so points stay in frame (no out-of-frame sweep).
    contain=False (legacy A/B): FIXED side crop_px centered on the centroid -- points sweep out."""
    flat = cluster.reshape(-1, 3)
    out = []
    for c in cams:
        uv = project_cam(c, flat).reshape(-1, 2)
        uv = uv[torch.all(torch.isfinite(uv), dim=1)]
        if uv.shape[0] < 2:
            continue
        size = c['size'].double()
        if contain:
            pmin, pmax = uv.min(0).values, uv.max(0).values
            traj_side = float((pmax - pmin).max()) + 40.0        # +40 ~ the natural crop pad
            S = min(max(float(crop_px), traj_side), float(size[0]), float(size[1]))
            ctr = (pmin + pmax) / 2
        else:
            S = min(float(crop_px), float(size[0]), float(size[1]))
            ctr = uv.mean(0)
        if S < 1:
            continue
        low = torch.clamp(ctr - S / 2, torch.zeros(2, dtype=torch.float64), (size - S).clamp_min(0))
        scale = image_size / S
        mat = c['mat'].clone().double() * scale
        mat[2, 2] = 1.0
        out.append((c, {'name': c['name'], 'type': c['type'], 'mat': mat, 'ext': c['ext'].double(),
                    'dist': c['dist'].double(), 'offset': (c['offset'].double() + low) * scale,
                    'size': torch.tensor([image_size, image_size], dtype=torch.float64)}))
    return out


def realized_head(cams, cluster, crop_px, image_size, vis_mask=True, contain=True):
    """The per-axis |head| the loss would see for the cluster at the forced crop: crop+resize,
    recompute cube on the cropped cams, then (p_raylocal-anchor)/(cube*image_size) over VISIBLE
    (in-crop) non-anchor frames/points/axes across all cameras. Returns a dict with the pooled
    per-axis |head| ('all'), split into lateral x,y ('lat') and depth z ('dep'), plus the
    per-track peak |head| over visible frames+axes ('tmax')."""
    empty = dict(all=np.zeros(0), lat=np.zeros(0), dep=np.zeros(0), tmax=np.zeros(0))
    pairs = forced_fcams(cams, cluster, crop_px, image_size, contain=contain)
    if not pairs:
        return empty
    fcams = [fc for _, fc in pairs]
    pts = cluster.reshape(-1, 3)
    pts = pts[torch.isfinite(pts).all(1)]
    if pts.shape[0] < 8:
        return empty
    cube_r = float(torch.nanmedian(get_camera_scale(fcams, pts[None].double())))
    if not np.isfinite(cube_r) or cube_r <= 0:
        return empty
    valid = torch.isfinite(cluster).all(-1)
    fv = valid.float().argmax(0)
    k = cluster.shape[1]
    L = cluster.shape[0]
    centre = torch.tensor([[image_size / 2, image_size / 2]], dtype=torch.float64)
    allv, latv, depv, tmax = [], [], [], []
    oof_num = oof_den = 0                                                     # crop-induced out-of-frame
    flat3 = cluster.reshape(-1, 3)
    base0 = valid.clone(); base0[fv, torch.arange(k)] = False                 # non-anchor valid frames
    for src, c in pairs:
        rays = points_to_rays(c, centre, normalize_t=False)[0]
        p_rl = from_homogeneous(torch.einsum('xr,...r->...x', rays, to_homogeneous(cluster)))
        anchor = p_rl[fv, torch.arange(k)]
        head = ((p_rl - anchor[None]) / (cube_r * image_size)).abs()      # (L,k,3)
        vis = is_point_visible(c, flat3).reshape(L, k)                        # (L,k) visible in the crop
        vis_full = is_point_visible(src, flat3).reshape(L, k)                 # visible in the full frame
        # crop-INDUCED loss: in the full frame but pushed out by the crop (isolates the sweep,
        # excludes normal cross-camera / behind-camera invisibility).
        oof_num += int((base0 & vis_full & ~vis).sum()); oof_den += int((base0 & vis_full).sum())
        keep = (vis & valid) if vis_mask else valid.clone()                  # supervise out-of-frame too
        keep[fv, torch.arange(k)] = False
        km = keep[..., None].expand(-1, -1, 3) & torch.isfinite(head)
        allv.append(head[km])
        latv.append(head[..., :2][km[..., :2]])
        depv.append(head[..., 2][km[..., 2]])
        # per-track peak |head| over visible frames + all axes (one value per point)
        hmask = torch.where(km, head, torch.zeros_like(head))
        pk = hmask.reshape(L, k, 3).amax(dim=(0, 2))                      # (k,)
        seen = keep.any(0)
        tmax.append(pk[seen])
    cat = lambda xs: torch.cat(xs).numpy().reshape(-1) if xs else np.zeros(0)
    return dict(all=cat(allv), lat=cat(latv), dep=cat(depv), tmax=cat(tmax),
                oof=np.array([oof_num / max(oof_den, 1)]))


def solve_crop(win, med_cam, cube, s_star, ref_pct, n_radii, frac_lo, min_kpts,
               s_lo, s_hi, fallback):
    """Pick a cluster mask + crop_px so its per-axis head ref_pct-percentile ~ s* (resampling
    s* within the clip's achievable band if needed). Returns (mask, crop_px) or None."""
    L, P, _ = win.shape
    valid = torch.isfinite(win).all(-1)
    has = valid.any(0)
    if int(has.sum()) < min_kpts:
        return None
    fv = valid.float().argmax(0)
    kpt = win[fv, torch.arange(P)]                              # (P,3) anchor coord per point
    # movement-weighted center (extent proxy)
    w2 = win.clone(); w2[~valid] = float('nan')
    mov = torch.nan_to_num(w2.amax(0) - w2.amin(0), nan=0.0).norm(dim=-1)
    cand = torch.where(has)[0]
    weight = mov[cand] + 2.0
    ci = cand[torch.multinomial(weight / weight.sum(), 1).item()]
    dists = torch.linalg.norm(kpt - kpt[ci], dim=-1)
    dists = torch.where(has, dists, torch.full_like(dists, float('inf')))
    fd = dists[torch.isfinite(dists)]
    if fd.numel() < min_kpts or float(fd.max()) == 0:
        return None
    d_max = float(fd.max())
    frame_px = float(max(med_cam['size'][0], med_cam['size'][1]))

    cands = []
    for frac in np.geomspace(frac_lo, 1.0, n_radii):
        mask = (dists <= d_max * float(frac)) & has
        if int(mask.sum()) < min_kpts:
            continue
        cluster = win[:, mask]
        tbox = _pad_bbox_side(project_cam(med_cam, cluster.reshape(-1, 3)))  # trajectory bbox
        if tbox is None or tbox <= 0:
            continue
        tbox = min(tbox, frame_px)
        # cube on the CLUSTER points (uncropped med_cam): resize preserves the median
        # sensitivity over the SAME points, so cube_c*crop == cube_r_cropped*image_size
        # exactly -- i.e. this is the scale the loss will actually apply. Using the full
        # scene's cube (different visible-point set) under-targets badly.
        cpts = cluster.reshape(-1, 3)
        cpts = cpts[torch.isfinite(cpts).all(1)]
        if cpts.shape[0] > 512:
            cpts = cpts[torch.randperm(cpts.shape[0])[:512]]
        cube_c = float(torch.nanmedian(get_camera_scale([med_cam], cpts[None].double())))
        if not np.isfinite(cube_c) or cube_c <= 0:
            continue
        ctr_px = project_cam(med_cam, kpt[mask].mean(0, keepdim=True))  # cluster centroid pixel
        rvals, _ = _axis_resid(med_cam, cluster, ctr_px)
        if rvals.numel() < 10:
            continue
        motion_ref = float(np.percentile(rvals.cpu().numpy(), ref_pct))
        if motion_ref <= 0:
            continue
        cands.append(dict(mask=mask, tbox=tbox, motion_ref=motion_ref, cube=cube_c,
                          hi=motion_ref / (cube_c * tbox), lo=motion_ref / (cube_c * frame_px)))
    if not cands:
        return None

    s_lo_all = min(c['lo'] for c in cands)
    s_hi_all = max(c['hi'] for c in cands)
    if not (s_lo_all <= s_star <= s_hi_all):
        if fallback == 'resample_in_range':
            b_lo, b_hi = max(s_lo, s_lo_all), min(s_hi, s_hi_all)
            s_star = (float(np.random.uniform(b_lo, b_hi)) if b_lo < b_hi
                      else float(np.clip(s_star, s_lo_all, s_hi_all)))
        else:
            s_star = float(np.clip(s_star, s_lo_all, s_hi_all))

    bracket = [c for c in cands if c['lo'] <= s_star <= c['hi']]
    chosen = (min(bracket, key=lambda c: int(c['mask'].sum())) if bracket
              else min(cands, key=lambda c: min(abs(c['lo'] - s_star), abs(c['hi'] - s_star))))
    crop_px = float(np.clip(chosen['motion_ref'] / (chosen['cube'] * s_star), chosen['tbox'], frame_px))
    return chosen['mask'], int(round(crop_px))


def draw_sstar(s_lo, s_hi, high_bias):
    return float(s_lo + (s_hi - s_lo) * (np.random.uniform() ** high_bias))


def pick_med_cam(cams, pts):
    """The camera whose cube_scale is closest to the median (the loss's aggregate scale)."""
    cs = get_camera_scale(cams, pts[None].double()).reshape(-1).cpu().numpy()  # (n_cams,)
    med = np.nanmedian(cs)
    order = np.argsort(np.abs(np.nan_to_num(cs, nan=np.inf) - med))
    return cams[int(order[0])], float(med)


def sphere_mask(win, speed_thresh, f_lo, f_hi):
    """Legacy 'speed_thresh' crop: keep a log-uniform sphere fraction of points about a
    speed_thresh-biased center (mirrors sample_keypoints_sphere). Returns a point mask, or None.
    The crop itself is the natural in-frame trajectory bbox of the subset (containment with a
    tiny target), so this is an apples-to-apples head measurement vs the contained speed_target."""
    L, P, _ = win.shape
    valid = torch.isfinite(win).all(-1)
    has = valid.any(0)
    if int(has.sum()) < 2:
        return None
    fv = valid.float().argmax(0)
    kpt = win[fv, torch.arange(P)]
    # per-point avg speed = mean frame-to-frame displacement over valid steps
    step = torch.linalg.norm(win[1:] - win[:-1], dim=-1)                 # (L-1,P)
    sv = torch.isfinite(step)
    avg_speed = torch.where(sv, step, torch.zeros_like(step)).sum(0) / sv.float().sum(0).clamp_min(1)
    tot_mov = torch.nan_to_num(win.amax(0) - win.amin(0)).norm(dim=-1)   # extent proxy
    cand = torch.where(has)[0]
    dyn = (avg_speed >= speed_thresh) & has
    if bool(dyn.any()):
        pool = torch.where(dyn)[0]
        ci = int(pool[np.random.randint(len(pool))])
    else:
        w = tot_mov[cand] + 2.0
        ci = int(cand[torch.multinomial(w / w.sum(), 1).item()])
    dists = torch.linalg.norm(kpt - kpt[ci], dim=-1)
    dists = torch.where(has, dists, torch.full_like(dists, float('inf')))
    fd = dists[torch.isfinite(dists)]
    if fd.numel() < 2 or float(fd.max()) == 0:
        return None
    frac = float(np.exp(np.random.uniform(np.log(f_lo), np.log(f_hi))))
    mask = (dists <= float(fd.max()) * frac) & has
    return mask if int(mask.sum()) >= 2 else None


def process_clip(tpath, args, rng):
    traj = _load_traj(os.path.join(tpath, "pose3d.npz"))
    if traj is None:
        return None
    tt = traj.shape[0]
    wl = max(2, min(args.n_frames, tt))
    t0 = rng.randint(0, tt - wl) if tt > wl else 0
    win = traj[t0:t0 + wl].double()
    cams = build_cams(load_yaml(os.path.join(tpath, "metadata.yaml")))
    if not cams:
        return None
    scale_cams = [c for c in cams
                  if 0.5 * (float(c['mat'][0, 0]) + float(c['mat'][1, 1])) < 1e6]
    if not scale_cams:
        return None
    pts0 = _flat_pts(win, args.max_points, rng)
    if pts0 is None:
        return None
    med_cam, cube = pick_med_cam(scale_cams, pts0)
    if not np.isfinite(cube) or cube <= 0:
        return None
    acc = {k: [] for k in ('all', 'lat', 'dep', 'tmax', 'oof')}
    for _ in range(args.n_draws):
        if args.mode == 'speed_thresh':
            mask = sphere_mask(win, args.speed_thresh, args.st_frac_lo, args.st_frac_hi)
            if mask is None:
                continue
            crop_px = 1   # containment bumps to the natural in-frame trajectory bbox
        else:
            s_star = draw_sstar(args.s_lo, args.s_hi, args.high_bias)
            sol = solve_crop(win, med_cam, cube, s_star, args.ref_pct, args.n_radii,
                             args.frac_lo, args.min_kpts, args.s_lo, args.s_hi, args.fallback)
            if sol is None:
                continue
            mask, crop_px = sol
        h = realized_head(scale_cams, win[:, mask], crop_px, args.image_size,
                          vis_mask=not args.no_vis_mask, contain=not args.legacy_crop)
        for kk in acc:
            if h[kk].size:
                acc[kk].append(h[kk])
    if not acc['all']:
        return None
    return {kk: (np.concatenate(v) if v else np.zeros(0)) for kk, v in acc.items()}


def _bars(a, s_hi, nb=20):
    edges = np.linspace(0.0, s_hi, nb + 1)
    hist, _ = np.histogram(a, bins=edges)
    frac = hist / max(hist.sum(), 1)
    return "".join("▁▂▃▄▅▆▇█"[min(7, int(f * 8 / max(frac.max(), 1e-9)))] for f in frac)


def _pcts(a):
    return (np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 99), np.percentile(a, 99.9)) \
        if a.size else (float('nan'),) * 4


def report(name, n_ok, d, s_lo, s_hi):
    a, lat, dep, tmax = d['all'], d['lat'], d['dep'], d['tmax']
    oof = 100 * float(np.mean(d['oof'])) if d.get('oof') is not None and d['oof'].size else float('nan')
    oog = 100 * np.mean(a > s_hi) if a.size else float('nan')
    # per-track peak: the meaningful "does the track need an edge bin" distribution
    t_band = 100 * np.mean((tmax >= s_lo) & (tmax <= s_hi)) if tmax.size else float('nan')
    t_hi = 100 * np.mean(tmax >= s_lo) if tmax.size else float('nan')
    p = _pcts(a); pl = _pcts(lat); pd = _pcts(dep)
    print(f"  {name:>24s} clips={n_ok:>3} N={a.size:>9,d}  crop-induced-oof={oof:5.2f}%")
    print(f"      per-axis|head| all  p50={p[0]:4.2f} p90={p[1]:4.2f} p99={p[2]:4.2f} p99.9={p[3]:4.2f}  "
          f"OOG={oog:5.2f}%   {_bars(a, s_hi)}")
    print(f"      lateral(x,y)        p90={pl[1]:4.2f} p99={pl[2]:4.2f} p99.9={pl[3]:4.2f}   "
          f"depth(z) p90={pd[1]:4.2f} p99={pd[2]:4.2f} p99.9={pd[3]:4.2f}")
    print(f"      per-TRACK peak|head| p50={_pcts(tmax)[0]:4.2f} p90={_pcts(tmax)[1]:4.2f} "
          f"| tracks>= {s_lo}: {t_hi:4.1f}%  in[{s_lo},{s_hi}]: {t_band:4.1f}%   {_bars(tmax, s_hi)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/groups/karashchuk/karashchuklab/"
                    "animal-datasets-processed/posetail-finetuning-v3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--s-lo", type=float, default=0.5)
    ap.add_argument("--s-hi", type=float, default=2.0)
    ap.add_argument("--ref-pct", type=float, default=99.0)
    ap.add_argument("--high-bias", type=float, default=0.5, help="s* = s_lo+(s_hi-s_lo)*U**bias; <1 high-emph")
    ap.add_argument("--frac-lo", type=float, default=0.03, help="tightest cluster radius fraction")
    ap.add_argument("--n-radii", type=int, default=6)
    ap.add_argument("--min-kpts", type=int, default=4)
    ap.add_argument("--fallback", default="resample_in_range", choices=["resample_in_range", "clip"])
    ap.add_argument("--n-draws", type=int, default=12)
    ap.add_argument("--max-clips", type=int, default=20)
    ap.add_argument("--max-points", type=int, default=512)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-vis-mask", action="store_true",
                    help="supervise out-of-frame points too (fills edge bins with the sweep-out motion)")
    ap.add_argument("--legacy-crop", action="store_true",
                    help="A/B: use the OLD fixed centroid crop (points sweep out) instead of the contained crop")
    ap.add_argument("--mode", default="speed_target", choices=["speed_target", "speed_thresh"],
                    help="which sampling mode to simulate for the head-coverage comparison")
    ap.add_argument("--speed-thresh", type=float, default=3.0, help="speed_thresh mode: dynamic-center threshold")
    ap.add_argument("--st-frac-lo", type=float, default=0.05, help="speed_thresh mode: sphere-fraction low")
    ap.add_argument("--st-frac-hi", type=float, default=0.75, help="speed_thresh mode: sphere-fraction high")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets = args.datasets or get_dirs(args.root)
    print(f"root={args.root}  split={args.split}  n_frames={args.n_frames}  image_size={args.image_size}")
    print(f"target s* ~ [{args.s_lo}, {args.s_hi}] high_bias={args.high_bias}  ref_pct={args.ref_pct}  "
          f"frac_lo={args.frac_lo} n_radii={args.n_radii}  n_draws={args.n_draws} max_clips={args.max_clips}")
    print("head = |raylocal_residual| / (cube_scale_orig * crop_size); grid covers |head| <= s_hi.\n")

    pool_acc = {k: [] for k in ('all', 'lat', 'dep', 'tmax', 'oof')}
    n_total = 0
    for ds in datasets:
        if not os.path.isdir(os.path.join(args.root, ds, args.split)):
            continue
        pool = candidate_pool(args.root, ds, args.split, args.max_clips * 6, rng)
        accs, n_ok = {k: [] for k in pool_acc}, 0
        for c in pool:
            if n_ok >= args.max_clips:
                break
            try:
                h = process_clip(c, args, rng)
            except Exception:
                continue
            if h is not None and h['all'].size:
                for kk in accs:
                    accs[kk].append(h[kk])
                n_ok += 1
        if accs['all']:
            d = {kk: np.concatenate(v) for kk, v in accs.items()}
            report(ds, n_ok, d, args.s_lo, args.s_hi)
            for kk in pool_acc:
                pool_acc[kk].append(d[kk])
            n_total += n_ok
            print()
    if pool_acc['all']:
        report("ALL (pooled)", n_total, {kk: np.concatenate(v) for kk, v in pool_acc.items()},
               args.s_lo, args.s_hi)
    print("\nKey: lateral head is visibility-capped ~1 (points leave the crop); depth(z) is the")
    print("only unbounded axis. per-TRACK peak = does a track's motion demand an edge bin.")


if __name__ == "__main__":
    main()
