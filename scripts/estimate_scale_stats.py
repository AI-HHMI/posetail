#!/usr/bin/env python
"""Estimate per-dataset coordinate / camera-scale statistics for posetail datasets.

Motivation
----------
``TrackerEncoder`` scales its 3D and depth head outputs by a fixed constant
(``scale_3d == scale_depth == 500``) *and* by a per-scene ``cube_scale``
(= world units per pixel, from ``cube.get_camera_scale``). The 2D head uses
``scale_2d == 128 == image_size / 2``, which is principled: its output is pixels
in the 256-px crop. The fixed ``500`` on the metric heads is a magic number
whose appropriateness depends on the actual data scale, and it sits *on top of*
``cube_scale`` (so the head must produce ``residual_world / (500 * cube_scale)``).
A too-large constant means the head operates far from O(1) and amplifies the
gradient flowing back into the shared backbone — an effect that worsens as
``latent_dim`` grows. This script measures the real magnitudes so the
output-scale parametrization can be chosen from data rather than guessed.

What it does
------------
* Samples one ``n_frames`` window per clip (like a training item) and builds the
  EXACT cameras training sees: ``crop_to_points`` (bbox +/- 20 px, squared to
  ``max(min_crop_dim, w, h)``) then ``resize`` to ``image_size`` (focal *= scale).
  ``cube_scale`` is then the repo's ``get_camera_scale`` on those final cameras.
* Bounded, image-free file-walk over ``root/<dataset>/<split>/<session>/<trial>``
  (the nesting ``PosetailDataset._generate_metadata`` assumes), sampling a few
  trials per dataset. Fast even for huge datasets (kubric) — never enters ``img/``.

Reported scales (median magnitude a head must output if its raw output were O(1))
--------------------------------------------------------------------------------
For each of {2D, 3D} x {direct, residual}:
* ``2d_direct``   : pixel offset from crop centre (bounded by image_size/2 = 128).
* ``2d_residual`` : per-window 2D displacement, in final crop pixels.
* ``3d_direct``   : ray-local position ~ camera->point depth, in cube_scale units
                    (~ effective focal); this is what ``output_mode='direct'`` regresses.
* ``3d_residual`` : per-window 3D motion (world units) / cube_scale.
A well-conditioned head wants scale ~= the measured value. Compare ``3d_direct``
against the current fixed ``scale_3d = 500``.

Usage
-----
    pixi run python scripts/estimate_scale_stats.py \
        --root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 \
        --split train --max-clips 4
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from posetail.posetail.cube import get_camera_scale, project_cam, to_homogeneous  # noqa: E402
from posetail.datasets.utils import get_dirs, load_yaml  # noqa: E402


def _mat(x, shape):
    """Coerce a (possibly nested) yaml matrix into a square/rect torch tensor."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(np.prod(shape))
    if a.size < n:
        raise ValueError(f"expected {n} values, got {a.size}")
    return torch.from_numpy(a[:n].reshape(shape))


def candidate_pool(root, dataset, split, pool_size, rng):
    """Return a shuffled pool of trial dirs (root/dataset/split/session/trial).

    Randomly orders sessions and stops listing trials once the pool is large
    enough, so it never enumerates every trial of a huge dataset (e.g.
    kubric-multiview). Existence of metadata/pose is checked later, in analyze().
    """
    base = os.path.join(root, dataset, split)
    if not os.path.isdir(base):
        return []
    sessions = get_dirs(base)
    rng.shuffle(sessions)
    pool = []
    for session in sessions:
        spath = os.path.join(base, session)
        trials = get_dirs(spath)
        rng.shuffle(trials)
        pool.extend(os.path.join(spath, t) for t in trials)
        if len(pool) >= pool_size:
            break
    rng.shuffle(pool)
    return pool


def _load_traj(pose_path):
    """Return the pose trajectory as a (T, P, 3) torch tensor (NaNs preserved).

    pose3d.npz['pose'] is (S, T, K, 3) per the dataset convention
    's t n r -> t (s n) r' (3D arrays are treated as a single track, S=1).
    """
    npz = np.load(pose_path, allow_pickle=True)
    key = "pose" if "pose" in npz else next(
        (k for k in npz.files if getattr(npz[k], "ndim", 0) >= 3
         and npz[k].shape[-1] == 3), None)
    if key is None:
        return None
    a = np.asarray(npz[key], dtype=np.float64)
    if a.ndim == 3:        # (T, K, 3) -> (S=1, T, K, 3)
        a = a[None]
    if a.ndim != 4:
        return None
    s, t, k, _ = a.shape
    traj = a.transpose(1, 0, 2, 3).reshape(t, s * k, 3)  # (T, P, 3)
    return torch.from_numpy(traj)


def _flat_pts(traj, max_pts, rng):
    """Finite points (N, 3) for cube_scale, subsampled for fast SVD."""
    pts = traj.reshape(-1, 3)
    pts = pts[torch.isfinite(pts).all(1)]
    if len(pts) < 8:
        return None
    if max_pts and len(pts) > max_pts:
        idx = torch.tensor(rng.sample(range(len(pts)), max_pts))
        pts = pts[idx]
    return pts


def project_px(cam, P3):
    """Distortion-aware projection of world points to pixels; returns (uv, in_front).

    Uses the repo's project_cam so coordinates match training exactly.
    """
    flat = P3.reshape(-1, 3)
    uv = project_cam(cam, flat).reshape(*P3.shape[:-1], 2)
    z = (to_homogeneous(flat) @ cam["ext"].T)[..., 2].reshape(*P3.shape[:-1])
    return uv, z > 1e-6


def crop_resize_cams(cams, pts, image_size, min_crop_dim, pad=20.0):
    """Apply crop_to_points + resize-to-image_size, mirroring PosetailDataset.

    crop_cgroup_to_points: bbox of projected points +/- pad px, squared to
    max(min_crop_dim, w, h) and capped at the image size (changes offset/size,
    not focal). resize_camera_group: scale = image_size / max(crop_size), with
    mat (focal+pp), offset and size all multiplied by scale. Cameras that see
    too few points are dropped. Returns the final cam dicts.
    """
    out = []
    for c in cams:
        uv, infront = project_px(c, pts)
        uv = uv[infront & torch.isfinite(uv).all(-1)]
        if len(uv) < 4:
            continue
        size = c["size"].to(torch.float64)
        low = torch.clamp(uv.min(0).values - pad, torch.zeros(2, dtype=torch.float64), size)
        high = torch.clamp(uv.max(0).values + pad, torch.zeros(2, dtype=torch.float64), size)
        base = max(float(min_crop_dim), float(high[0] - low[0]), float(high[1] - low[1]))
        crop_w = min(base, float(size[0]))
        crop_h = min(base, float(size[1]))
        scale = image_size / max(crop_w, crop_h)            # resize_camera_group
        mat = c["mat"].clone() * scale
        mat[2, 2] = 1.0
        out.append({
            "name": c["name"], "type": c["type"], "mat": mat, "ext": c["ext"],
            "dist": c["dist"], "offset": (c["offset"].to(torch.float64) + low) * scale,
            "size": torch.tensor([round(crop_w * scale), round(crop_h * scale)],
                                  dtype=torch.float64),
            "crop_px": max(crop_w, crop_h),                 # raw crop size (context)
        })
    return out


def build_cams(md):
    """Build the minimal cam dicts get_camera_scale / project_cam need.

    Robust to per-dataset yaml quirks: distortion may be flat [5] (rat7m) or
    nested [[5]] (3dpop); extrinsics may be 3x4 or 4x4.
    """
    ints = md.get("intrinsic_matrices", {})
    exts = md.get("extrinsic_matrices", {})
    dists = md.get("distortion_matrices", {})
    widths = md.get("camera_widths", {})
    heights = md.get("camera_heights", {})
    cams = []
    for cn, K in ints.items():
        if cn not in exts:
            continue
        mat = _mat(K, (3, 3))
        ext_flat = np.asarray(exts[cn], dtype=np.float64).reshape(-1)
        if ext_flat.size == 12:                       # 3x4 -> 4x4
            ext = torch.eye(4, dtype=torch.float64)
            ext[:3] = torch.from_numpy(ext_flat.reshape(3, 4))
        else:
            ext = _mat(exts[cn], (4, 4))
        dist_flat = np.asarray(dists.get(cn, []), dtype=np.float64).reshape(-1)
        dist = torch.zeros(5, dtype=torch.float64)
        dist[:min(5, dist_flat.size)] = torch.from_numpy(dist_flat[:5])
        w = float(widths.get(cn, 2 * mat[0, 2]))
        h = float(heights.get(cn, 2 * mat[1, 2]))
        cams.append({
            "name": cn, "type": "pinhole", "mat": mat, "ext": ext,
            "dist": dist, "size": torch.tensor([w, h]), "offset": torch.zeros(2),
        })
    return cams


def _med_finite(x):
    x = x[torch.isfinite(x)]
    return float(torch.median(x)) if x.numel() else float("nan")


def analyze(tpath, image_size, n_frames, min_crop_dim, max_pts, rng):
    md_path = os.path.join(tpath, "metadata.yaml")
    pose_path = os.path.join(tpath, "pose3d.npz")
    if not (os.path.exists(md_path) and os.path.exists(pose_path)):
        raise ValueError("missing metadata.yaml or pose3d.npz")
    traj = _load_traj(pose_path)                       # (T, P, 3), NaNs preserved
    if traj is None:
        raise ValueError("no (S,T,K,3) pose array")
    # sample one n_frames window, like a training item; the crop is built from it
    t_total = traj.shape[0]
    win_len = max(2, min(n_frames, t_total))
    t0 = rng.randint(0, t_total - win_len) if t_total > win_len else 0
    win = traj[t0:t0 + win_len]                        # (L, P, 3)
    pts = _flat_pts(win, max_pts, rng)                 # finite points in the window
    if pts is None:
        raise ValueError("< 8 finite 3D points in window")
    cams = build_cams(load_yaml(md_path))
    if not cams:
        raise ValueError("no usable cameras in metadata")

    # crop_to_points + resize-to-image_size -> the exact cameras training sees
    fcams = crop_resize_cams(cams, pts, image_size, min_crop_dim)
    if not fcams:
        raise ValueError("no camera sees the points")
    cube = float(torch.nanmedian(get_camera_scale(fcams, pts[None].double())))
    if not np.isfinite(cube) or cube <= 0:
        raise ValueError("cube_scale non-finite")
    crop_px = float(np.median([c["crop_px"] for c in fcams]))
    # effective focal of the cropped+resized camera (= fx the model is fed)
    f_eff = float(np.median([0.5 * (float(c["mat"][0, 0]) + float(c["mat"][1, 1]))
                             for c in fcams]))

    # DIRECT 3D: head regresses ray-local position ~ camera->point depth
    depths = []
    for c in fcams:
        R, t = c["ext"][:3, :3], c["ext"][:3, 3]
        depths.append(_med_finite(torch.linalg.norm(pts - (-R.T @ t), dim=1)))
    scale_3d_direct = float(np.nanmedian(depths)) / cube
    # if we instead scale by cube*f_eff (~ scene depth Z), the head outputs this:
    scale_3d_direct_norm = scale_3d_direct / f_eff

    # RESIDUAL 3D: head regresses motion across the window (world units / cube)
    scale_3d_residual = _med_finite(torch.linalg.norm(win[-1] - win[0], dim=-1)) / cube
    # same f_eff normalization, to test whether the direct-mode trick helps here
    scale_3d_residual_norm = scale_3d_residual / f_eff

    # 2D measured in the FINAL cropped+resized pixel frame (exact, no fudge)
    centre = image_size / 2.0
    off2d, disp2d = [], []
    for c in fcams:
        uv, infront = project_px(c, win)              # (L, P, 2)
        ok = infront.reshape(-1) & torch.isfinite(uv).all(-1).reshape(-1)
        flat = uv.reshape(-1, 2)[ok]
        if len(flat):
            off2d.append(_med_finite(torch.linalg.norm(flat - centre, dim=1)))
        valid = infront[-1] & infront[0]
        d = torch.linalg.norm(uv[-1] - uv[0], dim=-1)[valid]
        if d.numel():
            disp2d.append(_med_finite(d))
    scale_2d_direct = float(np.nanmedian(off2d)) if off2d else float("nan")
    scale_2d_residual = float(np.nanmedian(disp2d)) if disp2d else float("nan")

    return dict(cube=cube, crop_px=crop_px, f_eff=f_eff,
                scale_2d_direct=scale_2d_direct, scale_2d_residual=scale_2d_residual,
                scale_3d_direct=scale_3d_direct, scale_3d_residual=scale_3d_residual,
                scale_3d_direct_norm=scale_3d_direct_norm,
                scale_3d_residual_norm=scale_3d_residual_norm)


def _pct(xs, q):
    xs = [x for x in xs if x == x]
    return float(np.percentile(xs, q)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/groups/karashchuk/karashchuklab/"
                    "animal-datasets-processed/posetail-finetuning-v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="default: autodetect subdirs of --root")
    ap.add_argument("--max-clips", type=int, default=4,
                    help="usable clips to analyze per dataset")
    ap.add_argument("--max-points", type=int, default=2000,
                    help="subsample 3D points per clip for fast cube_scale SVD")
    ap.add_argument("--n-frames", type=int, default=16,
                    help="window length for residual (displacement) scales + crop")
    ap.add_argument("--min-crop-dim", type=int, default=64,
                    help="crop_to_points min crop size (matches dataset config)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--verbose", action="store_true",
                    help="print why individual clips are skipped")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    datasets = args.datasets or get_dirs(args.root)
    print(f"root={args.root}  split={args.split}  image_size={args.image_size}  "
          f"n_frames={args.n_frames}  min_crop_dim={args.min_crop_dim}  max_clips={args.max_clips}")
    print("Scales = median magnitude a head must output if its raw output were O(1), "
          "measured on the EXACT cropped+resized cameras (crop_to_points + resize).")
    print("2D in final crop pixels; 3D = world units / cube_scale. crop_px = raw crop size.\n")
    hdr = (f"{'dataset':20s} {'n':>3s} {'f_eff':>7s} | {'2d_direct':>9s} "
           f"{'2d_resid':>9s} | {'3d_direct':>9s} {'3d_dir/f':>9s} {'3d_resid':>9s}")
    print(hdr)
    print("-" * len(hdr))

    agg = {k: [] for k in ("scale_2d_direct", "scale_2d_residual", "scale_3d_direct",
                           "scale_3d_direct_norm", "scale_3d_residual",
                           "scale_3d_residual_norm")}
    for ds in datasets:
        if not os.path.isdir(os.path.join(args.root, ds, args.split)):
            print(f"{ds:20s}  (no '{args.split}' split)")
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
            print(f"{ds:20s}  (no usable clips)")
            continue
        med = lambda k: float(np.nanmedian([r[k] for r in rows]))  # noqa: E731
        for k in agg:
            agg[k] += [r[k] for r in rows]
        print(f"{ds:20s} {len(rows):3d} {med('f_eff'):7.0f} | {med('scale_2d_direct'):9.1f} "
              f"{med('scale_2d_residual'):9.1f} | {med('scale_3d_direct'):9.0f} "
              f"{med('scale_3d_direct_norm'):9.2f} {med('scale_3d_residual'):9.1f}")

    def spread(k):
        lo, mid, hi = _pct(agg[k], 10), _pct(agg[k], 50), _pct(agg[k], 90)
        return f"{mid:8.2f}   (p10 {lo:.2f} / p90 {hi:.2f}, {hi / max(lo, 1e-9):.1f}x spread)"

    print("\nOVERALL (median across all clips):")
    for k in ("scale_2d_direct", "scale_2d_residual", "scale_3d_direct",
              "scale_3d_direct_norm", "scale_3d_residual", "scale_3d_residual_norm"):
        print(f"  {k:24s} = {spread(k)}")
    print("\nCurrent code: output_mode='direct' uses scale_2d=128, scale_3d=scale_depth=500.")
    print("Interpretation:")
    print("  * 2d_direct  bounded by image_size/2=128 -> current scale_2d=128 is correct.")
    print("  * 3d_direct  ~ effective focal f_eff; one fixed 500 can't fit its ~15x spread.")
    print("  * 3d_dir/f = 3d_direct / f_eff: scaling the direct output by (cube_scale * f_eff)")
    print("    instead of (cube_scale * 500) makes the target ~1 and MUCH more uniform across")
    print("    datasets (see scale_3d_direct_norm spread vs scale_3d_direct). f_eff = fx of the")
    print("    cropped+resized camera, already available to the model.")
    print("  * RESIDUAL: f_eff does NOT help. residual target = per-window motion in pixels;")
    print("    its spread is from motion speed (temporal), not geometry. Compare the spreads of")
    print("    scale_3d_residual vs scale_3d_residual_norm (dividing by f_eff does not shrink it).")


if __name__ == "__main__":
    main()
