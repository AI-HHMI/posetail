"""Synthetic track corruption + triplet assembly for the track-quality scorer.

Given a clean loaded sample (a "good" track), we build a triplet of three samples:

    good   = the clean track                              (label +1, every point)
    bad    = corrupt(good): every point independently     (label -1, every point)
             gets >=1 corruption applied over time
    anchor = augment(good or bad, 50/50): a posetail       (label = source's label)
             geometric augmentation that preserves track validity

The scorer must rank `bad` as worse than `good` while being invariant to the augmentation
that produced `anchor`. Corruptions move COORDINATES ONLY (never pixels), so `good` and
`bad` share scene features; the 3D anchor augmentation (world-frame rotation) also leaves
pixels unchanged, while the 2D anchor augmentation (in-plane image rotation) warps them.

Adapted from miss-alignment's composable `ShiftConfig`/`ShiftGenerator`
(`data/shift_generation.py`), extended to per-point, per-time shifts `[K, T, D]`.
"""

import random

import numpy as np
import torch

from posetail.posetail.cube import get_camera_scale
from posetail.datasets.posetail_dataset import rotate_points_image_plane


# --------------------------------------------------------------------------------------
# Per-point shift generators. Each returns [P, T, D] in *unit* magnitude space (the
# per-type magnitude is applied by the caller; 3D shifts are later scaled by cube_scale,
# 2D shifts are already in pixels). Randomness is independent per point.
# --------------------------------------------------------------------------------------

def gen_constant_offset(P, T, D, device, mag):
    """One random offset vector per point, added to every frame -> [P, T, D]."""
    off = (torch.rand(P, 1, D, device=device) * 2 - 1) * mag
    return off.expand(P, T, D).clone()


def gen_frame_noise(P, T, D, device, mag):
    """Gaussian noise (random per-point std) on a random contiguous frame subset."""
    std = torch.rand(P, 1, 1, device=device) * mag                      # [P,1,1]
    noise = torch.randn(P, T, D, device=device) * std
    # per-point contiguous window [start, start+length)
    length = torch.randint(1, T + 1, (P,), device=device)
    start = (torch.rand(P, device=device) * (T - length + 1).float()).long()
    ar = torch.arange(T, device=device)[None, :]                        # [1,T]
    mask = (ar >= start[:, None]) & (ar < (start + length)[:, None])    # [P,T]
    return noise * mask[:, :, None].float()


def gen_gradual_drift(P, T, D, device, mag):
    """Linear ramp from 0 to a random per-point offset over time -> [P, T, D]."""
    off = (torch.rand(P, 1, D, device=device) * 2 - 1) * mag
    ramp = torch.linspace(0, 1, T, device=device)[None, :, None]        # [1,T,1]
    drift = ramp * off
    # randomly flip the ramp direction per point (drift toward vs away from origin)
    flip = (torch.rand(P, 1, 1, device=device) < 0.5).float()
    return drift * (1 - flip) + torch.flip(drift, dims=(1,)) * flip


def gen_sinusoid(P, T, D, device, mag, omega_lo=0.3, omega_hi=2.0):
    """amp * sin(omega * t + phase), random per-point amp/omega/phase (cos == phase shift)."""
    amp = torch.rand(P, 1, D, device=device) * mag
    omega = (torch.rand(P, 1, D, device=device) * (omega_hi - omega_lo) + omega_lo)
    phase = torch.rand(P, 1, D, device=device) * (2 * np.pi)
    t = torch.arange(T, device=device).float()[None, :, None]          # [1,T,1]
    return amp * torch.sin(omega * t + phase)


GENERATORS = {
    'const_offset': gen_constant_offset,
    'frame_noise': gen_frame_noise,
    'gradual_drift': gen_gradual_drift,
    'sinusoid': gen_sinusoid,
}


class PointCorruptor:
    """Composable per-point corruptor. Each tracked point INDEPENDENTLY samples which
    corruption types to apply (each type gated by its probability); points that draw no
    type get one random type forced on, so EVERY point is corrupted (clean -1 labels).

    `probs` and `mags` are dicts keyed by the GENERATORS names. `mags` holds the
    per-type magnitudes for the *current* coordinate space (cube-scale units for 3D,
    pixels for 2D)."""

    def __init__(self, probs, mags):
        self.probs = probs
        self.mags = mags
        self.names = [n for n in GENERATORS if probs.get(n, 0.0) > 0.0]
        assert self.names, 'PointCorruptor needs at least one enabled corruption type'

    def __call__(self, P, T, D, device):
        contributions = {}                                   # name -> [P,T,D]
        applied = torch.zeros(P, dtype=torch.bool, device=device)
        total = torch.zeros(P, T, D, device=device)
        for name in self.names:
            shift = GENERATORS[name](P, T, D, device, self.mags[name])
            contributions[name] = shift
            mask = torch.rand(P, device=device) < self.probs[name]
            total = total + shift * mask[:, None, None].float()
            applied = applied | mask

        # ensure at least one corruption per point: force a random type where none fired
        need = ~applied
        if need.any():
            pick = torch.randint(0, len(self.names), (P,), device=device)
            for ci, name in enumerate(self.names):
                m = need & (pick == ci)
                if m.any():
                    total = total + contributions[name] * m[:, None, None].float()
        return total


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _clone_cam(cam):
    """Shallow-copy a camera dict, cloning tensor fields so in-place edits (e.g. by
    rotate_points_image_plane, which mutates cam['mat']) can't corrupt the shared group."""
    out = dict(cam)
    for k, v in cam.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
    return out


def compute_cube_scale_b(cgroup, coords_full):
    """Median-over-cameras world-units scale per batch item -> [b]. coords_full: [b,t,n,3]."""
    b, t, n, _ = coords_full.shape
    cs = get_camera_scale(cgroup, coords_full.reshape(b, t * n, 3))      # [n_cams, b]
    return torch.median(cs, dim=0).values                               # [b]


def corrupt_coords(coords_full, corruptor, mode, cube_scale_b=None):
    """Return a corrupted copy of coords_full [b,t,n,R]; every point corrupted over time.

    3D shifts (cube-scale units) are multiplied by the per-batch cube_scale -> world units;
    2D shifts are already in pixels.
    """
    b, t, n, D = coords_full.shape
    device = coords_full.device
    shifts = corruptor(b * n, t, D, device)                            # [b*n, t, D]
    shifts = shifts.reshape(b, n, t, D).permute(0, 2, 1, 3)            # [b,t,n,D]
    if mode == '3d':
        assert cube_scale_b is not None
        shifts = shifts * cube_scale_b[:, None, None, None]
    return coords_full + shifts


def apply_appearance_aug(views, dataset):
    """Run the dataset's imgaug appearance pipeline on each view -> new view list.

    views: list of [b,t,h,w,3] float tensors in [0,1]. Returns CPU float tensors.
    Coordinates are unchanged (appearance-only). Pixels change -> recompute scene feats.
    """
    out = []
    for v in views:
        arr = (v.detach().cpu().numpy() * 255.0).astype(np.uint8)      # [b,t,h,w,3]
        b, t = arr.shape[:2]
        aug_cam = dataset.aug_per_camera.to_deterministic()
        for bi in range(b):
            imgs = [aug_cam(image=arr[bi, ti]) for ti in range(t)]
            imgs = [dataset.aug_per_image(image=im) for im in imgs]
            arr[bi] = np.stack(imgs, axis=0)
        out.append(torch.from_numpy(arr.astype(np.float32) / 255.0).to(v.device))
    return out


def rotate_anchor_2d(views, cgroup, coords, angle_range=45.0):
    """In-plane image rotation for the single-camera 2D path.

    Warps the image (cv2) and rotates the pixel coords + camera together, so the track
    stays valid. Returns (warped_views, rotated_coords, rotated_cgroup). One random angle
    for the whole batch (the camera group is shared).
    """
    import cv2
    cam = _clone_cam(cgroup[0])
    angle = float(np.random.uniform(-angle_range, angle_range))
    cam_rot, coords_rot, (M_2x3, (cw, ch)) = rotate_points_image_plane(cam, coords, angle)

    v = views[0]
    device = v.device
    imgs = v.detach().cpu().numpy()                                    # [b,t,h,w,3]
    b, t = imgs.shape[:2]
    warped = np.zeros((b, t, ch, cw, 3), dtype=np.float32)
    for bi in range(b):
        for ti in range(t):
            warped[bi, ti] = cv2.warpAffine(imgs[bi, ti], M_2x3, (cw, ch),
                                            flags=cv2.INTER_LINEAR)
    warped_views = [torch.from_numpy(warped).to(device)]
    return warped_views, coords_rot, [cam_rot]


# --------------------------------------------------------------------------------------
# Triplet assembly
# --------------------------------------------------------------------------------------

def make_triplet(batch, dataset, corruptor_3d, corruptor_2d, cfg):
    """Build a (good, bad, anchor) triplet from one collated batch.

    Returns a dict with three samples, each a tuple (views, coords_full, cgroup), plus the
    anchor's source label and a flag for whether the anchor's pixels changed (so the
    trainer can reuse `good`'s scene features for the anchor when they did not).
    """
    coords = batch.coords                                              # [b,t,n,R]
    R = coords.shape[-1]
    mode = '3d' if R == 3 else '2d'
    views = batch.views
    cgroup = batch.cgroup

    if mode == '3d':
        cube_scale_b = compute_cube_scale_b(cgroup, coords)
        bad_coords = corrupt_coords(coords, corruptor_3d, '3d', cube_scale_b)
    else:
        bad_coords = corrupt_coords(coords, corruptor_2d, '2d')

    good = (views, coords, cgroup)
    bad = (views, bad_coords, cgroup)                                  # same pixels

    from_good = random.random() < 0.5
    src_views, src_coords, src_cgroup = good if from_good else bad

    if mode == '3d':
        a_cgroup, a_coords = dataset.rotate_camera_group(
            [_clone_cam(c) for c in src_cgroup], src_coords)
        a_views = src_views
        pixels_changed = False
        if cfg.get('anchor_appearance_aug', False):
            a_views = apply_appearance_aug(src_views, dataset)
            pixels_changed = True
    else:
        a_views, a_coords, a_cgroup = rotate_anchor_2d(
            src_views, src_cgroup, src_coords,
            angle_range=cfg.get('anchor_rotate_2d_deg', 45.0))
        pixels_changed = True
        if cfg.get('anchor_appearance_aug', False):
            a_views = apply_appearance_aug(a_views, dataset)

    anchor = (a_views, a_coords, a_cgroup)
    anchor_label = 1.0 if from_good else -1.0
    return {
        'good': good, 'bad': bad, 'anchor': anchor,
        'anchor_label': anchor_label, 'mode': mode,
        'reuse_scene_for_anchor': not pixels_changed,
    }


def build_corruptors(corruption_cfg):
    """Construct the 3D and 2D PointCorruptors from a config block.

    Expects per-type probabilities (`*_prob`) plus `mag_3d`/`mag_2d` magnitude dicts keyed
    by the GENERATORS names.
    """
    probs = {n: corruption_cfg.get(f'{n}_prob', 0.0) for n in GENERATORS}
    mag_3d = dict(corruption_cfg.get('mag_3d', {}))
    mag_2d = dict(corruption_cfg.get('mag_2d', {}))
    corruptor_3d = PointCorruptor(probs, mag_3d)
    corruptor_2d = PointCorruptor(probs, mag_2d)
    return corruptor_3d, corruptor_2d
