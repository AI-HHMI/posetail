"""Synthetic track corruption + triplet assembly for the track-quality scorer.

Given a clean loaded sample, we build a triplet of three samples. ALL three get both rotation
and appearance augmentation (the scorer never sees a raw un-augmented view):

    good   = clean track,   augmentation A                (label +1, every point)
    bad    = corrupt(good),  augmentation A (SAME as good) (label -1, every point)
             every point independently gets >=1 corruption applied over time
    anchor = same track as good|bad (50/50), augmentation B (label = source's label)
             an INDEPENDENT augmentation the scorer must be invariant to

`good` and `bad` share ONE augmentation (A) so they stay pixel-matched -- corruptions move
COORDINATES ONLY, so they share scene features and the good-vs-bad gap isolates track quality.
The `anchor` gets an independent augmentation (B) of the same underlying track, so its pixels
always differ (reuse_scene_for_anchor is always False). Rotation uses the dataset's real
image-plane rotation: `rotate_anchor_2d` (single camera) / `rotate_anchor_3d` (per camera,
extrinsic Z-roll, projection-consistent); appearance uses the dataset's imgaug pipeline.

Adapted from miss-alignment's composable `ShiftConfig`/`ShiftGenerator`
(`data/shift_generation.py`), extended to per-point, per-time shifts `[K, T, D]`.
"""

import math
import random

import numpy as np
import torch
from easydict import EasyDict as edict

from posetail.posetail.cube import get_camera_scale
from posetail.datasets.posetail_dataset import (rotate_points_image_plane,
                                                rotate_camera_image_plane_3d)


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


def compute_drop_mask(coords, cfg, k_min):
    """Boolean [b,t,n] mask of coord slots to NaN out. Per point, randomly use a contiguous
    window OR per-frame random drops (50/50). Only currently-valid frames are dropped, and
    every point keeps >= k_min valid frames (the dataset's min_valid_frames, so drops never
    push a point below what the base loader would keep). torch-only RNG (worker-safe: torch
    is auto-seeded per DataLoader worker, unlike numpy)."""
    p = cfg.get('point_drop_prob', 0.0)
    if p <= 0.0:
        return None
    max_frac = cfg.get('point_drop_max_frac', 0.4)
    rate     = cfg.get('point_drop_bernoulli_rate', 0.2)
    b, t, n, _ = coords.shape
    device = coords.device

    valid    = torch.isfinite(coords).all(dim=-1)              # [b,t,n]
    n_valid  = valid.sum(dim=1)                                # [b,n]
    max_drop = (n_valid - k_min).clamp(min=0)                  # [b,n]

    # contiguous window per point
    L_max  = max(1, int(math.ceil(max_frac * t)))
    length = torch.randint(1, L_max + 1, (b, n), device=device)
    start  = (torch.rand(b, n, device=device) * (t - length + 1).clamp(min=1).float()).long()
    ar     = torch.arange(t, device=device)[None, :, None]     # [1,t,1]
    window = (ar >= start[:, None, :]) & (ar < (start + length)[:, None, :])   # [b,t,n]

    # per-frame random per point
    bern = torch.rand(b, t, n, device=device) < rate           # [b,t,n]

    # 50/50 pattern choice per point, gated by point_drop_prob, restricted to valid frames
    use_window = (torch.rand(b, n, device=device) < 0.5)[:, None, :]
    gate       = (torch.rand(b, n, device=device) < p)[:, None, :]
    cand = torch.where(use_window, window, bern) & gate & valid

    # cap per point so >= k_min valid frames remain (cumulative count over time)
    order = cand.cumsum(dim=1)
    return cand & (order <= max_drop[:, None, :])              # [b,t,n]


def apply_drop_mask(coords, mask):
    """Return a copy of coords [b,t,n,R] with `mask` [b,t,n] slots set to NaN."""
    if mask is None:
        return coords
    nan = torch.full_like(coords, float('nan'))
    return torch.where(mask.unsqueeze(-1), nan, coords)


def apply_appearance_aug(views, dataset):
    """Run the dataset's imgaug appearance pipeline on each view -> new view list.

    views: list of [b,t,h,w,3] float tensors in [0,1]. Returns CPU float tensors.
    Coordinates are unchanged (appearance-only). Pixels change -> recompute scene feats.
    """
    out = []
    for v in views:
        arr = (v.detach().cpu().numpy() * 255.0).astype(np.uint8)      # [b,t,h,w,3]
        b, t, h, w = arr.shape[:4]
        # imgcorruptlike augmenters (DefocusBlur) assert h,w >= 32. Tiny views (e.g. a rotated
        # crop of a small source frame) would crash; symmetric-pad up to 32 for the aug, then
        # crop back so the view shape and its coords stay unchanged.
        pad_h, pad_w = max(0, 32 - h), max(0, 32 - w)
        top, left = pad_h // 2, pad_w // 2
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, 0), (top, pad_h - top), (left, pad_w - left), (0, 0)),
                         mode='symmetric')
        aug_cam = dataset.aug_per_camera.to_deterministic()
        for bi in range(b):
            imgs = [aug_cam(image=arr[bi, ti]) for ti in range(t)]
            imgs = [dataset.aug_per_image(image=im) for im in imgs]
            arr[bi] = np.stack(imgs, axis=0)
        if pad_h or pad_w:
            arr = arr[:, :, top:top + h, left:left + w, :]
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


def rotate_anchor_3d(views, cgroup, angle_range=45.0):
    """Per-view in-plane image rotation for the multi-camera 3D path.

    Each camera gets an INDEPENDENT random angle (mirrors dataset.augment_image_rotation, but
    forced -- no aug_prob gate). Warps that camera's images (cv2) and applies the matching
    extrinsic Z-roll so the 3D->2D projection stays consistent; the 3D world coords are left
    unchanged (rotation is camera-only), so the caller keeps using the same coords.

    views: list (per camera) of [b,t,h,w,3] float tensors. Returns (warped_views, rotated_cgroup).
    """
    import cv2
    new_views = []
    new_cgroup = []
    for cam, v in zip(cgroup, views):
        angle = float(np.random.uniform(-angle_range, angle_range))
        cam_rot, (M_2x3, (cw, ch)) = rotate_camera_image_plane_3d(_clone_cam(cam), angle)

        device = v.device
        imgs = v.detach().cpu().numpy()                                # [b,t,h,w,3]
        b, t = imgs.shape[:2]
        warped = np.zeros((b, t, ch, cw, 3), dtype=np.float32)
        for bi in range(b):
            for ti in range(t):
                warped[bi, ti] = cv2.warpAffine(imgs[bi, ti], M_2x3, (cw, ch),
                                                flags=cv2.INTER_LINEAR)
        new_views.append(torch.from_numpy(warped).to(device))
        new_cgroup.append(cam_rot)
    return new_views, new_cgroup


def crop_views(views, crops):
    """Slice each view [b,t,h,w,3] to its per-camera [x1,y1,x2,y2] crop rect."""
    out = []
    for v, crop in zip(views, crops):
        x1, y1, x2, y2 = (int(c) for c in crop)
        out.append(v[:, :, y1:y2, x1:x2, :])
    return out


def resize_views(views, cgroup, target_res, coords=None):
    """Resize each view so its longest side == target_res, scaling the camera (and 2D coords).

    Mirrors `PosetailDataset.resize_camera_group` (fixed target, no randint jitter) but also
    resizes the actual image frames. `views`: list (per camera) of [b,t,h,w,3] float tensors.
    Returns (views, cgroup) for 3D, or (views, cgroup, coords) when 2D pixel `coords` are given.
    """
    import cv2
    if target_res is None or target_res == -1:
        return (views, cgroup) if coords is None else (views, cgroup, coords)
    out_views, out_cgroup, scale0 = [], [], None
    for ci, (v, cam) in enumerate(zip(views, cgroup)):
        cam = dict(cam)
        size = cam['size']
        scale = float(target_res) / float(size.max())
        new_size = torch.round(size.float() * scale).to(torch.int32)
        nw, nh = int(new_size[0]), int(new_size[1])
        arr = v.detach().cpu().numpy()                                 # [b,t,h,w,3]
        b, t = arr.shape[:2]
        resized = np.zeros((b, t, nh, nw, 3), dtype=arr.dtype)
        for bi in range(b):
            for ti in range(t):
                resized[bi, ti] = cv2.resize(arr[bi, ti], (nw, nh), interpolation=cv2.INTER_LINEAR)
        out_views.append(torch.from_numpy(resized).to(v.device))
        cam['size'] = new_size
        cam['mat'] = cam['mat'] * scale
        cam['mat'][2, 2] = 1
        if 'offset' in cam:
            cam['offset'] = cam['offset'] * scale
        out_cgroup.append(cam)
        if ci == 0:
            scale0 = scale
    if coords is None:
        return out_views, out_cgroup
    return out_views, out_cgroup, coords * scale0                      # 2D: single camera


# --------------------------------------------------------------------------------------
# Triplet assembly
# --------------------------------------------------------------------------------------

def make_triplet(batch, dataset, corruptor_3d, corruptor_2d, cfg):
    """Build a (good, bad, anchor) triplet from one collated batch.

    All three samples get both rotation and appearance augmentation (the network never sees a
    raw un-augmented view). `good` and `bad` share ONE augmentation so they stay pixel-matched
    (only the coord corruption distinguishes them, and `score_triplet` can reuse good's scene
    features for bad); the `anchor` gets an INDEPENDENT augmentation of the SAME track as its
    source, which is the invariance target the scorer must match.

    Returns a dict with three samples, each a tuple (views, coords_full, cgroup), the anchor's
    source label, the mode, and reuse_scene_for_anchor (always False: the anchor's pixels differ).
    """
    coords = batch.coords                                              # [b,t,n,R]
    R = coords.shape[-1]
    mode = '3d' if R == 3 else '2d'
    views = batch.views
    cgroup = batch.cgroup
    ang = cfg.get('anchor_rotate_2d_deg', 45.0)                        # shared 2D & 3D angle range
    do_crop = cfg.get('crop_to_points', True)                          # scorer owns crop-to-points
    target_res = cfg.get('target_res', 256)                            # scorer owns the resize
    aug_prob = cfg.get('should_augment_prob', 0.0)                     # gates the appearance aug

    # Each view runs rotate -> crop-to-points -> resize -> appearance aug (mirrors the base
    # pipeline order, so the crop's min_crop_dim guarantee and the resize both hold BEFORE the
    # imgaug appearance pipeline sees the image).

    # (A) one shared transform for the good/bad scene -> they stay pixel-matched
    if mode == '3d':
        s_views, s_cgroup = rotate_anchor_3d(views, cgroup, angle_range=ang)
        s_coords = coords                                             # world coords unchanged by image rotation
        if do_crop:
            s_cgroup, crops = dataset.crop_cgroup_to_points(s_cgroup, s_coords[0])
            s_views = crop_views(s_views, crops)
        s_views, s_cgroup = resize_views(s_views, s_cgroup, target_res)
        if random.random() < aug_prob:
            s_views = apply_appearance_aug(s_views, dataset)
        cube_scale_b = compute_cube_scale_b(s_cgroup, s_coords)
        bad_coords = corrupt_coords(s_coords, corruptor_3d, '3d', cube_scale_b)
        drop_mask = compute_drop_mask(s_coords, cfg, dataset.min_valid_frames)  # shared: good & bad drop identically
        s_coords = apply_drop_mask(s_coords, drop_mask)
        bad_coords = apply_drop_mask(bad_coords, drop_mask)
    else:
        s_views, s_coords, s_cgroup = rotate_anchor_2d(views, cgroup, coords, angle_range=ang)
        if do_crop:
            s_cgroup, crops, s_coords_u = dataset.crop_cgroup_to_points_2d(s_cgroup, s_coords[0])
            s_views = crop_views(s_views, crops)
            s_coords = s_coords_u[None]
        s_views, s_cgroup, s_coords_u = resize_views(s_views, s_cgroup, target_res, coords=s_coords[0])
        s_coords = s_coords_u[None]
        if random.random() < aug_prob:
            s_views = apply_appearance_aug(s_views, dataset)
        bad_coords = corrupt_coords(s_coords, corruptor_2d, '2d')
        drop_mask = compute_drop_mask(s_coords, cfg, dataset.min_valid_frames)  # shared: good & bad drop identically
        s_coords = apply_drop_mask(s_coords, drop_mask)
        bad_coords = apply_drop_mask(bad_coords, drop_mask)

    good = (s_views, s_coords, s_cgroup)
    bad = (s_views, bad_coords, s_cgroup)                              # shares good's augmented pixels

    # (B) independent transform for the anchor, on the SAME track as its source
    from_good = random.random() < 0.5
    a_src_coords = s_coords if from_good else bad_coords

    if mode == '3d':
        # 3D image rotation is camera-only, so build from the ORIGINAL views (single warp) and
        # keep the identical world coords -> same track, independent view.
        a_views, a_cgroup = rotate_anchor_3d(views, cgroup, angle_range=ang)
        a_coords = a_src_coords
        if do_crop:
            # crop from PRE-drop coords so heavy point-drops can never empty the crop bbox
            a_cgroup, a_crops = dataset.crop_cgroup_to_points(a_cgroup, coords[0])
            a_views = crop_views(a_views, a_crops)
        a_views, a_cgroup = resize_views(a_views, a_cgroup, target_res)
        if random.random() < aug_prob:
            a_views = apply_appearance_aug(a_views, dataset)
    else:
        # 2D coords ARE pixels, so rotate the shared source again -> the anchor's rotated track
        # stays identical to its source. Crop-to-points + resize exactly like the 3D anchor (gated
        # by the same crop_to_points flag) so both modes apply cropping consistently.
        a_views, a_coords_u, a_cgroup = rotate_anchor_2d(s_views, s_cgroup, a_src_coords[0],
                                                         angle_range=ang)
        if do_crop:
            a_cgroup, a_crops, a_coords_u = dataset.crop_cgroup_to_points_2d(a_cgroup, a_coords_u)
            a_views = crop_views(a_views, a_crops)
        a_views, a_cgroup, a_coords_u = resize_views(a_views, a_cgroup, target_res, coords=a_coords_u)
        a_coords = a_coords_u[None]
        if random.random() < aug_prob:
            a_views = apply_appearance_aug(a_views, dataset)

    anchor = (a_views, a_coords, a_cgroup)
    anchor_label = 1.0 if from_good else -1.0
    return {
        'good': good, 'bad': bad, 'anchor': anchor,
        'anchor_label': anchor_label, 'mode': mode,
        'reuse_scene_for_anchor': False,
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


# --------------------------------------------------------------------------------------
# Dataset wrapper -- run the whole triplet build (rotation + appearance aug + corruption)
# in the DataLoader worker processes instead of the training loop's main process.
# --------------------------------------------------------------------------------------

class ScorerTripletDataset(torch.utils.data.Dataset):
    """Wraps a `PosetailDataset` so each item is a ready `(good, bad, anchor)` triplet.

    Building the triplet is CPU-bound (`cv2.warpAffine`, imgaug appearance aug, torch CPU
    ops), so doing it here means it runs inside the DataLoader workers -- pipelined with the
    GPU step -- rather than synchronously in the training loop. Base items come off disk as
    CPU tensors, so `make_triplet`'s internal `.to(device)` calls are no-ops here; the triplet
    stays on CPU until the main loop moves it to the GPU. On Linux fork each worker already
    holds the base dataset's imgaug pipelines, so `apply_appearance_aug` works unchanged.
    """

    GETITEM_MAX_RETRIES = 10

    def __init__(self, base, corruption_cfg):
        self.base = base
        self.cfg = dict(corruption_cfg)
        # plain-dict PointCorruptors -> fork-safe, no CUDA state
        self.corruptor_3d, self.corruptor_2d = build_corruptors(self.cfg)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # A single worker exception here (e.g. a degenerate crop) would crash the rank and hang
        # the whole DDP job on the next NCCL collective, so any triplet-build failure retries a
        # different sample instead of propagating.
        for _ in range(self.GETITEM_MAX_RETRIES):
            item = self.base[idx]
            if item is not None:
                try:
                    views, coords, vis, fnums, cgroup, row, query_times, vis_2d, p2d = item
                    mini = edict({'views': [v[None] for v in views],   # add b=1 -> [1,t,h,w,3]
                                  'coords': coords[None],               # [1,t,n,R]
                                  'cgroup': cgroup})
                    trip = make_triplet(mini, self.base, self.corruptor_3d, self.corruptor_2d, self.cfg)
                    trip['sample_info'] = row      # carry for the bad-gradient error print
                    return trip
                except Exception as e:
                    print(f"[ScorerTripletDataset] triplet build failed for idx {idx}: "
                          f"{type(e).__name__}: {e}; retrying another sample")
            idx = int(np.random.randint(len(self.base)))
        return None                            # exhausted retries (the loop's None-skip handles it)


def triplet_collate(batch):
    """Trivial collate for the scorer: one full b=1 triplet dict per step.

    The scorer runs `batch_size=1` because each camera's rotated crop has a variable size, so
    there is nothing to stack -- just unwrap the single item.
    """
    assert len(batch) == 1, 'scorer runs batch_size=1 (variable per-camera rotated sizes)'
    return batch[0]


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: decorrelate numpy RNG across workers.

    PyTorch auto-seeds torch and Python's `random` per worker (so corruption and the anchor
    coin-flip are already independent), but NOT numpy -- and the rotation angles use
    `np.random.uniform`. Seed numpy per worker so the workers don't emit correlated angles.
    """
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
