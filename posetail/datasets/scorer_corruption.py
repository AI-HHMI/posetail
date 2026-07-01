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
    app = cfg.get('anchor_appearance_aug', False)

    # (A) one shared transform for the good/bad scene -> they stay pixel-matched
    if mode == '3d':
        s_views, s_cgroup = rotate_anchor_3d(views, cgroup, angle_range=ang)
        s_coords = coords                                             # world coords unchanged by image rotation
        if app:
            s_views = apply_appearance_aug(s_views, dataset)
        cube_scale_b = compute_cube_scale_b(s_cgroup, s_coords)
        bad_coords = corrupt_coords(s_coords, corruptor_3d, '3d', cube_scale_b)
    else:
        s_views, s_coords, s_cgroup = rotate_anchor_2d(views, cgroup, coords, angle_range=ang)
        if app:
            s_views = apply_appearance_aug(s_views, dataset)
        bad_coords = corrupt_coords(s_coords, corruptor_2d, '2d')

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
        if app:
            a_views = apply_appearance_aug(a_views, dataset)
    else:
        # 2D coords ARE pixels, so rotate the shared-augmented source again -> the anchor's
        # (rotated) track stays identical to its source.
        a_views, a_coords, a_cgroup = rotate_anchor_2d(s_views, s_cgroup, a_src_coords,
                                                       angle_range=ang)
        if app:
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

    def __init__(self, base, corruption_cfg):
        self.base = base
        self.cfg = dict(corruption_cfg)
        # plain-dict PointCorruptors -> fork-safe, no CUDA state
        self.corruptor_3d, self.corruptor_2d = build_corruptors(self.cfg)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if item is None:
            return None                        # mirror PosetailDataset (None is rare, retry-guarded)
        views, coords, vis, fnums, cgroup, row, query_times, vis_2d, p2d = item
        mini = edict({'views': [v[None] for v in views],   # add b=1 -> [1,t,h,w,3]
                      'coords': coords[None],               # [1,t,n,R]
                      'cgroup': cgroup})
        trip = make_triplet(mini, self.base, self.corruptor_3d, self.corruptor_2d, self.cfg)
        trip['sample_info'] = row              # carry for the bad-gradient error print
        return trip


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
