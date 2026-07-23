"""
Example invocation (trial directory with img/ folder):

    python inference_video.py \
        --base-folder /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX \
        --trial-path /path/to/session/trial/ \
        --start-frame 0 \
        --n-frames 256 \
        --n-overlap 2 \
        --checkpoint 10000 \
        --device cuda:0 \
        --outpath /path/to/output.npz

The trial directory should contain:
    - metadata.yaml (camera calibration)
    - pose3d.npz (3D pose data, used for initial query points)
    - img/ (per-camera subdirectories of images) or vid/ (per-camera .mp4 files)
"""
import os
import cv2
import glob
import json
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

from decord import VideoReader, cpu
from aniposelib.cameras import CameraGroup, Camera

from posetail.datasets.utils import disassemble_extrinsics
from posetail.posetail.cube import project_points_torch
from posetail.posetail.tracker_encoder import TrackerEncoder
from train_utils import dict_to_device, load_config, load_checkpoint, format_camera_group


class ImageFolderReader:
    """Mimics the VideoReader interface but reads from a folder of images."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.filenames = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        if len(self.filenames) == 0:
            raise FileNotFoundError(f'No images found in {folder_path}')

    def __len__(self):
        return len(self.filenames)

    def get_batch(self, frame_ids):
        imgs = []
        for idx in frame_ids:
            path = os.path.join(self.folder_path, self.filenames[idx])
            img = cv2.imread(path)
            if img is None:
                raise IOError(f'Failed to read image: {path}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        return np.stack(imgs, axis=0)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.filenames)))
            return self.get_batch(list(indices))
        elif isinstance(key, int):
            return self.get_batch([key])[0]
        else:
            raise TypeError(f'Invalid key type: {type(key)}')


def build_video_readers(video_paths):
    readers = []
    for video_path in video_paths:
        if os.path.isdir(video_path):
            readers.append(ImageFolderReader(video_path))
        else:
            readers.append(VideoReader(video_path, ctx=cpu(0)))
    lengths = [len(reader) for reader in readers]
    return readers, lengths


def camera_group_to_device(camera_group, device):
    return [dict_to_device(cam_dict, device) for cam_dict in camera_group]


def crop_frames_with_padding(frames, x1, y1, x2, y2, pad_value=0):
    """Crop frames[..., y1:y2, x1:x2] with zero-padding for out-of-frame regions.

    Unlike a raw numpy slice (which wraps for negative coords and truncates for
    coords past the edge), this returns exactly a (x2-x1) x (y2-y1) crop with any
    off-frame area filled with ``pad_value``, so the pixels stay aligned with the
    camera geometry that assumes the crop spans [x1,x2) x [y1,y2).

    ``frames`` is (..., H, W, C); x/y are ints that may fall outside [0, W)/[0, H).
    """
    *lead, H, W, C = frames.shape
    out = np.full((*lead, y2 - y1, x2 - x1, C), pad_value, dtype=frames.dtype)
    sx1, sy1 = max(x1, 0), max(y1, 0)
    sx2, sy2 = min(x2, W), min(y2, H)
    if sx2 > sx1 and sy2 > sy1:
        out[..., sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1, :] = frames[..., sy1:sy2, sx1:sx2, :]
    return out


def load_multiview_clip(readers, start_frame, n_frames, crop_boxes=None, target_sizes=None):
    max_available = min(len(reader) for reader in readers)
    end_frame = min(start_frame + n_frames, max_available)

    if end_frame <= start_frame:
        raise ValueError(f'No synchronized frames available for start_frame={start_frame}')

    views = []

    for cam_idx, reader in enumerate(readers):
        frames = reader[start_frame:end_frame]
        if hasattr(frames, 'asnumpy'):
            frames = frames.asnumpy()

        if crop_boxes is not None:
            x1, y1, x2, y2 = crop_boxes[cam_idx].cpu().to(torch.int32).tolist()
            frames = crop_frames_with_padding(frames, x1, y1, x2, y2)

        if target_sizes is not None:
            target_size_cam = target_sizes[cam_idx]
            resized = [cv2.resize(frame, target_size_cam) for frame in frames]
            frames = np.stack(resized, axis=0)

        views.append(torch.from_numpy(frames))

    return views, end_frame - start_frame


def crop_camera_group_to_queries(camera_group, query_coords, min_crop_dim, padding=20, is_2d=False):
    if is_2d:
        # 2D queries are already pixel coords in the (single) camera frame; no
        # projection. Shape to (cams=1, ..., 2) so the cropping math below matches.
        p2d = query_coords.reshape(1, -1, 2)
    else:
        p2d = project_points_torch(camera_group, query_coords)
    crops = []

    for cnum in range(p2d.shape[0]):
        size = camera_group[cnum]['size']
        pflat = p2d[cnum].reshape(-1, 2)
        good = torch.all(torch.isfinite(pflat), dim=1)
        pflat = pflat[good]

        if pflat.shape[0] == 0:
            low = torch.tensor([0, 0], dtype=torch.int32, device=size.device)
            high = size.to(torch.int32)
        else:
            low = torch.clamp(
                torch.min(pflat, dim=0).values - padding,
                torch.tensor([0, 0], device=pflat.device),
                size.to(pflat.device),
            ).to(torch.int32)
            high = torch.clamp(
                torch.max(pflat, dim=0).values + padding,
                torch.tensor([0, 0], device=pflat.device),
                size.to(pflat.device),
            ).to(torch.int32)

            current_width = high[0] - low[0]
            current_height = high[1] - low[1]

            # Cap each axis at the image dimension so the crop never exceeds the
            # frame (matches posetail_dataset.crop_cgroup_to_points). Without the
            # per-axis cap, a wide bbox forces min_dim > frame height, and
            # `low = high - min_dim` yields a negative offset -> load_multiview_clip's
            # numpy slice wraps instead of padding, breaking the 2D<->3D geometry.
            base = max(min_crop_dim, int(current_width), int(current_height))
            min_dim_x = min(base, int(size[0]))
            min_dim_y = min(base, int(size[1]))

            if current_width < min_dim_x:
                center_x = (low[0] + high[0]) // 2
                low[0] = torch.clamp(center_x - min_dim_x // 2, 0, size[0] - min_dim_x)
                high[0] = low[0] + min_dim_x

            if current_height < min_dim_y:
                center_y = (low[1] + high[1]) // 2
                low[1] = torch.clamp(center_y - min_dim_y // 2, 0, size[1] - min_dim_y)
                high[1] = low[1] + min_dim_y

        crops.append(torch.cat([low, high]))

    camera_group_cropped = []
    for cnum in range(len(camera_group)):
        x1, y1, x2, y2 = crops[cnum]
        cam = dict(camera_group[cnum])
        cam['offset'] = cam['offset'] + torch.tensor(
            [x1, y1],
            dtype=torch.int32,
            device=cam['offset'].device,
        )
        cam['size'] = torch.tensor(
            [x2 - x1, y2 - y1],
            dtype=torch.int32,
            device=cam['size'].device,
        )
        camera_group_cropped.append(cam)

    return camera_group_cropped, crops


def resize_camera_group(camera_group, target_res):
    camera_group_scaled = []

    for cnum in range(len(camera_group)):
        cam = dict(camera_group[cnum])
        size = cam['size']
        scale = float(target_res) / max(size)
        cam['size'] = torch.round(size * scale).to(torch.int32)
        cam['mat'] = cam['mat'] * scale
        cam['mat'][2, 2] = 1

        if 'offset' in cam:
            # Keep offset float (matches posetail_dataset.resize_camera_group).
            # Rounding to int here left a sub-pixel (<0.5px) mismatch against the
            # exact `mat * scale`, since project_cam does `mat @ X - offset`.
            cam['offset'] = cam['offset'] * scale

        camera_group_scaled.append(cam)

    return camera_group_scaled


def resolve_config_and_checkpoint(base_folder, checkpoint=None):
    checkpoint_dir = os.path.join(base_folder, 'files', 'checkpoints')

    if checkpoint is None:
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')
        checkpoint_path = checkpoint_paths[-1]
    else:
        checkpoint_name = f'checkpoint_{str(checkpoint).zfill(8)}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    config_path = os.path.join(base_folder, 'files', 'config.toml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')

    return config_path, checkpoint_path


def load_camera_group_from_metadata(metadata_path, device='cpu'):
    with open(metadata_path, 'r') as f:
        cam_metadata = yaml.safe_load(f)

    offset_dict = cam_metadata.get('offset_dict', None)
    cam_type = cam_metadata.get('cam_type', 'pinhole')

    intrinsics_dict = cam_metadata['intrinsic_matrices']
    extrinsics_dict = cam_metadata['extrinsic_matrices']
    distortions_dict = cam_metadata['distortion_matrices']
    heights_dict = cam_metadata['camera_heights']
    widths_dict = cam_metadata['camera_widths']

    cam_names = list(intrinsics_dict.keys())
    if all(cam_name.isdigit() for cam_name in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)

    cams = []
    for cam_name in cam_names:
        rvec, tvec = disassemble_extrinsics(extrinsics_dict[cam_name])

        cam = Camera(
            matrix=intrinsics_dict[cam_name],
            dist=distortions_dict[cam_name],
            rvec=rvec,
            tvec=tvec,
            name=cam_name,
        )

        width = widths_dict[cam_name]
        height = heights_dict[cam_name]
        cam.set_size((width, height))
        cams.append(cam)

    camera_group = CameraGroup(cams)
    camera_group = format_camera_group(camera_group, offset_dict, cam_type, device=device)

    return camera_group


def load_camera_group_2d(video_paths, metadata_path=None, device='cpu'):
    """Build a single-camera group for a 2D (uncalibrated) trial.

    Mirrors PosetailDataset._build_2d_cgroup:
      (1) fully-calibrated metadata.yaml -> real camera;
      (2) metadata.yaml with only camera_widths/heights -> nominal pinhole;
      (3) no usable metadata -> read one image to get the size -> nominal pinhole.
    """
    from posetail.datasets.posetail_dataset import _make_nominal_2d_camera

    cam_meta = None
    if metadata_path is not None and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            cam_meta = yaml.safe_load(f)

    if cam_meta is not None and all(
            k in cam_meta for k in ('intrinsic_matrices', 'extrinsic_matrices',
                                    'distortion_matrices',
                                    'camera_heights', 'camera_widths')):
        # fully calibrated 2D metadata — use the real camera(s)
        return load_camera_group_from_metadata(metadata_path, device=device)

    if cam_meta is not None and 'camera_widths' in cam_meta and 'camera_heights' in cam_meta:
        widths, heights = cam_meta['camera_widths'], cam_meta['camera_heights']
        key = next(iter(widths))
        w, h = int(widths[key]), int(heights[key])
    else:
        # fall back to reading one image from the (single) camera folder
        cam_dir = video_paths[0]
        if os.path.isdir(cam_dir):
            sample = sorted(os.listdir(cam_dir))[0]
            from PIL import Image
            w, h = Image.open(os.path.join(cam_dir, sample)).size
        else:
            reader = VideoReader(cam_dir, ctx=cpu(0))
            frame = reader[0]
            if hasattr(frame, 'asnumpy'):
                frame = frame.asnumpy()
            h, w = frame.shape[:2]

    cgroup = _make_nominal_2d_camera(w, h)
    return camera_group_to_device(cgroup, device)


def load_model_from_base_folder(base_folder, checkpoint=None, device=None):
    config_path, checkpoint_path = resolve_config_and_checkpoint(base_folder, checkpoint=checkpoint)

    config = load_config(config_path)

    if device is None:
        device = torch.device(config.devices.device) if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_dict = load_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model = checkpoint_dict['model']
    model.eval()

    return model, config, config_path, checkpoint_path


def run_tracker_encoder_on_videos(
    model,
    video_paths,
    camera_group,
    query_points_3d,
    start_frame=0,
    n_frames=128,
    n_overlap=2,
    max_kpts=None,
    device=None,
    pred_key_3d='coords_pred',
    clip_len=None,
    occlusion_gt=None,
    query_times=None,
    motion_margin=True,
    conf_crop_thresh=0.1,
    carry_latent=False,
):
    """Windowed multi-view tracking whose per-chunk crop FOLLOWS the subject.

    Each chunk re-crops the video on the current (re-anchored) query so a moving subject
    stays in the model's field of view — the long-horizon fix that a single static crop
    over the whole clip lacks.

    query_times (per-point, window-relative to start_frame; None -> all zeros): each point
    is SEEDED in the chunk containing its query frame (mvtracker / query-first) and, once
    seeded, propagated forward by re-anchoring on the previous chunk's prediction. With all
    query_times == 0 (and motion_margin off) this reduces bit-for-bit to the legacy
    all-points-at-start behavior.

    Keypoint chunking is delegated to the model's internal ``kpt_chunk`` (scene encoded
    ONCE per window, reused across point slices — numerically identical, lower memory).
    NOTE: the model disables ``kpt_chunk`` for ``output_mode='gridnorm'`` (its per-camera
    gauge couples points), so ``max_kpts`` is a no-op there and a large point set may OOM.

    motion_margin: expand each chunk's crop by the previous chunk's per-point velocity
    (a causal approximation of the training trajectory-bounded crop) so a fast subject
    stays in-frame across the window; ~inert for slow subjects.
    conf_crop_thresh: active points whose predicted visibility (sigmoid) falls below this
    are excluded from the crop-bbox computation so a lost point can't drag the crop off the
    still-tracked cluster. None disables the exclusion.
    """

    # Dimensionality of the queries: R==3 -> 3D world coords (multi-view), R==2 ->
    # 2D pixel coords in a single camera frame. The model supports both.
    # pred_key_3d selects which 3D model output to use as the prediction (and the
    # recurrence query) — e.g. 'coords_pred' (default) or '3d_pred_triangulate'.
    R = query_points_3d.shape[-1]
    is_2d = (R == 2)

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    camera_group = camera_group_to_device(camera_group, device)

    readers, reader_lengths = build_video_readers(video_paths)

    max_available = min(reader_lengths)
    if start_frame < 0:
        raise ValueError('start_frame must be >= 0')
    if n_frames <= 0:
        raise ValueError('n_frames must be > 0')
    if n_overlap < 1:
        raise ValueError('n_overlap must be >= 1')
    if not hasattr(model, 'image_size'):
        raise AttributeError('model does not have an image_size attribute')
    if not hasattr(model, 'n_frames'):
        raise AttributeError('model does not have an n_frames attribute')
    if n_overlap >= model.n_frames:
        raise ValueError(f'n_overlap ({n_overlap}) must be less than model.n_frames ({model.n_frames})')
    if start_frame >= max_available:
        raise ValueError('start_frame is beyond the available frames in at least one video')

    current_frame = start_frame
    end_frame = min(start_frame + n_frames, max_available)

    current_queries = query_points_3d.to(device=device, dtype=torch.float32)
    if current_queries.ndim == 2:
        current_queries = current_queries.unsqueeze(0)
    N = current_queries.shape[1]
    # True query coordinate per point; seeded / not-yet-appeared points anchor on this
    # (their re-anchored prediction is only meaningful once they are active).
    original_queries = current_queries.clone()

    # Per-point ABSOLUTE first-visible frame (query-first). None -> everyone at start_frame,
    # which makes `before` always False and `seeded` true in chunk 0 -> the legacy path.
    if query_times is None:
        qt_abs = torch.full((N,), start_frame, device=device, dtype=torch.int64)
    else:
        qt_abs = start_frame + query_times.to(device=device, dtype=torch.int64).view(-1)

    coords_pred_all = []
    vis_pred_all = []
    conf_pred_all = []
    vis_pred_2d_all = []          # per-camera visibility logits (cams,b,t,n), for occlusion eval
    frame_numbers_all = []
    crop_history = []
    is_first_chunk = True
    # Latent threaded across chunks (cross-chunk carry), only when carry_latent=True. The
    # carried latent is the full-N decoder latent (b,t,N,cams,D) and is reassembled to full N
    # (immune to kpt_chunk), so on chunk 2+ it coexists on-GPU with the current window's latent
    # -> ~2x peak. For dense point sets (tens of thousands of points) that alone can OOM, so the
    # carry is OFF by default; cross-chunk continuity is still provided by query re-anchoring
    # (current_queries below). With carry ON, a windowed model can also feed clip_len >
    # model.n_frames so the internal windowing + carry engage within each chunk.
    init_latent = None
    clip_len = clip_len if clip_len is not None else model.n_frames

    # Cross-chunk crop-robustness state: per-point velocity (world/pixel per frame) and
    # per-point predicted visibility at the previous chunk's last frame.
    prev_vel = None                                  # (1, N, R)
    prev_vis = None                                  # (1, N)

    # Occlusion carried across chunks (occlusion_embedding). First chunk (and each point's
    # SEED chunk) uses the GT occlusion at the query frame ({0,1,-1}) when available, else
    # unknown (-1); already-active points use the model's PREDICTED per-camera visibility
    # ({0,1}), which propagates window-to-window even when no GT vis exists. None -> all-unknown.
    occ_gt_dev = occlusion_gt.to(device=device).unsqueeze(0) if occlusion_gt is not None else None  # (1,N,cams)
    occ_pred = None                                  # predicted occ carried from previous chunk

    with torch.no_grad():
        pbar = tqdm(total=end_frame - start_frame, desc='Tracking', unit='frames')
        while current_frame < end_frame:
            remaining = end_frame - current_frame
            current_clip_len = clip_len
            win_end = current_frame + current_clip_len

            # Per-point role this chunk (query-first, forward-only seeding):
            before = qt_abs < current_frame                          # active (from earlier)
            seeded = (qt_abs >= current_frame) & (qt_abs < win_end)  # query frame lands here
            appeared = before | seeded                               # produced this chunk

            # Anchor coord per point: active -> carried prediction, else true query coord.
            coords_chunk = torch.where(before.view(1, N, 1), current_queries, original_queries)

            # Model query time (offset within the window): seed offset for points seeded
            # here, the re-anchor overlap offset for active points, 0 otherwise. With every
            # qt_abs == start_frame this is 0 in chunk 0 and n_overlap-1 after (legacy).
            seed_off = (qt_abs - current_frame).clamp(0, current_clip_len - 1)
            active_off = torch.full_like(qt_abs, n_overlap - 1)
            times_chunk = torch.where(before, active_off,
                                      torch.where(seeded, seed_off, torch.zeros_like(qt_abs)))
            times_chunk = times_chunk.to(torch.int32).unsqueeze(0)   # (1, N)

            # --- Crop point set: appeared anchors, minus lost (low-vis) active points, plus
            # a causal motion look-ahead so a fast subject stays in-frame over the window.
            crop_mask = appeared.clone()
            if conf_crop_thresh is not None and prev_vis is not None:
                lost = before & (torch.sigmoid(prev_vis[0]) < conf_crop_thresh)
                if bool((crop_mask & ~lost).any()):        # never empty the crop
                    crop_mask = crop_mask & ~lost
            crop_pts = coords_chunk[:, crop_mask]
            if motion_margin and prev_vel is not None:
                act = before & crop_mask
                if bool(act.any()):
                    look = current_queries[:, act] + prev_vel[:, act] * float(current_clip_len)
                    crop_pts = torch.cat([crop_pts, look], dim=1)
            if crop_pts.shape[1] == 0:
                crop_pts = coords_chunk                     # fallback (e.g. all not-yet)

            camera_group_chunk, crop_boxes = crop_camera_group_to_queries(
                camera_group=camera_group,
                query_coords=crop_pts,
                min_crop_dim=model.image_size,
                padding=20,
                is_2d=is_2d,
            )
            camera_group_chunk = resize_camera_group(
                camera_group_chunk,
                model.image_size,
            )

            # For 2D, queries are pixel coords in the ORIGINAL frame; the model
            # expects them in the cropped+resized model-input frame. resize_camera_group
            # scales by target/max(width,height), so use the SAME max here (crops are
            # non-square after the per-axis cap; width-only would mis-scale portrait crops).
            if is_2d:
                crop_low = crop_boxes[0][:2].to(device=device, dtype=torch.float32)
                crop_wh = (crop_boxes[0][2:] - crop_boxes[0][:2]).to(torch.float32)
                crop_side = torch.max(crop_wh)
                model_scale = float(model.image_size) / float(crop_side)
                queries_model = (coords_chunk - crop_low) * model_scale
            else:
                queries_model = coords_chunk

            target_sizes = [
                tuple(cam['size'].tolist())
                for cam in camera_group_chunk
            ]

            views, actual_clip_len = load_multiview_clip(
                readers,
                current_frame,
                current_clip_len,
                crop_boxes=crop_boxes,
                target_sizes=target_sizes,
            )

            if actual_clip_len == 0:
                break

            for i in range(len(views)):
                if views[i].shape[0] != actual_clip_len:
                    raise ValueError('All videos must provide the same number of frames')

            # Model requires exactly model.n_frames; pad if we got fewer
            if actual_clip_len < model.n_frames:
                pad_len = model.n_frames - actual_clip_len
                for i in range(len(views)):
                    last_frame = views[i][-1:]  # (1, H, W, 3)
                    padding = last_frame.expand(pad_len, -1, -1, -1)
                    views[i] = torch.cat([views[i], padding], dim=0)

            # Only keep predictions for frames we actually need
            keep_len = min(actual_clip_len, remaining)

            crop_history.append({
                'start_frame': current_frame,
                'n_frames': actual_clip_len,
                'crop_boxes': crop_boxes,
            })

            views = [v.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0 for v in views]

            # Occlusion fed to the model this chunk. Active points carry the previous chunk's
            # PREDICTED per-camera occlusion so the visibility state propagates across windows;
            # points whose query frame lands in THIS chunk are (re)seeded with GT occlusion when
            # available, else marked unknown (-1). occ_pred is only ever set when the model uses
            # the occlusion term (see below), so a model without it always takes the first
            # branch and this reduces to the legacy GT-or-None behavior.
            if is_first_chunk or occ_pred is None:
                occ_chunk = occ_gt_dev.clone() if occ_gt_dev is not None else None
            else:
                occ_chunk = occ_pred.clone()
                if bool(seeded.any()):
                    occ_chunk[:, seeded] = (occ_gt_dev[:, seeded] if occ_gt_dev is not None
                                            else -1)

            outputs = model(
                views=views,
                coords=queries_model,
                query_times=times_chunk,
                camera_group=camera_group_chunk,
                init_latent=init_latent,
                kpt_chunk=max_kpts,
                occlusion=occ_chunk,
            )

            if is_2d:
                # 2D predictions live in model-input pixel space; map back to the
                # original full-image frame so outputs and the recurrence stay there.
                coords_pred = outputs['2d_pred'][0]  # single cam: (b, t, n, 2)
                coords_pred = crop_low + coords_pred / model_scale
                coords_pred = coords_pred[:, :keep_len]
            else:
                coords_pred = outputs[pred_key_3d][:, :keep_len]
            vis_pred = outputs['vis_pred'][:, :keep_len]
            conf_pred = outputs['conf_pred'][:, :keep_len]
            # Per-camera visibility logits (cams,b,t,n); time axis is dim 2, not dim 1.
            vis_pred_2d = (outputs['vis_pred_2d'][:, :, :keep_len]
                           if 'vis_pred_2d' in outputs else None)

            if coords_pred.shape[1] == 0:
                break

            # Drop the overlap frames already emitted by the previous chunk (temporal
            # dedup); the first chunk keeps everything from its start.
            discard = 0 if is_first_chunk else min(n_overlap, keep_len)

            coords_pred_all.append(coords_pred[:, discard:].cpu())
            vis_pred_all.append(vis_pred[:, discard:].cpu())
            conf_pred_all.append(conf_pred[:, discard:].cpu())
            if vis_pred_2d is not None:
                vis_pred_2d_all.append(vis_pred_2d[:, :, discard:].cpu())
            frame_numbers_all.append(torch.arange(current_frame + discard, current_frame + keep_len, dtype=torch.int64))

            # --- Carry state to the next chunk.
            # Re-anchor appeared points on this chunk's last prediction; not-yet points keep
            # their true query coord for a later chunk to seed.
            appeared_by_end = (qt_abs < current_frame + keep_len).view(1, N, 1)
            current_queries = torch.where(appeared_by_end, coords_pred[:, -1], original_queries)

            # Per-point velocity (per frame, from the last two kept frames) for the next
            # chunk's causal motion margin, and predicted visibility for low-conf exclusion.
            prev_vel = (coords_pred[:, -1] - coords_pred[:, -2]) if keep_len >= 2 else None
            prev_vis = vis_pred[:, -1, :, 0] if vis_pred.dim() == 4 else vis_pred[:, -1]

            # Predicted per-camera occlusion for the re-anchored queries of the NEXT chunk:
            # threshold the model's vis_pred_2d at the last kept frame (same frame the
            # re-anchor query comes from). visible(sigmoid>=0.5)->1, occluded->0 (no -1 from
            # a prediction). Only fed forward when the model uses the occlusion term.
            if getattr(model, 'occlusion_embedding', False) and 'vis_pred_2d' in outputs:
                vp2d = outputs['vis_pred_2d']                    # (cams, b, t, n) logits
                vp2d_last = vp2d[:, :, keep_len - 1]             # (cams, b, n)
                occ_p = (torch.sigmoid(vp2d_last) >= 0.5).to(torch.int64)
                occ_pred = occ_p.permute(1, 2, 0).contiguous()  # (cams,b,n) -> (b,n,cams)
            else:
                occ_pred = None

            # Thread the decoder latent into the next chunk so the carry spans the whole video
            # (not just within a chunk) -- only when carry_latent. Left None otherwise (the
            # default) to avoid the full-N latent's ~2x peak on dense point sets; None also for
            # non-windowed models / single-pass.
            init_latent = outputs.get('final_latent', None) if carry_latent else None

            current_frame += keep_len - n_overlap
            pbar.update(keep_len - discard)
            is_first_chunk = False

            # Release this chunk's GPU tensors before the next window builds its own full-N
            # latent + grid. Chiefly the ~full-N 'final_latent' held in `outputs`: with carry
            # off it is dead weight, and holding it across the next model() call would keep two
            # full-N latents live and double peak memory (the OOM on dense point sets). When
            # carrying, init_latent already holds the latent, so dropping the dict is safe.
            del coords_pred, vis_pred, conf_pred, outputs
            if vis_pred_2d is not None:
                del vis_pred_2d

            # If we've reached or passed the end, stop
            if current_frame + n_overlap >= end_frame:
                break

        pbar.close()

    del readers

    if len(coords_pred_all) == 0:
        return {
            'coords_pred': torch.empty((0, 0, 0, R)),
            'vis_pred': torch.empty((0, 0, 0, 1)),
            'conf_pred': torch.empty((0, 0, 0, 1)),
            'vis_pred_2d': torch.empty((0, 0, 0, 0)),
            'frame_numbers': torch.empty((0,), dtype=torch.int64),
            'crop_history': crop_history,
        }

    result = {
        'coords_pred': torch.cat(coords_pred_all, dim=1),
        'vis_pred': torch.cat(vis_pred_all, dim=1),
        'conf_pred': torch.cat(conf_pred_all, dim=1),
        'frame_numbers': torch.cat(frame_numbers_all, dim=0),
        'crop_history': crop_history,
    }
    if len(vis_pred_2d_all) > 0:
        # (cams, b, t, n) -> concat over the time axis (dim 2)
        result['vis_pred_2d'] = torch.cat(vis_pred_2d_all, dim=2)
    return result


def load_tracker_encoder_checkpoint(checkpoint_path, model_kwargs, device=None,
                                    config_path=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackerEncoder(**model_kwargs).to(device)

    # When a config is available, route through load_checkpoint so schedule-free runs get the
    # averaged EVAL weights (eval_weights defaults to 'auto' -> swaps since optimizer is None).
    if config_path is not None:
        load_checkpoint(config_path, checkpoint_path, model=model, device=device)
        model.eval()
        return model

    # No config: fall back to a raw model_state load (uses raw training weights for schedule-free).
    print('  [warn] no config_path given; loading raw model_state '
          '(schedule-free averaged eval weights will NOT be applied)')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def sort_by_camera_name(paths):
    """Sort paths using the same camera name ordering as load_camera_group_from_metadata."""
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if all(n.isdigit() for n in names):
        return [p for _, p in sorted(zip(names, paths), key=lambda x: int(x[0]))]
    else:
        return [p for _, p in sorted(zip(names, paths))]


def resolve_video_paths(video_paths):
    """If a single directory is given that contains subdirectories (one per camera),
    expand it into a list of per-camera image folder paths, matching the
    PosetailDataset convention of img_path/cam_name/frame.png."""
    if len(video_paths) == 1 and os.path.isdir(video_paths[0]):
        root = video_paths[0]
        subdirs = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
        if all(d.isdigit() for d in subdirs):
            subdirs = sorted(subdirs, key=int)
        else:
            subdirs = sorted(subdirs)
        if len(subdirs) > 0:
            return [os.path.join(root, d) for d in subdirs]
    return video_paths


def compute_query_first(coords, vis_gt, start_frame, n_frames):
    """mvtracker/training 'query_first': anchor each point at its FIRST valid frame.

    A frame is valid for a point when its coord is finite AND (if vis available) the point
    is visible in >=1 camera. Returns per-point:
      query_time (N,) int  -- index within [start_frame, start_frame+n_frames) of the first
                              valid frame (0 for points that are never valid; they are dropped
                              via `valid`),
      query_coord (N, R)   -- the coordinate at that frame,
      valid (N,) bool      -- point has at least one valid frame in the window.

    coords: (T_full, N, R) for one subject. vis_gt: (T_full, N, cams) or None.
    """
    T_full = coords.shape[0]
    hi = T_full if n_frames is None else min(T_full, start_frame + n_frames)
    sl = slice(start_frame, hi)
    c = coords[sl]                                       # (T, N, R)
    good = np.all(np.isfinite(c), axis=-1)              # (T, N)
    if vis_gt is not None:
        good = good & vis_gt[sl].any(axis=-1)
    valid = good.any(axis=0)                            # (N,)
    query_time = np.argmax(good, axis=0).astype(np.int32)   # first True frame (0 if none)
    query_coord = c[query_time, np.arange(c.shape[1])]      # (N, R)
    return query_time, query_coord, valid


def occlusion_at_query(vis_used_s, query_time_valid, valid_mask, start_frame):
    """Per-camera GT occlusion at each valid point's query frame, for the
    occlusion_embedding query term.

    vis_used_s: (T_full, n_kpts, cams) GT visibility for one subject (NaN=unknown), on
        the cameras ACTUALLY USED (already camera-subsampled), or None.
    query_time_valid: (N_valid,) window-relative query frame per VALID point (0 ==
        start_frame), in the same order as valid_mask's selected keypoints (this is what
        load_trial stores in per_subject_query_times).
    valid_mask: (n_kpts,) bool selecting the tracked points (its nonzero indices give the
        original keypoint columns, aligned with query_time_valid).
    start_frame: absolute offset added to the window-relative query time.

    Returns an int64 tensor (N_valid, cams) with values {0=occluded, 1=visible,
    -1=unknown}, or None when no GT visibility is available.
    """
    if vis_used_s is None:
        return None
    kpt_idx = np.nonzero(valid_mask)[0]                            # (N_valid,) original columns
    frames = np.clip(start_frame + np.asarray(query_time_valid),
                     0, vis_used_s.shape[0] - 1)                   # (N_valid,) absolute frames
    v = vis_used_s[frames, kpt_idx]                               # (N_valid, cams), NaN-bearing
    occ = np.where(np.isnan(v), -1.0, (v > 0.5).astype(np.float32))  # NaN->-1, else 0/1
    return torch.as_tensor(occ, dtype=torch.int64)


def load_trial(trial_path, start_frame=0, n_frames=None, query_first=True):
    """Load metadata, video/image paths, and query points from a trial directory,
    following the PosetailDataset convention.

    Detects 2D vs 3D from which pose file is present: pose3d.npz -> 3D (multi-view,
    metadata.yaml required), pose2d.npz -> 2D (single camera, metadata optional).
    Query points have last dim 3 (3D world coords) or 2 (2D pixel coords) accordingly.

    query_first (3D only): anchor each point at its first valid+visible frame (mvtracker /
    training convention) instead of all points at start_frame. Returns per-point query_times.

    Returns a dict with keys: 'mode', 'metadata_path', 'cam_names', 'video_paths',
    'query_points', 'per_subject_queries', 'coords', 'vis_gt', 'valid_flat',
    'per_subject_valid_masks', 'query_times_flat', 'per_subject_query_times'.
    """

    pose3d_path = os.path.join(trial_path, 'pose3d.npz')
    pose2d_path = os.path.join(trial_path, 'pose2d.npz')
    if os.path.exists(pose3d_path):
        mode = '3d'
        pose_path = pose3d_path
    elif os.path.exists(pose2d_path):
        mode = '2d'
        pose_path = pose2d_path
    else:
        raise FileNotFoundError(f'No pose3d.npz or pose2d.npz found in {trial_path}')

    # metadata.yaml required for 3D; optional for 2D (uncalibrated single camera)
    metadata_path = os.path.join(trial_path, 'metadata.yaml')
    if mode == '3d' and not os.path.exists(metadata_path):
        raise FileNotFoundError(f'metadata.yaml not found in {trial_path}')
    if not os.path.exists(metadata_path):
        metadata_path = None

    # determine image or video paths
    img_path = os.path.join(trial_path, 'img')
    vid_path = os.path.join(trial_path, 'vid')

    # Camera ordering: from metadata when available (3D, or calibrated 2D),
    # otherwise from the img/ subdirectories (uncalibrated 2D -> single camera).
    if metadata_path is not None:
        with open(metadata_path, 'r') as f:
            cam_metadata = yaml.safe_load(f)
        cam_names = list(cam_metadata['intrinsic_matrices'].keys())
    elif os.path.exists(img_path):
        cam_names = [d for d in os.listdir(img_path)
                     if os.path.isdir(os.path.join(img_path, d))]
    else:
        cam_names = [os.path.splitext(f)[0] for f in os.listdir(vid_path)]
    if all(n.isdigit() for n in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)
    if mode == '2d':
        cam_names = cam_names[:1]  # 2D trials are always single-camera

    if os.path.exists(img_path) and len(os.listdir(img_path)) > 0:
        # Sort image subdirectories using camera names from metadata
        # to ensure alignment with the camera group
        video_paths = [os.path.join(img_path, cam_name) for cam_name in cam_names]
        for vp in video_paths:
            if not os.path.exists(vp):
                raise FileNotFoundError(f'Expected image folder {vp} not found')
    elif os.path.exists(vid_path):
        video_paths = [os.path.join(vid_path, f'{cam_name}.mp4') for cam_name in cam_names]
        for vp in video_paths:
            if not os.path.exists(vp):
                raise FileNotFoundError(f'Expected video file {vp} not found')
    else:
        raise FileNotFoundError(f'Neither img/ nor vid/ folder found in {trial_path}')


    data = np.load(pose_path)
    coords = data['pose']  # (subjects, time, n_kpts, R), R=3 for 3D / 2 for 2D
    vis_gt = data['vis'] if 'vis' in data else None  # 2D trials never have vis

    R = coords.shape[-1]
    n_subjects = coords.shape[0]
    n_kpts = coords.shape[2]

    # query_first is 3D-only (needs vis to define first-visible frame); 2D falls back.
    use_query_first = query_first and mode == '3d'

    per_subject_queries = []
    per_subject_valid_masks = []
    per_subject_query_times = []
    for s in range(n_subjects):
        if use_query_first:
            qt_s, qc_s, valid = compute_query_first(
                coords[s], vis_gt[s] if vis_gt is not None else None, start_frame, n_frames)
        else:
            qc_s = coords[:, start_frame, :, :][s]              # (n_kpts, R)
            valid = np.all(np.isfinite(qc_s), axis=1)
            qt_s = np.zeros(qc_s.shape[0], dtype=np.int32)
        per_subject_valid_masks.append(valid)
        per_subject_queries.append(torch.as_tensor(qc_s[valid], dtype=torch.float32))
        per_subject_query_times.append(torch.as_tensor(qt_s[valid], dtype=torch.int32))

    # flat (all subjects concatenated, s-major then kpt) — matches GT-extraction ordering
    query_points = torch.cat(per_subject_queries, dim=0) if per_subject_queries \
        else torch.empty((0, R))
    query_times_flat = torch.cat(per_subject_query_times, dim=0) if per_subject_query_times \
        else torch.empty((0,), dtype=torch.int32)
    valid_flat = np.concatenate(per_subject_valid_masks) if per_subject_valid_masks \
        else np.zeros(0, dtype=bool)

    if query_points.shape[0] == 0:
        raise ValueError(f'No valid query points in trial {trial_path}')

    return {
        'mode': mode,
        'metadata_path': metadata_path,
        'cam_names': cam_names,
        'video_paths': video_paths,
        'query_points': query_points,
        'per_subject_queries': per_subject_queries,
        'coords': coords,
        'vis_gt': vis_gt,
        'valid_flat': valid_flat,
        'per_subject_valid_masks': per_subject_valid_masks,
        'query_times_flat': query_times_flat,
        'per_subject_query_times': per_subject_query_times,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base-folder', type=str, required=True,
                        help='Wandb run folder containing files/config.toml and files/checkpoints/')
    parser.add_argument('--trial-path', type=str, required=True,
                        help='Path to a trial directory containing metadata.yaml, '
                             'pose3d.npz, and an img/ or vid/ folder')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=128)
    parser.add_argument('--n-overlap', type=int, default=2)
    parser.add_argument('--n-views', type=int, default=None, help='Evaluate on a random subset of the cameras')
    parser.add_argument('--view-seed', type=int, default=None, help='Random seed for subsampling cameras')
    parser.add_argument('--max-kpts', type=int, default=None, help='Max keypoints per model forward pass.')
    parser.add_argument('--per-subject', action='store_true', default=False,
                        help='Track each subject independently instead of concatenating all keypoints')
    parser.add_argument('--no-query-first', dest='query_first', action='store_false', default=True,
                        help='disable query-first (default ON): with this flag all points are '
                             'anchored at start_frame instead of their first valid+visible frame')
    parser.add_argument('--no-motion-margin', dest='motion_margin', action='store_false', default=True,
                        help='disable the causal motion-margin crop expansion (default ON); with '
                             'this flag + --no-query-first the path is legacy-identical')
    parser.add_argument('--carry-latent', action='store_true', default=False,
                        help='thread the decoder latent across chunks (default OFF): the carried '
                             'full-N latent (b,t,N,cams,D) is immune to --max-kpts chunking and '
                             'roughly doubles peak memory on dense point sets, so it is off by '
                             'default; query re-anchoring still provides cross-chunk continuity')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Optional checkpoint step number; if omitted, use latest checkpoint')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--pred-key-3d', type=str, default='coords_pred',
                        help="Which 3D model output to use as the prediction "
                             "(e.g. 'coords_pred' or '3d_pred_triangulate')")
    parser.add_argument('--clip-len', type=int, default=None,
                        help='Frames fed to the model per forward. Defaults to '
                             'model.n_frames (= stride_length). For a windowed model set '
                             'this > stride_length (e.g. 16) so internal windowing + the '
                             'latent carry engage per chunk; the latent is also threaded '
                             'across chunks.')
    parser.add_argument('--outpath', type=str, default=None,
                        help='Optional output .npz path')

    return parser.parse_args()


def run_inference(
    model,
    config_path,
    checkpoint_path,
    trial_path,
    start_frame=0,
    n_frames=128,
    n_overlap=2,
    n_views=None,
    view_seed=None,
    max_kpts=None,
    per_subject=False,
    device=None,
    outpath=None,
    pred_key_3d='coords_pred',
    clip_len=None,
    query_first=True,
    motion_margin=True,
    carry_latent=False,
):
    """Run inference on one trial with an already-loaded model.

    Factored out of main() so a caller (e.g. a batch driver) can load the model
    once and reuse it across many trials. If outpath is given the outputs are
    saved as .npz; the outputs dict is always returned.

    query_first (3D only): seed each point at its first valid+visible frame (mvtracker /
    training convention) within the windowed, per-chunk re-cropping tracker, instead of all
    points at start_frame. motion_margin: expand each chunk's crop by the previous chunk's
    per-point velocity so fast subjects stay in-frame (disable for the legacy-identical path).
    """
    if device is None:
        device = next(model.parameters()).device

    trial = load_trial(trial_path, start_frame=start_frame, n_frames=n_frames,
                       query_first=query_first)
    mode                    = trial['mode']
    metadata_path           = trial['metadata_path']
    cam_names               = trial['cam_names']
    video_paths             = trial['video_paths']
    query_points_3d         = trial['query_points']
    per_subject_queries     = trial['per_subject_queries']
    coords_gt               = trial['coords']
    vis_gt_raw              = trial['vis_gt']
    valid_flat              = trial['valid_flat']
    per_subject_valid_masks = trial['per_subject_valid_masks']
    query_times_flat        = trial['query_times_flat']
    per_subject_query_times = trial['per_subject_query_times']
    use_query_first = query_first and mode == '3d'

    R = coords_gt.shape[-1]

    if mode == '2d':
        camera_group = load_camera_group_2d(video_paths, metadata_path, device='cpu')
    else:
        camera_group = load_camera_group_from_metadata(metadata_path, device='cpu')

    if len(video_paths) != len(camera_group):
        raise ValueError(
            f'Number of video paths ({len(video_paths)}) does not match '
            f'number of cameras ({len(camera_group)}) in metadata'
        )
    
    # subsample cameras
    n_cams_total = len(camera_group)
    cam_indices_used = list(range(n_cams_total))

    if n_views is not None and n_views < n_cams_total:
        rng = np.random.default_rng(view_seed)
        cam_indices_used = sorted(rng.choice(n_cams_total, n_views, replace=False).tolist())
        print(f'Subsampling {n_views}/{n_cams_total} cameras: indices {cam_indices_used}')
        camera_group = [camera_group[i] for i in cam_indices_used]
        video_paths  = [video_paths[i] for i in cam_indices_used]
    elif n_views is not None:
        print(f'--n-views={n_views} >= available cameras ({n_cams_total}); using all cameras')

    # Record the cameras actually used, after any subsampling above.
    cam_names_used = [cam_names[i] for i in cam_indices_used]

    # GT visibility restricted to the cameras actually used, for the occlusion query term.
    # vis_gt_raw is (subjects, time, kpts, n_cams_total); index the camera axis so the
    # occlusion cams dim matches the model input. None when the trial has no visibility.
    vis_gt_used = vis_gt_raw[..., cam_indices_used] if vis_gt_raw is not None else None

    def _subject_occlusion(s):
        vis_used_s = vis_gt_used[s] if vis_gt_used is not None else None
        return occlusion_at_query(vis_used_s, per_subject_query_times[s],
                                  per_subject_valid_masks[s], start_frame)

    if per_subject:
        all_subject_outputs = []
        for subj_idx, subj_queries in enumerate(per_subject_queries):
            if subj_queries.shape[0] == 0:
                print(f'Skipping subject {subj_idx}: no valid query points')
                continue
            print(f'Tracking subject {subj_idx} ({subj_queries.shape[0]} keypoints)')
            subj_occlusion = _subject_occlusion(subj_idx)
            subj_out = run_tracker_encoder_on_videos(
                model=model,
                video_paths=video_paths,
                camera_group=camera_group,
                query_points_3d=subj_queries,
                start_frame=start_frame,
                n_frames=n_frames,
                n_overlap=n_overlap,
                max_kpts=max_kpts,
                device=device,
                pred_key_3d=pred_key_3d,
                clip_len=clip_len,
                occlusion_gt=subj_occlusion,
                query_times=per_subject_query_times[subj_idx] if use_query_first else None,
                motion_margin=motion_margin,
                carry_latent=carry_latent,
            )
            subj_out['subject_idx'] = subj_idx
            all_subject_outputs.append(subj_out)

        # Combine per-subject results: stack along a new subject dimension
        if len(all_subject_outputs) == 0:
            outputs = {
                'coords_pred': torch.empty((0, 0, 0, R)),
                'vis_pred': torch.empty((0, 0, 0, 1)),
                'conf_pred': torch.empty((0, 0, 0, 1)),
                'frame_numbers': torch.empty((0,), dtype=torch.int64),
                'crop_history': [],
            }
        else:
            # Use frame_numbers from the first subject (all should be identical
            # since they share start_frame / n_frames / n_overlap)
            outputs = {
                'frame_numbers': all_subject_outputs[0]['frame_numbers'],
                'crop_history': [],
            }
            # Stack per-subject predictions: each is (1, T, K, D) -> collect into lists
            coords_list = []
            vis_list = []
            conf_list = []
            vp2d_list = []          # per-camera logits, each (cams, 1, T, K_s)
            for so in all_subject_outputs:
                coords_list.append(so['coords_pred'])
                vis_list.append(so['vis_pred'])
                conf_list.append(so['conf_pred'])
                if 'vis_pred_2d' in so:
                    vp2d_list.append(so['vis_pred_2d'])
                outputs['crop_history'].extend(so['crop_history'])

            # Concatenate along the keypoint dimension (dim=2) with batch dim=0
            # Result shape: (1, T, total_kpts, D) but grouped by subject
            outputs['coords_pred'] = torch.cat(coords_list, dim=2)
            outputs['vis_pred'] = torch.cat(vis_list, dim=2)
            outputs['conf_pred'] = torch.cat(conf_list, dim=2)
            # vis_pred_2d is (cams, b, t, n): keypoints are dim 3. Only when every
            # subject produced it (same camera/time layout across subjects).
            if len(vp2d_list) == len(all_subject_outputs) and len(vp2d_list) > 0:
                outputs['vis_pred_2d'] = torch.cat(vp2d_list, dim=3)

            # Also store per-subject slicing info
            subject_kpt_counts = [so['coords_pred'].shape[2] for so in all_subject_outputs]
            subject_indices = [so['subject_idx'] for so in all_subject_outputs]
            outputs['subject_kpt_counts'] = np.array(subject_kpt_counts, dtype=np.int32)
            outputs['subject_indices'] = np.array(subject_indices, dtype=np.int32)
    else:
        # Flat (all subjects concatenated, s-major): build occlusion in the same order as
        # query_points (torch.cat of per_subject_queries). None if the trial has no vis.
        if vis_gt_used is not None:
            occ_parts = [_subject_occlusion(s) for s in range(len(per_subject_queries))]
            occlusion_flat = (torch.cat(occ_parts, dim=0)
                              if len(occ_parts) > 0 else None)
        else:
            occlusion_flat = None

        outputs = run_tracker_encoder_on_videos(
            model=model,
            video_paths=video_paths,
            camera_group=camera_group,
            query_points_3d=query_points_3d,
            start_frame=start_frame,
            n_frames=n_frames,
            n_overlap=n_overlap,
            max_kpts=max_kpts,
            device=device,
            pred_key_3d=pred_key_3d,
            clip_len=clip_len,
            occlusion_gt=occlusion_flat,
            query_times=query_times_flat if use_query_first else None,
            motion_margin=motion_margin,
            carry_latent=carry_latent,
        )

    # --- Ground-truth extraction ---
    frame_nums = (outputs['frame_numbers'].numpy()
                  if isinstance(outputs['frame_numbers'], torch.Tensor)
                  else np.asarray(outputs['frame_numbers']))
    n_time_gt = coords_gt.shape[1]
    frame_nums_gt = np.clip(frame_nums, 0, n_time_gt - 1)

    if per_subject and 'subject_indices' in outputs:
        coords_true_parts, vis_true_parts, vis_cam_parts = [], [], []
        for s_idx in outputs['subject_indices']:
            vmask = per_subject_valid_masks[s_idx]
            subj_coords = coords_gt[s_idx, frame_nums_gt]          # (T, n_kpts, 3)
            subj_coords = subj_coords[:, vmask, :]                 # (T, K_s, 3)
            coords_true_parts.append(subj_coords)

            if vis_gt_raw is not None:
                subj_vis = vis_gt_raw[s_idx, frame_nums_gt]        # (T, n_kpts, n_cams)
                subj_vis = subj_vis[:, vmask, :].any(axis=-1, keepdims=True)  # (T, K_s, 1)
            else:
                subj_vis = np.all(np.isfinite(subj_coords), axis=-1, keepdims=True)
            vis_true_parts.append(subj_vis)

            # Per-camera GT visibility on the USED cameras (NaN=unknown preserved), for the
            # per-camera occlusion metric. Aligned to vis_pred_2d's camera axis.
            if vis_gt_used is not None:
                vis_cam_parts.append(vis_gt_used[s_idx, frame_nums_gt][:, vmask, :])  # (T, K_s, cams)

        coords_true = np.concatenate(coords_true_parts, axis=1)[np.newaxis]  # (1, T, K, 3)
        vis_true    = np.concatenate(vis_true_parts,    axis=1)[np.newaxis]  # (1, T, K, 1)
        vis_true_cams = (np.concatenate(vis_cam_parts, axis=1)[np.newaxis]   # (1, T, K, cams)
                         if vis_cam_parts else None)

    else:
        n_subj, n_time_full, n_kpts_full, _ = coords_gt.shape

        # coords_gt is (n_subj, n_time, n_kpts, R)
        # transpose to (n_subj, n_kpts, n_time, R) before flattening subjects+kpts
        coords_flat_all = coords_gt.transpose(0, 2, 1, 3)                    # (n_subj, n_kpts, n_time, R)
        coords_flat_all = coords_flat_all.reshape(n_subj * n_kpts_full, n_time_full, R)  # (S*K, n_time, R)
        coords_flat_all = coords_flat_all[valid_flat]                         # (K_valid, n_time, R)
        coords_true = coords_flat_all[:, frame_nums_gt, :].transpose(1, 0, 2)[np.newaxis]  # (1, T, K_valid, R)

        if vis_gt_raw is not None:
            # vis_gt_raw is (n_subj, n_time, n_kpts, n_cams) — aggregate across cameras
            vis_agg = vis_gt_raw.any(axis=-1)                                 # (n_subj, n_time, n_kpts)
            vis_flat_all = vis_agg.transpose(0, 2, 1)                        # (n_subj, n_kpts, n_time)
            vis_flat_all = vis_flat_all.reshape(n_subj * n_kpts_full, n_time_full)  # (S*K, n_time)
            vis_flat_all = vis_flat_all[valid_flat]                           # (K_valid, n_time)
            vis_true = vis_flat_all[:, frame_nums_gt].T[:, :, np.newaxis][np.newaxis]  # (1, T, K_valid, 1)
        else:
            vis_true = np.all(np.isfinite(coords_true), axis=-1, keepdims=True)

        # Per-camera GT (NaN=unknown preserved) on the USED cameras, mirroring the coords_true
        # reshape but keeping the camera axis: (n_subj,n_time,n_kpts,cams) -> (1,T,K_valid,cams).
        if vis_gt_used is not None:
            n_cams_used = vis_gt_used.shape[-1]
            vc = vis_gt_used.transpose(0, 2, 1, 3)                            # (n_subj, n_kpts, n_time, cams)
            vc = vc.reshape(n_subj * n_kpts_full, n_time_full, n_cams_used)   # (S*K, n_time, cams)
            vc = vc[valid_flat]                                              # (K_valid, n_time, cams)
            vis_true_cams = vc[:, frame_nums_gt, :].transpose(1, 0, 2)[np.newaxis]  # (1, T, K_valid, cams)
        else:
            vis_true_cams = None

    # per-point query times, in the same keypoint order as coords_pred/coords_true
    if per_subject and 'subject_indices' in outputs:
        qt_parts = [np.asarray(per_subject_query_times[s]) for s in outputs['subject_indices']]
        query_times_out = np.concatenate(qt_parts) if qt_parts else np.zeros(0, dtype=np.int32)
    else:
        query_times_out = np.asarray(query_times_flat)
    outputs['query_times'] = query_times_out.astype(np.int32)   # (K,)
    outputs['query_first'] = bool(use_query_first)

    outputs['coords_true'] = coords_true   # (1, T, K, 3)
    outputs['vis_true'] = vis_true      # (1, T, K, 1)
    # Per-camera GT visibility for the per-camera occlusion metric; paired with the model's
    # vis_pred_2d (already in outputs from the tracker). Both omitted when GT vis is absent.
    if vis_true_cams is not None:
        outputs['vis_true_cams'] = vis_true_cams   # (1, T, K, cams_used), NaN=unknown
    outputs['video_paths'] = np.array(video_paths, dtype=str)
    outputs['mode'] = mode
    outputs['metadata_path'] = metadata_path if metadata_path is not None else ''
    outputs['trial_path'] = trial_path
    outputs['config_path'] = config_path
    outputs['checkpoint_path'] = checkpoint_path
    outputs['start_frame'] = start_frame
    outputs['n_frames_requested'] = n_frames
    outputs['n_overlap'] = n_overlap
    outputs['per_subject'] = per_subject
    outputs['n_cams_total'] = n_cams_total
    outputs['cam_names_used'] = np.array(cam_names_used, dtype=str)

    if isinstance(outputs['coords_pred'], torch.Tensor) and outputs['coords_pred'].ndim >= 2:
        outputs['n_frames_returned'] = outputs['coords_pred'].shape[1]
    else:
        outputs['n_frames_returned'] = 0

    if outpath is not None:
        save_dict = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.cpu().numpy()
            else:
                save_dict[k] = v

        crop_history_serializable = [
            {
                'start_frame': int(ch['start_frame']),
                'n_frames': int(ch['n_frames']),
                'crop_boxes': [cb.cpu().numpy().tolist() for cb in ch['crop_boxes']],
            }
            for ch in outputs['crop_history']
        ]
        save_dict['crop_history'] = json.dumps(crop_history_serializable)

        outdir = os.path.dirname(outpath)
        if outdir != '':
            os.makedirs(outdir, exist_ok=True)
        np.savez(outpath, **save_dict)
        print(f'Saved outputs to {outpath}')
    else:
        print('Inference completed.')
        print(f'checkpoint_path: {checkpoint_path}')
        print(f'config_path: {config_path}')
        print(f'n_frames_returned: {outputs["coords_pred"].shape[1] if outputs["coords_pred"].ndim >= 2 else 0}')

    return outputs


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device is not None else None

    model, config, config_path, checkpoint_path = load_model_from_base_folder(
        args.base_folder,
        checkpoint=args.checkpoint,
        device=device,
    )

    run_inference(
        model=model,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        trial_path=args.trial_path,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
        n_overlap=args.n_overlap,
        n_views=args.n_views,
        view_seed=args.view_seed,
        max_kpts=args.max_kpts,
        per_subject=args.per_subject,
        device=device,
        outpath=args.outpath,
        pred_key_3d=args.pred_key_3d,
        clip_len=args.clip_len,
        query_first=args.query_first,
        motion_margin=args.motion_margin,
        carry_latent=args.carry_latent,
    )


if __name__ == '__main__':
    main()
