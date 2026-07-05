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
            frames = frames[:, y1:y2, x1:x2, :]

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

            min_dim = max(min_crop_dim, current_width, current_height)

            if current_width < min_dim:
                center_x = (low[0] + high[0]) // 2
                low[0] = torch.clamp(center_x - min_dim // 2, 0, size[0] - min_dim)
                high[0] = torch.clamp(low[0] + min_dim, 0, size[0])
                low[0] = high[0] - min_dim

            if current_height < min_dim:
                center_y = (low[1] + high[1]) // 2
                low[1] = torch.clamp(center_y - min_dim // 2, 0, size[1] - min_dim)
                high[1] = torch.clamp(low[1] + min_dim, 0, size[1])
                low[1] = high[1] - min_dim

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
            cam['offset'] = torch.round(cam['offset'] * scale).to(torch.int32)

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
):

    # Dimensionality of the queries: R==3 -> 3D world coords (multi-view), R==2 ->
    # 2D pixel coords in a single camera frame. The model supports both.
    # pred_key_3d selects which 3D model output to use as the prediction (and the
    # recurrence query) — e.g. 'coords_pred' (default) or '3d_pred_triangulate'.
    R = query_points_3d.shape[-1]
    is_2d = (R == 2)

    # --- keypoint chunking (insert here) ---
    n_kpts = query_points_3d.shape[0] if query_points_3d.ndim == 2 else query_points_3d.shape[1]
    if max_kpts is not None and n_kpts > max_kpts:
        chunk_outputs = []
        for start_k in range(0, n_kpts, max_kpts):
            end_k = min(start_k + max_kpts, n_kpts)
            chunk_queries = (query_points_3d[start_k:end_k]
                             if query_points_3d.ndim == 2
                             else query_points_3d[:, start_k:end_k])
            print(f'  keypoint chunk {start_k}:{end_k} of {n_kpts}')
            chunk_out = run_tracker_encoder_on_videos(
                model=model,
                video_paths=video_paths,
                camera_group=camera_group,
                query_points_3d=chunk_queries,
                start_frame=start_frame,
                n_frames=n_frames,
                n_overlap=n_overlap,
                max_kpts=None,
                device=device,
                pred_key_3d=pred_key_3d,
                clip_len=clip_len,
            )
            chunk_outputs.append(chunk_out)
        return {
            'coords_pred':   torch.cat([o['coords_pred']  for o in chunk_outputs], dim=2),
            'vis_pred':      torch.cat([o['vis_pred']     for o in chunk_outputs], dim=2),
            'conf_pred':     torch.cat([o['conf_pred']    for o in chunk_outputs], dim=2),
            'frame_numbers': chunk_outputs[0]['frame_numbers'],
            'crop_history':  [h for o in chunk_outputs for h in o['crop_history']],
        }

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

    coords_pred_all = []
    vis_pred_all = []
    conf_pred_all = []
    frame_numbers_all = []
    crop_history = []
    query_times = None
    # Latent threaded across chunks (cross-chunk carry). For windowed models, feed more
    # than stride_length frames per forward (clip_len > model.n_frames) so the internal
    # windowing + latent carry actually engage within each chunk.
    init_latent = None
    clip_len = clip_len if clip_len is not None else model.n_frames

    with torch.no_grad():
        pbar = tqdm(total=end_frame - start_frame, desc='Tracking', unit='frames')
        while current_frame < end_frame:
            remaining = end_frame - current_frame
            current_clip_len = clip_len

            camera_group_chunk, crop_boxes = crop_camera_group_to_queries(
                camera_group=camera_group,
                query_coords=current_queries,
                min_crop_dim=model.image_size,
                padding=20,
                is_2d=is_2d,
            )
            camera_group_chunk = resize_camera_group(
                camera_group_chunk,
                model.image_size,
            )

            # For 2D, queries are pixel coords in the ORIGINAL frame; the model
            # expects them in the cropped+resized model-input frame. The crop is
            # square, so a single uniform scale maps between the two spaces.
            if is_2d:
                crop_low = crop_boxes[0][:2].to(device=device, dtype=torch.float32)
                crop_side = (crop_boxes[0][2] - crop_boxes[0][0]).to(torch.float32)
                model_scale = float(model.image_size) / float(crop_side)
                queries_model = (current_queries - crop_low) * model_scale
            else:
                queries_model = current_queries

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

            outputs = model(
                views=views,
                coords=queries_model,
                query_times=query_times,
                camera_group=camera_group_chunk,
                init_latent=init_latent,
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

            if coords_pred.shape[1] == 0:
                break

            if query_times is not None:
                discard = min(n_overlap, keep_len)
            else:
                discard = 0

            coords_pred_all.append(coords_pred[:, discard:].cpu())
            vis_pred_all.append(vis_pred[:, discard:].cpu())
            conf_pred_all.append(conf_pred[:, discard:].cpu())
            frame_numbers_all.append(torch.arange(current_frame + discard, current_frame + keep_len, dtype=torch.int64))

            current_queries = coords_pred[:, -1]

            # Thread the decoder latent into the next chunk so the carry spans the whole
            # video (not just within a chunk). None for non-windowed models / single-pass.
            init_latent = outputs.get('final_latent', None)

            query_times = torch.full(
                (current_queries.shape[0], current_queries.shape[1]),
                n_overlap - 1,
                device=device,
                dtype=torch.int32,
            )

            current_frame += keep_len - n_overlap
            pbar.update(keep_len - discard)

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
            'frame_numbers': torch.empty((0,), dtype=torch.int64),
            'crop_history': crop_history,
        }

    return {
        'coords_pred': torch.cat(coords_pred_all, dim=1),
        'vis_pred': torch.cat(vis_pred_all, dim=1),
        'conf_pred': torch.cat(conf_pred_all, dim=1),
        'frame_numbers': torch.cat(frame_numbers_all, dim=0),
        'crop_history': crop_history,
    }


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


def run_query_first_on_videos(model, video_paths, camera_group, query_points_3d, query_times,
                              start_frame=0, n_frames=128, max_kpts=None, device=None,
                              pred_key_3d='coords_pred'):
    """Single-forward tracking with per-point query times (mvtracker query_first).

    Unlike run_tracker_encoder_on_videos (which anchors all points at the chunk start and
    re-anchors across chunks), this feeds the whole clip in one forward and lets the model's
    internal windowing + latent carry handle each point's introduction at its query frame
    (TrackerEncoder._forward_windows already masks frames before a track's query time). 3D-only.
    """
    R = query_points_3d.shape[-1]
    if R == 2:
        raise ValueError('run_query_first_on_videos is 3D-only')
    n_kpts = query_points_3d.shape[0]

    # Memory: keypoints are chunked INSIDE the model (kpt_chunk) so the V-JEPA scene encoding
    # is computed once per window and reused across chunks (numerically identical; see
    # TrackerEncoder._forward_window). This also keeps a single, consistent crop over all
    # query points -- unlike the old external per-chunk recursion, which cropped each chunk
    # separately. `max_kpts` becomes the internal chunk size.

    if device is None:
        device = next(model.parameters()).device
    model = model.to(device).eval()
    camera_group = camera_group_to_device(camera_group, device)
    readers, reader_lengths = build_video_readers(video_paths)
    max_available = min(reader_lengths)
    end_frame = min(start_frame + n_frames, max_available)
    T = end_frame - start_frame

    q = query_points_3d.to(device=device, dtype=torch.float32).unsqueeze(0)          # (1,N,3)
    qt = query_times.to(device=device, dtype=torch.int32).clamp_(max=max(T - 1, 0)).unsqueeze(0)

    with torch.no_grad():
        camera_group_chunk, crop_boxes = crop_camera_group_to_queries(
            camera_group=camera_group, query_coords=q,
            min_crop_dim=model.image_size, padding=20, is_2d=False)
        camera_group_chunk = resize_camera_group(camera_group_chunk, model.image_size)
        target_sizes = [tuple(cam['size'].tolist()) for cam in camera_group_chunk]

        views, actual = load_multiview_clip(
            readers, start_frame, T, crop_boxes=crop_boxes, target_sizes=target_sizes)
        if actual < model.n_frames:                     # model needs >= n_frames; pad with last
            for i in range(len(views)):
                pad = views[i][-1:].expand(model.n_frames - actual, -1, -1, -1)
                views[i] = torch.cat([views[i], pad], dim=0)
        views = [v.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0 for v in views]

        outputs = model(views=views, coords=q, query_times=qt,
                        camera_group=camera_group_chunk, init_latent=None, kpt_chunk=max_kpts)
        coords_pred = outputs[pred_key_3d][:, :T]
        vis_pred = outputs['vis_pred'][:, :T]
        conf_pred = outputs['conf_pred'][:, :T]

    del readers
    Tret = coords_pred.shape[1]
    return {
        'coords_pred':   coords_pred.cpu(),
        'vis_pred':      vis_pred.cpu(),
        'conf_pred':     conf_pred.cpu(),
        'frame_numbers': torch.arange(start_frame, start_frame + Tret, dtype=torch.int64),
        'crop_history':  [{'start_frame': start_frame, 'n_frames': Tret, 'crop_boxes': crop_boxes}],
    }


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
):
    """Run inference on one trial with an already-loaded model.

    Factored out of main() so a caller (e.g. a batch driver) can load the model
    once and reuse it across many trials. If outpath is given the outputs are
    saved as .npz; the outputs dict is always returned.

    query_first (3D only): anchor each point at its first valid+visible frame (mvtracker /
    training convention), via a single full-clip forward, instead of all points at start_frame.
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
    cam_names_used = [cam_names[i] for i in cam_indices_used]

    if n_views is not None and n_views < n_cams_total:
        rng = np.random.default_rng(view_seed)
        cam_indices_used = sorted(rng.choice(n_cams_total, n_views, replace=False).tolist())
        print(f'Subsampling {n_views}/{n_cams_total} cameras: indices {cam_indices_used}')
        camera_group = [camera_group[i] for i in cam_indices_used]
        video_paths  = [video_paths[i] for i in cam_indices_used]
    elif n_views is not None:
        print(f'--n-views={n_views} >= available cameras ({n_cams_total}); using all cameras')

    if per_subject:
        all_subject_outputs = []
        for subj_idx, subj_queries in enumerate(per_subject_queries):
            if subj_queries.shape[0] == 0:
                print(f'Skipping subject {subj_idx}: no valid query points')
                continue
            print(f'Tracking subject {subj_idx} ({subj_queries.shape[0]} keypoints)')
            if use_query_first:
                subj_out = run_query_first_on_videos(
                    model=model, video_paths=video_paths, camera_group=camera_group,
                    query_points_3d=subj_queries, query_times=per_subject_query_times[subj_idx],
                    start_frame=start_frame, n_frames=n_frames, max_kpts=max_kpts,
                    device=device, pred_key_3d=pred_key_3d)
            else:
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
            for so in all_subject_outputs:
                coords_list.append(so['coords_pred'])
                vis_list.append(so['vis_pred'])
                conf_list.append(so['conf_pred'])
                outputs['crop_history'].extend(so['crop_history'])

            # Concatenate along the keypoint dimension (dim=2) with batch dim=0
            # Result shape: (1, T, total_kpts, D) but grouped by subject
            outputs['coords_pred'] = torch.cat(coords_list, dim=2)
            outputs['vis_pred'] = torch.cat(vis_list, dim=2)
            outputs['conf_pred'] = torch.cat(conf_list, dim=2)

            # Also store per-subject slicing info
            subject_kpt_counts = [so['coords_pred'].shape[2] for so in all_subject_outputs]
            subject_indices = [so['subject_idx'] for so in all_subject_outputs]
            outputs['subject_kpt_counts'] = np.array(subject_kpt_counts, dtype=np.int32)
            outputs['subject_indices'] = np.array(subject_indices, dtype=np.int32)
    elif use_query_first:
        outputs = run_query_first_on_videos(
            model=model, video_paths=video_paths, camera_group=camera_group,
            query_points_3d=query_points_3d, query_times=query_times_flat,
            start_frame=start_frame, n_frames=n_frames, max_kpts=max_kpts,
            device=device, pred_key_3d=pred_key_3d)
    else:
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
        )

    # --- Ground-truth extraction ---
    frame_nums = (outputs['frame_numbers'].numpy()
                  if isinstance(outputs['frame_numbers'], torch.Tensor)
                  else np.asarray(outputs['frame_numbers']))
    n_time_gt = coords_gt.shape[1]
    frame_nums_gt = np.clip(frame_nums, 0, n_time_gt - 1)

    if per_subject and 'subject_indices' in outputs:
        coords_true_parts, vis_true_parts = [], []
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

        coords_true = np.concatenate(coords_true_parts, axis=1)[np.newaxis]  # (1, T, K, 3)
        vis_true    = np.concatenate(vis_true_parts,    axis=1)[np.newaxis]  # (1, T, K, 1)

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
    outputs['n_cams_total'] = len(camera_group)
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
    )


if __name__ == '__main__':
    main()
