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
from train_utils import dict_to_device, load_config, load_checkpoint, format_camera_group, _recompute_ortho_derived


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


def crop_camera_group_to_queries(camera_group, query_coords_3d, min_crop_dim, padding=20):
    p2d = project_points_torch(camera_group, query_coords_3d)
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

        if cam['type'] == 'orthographic':
            P = cam['proj_mat'].clone()
            P[:2, :] = P[:2, :] * scale
            cam['proj_mat'] = P
            _recompute_ortho_derived(cam)
        else:
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

    if cam_type == 'orthographic':
        dlt_dict = cam_metadata['dlt_coefficients']
        heights_dict = cam_metadata['camera_heights']
        widths_dict = cam_metadata['camera_widths']
        cam_names = list(dlt_dict.keys())
        cam_names = sorted(cam_names, key=int) if all(n.isdigit() for n in cam_names) else sorted(cam_names)
        cgroup = [
            {'name': n, 'L': list(dlt_dict[n]),
             'size': [widths_dict[n], heights_dict[n]]}
            for n in cam_names
        ]
        return format_camera_group(cgroup, offset_dict, cam_type, device=device)

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
    device=None,
):
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

    with torch.no_grad():
        pbar = tqdm(total=end_frame - start_frame, desc='Tracking', unit='frames')
        while current_frame < end_frame:
            remaining = end_frame - current_frame
            current_clip_len = model.n_frames

            camera_group_chunk, crop_boxes = crop_camera_group_to_queries(
                camera_group=camera_group,
                query_coords_3d=current_queries,
                min_crop_dim=model.image_size,
                padding=20,
            )
            camera_group_chunk = resize_camera_group(
                camera_group_chunk,
                model.image_size,
            )

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

            # Model requires exactly model.n_frames; pad if we got fewer
            if actual_clip_len < model.n_frames:
                pad_len = model.n_frames - actual_clip_len
                for i in range(len(views)):
                    last_frame = views[i][-1:]  # (1, H, W, 3)
                    padding = last_frame.expand(pad_len, -1, -1, -1)
                    views[i] = torch.cat([views[i], padding], dim=0)

            # Only keep predictions for frames we actually need
            keep_len = min(actual_clip_len, remaining)

            for i in range(len(views)):
                if views[i].shape[0] != actual_clip_len:
                    raise ValueError('All videos must provide the same number of frames')

            crop_history.append({
                'start_frame': current_frame,
                'n_frames': actual_clip_len,
                'crop_boxes': crop_boxes,
            })

            views = [v.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0 for v in views]

            outputs = model(
                views=views,
                coords=current_queries,
                query_times=query_times,
                camera_group=camera_group_chunk,
            )

            coords_pred = outputs['coords_pred'][:, :keep_len]
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
            'coords_pred': torch.empty((0, 0, 0, 3)),
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


def load_tracker_encoder_checkpoint(checkpoint_path, model_kwargs, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackerEncoder(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
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


def load_trial(trial_path, start_frame=0):
    """Load metadata, video/image paths, and query points from a trial directory,
    following the PosetailDataset convention."""

    metadata_path = os.path.join(trial_path, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'metadata.yaml not found in {trial_path}')

    pose_path = os.path.join(trial_path, 'pose3d.npz')
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f'pose3d.npz not found in {trial_path}')

    # determine image or video paths
    img_path = os.path.join(trial_path, 'img')
    vid_path = os.path.join(trial_path, 'vid')

    # Load camera names from metadata to ensure proper ordering
    with open(metadata_path, 'r') as f:
        cam_metadata = yaml.safe_load(f)
    name_source = (cam_metadata['dlt_coefficients']
                   if cam_metadata.get('cam_type') == 'orthographic'
                   else cam_metadata['intrinsic_matrices'])
    cam_names = list(name_source.keys())
    if all(n.isdigit() for n in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)

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

    # load query points from pose3d.npz at start_frame
    data = np.load(pose_path)
    coords = data['pose']  # (subjects, time, n_kpts, 3)
    n_subjects = coords.shape[0]
    n_kpts = coords.shape[2]

    coords_at_start = coords[:, start_frame, :, :]  # (subjects, n_kpts, 3)

    # Build per-subject query points (list of tensors, each (n_valid_kpts, 3))
    per_subject_queries = []
    per_subject_valid_masks = []
    for s in range(n_subjects):
        subject_coords = coords_at_start[s]  # (n_kpts, 3)
        valid = np.all(np.isfinite(subject_coords), axis=1)
        per_subject_valid_masks.append(valid)
        per_subject_queries.append(
            torch.as_tensor(subject_coords[valid], dtype=torch.float32)
        )

    # Also build the concatenated version (original behavior)
    coords_flat = coords_at_start.reshape(-1, 3)  # (subjects * n_kpts, 3)
    valid = np.all(np.isfinite(coords_flat), axis=1)
    coords_flat = coords_flat[valid]

    if coords_flat.shape[0] == 0:
        raise ValueError(f'No valid (non-NaN) query points at frame {start_frame}')

    query_points_3d = torch.as_tensor(coords_flat, dtype=torch.float32)

    return metadata_path, video_paths, query_points_3d, per_subject_queries


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
    parser.add_argument('--per-subject', action='store_true', default=False,
                        help='Track each subject independently instead of concatenating all keypoints')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Optional checkpoint step number; if omitted, use latest checkpoint')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--outpath', type=str, default=None,
                        help='Optional output .npz path')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = None

    model, config, config_path, checkpoint_path = load_model_from_base_folder(
        args.base_folder,
        checkpoint=args.checkpoint,
        device=device,
    )

    if device is None:
        device = next(model.parameters()).device

    metadata_path, video_paths, query_points_3d, per_subject_queries = load_trial(
        args.trial_path, start_frame=args.start_frame)

    camera_group = load_camera_group_from_metadata(metadata_path, device='cpu')

    if len(video_paths) != len(camera_group):
        raise ValueError(
            f'Number of video paths ({len(video_paths)}) does not match '
            f'number of cameras ({len(camera_group)}) in metadata'
        )

    if args.per_subject:
        all_subject_outputs = []
        for subj_idx, subj_queries in enumerate(per_subject_queries):
            if subj_queries.shape[0] == 0:
                print(f'Skipping subject {subj_idx}: no valid query points')
                continue
            print(f'Tracking subject {subj_idx} ({subj_queries.shape[0]} keypoints)')
            subj_out = run_tracker_encoder_on_videos(
                model=model,
                video_paths=video_paths,
                camera_group=camera_group,
                query_points_3d=subj_queries,
                start_frame=args.start_frame,
                n_frames=args.n_frames,
                n_overlap=args.n_overlap,
                device=device,
            )
            subj_out['subject_idx'] = subj_idx
            all_subject_outputs.append(subj_out)

        # Combine per-subject results: stack along a new subject dimension
        if len(all_subject_outputs) == 0:
            outputs = {
                'coords_pred': torch.empty((0, 0, 0, 3)),
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
    else:
        outputs = run_tracker_encoder_on_videos(
            model=model,
            video_paths=video_paths,
            camera_group=camera_group,
            query_points_3d=query_points_3d,
            start_frame=args.start_frame,
            n_frames=args.n_frames,
            n_overlap=args.n_overlap,
            device=device,
        )

    outputs['video_paths'] = np.array(video_paths, dtype=str)
    outputs['metadata_path'] = metadata_path
    outputs['trial_path'] = args.trial_path
    outputs['config_path'] = config_path
    outputs['checkpoint_path'] = checkpoint_path
    outputs['start_frame'] = args.start_frame
    outputs['n_frames_requested'] = args.n_frames
    outputs['n_overlap'] = args.n_overlap
    outputs['per_subject'] = args.per_subject
    if isinstance(outputs['coords_pred'], torch.Tensor) and outputs['coords_pred'].ndim >= 2:
        outputs['n_frames_returned'] = outputs['coords_pred'].shape[1]
    else:
        outputs['n_frames_returned'] = 0

    if args.outpath is not None:
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

        outdir = os.path.dirname(args.outpath)
        if outdir != '':
            os.makedirs(outdir, exist_ok=True)
        np.savez(args.outpath, **save_dict)
        print(f'Saved outputs to {args.outpath}')
    else:
        print('Inference completed.')
        print(f'checkpoint_path: {checkpoint_path}')
        print(f'config_path: {config_path}')
        print(f'n_frames_returned: {outputs["coords_pred"].shape[1] if outputs["coords_pred"].ndim >= 2 else 0}')


if __name__ == '__main__':
    main()
