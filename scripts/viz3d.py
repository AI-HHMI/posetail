import argparse
import os
import glob
import cv2
import torch

import numpy as np
import rerun as rr

from einops import rearrange
from matplotlib.colors import to_rgb

from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera

from posetail.inference.inference_utils import *
from posetail.datasets.utils import get_dirs, load_yaml, disassemble_extrinsics


def format_scheme(scheme, keypoint_names):

    new_scheme = [] 
    kpt_to_ix = dict(zip(keypoint_names, range(len(keypoint_names))))

    for kpt1, kpt2 in scheme: 
        new_scheme.append([kpt_to_ix[kpt1], kpt_to_ix[kpt2]])

    return new_scheme


def viz_predictions_3d_rerun(coords_pred, coords_true, outpath = None, 
                             subject_ids = None, keypoint_names = None, scheme = None,
                             color_pred = 'xkcd:red', color_true = 'xkcd:green', 
                             color_connection = 'xkcd:sky blue', 
                             kpt_radius = 0.05, connection_radius = 0.01, 
                             connect_pred_to_gt = False, spawn = False):

    # get colors for visualization
    color_pred_rgb = to_rgb(color_pred)
    color_true_rgb = to_rgb(color_true)
    color_connection_rgb = to_rgb(color_connection)

    # TODO: convert scheme from keypoint names to index 
    if scheme is not None: 
        scheme = format_scheme(scheme, keypoint_names)
        print(scheme)

    # format subject ids if present, and reshape the coordinates
    if subject_ids is not None: 
        subject_ids = [f'_{sid}' for sid in subject_ids]
    else: 
        subject_ids = ['']

    T = coords_pred.shape[0]
    n_subjects = len(subject_ids)
    coords_pred = rearrange(coords_pred, 't (s n) r -> s t n r', s = n_subjects)
    coords_true = rearrange(coords_true, 't (s n) r -> s t n r', s = n_subjects)

    rr.init('posetail_vis_3d', spawn = spawn)

    for i in range(T):

        rr.set_time_seconds('iteration', i)

        for j in range(n_subjects): 

            # subject name
            subject = subject_ids[j]

            # get keypoints for the current frame
            kpts_true = coords_true[j, i, :, :]
            kpts_pred = coords_pred[j, i, :, :]

            # remove nans 
            kpts_true = kpts_true[torch.isfinite(kpts_true).all(dim = 1)]
            kpts_pred = kpts_pred[torch.isfinite(kpts_pred).all(dim = 1)]
            
            # log true keypoints in green
            rr.log(f'pose_true{subject}', 
                rr.Points3D(
                    kpts_true, 
                    colors = color_true_rgb,
                    radii = kpt_radius
                )
            )

            # log predicted keypoints in red
            rr.log(f'pose_pred{subject}', 
                rr.Points3D(
                    kpts_pred, 
                    colors = color_pred_rgb,
                    radii = kpt_radius
                )
            )
        
            # log connections between ground truth points and 
            # the corresponding predictions
            if connect_pred_to_gt:

                for j in range(kpts_pred.shape[0]):
                    rr.log(
                        f'connections{subject}/point_{j}', # NOTE: depending on nans, this kpt number may not correspond across frames
                        rr.LineStrips3D(
                            strips = [[kpts_pred[j], kpts_true[j]]],
                            colors = color_connection_rgb,
                            radii = connection_radius
                        )
                    )

            # log connections between coords if given a pose skeleton
            valid_connections_true = []
            valid_connections_pred = []

            if scheme: 

                for start_ix, end_ix in scheme:

                    # save valid connections between keypoints in ground truth
                    if not any(torch.isnan(kpts_true[start_ix])) and not any(torch.isnan(kpts_true[end_ix])):

                        valid_connections_true.append([
                            kpts_true[start_ix, :],
                            kpts_true[end_ix, :]
                        ])

                    # save valid connections between keypoints in model predictions
                    if not any(torch.isnan(kpts_pred[start_ix])) and not any(torch.isnan(kpts_pred[end_ix])):

                        valid_connections_pred.append([
                            kpts_pred[start_ix, :],
                            kpts_pred[end_ix, :]
                        ])

                # log valid skeleton between true keypoints in green
                if valid_connections_true:

                    rr.log(f'pose/connections_true{subject}',
                        rr.LineStrips3D(
                            strips = valid_connections_true,
                            colors = color_true_rgb, 
                            radii = connection_radius
                        )
                    )

                # log valid skeleton between predicted keypoints in red
                if valid_connections_pred:

                    rr.log(f'pose/connections_pred{subject}',
                        rr.LineStrips3D(
                            strips = valid_connections_pred,
                            colors = color_pred_rgb, 
                            radii = connection_radius
                        )
                    )

    if outpath: 
        rr.save(outpath)
    
    return outpath


def viz_predictions_3d(split_path, spawn = False, **kwargs):

    device = torch.device('cpu')

    for session in get_dirs(split_path): 

        session_path = os.path.join(split_path, session)

        for trial in get_dirs(session_path):

            trial_path = os.path.join(session_path, trial)

            # skip if there are no npz prediction files
            predictions_path = os.path.join(trial_path, 'predictions.npz')
            if not os.path.exists(predictions_path): 
                print(f'skipping... no predictions found at {trial_path}')
                continue

            # load the coords 
            data = np.load(predictions_path)
            coords_true = torch.from_numpy(data['coords_true'])
            coords_pred = torch.from_numpy(data['coords_pred'])

            subject_ids = None
            if 'subject_ids' in data:
                subject_ids = data['subject_ids']

            # get keypoint names and scheme (if present) from the trial path
            pose_path = data['pose_path']
            scheme = None 
            keypoint_names = None

            if os.path.exists(pose_path): 

                pose_data = np.load(pose_path)
                keypoint_names = pose_data['keypoints']

                if 'scheme' in pose_data: 
                    scheme = pose_data['scheme']

            # save to rrd file (to visualize with rerun)
            rrd_outpath = os.path.join(trial_path, f'predictions_3d.rrd')
            rrd_outpath = viz_predictions_3d_rerun(
                coords_pred, coords_true, rrd_outpath,
                subject_ids = subject_ids, scheme = scheme,
                keypoint_names = keypoint_names, spawn = spawn,
                **kwargs)
            print(f'saved 3d predictions to {rrd_outpath}')


# ---------------------------------------------------------------------------
# New inference pipeline (inference_video.py / inference_dataset.py) support.
#
# inference_video.py writes a single output.npz per trial with
# coords_pred / coords_true shaped (1, T, K, 3) (leading batch dim) and stores
# the source trial_path. The helpers below adapt that format to the rerun
# visualization above without depending on the old predictions.npz layout.
# ---------------------------------------------------------------------------


def load_pose_metadata(trial_path):
    '''
    load keypoint names, skeleton scheme, and subject ids from a trial's
    pose3d.npz. any of the three may be None if not present in the file
    (e.g. cmupanoptic_3dgs has keypoints but no scheme/ids).
    '''
    pose_path = os.path.join(trial_path, 'pose3d.npz')

    if not os.path.exists(pose_path):
        print(f'  no pose3d.npz at {trial_path}; plotting points without skeleton')
        return None, None, None

    pose_data = np.load(pose_path, allow_pickle = True)
    keypoint_names = pose_data['keypoints'] if 'keypoints' in pose_data.files else None
    scheme = pose_data['scheme'] if 'scheme' in pose_data.files else None
    ids = pose_data['ids'] if 'ids' in pose_data.files else None

    return keypoint_names, scheme, ids


def viz_trial(output_path, rrd_name = 'predictions_3d.rrd', spawn = False, **kwargs):
    '''
    write a rerun .rrd visualizing the 3d predictions in a single
    inference_video.py output.npz. the .rrd is written alongside output_path.

    extra kwargs are forwarded to viz_predictions_3d_rerun
    (kpt_radius, connection_radius, colors, connect_pred_to_gt, ...).
    '''
    output_path = str(output_path)
    data = np.load(output_path, allow_pickle = True)

    if 'coords_pred' not in data.files or 'coords_true' not in data.files:
        print(f'  skipping {output_path}: missing coords_pred / coords_true')
        return None

    coords_pred = torch.from_numpy(data['coords_pred'].astype('float32'))
    coords_true = torch.from_numpy(data['coords_true'].astype('float32'))

    # inference_video.py stores a leading batch dim of 1: (1, T, K, 3) -> (T, K, 3)
    if coords_pred.ndim == 4:
        coords_pred = coords_pred[0]
    if coords_true.ndim == 4:
        coords_true = coords_true[0]

    if coords_pred.numel() == 0 or coords_pred.shape[0] == 0:
        print(f'  skipping {output_path}: no predicted frames')
        return None

    # recover keypoint names / skeleton / subject ids from the source trial
    trial_path = str(data['trial_path']) if 'trial_path' in data.files else None
    keypoint_names = scheme = ids = None
    if trial_path is not None:
        keypoint_names, scheme, ids = load_pose_metadata(trial_path)

    # infer the subject count from the number of predicted keypoints.
    # inference_video.py drops NaN query points, so the per-subject skeleton
    # scheme is only valid when no keypoints were masked out (K divisible by n_kpts).
    n_kpts = coords_pred.shape[1]
    n_kpts_full = len(keypoint_names) if keypoint_names is not None else None

    if n_kpts_full and n_kpts % n_kpts_full == 0:
        n_subjects = n_kpts // n_kpts_full
        scheme_to_use = scheme
    else:
        if n_kpts_full and scheme is not None:
            print(f'  note: {n_kpts} predicted kpts not divisible by {n_kpts_full} '
                  f'(NaN query points were dropped); skipping skeleton')
        n_subjects = 1
        scheme_to_use = None

    subject_ids = None
    if n_subjects > 1:
        if ids is not None and len(ids) == n_subjects:
            subject_ids = list(ids)
        else:
            subject_ids = list(range(n_subjects))

    rrd_outpath = os.path.join(os.path.dirname(output_path), rrd_name)
    rrd_outpath = viz_predictions_3d_rerun(
        coords_pred, coords_true, rrd_outpath,
        subject_ids = subject_ids, scheme = scheme_to_use,
        keypoint_names = keypoint_names, spawn = spawn,
        **kwargs)
    print(f'  saved 3d predictions to {rrd_outpath}')

    return rrd_outpath


def find_output_files(output_root, input_name = 'output.npz',
                      datasets = None, splits = None, trials = None):
    '''
    return (label, npz_path) for every input_name found anywhere under
    output_root, at any depth. this is depth-agnostic so it works regardless of
    how the tree is nested, e.g.:
        dataset/split/trial/output.npz
        dataset/split/subject/trial/output.npz
        run_id/dataset/split/subject/trial/output.npz
    label is the trial directory path relative to output_root.

    datasets / splits / trials filter by matching the value against any
    directory component of the path (so they work no matter the nesting depth).
    '''
    results = []

    pattern = os.path.join(output_root, '**', input_name)

    for npz_path in sorted(glob.glob(pattern, recursive = True)):

        rel_dir = os.path.relpath(os.path.dirname(npz_path), output_root)
        parts = rel_dir.split(os.sep)

        if datasets and not any(d in parts for d in datasets):
            continue
        if splits and not any(s in parts for s in splits):
            continue
        if trials and not any(t in parts for t in trials):
            continue

        results.append((rel_dir, npz_path))

    return results


def viz_predictions_dataset(output_root, input_name = 'output.npz',
                            rrd_name = 'predictions_3d.rrd',
                            datasets = None, splits = None, trials = None,
                            force = False, spawn = False, **kwargs):
    '''
    walk an inference output tree (as produced by inference_dataset.py) and
    write a rerun .rrd next to every per-trial output.npz.
    '''
    found = find_output_files(output_root, input_name = input_name,
                              datasets = datasets, splits = splits, trials = trials)

    if len(found) == 0:
        print(f'No {input_name} files found under {output_root}')
        return

    print(f'Found {len(found)} trial(s) under {output_root}')

    n_done = n_skip = n_fail = 0

    for label, npz_path in found:

        rrd_path = os.path.join(os.path.dirname(npz_path), rrd_name)

        if os.path.exists(rrd_path) and not force:
            print(f'  skip {label} (rrd exists)')
            n_skip += 1
            continue

        try:
            viz_trial(npz_path, rrd_name = rrd_name, spawn = spawn, **kwargs)
            n_done += 1
        except Exception as e:
            print(f'  FAILED {label}: {e}')
            n_fail += 1

    print(f'\nDone: {n_done} visualized, {n_skip} skipped, {n_fail} failed')


# ===========================================================================
# 2D projection / video rendering
#
# Projects the 3D predictions stored in an inference_video.py output.npz onto
# each camera's image plane and draws them on the source frames (img/ folders
# or vid/ .mp4 files), writing one annotated .mp4 per camera. The model only
# predicts in 3D, so 2D points are obtained by projecting through the camera
# calibration in metadata.yaml. Colors are BGR (cv2): green = ground truth,
# red = prediction. This is opt-in and never runs as part of inference_dataset.py.
# ===========================================================================


def build_cgroup(metadata, cam_names = None):
    '''
    build an aniposelib CameraGroup from a metadata.yaml dict. if cam_names is
    given, build (only) those cameras in that order -- this is what we want when
    matching the cam_names_used / video_paths stored in an output.npz (which may
    be a numerically-sorted subset of the cameras when --n-views was used).
    '''
    cam_type = metadata.get('cam_type') or 'pinhole'

    if cam_names is None:
        cam_names = list(metadata['intrinsic_matrices'].keys())
        if all(str(c).isdigit() for c in cam_names):
            cam_names = sorted(cam_names, key = int)
        else:
            cam_names = sorted(cam_names)

    cam_objs = []
    for cam_name in cam_names:

        intrinsics = metadata['intrinsic_matrices'][cam_name]
        extrinsics = metadata['extrinsic_matrices'][cam_name]
        distortions = metadata['distortion_matrices'][cam_name]
        rvec, tvec = disassemble_extrinsics(extrinsics)
        tvec = np.asarray(tvec).reshape(3, 1)

        cam_cls = FisheyeCamera if cam_type == 'fisheye' else Camera
        cam_objs.append(cam_cls(matrix = intrinsics, dist = distortions,
                                rvec = rvec, tvec = tvec, name = str(cam_name)))

    return CameraGroup(cam_objs)


def project_to_cameras(cgroup, coords_3d):
    '''
    project (T, K, 3) world coordinates to (C, T, K, 2) pixel coordinates for
    every camera in cgroup. NaN/inf 3D points are projected as NaN (so they can
    be masked out at draw time) rather than crashing cv2.projectPoints.
    '''
    T, K, _ = coords_3d.shape
    flat = coords_3d.reshape(-1, 3)

    valid = np.isfinite(flat).all(axis = 1)
    safe = np.where(valid[:, None], flat, 0.0)

    p2d = cgroup.project(safe)                      # (C, T*K, 2)
    C = p2d.shape[0]
    p2d = p2d.reshape(C, T, K, 2).astype('float64')

    valid_tk = valid.reshape(T, K)
    p2d[:, ~valid_tk] = np.nan

    return p2d


class CameraFrameReader:
    '''
    read frames by absolute index from either a folder of images
    (img/<cam>/000000.jpg ...) or a single video file (vid/<cam>.mp4).
    exposes .size = (width, height) and .fps (None for image folders).
    '''

    def __init__(self, path):
        self.path = path
        self.is_dir = os.path.isdir(path)

        if self.is_dir:
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            self.files = sorted(
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in exts
            )
            if len(self.files) == 0:
                raise FileNotFoundError(f'no images found in {path}')
            first = cv2.imread(os.path.join(path, self.files[0]))
            if first is None:
                raise IOError(f'failed to read {self.files[0]} in {path}')
            h, w = first.shape[:2]
            self.size = (w, h)
            self.fps = None
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f'failed to open video {path}')
            self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else None
            self.files = None

    def read(self, idx):
        if self.is_dir:
            if idx < 0 or idx >= len(self.files):
                return None
            return cv2.imread(os.path.join(self.path, self.files[idx]))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


def draw_points_2d(frame, pts_2d, color, radius = 3):
    '''draw finite, in-frame points (K, 2) onto frame as filled circles.'''
    h, w = frame.shape[:2]
    for k in range(pts_2d.shape[0]):
        x, y = pts_2d[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(frame, (xi, yi), radius, color, -1)
    return frame


def draw_skeleton_2d(frame, pts_2d, scheme_idx, color, n_subjects = 1,
                     n_kpts = None, thickness = 1):
    '''
    draw skeleton edges. scheme_idx is a list of [i, j] index pairs into a
    single subject's keypoints; edges are repeated for each subject block.
    '''
    if scheme_idx is None:
        return frame
    for s in range(n_subjects):
        base = s * n_kpts
        for a, b in scheme_idx:
            ia, ib = base + a, base + b
            pa, pb = pts_2d[ia], pts_2d[ib]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(frame,
                         (int(round(pa[0])), int(round(pa[1]))),
                         (int(round(pb[0])), int(round(pb[1]))),
                         color, thickness)
    return frame


def viz_trial_2d(output_path, output_subdir = 'videos_2d', fps = 30,
                 draw_skeleton = True, point_radius = 3, line_thickness = 1,
                 max_frames = None,
                 color_pred = (0, 0, 255), color_true = (0, 255, 0)):
    '''
    project the 3D predictions in a single inference_video.py output.npz onto
    every used camera and write an annotated <cam>.mp4 per camera into
    <trial_dir>/<output_subdir>/. colors are BGR. returns the output directory.
    '''
    output_path = str(output_path)
    data = np.load(output_path, allow_pickle = True)

    required = ('coords_pred', 'coords_true', 'metadata_path',
                'video_paths', 'cam_names_used', 'frame_numbers')
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f'  skipping {output_path}: missing {missing} (re-run inference_video.py)')
        return None

    coords_pred = data['coords_pred'].astype('float64')
    coords_true = data['coords_true'].astype('float64')
    if coords_pred.ndim == 4:
        coords_pred = coords_pred[0]      # (1, T, K, 3) -> (T, K, 3)
    if coords_true.ndim == 4:
        coords_true = coords_true[0]

    if coords_pred.size == 0 or coords_pred.shape[0] == 0:
        print(f'  skipping {output_path}: no predicted frames')
        return None

    metadata_path = str(data['metadata_path'])
    video_paths = [str(p) for p in data['video_paths']]
    cam_names_used = [str(c) for c in data['cam_names_used']]
    frame_numbers = np.asarray(data['frame_numbers']).astype(int)

    T = coords_pred.shape[0]
    if max_frames is not None:
        T = min(T, max_frames)
    K = coords_pred.shape[1]

    metadata = load_yaml(metadata_path)
    cgroup = build_cgroup(metadata, cam_names_used)

    if len(cgroup.cameras) != len(video_paths):
        print(f'  warning: {len(cgroup.cameras)} cameras built but {len(video_paths)} video paths')

    # project 3D -> 2D for each camera: (C, T, K, 2)
    pred_2d = project_to_cameras(cgroup, coords_pred[:T])
    true_2d = project_to_cameras(cgroup, coords_true[:T])

    # optional skeleton (only when the source pose has a scheme and no query
    # points were dropped, i.e. K is a whole number of full skeletons)
    scheme_idx = None
    n_subjects = 1
    n_kpts = None
    if draw_skeleton:
        trial_path = (str(data['trial_path']) if 'trial_path' in data.files
                      else os.path.dirname(metadata_path))
        keypoint_names, scheme, _ = load_pose_metadata(trial_path)
        n_kpts_full = len(keypoint_names) if keypoint_names is not None else None
        if scheme is not None and keypoint_names is not None and n_kpts_full and K % n_kpts_full == 0:
            scheme_idx = format_scheme(scheme, list(keypoint_names))
            n_kpts = n_kpts_full
            n_subjects = K // n_kpts_full

    out_dir = os.path.join(os.path.dirname(output_path), output_subdir)
    os.makedirs(out_dir, exist_ok = True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    n_written = 0
    for c, (vpath, cname) in enumerate(zip(video_paths, cam_names_used)):

        try:
            reader = CameraFrameReader(vpath)
        except Exception as e:
            print(f'  cam {cname}: cannot open {vpath} ({e})')
            continue

        w, h = reader.size
        out_fps = fps or reader.fps or 30
        out_file = os.path.join(out_dir, f'{cname}.mp4')
        writer = cv2.VideoWriter(out_file, fourcc, out_fps, (w, h))

        for i in range(T):
            frame = reader.read(int(frame_numbers[i]))
            if frame is None:
                continue
            draw_skeleton_2d(frame, true_2d[c, i], scheme_idx, color_true, n_subjects, n_kpts, line_thickness)
            draw_skeleton_2d(frame, pred_2d[c, i], scheme_idx, color_pred, n_subjects, n_kpts, line_thickness)
            draw_points_2d(frame, true_2d[c, i], color_true, point_radius)
            draw_points_2d(frame, pred_2d[c, i], color_pred, point_radius)
            writer.write(frame)

        writer.release()
        reader.release()
        n_written += 1

    print(f'  saved {n_written} 2d video(s) to {out_dir}')
    return out_dir


def viz_predictions_dataset_2d(output_root, input_name = 'output.npz',
                               output_subdir = 'videos_2d',
                               datasets = None, splits = None, trials = None,
                               force = False, **kwargs):
    '''
    walk an inference output tree and write annotated 2d videos for every trial.
    extra kwargs are forwarded to viz_trial_2d (fps, max_frames, draw_skeleton,
    point_radius, line_thickness, colors).
    '''
    found = find_output_files(output_root, input_name = input_name,
                              datasets = datasets, splits = splits, trials = trials)

    if len(found) == 0:
        print(f'No {input_name} files found under {output_root}')
        return

    print(f'Found {len(found)} trial(s) under {output_root}')

    n_done = n_skip = n_fail = 0

    for label, npz_path in found:

        out_dir = os.path.join(os.path.dirname(npz_path), output_subdir)

        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0 and not force:
            print(f'  skip {label} ({output_subdir} exists)')
            n_skip += 1
            continue

        try:
            viz_trial_2d(npz_path, output_subdir = output_subdir, **kwargs)
            n_done += 1
        except Exception as e:
            print(f'  FAILED {label}: {e}')
            n_fail += 1

    print(f'\nDone: {n_done} visualized, {n_skip} skipped, {n_fail} failed')