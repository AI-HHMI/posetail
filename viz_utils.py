import os 

import cv2
import torch

from einops import rearrange
import numpy as np

from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
from posetail.datasets.utils import load_yaml, disassemble_extrinsics


def load_cgroup(metadata): 

    cam_names = sorted(list(metadata['intrinsic_matrices'].keys()))
    cam_objs = []

    cam_type = 'pinhole'
    if 'cam_type' in metadata and metadata['cam_type'] is not None: 
        cam_type = metadata['cam_type']
 
    for cam_name in cam_names:
        
        intrinsics = metadata['intrinsic_matrices'][cam_name]
        extrinsics = metadata['extrinsic_matrices'][cam_name]
        distortions = metadata['distortion_matrices'][cam_name]
        rvec, tvec = disassemble_extrinsics(extrinsics)

        if cam_type == 'fisheye': 
            cam_obj = FisheyeCamera(matrix = intrinsics,
                             dist = distortions,
                             rvec = rvec, 
                             tvec = tvec.reshape(3, 1),
                             name = cam_name)           
        else: 
            cam_obj = Camera(matrix = intrinsics,
                             dist = distortions,
                             rvec = rvec, 
                             tvec = tvec.reshape((3, 1)),
                             name = cam_name)

        cam_objs.append(cam_obj)

    cgroup = CameraGroup(cam_objs)

    return cgroup


def project_points(cgroup, coords_3d):

    coords_2d = cgroup.project(coords_3d)
    t, n, _ = coords_3d.shape
    c = coords_2d.shape[0]
    coords_2d = coords_2d.reshape(c, t, n, -1) # (cameras, time, kpts * n_subjects, 2)

    # apply camera offsets
    for j, cam in enumerate(cgroup): 

        if 'offset_dict' in cam and cam['offset_dict'] is not None: 
            offset = cam['offset_dict']
            coords_2d[j, :, :, 0] -= offset[0]
            coords_2d[j, :, :, 1] -= offset[1]

    return coords_2d


def draw_joints(frame, coords_2d, valid_mask, color):

    for i, valid in enumerate(valid_mask):

        if not valid:
            continue

        x, y = coords_2d[i].astype(int)
        cv2.circle(frame, (x, y), 3, color, -1)

    return frame

def draw_connections(frame, coords_2d, valid_mask, scheme, color):

    for i, j in scheme:
        if valid_mask[i] and valid_mask[j]:
            pt1 = tuple(pts_2d[i].astype(int))
            pt2 = tuple(pts_2d[j].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)

    return frame


def generate_videos_2d(data_path, predictions_path, output_dir):

    metadata_path = os.path.join(data_path, 'metadata.yaml')
    pose_path = os.path.join(data_path, 'pose3d.npz')
    pred_path = os.path.join(predictions_path, 'predictions.npz')
    outpath = os.path.join(predictions_path, 'videos_2d')
    os.makedirs(outpath, exist_ok = True)

    # load camera metadata
    metadata = load_yaml(metadata_path)
    cgroup = load_cgroup(metadata)

    # load pose metadata
    pose_data = np.load(pose_path)
    keypoint_names = pose_data['keypoints']

    scheme = None 
    if 'scheme' in pose_data:   
        scheme = data['scheme']

    n_subjects = 1 
    if 'ids' in pose_data: 
        n_subjects = len(pose_data['ids'])

    # load gt and predictions, then project coordinates
    data = np.load(pred_path)
    coords_true_3d = data['coords_true']
    coords_pred_3d = data['coords_pred']

    coords_true_2d = project_points(cgroup, coords_true_3d)
    coords_pred_2d = project_points(cgroup, coords_pred_3d)

    coords_pred_2d = rearrange(coords_pred_3d, 'c t (s n) r -> c s t n r', s = n_subjects)
    coords_true_2d = rearrange(coords_true_3d, 'c t (s n) r -> c s t n r', s = n_subjects)
    
    T, S, _, _ = coords_true_3d.shape

    # determine format of the data (videos or images)
    if os.path.exists(os.path.join(pred_path, 'vid')):
        pass 

    elif os.path.exists(os.path.join(pred_path, 'img')):
        pass 
    else: 
        print('skipping... no videos or images found')


    caps = [cv2.VideoCapture(str(p)) for p in video_paths]

    writers = []
    for cam, path, cap in zip(camera_group, video_paths, caps):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = os.path.join(output_dir, f'{cam['name']}.mp4")

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        writers.append(writer)

    for t in range(T):
        for cam, cap, writer in zip(camera_group, caps, writers):
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            for s in range(S):
                pts_3d_frame = points_3d[t, s]

                pts_2d, z = project_points(pts_3d_frame, cam)
                pts_2d = pts_2d.cpu().numpy()
                z = z.cpu().numpy()

                # Valid points: in front of camera + inside frame
                valid = (
                    (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
                    (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
                )

                color = colors[s % len(colors)]

                if scheme is not None: 
                    frame = draw_connections(frame, points_2d, valid, scheme, color)

                frame = draw_joints(frame, points_2d, valid, color)

            writer.write(frame)

    for cap in caps:
        cap.release()

    for writer in writers:
        writer.release()

    print(f"Saved projected videos to: {output_dir}")