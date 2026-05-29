import glob
import os

import cv2
import json 

import torch 
from torch.utils.data import Dataset

import numpy as np
import pandas as pd 

from aniposelib.cameras import CameraGroup, Camera
from easydict import EasyDict as edict
from einops import rearrange

from posetail.datasets.utils import get_dirs, load_yaml, disassemble_extrinsics, format_sample_input
from posetail.posetail.cube import project_points_torch, is_point_visible
from train_utils import format_camera_group, dict_to_device, _recompute_ortho_derived



def custom_collate(batch):
    ''' 
    custom collate functon to enable returning 
    non-tensor, non-list, etc type objects from 
    the default collate function
    '''
    batch = list(zip(*batch))

    views = [torch.stack(v, dim = 0) for v in zip(*list(batch[0]))]
    fnums = torch.stack(batch[3], axis = 0)
    cgroup = batch[4][0]

    # get coords (no nan masking, handle downstream)
    coords = torch.stack(batch[1], axis = 0)

    # get corresponding visibilities if present
    vis = None
    if batch[2][0] is not None: 
        vis = torch.stack(batch[2], axis = 0).unsqueeze(-1)

    rows = batch[5][0]
        
    batch = edict({'views': views, 
                   'coords': coords,
                   'vis': vis,
                   'fnums': fnums,
                   'cgroup': cgroup, 
                   'sample_info': rows})

    return batch


class PosetailInferenceDataset(Dataset): 

    def __init__(self, dataset_path, config, split): 

        self.split = split
        assert split in {'train', 'val', 'test'}

        self.dataset_path = dataset_path
        self.split_path = os.path.join(dataset_path, split)
        os.makedirs(self.split_path, exist_ok = True)

        self.n_frames = config.dataset[split].get('n_frames', 100)
        self.max_res = config.dataset[split].get('max_res', -1) # -1 means no resizing
        self.min_res = config.dataset[split].get('min_res', self.max_res) # only used when max_res != -1

        # for sampling cameras, keypoints
        self.cams_to_sample = format_sample_input(config.dataset[split].get('cams_to_sample', None))
        self.kpts_to_sample = format_sample_input(config.dataset[split].get('kpts_to_sample', None))
        self.cam_thresh_for_vis = config.dataset[split].get('cam_thresh_for_vis', 1) 
        # self.enable_kpt_filtering = config.dataset[split].get('enable_kpt_filtering', False)
        
        # generate metadata for the provided data path (requires a specific format)
        self.metadata = self._generate_metadata()


    def __len__(self): 
        return len(self.metadata)


    def __getitem__(self, idx): 
        
        row = self.metadata.loc[idx].to_dict()
        start_ix = row['start_ix']
        interval = row['interval']
        end_ix = start_ix + self.n_frames * interval 
        fnums = torch.arange(start_ix, end_ix, interval)

        # load keypoints and visibilities (if present)
        data = np.load(row['pose_path'])
        coords = data['pose'][:, start_ix:end_ix:interval, :, :]
        coords = torch.tensor(coords, dtype = torch.float32, device = 'cpu')

        vis = None
        if 'vis' in data: 
            vis = data['vis'][:, start_ix:end_ix:interval, :, :]
            vis = torch.tensor(vis, dtype = torch.float32, device = 'cpu')
            vis[torch.isnan(vis)] = 1
            vis = vis.bool()
            
        coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
        if vis is not None:
            vis = rearrange(vis, 's t n c -> t (s n) c') # (time, n_kpts, cams)

        # load camera group and optionally sample camera views
        cgroup, offset_dict, cam_type = self._load_cameras(row['camera_metadata_path']) 
        cam_names = sorted([cam.name for cam in cgroup.cameras]) 

        if self.cams_to_sample: 
            coords, vis, cam_names = self.sample_cameras(coords, vis, cam_names)

        cgroup = cgroup.subset_cameras_names(cam_names)
        cgroup = format_camera_group(cgroup, offset_dict, cam_type, device = 'cpu')

        # if there is no vis provided, a point is considered visible
        # if it can be seen in at least cam_thresh_for_vis cameras
        if vis is not None:
            vis = vis.sum(dim = -1) >= self.cam_thresh_for_vis # (time, n_kpts)                
                
        # # filter coords based on which coords are visible
        # # in the first frame (will sample from these)
        # if vis is not None:
        #     mask = vis[0].bool()
        #     coords = coords[:, mask, :]
        #     vis = vis[:, mask]

        # # filter coords that are not nan throughout
        # mask = torch.all(torch.isfinite(coords), dim=(0, 2))
        # coords = coords[:, mask]

        # if vis is not None: 
        #     vis = vis[:, mask]
        
        # # filter points that are visible in enough views
        # if self.enable_kpt_filtering:
        #     coords, vis = self.filter_keypoints(coords, vis)

        # # sample a random number of keypoints from available tracks 
        # if self.kpts_to_sample: 
        #     coords, vis = self.sample_keypoints(coords, vis)

        # resize cameras
        cgroup = self.resize_camera_group(cgroup)
        
        # load image/video data 
        img_path = row['img_path']
        video_paths = glob.glob(os.path.join(img_path, '*.mp4'))

        # data is in image format 
        if len(video_paths) == 0: 
            views = self.load_images(img_path, cgroup, start_ix, end_ix, interval)

        # data is in video format
        else:
            views = self.load_videos(video_paths, cgroup, start_ix, end_ix, interval)

        return views, coords, vis, fnums, cgroup, row
    

    def load_videos(self, video_paths, cgroup, start_ix, end_ix, interval): 

        cam_names = sorted([cam['name'] for cam in cgroup])
        
        video_paths_subset = [video_path for video_path in video_paths 
                              if os.path.splitext(os.path.basename(video_path))[0] in cam_names]
        
        n_frames = end_ix - start_ix
        views = []

        for cnum, video_path in enumerate(video_paths_subset): 

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_ix)
            imgs = []

            for i in range(n_frames): 

                ret, img = cap.read()

                if not ret:
                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.max_res != -1:
                    img = cv2.resize(img, dsize = cgroup[cnum]['size'].tolist())

                if i % interval == 0: 
                    imgs.append(img)

            cap.release()
            views.append(torch.tensor(np.array(imgs), dtype = torch.float32) / 255.0)

        return views


    def load_images(self, img_path, cgroup, start_ix, end_ix, interval): 

        # load images from paths and resize to desired resolution
        cam_names = sorted([cam['name'] for cam in cgroup])
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix:interval]
        views = []

        for cnum, cam_name in enumerate(cam_names):

            imgs = []

            for img_fname in img_fnames: 

                cam_img_path = os.path.join(img_path, cam_name, img_fname)
                img = cv2.imread(cam_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if self.max_res != -1:
                    img = cv2.resize(img, dsize = cgroup[cnum]['size'].tolist())

                imgs.append(img)

            views.append(torch.tensor(np.array(imgs), dtype = torch.float32) / 255.0)

        return views


    def resize_camera_group(self, cgroup):

        # resize cameras
        if self.max_res != -1:

            target_res = np.random.randint(self.min_res, self.max_res + 1)
            camera_group_scaled = []

            for cnum in range(len(cgroup)):

                cam = dict(cgroup[cnum])
                name = cam['name']
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
                    cam['offset'] = cam['offset'] * scale

                camera_group_scaled.append(cam)

        return camera_group_scaled


    def filter_keypoints(self, coords, vis, cgroup): 

        # filter keypoints that are not visible from enough views 
        s, n, _ = coords.shape
        coords_flat = rearrange(coords, 's n r -> (s n) r')
        all_visible = torch.stack([is_point_visible(cam, coords_flat) 
                                    for cam in cgroup])
        count_flat = torch.sum(all_visible, dim = 0)
        count = rearrange(count_flat, '(s n) -> s n', s = s, n = n)
        good = torch.all(count >= self.cam_thresh_for_vis, dim = 0)
        coords = coords[:, good, :]

        # filter vis if available
        if vis is not None:
            vis = vis[:, good]

        return coords, vis


    def sample_keypoints(self, coords, vis): 

        # sample a random number of keypoints from available tracks 
        if isinstance(self.kpts_to_sample, int): 
            num_kpts_to_sample = self.kpts_to_sample
        else: # sample between a high and low bound 
            num_kpts_to_sample = np.random.randint(self.kpts_to_sample[0], self.kpts_to_sample[1] + 1)

        # sample if there are more keypoints than the number to sample
        if coords.shape[1] > num_kpts_to_sample:   
            ix_p = np.random.choice(coords.shape[1], size = num_kpts_to_sample, replace = False)
            coords = coords[:, ix_p]

            # sample corresponding visibilities
            if vis is not None: 
                vis = vis[:, ix_p]

        return coords, vis
    
    def sample_cameras(self, coords, vis, cam_names): 

        # sample a number of camera views from a set of calibrated cameras
        if isinstance(self.cams_to_sample, int): 
            num_cams_to_sample = self.cams_to_sample
        else: # sample between a high and low bound
            num_cams_to_sample = np.random.randint(self.cams_to_sample[0], self.cams_to_sample[1] + 1)

        if len(cam_names) > num_cams_to_sample:

            ix_cams = np.random.choice(len(cam_names), size = num_cams_to_sample, replace = False)
            cam_names = [cam_names[i] for i in ix_cams]

            # determine visibilities only from the sampled cameras
            if vis is not None: 
                vis = vis[:, :, ix_cams]

        return coords, vis, cam_names

    def _get_start_ixs(self, coords, min_frames = 8):

        start_ixs = []
        intervals = []

        n_start = np.ceil(coords.shape[0] / self.n_frames).astype(int)
        
        for i in range(n_start): 

            coords_subset = coords[i:i + self.n_frames, :, :]
            enough_frames = coords_subset.shape[0] >= self.n_frames
            
            # if there is enough frames (or enough to get by)
            if enough_frames or coords_subset.shape[0] >= min_frames:
                start_ixs.append(i * self.n_frames)
                intervals.append(1)

        start_ixs = np.array(start_ixs)
        intervals = np.array(intervals)
        
        return start_ixs, intervals


    def _generate_metadata(self, track_3d = True): 
            
        rows = []
        mode = '3d' # if track_3d else '2d' - not yet implemented

        for session in get_dirs(self.split_path): 
            session_path = os.path.join(self.split_path, session)

            for trial in get_dirs(session_path):
                # get paths to metadata, 3d pose, and images
                trial_path = os.path.join(session_path, trial)

                metadata_path = os.path.join(trial_path, 'metadata.yaml')
                assert os.path.exists(metadata_path)
                cam_metadata = load_yaml(metadata_path)

                camera_height_dict = cam_metadata['camera_heights']
                camera_width_dict = cam_metadata['camera_widths']

                pose_path = os.path.join(trial_path, f'pose{mode}.npz')
                assert os.path.exists(pose_path)

                # check for image or video data
                img_path = None
                n_cams = None

                if os.path.exists(os.path.join(trial_path, 'img')):
                    img_path = os.path.join(trial_path, 'img')
                    n_cams = len(os.listdir(img_path))

                elif os.path.exists(os.path.join(trial_path, 'vid')): 
                    img_path = os.path.join(trial_path, 'vid')
                    n_cams = len(os.listdir(img_path))
                
                assert img_path is not None and n_cams > 0

                # load coords and subject ids 
                data = np.load(pose_path, allow_pickle = True)
                coords = data['pose']
                ids = None
                if 'ids' in data: 
                    ids = data['ids']

                # get starting indices
                coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
                start_ixs, intervals = self._get_start_ixs(coords, min_frames = self.n_frames)

                # add a row to the metadata that will correspond
                # to each sample within a batch
                for start_ix, interval in zip(start_ixs, intervals): 
                    row = [session, trial, metadata_path, pose_path, 
                           img_path, start_ix, interval, ids, 
                           camera_height_dict, camera_width_dict]
                    rows.append(row)

        columns = ['session', 'trial', 'camera_metadata_path', 
                   'pose_path', 'img_path', 'start_ix', 'interval', 
                   'subject_ids', 'camera_heights', 'camera_widths']

        df = pd.DataFrame(rows, columns = columns)
        df['camera_heights'] = df['camera_heights'].apply(json.dumps)
        df['camera_widths'] = df['camera_widths'].apply(json.dumps)

        return df 
    

    def _load_cameras(self, camera_metadata_path):

        cam_metadata = load_yaml(camera_metadata_path)
        offset_dict = None
        cam_type = 'pinhole'

        if 'cam_type' in cam_metadata:
            cam_type = cam_metadata['cam_type']

        if cam_type == 'orthographic':
            dlt_dict = cam_metadata['dlt_coefficients']
            heights_dict = cam_metadata['camera_heights']
            widths_dict = cam_metadata['camera_widths']
            if 'offset_dict' in cam_metadata:
                offset_dict = cam_metadata['offset_dict']
            cam_names = list(dlt_dict.keys())
            cam_names = sorted(cam_names, key=int) if all(n.isdigit() for n in cam_names) else sorted(cam_names)
            cgroup = [
                {'name': n,
                 'L': list(dlt_dict[n]),
                 'size': [widths_dict[n], heights_dict[n]]}
                for n in cam_names
            ]
            return cgroup, offset_dict, cam_type

        intrinsics_dict = cam_metadata['intrinsic_matrices']
        extrinsics_dict = cam_metadata['extrinsic_matrices']
        distortions_dict = cam_metadata['distortion_matrices']
        heights_dict = cam_metadata['camera_heights']
        widths_dict = cam_metadata['camera_widths']

        if 'offset_dict' in cam_metadata:
            offset_dict = cam_metadata['offset_dict']

        # sort camera names either numerically or alphabetically
        cam_names = list(intrinsics_dict.keys())

        if all(cam_name.isdigit() for cam_name in cam_names):
            cam_names = sorted(cam_names, key = int)
        else: 
            cam_names = sorted(cam_names) 

        cams = []

        for cam_name in cam_names: 

            rvec, tvec = disassemble_extrinsics(extrinsics_dict[cam_name])

            cam = Camera(
                matrix = intrinsics_dict[cam_name],
                dist = distortions_dict[cam_name],
                rvec = rvec,
                tvec = tvec,
                name = cam_name)

            width = widths_dict[cam_name]
            height = heights_dict[cam_name]
            cam.set_size((width, height))
            cams.append(cam)

        cgroup = CameraGroup(cams)

        return cgroup, offset_dict, cam_type
