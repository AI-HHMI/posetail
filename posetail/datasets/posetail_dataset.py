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
from train_utils import format_camera_group, dict_to_device

from pprint import pprint

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imagecorruptions')

import imgaug.augmenters as iaa

from concurrent.futures import ThreadPoolExecutor


def _rotated_rect_max_inscribed(w, h, angle_rad):
    """Largest axis-aligned rectangle inscribed in a w×h rectangle rotated by angle_rad."""
    sin_a = abs(np.sin(angle_rad))
    cos_a = abs(np.cos(angle_rad))
    width_is_longer = w >= h
    long, short = (w, h) if width_is_longer else (h, w)
    if short <= 2 * sin_a * cos_a * long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short
        if width_is_longer:
            cw, ch = x / sin_a, x / cos_a
        else:
            cw, ch = x / cos_a, x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        cw = (w * cos_a - h * sin_a) / cos_2a
        ch = (h * cos_a - w * sin_a) / cos_2a
    return cw, ch


def load_image(cam_img_path, crop_coords=None, target_size=None, rotation=None):
    img = cv2.imread(cam_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if rotation is not None:
        M_2x3, new_size = rotation
        img = cv2.warpAffine(img, M_2x3, new_size)

    if crop_coords is not None:
        x1, y1, x2, y2 = crop_coords
        img = img[y1:y2, x1:x2]

    if target_size is not None:
        img = cv2.resize(img, target_size)

    return img

def get_rows_trial(trial_path, n_frames, split, context,
                   mode='3d'):
    dataset, session, trial = context
    
    metadata_path = os.path.join(trial_path, 'metadata.yaml')
    assert os.path.exists(metadata_path)
    # cam_metadata = load_yaml(metadata_path)
    # camera_height_dict = cam_metadata['camera_heights']
    # camera_width_dict = cam_metadata['camera_widths']

    img_path = os.path.join(trial_path, 'img')
    cams = os.listdir(img_path)
    assert len(cams) > 0

    pose_path = os.path.join(trial_path, f'pose{mode}.npz')
    assert os.path.exists(pose_path)

    # get starting indices 
    data = np.load(pose_path)
    coords = torch.as_tensor(data['pose'])

    coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
    start_ixs, intervals = get_start_ixs(coords, n_frames, split)

    # n_batches = len(imgs) // self.n_frames
    # start_ixs = np.arange(0, len(imgs), self.n_frames)[:n_batches]
    # end_ixs = start_ixs + self.n_frames

    # add a row to the metadata that will correspond
    # to each sample within a batch
    rows = []
    for start_ix, interval in zip(start_ixs, intervals): 
        row = [dataset, session, trial, metadata_path,
               pose_path, img_path, start_ix, interval, 
               # camera_height_dict, camera_width_dict
               ]
        rows.append(row)

    return rows


def get_start_ixs(coords, n_frames, split):

    if split == 'train': 
        start_ixs = get_start_ixs_train(coords, n_frames)
    else: 
        start_ixs = get_start_ixs_test(coords, n_frames)

    return start_ixs


def get_start_ixs_train(coords, n_frames):

    start_ixs = []
    intervals = []

    for interval in [1, 2, 4]:
        for i in range(coords.shape[0] - n_frames * interval + 1): 

            start = i
            end = i + n_frames * interval
            coords_subset = coords[start:end:interval, :, :]        

            # if not all nans in the starting frame 
            if np.isfinite(coords_subset[0]).any():
                start_ixs.append(i)
                intervals.append(interval)


    start_ixs = np.array(start_ixs)
    intervals = np.array(intervals)

    return start_ixs, intervals

def get_start_ixs_test(coords, n_frames):

    safe = 0
    start_ixs = []
    intervals = []

    for i in range(coords.shape[0]): 

        if safe > 0:
            safe = safe - 1 
            continue

        coords_subset = coords[i:i + n_frames, :, :]
        enough_frames = coords_subset.shape[0] == n_frames

        # if not all nans in the starting frame and enough_frames: 
        if np.isfinite(coords_subset[0]).any() and enough_frames:
            start_ixs.append(i)
            intervals.append(1)
            safe = n_frames - 1

    start_ixs = np.array(start_ixs)
    intervals = np.array(intervals)

    return start_ixs, intervals

    
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

    # # mask nan coordinates in the first frame
    coords = torch.stack(batch[1], axis = 0) # (b, t, n_kpts, 3)
    # mask = torch.isfinite(coords).all(dim = -1).all(dim = 1).all(dim = 0)
    # coords_masked = coords[:, :, mask, :]
    #
    p2d = None
    if batch[8][0] is not None:
        p2d = torch.stack(batch[8], axis=0) # (b, cams, t, n_kpts, 2)

    # # get corresponding visibilities if present
    # vis_masked = None
    # if batch[2][0] is not None: 
    #     vis = torch.stack(batch[2], axis = 0)
    #     vis_masked = vis[:, :, mask].unsqueeze(-1)

    vis = None
    if batch[2][0] is not None: 
        vis = torch.stack(batch[2], axis = 0)[..., None]

    vis_2d = None
    if batch[7][0] is not None:
        vis_2d = torch.stack(batch[7], axis=0)[..., None]

        
    rows = batch[5][0]
    query_times = torch.stack(batch[6])
    
    batch = edict({'views': views, 
                   'coords': coords,
                   'p2d': p2d,
                   'query_times': query_times,
                   'vis': vis,
                   'vis_2d': vis_2d,
                   'fnums': fnums,
                   'cgroup': cgroup, 
                   'sample_info': rows})

    return batch


class PosetailDataset(Dataset): 

    def __init__(self, config, split): 

        self.split = split
        assert split in {'train', 'val', 'test'}
        self.split_dir = config.dataset[split].get('split_dir')

        self.data_path = config.dataset.prefix
        self.datasets_to_exclude = config.dataset.get('datasets_to_exclude', [])
        self.n_frames = config.dataset[split].get('n_frames', 16)
        self.max_res = config.dataset[split].get('max_res', -1) # -1 means no resizing
        self.min_res = config.dataset[split].get('min_res', self.max_res) # only used when max_res != -1
        self.aug_prob = config.dataset[split].get('aug_prob', 0.25)
        self.per_image_aug_prob = config.dataset[split].get('per_image_aug_prob', self.aug_prob)
        self.prob_2d_only = config.dataset[split].get('prob_2d_only', 0.0)

        self.crop_to_points = config.dataset[split].get('crop_to_points', True)
        self.min_crop_dim = config.dataset[split].get('min_crop_dim', 64)

        # for sampling cameras, keypoints
        self.cams_to_sample = format_sample_input(config.dataset[split].get('cams_to_sample', None))
        self.kpts_to_sample = format_sample_input(config.dataset[split].get('kpts_to_sample', None))
        self.speed_thresh = config.dataset[split].get('speed_thresh', None) 
        self.prop_dynamic_kpts_to_sample = config.dataset[split].get('prop_dynamic_kpts_to_sample', 0.7)
        self.cam_thresh_for_vis = config.dataset[split].get('cam_thresh_for_vis', 1) 
        self.enable_kpt_filtering = config.dataset[split].get('enable_kpt_filtering', False)
        self.query_anytime = config.dataset[split].get('query_anytime', False)
        self.prob_query_anytime = config.dataset[split].get('prob_query_anytime', 1.0)
        self.no_nan_coords = config.dataset[split].get('no_nan_coords', True)

        # 3D sphere subvolume crop augmentation
        self.crop_3d_enabled = config.dataset[split].get('crop_3d_enabled', False)
        self.crop_3d_fraction = config.dataset[split].get('crop_3d_fraction', [0.3, 0.7])
        self.crop_3d_min_kpts = config.dataset[split].get('crop_3d_min_kpts', 4)
        self.crop_3d_prob = config.dataset[split].get('crop_3d_prob', self.aug_prob)

        # for balancing datasets
        self.balance_datasets = config.dataset[split].get('balance_datasets', True)
        self.n_samples_per_dataset = config.dataset[split].get('n_samples_per_dataset', -1) # default balances based on dataset with the most samples

        
        # per-camera augmentations: same parameters applied to all frames of one camera
        self.aug_per_camera = iaa.Sequential([
            iaa.Sometimes(self.aug_prob, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))),
            # iaa.Sometimes(self.aug_prob, iaa.imgcorruptlike.Contrast(severity=(1,2))),
            iaa.Sometimes(self.aug_prob, iaa.GammaContrast((0.5, 1.8))),
            iaa.Sometimes(self.aug_prob, iaa.AddToSaturation((-50, 30))),
            iaa.Sometimes(self.aug_prob, iaa.AddToHue((-10, 10))),
            # iaa.Sometimes(self.aug_prob, iaa.UniformColorQuantizationToNBits(nb_bits=(3,7))),
            iaa.Sometimes(self.aug_prob, iaa.JpegCompression(compression=(30, 70))),
            # iaa.Sometimes(self.aug_prob, iaa.imgcorruptlike.Pixelate(severity=(1,2))),
        ])

        # per-image augmentations: independently resampled for each frame
        self.aug_per_image = iaa.Sequential([
            iaa.Sometimes(self.per_image_aug_prob, iaa.MotionBlur(k=(3,5))),
            iaa.Sometimes(self.per_image_aug_prob, iaa.AdditiveGaussianNoise(scale=(0, 0.07*255))),
            iaa.Sometimes(self.per_image_aug_prob, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(self.per_image_aug_prob, iaa.SaltAndPepper(0.01)),
        ])
        
        # generate metadata for the provided data path (requires a specific format)
        self.metadata = self._generate_metadata()

        # self.metadata[['scale_dict', 'res_dict', 'new_res_dict']] = self.metadata.apply(
        #     self._get_scale, axis = 1, result_type = 'expand')

        # balances datasets
        if self.balance_datasets:
            print('blancing datasets...') 
            self.metadata = self._balance_metadata(n_samples = self.n_samples_per_dataset)
            print(self.metadata.groupby('dataset').size())

        print("total length:", len(self.metadata))
        self.good_index = np.ones(len(self.metadata), dtype='bool')
        # self.metadata_path = os.path.join(data_path, 'posetail_metadata.csv')
        # self.metadata.to_csv(self.metadata_path, index = False)


    def __len__(self): 
        return len(self.metadata)


    def __getitem__(self, idx):
        start = idx
        out = None
        while True:
            if self.good_index[start]:
                out = self.get_item_actual(start)
            if out is not None:
                return out
            
            self.good_index[start] = False
            start = np.random.randint(len(self.metadata))
            if np.sum(self.good_index) == 0:
                return None # no valid samples
            
            # if start >= self.__len__():
            #     start = np.random.randint(self.__len__())
                
        
    def get_item_actual(self, idx):
        # print(idx, "started")
        
        row = self.metadata.loc[idx].to_dict()
        start_ix = row['start_ix']
        # end_ix = row['end_ix']
        interval = row['interval']
        end_ix = start_ix + self.n_frames * interval 
        fnums = torch.arange(start_ix, end_ix, interval)
        
        # load keypoints 
        data = np.load(row['pose_path'])
        coords = data['pose'][:, start_ix:end_ix:interval, :, :] 
        coords = torch.tensor(coords, dtype = torch.float32, device = 'cpu')

        # load visibilities (if present)
        vis = None
        vis_2d = None
        if 'vis' in data: 
            vis = data['vis'][:, start_ix:end_ix:interval, :, :]
            vis = torch.tensor(vis, dtype = torch.float32, device = 'cpu')
            vis_2d = vis.clone()
            vis[torch.isnan(vis)] = 1
            vis = vis.bool()

        # only augment some of the samples
        should_augment = np.random.random() < 0.75
        should_grayscale = self.split == 'train' and np.random.random() < 0.2
            
        # sample a random subject with 0.5 probability if using a 
        # multi-subject dataset
        if np.random.random() < 0.5:
            ix_sample = np.random.randint(coords.shape[0])
            coords = coords[ix_sample, None]
            if vis is not None:
                vis = vis[ix_sample, None]
                vis_2d = vis_2d[ix_sample, None]
            
        coords = rearrange(coords, 's t n r -> t (s n) r') # (time, n_kpts, 3)
        if vis is not None:
            vis = rearrange(vis, 's t n c -> t (s n) c') # (time, n_kpts, cams)
            vis_2d = rearrange(vis_2d, 's t n c -> t (s n) c')
        
        # load camera resolutions for resizing
        # res_dict = json.loads(row['res_dict'])
        # new_res_dict = json.loads(row['new_res_dict'])
        # scale_dict = json.loads(row['scale_dict'])

        img_path = row['img_path']
        cam_names = get_dirs(img_path)
        img_fnames = sorted(os.listdir(os.path.join(img_path, cam_names[0])))[start_ix:end_ix:interval]

        is_2d_mode = self.prob_2d_only > 0 and np.random.random() < self.prob_2d_only

        if is_2d_mode:
            # Force sample 1 camera
            ix_cams = [np.random.randint(len(cam_names))]
            cam_names = [cam_names[i] for i in ix_cams]
            if vis is not None: 
                vis = vis[:, :, ix_cams]
                vis_2d = vis_2d[:, :, ix_cams]
        elif self.cams_to_sample: 
            coords, vis, vis_2d, cam_names = self.sample_cameras(coords, vis, vis_2d, cam_names)

        if vis is not None:
            vis = vis.sum(dim = -1) >= self.cam_thresh_for_vis # (time, n_kpts)                

  
        # load cameras
        cgroup, offset_dict, cam_type = self._load_cameras(row['camera_metadata_path']) 
        cgroup = cgroup.subset_cameras_names(cam_names)
        cgroup = format_camera_group(cgroup, offset_dict, cam_type, device = 'cpu')


        # per-camera image-plane rotation augmentation (before cropping)
        if should_augment:
            cgroup, rotation_info = self.augment_image_rotation(cgroup)
            # Mark keypoints cropped out by the rotation as not visible.
            if vis_2d is not None:
                s, n, _ = coords.shape
                coords_flat = rearrange(coords, 's n r -> (s n) r')
                for cnum, cam in enumerate(cgroup):
                    if rotation_info[cnum] is None:
                        continue
                    visible = rearrange(is_point_visible(cam, coords_flat),
                                        '(s n) -> s n', s=s, n=n)
                    vis_2d[:, :, cnum][~visible] = 0
                # Re-aggregate vis across cameras (NaN treated as visible, matching the
                # initial collapse at the top of this method).
                if vis is not None:
                    per_cam = vis_2d.clone()
                    per_cam[torch.isnan(per_cam)] = 1
                    vis = per_cam.bool().sum(dim=-1) >= self.cam_thresh_for_vis
        else:
            rotation_info = [None] * len(cam_names)

        # compute per-frame validity mask
        valid_mask = torch.isfinite(coords[..., 0])  # (time, n_kpts)
        if vis is not None:
            valid_mask = valid_mask & vis
        else:
            t, n, _ = coords.shape
            coords_flat = rearrange(coords, 't n r -> (t n) r')
            cam_visible = torch.stack([is_point_visible(cam, coords_flat) for cam in cgroup])
            proxy_vis = rearrange(cam_visible, 'c (t n) -> t n c', t=t, n=n).sum(dim=-1) >= self.cam_thresh_for_vis
            valid_mask = valid_mask & proxy_vis

        # filter coords based on which coords are visible enough times
        if self.query_anytime:
            use_anytime = np.random.random() < self.prob_query_anytime
            if use_anytime:
                mask = valid_mask.sum(dim=0) >= 2
            else:
                edge_frame = np.random.choice([0, self.n_frames - 1])
                mask = valid_mask[edge_frame]
        else:
            use_anytime = False
            edge_frame = 0
            mask = valid_mask[0]

        coords = coords[:, mask]
        if vis is not None:
            vis = vis[:, mask]
            vis_2d = vis_2d[:, mask]                

        # filter coords that are not nan throughout
        if self.no_nan_coords:
            mask = torch.all(torch.isfinite(coords), dim=(0, 2))
            coords = coords[:, mask]
            if vis is not None: 
                vis = vis[:, mask]
                vis_2d = vis_2d[:, mask]
                        
            
        # filter points that are visible in enough views
        if self.enable_kpt_filtering:
            coords, vis, vis_2d = self.filter_keypoints(coords, vis, vis_2d, cgroup)

        if coords.shape[1] < 2:
            return None
                
        # compute total movement and speed in pixels, averaged across cameras
        p2d = project_points_torch(cgroup, coords) # (cams, t, n_kpts, 2)
        movement = torch.linalg.norm(torch.diff(p2d, dim = 1), dim = -1)
        movement = torch.nan_to_num(movement, 0.0)
        total_movement = torch.mean(torch.sum(movement, dim = 1), dim = 0)
        avg_speed = torch.mean(torch.mean(movement, dim = 1), dim = 0) 

        # should have at least 12 pixels of movement over the sampled frames
        good = total_movement >= 12
        if torch.sum(good) < 2: # not enough points with movement
            return None

        # sample a random number of keypoints from available tracks
        fire_3d = self.crop_3d_enabled and (np.random.rand() < self.crop_3d_prob)
        if fire_3d:
            coords, vis, vis_2d = self.sample_keypoints_sphere(
                coords, vis, vis_2d, total_movement, avg_speed)
            if coords.shape[1] < self.crop_3d_min_kpts:
                return None
        elif self.kpts_to_sample:
            coords, vis, vis_2d = self.sample_keypoints(coords, vis, vis_2d, total_movement, avg_speed)

        # failed to sample coordinates, just get another random sample
        if coords.shape[1] < 2:
            return None

        # cropping around coordinates
        # helps for small animals in large arenas
        if self.crop_to_points:
            cgroup, crops = self.crop_cgroup_to_points(cgroup, coords)

        # resize cameras
        if self.max_res != -1:
            cgroup = self.resize_camera_group(cgroup)

        # arbitrary camera rotation
        cgroup, coords = self.rotate_camera_group(cgroup, coords)

        # setup possible query_times (n_kpts)
        if use_anytime:
            query_times = []
            for kpt_idx in range(coords.shape[1]):
                good = torch.isfinite(coords[:, kpt_idx, 0])
                if vis is not None:
                    good = good & vis[:, kpt_idx]
                valid_times = torch.where(good)[0]
                query_time = valid_times[torch.randint(len(valid_times), (1,))]
                query_times.append(query_time.item())
            query_times = torch.tensor(query_times, dtype=torch.int32, device='cpu')
        else:
            query_times = torch.full((coords.shape[1],), edge_frame, dtype=torch.int32)
        
        if is_2d_mode:
            p2d = project_points_torch(cgroup, coords) # (1, t, n_kpts, 2)
        else:
            p2d = None

        # generate per-camera cutout rectangles for random erasing augmentation
        cutout_rects = []
        if should_augment:
            p2d_aug = project_points_torch(cgroup, coords)  # (cams, t, n_kpts, 2)
            for cnum_r in range(len(cam_names)):
                cam_rects = []
                if np.random.random() < self.aug_prob:
                    img_w = cgroup[cnum_r]['size'][0].item()
                    img_h = cgroup[cnum_r]['size'][1].item()
                    n_rects = np.random.randint(1, 4)
                    for _ in range(n_rects):
                        rw = int(img_w * 0.15)
                        rh = int(img_h * 0.15)
                        rx = np.random.randint(0, max(img_w - rw, 1))
                        ry = np.random.randint(0, max(img_h - rh, 1))
                        fill_color = np.random.randint(0, 256, size=3).tolist()
                        cam_rects.append((rx, ry, rx + rw, ry + rh, fill_color))
                        if vis_2d is not None:
                            pts = p2d_aug[cnum_r]  # (t, n_kpts, 2)
                            inside = ((pts[..., 0] >= rx) & (pts[..., 0] <= rx + rw) &
                                      (pts[..., 1] >= ry) & (pts[..., 1] <= ry + rh))
                            vis_2d[:, :, cnum_r][inside] = 0
                cutout_rects.append(cam_rects)
        else:
            cutout_rects = [[] for _ in cam_names]

        # apply augmentation
        with ThreadPoolExecutor(max_workers=24) as executor:
            views_unloaded = []
            for cnum, cam_name in enumerate(cam_names):

                # we apply the same augmentation per camera
                # (thus assuming that each recording is at least self-consistent)
                #
                if self.max_res != -1:
                    target_size = cgroup[cnum]['size'].tolist()
                else:
                    target_size = None

                if self.crop_to_points:
                    crop_coords = crops[cnum]
                else:
                    crop_coords = None

                rotation = rotation_info[cnum]

                futures = []
                # load images from paths and resize to desired resolution
                for img_fname in img_fnames:
                    cam_img_path = os.path.join(img_path, cam_name, img_fname)
                    future = executor.submit(
                        load_image,
                        cam_img_path, crop_coords, target_size, rotation)
                    futures.append(future)
                views_unloaded.append(futures)

            views = []
            for cnum, futures in enumerate(views_unloaded):
                imgs = [f.result() for f in futures]
                if should_augment:
                    aug_cam_det = self.aug_per_camera.to_deterministic()
                    imgs = [aug_cam_det(image=img) for img in imgs]
                    imgs = [self.aug_per_image(image=img) for img in imgs]

                for rect in cutout_rects[cnum]:
                    rx1, ry1, rx2, ry2, fill_color = rect
                    for img in imgs:
                        img[ry1:ry2, rx1:rx2] = fill_color

                if should_grayscale:
                    imgs = [np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
                            for img in imgs]

                imgs = torch.tensor(np.array(imgs), dtype = torch.float32, device='cpu')
                imgs = imgs / 255.0
                views.append(imgs)


        # p2d = project_points_torch(cgroup, coords) # (cams, t, n_kpts, 2)
        
        return views, coords, vis, fnums, cgroup, row, query_times, vis_2d, p2d


    def crop_cgroup_to_points(self, cgroup, coords): 
            
        # compute crops locations
        p2d = project_points_torch(cgroup, coords)
        crops = []

        for cnum in range(p2d.shape[0]):
            
            size = cgroup[cnum]['size']
            pflat = p2d[cnum].reshape(-1, 2)
            good = torch.all(torch.isfinite(pflat), dim=1)
            pflat = pflat[good]
            low = torch.clamp(torch.min(pflat, dim=0).values - 20, torch.tensor([0,0]), size).to(torch.int32)
            high = torch.clamp(torch.max(pflat, dim=0).values + 20, torch.tensor([0,0]), size).to(torch.int32)

            current_width = high[0] - low[0]
            current_height = high[1] - low[1]

            # Each axis is capped at the image dimension so the crop never
            # exceeds image bounds. Without the cap, a wide bbox (e.g. 700 px
            # on a 540-tall image) forces min_dim=700 > size[1]=540, making
            # torch.clamp(x, 0, size[1]-min_dim) return a negative max value
            # and producing a negative cam['offset'] that breaks project_cam.
            base = max(self.min_crop_dim, int(current_width), int(current_height))
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

        # camera crops
        camera_group_cropped = []
        for cnum in range(len(cgroup)):
            x1, y1, x2, y2 = crops[cnum]
            cam = dict(cgroup[cnum])
            cam['offset'] = cam['offset'] + torch.tensor([x1, y1], dtype=torch.int32, device='cpu')
            cam['size'] = torch.tensor([x2 - x1, y2 - y1], dtype=torch.int32, device='cpu')
            camera_group_cropped.append(cam)
        
        return camera_group_cropped, crops


    def filter_keypoints(self, coords, vis, vis_2d, cgroup): 

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
            vis_2d = vis_2d[:, good]

        return coords, vis, vis_2d


    def sample_cameras(self, coords, vis, vis_2d, cam_names): 

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
                vis_2d = vis_2d[:, :, ix_cams]

        return coords, vis, vis_2d, cam_names


    def sample_keypoints(self, coords, vis, vis_2d, total_movement, avg_speed): 

        if isinstance(self.kpts_to_sample, int): 
            num_kpts_to_sample = self.kpts_to_sample

        else: # sample between a high and low bound 
            num_kpts_to_sample = np.random.randint(self.kpts_to_sample[0], self.kpts_to_sample[1] + 1)

        # sample if there are more keypoints than the number to sample
        if coords.shape[1] > num_kpts_to_sample:
            # sample a proportion of static vs dynamic points if a speed thresh is provided
            if self.speed_thresh is not None: 

                dynamic_mask = avg_speed >= self.speed_thresh
                static_mask = ~dynamic_mask

                num_dynamic = int(num_kpts_to_sample * self.prop_dynamic_kpts_to_sample)
                num_static = num_kpts_to_sample - num_dynamic

                dynamic_idx = torch.where(dynamic_mask)[0].cpu().numpy()
                static_idx = torch.where(static_mask)[0].cpu().numpy()

                num_dynamic = min(num_dynamic, len(dynamic_idx))
                num_static = min(num_static, len(static_idx))

                sampled_dynamic = np.random.choice(dynamic_idx, size = num_dynamic, replace = False) if len(dynamic_idx) > 0 else []
                sampled_static = np.random.choice(static_idx, size = num_static, replace = False) if len(static_idx) > 0 else []

                ix_p = np.concatenate([sampled_dynamic, sampled_static])
                np.random.shuffle(ix_p)
                coords = coords[:, ix_p]

            # otherwise, default to sampling probabilities based on total movement
            else: 
                prob = (total_movement + 2) / torch.sum(total_movement + 2)
                prob = prob.numpy()
                
                ix_p = np.random.choice(coords.shape[1], size = num_kpts_to_sample,
                                        replace = False, p = prob)
                coords = coords[:, ix_p]

            # sample corresponding visibilities
            if vis is not None: 
                vis = vis[:, ix_p]
                vis_2d = vis_2d[:, ix_p]

        return coords, vis, vis_2d


    def sample_keypoints_sphere(self, coords, vis, vis_2d, total_movement, avg_speed):
        T, N, _ = coords.shape

        valid = torch.isfinite(coords).all(dim=-1)   # (T, N)
        has_any = valid.any(dim=0)                   # (N,)
        if has_any.sum() < 2:
            return coords, vis, vis_2d

        first_valid_t = valid.float().argmax(dim=0)              # (N,)
        kpt_coords = coords[first_valid_t, torch.arange(N)]     # (N, 3)

        # pick center kpt from dynamic subset, fall back to movement-weighted
        if self.speed_thresh is not None:
            dynamic = (avg_speed >= self.speed_thresh) & has_any
            if dynamic.any():
                cand = torch.where(dynamic)[0]
                center_kpt = int(cand[np.random.randint(len(cand))])
            else:
                cand = torch.where(has_any)[0]
                probs = (total_movement[cand] + 2)
                probs = probs / probs.sum()
                center_kpt = int(cand[torch.multinomial(probs, 1).item()])
        else:
            cand = torch.where(has_any)[0]
            probs = (total_movement[cand] + 2)
            probs = probs / probs.sum()
            center_kpt = int(cand[torch.multinomial(probs, 1).item()])

        center = kpt_coords[center_kpt]   # (3,)

        # compute distances; treat kpts with no valid time as inf
        dists = torch.linalg.norm(kpt_coords - center, dim=-1)
        dists = torch.where(has_any, dists, torch.full_like(dists, float('inf')))
        finite_d = dists[torch.isfinite(dists)]
        if finite_d.numel() < 2 or finite_d.max() == 0:
            return coords, vis, vis_2d

        f_lo, f_hi = self.crop_3d_fraction
        fraction = float(np.random.uniform(f_lo, f_hi))
        radius = finite_d.max() * fraction
        in_sphere = dists <= radius

        coords = coords[:, in_sphere]
        if vis is not None:
            vis = vis[:, in_sphere]
            vis_2d = vis_2d[:, in_sphere]
        tm_s = total_movement[in_sphere]
        sp_s = avg_speed[in_sphere]

        if self.kpts_to_sample:
            coords, vis, vis_2d = self.sample_keypoints(coords, vis, vis_2d, tm_s, sp_s)

        return coords, vis, vis_2d


    def resize_camera_group(self, cgroup): 

        target_res = np.random.randint(self.min_res, self.max_res + 1)
        camera_group_scaled = []

        for cnum in range(len(cgroup)):

            cam = dict(cgroup[cnum])
            size = cam['size']
            scale = float(target_res) / max(size)
            cam['size'] = torch.round(size * scale).to(torch.int32)
            cam['mat'] = cam['mat'] * scale
            cam['mat'][2, 2] = 1

            if 'offset' in cam:
                cam['offset'] = cam['offset'] * scale

            camera_group_scaled.append(cam)

        return camera_group_scaled


    def augment_image_rotation(self, cgroup):
        rotation_info = []
        cgroup_rotated = []

        for cam in cgroup:
            if np.random.random() >= self.aug_prob:
                cgroup_rotated.append(cam)
                rotation_info.append(None)
                continue

            angle = float(np.random.uniform(-45, 45))
            angle_rad = np.radians(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            w, h = cam['size'].tolist()
            cx = float(cam['mat'][0, 2].item())
            cy = float(cam['mat'][1, 2].item())
            off_x = float(cam['offset'][0].item())
            off_y = float(cam['offset'][1].item())

            # rotate around the cropped principal point so the image rotation
            # matches the extrinsic Z-roll. equals (cx, cy) when offset is 0.
            center_x = cx - off_x
            center_y = cy - off_y
            M_2x3 = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
            corners_rot = corners @ M_2x3[:, :2].T + M_2x3[:, 2]
            min_x, min_y = corners_rot.min(axis=0)
            max_x, max_y = corners_rot.max(axis=0)
            tx, ty = -min_x, -min_y
            M_2x3[0, 2] += tx
            M_2x3[1, 2] += ty

            # Crop to the largest axis-aligned rectangle with no black borders.
            # The crop is centered on the rotated image center in the expanded canvas.
            cw, ch = _rotated_rect_max_inscribed(w, h, angle_rad)
            cw_i, ch_i = int(np.floor(cw)), int(np.floor(ch))
            img_ctr = M_2x3[:, :2] @ np.array([w / 2, h / 2]) + M_2x3[:, 2]
            x1 = img_ctr[0] - cw_i / 2
            y1 = img_ctr[1] - ch_i / 2
            M_2x3[0, 2] -= x1
            M_2x3[1, 2] -= y1

            cam_rot = dict(cam)

            # OpenCV y-down sign convention: R_roll[:2,:2] = [[c, s], [-s, c]]
            R_roll = torch.eye(4, dtype=cam['ext'].dtype, device=cam['ext'].device)
            R_roll[0, 0] = cos_a
            R_roll[0, 1] = sin_a
            R_roll[1, 0] = -sin_a
            R_roll[1, 1] = cos_a
            cam_rot['ext'] = R_roll @ cam['ext']
            cam_rot['ext_inv'] = torch.linalg.inv(cam_rot['ext'])
            cam_rot['center'] = -cam_rot['ext'][:3, :3].T @ cam_rot['ext'][:3, 3]

            # mat[:2,:2] is unchanged (stays diagonal); principal point tracks the
            # canvas expansion (tx, ty) and the crop offset (x1, y1). offset unchanged.
            cam_rot['mat'] = cam['mat'].clone()
            cam_rot['mat'][0, 2] = cam['mat'][0, 2] + tx - x1
            cam_rot['mat'][1, 2] = cam['mat'][1, 2] + ty - y1

            cam_rot['offset'] = cam['offset'].clone()

            cam_rot['size'] = torch.tensor([cw_i, ch_i], dtype=torch.int32,
                                           device=cam['size'].device)

            cgroup_rotated.append(cam_rot)
            rotation_info.append((M_2x3, (cw_i, ch_i)))

        return cgroup_rotated, rotation_info


    def rotate_camera_group(self, cgroup, coords):
                
        rvec = np.random.uniform(-2*np.pi, 2*np.pi, size=3)
        rotmat, _ = cv2.Rodrigues(np.array(rvec))
        rotmat = torch.as_tensor(rotmat, device=coords.device, dtype=coords.dtype)
        coords = torch.matmul(coords, rotmat)

        rmat = torch.eye(4, device=coords.device, dtype=coords.dtype)
        rmat[:3,:3] = rotmat
        camera_group_rotated = list()

        for cam in cgroup:
            cam_rot = dict(cam)
            cam_rot['ext'] = torch.matmul(cam['ext'], rmat)
            cam_rot['ext_inv'] = torch.linalg.inv(cam_rot['ext'])

            R = cam_rot['ext'][:3,:3]
            t = cam_rot['ext'][:3, 3]
            cam_rot['center'] = -R.T @ t

            camera_group_rotated.append(cam_rot)

        cgroup = camera_group_rotated 

        return cgroup, coords


    def _generate_metadata(self, track_3d = True): 
            
        rows = []
        mode = '3d' # if track_3d else '2d' - not yet implemented

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for dataset in get_dirs(self.data_path):

                if dataset in self.datasets_to_exclude: 
                    continue

                # NOTE: split folder structure must match here
                dataset_path = os.path.join(self.data_path, dataset, self.split_dir)

                # skip dataset if this particular split doesn't exist
                if not os.path.exists(dataset_path): 
                    continue

                for session in get_dirs(dataset_path): 
                    session_path = os.path.join(dataset_path, session)

                    for trial in get_dirs(session_path):
                        # get paths to metadata, 3d pose, and images
                        trial_path = os.path.join(session_path, trial)
                        future = executor.submit(
                            get_rows_trial,
                            trial_path, self.n_frames, self.split,
                            (dataset, session, trial),
                            mode)
                        futures.append(future)
                        
            for future in futures:
                add_rows = future.result()
                rows.extend(add_rows)

        columns = ['dataset', 'session', 'trial', 'camera_metadata_path', 
                   'pose_path', 'img_path', 'start_ix', 'interval', 
                   # 'camera_heights', 'camera_widths'
                   ]

        df = pd.DataFrame(rows, columns = columns)
        # df['camera_heights'] = df['camera_heights'].apply(json.dumps)
        # df['camera_widths'] = df['camera_widths'].apply(json.dumps)

        return df 
    
    def _balance_group(self, df, n_samples = 1000, random_state = 3): 

        duplicates = int(np.ceil(n_samples / len(df)))

        if duplicates > 1: 
            df = pd.concat([df] * duplicates, axis = 0)# .reset_index(drop = True)

        df_balanced = df.sample(n = n_samples, random_state = random_state)

        return df_balanced

    def _balance_metadata(self, n_samples = -1, random_state = 3): 

        self.metadata['dataset2'] = self.metadata['dataset'].copy()

        # balance the dataset according to the dataset with the most samples
        if n_samples == -1: 
            n_samples = self.metadata.groupby('dataset2').size().max()

        # balance and sample based on a predefined number of samples
        df_balanced = (self.metadata.groupby('dataset2')
                           .apply(lambda x: self._balance_group(x, 
                                                                n_samples = n_samples, 
                                                                random_state = random_state), 
                                                                include_groups = False)
                           .reset_index(drop = True))
        
        return df_balanced

    # def _get_scale(self, row): 

    #     scale_dict = {}
    #     res_dict = {}
    #     new_res_dict = {}

    #     camera_height_dict = json.loads(row['camera_heights'])
    #     camera_width_dict = json.loads(row['camera_widths'])

    #     for cam_name, height in camera_height_dict.items():

    #         width = camera_width_dict[cam_name]

    #         if self.max_res != -1: 
    #             scale = self.max_res / max(height, width)
    #         else: 
    #             scale = 1

    #         orig_res = [width, height]
    #         new_res = [round(width * scale), round(height * scale)]
    #         # xy_scale = (orig_res[0] / new_res[0], orig_res[1] / new_res[1])

    #         scale_dict[cam_name] = scale
    #         res_dict[cam_name] = orig_res
    #         new_res_dict[cam_name] = new_res
        
    #     scale_dict = json.dumps(scale_dict)
    #     res_dict = json.dumps(res_dict)
    #     new_res_dict = json.dumps(new_res_dict)

    #     return scale_dict, res_dict, new_res_dict


    def _load_cameras(self, camera_metadata_path):

        cam_metadata = load_yaml(camera_metadata_path)
        offset_dict = None
        cam_type = 'pinhole'

        intrinsics_dict = cam_metadata['intrinsic_matrices']
        extrinsics_dict = cam_metadata['extrinsic_matrices']
        distortions_dict = cam_metadata['distortion_matrices']
        heights_dict = cam_metadata['camera_heights']
        widths_dict = cam_metadata['camera_widths']

        if 'offset_dict' in cam_metadata: 
            offset_dict = cam_metadata['offset_dict']

        if 'cam_type' in cam_metadata: 
            cam_type = cam_metadata['cam_type']

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
            # cam.resize_camera(scale_dict[cam_name])
            cams.append(cam)

            # if offset_dict: 
            #     offsets = offset_dict[cam_name]
            #     offset_dict[cam_name] = [offsets[0] * scale_dict[cam_name], offsets[1] * scale_dict[cam_name]]

        cgroup = CameraGroup(cams)

        return cgroup, offset_dict, cam_type
