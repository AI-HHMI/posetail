import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from posetail.posetail.cube import get_camera_scale, project_points_torch, is_point_visible

from collections import defaultdict

class TotalLoss(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, delta = 6,
                 use_huber_loss = False, vis_loss_weight = 1,
                 conf_loss_weight = 1, coords_loss_weight = 1,
                 occluded_coords_loss_weight = 1,
                 feature_loss_weight = 0.5,
                 coords_loss_direct_weight = 0.1,
                 coords_loss_rays_weight = 0.001,
                 coords_loss_triangulate_weight = 0.1,
                 coords_loss_2d_weight = 1,
                 coords_loss_depth_weight = 1,
                 conf_2d_loss_weight = 0,
                 per_camera_cube_scale = False):
        super().__init__()

        self.gamma = gamma
        self.pixel_thresh = pixel_thresh
        self.delta = delta
        
        self.use_huber_loss = use_huber_loss

        # weight for each loss (0 to not use or compute)
        self.vis_loss_weight = vis_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.coords_loss_weight = coords_loss_weight
        self.occluded_coords_loss_weight = occluded_coords_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.conf_2d_loss_weight = conf_2d_loss_weight
        
        self.coords_loss_direct_weight = coords_loss_direct_weight
        self.coords_loss_rays_weight = coords_loss_rays_weight
        self.coords_loss_triangulate_weight = coords_loss_triangulate_weight
        self.coords_loss_2d_weight = coords_loss_2d_weight
        self.coords_loss_depth_weight = coords_loss_depth_weight
        self.per_camera_cube_scale = per_camera_cube_scale

        self.bce_loss_vis = BCELossVis(
            gamma = self.gamma, 
            weight = self.vis_loss_weight
        )

        self.bce_loss_conf = BCELossConf(
            gamma = self.gamma, 
            pixel_thresh = self.pixel_thresh, 
            weight = self.conf_loss_weight
        )

        self.mae_loss_coords = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_weight
        )

        self.mae_loss_coords_direct = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_direct_weight
        )

        self.mae_loss_coords_rays = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_rays_weight
        )

        self.mae_loss_coords_triangulate = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_triangulate_weight
        )
        
        self.mae_loss_coords_2d = WeightedMAELoss(
            gamma = self.gamma, 
            delta = 16, 
            use_huber_loss = True, 
            weight = self.coords_loss_2d_weight / 16.0
        )

        self.mae_loss_coords_depth = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.coords_loss_depth_weight
        )
        
        self.mae_loss_occluded_coords = WeightedMAELoss(
            gamma = self.gamma, 
            delta = self.delta, 
            use_huber_loss = self.use_huber_loss, 
            weight = self.occluded_coords_loss_weight
        )

        # self.mae_loss_occluded_coords_2d = WeightedMAELoss(
        #     gamma = self.gamma, 
        #     delta = 16, 
        #     use_huber_loss = True,
        #     weight = self.occluded_coords_loss_weight * 10 / 16.0
        # )
        
        self.bce_loss_conf_2d = BCELossConf(
            gamma = self.gamma,
            pixel_thresh = self.pixel_thresh,
            weight = self.conf_2d_loss_weight
        )

        self.feature_loss = FeatureLoss(
            weight = self.feature_loss_weight)

        # loss_names = ['vis_loss', 'conf_loss', 
        #               'occluded_coords_loss', 'coords_loss',
        #               '2d_loss', 'depth_loss',
        #               # 'feature_loss','bad_feature_loss',
        #               'total_loss', 'cube_scale']
        # self.loss_history = {loss_name: [] for loss_name in loss_names}
        self.loss_history = defaultdict(list)
        
    def collapse_history(self, prefix = ''): 
        
        loss_summary = {}

        for name, losses in self.loss_history.items():
            loss_summary[f'{prefix}{name}'] = float(np.nanmean(losses))
            
        return loss_summary

    def reset_history(self):
        self.loss_history = {name: [] for name in list(self.loss_history.keys())}


    def forward(self, model, outputs, coords_true, 
                vis_true, vis_true_cams, cgroup = None, p2d = None, device = None):

        B, T, N, R = coords_true.shape
        
        coords_pred = outputs['coords_pred']
        vis_pred = outputs['vis_pred']
        conf_pred = outputs['conf_pred']

        # Handle 2D predictions
        if '2d_pred' in outputs:
            coords_pred_2d = outputs['2d_pred']
        else:
            coords_pred_2d = None
            
        # Handle depth predictions
        if 'depth_pred' in outputs:
            depth_pred = outputs['depth_pred'][..., None]
        else:
            depth_pred = None

        if p2d is not None:
            coords_true_2d = p2d
        else:
            coords_true_2d = project_points_torch(cgroup, coords_true)
            
        coords_true_cams = repeat(coords_true, 'b t n r -> cams b t n r',
                                  cams=len(cgroup)) 

        centers = rearrange(torch.stack([cam['center'] for cam in cgroup]),
                            'cams r -> cams 1 1 1 r')
        
        depths_true = torch.linalg.norm(coords_true_cams - centers, dim=-1)[..., None]
        
        if vis_true is None:
            valid_vis = False
            vis_true = get_vis_true(coords_true)
            # vis_true_cams = repeat(vis_true, 'b t n 1 -> cams b t n 1', cams=len(cgroup))

            qflat = rearrange(coords_true, 'b t n r -> (b t n) r')
            visible = torch.stack([
                is_point_visible(cam, qflat, margin=2)
                for cam in cgroup
            ])
            vis_true_cams = rearrange(visible, 'cams (b t n) -> cams b t n 1', b=B, t=T, n=N)
            
        else:
            valid_vis = True
            vis_true_cams = rearrange(vis_true_cams, 'b t n cams 1 -> cams b t n 1')
            

        scale = get_camera_scale(cgroup, coords_true.reshape(B, -1, 3))  # (cams, B)
        if not self.per_camera_cube_scale:
            med = scale.median(dim=0).values  # (B,)
            scale = med[None, :].expand_as(scale).contiguous()

        if p2d is not None:
            # for 2d prediction, the cube_scale is 1
            scale_cb = rearrange(scale, 'cams b -> cams b 1 1 1')
            depths_true = depths_true / scale_cb

            # Scale 3D targets down relative to the camera center
            # Since R=2 implies len(cgroup) == 1, we use the single camera center
            C = centers[0]
            coords_true_cams = C + (coords_true_cams - C) / scale_cb
            coords_true = C + (coords_true - C) / scale[0].reshape(B, 1, 1, 1)
            
        occluded_true = ~vis_true

        if p2d is None:
            # 3D mode: pre-compute per-(cam,B) and per-batch scales for loss normalization
            scale_cb = rearrange(scale, 'cams b -> cams b 1 1 1')
            scale_b  = scale.median(dim=0).values.reshape(B, 1, 1, 1)
        else:
            # 2D mode: targets already normalized; leave MAE losses unnormalized
            scale_cb = 1.0
            scale_b  = 1.0

        training_iters = model.training and 'coords_pred_iters' in outputs
        
        if training_iters:
            coords_pred_iters = outputs['coords_pred_iters']
            vis_pred_iters = outputs['vis_pred_iters']
            conf_pred_iters = outputs['conf_pred_iters']

            coords_true_unrolled, vis_true_unrolled, occluded_true_unrolled = unroll_batch(
                coords = coords_true, 
                vis = vis_true, 
                stride = model.S, 
                stride_overlap = model.stride_overlap)

        # compute losses
        if valid_vis:
            vis_loss = self.bce_loss_vis(
                vis_pred = vis_pred_iters if training_iters else vis_pred,
                vis_true = vis_true_unrolled if training_iters else vis_true,
                device = device
            )
            if 'vis_pred_2d' in outputs:
                vis_loss_cams = self.bce_loss_vis(
                    vis_pred = rearrange(outputs['vis_pred_2d'], 'b t n cams -> b t n cams 1'),
                    vis_true = vis_true_cams,
                    device = device
                )
            else:
                vis_loss_cams = torch.tensor(0.0, device=device)
        else:
            vis_loss = torch.tensor(0.0, device=device)
            vis_loss_cams = torch.tensor(0.0, device=device)

        if 'conf_3d' in outputs:
            conf_loss = self.bce_loss_conf(
                conf_pred = outputs['conf_3d'][..., None],
                coords_pred = outputs['3d_pred_cams_direct'],
                coords_true = coords_true_cams,
                vis_true = vis_true_cams,
                scale = rearrange(scale, 'cams b -> cams b 1 1'),
                device = device
            )
        else:
            conf_loss = self.bce_loss_conf(
                conf_pred = conf_pred_iters if training_iters else conf_pred,
                coords_pred = coords_pred_iters if training_iters else coords_pred,
                coords_true = coords_true_unrolled if training_iters else coords_true,
                vis_true = vis_true_unrolled if training_iters else vis_true,
                scale = scale.mean(dim=0).reshape(B, 1, 1),
                device = device
            )

        conf_loss_2d = torch.tensor(0.0, device=device)
        if 'conf_pred_2d' in outputs and coords_pred_2d is not None:
            conf_loss_2d = self.bce_loss_conf_2d(
                conf_pred = outputs['conf_pred_2d'][..., None],
                coords_pred = coords_pred_2d,
                coords_true = coords_true_2d,
                vis_true = vis_true_cams,
                scale = 1.0,
                device = device
            )

        coords_loss = self.mae_loss_coords(
            coords_pred = coords_pred_iters if training_iters else coords_pred,
            coords_true = coords_true_unrolled if training_iters else coords_true,
            vis_true = vis_true_unrolled if training_iters else vis_true,
            scale = scale_b,
            device = device
        )

        coords_loss_direct = torch.tensor(0.0, device=device)
        coords_loss_rays = torch.tensor(0.0, device=device)
        coords_loss_triangulate = torch.tensor(0.0, device=device)

        if '3d_pred_cams_direct' in outputs:
            coords_loss_direct += self.mae_loss_coords_direct(
                coords_pred = outputs['3d_pred_cams_direct'],
                coords_true = coords_true_cams,
                vis_true = vis_true_cams,
                scale = scale_cb,
                device = device
            )

        if '3d_pred_direct' in outputs:
            coords_loss_direct += self.mae_loss_coords_direct(
                coords_pred = outputs['3d_pred_direct'],
                coords_true = coords_true,
                vis_true = vis_true,
                scale = scale_b,
                device = device
            )

        if '3d_pred_cams_rays' in outputs:
            coords_loss_rays += self.mae_loss_coords_rays(
                coords_pred = outputs['3d_pred_cams_rays'],
                coords_true = coords_true_cams,
                vis_true = vis_true_cams,
                scale = scale_cb,
                device = device
            )

        if '3d_pred_rays' in outputs:
            coords_loss_rays += self.mae_loss_coords_rays(
                coords_pred = outputs['3d_pred_rays'],
                coords_true = coords_true,
                vis_true = vis_true,
                scale = scale_b,
                device = device
            )

        if '3d_pred_triangulate' in outputs and outputs['3d_pred_triangulate'] is not None:
            coords_loss_triangulate += self.mae_loss_coords_triangulate(
                coords_pred = outputs['3d_pred_triangulate'],
                coords_true = coords_true,
                vis_true = vis_true,
                scale = scale_b,
                device = device
            )

        if coords_pred_2d is not None:
            coords_loss_2d = self.mae_loss_coords_2d(
                coords_pred = coords_pred_2d,
                coords_true = coords_true_2d,
                vis_true = vis_true_cams,
                device = device
            )
        else:
            coords_loss_2d = torch.tensor(0.0, device=device)

        if depth_pred is not None:
            coords_loss_depth = self.mae_loss_coords_depth(
                coords_pred = depth_pred,
                coords_true = depths_true,
                vis_true = vis_true_cams,
                scale = scale_cb,
                device = device
            )
        else:
            coords_loss_depth = torch.tensor(0.0, device=device)

        if valid_vis:
            occluded_coords_loss = self.mae_loss_occluded_coords(
                coords_pred = coords_pred_iters if training_iters else coords_pred,
                coords_true = coords_true_unrolled if training_iters else coords_true,
                vis_true = occluded_true_unrolled if training_iters else occluded_true,
                scale = scale_b,
                device = device
            )
            
            # if coords_pred_2d is not None:
            #     occluded_coords_loss_2d = self.mae_loss_occluded_coords_2d(
            #         coords_pred = coords_pred_2d,
            #         coords_true = coords_true_2d,
            #         vis_true = (1.0 - vis_true_cams),
            #         device = device
            #     )            
            # else:
            #     occluded_coords_loss_2d = torch.tensor(0.0, device=device)
        else:
            occluded_coords_loss = torch.tensor(0.0, device=device)
            # occluded_coords_loss_2d = torch.tensor(0.0, device=device)

        if 'feature_planes_levels' in outputs and self.feature_loss_weight > 0:
            feature_loss, bad_feature_loss = self.feature_loss(
                    model = model, 
                    coords_true = coords_true, 
                    feature_planes_levels = outputs['feature_planes_levels'], 
                    cgroup = cgroup,
                    device = device
            )
        else:
            feature_loss = torch.tensor(0.0, device=device)
            bad_feature_loss = torch.tensor(0.0, device=device)

        losses = [
            coords_loss, occluded_coords_loss,
            coords_loss_direct,
            coords_loss_rays,
            coords_loss_triangulate,
            # vis_loss, # replaced by vis_loss_cams
            vis_loss_cams,
            conf_loss,
            conf_loss_2d,
            coords_loss_2d, coords_loss_depth,
            # occluded_coords_loss_2d, # too crazy
            feature_loss, bad_feature_loss
        ]
        
        # total_loss = 0
        # for loss in losses: 
        #     if torch.isfinite(loss).item(): 
        #         total_loss += loss
        losses = torch.stack(losses)
        losses = losses[torch.isfinite(losses)]
        total_loss = losses.sum() / 50.0 # normalize to bring in 0-1 range

        self.loss_history['coords_loss'].append(coords_loss.item())
        self.loss_history['occluded_coords_loss'].append(occluded_coords_loss.item())

        self.loss_history['3d_direct'].append(coords_loss_direct.item())
        self.loss_history['3d_rays'].append(coords_loss_rays.item())
        self.loss_history['3d_triangulate'].append(coords_loss_triangulate.item())
        
        self.loss_history['vis_loss'].append(vis_loss.item())
        self.loss_history['vis_2d_loss'].append(vis_loss_cams.item())
        self.loss_history['conf_loss'].append(conf_loss.item())
        self.loss_history['conf_2d_loss'].append(conf_loss_2d.item())
        self.loss_history['total_loss'].append(total_loss.item())

        # Record 2D and depth losses if they were computed
        self.loss_history['2d_loss'].append(coords_loss_2d.item())
        # self.loss_history['2d_occluded_loss'].append(occluded_coords_loss_2d.item())
        self.loss_history['depth_loss'].append(coords_loss_depth.item())

        # feature loss
        self.loss_history['feature_loss'].append(feature_loss.item())
        self.loss_history['bad_feature_loss'].append(bad_feature_loss.item())

        self.loss_history['cube_scale'].append(scale.detach().mean().item())
        
        return total_loss


class BCELossVis(nn.Module): 
    
    def __init__(self, gamma = 0.8, weight = 1):
        super().__init__()

        self.gamma = gamma 
        self.weight = weight

    def _compute_loss(self, vis_pred, vis_true): 

        loss = F.binary_cross_entropy_with_logits(
            vis_pred, 
            vis_true.float(), 
            reduction = 'mean')

        return loss 

    def forward(self, vis_pred, vis_true, device = None):

        # don't compute if the weight is 0
        if self.weight == 0: 
            return torch.tensor(float('nan'), device = device)

        if isinstance(vis_pred, torch.Tensor): 
            total_loss = self._compute_loss(vis_pred, vis_true)
            return self.weight * total_loss 

        n_strides = len(vis_pred)
        n_iters = len(vis_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(vis_pred[i][j], vis_true[i])

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss


class BCELossConf(nn.Module): 

    def __init__(self, gamma = 0.8, pixel_thresh = 12, weight = 1): 
        super().__init__()

        self.gamma = gamma 
        self.pixel_thresh = pixel_thresh
        self.weight = weight

    def _compute_loss(self, conf_pred, coords_pred, coords_true, vis_true, scale=1): 

        dist = torch.sum((coords_pred - coords_true) ** 2, dim = -1) ** 0.5
        mask = (dist <= self.pixel_thresh * scale).float().unsqueeze(dim = -1)

        loss = F.binary_cross_entropy_with_logits(
            conf_pred, 
            mask, 
            reduction = 'mean')

        loss = torch.nanmean(loss * vis_true)

        return loss 

    def forward(self, conf_pred, coords_pred, coords_true, vis_true, scale=1, device = None): 

        # don't compute if the weight is 0
        if self.weight == 0: 
            return torch.tensor(float('nan'), device = device)
 
        if isinstance(coords_pred, torch.Tensor): 
            total_loss = self._compute_loss(conf_pred, coords_pred, coords_true, vis_true, scale)
            return self.weight * total_loss 

        n_strides = len(conf_pred)
        n_iters = len(conf_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides): 
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(
                    conf_pred[i][j], 
                    coords_pred[i][j], 
                    coords_true[i], 
                    vis_true[i],
                    scale)

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss 

class WeightedMAELoss(nn.Module):

    def __init__(self, gamma = 0.8, delta = 6, use_huber_loss = False, weight = 1):
        super().__init__()

        self.gamma = gamma
        self.delta = delta
        self.use_huber_loss = use_huber_loss
        self.weight = weight

    def huber_loss(self, coords_pred, coords_true):

        diff = coords_pred - coords_true
        mask = torch.abs(diff) <= self.delta
        
        loss_masked = 0.5 * ((diff * mask) ** 2) 
        loss_unmasked = ~mask * self.delta * (torch.abs(diff * ~mask) - 0.5 * self.delta)

        total_loss = loss_masked + loss_unmasked

        return total_loss
    
    def _compute_loss(self, coords_pred, coords_true, vis_true, scale=1.0):

        if self.use_huber_loss:
            loss = self.huber_loss(coords_pred, coords_true)
        else:
            loss = torch.abs(coords_pred - coords_true)

        loss = torch.nanmean((loss / scale) * vis_true)

        return loss

    def forward(self, coords_pred, coords_true, vis_true, scale=1.0, device=None):

        # don't compute if the weight is 0
        if self.weight == 0:
            return torch.tensor(float('nan'), device = device)

        if isinstance(coords_pred, torch.Tensor):
            total_loss = self._compute_loss(coords_pred, coords_true, vis_true, scale)
            return self.weight * total_loss

        n_strides = len(coords_pred)
        n_iters = len(coords_pred[0])

        losses = torch.ones((n_strides, n_iters), device = device)
        weights = self.gamma ** torch.arange(n_iters, device = device).flip(0)

        for i in range(n_strides):
            for j in range(n_iters):
                losses[i, j] = self._compute_loss(coords_pred[i][j], coords_true[i], vis_true[i], scale)

        total_loss = self.weight * torch.nanmean(weights * torch.nanmean(losses, axis = 0), axis = 0)

        return total_loss


class FeatureLoss(nn.Module): 

    def __init__(self, weight):
        super().__init__()

        self.weight = weight

    def forward(self, model, coords_true, feature_planes_levels, cgroup, device = None): 

        # don't compute if the weight is 0
        if self.weight == 0: 
            feature_loss = torch.tensor(float('nan'), device = device)
            bad_feature_loss = torch.tensor(float('nan'), device = device)
            return feature_loss, bad_feature_loss

        feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords_true, 
            camera_group = cgroup)
        
        b, s, n, r = coords_true.shape
        coords_flat = rearrange(coords_true, 'b s n r -> (b s n) r')
        ixs_perm = torch.randperm(coords_flat.shape[0])
        coords_shuffle = rearrange(coords_flat[ixs_perm], '(b s n) r -> b s n r',
            b = b, s = s, n = n)

        bad_feature_loss = model.get_feature_loss(
            feature_planes_levels = feature_planes_levels, 
            coords_full = coords_shuffle, 
            camera_group = cgroup)
        bad_feature_loss  = 1 - bad_feature_loss
        
        feature_loss *= self.weight
        bad_feature_loss *= self.weight
    
        return feature_loss, bad_feature_loss


def get_vis_true(coords):

    # vis = ~torch.isnan(torch.einsum('bsnr->bsn', coords))
    # vis = rearrange(vis, 'b s n -> b s n 1')

    vis = torch.isfinite(coords[..., 0])[..., None]

    return vis 


def unroll_batch(coords, vis, stride = 8, stride_overlap = 4): 

    T = coords.shape[1]
    stride_remainder = stride - stride_overlap
    n_windows = T // (stride_remainder)
    
    coords_unrolled = []
    vis_unrolled = []
    occluded_unrolled = []

    for i in range(n_windows): 

        ix = stride_remainder * i
        coords_subset = coords[:, ix:ix + stride, ...]
        vis_subset = vis[:, ix:ix + stride, ...]

        coords_unrolled.append(coords_subset)
        vis_unrolled.append(vis_subset)
        occluded_unrolled.append(~vis_subset)

    return coords_unrolled, vis_unrolled, occluded_unrolled
