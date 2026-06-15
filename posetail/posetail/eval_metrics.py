import torch
import numpy as np
from posetail.posetail.losses import get_vis_true


def _sigmoid(x):
    # vis_pred is emitted as a raw logit (max over per-camera logits, see
    # tracker_tapnext.py); convert to a probability so the 0.5 threshold below
    # matches the decision boundary the BCE-with-logits vis loss optimizes toward.
    return 1.0 / (1.0 + np.exp(-x))

def get_eval_metrics(vis_pred, vis_true, coords_pred, 
                     coords_true, thresholds = None, 
                     survival_threshold = 50, prefix = 'eval/'):

    if vis_true is None:
        vis_true = get_vis_true(coords_true) 

    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]
    
    vis_pred = vis_pred.detach().cpu().to(torch.float32).numpy()
    vis_true = vis_true.detach().cpu().numpy().astype(bool)
    coords_pred = coords_pred.detach().cpu().to(torch.float32).numpy()
    coords_true = coords_true.detach().cpu().to(torch.float32).numpy()

    mte = get_mte(coords_pred, coords_true, vis_true)

    occlusion_acc = get_occlusion_accuracy(vis_pred, vis_true)

    mpjpe = get_mpjpe(coords_pred, coords_true, vis_pred, vis_true)

    delta_x_avg, delta_x_dict = get_delta_x_avg(coords_pred, 
        coords_true, vis_true, thresholds = thresholds)
    
    survival_rate = get_survival_rate(coords_pred, 
        coords_true, vis_true, threshold = survival_threshold)
    
    avg_jaccard, avg_jaccard_dict = get_average_jaccard(coords_pred, 
        coords_true, vis_pred, vis_true, thresholds=thresholds)

    metrics = {f'{prefix}mte': mte, 
               f'{prefix}delta_x_avg': delta_x_avg, 
               f'{prefix}occlusion_acc': occlusion_acc,
               f'{prefix}avg_jaccard': avg_jaccard,
               f'{prefix}survival_rate': survival_rate, 
               f'{prefix}mpjpe': mpjpe}

    # add per-threshold metrics for delta_x and avg_jaccard
    for k, v in delta_x_dict.items(): 
        metrics[f'{prefix}delta_x_{k:.3g}'] = v

    for k, v in avg_jaccard_dict.items():
        metrics[f'{prefix}jaccard_{k:.3g}'] = v

    return metrics


def get_mte(coords_pred, coords_true, vis_true):
    '''
    Median Trajectory Error: per-track median L2 over visible timesteps,
    then mean across tracks (MVTracker definition).

    parameters:
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_true:    B, T, N, 1  bool
    '''
    vis  = np.squeeze(vis_true.astype(bool), axis=-1)           # B, T, N
    dist = np.linalg.norm(coords_pred - coords_true, axis=-1)   # B, T, N
    B, T, N = dist.shape

    track_mtes = []
    for b in range(B):
        for n in range(N):
            visible = vis[b, :, n]
            if not np.any(visible):
                continue
            track_mtes.append(np.median(dist[b, visible, n]))

    if len(track_mtes) == 0:
        return np.nan
    
    return float(np.mean(track_mtes))


def get_occlusion_accuracy(vis_pred, vis_true): 
    ''' 
    parameters:
        vis_pred: B, T, N, 1
        vis_true: B, T, N, 1

    returns: 
        occlusion_acc (float)
    '''

    occlusion_pred = _sigmoid(vis_pred) < 0.5
    occlusion_true = ~vis_true

    occlusion_acc = np.mean(occlusion_pred == occlusion_true)

    return occlusion_acc


def get_delta_x(coords_pred, coords_true, vis_true, threshold):
    ''' 
    for points that are visible, measures the fraction of 
    points that are within a distance delta pixels from 
    their ground truth

    parameters: 
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_true: B, T, N, 1

    ''' 

    within_thresh = np.sum((coords_pred - coords_true) ** 2, axis=-1) < (threshold ** 2)
    good = within_thresh[..., None] & vis_true
    delta_x = np.sum(good, axis = (0, 1, 2)) / np.sum(vis_true)

    return delta_x 


def get_delta_x_avg(coords_pred, coords_true, 
                    vis_true, thresholds = None): 

    delta_xs = []
    
    # initialize to default values
    if thresholds is None: 
        thresholds = [1, 2, 4, 8, 16]

    for thresh in thresholds:

        delta_x = get_delta_x(
            coords_pred = coords_pred, 
            coords_true = coords_true, 
            vis_true = vis_true, 
            threshold = thresh)

        delta_xs.append(delta_x)

    delta_x_avg = np.mean(delta_xs)
    delta_x_dict = dict(zip(thresholds, delta_xs))

    return delta_x_avg, delta_x_dict 


def get_mpjpe(coords_pred, coords_true, vis_pred, vis_true, eps = 1e-8):
    ''' 
    calculates the mean per joint position error for all 
    keypoints (pixels for 2d, mm for 3d) and timepoints
    in a batch

    parameters: 
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_pred: B, T, N, 1
        vis_true: B, T, N, 1
        eps: a small constant to prevent divide by zero errors
    '''

    # mask = (vis_pred > 0.5) & vis_true
    mask = vis_true
    valid_mask = np.squeeze(mask, axis = -1)

    error_per_kpt = np.linalg.norm(coords_pred - coords_true, axis = -1, keepdims = False)
    error = np.nansum(error_per_kpt, axis = -1) / (np.sum(valid_mask, axis = -1) + eps)
    
    mask = np.sum(valid_mask, axis = -1) == 0
    error[mask] = np.nan

    mpjpe = np.nanmean(error)

    return mpjpe


def get_survival_rate(coords_pred, coords_true, vis_true, threshold = 50):
    '''
    Average frames-until-failure as a ratio of video length, where failure
    is defined as L2 distance exceeding `threshold` on a visible frame.
    Tracks that never fail contribute a ratio of 1.0.
    Only tracks with at least one visible GT frame are counted.

    parameters:
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_true:    B, T, N, 1  bool
        threshold:   failure distance (default 50px at 256x256 resolution)

    returns:
        survival_rate (float) in [0, 1]
    '''
    vis = np.squeeze(vis_true.astype(bool), axis=-1)           # B, T, N
    dist = np.linalg.norm(coords_pred - coords_true, axis=-1)  # B, T, N
    B, T, N = dist.shape

    survival_ratios = []
    for b in range(B):
        for n in range(N):
            if not np.any(vis[b, :, n]):
                continue  # no visible GT frames, skip
            # failure: visible frame where L2 exceeds threshold
            failed_frames = np.where((dist[b, :, n] > threshold) & vis[b, :, n])[0]
            if len(failed_frames) == 0:
                frames_survived = T
            else:
                frames_survived = int(failed_frames[0])  # frames before first failure
            survival_ratios.append(frames_survived / T)

    if len(survival_ratios) == 0:
        return np.nan
    return float(np.mean(survival_ratios))


def get_average_jaccard(coords_pred, coords_true, vis_pred, vis_true, thresholds=None):
    '''
    Average Jaccard (MVTracker / TAP-Vid definition).

    Per track i, per threshold x:
        numerator   = sum_t( v_t * v̂_t * α_t )
        denominator = sum_t( v_t + (1−v_t)*v̂_t + v_t*v̂_t*(1−α_t) )
        AJ^i_x = numerator / denominator

    "Both occluded" frames contribute 0 to both — excluded from the ratio.
    AJ per threshold = mean over tracks. AJ = mean over thresholds.

    parameters:
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_pred:    B, T, N, 1  float in [0, 1]
        vis_true:    B, T, N, 1  bool
    '''
    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]

    gt_vis  = np.squeeze(vis_true.astype(bool),  axis=-1)   # B, T, N
    pred_vis = np.squeeze(_sigmoid(vis_pred) > 0.5, axis=-1) # B, T, N
    dist    = np.linalg.norm(coords_pred - coords_true, axis=-1)  # B, T, N
    B, T, N = dist.shape

    per_thresh = {t: [] for t in thresholds}

    for b in range(B):
        for n in range(N):
            vt = gt_vis[b, :, n]    # (T,)
            vh = pred_vis[b, :, n]  # (T,)
            d  = dist[b, :, n]      # (T,)

            for thresh in thresholds:
                alpha = d < thresh  # (T,) bool

                tp    = np.sum(vt & vh & alpha)
                denom = np.sum(vt) + np.sum(~vt & vh) + np.sum(vt & vh & ~alpha)

                if denom == 0:
                    continue  # all frames both occluded — skip track at this threshold
                per_thresh[thresh].append(float(tp / denom))

    jaccard_dict = {
        t: float(np.mean(vals)) if vals else np.nan
        for t, vals in per_thresh.items()
    }
    aj = float(np.nanmean(list(jaccard_dict.values())))
    return aj, jaccard_dict
