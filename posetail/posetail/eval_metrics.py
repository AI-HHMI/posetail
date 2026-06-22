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


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    return np.asarray(x)


def get_metrics_by_horizon(coords_pred, coords_true, vis_true,
                           query_times=None, horizons=None, thresholds=None,
                           prefix='eval/', emit_all=False):
    '''
    Drift-vs-horizon: error and delta_x bucketed by temporal distance from the
    source/query frame, h = |t - t_src|. The aggregate delta_x_avg averages over
    all frames and HIDES drift; this exposes it.

    parameters:
        coords_pred: B, T, N, R
        coords_true: B, T, N, R
        vis_true:    B, T, N, 1  (bool; None -> finite(coords_true))
        query_times: B, N  source-frame index per track (None -> all 0, i.e. h=t)
        horizons:    list of integer horizons to report (default 1,2,4,8,16,24,32
                     clipped to < T)
        thresholds:  delta_x pixel/metric thresholds (default 1,2,4,8,16)
        emit_all:    when True, always emit every requested horizon key (NaN for
                     empty buckets) and mte_fwd/mte_bwd, so the key set is stable
                     across batches (required by average_metrics, which assumes
                     uniform keys). When False, empty buckets are skipped.

    returns a flat dict for logging:
        {prefix}mte_h{h}      = mean L2 over visible points at horizon h
        {prefix}delta_x_{x}_h{h}
        {prefix}n_h{h}        = #points in that bucket
        {prefix}mte_fwd / mte_bwd = mean L2 for h>0 vs h<0 (forward/backward asym.)
    '''
    cp = _to_np(coords_pred)
    ct = _to_np(coords_true)
    vt = (_to_np(vis_true) if vis_true is not None
          else np.isfinite(ct[..., :1])).astype(bool)
    B, T, N, _ = cp.shape

    if query_times is None:
        q = np.zeros((B, N), dtype=np.int64)
    else:
        q = _to_np(query_times).astype(np.int64).reshape(B, N)
    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]
    if horizons is None:
        horizons = [h for h in (1, 2, 4, 8, 16, 24, 32) if h < T]

    horizon = np.arange(T)[None, :, None] - q[:, None, :]   # B, T, N  (signed)
    absh = np.abs(horizon)
    vis = vt[..., 0]                                         # B, T, N
    err = np.linalg.norm(cp - ct, axis=-1)                  # B, T, N
    finite = np.isfinite(err)

    metrics = {}
    for h in horizons:
        sel = (absh == h) & vis & finite
        nsel = int(sel.sum())
        if nsel == 0 and not emit_all:
            continue
        e = err[sel]
        metrics[f'{prefix}mte_h{h}'] = float(np.mean(e)) if nsel else float('nan')
        for thr in thresholds:
            metrics[f'{prefix}delta_x_{thr:.3g}_h{h}'] = (
                float(np.mean(e < thr)) if nsel else float('nan'))
        metrics[f'{prefix}n_h{h}'] = nsel

    fwd = (horizon > 0) & vis & finite
    bwd = (horizon < 0) & vis & finite
    if fwd.any() or emit_all:
        metrics[f'{prefix}mte_fwd'] = float(np.mean(err[fwd])) if fwd.any() else float('nan')
    if bwd.any() or emit_all:
        metrics[f'{prefix}mte_bwd'] = float(np.mean(err[bwd])) if bwd.any() else float('nan')
    return metrics


def get_metrics_by_motion(coords_pred, coords_true, vis_true, query_times=None,
                          cube_scale=None, prefix='eval/'):
    '''Error binned by each point's DISPLACEMENT from its query frame, normalized by
    cube_scale (world-units-per-pixel) so "fast motion" is in pixel-equivalent units and
    comparable across datasets. This is the watchable "how good on fast motion" signal:
    mte_mo_{slow,med,fast,vfast} (median px error). Bins (px displacement-from-query):
    slow <4, med 4-16, fast 16-64, vfast >=64.

    cube_scale: (B,) world units per pixel (median over cameras); None -> 1 (raw units).
    Keys are STABLE (NaN for empty bins) so average_metrics can aggregate them.'''
    cp = _to_np(coords_pred)
    ct = _to_np(coords_true)
    vt = (_to_np(vis_true) if vis_true is not None else np.isfinite(ct[..., :1])).astype(bool)
    B, T, N, R = cp.shape
    q = (np.zeros((B, N), np.int64) if query_times is None
         else _to_np(query_times).astype(np.int64).reshape(B, N))
    cs = (_to_np(cube_scale).reshape(B) if cube_scale is not None else np.ones(B))
    cs = np.where(cs > 1e-9, cs, 1.0)[:, None, None]                       # (B,1,1)
    qidx = np.broadcast_to(q[:, None, :, None], (B, 1, N, R))
    qc = np.take_along_axis(ct, qidx, axis=1)                             # (B,1,N,R) GT at query
    disp = np.linalg.norm(ct - qc, axis=-1) / cs                         # (B,T,N) px-equiv
    err = np.linalg.norm(cp - ct, axis=-1) / cs
    vis = vt[..., 0]
    finite = np.isfinite(err) & np.isfinite(disp)
    metrics = {}
    for lo, hi, name in [(0, 4, 'slow'), (4, 16, 'med'), (16, 64, 'fast'), (64, 1e18, 'vfast')]:
        sel = (disp >= lo) & (disp < hi) & vis & finite
        n = int(sel.sum())
        metrics[f'{prefix}mte_mo_{name}'] = float(np.median(err[sel])) if n else float('nan')
    return metrics


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
