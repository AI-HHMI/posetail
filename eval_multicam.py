#!/usr/bin/env python3
"""Ground-truth eval of a finetuned encoder checkpoint on the multi-view benchmark sets
(kubric-multiview, cmupanoptic) at a configurable camera count. The training-time val uses
cams_to_sample=5, which under-estimates multi-view 3D accuracy; the real benchmark uses many
more cams (kubric=10, cmu=31). This reports mte per dataset at the requested cam count, using
the schedulefree EVAL weights (optimizer.eval()).

Usage:
  CUDA_VISIBLE_DEVICES=0 pixi run python eval_multicam.py \
      --config configs/config_encoder_gridresid_finetune_test.toml \
      --checkpoint <ref final .pth> --cams 8 --max-batches 80 --datasets kubric-multiview cmupanoptic
"""
import argparse, warnings, collections
warnings.filterwarnings('ignore')
import numpy as np, torch
from torch.utils.data import DataLoader

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from posetail.posetail.eval_metrics import get_eval_metrics
from posetail.posetail.losses import get_vis_true, normalize_by_mean_depth
from train_utils import load_config, load_checkpoint, set_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cams', type=int, default=8)
    ap.add_argument('--max-batches', type=int, default=80)
    ap.add_argument('--datasets', nargs='+', default=['kubric-multiview', 'cmupanoptic'])
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--kpts', type=int, default=256)
    ap.add_argument('--split', default='val')
    ap.add_argument('--decompose', action='store_true', help='split error into depth(radial) vs lateral(tangential)')
    ap.add_argument('--image-size', type=int, default=None, help='override input resolution (model + dataset min/max res)')
    ap.add_argument('--n-frames', type=int, default=None, help='override clip length (dataset only; tests pos_embed temporal interp)')
    ap.add_argument('--no-opt', action='store_true', help='skip schedulefree optimizer eval-swap (use raw model_state; for res-changed smoke tests)')
    ap.add_argument('--wise-base', default=None, help='WiSE-FT: pretrained checkpoint to interpolate the finetuned weights TOWARD (weight-space ensembling, OOD robustness)')
    ap.add_argument('--wise-alpha', type=float, default=1.0, help='WiSE-FT blend theta=alpha*finetuned+(1-alpha)*base; 1.0=pure finetuned (default, no blend). Sweep ~0.5-0.9.')
    args = ap.parse_args()
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)
    # override the eval protocol on BOTH val and test sub-configs: more cameras, more query
    # points, ONLY the target datasets.
    for sub in ('val', 'test'):
        if sub in config.dataset:
            config.dataset[sub]['cams_to_sample'] = args.cams
            config.dataset[sub]['kpts_to_sample'] = args.kpts
            if args.image_size:
                config.dataset[sub]['min_res'] = args.image_size
                config.dataset[sub]['max_res'] = args.image_size
            if args.n_frames:
                config.dataset[sub]['n_frames'] = args.n_frames
    if args.image_size:
        config.model['image_size'] = args.image_size
    ALL = ['3dpop','3dzef','allen-mouse','branson-fly','cmupanoptic','cmupanoptic_3dgs',
           'dex_ycb','johnson-fly','johnson-mouse','kubric-multiview',
           'point-odyssey-animal-sub256','point-odyssey-human','rat7m','rat-city',
           'ravan-fish-sim','sober-zebrafinch','tuthill-fly']
    config.dataset['datasets_to_exclude'] = [d for d in ALL if d not in args.datasets]

    # build model, load ckpt, and (unless --no-opt) swap to the schedule-free EVAL weights.
    from posetail.posetail.tracker_encoder import TrackerEncoder
    model = TrackerEncoder(**config.model).to(device)
    load_checkpoint(args.config, args.checkpoint, model=model, device=device,
                    eval_weights=not args.no_opt)
    if args.wise_base and args.wise_alpha < 1.0:
        # WiSE-FT (Wortsman et al., CVPR'22): theta = alpha*finetuned + (1-alpha)*pretrained.
        # A post-hoc weight-space ensemble that recovers OOD robustness lost to finetuning -- free,
        # no retraining. Sweep alpha on val. alpha=1.0 (default) skips this entirely.
        base_ckpt = torch.load(args.wise_base, map_location=device)
        base_sd = base_ckpt.get('model_state', base_ckpt.get('model_state_dict', base_ckpt))
        cur = model.state_dict(); a = args.wise_alpha; n = 0
        for k, v in cur.items():
            if k in base_sd and base_sd[k].shape == v.shape and v.is_floating_point():
                cur[k] = a * v + (1.0 - a) * base_sd[k].to(device=device, dtype=v.dtype); n += 1
        model.load_state_dict(cur)
        print(f"WiSE-FT: blended {n} tensors with base {args.wise_base.split('/')[-1]} at alpha={a}")
    model.eval()

    val_ds = PosetailDataset(config, split=args.split)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(val_ds, batch_size=config.dataset.batch_size, collate_fn=custom_collate,
                        shuffle=True, generator=gen, num_workers=8, pin_memory=True)

    per = collections.defaultdict(list)
    per_rad, per_tang = [], []
    seen = collections.Counter()
    for j, batch in enumerate(loader):
        if sum(seen.values()) >= args.max_batches * len(args.datasets):
            break
        ds = batch.sample_info.get('dataset', 'unknown')
        if ds not in args.datasets or seen[ds] >= args.max_batches:
            continue
        views = [v.to(device) for v in batch.views]
        coords = batch.coords.to(device)
        vis = batch.vis
        cgroup = batch.cgroup
        query_times = batch.query_times
        if isinstance(query_times, torch.Tensor):
            query_times = query_times.to(device)
        elif isinstance(query_times, (list, tuple)):
            query_times = [q.to(device) if isinstance(q, torch.Tensor) else q for q in query_times]
        p2d = batch.p2d.to(device) if batch.p2d is not None else None
        if coords.shape[-1] != 3:
            continue
        if p2d is None:
            qc = coords[:, query_times[0], torch.arange(len(query_times[0]))]
        else:
            qc = p2d[:, 0, query_times[0], torch.arange(len(query_times[0]))]
        if cgroup:
            cgroup = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in c.items()} for c in cgroup]
        with torch.no_grad():
            out = model(views=list(views), coords=qc, query_times=query_times, camera_group=cgroup)
        coords_pred = out['coords_pred']; vis_pred = out['vis_pred']
        if p2d is not None:
            C = cgroup[0]['center']
            vfn = vis.to(coords.device) if vis is not None else get_vis_true(coords)
            _, pmd = normalize_by_mean_depth(coords_pred, vfn, C)
            _, tmd = normalize_by_mean_depth(coords, vfn, C)
            coords_pred = C + (coords_pred - C) * (tmd / pmd)
        m = get_eval_metrics(vis_pred=vis_pred, vis_true=vis, coords_pred=coords_pred,
                             coords_true=coords, prefix='')
        if m.get('mte') is not None and m['mte'] == m['mte']:
            per[ds].append(m['mte'])
            seen[ds] += 1
            if args.decompose and cgroup:
                # decompose error into radial (along cam0 viewing ray = depth) vs tangential
                # (lateral). cam0 center is the viewpoint.
                cc = cgroup[0]['center'].reshape(1, 1, 1, 3).to(coords.device).float()
                cp = coords_pred.float(); ct = coords.float()
                ray = ct - cc                                  # viewpoint->true point
                ray = ray / (ray.norm(dim=-1, keepdim=True) + 1e-6)
                err = cp - ct                                  # (B,T,N,3)
                rad = (err * ray).sum(-1)                       # signed depth-axis error
                tang = (err - rad.unsqueeze(-1) * ray).norm(dim=-1)
                vmask = vis.squeeze(-1).bool().to(coords.device) if vis is not None else torch.isfinite(ct[...,0])
                vmask = vmask & torch.isfinite(rad) & torch.isfinite(tang)
                if vmask.any():
                    per_rad.append(rad[vmask].abs().median().item())
                    per_tang.append(tang[vmask].median().item())

    print(f"\n=== multicam eval  cams={args.cams}  ckpt={args.checkpoint.split('/')[-1]} ===")
    for ds in args.datasets:
        v = per.get(ds, [])
        if v:
            print(f"  {ds:<20} mte: mean={np.mean(v):.4f} median={np.median(v):.4f} "
                  f"p25={np.percentile(v,25):.4f} n={len(v)}")
        else:
            print(f"  {ds:<20} (no batches seen)")
    if args.decompose and per_rad:
        print(f"  ERROR DECOMP (median per clip): depth(radial)={np.median(per_rad):.4f}  "
              f"lateral(tangential)={np.median(per_tang):.4f}  ratio_depth/lat={np.median(per_rad)/max(np.median(per_tang),1e-6):.2f}")


if __name__ == '__main__':
    main()
