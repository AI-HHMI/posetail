#!/usr/bin/env python
"""Full test-set evaluation with COMPLETE error metrics, in one script.

For each requested dataset, runs query-first inference (mvtracker/training convention -- the
default in run_inference) on every test trial and computes the full metric set via
get_eval_metrics (MTE, MPJPE, delta-avg + per-threshold, survival, occlusion-acc,
occlusion-acc-percam, AJ + per-threshold). Metrics use the corrected masking: non-finite GT
(cleaned (0,0,-1)->NaN placeholders) and pre-query frames are excluded via the per-point
query_times -- this now includes both occlusion metrics (frame counts, not the visibility gate,
so occluded frames still count). occlusion-acc-percam scores the model's per-camera visibility
logits (vis_pred_2d) against per-camera GT; it is NaN for predictions cached before this column
existed (regenerate with --force to populate it).

Crop setup (best for moving subjects): query-first runs through the windowed tracker, which
RE-CROPS each chunk to follow the subject (n_overlap sets the cadence) and expands the crop by
a causal motion margin (--no-motion-margin to disable). This replaced an earlier static-crop
path that held one crop over the whole clip -- so predictions cached by that old path are
stale; pass --force (or delete the npz cache) to regenerate them through the fixed crop.

Predictions already present in the output folder are REUSED (metrics recomputed from the saved
npz) instead of re-running inference -- so the model / wandb folder / GPU are only needed for
trials that must still be computed. Writes per-dataset metrics.json plus a combined summary.json
and metrics.md (mean + median tables).

Usage:
    # regenerate tables from existing predictions (no model/GPU needed):
    pixi run python scripts/eval_testset_metrics.py \\
        --datasets dex_ycb kubric-multiview cmupanoptic_3dgs --out <preds_dir>

    # compute missing predictions from a wandb run (latest checkpoint by default):
    pixi run python scripts/eval_testset_metrics.py --datasets dex_ycb \\
        --out <preds_dir> --wandb-folder <wandb_run_dir> [--checkpoint STEP] [--force]
"""
import argparse
import glob
import json
import os
import sys
import time

# reduce CUDA fragmentation on dense point sets (cmupanoptic) before torch is imported
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posetail.inference.inference_utils import load_model_from_base_folder, run_inference   # noqa: E402
from posetail.posetail.eval_metrics import get_eval_metrics              # noqa: E402

DEFAULT_WANDB = '/groups/karashchuk/home/karashchukl/results/posetail-finetuning-v3/wandb/run-20260628_134003-39yczenk'
ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v3'

# per-dataset run + scoring settings. thresholds/survival are the mvtracker evaluator_3dpt
# settings; n_views/max_kpts are the inference caps (cmupanoptic_3dgs is dense + many-camera).
SETTINGS = {
    'dex_ycb':          dict(thresholds=[0.01, 0.02, 0.05, 0.10, 0.20], survival=0.10,
                             n_views=None, max_kpts=2500),
    'kubric-multiview': dict(thresholds=[0.05, 0.1, 0.2, 0.4, 0.8],     survival=0.50,
                             n_views=None, max_kpts=2500),
    'cmupanoptic_3dgs': dict(thresholds=[0.05, 0.10, 0.20, 0.40],       survival=1.0,
                             n_views=14, max_kpts=600),
}
METRIC_KEYS = ['mte', 'mpjpe', 'delta_x_avg', 'survival_rate', 'occlusion_acc',
               'occlusion_acc_percam', 'avg_jaccard']


def find_test_trials(dataset):
    # glob follows the v3->v2 symlink for cmupanoptic_3dgs
    trials = sorted(glob.glob(os.path.join(ROOT, dataset, 'test', '*', 'trial')))
    return [t for t in trials if os.path.exists(os.path.join(t, 'pose3d.npz'))]


def eval_outputs(out, thresholds, survival):
    """Compute the full metric set from a predictions dict (an in-memory outputs dict or an
    np.load NpzFile -- both index by key and expose .files/keys)."""
    keys = set(out.files) if hasattr(out, 'files') else set(out)
    cp = torch.as_tensor(np.asarray(out['coords_pred']), dtype=torch.float32)
    ct = torch.as_tensor(np.asarray(out['coords_true']), dtype=torch.float32)
    vp = torch.as_tensor(np.asarray(out['vis_pred']), dtype=torch.float32)
    vt = torch.as_tensor(np.asarray(out['vis_true']), dtype=torch.bool)
    qt = torch.as_tensor(np.asarray(out['query_times']), dtype=torch.long) \
        if 'query_times' in keys else None
    # Per-camera occlusion: model logits vs per-camera GT (both saved by run_inference when the
    # trial has visibility GT). Absent from predictions cached before this was added -> the
    # per-cam metric is left as NaN (regenerate with --force to populate it).
    vp2d = np.asarray(out['vis_pred_2d']) if 'vis_pred_2d' in keys else None
    vtc = np.asarray(out['vis_true_cams']) if 'vis_true_cams' in keys else None
    m = get_eval_metrics(vp, vt, cp, ct, thresholds=thresholds,
                         survival_threshold=survival, prefix='', query_times=qt,
                         vis_pred_2d=vp2d, vis_true_cams=vtc)
    m = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()}
    m.setdefault('occlusion_acc_percam', float('nan'))
    m['n_kpts'] = int(cp.shape[2])
    m['n_late_query'] = int((np.asarray(out['query_times']) > 0).sum()) if 'query_times' in keys else 0
    return m


def fmt_dur(seconds):
    s = int(round(seconds)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def write_report(all_summary, out_dir):
    """Write combined summary.json + metrics.md (mean and median tables) over all datasets."""
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=2)

    def table(agg):
        L = ['| dataset | #tr | ' + ' | '.join(k.replace('_', '-') for k in METRIC_KEYS) + ' |',
             '|---|---|' + '---|' * len(METRIC_KEYS)]
        for ds, s in all_summary.items():
            L.append(f'| {ds} | {s["n_trials"]} | '
                     + ' | '.join(f'{s[f"{k}_{agg}"]:.4f}' for k in METRIC_KEYS) + ' |')
        return '\n'.join(L)

    md = ['# Test-set error metrics (query-first)', '',
          'Per-point first-visible query anchoring; pre-query + non-finite GT masked. Metrics via '
          '`get_eval_metrics`, aggregated over test trials.', '',
          '## Mean over trials', '', table('mean'), '',
          '## Median over trials', '', table('median'), '',
          'Thresholds (world units): '
          + '; '.join(f'{ds} {SETTINGS[ds]["thresholds"]} surv={SETTINGS[ds]["survival"]}'
                      for ds in all_summary if ds in SETTINGS)
          + '. Per-trial breakdowns in `<dataset>/metrics.json`.']
    with open(os.path.join(out_dir, 'metrics.md'), 'w') as f:
        f.write('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True, choices=list(SETTINGS),
                    help='which datasets to evaluate')
    ap.add_argument('--out', required=True,
                    help='predictions/output folder: <out>/<dataset>/<trial>.npz (reused if present)')
    ap.add_argument('--wandb-folder', default=DEFAULT_WANDB,
                    help='wandb run dir (files/config.toml + files/checkpoints/); only loaded '
                         'when a trial needs inference')
    ap.add_argument('--checkpoint', type=int, default=None,
                    help='checkpoint step; default = latest checkpoint in the wandb folder')
    ap.add_argument('--force', action='store_true', help='recompute even if predictions exist')
    ap.add_argument('--n-frames', type=int, default=256, help='cap; uses all available frames')
    ap.add_argument('--max-kpts', type=int, default=None, help='override per-dataset max_kpts')
    ap.add_argument('--n-views', type=int, default=None, help='override per-dataset n_views')
    ap.add_argument('--view-seed', type=int, default=0)
    # Crop setup: query-first runs through the windowed tracker, which re-crops every chunk
    # to FOLLOW the subject (the fix for the static-crop regression). motion-margin expands
    # each chunk's crop by the previous chunk's velocity so fast subjects (dex_ycb) stay
    # in-frame; it is ~inert for slow subjects. Both default ON = best crop setup.
    ap.add_argument('--n-overlap', type=int, default=8,
                    help='re-crop / re-anchor cadence: window advances by (clip_len - n_overlap) '
                         'frames, so a smaller value re-crops more often for very fast subjects')
    ap.add_argument('--no-motion-margin', dest='motion_margin', action='store_false', default=True,
                    help='disable the causal motion-margin crop expansion (default ON)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Lazy model load: only pay for the wandb folder / GPU when a trial must be computed.
    model_state = {}

    def get_model():
        if not model_state:
            device = torch.device(args.device)
            print(f'Loading model from {args.wandb_folder} '
                  f'(checkpoint={"latest" if args.checkpoint is None else args.checkpoint}) ...')
            model, _cfg, cfg_path, ckpt = load_model_from_base_folder(
                args.wandb_folder, checkpoint=args.checkpoint, device=device)
            print(f'Model loaded: {ckpt}')
            model_state.update(model=model, config_path=cfg_path, checkpoint_path=ckpt, device=device)
        return model_state

    # Build the full work list up front so we can show a global counter + an ETA that knows
    # which trials are cached (fast) vs need inference (slow).
    work = []
    for ds in args.datasets:
        cfg = SETTINGS[ds]
        n_views = args.n_views if args.n_views is not None else cfg['n_views']
        max_kpts = args.max_kpts if args.max_kpts is not None else cfg['max_kpts']
        ds_out = os.path.join(args.out, ds); os.makedirs(ds_out, exist_ok=True)
        for tp in find_test_trials(ds):
            tid = os.path.basename(os.path.dirname(tp))
            npz = os.path.join(ds_out, f'{tid}.npz')
            work.append(dict(ds=ds, cfg=cfg, n_views=n_views, max_kpts=max_kpts, tp=tp,
                             tid=tid, npz=npz, cached=os.path.exists(npz) and not args.force))
    total = len(work)
    n_cached = sum(w['cached'] for w in work)
    print(f'\n{total} trials across {len(args.datasets)} dataset(s): '
          f'{n_cached} cached, {total - n_cached} to infer')
    for ds in args.datasets:
        ws = [w for w in work if w['ds'] == ds]
        print(f'  {ds}: {len(ws)} trials (n_views={ws[0]["n_views"] if ws else "-"}, '
              f'max_kpts={ws[0]["max_kpts"] if ws else "-"}, '
              f'{sum(w["cached"] for w in ws)} cached)')

    # Running-mean per-trial durations (seeded with rough priors) -> ETA over remaining work,
    # counting each remaining trial as cached (~instant) or infer (slow) since we know which.
    avg = {'cached': 1.0, 'infer': 90.0}
    seen = {'cached': 0, 'infer': 0}
    results = {ds: [] for ds in args.datasets}
    t_start = time.time()
    for i, w in enumerate(work):
        ds, cfg, tid = w['ds'], w['cfg'], w['tid']
        t0 = time.time()
        try:
            if w['cached']:
                with np.load(w['npz'], allow_pickle=True) as data:
                    m = eval_outputs(data, cfg['thresholds'], cfg['survival'])
                src = 'cached'
            else:
                ms = get_model()
                out = run_inference(
                    model=ms['model'], config_path=ms['config_path'],
                    checkpoint_path=ms['checkpoint_path'], trial_path=w['tp'], start_frame=0,
                    n_frames=args.n_frames, n_overlap=args.n_overlap, per_subject=True,
                    device=ms['device'], max_kpts=w['max_kpts'], n_views=w['n_views'],
                    seed=args.view_seed, outpath=w['npz'], query_first=True,
                    motion_margin=args.motion_margin)
                torch.cuda.empty_cache()
                m = eval_outputs(out, cfg['thresholds'], cfg['survival'])
                src = 'infer'
            m['trial'] = tid; m['status'] = 'ok'
            results[ds].append(m)
            info = (f'mte={m["mte"]:.4f} dx={m["delta_x_avg"]:.3f} surv={m["survival_rate"]:.3f} '
                    f'aj={m["avg_jaccard"]:.3f} K={m["n_kpts"]}')
        except Exception as exc:                    # noqa: BLE001
            results[ds].append({'trial': tid, 'status': 'FAILED', 'err': str(exc)})
            src, info = 'FAILED', str(exc).splitlines()[0][:80]

        dt = time.time() - t0
        kind = 'cached' if w['cached'] else 'infer'
        seen[kind] += 1
        avg[kind] += (dt - avg[kind]) / seen[kind]        # running mean
        eta = sum(avg['cached' if r['cached'] else 'infer'] for r in work[i + 1:])
        print(f'[{i + 1:>3}/{total}] {ds}/{tid} [{src}] {info} | {dt:4.0f}s  '
              f'elapsed {fmt_dur(time.time() - t_start)}  eta {fmt_dur(eta)}', flush=True)

    # per-dataset aggregation + save
    all_summary = {}
    for ds in args.datasets:
        per_trial = results[ds]
        ok = [r for r in per_trial if r['status'] == 'ok']
        summary = {'n_trials': len(ok)}
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in ok], dtype=float)
            summary[f'{k}_mean'] = float(np.nanmean(vals)) if vals.size else float('nan')
            summary[f'{k}_median'] = float(np.nanmedian(vals)) if vals.size else float('nan')
        all_summary[ds] = summary
        with open(os.path.join(args.out, ds, 'metrics.json'), 'w') as f:
            json.dump({'summary': summary, 'per_trial': per_trial}, f, indent=2)
        print(f'--- {ds} over {len(ok)} trials: '
              + '  '.join(f'{k}={summary[f"{k}_mean"]:.4f}' for k in METRIC_KEYS))

    print(f'\nTotal wall time: {fmt_dur(time.time() - t_start)}')
    write_report(all_summary, args.out)
    print('\n========== TEST-SET SUMMARY (mean / median over trials) ==========')
    hdr = f'{"dataset":18s} ' + ' '.join(f'{k:>13s}' for k in METRIC_KEYS)
    print(hdr)
    for ds, s in all_summary.items():
        print(f'{ds:18s} ' + ' '.join(f'{s[f"{k}_mean"]:13.4f}' for k in METRIC_KEYS))
    print(f'\nSaved: {os.path.join(args.out, "summary.json")} + metrics.md')


if __name__ == '__main__':
    main()
