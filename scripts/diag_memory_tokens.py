#!/usr/bin/env python
"""How close are memory-frame tokens to the tokens the model actually reasons in?

Memory frames are encoded in ISOLATION during training (they come from anywhere in the
video, with their own crops, so there is no clip to read them out of), while at inference
they are read straight out of the chunk encode. Same backbone, same patch grid, same tubelet
size -- but an isolated pair has not attended the other 11 timesteps of a clip. That gap is
the one real cost of reusing scene features at inference, and it is measurable.

For a real clip, encode the same frames three ways and score each against the clip route by
same-patch nearest-neighbour top-1 (does a patch's token retrieve ITSELF among all the
clip-route tokens of that tubelet?):

  clip        the reference (trivially 1.0)
  tubelet     the real adjacent pair [t, t+1] encoded alone -- the TRAINING path
  duplicate   frame t duplicated into a 2-frame clip -- previously measured 0.52
  image       the encoder's own image path -- previously measured 0.43

`tubelet` is the number that matters. If it does not clearly beat `duplicate`, the 2-frame
dataset change bought nothing; if it is weak in absolute terms, inference should fall back to
the isolated encode (scene_features=None in build_chunk_memory) so it matches training.

  pixi run python scripts/diag_memory_tokens.py --config configs/config_encoder_memory.toml
"""
import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate  # noqa: E402
from posetail.inference.inference_utils import load_model_from_base_folder      # noqa: E402
from posetail.posetail.tracker_encoder import TrackerEncoder                    # noqa: E402
from posetail.posetail.train_utils import load_config                           # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--base-folder', help='wandb run folder (trained checkpoint)')
    g.add_argument('--config', help='config .toml -> pretrained backbone, untrained head')
    p.add_argument('--split', default='val')
    p.add_argument('--n-samples', type=int, default=6)
    p.add_argument('--device', default=None)
    return p.parse_args()


def nn_top1(a, b):
    """Fraction of rows where b[i]'s nearest neighbour among all of `a` is a[i].

    a, b: (P, D) tokens for the same P patch positions, encoded two different ways.
    """
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    sim = b @ a.t()
    return float((sim.argmax(-1) == torch.arange(a.shape[0], device=a.device)).float().mean())


def off_diag_cos(a, b):
    """Mean cosine between DIFFERENT patches -- the floor any top-1 number sits above."""
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    sim = b @ a.t()
    n = sim.shape[0]
    return float((sim.sum() - sim.diag().sum()) / (n * n - n))


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.base_folder:
        model, config, _, ck = load_model_from_base_folder(
            args.base_folder, checkpoint=None, device=device)
        tag = f'trained: {os.path.basename(ck)}'
    else:
        config = load_config(args.config)
        model = TrackerEncoder(**dict(config.model)).to(device)
        tag = f'pretrained backbone from {os.path.basename(args.config)}'
    model.eval()
    print(tag)

    tub = model.scene_encoder.tubelet_size
    ps = model.scene_encoder.patch_size

    d = config.dataset[args.split]
    d['kpts_to_sample'] = 16
    d['memory_prob'] = 0.0            # only the CLIP is needed here
    d['memory_only_prob'] = 0.0
    d['balance_datasets'] = True
    d['n_samples_per_dataset'] = 1
    d['aug_prob'] = 0.0
    ds = PosetailDataset(config, split=args.split)
    dl = DataLoader(ds, batch_size=1, collate_fn=custom_collate, shuffle=False, num_workers=4)

    rows = []
    with torch.inference_mode():
        for b in dl:
            if b is None or b.views is None:
                continue
            vn = model._normalize_views([v.to(device) for v in b.views], device)[0]  # cam 0
            B, T = vn.shape[:2]
            if T < 2 * tub:
                continue
            gH = vn.shape[-2] // ps
            n_sp = gH * (vn.shape[-1] // ps)

            clip = model.encode_scene([vn])[0]                       # (B, gT*n_sp, D)
            gT = clip.shape[1] // n_sp
            k = gT // 2                                              # a middle tubelet
            t0 = k * tub
            ref = clip[0, k * n_sp:(k + 1) * n_sp]                   # (n_sp, D)

            pair = vn[:, t0:t0 + tub]                                # the REAL adjacent pair
            dup = vn[:, t0:t0 + 1].repeat(1, tub, 1, 1, 1)           # duplicated still
            got = {'tubelet': model.encode_scene([pair])[0][0],
                   'duplicate': model.encode_scene([dup])[0][0]}
            # the encoder's image path: temporal dim == img_temporal_dim_size (1)
            got['image'] = model.encode_scene([vn[:, t0:t0 + 1]])[0][0]

            rows.append({name: (nn_top1(ref, tok), off_diag_cos(ref, tok))
                         for name, tok in got.items()
                         if tok.shape[0] == n_sp})
            if len(rows) >= args.n_samples:
                break

    if not rows:
        sys.exit('no usable clips')
    names = [n for n in ('tubelet', 'duplicate', 'image') if n in rows[0]]
    print(f'\n{len(rows)} clips, {ps}px patches, tubelet_size={tub}')
    print(f'{"route":>10}  {"same-patch NN top-1":>20}  {"off-diagonal cos":>17}')
    for n in names:
        t1 = np.mean([r[n][0] for r in rows])
        od = np.mean([r[n][1] for r in rows])
        print(f'{n:>10}  {t1:>20.3f}  {od:>17.4f}')
    print('\ntubelet is the training path; it must clearly beat duplicate, and its absolute '
          'value is\nthe train/inference gap that scene-feature reuse introduces.')


if __name__ == '__main__':
    main()
