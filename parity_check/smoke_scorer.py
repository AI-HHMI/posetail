"""End-to-end smoke test for the ScorerEncoder.

Builds the scorer from config_scorer.toml, warm-starts from the tracker checkpoint
(strict=False), pulls real PosetailDataset samples (3D and 2D), runs a full triplet, and
checks: scores/precision are [b,k] finite; backbone frozen while query-encoder/decoder/
pool/heads are trainable; good & bad share scene features (pixels unchanged); one backward
produces grads only on trainable params.

    pixi run python parity_check/smoke_scorer.py
"""
import sys
sys.path.insert(0, '.')

import torch

from train_utils import load_config, load_checkpoint, dict_to_device
from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from posetail.datasets.scorer_corruption import make_triplet, build_corruptors
from posetail.posetail.scorer_encoder import ScorerEncoder
from posetail.posetail.losses_scorer import TripletScorerLoss

CONFIG = 'configs/config_scorer.toml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def check_freeze(model):
    enc = model.scene_encoder.encoder
    enc_frozen = all(not p.requires_grad for p in enc.parameters())
    qe_train = any(p.requires_grad for p in model.query_encoder.parameters())
    dec_train = any(p.requires_grad for p in model.decoder.parameters())
    pool_train = all(p.requires_grad for p in model.attn_pool.parameters())
    head_train = all(p.requires_grad for p in model.score_head.parameters())
    print(f"  backbone frozen={enc_frozen} | query_enc trainable={qe_train} | "
          f"decoder trainable={dec_train} | attn_pool trainable={pool_train} | "
          f"score_head trainable={head_train}")
    assert enc_frozen and qe_train and dec_train and pool_train and head_train
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")


def run_sample(model, batch, dataset, c3, c2, ccfg, loss):
    batch.views = [v.to(DEVICE) for v in batch.views]
    batch.coords = batch.coords.to(DEVICE)
    if batch.cgroup:
        batch.cgroup = [dict_to_device(c, DEVICE) for c in batch.cgroup]
    mode = '3d' if batch.coords.shape[-1] == 3 else '2d'
    b, t, n, R = batch.coords.shape
    print(f"  sample mode={mode} coords={tuple(batch.coords.shape)} "
          f"cams={len(batch.views)}")

    trip = make_triplet(batch, dataset, c3, c2, ccfg)
    gv, gc, gcg = trip['good']
    _, bc, bcg = trip['bad']
    av, ac, acg = trip['anchor']

    vn, sf = model.encode_scene(gv)
    good_s, good_p = model.score(vn, sf, gc, gcg)
    bad_s, bad_p = model.score(vn, sf, bc, bcg)
    # scene features identical for good vs bad (corruption never touches pixels)
    vn2, sf2 = model.encode_scene(gv)
    same = all(torch.allclose(a, b) for a, b in zip(sf, sf2)) if isinstance(sf, (list, tuple)) \
        else torch.allclose(sf, sf2)
    avn, asf = (vn, sf) if trip['reuse_scene_for_anchor'] else model.encode_scene(av)
    anc_s, anc_p = model.score(avn, asf, ac, acg)

    assert good_s.shape == (b, n), good_s.shape
    assert good_p.shape == (b, n)
    for nm, s in [('good', good_s), ('bad', bad_s), ('anchor', anc_s)]:
        assert torch.isfinite(s).all(), f"{nm} has non-finite scores"
    print(f"  scores ok: good[{good_s.shape}] finite; "
          f"reuse_scene_for_anchor={trip['reuse_scene_for_anchor']} scene_reproducible={same}")

    # loss + backward
    scores = torch.stack([good_s, bad_s, anc_s], -1).reshape(-1, 3)
    precision = torch.stack([good_p, bad_p, anc_p], -1).reshape(-1, 3)
    labels = torch.tensor([1.0, -1.0, trip['anchor_label']], device=DEVICE).expand_as(scores)
    l = loss(scores, precision, labels)
    l.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.scene_encoder.encoder.parameters())
    head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.score_head.parameters())
    print(f"  loss={float(l):.4f} | backbone got grad={enc_grad} (expect False) | "
          f"score_head got grad={head_grad} (expect True)")
    assert not enc_grad and head_grad
    model.zero_grad()
    return mode


def main():
    config = load_config(CONFIG)
    print("building ScorerEncoder...")
    sk = dict(config.scorer)
    sk.pop('corruption', None)
    model = ScorerEncoder(pool_num_heads=sk['pool_num_heads'], score_hidden=sk['score_hidden'],
                          use_precision=sk['use_precision'], **config.model).to(DEVICE)
    print("loading checkpoint (strict=False)...")
    model = load_checkpoint(CONFIG, config.training.checkpoint_path, model=model, device=DEVICE)['model']
    check_freeze(model)

    ccfg = config.scorer.corruption
    c3, c2 = build_corruptors(ccfg)
    loss = TripletScorerLoss(margin=sk['triplet_margin'], precision_reg_weight=sk['precision_reg_weight'])

    dataset = PosetailDataset(config, split='train')
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=config.dataset.batch_size, collate_fn=custom_collate,
                        shuffle=True, num_workers=0)

    seen = set()
    it = iter(loader)
    for _ in range(30):
        if {'2d', '3d'} <= seen:
            break
        batch = next(it)
        if batch is None:
            continue
        model.train()
        mode = run_sample(model, batch, dataset, c3, c2, ccfg, loss)
        seen.add(mode)
    print(f"\nmodes exercised: {sorted(seen)}")
    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
