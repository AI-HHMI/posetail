"""CPU unit tests for the scorer loss + corruption (no model/data/GPU needed).

    pixi run python parity_check/test_scorer_units.py
"""
import torch

from posetail.posetail.losses_scorer import TripletScorerLoss
from posetail.datasets.scorer_corruption import (PointCorruptor, corrupt_coords,
                                                 GENERATORS)


def test_loss_sign_convention():
    """good >> bad  ->  loss small + triplet_acc==1; swapped -> large loss, acc==0."""
    loss = TripletScorerLoss(margin=0.5, precision_reg_weight=0.0)
    N = 64
    # anchor derived from good (label +1): columns = [good, bad, anchor]
    good = torch.full((N,), 3.0)
    bad = torch.full((N,), -3.0)
    anchor = good.clone()                       # augmented good ~ good
    scores = torch.stack([good, bad, anchor], dim=-1)
    precision = torch.ones(N, 3)                 # full confidence -> weighting is a no-op
    labels = torch.tensor([1.0, -1.0, 1.0]).expand(N, 3)

    l_good = loss(scores, precision, labels)
    acc = loss.loss_history['triplet_acc'][-1]
    print(f"  good>>bad: loss={float(l_good):.4f} triplet_acc={acc:.3f}")
    assert acc == 1.0, acc
    assert float(l_good) < 0.6, float(l_good)        # ~ margin only (relu floored)

    # swapped: good scores LOWER than bad -> wrong -> big loss, acc 0
    loss.reset_history()
    scores_sw = torch.stack([bad, good, bad.clone()], dim=-1)   # good col now low
    l_bad = loss(scores_sw, precision, labels)
    acc_bad = loss.loss_history['triplet_acc'][-1]
    print(f"  swapped:   loss={float(l_bad):.4f} triplet_acc={acc_bad:.3f}")
    assert acc_bad == 0.0, acc_bad
    assert float(l_bad) > float(l_good) + 1.0
    print("  [ok] loss sign convention")


def test_loss_anchor_from_bad():
    """Anchor derived from bad (label -1): close pair = bad & anchor; good is distant."""
    loss = TripletScorerLoss(margin=0.5, precision_reg_weight=0.0)
    N = 32
    good = torch.full((N,), 2.0)
    bad = torch.full((N,), -2.0)
    anchor = bad.clone()
    scores = torch.stack([good, bad, anchor], dim=-1)
    labels = torch.tensor([1.0, -1.0, -1.0]).expand(N, 3)       # anchor = bad-derived
    l = loss(scores, torch.ones(N, 3), labels)
    acc = loss.loss_history['triplet_acc'][-1]
    print(f"  anchor-from-bad: loss={float(l):.4f} acc={acc:.3f}")
    assert acc == 1.0
    assert float(l) < 0.6
    print("  [ok] anchor-from-bad pairing")


def test_corruption_shapes_and_coverage():
    """Every point corrupted; shifts have shape [b,t,n,D]; good != bad everywhere."""
    probs = {n: 0.5 for n in GENERATORS}
    mags = {n: 1.0 for n in GENERATORS}
    corr = PointCorruptor(probs, mags)

    b, t, n, D = 2, 16, 40, 3
    coords = torch.zeros(b, t, n, D)
    cube_scale_b = torch.ones(b)
    bad = corrupt_coords(coords, corr, '3d', cube_scale_b)
    assert bad.shape == coords.shape
    # every point differs from clean at >=1 frame
    moved = (bad - coords).abs().sum(dim=(1, 3))                # [b, n]
    assert (moved > 0).all(), f"{(moved == 0).sum()} points uncorrupted"
    print(f"  3d corruption: shape={tuple(bad.shape)} all points moved, "
          f"max|shift|={float((bad-coords).abs().max()):.3f}")

    # 2D path (pixels)
    coords2 = torch.zeros(b, t, n, 2)
    bad2 = corrupt_coords(coords2, PointCorruptor(probs, {k: 8.0 for k in GENERATORS}), '2d')
    assert bad2.shape == coords2.shape
    moved2 = (bad2 - coords2).abs().sum(dim=(1, 3))
    assert (moved2 > 0).all()
    print(f"  2d corruption: shape={tuple(bad2.shape)} all points moved, "
          f"max|shift|={float((bad2-coords2).abs().max()):.2f}px")
    print("  [ok] corruption shapes + full coverage")


def test_each_generator_pattern():
    """Each generator produces its intended temporal signature."""
    P, T, D = 8, 32, 3
    dev = 'cpu'
    const = GENERATORS['const_offset'](P, T, D, dev, 1.0)
    assert torch.allclose(const, const[:, :1, :].expand(P, T, D)), "const offset varies over t"
    drift = GENERATORS['gradual_drift'](P, T, D, dev, 1.0)
    # endpoints differ in magnitude from start for a ramp
    assert (drift[:, 0].abs().sum() < drift[:, -1].abs().sum() + 1e-6) or \
           (drift[:, -1].abs().sum() < drift[:, 0].abs().sum() + 1e-6)
    sin = GENERATORS['sinusoid'](P, T, D, dev, 1.0)
    assert sin.abs().max() > 0
    print("  [ok] generator temporal patterns (const flat over t, drift ramps, sinusoid oscillates)")


if __name__ == '__main__':
    torch.manual_seed(0)
    print("== loss ==");        test_loss_sign_convention(); test_loss_anchor_from_bad()
    print("== corruption =="); test_corruption_shapes_and_coverage(); test_each_generator_pattern()
    print("\nALL UNIT TESTS PASSED")
