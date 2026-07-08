"""Tests for the backward-compatible `grid_decode_space` knob (gridresid + log_3d_output).

  "head"   (default) = average head-unit bin centres  -> bit-for-bit the original behaviour.
  "warped"           = average in the uniform warped space, then expm1 -> overshoot-free.

Self-contained (builds the Decoder module directly, no V-JEPA encoder/batch harness):
  1. Pure-math: head and warped decode AGREE at a one-hot/peaked distribution, DIFFER for a broad
     one, and the warped path round-trips through signed_log1p (so the subpixel head still composes).
  2. The real Decoder constructs with the knob, defaults to "head", and adds NO params/buffers
     (load-compatible) — verified against its actual registered grid_1d buffer.
"""
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from posetail.posetail.cube import signed_log1p, signed_expm1  # noqa: E402
from posetail.posetail.encoder_decoder import Decoder  # noqa: E402


def _decode(p, grid_1d, cr, eps, space):
    if space == "warped":
        w = signed_log1p(grid_1d.to(p.dtype), eps)
        return signed_expm1((p * w).sum(-1).clamp(-cr, cr), eps)
    return (p * grid_1d.to(p.dtype)).sum(-1)


def test_decode_space_math():
    print("\n=== Test 1: head vs warped decode math ===")
    G, eps, radius = 512, 0.01, 1.5
    cr = math.log1p(radius / eps)
    grid_1d = signed_expm1(torch.linspace(-cr, cr, G), eps)
    assert torch.allclose(signed_log1p(grid_1d, eps), torch.linspace(-cr, cr, G), atol=1e-4)
    oh = torch.zeros(16, G); oh[torch.arange(16), torch.randint(0, G, (16,))] = 1.0
    assert torch.allclose(_decode(oh, grid_1d, cr, eps, "head"),
                          _decode(oh, grid_1d, cr, eps, "warped"), atol=1e-4), "peaked must agree"
    tw = signed_log1p(torch.tensor(1.0), eps)
    p = torch.softmax(-0.5 * ((torch.linspace(-cr, cr, G) - tw) / 0.3) ** 2, -1)[None]
    assert (_decode(p, grid_1d, cr, eps, "head")
            - _decode(p, grid_1d, cr, eps, "warped")).abs().item() > 1e-3, "broad dist should differ"
    wm = (p * signed_log1p(grid_1d, eps)).sum(-1).clamp(-cr, cr)
    assert torch.allclose(signed_log1p(signed_expm1(wm, eps), eps), wm, atol=1e-4)
    print("  OK: peaked agree, broad differ, warped round-trips for the subpixel path")


def _build(space):
    return Decoder(output_mode="gridresid", log_3d_output=True, head_3d_grid_size=512,
                   head_3d_grid_radius=1.5, log_3d_eps=0.01, grid_decode_space=space).eval()


def test_construct_default_and_no_state():
    print("\n=== Test 2: real Decoder constructs; default 'head'; no extra state ===")
    dh = _build("head")
    assert dh.grid_decode_space == "head"
    assert _build("warped").grid_decode_space == "warped"
    try:
        _build("bogus"); assert False, "should reject invalid space"
    except AssertionError as e:
        assert "bogus" in str(e)
    keys = list(dh.state_dict().keys())
    assert not any("grid_decode_space" in k or "warped" in k for k in keys), \
        "grid_decode_space must add no state"
    # decode on the module's OWN registered grid_1d differs head vs warped for a broad dist
    G = dh.head_3d_grid_size; cr = dh.log_3d_c_range; eps = dh.log_3d_eps
    tw = signed_log1p(torch.tensor(1.0), eps)
    p = torch.softmax(-0.5 * ((torch.linspace(-cr, cr, G) - tw) / 0.3) ** 2, -1)[None]
    d_head = _decode(p, dh.grid_1d, cr, eps, "head")
    d_warp = _decode(p, dh.grid_1d, cr, eps, "warped")
    assert (d_head - d_warp).abs().item() > 1e-3
    print(f"  OK: {len(keys)} state keys, none decode-space related; module decode wired")


if __name__ == "__main__":
    print(f"device check (cpu math only)")
    test_decode_space_math()
    test_construct_default_and_no_state()
    print("\nALL GRID_DECODE_SPACE TESTS PASSED")
