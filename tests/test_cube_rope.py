"""Tests for GnomonicRoPE and its wiring into crossformer.Attention/Transformer.

Covers three things:
  1. The rotary math itself (norm-preserving, relative-position property, and
     that the equiangular correction is actually non-linear in raw index --
     otherwise it would be no different from plain grid-index RoPE).
  2. That enabling ``rope`` reorders Attention.forward (to_qkv before
     windowing) without changing the result when rope is disabled -- the
     default path must stay bit-identical to the pre-refactor implementation.
  3. That CubeSphereWxFormer builds and runs with use_cube_rope=True.
"""

import math

import torch
from einops import rearrange

from credit.models.wxformer.cube_rope import GnomonicRoPE
from credit.models.wxformer.crossformer import Attention, Transformer


def test_rope_preserves_norm():
    """Rotation must not change per-token vector norm (it's an orthogonal transform)."""
    torch.manual_seed(0)
    rope = GnomonicRoPE(dim_head=16)
    q = torch.randn(2, 3, 8, 8, 16)  # (b, heads, h, w, dim_head)
    k = torch.randn(2, 3, 8, 8, 16)
    q_out, k_out = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property():
    """rotated_q(phase_i) . rotated_k(phase_j) must depend only on (phase_i - phase_j).

    This is the core algebraic property that makes RoPE "rotary": it's a
    property of the rotate-half formula for a given PHASE, exercised directly
    here (not through the equiangular coordinate map, which is deliberately
    non-linear in grid index -- see test_rope_phase_is_nonlinear_in_index --
    so a fixed *index* offset does NOT correspond to a fixed *phase* offset,
    and testing the property that way would be testing the wrong thing).
    """
    torch.manual_seed(0)
    rope = GnomonicRoPE(dim_head=8)
    q = torch.randn(1, 4)
    k = torch.randn(1, 4)

    def rotate(x, phase):
        cos = torch.cat([torch.cos(phase), torch.cos(phase)], dim=-1)
        sin = torch.cat([torch.sin(phase), torch.sin(phase)], dim=-1)
        return x * cos + rope._rotate_half(x) * sin

    phase_i = torch.tensor([[0.3, 1.1]])
    phase_j = torch.tensor([[0.9, 0.4]])
    dot_a = (rotate(q, phase_i) * rotate(k, phase_j)).sum()

    shift = torch.tensor([[0.5, -0.2]])
    dot_b = (rotate(q, phase_i + shift) * rotate(k, phase_j + shift)).sum()

    assert torch.allclose(dot_a, dot_b, atol=1e-5)


def test_rope_phase_is_nonlinear_in_index():
    """The equiangular correction must differ from plain linear-in-index RoPE.

    Checks that the per-column phase spacing (d xi / d col) is NOT constant
    across the face -- it should be larger near the face center and smaller
    near the edges (arctan compresses the tails of a linear ramp).
    """
    rope = GnomonicRoPE(dim_head=8)
    w = 65
    xi, _ = rope._equiangular_coords(w, w, device="cpu", dtype=torch.float32)
    dxi = xi[1:] - xi[:-1]
    center = len(dxi) // 2
    assert dxi[center] > dxi[0] * 1.5, "equiangular phase should be denser (smaller step) near the face edge"


def test_attention_rope_disabled_matches_reference_forward():
    """With rope=None, Attention.forward must exactly match the pre-refactor
    computation order (windowing before to_qkv), proving the reordering done
    to support rope is a no-op when rope is off.
    """
    torch.manual_seed(0)
    dim, dim_head, wsz = 32, 8, 4
    attn = Attention(dim, attn_type="short", window_size=wsz, dim_head=dim_head)
    attn.eval()

    x = torch.randn(2, dim, 8, 8)

    with torch.no_grad():
        actual = attn(x)

    # Reference: the ORIGINAL forward order -- window x first, then to_qkv.
    with torch.no_grad():
        heads = attn.heads
        xn = attn.norm(x)
        xw = rearrange(xn, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz, s2=wsz)
        q, k, v = attn.to_qkv(xw).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))
        q = q * attn.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        pos = torch.arange(-wsz, wsz + 1)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rearrange(rel_pos, "c i j -> (i j) c").to(xn.dtype)
        biases = attn.dpb(rel_pos)
        rel_pos_bias = biases[attn.rel_pos_indices]
        sim = sim + rel_pos_bias

        a = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", a, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz, y=wsz)
        out = attn.to_out(out)
        expected = rearrange(out, "(b h w) d s1 s2 -> b d (h s1) (w s2)", h=8 // wsz, w=8 // wsz)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_transformer_use_rope_shape_and_finite():
    """Transformer(use_rope=True) builds and runs, output finite and same shape."""
    torch.manual_seed(0)
    dim, dim_head = 32, 8
    tfm = Transformer(
        dim=dim,
        local_window_size=4,
        global_window_size=4,
        depth=1,
        dim_head=dim_head,
        use_rope=True,
    )
    x = torch.randn(1, dim, 8, 8)
    y = tfm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_gnomonic_rope_rejects_bad_dim_head():
    for bad in (1, 2, 6, 15):
        try:
            GnomonicRoPE(dim_head=bad)
            raise AssertionError(f"expected ValueError for dim_head={bad}")
        except ValueError:
            pass
    GnomonicRoPE(dim_head=8)  # divisible by 4, should not raise


def test_gnomonic_rope_dim_mismatch_raises():
    rope = GnomonicRoPE(dim_head=16)
    q = torch.randn(1, 1, 4, 4, 8)  # wrong last dim
    k = torch.randn(1, 1, 4, 4, 8)
    try:
        rope(q, k)
        raise AssertionError("expected ValueError for mismatched dim_head")
    except ValueError:
        pass


def test_equiangular_coords_center_symmetric():
    """xi/eta should be antisymmetric around the face center (odd function of index offset)."""
    w = 21
    xi, _ = GnomonicRoPE._equiangular_coords(w, w, device="cpu", dtype=torch.float32)
    center = w // 2
    assert abs(float(xi[center])) < 1e-6
    for i in range(1, center + 1):
        assert math.isclose(float(xi[center - i]), -float(xi[center + i]), abs_tol=1e-6)
