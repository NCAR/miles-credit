"""Tests for the cubed-sphere WXFormer.

These are self-contained: they synthesize a small SE permutation index for a
"full" cube (every cell is a node) at a couple of resolutions, so they exercise
the grid-agnostic geometry inference without needing the ne120 static files.
"""

import math

import numpy as np
import pytest
import torch

from credit.models.wxformer.cube_sphere_wxformer import (
    NFACE,
    CubeSphereWxFormer,
)


def _write_full_cube_index(tmp_path, edge):
    """SE index for a fully-populated cube of the given edge length."""
    se_index = np.arange(NFACE * edge * edge, dtype=np.int64)
    path = tmp_path / f"se_index_e{edge}.npy"
    np.save(path, se_index)
    return path, len(se_index)


def _make_model(se_index_path, **overrides):
    kwargs = dict(
        se_index_path=str(se_index_path),
        channels=2,
        levels=3,
        surface_channels=4,
        input_only_channels=5,  # keep C_out <= C_in for the residual
        output_only_channels=1,
        frames=1,
        dim=(8, 16, 32, 64),
        depth=(1, 1, 1, 1),
        dim_head=8,
        face_attn_heads=2,
        global_window_size=(6, 6, 6, 6),
        local_window_size=12,
        cross_embed_strides=(2, 2, 2, 2),
        adjacency_path="/does/not/exist.npz",  # disable halo/edge attention
    )
    kwargs.update(overrides)
    return CubeSphereWxFormer(**kwargs)


@pytest.mark.parametrize("edge", [30, 50])
def test_geometry_inferred_from_index(tmp_path, edge):
    """Face edge, padded size and crop offset derive from the index + config."""
    si, _ = _write_full_cube_index(tmp_path, edge)
    model = _make_model(si)

    assert model.nface_edge == edge
    assert model.cube_flat == NFACE * edge * edge
    assert model.total_stride == 16

    # padded size is the smallest multiple of total_stride * lcm(windows) >= edge
    unit = 16 * math.lcm(12, 6)
    assert model.padded_size == math.ceil(edge / unit) * unit
    assert model.padded_size % model.total_stride == 0
    # no halo file -> no offset
    assert model.crop_off == 0


@pytest.mark.parametrize("edge", [30, 50])
def test_forward_shape_roundtrips(tmp_path, edge):
    """Output returns to the SE grid with the output channel count."""
    si, ncol = _write_full_cube_index(tmp_path, edge)
    model = _make_model(si)
    c_in = 2 * 3 + 4 + 5
    c_out = 2 * 3 + 4 + 1

    x = torch.randn(1, c_in, 1, ncol)
    y = model(x)
    assert tuple(y.shape) == (1, c_out, 1, ncol)


def test_zero_init_head_gives_persistence(tmp_path):
    """Zero-init head => delta 0 => output equals the residual exactly.

    This also proves the final step is an unpad/crop, not an interpolation:
    a bilinear resize back to the face size would not reproduce the input.
    """
    si, ncol = _write_full_cube_index(tmp_path, 50)
    model = _make_model(si)
    model.eval()

    c_out = 2 * 3 + 4 + 1
    x = torch.randn(1, 2 * 3 + 4 + 5, 1, ncol)
    with torch.no_grad():
        y = model(x)

    x_res = x[:, :c_out, 0, :]
    assert torch.allclose(y[:, :, 0, :], x_res, atol=0.0)


def test_pad_crop_is_identity(tmp_path):
    """_crop_face exactly inverts _pad_face on the native face region."""
    si, _ = _write_full_cube_index(tmp_path, 50)
    model = _make_model(si)

    face = torch.randn(NFACE, 3, model.nface_edge, model.nface_edge)
    padded = model._pad_face(face)
    assert padded.shape[-1] == model.padded_size
    cropped = model._crop_face(padded)
    assert torch.equal(cropped, face)
