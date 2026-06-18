"""test_preblock_nan.py — unit tests for credit.preblock.nan.FillNan."""

import math

import pytest
import torch

from credit.preblock import build_preblocks
from credit.preblock.nan import FillNan


def create_nan_data() -> dict:
    """Nested CREDIT-convention batch dict with NaNs (and an inf) sprinkled in.

    Structure: data[data_type][source][var_name], tensor shape (2, 3, 1, 4, 4).
    Each variable gets a couple of NaN entries; T also gets an inf so we can
    assert infs are left untouched.
    """
    shape = (2, 3, 1, 4, 4)
    var_names = [
        "Test_ERA5/prognostic/3d/T",
        "Test_ERA5/prognostic/3d/U",
        "Test_ERA5/prognostic/2d/SP",
    ]

    def _make():
        out = {}
        for var in var_names:
            t = torch.randn(*shape)
            t[0, 0, 0, 0, 0] = float("nan")
            t[1, 1, 0, 1, 1] = float("nan")
            if var.endswith("/T"):
                t[0, 1, 0, 0, 0] = float("inf")
            out[var] = t
        return out

    return {split: {"Test_ERA5": _make()} for split in ("input", "target")}


def _all_vars(batch, data_type="input", source="Test_ERA5"):
    return batch[data_type][source]


# ---------------------------------------------------------------------------
# Default behaviour: fill every NaN with 0.0 across all variables / data types
# ---------------------------------------------------------------------------


def test_fill_nan_default_fills_all():
    """With no variables specified, every NaN in every variable is replaced by 0."""
    batch = create_nan_data()
    result = FillNan()(batch)
    for data_type in ("input", "target"):
        for var, tensor in _all_vars(result, data_type).items():
            assert not torch.isnan(tensor).any(), f"NaNs remain in {data_type}/{var}"


def test_fill_nan_replaces_with_zero_at_known_positions():
    """The exact NaN positions become the fill value, finite values are unchanged."""
    batch = create_nan_data()
    original = batch["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"].clone()
    result = FillNan()(batch)
    out = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    assert out[0, 0, 0, 0, 0] == 0.0
    assert out[1, 1, 0, 1, 1] == 0.0
    # A finite location is left exactly as it was.
    assert out[0, 2, 0, 2, 2] == original[0, 2, 0, 2, 2]


def test_fill_nan_preserves_inf_and_finite():
    """Only NaNs are touched; +/-inf and finite values are preserved."""
    batch = create_nan_data()
    result = FillNan()(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert math.isinf(t[0, 1, 0, 0, 0].item())
    assert not torch.isnan(t).any()


def test_fill_nan_custom_value():
    """A custom fill_value is used in place of the default 0."""
    batch = create_nan_data()
    result = FillNan(fill_value=-999.0)(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    assert t[0, 0, 0, 0, 0] == -999.0
    assert t[1, 1, 0, 1, 1] == -999.0


# ---------------------------------------------------------------------------
# Variable subset selection
# ---------------------------------------------------------------------------


def test_fill_nan_variable_subset_full_name():
    """Only the selected variable is filled; others keep their NaNs."""
    batch = create_nan_data()
    result = FillNan(variables=["Test_ERA5/prognostic/3d/U"])(batch)
    filled = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    untouched = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert not torch.isnan(filled).any()
    assert torch.isnan(untouched).any()


def test_fill_nan_variable_subset_partial_name_expands():
    """A partial name expands to every variable beneath it (via _parse_variable_selection)."""
    batch = create_nan_data()
    fn = FillNan(variables=["Test_ERA5/prognostic/3d"])
    result = fn(batch)
    # Both 3d variables should have been expanded and filled.
    assert sorted(fn.variables) == [
        "Test_ERA5/prognostic/3d/T",
        "Test_ERA5/prognostic/3d/U",
    ]
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]).any()
    # The 2d variable was not selected, so its NaNs remain.
    assert torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/2d/SP"]).any()


def test_fill_nan_data_types_scope():
    """Restricting data_types fills only those splits."""
    batch = create_nan_data()
    result = FillNan(data_types=["input"])(batch)
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()
    # target was not in scope, so its NaNs are untouched.
    assert torch.isnan(result["target"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_fill_nan_preserves_shape_and_dtype():
    """Output tensors keep their shape and dtype."""
    batch = create_nan_data()
    var = "Test_ERA5/prognostic/3d/T"
    original = batch["input"]["Test_ERA5"][var]
    shape, dtype = original.shape, original.dtype
    result = FillNan()(batch)
    out = result["input"]["Test_ERA5"][var]
    assert out.shape == shape
    assert out.dtype == dtype


def test_fill_nan_invalid_data_type_raises():
    """An unsupported data_type is rejected at construction time."""
    with pytest.raises(ValueError):
        FillNan(data_types=["prediction"])


# ---------------------------------------------------------------------------
# Registry / config construction
# ---------------------------------------------------------------------------


def test_fill_nan_registered():
    """FillNan is registered under the 'fill_nan' key."""
    from credit.preblock import _load_preblock_entry

    assert _load_preblock_entry("fill_nan") is FillNan


def test_fill_nan_built_from_config():
    """build_preblocks instantiates FillNan from a config dict and it works."""
    blocks = build_preblocks({"preblocks": {"per_step": {"fill": {"type": "fill_nan", "args": {"fill_value": 5.0}}}}})
    result = blocks["fill"](create_nan_data())
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    assert t[0, 0, 0, 0, 0] == 5.0
    assert not torch.isnan(t).any()
