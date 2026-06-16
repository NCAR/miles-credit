"""test_preblock_fill_values.py — unit tests for credit.preblock.fill_values.FillValues."""

import logging

import math

import pytest
import torch

from credit.preblock import PREBLOCK_REGISTRY, build_preblocks
from credit.preblock.fill_values import FillValues


def create_nan_data() -> dict:
    """Batch dict with NaNs (and an inf) in each variable, shape (2, 3, 1, 4, 4)."""
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


def create_mixed_data() -> dict:
    """Batch dict with NaN, zero, negative, positive, and inf values."""
    shape = (2, 3, 1, 4, 4)
    var_names = ["Test_ERA5/prognostic/3d/T", "Test_ERA5/prognostic/3d/U"]

    def _make():
        out = {}
        for var in var_names:
            t = torch.ones(*shape)
            t[0, 0, 0, 0, 0] = float("nan")
            t[0, 1, 0, 1, 1] = 0.0
            t[0, 2, 0, 2, 2] = -0.5
            t[1, 0, 0, 0, 0] = float("inf")
            out[var] = t
        return out

    return {split: {"Test_ERA5": _make()} for split in ("input", "target")}


def _all_vars(batch, data_type="input", source="Test_ERA5"):
    return batch[data_type][source]


# ---------------------------------------------------------------------------
# NaN replacement
# ---------------------------------------------------------------------------


def test_fill_values_default_fills_all_nans():
    """With no variables specified, every NaN in every variable is replaced."""
    batch = create_nan_data()
    result = FillValues(rules=[{"search": "nan", "fill": -1.0}])(batch)
    for data_type in ("input", "target"):
        for var, tensor in _all_vars(result, data_type).items():
            assert not torch.isnan(tensor).any(), f"NaNs remain in {data_type}/{var}"


def test_fill_values_nan_replaced_at_known_positions():
    """Exact NaN positions get fill value; finite values are unchanged."""
    batch = create_nan_data()
    original = batch["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"].clone()
    result = FillValues(rules=[{"search": "nan", "fill": -1.0}])(batch)
    out = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    assert out[0, 0, 0, 0, 0] == -1.0
    assert out[1, 1, 0, 1, 1] == -1.0
    assert out[0, 2, 0, 2, 2] == original[0, 2, 0, 2, 2]


def test_fill_values_preserves_inf():
    """Only NaN is matched by 'nan'; inf is left untouched."""
    batch = create_nan_data()
    result = FillValues(rules=[{"search": "nan", "fill": 0.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert math.isinf(t[0, 1, 0, 0, 0].item())
    assert not torch.isnan(t).any()


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------


def test_fill_values_eq_replaces_exact_zeros():
    """op=eq replaces values exactly equal to search."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 0.0, "op": "eq", "fill": 1e-4}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 1, 0, 1, 1] == pytest.approx(1e-4)  # was 0.0
    assert t[0, 2, 0, 2, 2] == pytest.approx(-0.5)  # was -0.5, untouched
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0, untouched


def test_fill_values_lt_clamps_negatives():
    """op=lt replaces all values strictly less than search."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 0.0, "op": "lt", "fill": 99.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 2, 0, 2, 2] == pytest.approx(99.0)  # was -0.5 → replaced
    assert t[0, 1, 0, 1, 1] == pytest.approx(0.0)  # was 0.0 → untouched (not < 0)
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0 → untouched


def test_fill_values_le_includes_zero():
    """op=le replaces values less than or equal to search."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 0.0, "op": "le", "fill": 1e-4}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 1, 0, 1, 1] == pytest.approx(1e-4)  # was 0.0 → replaced
    assert t[0, 2, 0, 2, 2] == pytest.approx(1e-4)  # was -0.5 → replaced
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0 → untouched


def test_fill_values_ge_replaces_large_values():
    """op=ge replaces values greater than or equal to search."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 1.0, "op": "ge", "fill": 0.5}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 0, 0, 2, 2] == pytest.approx(0.5)  # was 1.0 → replaced
    assert t[0, 2, 0, 2, 2] == pytest.approx(-0.5)  # was -0.5 → untouched
    assert t[1, 0, 0, 0, 0] == pytest.approx(0.5)  # was inf → also replaced (inf >= 1.0)


def test_fill_values_eq_replaces_inf():
    """op=eq with search=inf replaces only exact inf positions."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": float("inf"), "op": "eq", "fill": 0.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[1, 0, 0, 0, 0] == pytest.approx(0.0)  # was inf → replaced
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0 → untouched
    assert t[0, 2, 0, 2, 2] == pytest.approx(-0.5)  # was -0.5 → untouched


def test_fill_values_gt_catches_inf():
    """op=gt with a finite threshold also catches inf (inf > any finite number)."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 1e10, "op": "gt", "fill": 0.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[1, 0, 0, 0, 0] == pytest.approx(0.0)  # was inf → replaced (inf > 1e10)
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0 → untouched (1.0 not > 1e10)


def test_fill_values_nan_rule_does_not_affect_inf():
    """search='nan' does not match inf — only torch.isnan() positions are filled."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": "nan", "fill": 0.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert math.isinf(t[1, 0, 0, 0, 0].item())  # inf left untouched by nan rule


def test_fill_values_gt_excludes_exact_match():
    """op=gt replaces values strictly greater than search, leaving the exact value untouched."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 0.0, "op": "gt", "fill": 99.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 0, 0, 2, 2] == pytest.approx(99.0)  # was 1.0 → replaced (> 0)
    assert t[0, 1, 0, 1, 1] == pytest.approx(0.0)  # was 0.0 → untouched (not > 0)
    assert t[0, 2, 0, 2, 2] == pytest.approx(-0.5)  # was -0.5 → untouched


def test_fill_values_ne_replaces_non_matching():
    """op=ne replaces all values not equal to search."""
    batch = create_mixed_data()
    result = FillValues(rules=[{"search": 1.0, "op": "ne", "fill": 99.0}])(batch)
    t = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert t[0, 1, 0, 1, 1] == pytest.approx(99.0)  # was 0.0 → replaced (≠ 1.0)
    assert t[0, 2, 0, 2, 2] == pytest.approx(99.0)  # was -0.5 → replaced (≠ 1.0)
    assert t[0, 0, 0, 2, 2] == pytest.approx(1.0)  # was 1.0 → untouched (== 1.0)
    assert torch.isnan(t[0, 0, 0, 0, 0])  # NaN → untouched by numeric op


def test_fill_values_numeric_op_never_touches_nan():
    """Numeric ops leave NaN positions alone regardless of operator."""
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([float("nan"), 0.0, -1.0, 2.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}
    for op in ("eq", "ne", "lt", "le", "gt", "ge"):
        b = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t.clone()}}}
        result = FillValues(rules=[{"search": 0.0, "op": op, "fill": 99.0}])(b)
        out = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"].flatten()
        assert torch.isnan(out[0]), f"NaN was replaced by op={op!r}"


# ---------------------------------------------------------------------------
# Overlapping rules
# ---------------------------------------------------------------------------


def test_fill_values_overlap_warning(caplog):
    """A warning is logged on the first forward pass when two rules overlap."""
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([float("nan"), 0.0, 1.0, 2.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}

    # le: 0.0 and eq: 0.0 both match the position where x == 0.0
    fv = FillValues(
        rules=[
            {"search": 0.0, "op": "le", "fill": 1e-4},
            {"search": 0.0, "op": "eq", "fill": 99.0},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="credit.preblock.fill_values"):
        fv(batch)
    assert any("overlap" in msg for msg in caplog.messages)


def test_fill_values_overlap_warning_emitted_only_once(caplog):
    """The overlap warning is only logged on the first forward pass."""
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}

    fv = FillValues(
        rules=[
            {"search": 0.0, "op": "le", "fill": 1e-4},
            {"search": 0.0, "op": "eq", "fill": 99.0},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="credit.preblock.fill_values"):
        fv(batch)
        count_after_first = len(caplog.messages)
        fv({"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t.clone()}}})
    assert len(caplog.messages) == count_after_first  # no new messages on second call


def test_fill_values_no_overlap_no_warning(caplog):
    """No warning is logged when rules do not overlap."""
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([float("nan"), 0.0, 1.0, 2.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}

    fv = FillValues(
        rules=[
            {"search": "nan", "fill": -1.0},  # matches only NaN
            {"search": 0.0, "fill": 1e-4},  # matches only exact zeros (NaN excluded)
        ]
    )
    with caplog.at_level(logging.WARNING, logger="credit.preblock.fill_values"):
        fv(batch)
    assert not caplog.messages


# ---------------------------------------------------------------------------
# Simultaneous semantics
# ---------------------------------------------------------------------------


def test_fill_values_simultaneous_nan_not_caught_by_zero_rule():
    """NaN-filled zeros are NOT caught by a subsequent zero rule (masks are simultaneous)."""
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([float("nan"), 0.0, 1.0, 2.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}

    result = FillValues(
        rules=[
            {"search": "nan", "fill": 0.0},  # NaN → 0.0
            {"search": 0.0, "fill": 1e-4},  # physical 0.0 → 1e-4
        ]
    )(batch)
    out = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"].flatten()
    assert out[0] == pytest.approx(0.0)  # NaN-filled zero stays at 0.0
    assert out[1] == pytest.approx(1e-4)  # original 0.0 → 1e-4
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Variable subset selection
# ---------------------------------------------------------------------------


def test_fill_values_variable_subset_full_name():
    """Only the selected variable is filled; others keep their NaNs."""
    batch = create_nan_data()
    result = FillValues(
        rules=[{"search": "nan", "fill": 0.0}],
        variables=["Test_ERA5/prognostic/3d/U"],
    )(batch)
    filled = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]
    untouched = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]
    assert not torch.isnan(filled).any()
    assert torch.isnan(untouched).any()


def test_fill_values_variable_subset_partial_name_expands():
    """A partial name expands to every variable beneath it."""
    batch = create_nan_data()
    fv = FillValues(rules=[{"search": "nan", "fill": 0.0}], variables=["Test_ERA5/prognostic/3d"])
    result = fv(batch)
    assert sorted(fv.variables) == ["Test_ERA5/prognostic/3d/T", "Test_ERA5/prognostic/3d/U"]
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/U"]).any()
    assert torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/2d/SP"]).any()


def test_fill_values_data_types_scope():
    """Restricting data_types fills only those splits."""
    batch = create_nan_data()
    result = FillValues(rules=[{"search": "nan", "fill": 0.0}], data_types=["input"])(batch)
    assert not torch.isnan(result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()
    assert torch.isnan(result["target"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"]).any()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_fill_values_preserves_shape_and_dtype():
    """Output tensors keep their shape and dtype."""
    batch = create_nan_data()
    var = "Test_ERA5/prognostic/3d/T"
    original = batch["input"]["Test_ERA5"][var]
    shape, dtype = original.shape, original.dtype
    result = FillValues(rules=[{"search": "nan", "fill": 0.0}])(batch)
    out = result["input"]["Test_ERA5"][var]
    assert out.shape == shape
    assert out.dtype == dtype


def test_fill_values_invalid_data_type_raises():
    """An unsupported data_type is rejected at construction time."""
    with pytest.raises(ValueError):
        FillValues(rules=[{"search": "nan", "fill": 0.0}], data_types=["prediction"])


def test_fill_values_missing_key_raises():
    """A rule missing 'search' or 'fill' is rejected at construction time."""
    with pytest.raises(ValueError):
        FillValues(rules=[{"search": "nan"}])
    with pytest.raises(ValueError):
        FillValues(rules=[{"fill": 0.0}])


def test_fill_values_invalid_op_raises():
    """An unsupported op is rejected at construction time."""
    with pytest.raises(ValueError):
        FillValues(rules=[{"search": 0.0, "op": "contains", "fill": 0.0}])


# ---------------------------------------------------------------------------
# Registry / config construction
# ---------------------------------------------------------------------------


def test_fill_values_registered():
    """FillValues is registered under the 'fill_values' key."""
    assert PREBLOCK_REGISTRY["fill_values"] is FillValues


def test_fill_values_built_from_config():
    """build_preblocks instantiates FillValues from a config dict and it works."""
    cfg = {
        "per_step": {
            "fill": {
                "type": "fill_values",
                "args": {
                    "rules": [
                        {"search": "nan", "fill": -1.0},
                        {"search": 0.0, "op": "eq", "fill": 1e-4},
                    ]
                },
            }
        }
    }
    blocks = build_preblocks(cfg)
    shape = (1, 1, 1, 2, 2)
    t = torch.tensor([float("nan"), 0.0, 1.0, 2.0]).reshape(shape)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": t}}}
    result = blocks["fill"](batch)
    out = result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"].flatten()
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(1e-4)
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(2.0)
