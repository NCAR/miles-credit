"""test_preblock.py — unit tests for credit.preblock modules."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr

from bridgescaler.distributed_tensor import DStandardScalerTensor
from bridgescaler import save_scaler_dict, scale_var_dict
from credit.preblock import apply_preblocks, build_preblocks
from credit.preblock.concat import ConcatToTensor
from credit.preblock.log import LogTransform
from credit.preblock.regrid import Regridder
from credit.preblock.scaler import BridgeScalerTransform
from credit.preblock.sqrt import SqrtTransform
from credit.preblock._utils import _parse_variable_selection
from credit.postblock import build_postblocks


def create_synthetic_data() -> dict:
    """
    Creates synthetic data as a nested dictionary of torch tensors.

    Structure: data[data_type][source][var_name]
    - data_type: "input" | "target"
    - source: "Test_ERA5"
    - var_name: "Test_ERA5/prognostic/3d/T" | ...
    - tensor shape: (100, 16, 1, 8, 8)
    """
    shape = (100, 16, 1, 8, 8)
    var_names = [
        "Test_ERA5/prognostic/3d/T",
        "Test_ERA5/prognostic/3d/U",
        "Test_ERA5/prognostic/3d/V",
    ]

    return {split: {"Test_ERA5": {var: torch.randn(*shape) for var in var_names}} for split in ("input", "target")}


# ---------------------------------------------------------------------------
# Fixture — synthetic ESMF weight file (384×576 → 192×288)
# ---------------------------------------------------------------------------


@pytest.fixture
def weight_file(tmp_path):
    """Write a minimal ESMF-compatible weight file and return its path.

    Represents a 2:1 block-average downsampling from an 8×8 source grid to a
    4×4 destination grid, matching create_synthetic_data().  Each destination
    cell is the average of the 4 corresponding source cells (weight = 0.25 each).

    Only the variables read by Regrid.__init__ are written:
      - dst_grid_dims: [nlon, nlat] SCRIP/ESMF convention; Regrid reverses with [::-1]
      - row, col, S:   1-based COO sparse entries
      - mask_a/b:      stubs so xarray exposes the n_a / n_b dimension sizes
    """
    n_src_lat, n_src_lon = 8, 8
    n_dst_lat, n_dst_lon = 4, 4

    # Build the COO sparse entries with numpy (vectorized, no Python loop)
    j_dst, i_dst = np.indices((n_dst_lat, n_dst_lon))
    j_dst, i_dst = j_dst.ravel(), i_dst.ravel()  # (n_b,)
    dst_idx = j_dst * n_dst_lon + i_dst + 1  # 1-based

    dj = np.array([0, 0, 1, 1])
    di = np.array([0, 1, 0, 1])
    j_src = j_dst[:, None] * 2 + dj  # (n_b, 4)
    i_src = i_dst[:, None] * 2 + di  # (n_b, 4)
    src_idx = j_src * n_src_lon + i_src + 1  # 1-based

    rows = np.repeat(dst_idx, 4)  # (n_s,)
    cols = src_idx.ravel()  # (n_s,)
    vals = np.full(len(rows), 0.25)

    n_a = n_src_lat * n_src_lon
    n_b = n_dst_lat * n_dst_lon

    ds = xr.Dataset(
        {
            "dst_grid_dims": xr.DataArray(np.array([n_dst_lon, n_dst_lat], dtype=np.int32), dims=("dst_grid_rank",)),
            "mask_a": xr.DataArray(np.ones(n_a, dtype=np.int32), dims=("n_a",)),
            "mask_b": xr.DataArray(np.ones(n_b, dtype=np.int32), dims=("n_b",)),
            "row": xr.DataArray(rows.astype(np.int32), dims=("n_s",)),
            "col": xr.DataArray(cols.astype(np.int32), dims=("n_s",)),
            "S": xr.DataArray(vals.astype(np.float64), dims=("n_s",)),
        }
    )
    path = tmp_path / "weights.nc"
    ds.to_netcdf(path)
    return str(path), n_src_lat, n_src_lon, n_dst_lat, n_dst_lon


# ---------------------------------------------------------------------------
# Regrid tests
# ---------------------------------------------------------------------------


def test_regrid_output_shape(weight_file):
    """Downsampling regrid produces the destination grid shape for all splits."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/prognostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = create_synthetic_data()
    result = regrid(batch)
    for split in ("input", "target"):
        assert result[split]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"].shape == (100, 16, 1, n_dst_lat, n_dst_lon)


def test_regrid_uniform_input(weight_file):
    """Block-average regrid: uniform input maps to uniform output of the same value."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/prognostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/prognostic/3d/T": torch.ones(1, 1, 1, n_src_lat, n_src_lon)}}}
    result = regrid(batch)
    assert torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"],
        torch.ones(1, 1, 1, n_dst_lat, n_dst_lon),
        atol=1e-5,
    )


def test_regrid_reshape_false(weight_file):
    """reshape_to_xy=False returns a flat (prod(lead_dims), n_b) tensor."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/prognostic/3d/T"]
    regrid = Regridder(path, variables=variables, reshape_to_xy=False)
    batch = create_synthetic_data()
    result = regrid(batch)
    assert result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"].shape == (100 * 16 * 1, n_dst_lat * n_dst_lon)


def test_regrid_flip_axis(weight_file):
    """flip_axis is applied to the input before regridding."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/prognostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    regrid_flip = Regridder(path, variables=variables, flip_axis=[-1])
    batch = create_synthetic_data()
    result = regrid(copy.deepcopy(batch))
    result_flip = regrid_flip(copy.deepcopy(batch))
    assert not torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"],
        result_flip["input"]["Test_ERA5"]["Test_ERA5/prognostic/3d/T"],
    )


# ---------------------------------------------------------------------------
# Fixture — real DStandardScalerTensor fit on random data
# ---------------------------------------------------------------------------


@pytest.fixture
def scaler_file(tmp_path):
    """Fit a DStandardScalerTensor on random data, save to JSON, return path.

    Uses 16 channels to match typical CREDIT usage.  Spatial size is kept small
    (8×8) so the fixture stays fast.
    """
    x_dict = create_synthetic_data()
    variables = x_dict["input"]["Test_ERA5"].keys()
    scaler = DStandardScalerTensor(channels_last=False)
    scaler_dict = scale_var_dict(x_dict, scaler, method="fit")
    path = str(tmp_path / "scaler.json")
    save_scaler_dict(scaler_dict, path)
    return path, variables, x_dict


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------


def test_scaler_output_shape(scaler_file):
    """Transform preserves the input tensor shape for every variable."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransform(scaler_path=path, variables=list(variables), method="transform")
    original_shapes = {v: data["input"]["Test_ERA5"][v].shape for v in variables}
    result = scaler(data)
    for v in variables:
        assert result["input"]["Test_ERA5"][v].shape == original_shapes[v]


def test_scaler_transform_changes_values(scaler_file):
    """Transform produces different values than the raw input."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransform(scaler_path=path, variables=list(variables), method="transform")
    var = list(variables)[0]
    original = data["input"]["Test_ERA5"][var].clone()
    result = scaler(data)
    assert not torch.allclose(result["input"]["Test_ERA5"][var].float(), original.float())


def test_scaler_round_trip(scaler_file):
    """transform followed by inverse recovers the original tensor."""
    path, variables, data = scaler_file
    var_list = list(variables)
    fwd = BridgeScalerTransform(scaler_path=path, variables=var_list, method="transform")
    inv = BridgeScalerTransform(scaler_path=path, variables=var_list, method="inverse_transform")
    var = var_list[0]
    original = data["input"]["Test_ERA5"][var].clone()
    data = fwd(data)
    data = inv(data)
    assert torch.allclose(data["input"]["Test_ERA5"][var].float(), original.float(), atol=1e-5)


def test_scaler_data_types_input_only(scaler_file):
    """data_types=['input'] scales input tensors and leaves target tensors unchanged."""
    path, variables, data = scaler_file
    var = list(variables)[0]
    input_before = data["input"]["Test_ERA5"][var].clone()
    target_before = data["target"]["Test_ERA5"][var].clone()

    scaler = BridgeScalerTransform(
        scaler_path=path, variables=list(variables), method="transform", data_types=["input"]
    )
    result = scaler(data)

    assert not torch.allclose(result["input"]["Test_ERA5"][var].float(), input_before.float()), (
        "input tensor should have been scaled"
    )
    assert torch.allclose(result["target"]["Test_ERA5"][var].float(), target_before.float()), (
        "target tensor must not be touched when data_types=['input']"
    )


def test_scaler_data_types_target_only(scaler_file):
    """data_types=['target'] scales target tensors and leaves input tensors unchanged."""
    path, variables, data = scaler_file
    var = list(variables)[0]
    input_before = data["input"]["Test_ERA5"][var].clone()
    target_before = data["target"]["Test_ERA5"][var].clone()

    scaler = BridgeScalerTransform(
        scaler_path=path, variables=list(variables), method="transform", data_types=["target"]
    )
    result = scaler(data)

    assert torch.allclose(result["input"]["Test_ERA5"][var].float(), input_before.float()), (
        "input tensor must not be touched when data_types=['target']"
    )
    assert not torch.allclose(result["target"]["Test_ERA5"][var].float(), target_before.float()), (
        "target tensor should have been scaled"
    )


def test_scaler_data_types_none_scales_all(scaler_file):
    """data_types=None (default) scales both input and target tensors."""
    path, variables, data = scaler_file
    var = list(variables)[0]
    input_before = data["input"]["Test_ERA5"][var].clone()
    target_before = data["target"]["Test_ERA5"][var].clone()

    scaler = BridgeScalerTransform(scaler_path=path, variables=list(variables), method="transform")
    result = scaler(data)

    assert not torch.allclose(result["input"]["Test_ERA5"][var].float(), input_before.float()), (
        "input tensor should have been scaled with data_types=None"
    )
    assert not torch.allclose(result["target"]["Test_ERA5"][var].float(), target_before.float()), (
        "target tensor should have been scaled with data_types=None"
    )


# ---------------------------------------------------------------------------
# _parse_variable_selection
# ---------------------------------------------------------------------------


def _selection_state() -> dict:
    """A state dict following the CREDIT convention: state[data_type][source][var_name].

    var_name uses the canonical `source/field_type/dim/varname` layout, and the
    source key (e.g. "era5") matches the first segment of each variable name.
    """
    return {
        "input": {
            "era5": {
                "era5/prognostic/3d/T": object(),
                "era5/prognostic/3d/U": object(),
                "era5/prognostic/2d/SP": object(),
                "era5/static/2d/Z": object(),
            },
        },
        "target": {
            "era5": {
                "era5/prognostic/3d/T": object(),
                "era5/diagnostic/2d/precip": object(),
            },
        },
        "prediction": {
            "era5": {
                "era5/prognostic/3d/T": object(),
            },
        },
    }


def test_parse_variable_selection_expands_partial():
    """A partial name expands to every variable beneath it in the hierarchy."""
    state = _selection_state()
    result = _parse_variable_selection(["era5/prognostic/3d"], state)
    assert result == ["era5/prognostic/3d/T", "era5/prognostic/3d/U"]


def test_parse_variable_selection_full_name_matches_only_itself():
    """A full variable name matches exactly and does not pull in siblings."""
    state = _selection_state()
    result = _parse_variable_selection(["era5/prognostic/3d/T"], state)
    assert result == ["era5/prognostic/3d/T"]


def test_parse_variable_selection_empty_list_returns_all():
    """An empty selection returns every variable across all data types (deduped)."""
    state = _selection_state()
    result = _parse_variable_selection([], state)
    assert result == [
        "era5/prognostic/3d/T",
        "era5/prognostic/3d/U",
        "era5/prognostic/2d/SP",
        "era5/static/2d/Z",
        "era5/diagnostic/2d/precip",
    ]


def test_parse_variable_selection_dedupes_across_data_types():
    """A variable present in multiple data types appears only once."""
    state = _selection_state()
    result = _parse_variable_selection(["era5/prognostic/3d/T"], state)
    assert result == ["era5/prognostic/3d/T"]


def test_parse_variable_selection_data_types_filter():
    """Only the requested data types contribute candidate variables."""
    state = _selection_state()
    result = _parse_variable_selection([], state, data_types=["prediction"])
    assert result == ["era5/prognostic/3d/T"]

    result = _parse_variable_selection(["era5"], state, data_types=["target"])
    assert result == ["era5/prognostic/3d/T", "era5/diagnostic/2d/precip"]


def test_parse_variable_selection_missing_data_type_ignored():
    """A requested data type that is absent from the state is skipped, not an error."""
    state = _selection_state()
    result = _parse_variable_selection([], state, data_types=["prediction", "nonexistent"])
    assert result == ["era5/prognostic/3d/T"]


def test_parse_variable_selection_prefix_boundary():
    """Matching respects '/' boundaries: a prefix that is not a full path segment
    does not match."""
    state = _selection_state()
    # "era5/prognostic/3" is a string prefix of "era5/prognostic/3d/..." but not a
    # hierarchy ancestor, so nothing should match.
    result = _parse_variable_selection(["era5/prognostic/3"], state)
    assert result == []


def test_parse_variable_selection_preserves_selection_order():
    """Multiple partials expand in the order they are listed."""
    state = _selection_state()
    result = _parse_variable_selection(["era5/static", "era5/prognostic/2d"], state)
    assert result == ["era5/static/2d/Z", "era5/prognostic/2d/SP"]


def test_parse_variable_selection_multiple_sources():
    """Variables are collected across all sources within a data type, and a
    source-level partial selects only that source's variables."""
    state = {
        "input": {
            "era5": {
                "era5/prognostic/3d/T": object(),
                "era5/prognostic/2d/SP": object(),
            },
            "gfs": {
                "gfs/prognostic/3d/T": object(),
                "gfs/static/2d/Z": object(),
            },
        },
    }
    # Empty list pulls in every variable from every source.
    assert _parse_variable_selection([], state) == [
        "era5/prognostic/3d/T",
        "era5/prognostic/2d/SP",
        "gfs/prognostic/3d/T",
        "gfs/static/2d/Z",
    ]
    # A source-rooted partial selects only that source's variables.
    assert _parse_variable_selection(["gfs"], state) == [
        "gfs/prognostic/3d/T",
        "gfs/static/2d/Z",
    ]


# ---------------------------------------------------------------------------
# build_preblocks and build_postblocks — two-section format enforcement
# ---------------------------------------------------------------------------


class TestBuildPreblocks:
    """Tests for build_preblocks() config validation and section selection."""

    def test_flat_format_raises_value_error(self):
        """Old flat format (no ic_only/per_step sections) raises ValueError."""
        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_preblocks({"concat": {"type": "concat"}})

    def test_unknown_section_key_raises(self):
        """A key that is neither ic_only nor per_step raises ValueError."""
        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_preblocks({"per_step": {}, "bad_section": {}})

    def test_valid_two_section_per_step_builds(self):
        """per_step section builds an nn.ModuleDict with the named block."""
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        assert isinstance(preblocks, nn.ModuleDict)
        assert "concat" in preblocks

    def test_valid_two_section_ic_only_builds(self):
        """ic_only section builds an nn.ModuleDict with the named block."""
        preblocks = build_preblocks({"ic_only": {"concat": {"type": "concat"}}}, phase="ic_only")
        assert isinstance(preblocks, nn.ModuleDict)
        assert "concat" in preblocks

    def test_empty_config_returns_empty_module_dict(self):
        """Empty config builds an empty ModuleDict without error."""
        preblocks = build_preblocks({}, phase="per_step")
        assert isinstance(preblocks, nn.ModuleDict)
        assert len(preblocks) == 0

    def test_missing_phase_returns_empty_module_dict(self):
        """Requesting a phase absent from the config returns an empty ModuleDict."""
        # ic_only is configured but per_step is requested
        preblocks = build_preblocks({"ic_only": {"concat": {"type": "concat"}}}, phase="per_step")
        assert len(preblocks) == 0

    def test_invalid_phase_raises(self):
        """An unrecognized phase name raises ValueError."""
        with pytest.raises(ValueError, match="phase must be one of"):
            build_preblocks({}, phase="invalid_phase")

    def test_unknown_block_type_raises_with_valid_types(self):
        """An unregistered block type raises ValueError naming the block and listing valid types."""
        with pytest.raises(ValueError, match="unknown preblock type 'not_a_block'.*'concat'"):
            build_preblocks({"per_step": {"bogus": {"type": "not_a_block"}}}, phase="per_step")


class TestBuildPostblocks:
    """Tests for build_postblocks() config validation (mirrors build_preblocks)."""

    def test_flat_format_raises_value_error(self):
        """Old flat postblock format raises ValueError."""
        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_postblocks({"reconstruct": {"type": "reconstruct"}})

    def test_valid_two_section_per_step_builds(self):
        """per_step section builds a ModuleDict with the named block."""
        postblocks = build_postblocks({"per_step": {"reconstruct": {"type": "reconstruct"}}}, phase="per_step")
        assert isinstance(postblocks, nn.ModuleDict)
        assert "reconstruct" in postblocks

    def test_empty_config_returns_empty_module_dict(self):
        postblocks = build_postblocks({}, phase="per_step")
        assert len(postblocks) == 0

    def test_unknown_block_type_raises_with_valid_types(self):
        """An unregistered block type raises ValueError naming the block and listing valid types."""
        with pytest.raises(ValueError, match="unknown postblock type 'not_a_block'.*'reconstruct'"):
            build_postblocks({"per_step": {"bogus": {"type": "not_a_block"}}}, phase="per_step")

    def test_flatten_to_tensor_registered(self):
        """flatten_to_tensor builds without a scaler (scaler_path omitted)."""
        postblocks = build_postblocks({"per_step": {"flatten": {"type": "flatten_to_tensor"}}}, phase="per_step")
        assert "flatten" in postblocks

    def test_flatten_to_tensor_expands_env_vars_in_scaler_path(self, monkeypatch, tmp_path):
        """$VARS in scaler_path are expanded before the scaler file is opened."""
        import json

        from credit.postblock.reconstruct import FlattenToTensor

        scaler_file = tmp_path / "scaler.json"
        scaler_file.write_text(json.dumps({"target": {}}))
        monkeypatch.setenv("CREDIT_TEST_SCALER_DIR", str(tmp_path))
        block = FlattenToTensor(scaler_path="$CREDIT_TEST_SCALER_DIR/scaler.json")
        assert block.scaler_path == str(scaler_file)

    def test_global_energy_fixer_updown_alias(self):
        """global_energy_fixer_updown resolves to the same class as global_energy_fixer."""
        from credit.postblock import POSTBLOCK_REGISTRY

        assert POSTBLOCK_REGISTRY["global_energy_fixer_updown"] is POSTBLOCK_REGISTRY["global_energy_fixer"]


# ---------------------------------------------------------------------------
# ConcatToTensor — channel ordering and channel map correctness
# ---------------------------------------------------------------------------


class TestConcatToTensorChannelOrder:
    """ConcatToTensor must sort channels by FIELD_TYPE_RANK regardless of insertion order."""

    def test_channel_order_follows_field_type_rank(self):
        """prognostic(0) < static(1) < dynamic_forcing(2) regardless of dict insertion order."""
        B, H, W = 1, 4, 4
        # Insert in reverse-rank order to verify the sort is applied
        batch = {
            "input": {
                "era5": {
                    "era5/dynamic_forcing/2d/df": torch.full((B, 1, 1, H, W), 9.0),
                    "era5/static/2d/st": torch.full((B, 1, 1, H, W), 5.0),
                    "era5/prognostic/2d/p": torch.full((B, 1, 1, H, W), 3.0),
                }
            }
        }
        ct = ConcatToTensor()
        tensor, _meta = ct(batch)
        # Shape: (B, 3_channels, 1_timestep, H, W)
        assert tensor.shape == (B, 3, 1, H, W)
        assert tensor[0, 0, 0, 0, 0].item() == pytest.approx(3.0)  # prognostic
        assert tensor[0, 1, 0, 0, 0].item() == pytest.approx(5.0)  # static
        assert tensor[0, 2, 0, 0, 0].item() == pytest.approx(9.0)  # dynamic_forcing

    def test_input_channel_map_contains_all_variables(self):
        """input _channel_map has an entry for every variable key in the batch."""
        B, H, W = 1, 4, 4
        var_keys = [
            "era5/prognostic/2d/T",
            "era5/static/2d/z",
            "era5/dynamic_forcing/2d/insolation",
        ]
        batch = {"input": {"era5": {k: torch.randn(B, 1, 1, H, W) for k in var_keys}}}
        ct = ConcatToTensor()
        _, meta = ct(batch)
        channel_map = meta["input"]["_channel_map"]
        for key in var_keys:
            assert key in channel_map, f"Expected {key!r} in input _channel_map"

    def test_target_channel_map_excludes_non_predictable_fields(self):
        """Target _channel_map includes only prognostic and diagnostic; not static or dynfrc."""
        B, H, W = 1, 4, 4
        batch = {
            "input": {
                "era5": {
                    "era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W),
                    "era5/static/2d/z": torch.randn(B, 1, 1, H, W),
                    "era5/dynamic_forcing/2d/insolation": torch.randn(B, 1, 1, H, W),
                    "era5/diagnostic/2d/cape": torch.randn(B, 1, 1, H, W),
                }
            }
        }
        ct = ConcatToTensor()
        _, meta = ct(batch)
        target_map = meta["target"]["_channel_map"]
        assert "era5/prognostic/2d/T" in target_map
        assert "era5/diagnostic/2d/cape" in target_map
        assert "era5/static/2d/z" not in target_map
        assert "era5/dynamic_forcing/2d/insolation" not in target_map

    def test_channel_map_slices_are_non_overlapping(self):
        """Each variable's slice in _channel_map is disjoint from all others."""
        B, H, W = 1, 4, 4
        batch = {
            "input": {
                "era5": {
                    "era5/prognostic/2d/a": torch.randn(B, 1, 1, H, W),
                    "era5/prognostic/2d/b": torch.randn(B, 1, 1, H, W),
                    "era5/static/2d/c": torch.randn(B, 1, 1, H, W),
                }
            }
        }
        ct = ConcatToTensor()
        _, meta = ct(batch)
        channel_map = meta["input"]["_channel_map"]

        covered = []
        for info in channel_map.values():
            s = info["slice"]
            covered.extend(range(s.start, s.stop))

        assert len(covered) == len(set(covered)), "Channel slices must not overlap"


# ---------------------------------------------------------------------------
# apply_preblocks — return format and mutation safety (purity)
# ---------------------------------------------------------------------------


class TestApplyPreblocks:
    """Tests for apply_preblocks() return format and input-batch immutability."""

    def test_return_format_with_concat_has_x_y_metadata(self):
        """apply_preblocks returns {"x", "y", "metadata"} when ConcatToTensor is the final block."""
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        B, H, W = 1, 4, 4
        batch = {
            "input": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
            "target": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
        }
        result = apply_preblocks(preblocks, batch)

        assert {"x", "y", "metadata"} <= set(result.keys())
        assert isinstance(result["x"], torch.Tensor)
        assert isinstance(result["y"], torch.Tensor)

    def test_metadata_contains_input_and_target_channel_maps(self):
        """Metadata from apply_preblocks contains populated _channel_map for input and target."""
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        B, H, W = 1, 4, 4
        var_key = "era5/prognostic/2d/T"
        batch = {
            "input": {"era5": {var_key: torch.randn(B, 1, 1, H, W)}},
            "target": {"era5": {var_key: torch.randn(B, 1, 1, H, W)}},
        }
        result = apply_preblocks(preblocks, batch)

        meta = result["metadata"]
        assert "_channel_map" in meta["input"], "input _channel_map missing from metadata"
        assert "_channel_map" in meta["target"], "target _channel_map missing from metadata"
        assert var_key in meta["input"]["_channel_map"]
        assert var_key in meta["target"]["_channel_map"]

    def test_does_not_mutate_input_tensor_values(self):
        """apply_preblocks does not modify the caller's batch tensors in-place."""
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        B, H, W = 1, 4, 4
        original_tensor = torch.ones(B, 1, 1, H, W)
        batch = {
            "input": {"era5": {"era5/prognostic/2d/T": original_tensor}},
        }
        before = original_tensor.clone()
        _ = apply_preblocks(preblocks, batch)

        # Reference identity preserved — same object, same values
        assert batch["input"]["era5"]["era5/prognostic/2d/T"] is original_tensor
        torch.testing.assert_close(original_tensor, before)

    def test_does_not_add_keys_to_caller_batch(self):
        """apply_preblocks does not add or remove keys from the caller's batch dict."""
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        B, H, W = 1, 4, 4
        batch = {
            "input": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
        }
        original_keys = set(batch.keys())
        _ = apply_preblocks(preblocks, batch)

        assert set(batch.keys()) == original_keys

    def test_empty_chain_returns_nested_dict_unchanged(self):
        """An empty preblock chain passes the batch through unmodified."""
        preblocks = build_preblocks({}, phase="per_step")
        B, H, W = 1, 4, 4
        batch = {
            "input": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
        }
        result = apply_preblocks(preblocks, batch)

        # Without ConcatToTensor, _run_preblock_group returns the batch dict
        assert isinstance(result, dict)
        assert "input" in result
        assert "era5/prognostic/2d/T" in result["input"]["era5"]

    def test_concat_result_exposes_input_tensor_under_x(self):
        """The concat result is a dict keyed by "x"; the rollout apps read result["x"].

        rollout_to_netcdf_gen2 previously unpacked this as ``x, _ = apply_preblocks(...)``,
        which iterates the dict keys (strings) instead of returning the tensor, so the
        following ``x.to(device)`` raised AttributeError/ValueError. Reading result["x"]
        is the contract the apps must use.
        """
        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        B, H, W = 1, 4, 4
        batch = {
            "input": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
            "target": {"era5": {"era5/prognostic/2d/T": torch.randn(B, 1, 1, H, W)}},
        }
        result = apply_preblocks(preblocks, batch)

        assert "x" in result
        assert isinstance(result["x"], torch.Tensor)
        # the tensor is usable as a tensor (what the rollout app does next)
        assert result["x"].float().shape[0] == B


# ---------------------------------------------------------------------------
# LogTransform and SqrtTransform — lazy expansion (variables=[] and partial paths)
# ---------------------------------------------------------------------------

_TRANSFORM_SHAPE = (2, 4, 1, 4, 4)
_TRANSFORM_SOURCE = "era5"
_TRANSFORM_VARS = [
    "era5/prognostic/3d/T",
    "era5/prognostic/3d/U",
    "era5/static/2d/Z",
]


def _transform_batch_positive():
    """Batch with strictly positive values for transforms that require positivity."""
    return {
        split: {_TRANSFORM_SOURCE: {v: torch.rand(*_TRANSFORM_SHAPE) + 0.1 for v in _TRANSFORM_VARS}}
        for split in ("input", "target")
    }


def _transform_batch_nonneg():
    """Batch with non-negative values for SqrtTransform."""
    return {
        split: {_TRANSFORM_SOURCE: {v: torch.rand(*_TRANSFORM_SHAPE) for v in _TRANSFORM_VARS}}
        for split in ("input", "target")
    }


def test_log_transform_empty_variables_transforms_all():
    """variables=[] expands to all variables and transforms every one of them."""
    batch = _transform_batch_positive()
    originals = {v: batch["input"][_TRANSFORM_SOURCE][v].clone() for v in _TRANSFORM_VARS}
    result = LogTransform(variables=[])(batch)
    for v in _TRANSFORM_VARS:
        assert not torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"variables=[] should have transformed {v}"
        )


def test_log_transform_partial_path_expands_to_matching_vars():
    """A partial path transforms exactly the variables under that hierarchy."""
    batch = _transform_batch_positive()
    prog_vars = [v for v in _TRANSFORM_VARS if "prognostic" in v]
    non_prog_vars = [v for v in _TRANSFORM_VARS if "prognostic" not in v]
    originals = {v: batch["input"][_TRANSFORM_SOURCE][v].clone() for v in _TRANSFORM_VARS}

    result = LogTransform(variables=[f"{_TRANSFORM_SOURCE}/prognostic"])(batch)

    for v in prog_vars:
        assert not torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"partial path should have transformed {v}"
        )
    for v in non_prog_vars:
        assert torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"partial path should NOT have transformed {v}"
        )


def test_sqrt_transform_empty_variables_transforms_all():
    """variables=[] expands to all variables and transforms every one of them."""
    batch = _transform_batch_nonneg()
    originals = {v: batch["input"][_TRANSFORM_SOURCE][v].clone() for v in _TRANSFORM_VARS}
    result = SqrtTransform(variables=[])(batch)
    for v in _TRANSFORM_VARS:
        assert not torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"variables=[] should have transformed {v}"
        )


def test_sqrt_transform_partial_path_expands_to_matching_vars():
    """A partial path transforms exactly the variables under that hierarchy."""
    batch = _transform_batch_nonneg()
    prog_vars = [v for v in _TRANSFORM_VARS if "prognostic" in v]
    non_prog_vars = [v for v in _TRANSFORM_VARS if "prognostic" not in v]
    originals = {v: batch["input"][_TRANSFORM_SOURCE][v].clone() for v in _TRANSFORM_VARS}

    result = SqrtTransform(variables=[f"{_TRANSFORM_SOURCE}/prognostic"])(batch)

    for v in prog_vars:
        assert not torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"partial path should have transformed {v}"
        )
    for v in non_prog_vars:
        assert torch.allclose(result["input"][_TRANSFORM_SOURCE][v].float(), originals[v].float()), (
            f"partial path should NOT have transformed {v}"
        )
