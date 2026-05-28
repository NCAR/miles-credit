"""test_preblock.py — unit tests for credit.preblock modules."""

import numpy as np
import pytest
import torch
import xarray as xr

try:
    from bridgescaler.distributed_tensor import DStandardScalerTensor
    from bridgescaler import save_scaler_dict, scale_var_dict
    from credit.preblock.scaler import BridgeScalerTransformer

    _BRIDGESCALER_AVAILABLE = True
except (ImportError, Exception):
    _BRIDGESCALER_AVAILABLE = False

from credit.preblock.regrid import Regridder


def create_synthetic_data() -> dict:
    """
    Creates synthetic data as a nested dictionary of torch tensors.

    Structure: data[data_type][source][var_name]
    - data_type: "input" | "target"
    - source: "Test_ERA5"
    - var_name: "Test_ERA5/pronostic/3d/T" | ...
    - tensor shape: (100, 16, 1, 8, 8)
    """
    shape = (100, 16, 1, 8, 8)
    var_names = [
        "Test_ERA5/pronostic/3d/T",
        "Test_ERA5/pronostic/3d/U",
        "Test_ERA5/pronostic/3d/V",
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
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = create_synthetic_data()
    result = regrid(batch)
    for split in ("input", "target"):
        assert result[split]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"].shape == (100, 16, 1, n_dst_lat, n_dst_lon)


def test_regrid_uniform_input(weight_file):
    """Block-average regrid: uniform input maps to uniform output of the same value."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    batch = {"input": {"Test_ERA5": {"Test_ERA5/pronostic/3d/T": torch.ones(1, 1, 1, n_src_lat, n_src_lon)}}}
    result = regrid(batch)
    assert torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
        torch.ones(1, 1, 1, n_dst_lat, n_dst_lon),
        atol=1e-5,
    )


def test_regrid_reshape_false(weight_file):
    """reshape_to_xy=False returns a flat (prod(lead_dims), n_b) tensor."""
    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables, reshape_to_xy=False)
    batch = create_synthetic_data()
    result = regrid(batch)
    assert result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"].shape == (100 * 16 * 1, n_dst_lat * n_dst_lon)


def test_regrid_flip_axis(weight_file):
    """flip_axis is applied to the input before regridding."""
    import copy

    path, n_src_lat, n_src_lon, n_dst_lat, n_dst_lon = weight_file
    variables = ["Test_ERA5/pronostic/3d/T"]
    regrid = Regridder(path, variables=variables)
    regrid_flip = Regridder(path, variables=variables, flip_axis=[-1])
    batch = create_synthetic_data()
    result = regrid(copy.deepcopy(batch))
    result_flip = regrid_flip(copy.deepcopy(batch))
    assert not torch.allclose(
        result["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
        result_flip["input"]["Test_ERA5"]["Test_ERA5/pronostic/3d/T"],
    )


_skip_bridgescaler = pytest.mark.skipif(
    not _BRIDGESCALER_AVAILABLE,
    reason="bridgescaler not available in this environment",
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

# VAR_NAMES = ["era5/pronostic/3d/T", "era5/pronostic/3d/U", "era5/pronostic/3d/V"]


@_skip_bridgescaler
def test_scaler_output_shape(scaler_file):
    """Transform preserves the input tensor shape for every variable."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransformer(scaler_path=path, variables=list(variables), method="transform")
    original_shapes = {v: data["input"]["Test_ERA5"][v].shape for v in variables}
    result = scaler(data)
    for v in variables:
        assert result["input"]["Test_ERA5"][v].shape == original_shapes[v]


@_skip_bridgescaler
def test_scaler_transform_changes_values(scaler_file):
    """Transform produces different values than the raw input."""
    path, variables, data = scaler_file
    scaler = BridgeScalerTransformer(scaler_path=path, variables=list(variables), method="transform")
    var = list(variables)[0]
    original = data["input"]["Test_ERA5"][var].clone()
    result = scaler(data)
    assert not torch.allclose(result["input"]["Test_ERA5"][var].float(), original.float())


@_skip_bridgescaler
def test_scaler_round_trip(scaler_file):
    """transform followed by inverse recovers the original tensor."""
    path, variables, data = scaler_file
    var_list = list(variables)
    fwd = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="transform")
    inv = BridgeScalerTransformer(scaler_path=path, variables=var_list, method="inverse_transform")
    var = var_list[0]
    original = data["input"]["Test_ERA5"][var].clone()
    data = fwd(data)
    data = inv(data)
    assert torch.allclose(data["input"]["Test_ERA5"][var].float(), original.float(), atol=1e-5)


# ---------------------------------------------------------------------------
# build_preblocks and build_postblocks — two-section format enforcement
# ---------------------------------------------------------------------------


class TestBuildPreblocks:
    """Tests for build_preblocks() config validation and section selection."""

    def test_flat_format_raises_value_error(self):
        """Old flat format (no ic_only/per_step sections) raises ValueError."""
        from credit.preblock import build_preblocks

        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_preblocks({"concat": {"type": "concat"}})

    def test_unknown_section_key_raises(self):
        """A key that is neither ic_only nor per_step raises ValueError."""
        from credit.preblock import build_preblocks

        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_preblocks({"per_step": {}, "bad_section": {}})

    def test_valid_two_section_per_step_builds(self):
        """per_step section builds an nn.ModuleDict with the named block."""
        import torch.nn as nn
        from credit.preblock import build_preblocks

        preblocks = build_preblocks({"per_step": {"concat": {"type": "concat"}}}, phase="per_step")
        assert isinstance(preblocks, nn.ModuleDict)
        assert "concat" in preblocks

    def test_valid_two_section_ic_only_builds(self):
        """ic_only section builds an nn.ModuleDict with the named block."""
        import torch.nn as nn
        from credit.preblock import build_preblocks

        preblocks = build_preblocks({"ic_only": {"concat": {"type": "concat"}}}, phase="ic_only")
        assert isinstance(preblocks, nn.ModuleDict)
        assert "concat" in preblocks

    def test_empty_config_returns_empty_module_dict(self):
        """Empty config builds an empty ModuleDict without error."""
        import torch.nn as nn
        from credit.preblock import build_preblocks

        preblocks = build_preblocks({}, phase="per_step")
        assert isinstance(preblocks, nn.ModuleDict)
        assert len(preblocks) == 0

    def test_missing_phase_returns_empty_module_dict(self):
        """Requesting a phase absent from the config returns an empty ModuleDict."""
        from credit.preblock import build_preblocks

        # ic_only is configured but per_step is requested
        preblocks = build_preblocks({"ic_only": {"concat": {"type": "concat"}}}, phase="per_step")
        assert len(preblocks) == 0

    def test_invalid_phase_raises(self):
        """An unrecognized phase name raises ValueError."""
        from credit.preblock import build_preblocks

        with pytest.raises(ValueError, match="phase must be one of"):
            build_preblocks({}, phase="invalid_phase")


class TestBuildPostblocks:
    """Tests for build_postblocks() config validation (mirrors build_preblocks)."""

    def test_flat_format_raises_value_error(self):
        """Old flat postblock format raises ValueError."""
        from credit.postblock import build_postblocks

        with pytest.raises(ValueError, match="unexpected top-level keys"):
            build_postblocks({"reconstruct": {"type": "reconstruct"}})

    def test_valid_two_section_per_step_builds(self):
        """per_step section builds a ModuleDict with the named block."""
        import torch.nn as nn
        from credit.postblock import build_postblocks

        postblocks = build_postblocks({"per_step": {"reconstruct": {"type": "reconstruct"}}}, phase="per_step")
        assert isinstance(postblocks, nn.ModuleDict)
        assert "reconstruct" in postblocks

    def test_empty_config_returns_empty_module_dict(self):
        from credit.postblock import build_postblocks

        postblocks = build_postblocks({}, phase="per_step")
        assert len(postblocks) == 0


# ---------------------------------------------------------------------------
# ConcatToTensor — channel ordering and channel map correctness
# ---------------------------------------------------------------------------


class TestConcatToTensorChannelOrder:
    """ConcatToTensor must sort channels by FIELD_TYPE_RANK regardless of insertion order."""

    def test_channel_order_follows_field_type_rank(self):
        """prognostic(0) < static(1) < dynamic_forcing(2) regardless of dict insertion order."""
        from credit.preblock.concat import ConcatToTensor

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
        from credit.preblock.concat import ConcatToTensor

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
        from credit.preblock.concat import ConcatToTensor

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
        from credit.preblock.concat import ConcatToTensor

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
        from credit.preblock import apply_preblocks, build_preblocks

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
        from credit.preblock import apply_preblocks, build_preblocks

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
        from credit.preblock import apply_preblocks, build_preblocks

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
        from credit.preblock import apply_preblocks, build_preblocks

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
        from credit.preblock import apply_preblocks, build_preblocks

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
