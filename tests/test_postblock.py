"""test_postblock.py — tests for credit.postblock modules.

Covers: gen1 physics fixers (TracerFixer, GlobalMassFixer, GlobalWaterFixer,
GlobalEnergyFixer, GlobalEnergyFixerUpDown), Reconstruct, ExpTransform,
SquareTransform, and BridgeScalerTransform (postblock scaler).
"""

import yaml
import os
import logging

import pytest
import torch
from bridgescaler.distributed_tensor import DStandardScalerTensor
from bridgescaler import save_scaler_dict, scale_var_dict
from credit.postblock.gen1 import (
    GlobalWaterFixer,
    PostBlock,
    TracerFixer,
    GlobalMassFixer,
    GlobalEnergyFixer,
    GlobalEnergyFixerUpDown,
)
from credit.postblock.scaler import BridgeScalerTransform as PostScaler
from credit.preblock._utils import _flatten_spatial_tensors
from credit.postblock.exp import ExpTransform
from credit.postblock.reconstruct import Reconstruct
from credit.postblock.square import SquareTransform
from credit.preblock.log import LogTransform
from credit.preblock.sqrt import SqrtTransform
from credit.skebs import BackscatterFCNN
from credit.parser import credit_main_parser


CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]), "config")


def test_SKEBS_backscatter():
    config = os.path.join(CONFIG_FILE_DIR, "example-v2026.1.0.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf["model"]["post_conf"]["activate"] = True
    conf = credit_main_parser(conf)  # parser will copy model configs to post_conf
    post_conf = conf["model"]["post_conf"]

    image_height = post_conf["model"]["image_height"]
    image_width = post_conf["model"]["image_width"]
    channels = post_conf["model"]["channels"]
    levels = post_conf["model"]["levels"]
    surface_channels = post_conf["model"]["surface_channels"]
    output_only_channels = post_conf["model"]["output_only_channels"]
    frames = post_conf["model"]["frames"]

    out_channels = channels * levels + surface_channels + output_only_channels
    y_pred = torch.randn(2, out_channels, frames, image_height, image_width)

    model = BackscatterFCNN(out_channels, levels)

    pred = model(y_pred)

    target_shape = list(y_pred.shape)
    target_shape[1] = levels
    assert list(pred.shape) == target_shape
    assert not torch.isnan(pred).any()


def test_TracerFixer_rand():
    """Provides an I/O size test on TracerFixer at credit.postblock."""
    # initialize post_conf, turn-off other blocks
    conf = {"post_conf": {"skebs": {"activate": False}}}
    conf["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf["post_conf"]["global_water_fixer"] = {"activate": False}
    conf["post_conf"]["global_energy_fixer"] = {"activate": False}

    # tracer fixer specs
    conf["post_conf"]["tracer_fixer"] = {"activate": True, "denorm": False}
    conf["post_conf"]["tracer_fixer"]["tracer_inds"] = [
        0,
    ]
    conf["post_conf"]["tracer_fixer"]["tracer_thres"] = [
        0,
    ]

    # a random tensor with neg values
    input_tensor = -999 * torch.randn((1, 1, 10, 10))

    # initialize postblock for 'TracerFixer' only
    postblock = PostBlock(**conf)

    # verify that TracerFixer is registered in the postblock
    assert any(isinstance(module, TracerFixer) for module in postblock.modules())

    input_dict = {"y_pred": input_tensor}
    output_tensor = postblock(input_dict)

    # verify all values are non-negative after clamping
    assert output_tensor.min() >= 0


def test_GlobalMassFixer_rand():
    """Provides an I/O size test on GlobalMassFixer at credit.postblock."""
    # initialize post_conf, turn-off other blocks
    conf = {"post_conf": {"skebs": {"activate": False}}}
    conf["post_conf"]["tracer_fixer"] = {"activate": False}
    conf["post_conf"]["global_water_fixer"] = {"activate": False}
    conf["post_conf"]["global_energy_fixer"] = {"activate": False}

    # global mass fixer specs
    conf["post_conf"]["global_mass_fixer"] = {
        "activate": True,
        "activate_outside_model": False,
        "denorm": False,
        "grid_type": "pressure",
        "midpoint": False,
        "simple_demo": True,
        "fix_level_num": 3,
        "q_inds": [0, 1, 2, 3, 4, 5, 6],
    }

    # data specs
    conf["post_conf"]["data"] = {"lead_time_periods": 6}

    # initialize postblock
    postblock = PostBlock(**conf)

    # verify that GlobalMassFixer is registered in the postblock
    assert any(isinstance(module, GlobalMassFixer) for module in postblock.modules())

    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))

    input_dict = {"y_pred": y_pred, "x": x}

    # corrected output
    y_pred_fix = postblock(input_dict)

    # verify `y_pred_fix` and `y_pred` has the same size
    assert y_pred_fix.shape == y_pred.shape


def test_GlobalWaterFixer_rand():
    """Provides an I/O size test on GlobalWaterFixer at credit.postblock."""
    # initialize post_conf, turn-off other blocks
    conf = {"post_conf": {"skebs": {"activate": False}}}
    conf["post_conf"]["tracer_fixer"] = {"activate": False}
    conf["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf["post_conf"]["global_energy_fixer"] = {"activate": False}

    # global water fixer specs
    conf["post_conf"]["global_water_fixer"] = {
        "activate": True,
        "activate_outside_model": False,
        "denorm": False,
        "grid_type": "pressure",
        "midpoint": False,
        "simple_demo": True,
        "fix_level_num": 3,
        "q_inds": [0, 1, 2, 3, 4, 5, 6],
        "precip_ind": 7,
        "evapor_ind": 8,
    }

    # data specs
    conf["post_conf"]["data"] = {"lead_time_periods": 6}

    # initialize postblock
    postblock = PostBlock(**conf)

    # verify that GlobalWaterFixer is registered in the postblock
    assert any(isinstance(module, GlobalWaterFixer) for module in postblock.modules())

    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))

    input_dict = {"y_pred": y_pred, "x": x}

    # corrected output
    y_pred_fix = postblock(input_dict)

    # verify `y_pred_fix` and `y_pred` has the same size
    assert y_pred_fix.shape == y_pred.shape


def test_GlobalEnergyFixer_rand():
    """Provides an I/O size test on GlobalEnergyFixer at credit.postblock."""
    # turn-off other blocks
    conf = {"post_conf": {"skebs": {"activate": False}}}
    conf["post_conf"]["tracer_fixer"] = {"activate": False}
    conf["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf["post_conf"]["global_water_fixer"] = {"activate": False}

    # global energy fixer specs
    conf["post_conf"]["global_energy_fixer"] = {
        "activate": True,
        "activate_outside_model": False,
        "simple_demo": True,
        "denorm": False,
        "grid_type": "pressure",
        "midpoint": False,
        "T_inds": [0, 1, 2, 3, 4, 5, 6],
        "q_inds": [0, 1, 2, 3, 4, 5, 6],
        "U_inds": [0, 1, 2, 3, 4, 5, 6],
        "V_inds": [0, 1, 2, 3, 4, 5, 6],
        "TOA_rad_inds": [7, 8],
        "surf_rad_inds": [7, 8],
        "surf_flux_inds": [7, 8],
    }

    conf["post_conf"]["data"] = {"lead_time_periods": 6}

    # initialize postblock
    postblock = PostBlock(**conf)

    # verify that GlobalEnergyFixer is registered in the postblock
    assert any(isinstance(module, GlobalEnergyFixer) for module in postblock.modules())

    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))

    input_dict = {"y_pred": y_pred, "x": x}
    # corrected output
    y_pred_fix = postblock(input_dict)

    assert y_pred_fix.shape == y_pred.shape


def test_GlobalEnergyFixerUpDown_rand():
    """Provides an I/O size and registration test on GlobalEnergyFixerUpDown."""
    # demo grid: 7 pressure levels, midpoint=True → 6 midpoint levels
    LEV = 6

    conf = {"post_conf": {"skebs": {"activate": False}}}
    conf["post_conf"]["tracer_fixer"] = {"activate": False}
    conf["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf["post_conf"]["global_water_fixer"] = {"activate": False}
    conf["post_conf"]["global_energy_fixer"] = {"activate": False}

    conf["post_conf"]["global_energy_fixer_updown"] = {
        "activate": True,
        "activate_outside_model": False,
        "simple_demo": True,
        "midpoint": True,
        "denorm": False,
        "T_inds": [0, LEV - 1],
        "q_inds": [LEV, 2 * LEV - 1],
        "U_inds": [2 * LEV, 3 * LEV - 1],
        "V_inds": [3 * LEV, 4 * LEV - 1],
        "TOA_down_solar_ind": 4 * LEV,
        "TOA_up_solar_ind": 4 * LEV + 1,
        "TOA_up_OLR_ind": 4 * LEV + 2,
        "surf_down_solar_ind": 4 * LEV + 3,
        "surf_up_solar_ind": 4 * LEV + 4,
        "surf_down_LW_ind": 4 * LEV + 5,
        "surf_up_LW_ind": 4 * LEV + 6,
        "surf_SH_ind": 4 * LEV + 7,
        "surf_LH_ind": 4 * LEV + 8,
    }

    conf["post_conf"]["data"] = {"lead_time_periods": 6}

    postblock = PostBlock(**conf)

    assert any(isinstance(m, GlobalEnergyFixerUpDown) for m in postblock.modules())

    N_VARS = 4 * LEV + 9
    x = torch.randn((1, 4 * LEV, 2, 10, 18))
    y_pred = torch.randn((1, N_VARS, 1, 10, 18))

    y_pred_fix = postblock({"y_pred": y_pred, "x": x})

    assert y_pred_fix.shape == y_pred.shape


class TestReconstruct:
    """Tests for credit.postblock.reconstruct.Reconstruct."""

    # 4-part key format: source/field_type/dim/varname
    KEY_3D = "Test_ARCOERA5/prognostic/3d/temperature"
    KEY_2D = "Test_ARCOERA5/prognostic/2d/surface_pressure"

    def _output_map(self):
        """Minimal channel map matching ConcatToTensor output format.

        Simulates: one 3D variable (4 levels) and one 2D surface variable.
        Slash-joined key format: source/field_type/dim/var_name (4 parts).
        """
        return {
            self.KEY_3D: {"slice": slice(0, 4), "orig_shape": (4, 1)},
            self.KEY_2D: {"slice": slice(4, 5), "orig_shape": (1, 1)},
        }

    def _metadata(self, output_map):
        return {"target": {"_channel_map": output_map}}

    def _batch_dict(self, y_pred, extra=None):
        """Minimal batch_dict as the caller would build before apply_postblocks."""
        d = {"y_pred": y_pred, "metadata": self._metadata(self._output_map())}
        if extra:
            d.update(extra)
        return d

    def test_nested_dict_structure(self):
        """Output is y_processed[source][var_key] — source then flat 4-part slash key."""
        result = Reconstruct()(self._batch_dict(torch.randn(2, 5, 8, 8)))

        pred = result["y_processed"]
        assert "Test_ARCOERA5" in pred
        assert self.KEY_3D in pred["Test_ARCOERA5"]
        assert self.KEY_2D in pred["Test_ARCOERA5"]

    def test_tensor_shapes_4d_input(self):
        """3D var → (B, n_levels, 1, H, W), 2D var → (B, 1, 1, H, W)."""
        B, H, W = 2, 8, 8
        result = Reconstruct()(self._batch_dict(torch.randn(B, 5, H, W)))

        pred = result["y_processed"]
        assert pred["Test_ARCOERA5"][self.KEY_3D].shape == (B, 4, 1, H, W)
        assert pred["Test_ARCOERA5"][self.KEY_2D].shape == (B, 1, 1, H, W)

    def test_5d_input_no_extra_dim(self):
        """5D y_pred (B, C, 1, H, W) produces the same shape as 4D — no spurious singleton."""
        B, H, W = 2, 8, 8
        y_pred_4d = torch.randn(B, 5, H, W)
        y_pred_5d = y_pred_4d.unsqueeze(2)  # (B, 5, 1, H, W)

        result_4d = Reconstruct()(self._batch_dict(y_pred_4d))
        result_5d = Reconstruct()(self._batch_dict(y_pred_5d))

        shape_4d = result_4d["y_processed"]["Test_ARCOERA5"][self.KEY_3D].shape
        shape_5d = result_5d["y_processed"]["Test_ARCOERA5"][self.KEY_3D].shape
        assert shape_4d == shape_5d == (B, 4, 1, H, W)

    def test_values_match_input_channels(self):
        """Reconstructed tensors contain exactly the channels sliced from y_pred."""
        B, H, W = 1, 4, 4
        y_pred = torch.randn(B, 5, H, W)
        result = Reconstruct()(self._batch_dict(y_pred))
        pred = result["y_processed"]

        assert torch.equal(
            pred["Test_ARCOERA5"][self.KEY_3D],
            y_pred[:, 0:4].unflatten(1, (4, 1)),
        )
        assert torch.equal(
            pred["Test_ARCOERA5"][self.KEY_2D],
            y_pred[:, 4:5].unflatten(1, (1, 1)),
        )

    def test_other_keys_pass_through(self):
        """Keys other than 'y_pred' are preserved unchanged."""
        raw = {"Test_ERA5": {"input": {}}}
        batch = self._batch_dict(torch.randn(1, 5, 4, 4), extra={"input": torch.zeros(1), "_raw": raw})
        result = Reconstruct()(batch)
        assert result["_raw"] is raw
        assert "input" in result

    def test_metadata_passthrough(self):
        """metadata dict is returned at the same key, unchanged."""
        batch = self._batch_dict(torch.randn(1, 5, 4, 4))
        original_meta = batch["metadata"]
        result = Reconstruct()(batch)
        assert result["metadata"] is original_meta


# ---------------------------------------------------------------------------
# ExpTransform and SquareTransform — round-trip and edge-case tests
# ---------------------------------------------------------------------------

_VAR = "era5/prognostic/3d/Q"
_SOURCE = "era5"


def _postblock_batch_dict(tensor, key="y_processed"):
    return {key: {_SOURCE: {_VAR: tensor}}}


def _preblock_batch(tensor):
    return {"input": {_SOURCE: {_VAR: tensor}}, "target": {_SOURCE: {_VAR: tensor.clone()}}}


def test_postblock_scaler_empty_variables_scales_all(tmp_path):
    """variables=[] in the postblock scaler scales all variables (not none)."""

    var_names = ["Test_ERA5/prognostic/3d/T", "Test_ERA5/prognostic/3d/U"]
    source = "Test_ERA5"
    shape = (4, 4, 1, 8, 8)
    # Fit on the full {data_type: {source: {var_key: tensor}}} structure — same as preblock scaler
    x_dict = {"target": {source: {v: torch.randn(*shape) for v in var_names}}}
    scaler_dict = scale_var_dict(x_dict, DStandardScalerTensor(channels_last=False), method="fit")
    path = str(tmp_path / "scaler.json")
    save_scaler_dict(scaler_dict, path)

    y = {source: {v: torch.randn(*shape) for v in var_names}}
    original = y[source][var_names[0]].clone()

    scaler = PostScaler(scaler_path=path, variables=[], method="transform")
    result = scaler({"y_processed": y})
    assert not torch.allclose(result["y_processed"][source][var_names[0]].float(), original.float()), (
        "empty variables list should scale all variables, not none"
    )


def test_postblock_scaler_partial_path_expansion(tmp_path):
    """A partial path in variables expands to all matching variables."""

    var_names = ["Test_ERA5/prognostic/3d/T", "Test_ERA5/prognostic/3d/U"]
    source = "Test_ERA5"
    shape = (4, 4, 1, 8, 8)
    # Fit on the full {data_type: {source: {var_key: tensor}}} structure — same as preblock scaler
    x_dict = {"target": {source: {v: torch.randn(*shape) for v in var_names}}}
    scaler_dict = scale_var_dict(x_dict, DStandardScalerTensor(channels_last=False), method="fit")
    path = str(tmp_path / "scaler.json")
    save_scaler_dict(scaler_dict, path)

    y = {source: {v: torch.randn(*shape) for v in var_names}}
    originals = {v: y[source][v].clone() for v in var_names}

    scaler = PostScaler(scaler_path=path, variables=[source], method="transform")
    result = scaler({"y_processed": y})
    for v in var_names:
        assert not torch.allclose(result["y_processed"][source][v].float(), originals[v].float()), (
            f"partial path '{source}' should have expanded to include {v}"
        )


# ---------------------------------------------------------------------------
# Postblock scaler — spatial_variables (grid-wise scaling)
# ---------------------------------------------------------------------------


def _save_spatial_target_scaler(path, target_dict, spatial_variables):
    """Fit and save a scaler on the ``"target"`` slice, flattening
    *spatial_variables* per-gridpoint first (mirrors the preblock fit path)."""
    x_dict = {"target": {src: {v: t.clone() for v, t in vs.items()} for src, vs in target_dict.items()}}
    x_flat, _ = _flatten_spatial_tensors(x_dict, list(spatial_variables))
    scaler_dict = scale_var_dict(x_flat, DStandardScalerTensor(channels_last=False), method="fit")
    save_scaler_dict(scaler_dict, path)


def test_postblock_scaler_spatial_round_trip(tmp_path):
    """A spatial variable in y_processed transforms and inverse-transforms back
    to its original values and shape, alongside an ordinary per-level variable."""
    source = "Test_ERA5"
    spatial_var = "Test_ERA5/prognostic/2d/SP"
    level_var = "Test_ERA5/prognostic/3d/T"
    B, L, H, W = 32, 4, 8, 8

    target = {source: {spatial_var: torch.randn(B, 1, 1, H, W), level_var: torch.randn(B, L, 1, H, W)}}
    path = str(tmp_path / "scaler.json")
    _save_spatial_target_scaler(path, target, [spatial_var])

    y = {source: {spatial_var: torch.randn(B, 1, 1, H, W), level_var: torch.randn(B, L, 1, H, W)}}
    original = {v: t.clone() for v, t in y[source].items()}

    fwd = PostScaler(scaler_path=path, variables=[], method="transform", spatial_variables=[spatial_var])
    inv = PostScaler(scaler_path=path, variables=[], method="inverse_transform", spatial_variables=[spatial_var])
    out = inv(fwd({"y_processed": y}))

    for v in (spatial_var, level_var):
        got = out["y_processed"][source][v]
        assert got.shape == original[v].shape, f"{v} shape changed across round trip"
        assert torch.allclose(got.float(), original[v].float(), atol=1e-4), f"{v} not recovered"


def test_postblock_scaler_spatial_variables_must_be_subset(tmp_path):
    """spatial_variables not covered by `variables` is a config error and raises."""
    source = "Test_ERA5"
    spatial_var = "Test_ERA5/prognostic/2d/SP"
    level_var = "Test_ERA5/prognostic/3d/T"
    target = {source: {spatial_var: torch.randn(2, 1, 1, 4, 4), level_var: torch.randn(2, 3, 1, 4, 4)}}
    path = str(tmp_path / "scaler.json")
    _save_spatial_target_scaler(path, target, [spatial_var])

    block = PostScaler(
        scaler_path=path,
        variables=[level_var],  # SP deliberately omitted
        method="transform",
        spatial_variables=[spatial_var],
    )
    y = {source: {v: t.clone() for v, t in target[source].items()}}
    with pytest.raises(ValueError, match="must also be selected"):
        block({"y_processed": y})


def test_exp_transform_round_trip_base_e():
    """LogTransform → ExpTransform recovers the original tensor (base e)."""
    x = torch.rand(2, 4, 1, 8, 8) + 1e-6
    logged = LogTransform(variables=[_VAR], base="e")(_preblock_batch(x))
    y = logged["input"][_SOURCE][_VAR]
    result = ExpTransform(variables=[_VAR], base="e")(_postblock_batch_dict(y))["y_processed"][_SOURCE][_VAR]
    assert torch.allclose(result, x, atol=1e-5)


def test_exp_transform_round_trip_base_2():
    """LogTransform → ExpTransform recovers the original tensor (base 2)."""
    x = torch.rand(2, 4, 1, 8, 8) + 1e-6
    logged = LogTransform(variables=[_VAR], base="2")(_preblock_batch(x))
    y = logged["input"][_SOURCE][_VAR]
    result = ExpTransform(variables=[_VAR], base="2")(_postblock_batch_dict(y))["y_processed"][_SOURCE][_VAR]
    assert torch.allclose(result, x, atol=1e-5)


def test_exp_transform_round_trip_base_10():
    """LogTransform → ExpTransform recovers the original tensor (base 10)."""
    x = torch.rand(2, 4, 1, 8, 8) + 1e-6
    logged = LogTransform(variables=[_VAR], base="10")(_preblock_batch(x))
    y = logged["input"][_SOURCE][_VAR]
    result = ExpTransform(variables=[_VAR], base="10")(_postblock_batch_dict(y))["y_processed"][_SOURCE][_VAR]
    assert torch.allclose(result, x, atol=1e-5)


def test_exp_transform_skips_missing_variable():
    """ExpTransform silently skips variables absent from the batch."""
    exp_block = ExpTransform(variables=["era5/prognostic/3d/MISSING"])
    result = exp_block(_postblock_batch_dict(torch.ones(1, 1, 1, 4, 4)))
    assert _VAR in result["y_processed"][_SOURCE]


def test_exp_transform_invalid_base():
    """ExpTransform raises ValueError for an unsupported base."""
    with pytest.raises(ValueError):
        ExpTransform(variables=[_VAR], base="7")


def test_exp_transform_custom_key():
    """ExpTransform respects a non-default key."""
    x = torch.rand(1, 1, 1, 4, 4) + 1e-6
    logged = LogTransform(variables=[_VAR])(_preblock_batch(x))
    y = logged["input"][_SOURCE][_VAR]
    result = ExpTransform(variables=[_VAR], key="my_output")({"my_output": {_SOURCE: {_VAR: y}}})
    assert torch.allclose(result["my_output"][_SOURCE][_VAR], x, atol=1e-5)


def test_square_transform_round_trip():
    """SqrtTransform → SquareTransform recovers the original tensor."""
    x = torch.rand(2, 4, 1, 8, 8)
    sqrted = SqrtTransform(variables=[_VAR])(_preblock_batch(x))
    y = sqrted["input"][_SOURCE][_VAR]
    result = SquareTransform(variables=[_VAR])(_postblock_batch_dict(y))["y_processed"][_SOURCE][_VAR]
    assert torch.allclose(result, x, atol=1e-5)


def test_square_transform_skips_missing_variable():
    """SquareTransform silently skips variables absent from the batch."""
    square_block = SquareTransform(variables=["era5/prognostic/3d/MISSING"])
    result = square_block(_postblock_batch_dict(torch.ones(1, 1, 1, 4, 4)))
    assert _VAR in result["y_processed"][_SOURCE]


def test_square_transform_negative_input():
    """Squaring slightly negative values gives non-negative results."""
    y = torch.tensor([-0.01, 0.0, 0.1, 1.0]).reshape(1, 4, 1, 1, 1)
    result = SquareTransform(variables=[_VAR])(_postblock_batch_dict(y))["y_processed"][_SOURCE][_VAR]
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# ExpTransform and SquareTransform — lazy expansion (variables=[] and partial paths)
# ---------------------------------------------------------------------------

_LAZY_VARS = ["era5/prognostic/3d/Q", "era5/prognostic/3d/T", "era5/static/2d/Z"]
_LAZY_SOURCE = "era5"
_LAZY_SHAPE = (1, 1, 1, 4, 4)


def _lazy_postblock_batch(key="y_processed"):
    return {key: {_LAZY_SOURCE: {v: torch.rand(*_LAZY_SHAPE) + 0.1 for v in _LAZY_VARS}}}


def test_exp_transform_empty_variables_transforms_all():
    """variables=[] expands to all variables and transforms every one."""
    batch = _lazy_postblock_batch()
    originals = {v: batch["y_processed"][_LAZY_SOURCE][v].clone() for v in _LAZY_VARS}
    result = ExpTransform(variables=[])(batch)
    for v in _LAZY_VARS:
        assert not torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"variables=[] should have transformed {v}"
        )


def test_exp_transform_partial_path_expands_to_matching_vars():
    """A partial path transforms exactly the variables under that hierarchy."""
    batch = _lazy_postblock_batch()
    prog_vars = [v for v in _LAZY_VARS if "prognostic" in v]
    non_prog_vars = [v for v in _LAZY_VARS if "prognostic" not in v]
    originals = {v: batch["y_processed"][_LAZY_SOURCE][v].clone() for v in _LAZY_VARS}

    result = ExpTransform(variables=[f"{_LAZY_SOURCE}/prognostic"])(batch)

    for v in prog_vars:
        assert not torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"partial path should have transformed {v}"
        )
    for v in non_prog_vars:
        assert torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"partial path should NOT have transformed {v}"
        )


def test_square_transform_empty_variables_transforms_all():
    """variables=[] expands to all variables and transforms every one."""
    batch = {"y_processed": {_LAZY_SOURCE: {v: torch.rand(*_LAZY_SHAPE) for v in _LAZY_VARS}}}
    originals = {v: batch["y_processed"][_LAZY_SOURCE][v].clone() for v in _LAZY_VARS}
    result = SquareTransform(variables=[])(batch)
    for v in _LAZY_VARS:
        assert not torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"variables=[] should have transformed {v}"
        )


def test_square_transform_partial_path_expands_to_matching_vars():
    """A partial path transforms exactly the variables under that hierarchy."""
    batch = {"y_processed": {_LAZY_SOURCE: {v: torch.rand(*_LAZY_SHAPE) for v in _LAZY_VARS}}}
    prog_vars = [v for v in _LAZY_VARS if "prognostic" in v]
    non_prog_vars = [v for v in _LAZY_VARS if "prognostic" not in v]
    originals = {v: batch["y_processed"][_LAZY_SOURCE][v].clone() for v in _LAZY_VARS}

    result = SquareTransform(variables=[f"{_LAZY_SOURCE}/prognostic"])(batch)

    for v in prog_vars:
        assert not torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"partial path should have transformed {v}"
        )
    for v in non_prog_vars:
        assert torch.allclose(result["y_processed"][_LAZY_SOURCE][v].float(), originals[v].float()), (
            f"partial path should NOT have transformed {v}"
        )


if __name__ == "__main__":
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)
