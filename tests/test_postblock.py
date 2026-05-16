"""test_postblock.py provides I/O size tests.

-------------------------------------------------------
Content:
"""

import yaml
import os
import logging

import torch
from credit.postblock.gen1 import GlobalWaterFixer, PostBlock
from credit.skebs import BackscatterFCNN
from credit.postblock.gen1 import TracerFixer, GlobalMassFixer, GlobalEnergyFixer, GlobalEnergyFixerUpDown
from credit.parser import credit_main_parser


TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
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
    assert any([isinstance(module, TracerFixer) for module in postblock.modules()])

    input_dict = {"y_pred": input_tensor}
    output_tensor = postblock(input_dict)

    # verify negative values
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
    assert any([isinstance(module, GlobalMassFixer) for module in postblock.modules()])

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
    assert any([isinstance(module, GlobalWaterFixer) for module in postblock.modules()])

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
    assert any([isinstance(module, GlobalEnergyFixer) for module in postblock.modules()])

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

    assert any([isinstance(m, GlobalEnergyFixerUpDown) for m in postblock.modules()])

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
        d = {"prediction": y_pred, "metadata": self._metadata(self._output_map())}
        if extra:
            d.update(extra)
        return d

    def test_nested_dict_structure(self):
        """Output is prediction[source][var_key] — source then flat 4-part slash key."""
        from credit.postblock.reconstruct import Reconstruct

        result = Reconstruct()(self._batch_dict(torch.randn(2, 5, 8, 8)))

        pred = result["prediction"]
        assert "Test_ARCOERA5" in pred
        assert self.KEY_3D in pred["Test_ARCOERA5"]
        assert self.KEY_2D in pred["Test_ARCOERA5"]

    def test_tensor_shapes_4d_input(self):
        """3D var → (B, n_levels, 1, H, W), 2D var → (B, 1, 1, H, W)."""
        from credit.postblock.reconstruct import Reconstruct

        B, H, W = 2, 8, 8
        result = Reconstruct()(self._batch_dict(torch.randn(B, 5, H, W)))

        pred = result["prediction"]
        assert pred["Test_ARCOERA5"][self.KEY_3D].shape == (B, 4, 1, H, W)
        assert pred["Test_ARCOERA5"][self.KEY_2D].shape == (B, 1, 1, H, W)

    def test_5d_input_no_extra_dim(self):
        """5D y_pred (B, C, 1, H, W) produces the same shape as 4D — no spurious singleton."""
        from credit.postblock.reconstruct import Reconstruct

        B, H, W = 2, 8, 8
        y_pred_4d = torch.randn(B, 5, H, W)
        y_pred_5d = y_pred_4d.unsqueeze(2)  # (B, 5, 1, H, W)

        result_4d = Reconstruct()(self._batch_dict(y_pred_4d))
        result_5d = Reconstruct()(self._batch_dict(y_pred_5d))

        shape_4d = result_4d["prediction"]["Test_ARCOERA5"][self.KEY_3D].shape
        shape_5d = result_5d["prediction"]["Test_ARCOERA5"][self.KEY_3D].shape
        assert shape_4d == shape_5d == (B, 4, 1, H, W)

    def test_values_match_input_channels(self):
        """Reconstructed tensors contain exactly the channels sliced from y_pred."""
        from credit.postblock.reconstruct import Reconstruct

        B, H, W = 1, 4, 4
        y_pred = torch.randn(B, 5, H, W)
        result = Reconstruct()(self._batch_dict(y_pred))
        pred = result["prediction"]

        assert torch.equal(
            pred["Test_ARCOERA5"][self.KEY_3D],
            y_pred[:, 0:4].unflatten(1, (4, 1)),
        )
        assert torch.equal(
            pred["Test_ARCOERA5"][self.KEY_2D],
            y_pred[:, 4:5].unflatten(1, (1, 1)),
        )

    def test_other_keys_pass_through(self):
        """Keys other than 'prediction' are preserved unchanged."""
        from credit.postblock.reconstruct import Reconstruct

        raw = {"Test_ERA5": {"input": {}}}
        batch = self._batch_dict(torch.randn(1, 5, 4, 4), extra={"input": torch.zeros(1), "_raw": raw})
        result = Reconstruct()(batch)
        assert result["_raw"] is raw
        assert "input" in result

    def test_metadata_passthrough(self):
        """metadata dict is returned at the same key, unchanged."""
        from credit.postblock.reconstruct import Reconstruct

        batch = self._batch_dict(torch.randn(1, 5, 4, 4))
        original_meta = batch["metadata"]
        result = Reconstruct()(batch)
        assert result["metadata"] is original_meta


if __name__ == "__main__":
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)
