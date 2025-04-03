"""test_postblock.py provides I/O size tests.

-------------------------------------------------------
Content:
"""

import torch
from credit.postblock import PostBlock
from credit.postblock import (
    SKEBS,
    TracerFixer,
    GlobalMassFixer,
    GlobalWaterFixer,
    GlobalEnergyFixer,
)


def test_SKEBS_rand():
    """Provides an I/O size test on SKEBS at credit.postblock."""
    image_width = 100
    conf = {
        "post_conf": {
            "skebs": {"activate": True},
            "model": {
                "image_width": image_width,
            },
        }
    }
    conf["post_conf"]["tracer_fixer"] = {"activate": False}
    conf["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf["post_conf"]["global_water_fixer"] = {"activate": False}
    conf["post_conf"]["global_energy_fixer"] = {"activate": False}

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    input_dict = {"y_pred": input_tensor}

    y_pred = postblock(input_dict)

    assert y_pred.shape == input_tensor.shape


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
    assert any(
        [isinstance(module, GlobalEnergyFixer) for module in postblock.modules()]
    )

    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))

    input_dict = {"y_pred": y_pred, "x": x}
    # corrected output
    y_pred_fix = postblock(input_dict)

    assert y_pred_fix.shape == y_pred.shape


def test_SKEBS_era5():
    """test_SKEBS_era5."""
    pass
