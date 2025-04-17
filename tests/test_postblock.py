"""test_postblock.py provides I/O size tests.

-------------------------------------------------------
Content:
"""
import yaml
import os
import logging

import torch
from credit.postblock import GlobalWaterFixer, PostBlock
from credit.skebs import SKEBS, BackscatterFCNN
from credit.postblock import TracerFixer, GlobalMassFixer, GlobalEnergyFixer
from credit.parser import credit_main_parser


TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

def test_SKEBS_rand():
    """ unit test for CPU. testing that values make sense
    """
    config = os.path.join(CONFIG_FILE_DIR, "example-v2025.2.0.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf["data"]["save_loc_static"] = os.path.join(TEST_FILE_DIR, "data/level_info_test.nc")
    conf["data"]["mean_path"] = os.path.join(TEST_FILE_DIR, "data/mean_6h_1979_2018_16lev_0.25deg.nc")
    conf["data"]["std_path"] = os.path.join(TEST_FILE_DIR, "data/std_residual_6h_1979_2018_16lev_0.25deg.nc")
    conf['model']['post_conf']["activate"] = True

    conf['model']["post_conf"]["global_mass_fixer"] = {"activate": False}
    conf['model']["post_conf"]["global_water_fixer"] = {"activate": False}
    conf['model']["post_conf"]["global_energy_fixer"] = {"activate": False}
    conf['model']["post_conf"]["tracer_fixer"] = {"activate": False}

    conf['model']["post_conf"]["skebs"]["activate"] = True
    conf['model']["post_conf"]["skebs"]["dissipation_type"] = "uniform"

    conf['model']["post_conf"]["skebs"]["write_train_debug_files"] = False
    conf['model']["post_conf"]["skebs"]["write_rollout_debug_files"] = False

    image_height = 640 # needs to match level_info_test.nc
    image_width = 1280

    conf["model"]["image_height"] = image_height
    conf["model"]["image_width"] = image_width

    conf = credit_main_parser(conf) # parser will copy model configs to post_conf
    post_conf = conf['model']['post_conf']
    
   
    channels = post_conf["model"]["channels"]
    levels = post_conf["model"]["levels"]
    surface_channels = post_conf["model"]["surface_channels"]
    output_only_channels = post_conf["model"]["output_only_channels"]
    input_only_channels = post_conf["model"]["input_only_channels"]
    frames = post_conf["model"]["frames"]

    in_channels = channels * levels + surface_channels + input_only_channels
    x = torch.randn(2, in_channels, frames, image_height, image_width)
    out_channels = channels * levels + surface_channels + output_only_channels
    y_pred = torch.randn(2, out_channels, frames, image_height, image_width)

    post_conf["data"]["forecast_len"] = 2 # to turn on multistep
   
    postblock = PostBlock(post_conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    input_dict = {"x": x,
                "y_pred": y_pred}

    skebs_pred = postblock(input_dict)

    assert skebs_pred.shape == y_pred.shape
    assert not torch.isnan(skebs_pred).any()

def test_SKEBS_backscatter():
    config = os.path.join(CONFIG_FILE_DIR, "example-v2025.2.0.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf['model']['post_conf']["activate"] = True
    conf = credit_main_parser(conf) # parser will copy model configs to post_conf
    post_conf = conf['model']['post_conf']
    
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



if __name__ == "__main__":
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # test_SKEBS_integration()
    test_SKEBS_rand()
    # test_SKEBS_backscatter()
