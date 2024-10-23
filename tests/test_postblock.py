import pytest
import yaml
import os

import torch
from credit.models.crossformer import CrossFormer
from credit.postblock import PostBlock, Backscatter_FCNN
from credit.postblock import SKEBS, TracerFixer, GlobalMassFixer, GlobalEnergyFixer
from credit.parser import CREDIT_main_parser

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

@pytest.mark.skip(reason="need to have model weights, level_info, and surface area to test on gh")
def test_SKEBS_integration():
    '''
    integration testing to make sure everything goes on GPU, is loaded properly etc
    requires loading weights
    '''
    config = os.path.join(CONFIG_FILE_DIR, "example_skebs.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = CREDIT_main_parser(conf) # parser will copy model configs to post_conf
    post_conf = conf['model']['post_conf']
    
    image_height = post_conf["model"]["image_height"]
    image_width = post_conf["model"]["image_width"]
    channels = post_conf["model"]["channels"]
    levels = post_conf["model"]["levels"]
    surface_channels = post_conf["model"]["surface_channels"]
    output_only_channels = post_conf["model"]["output_only_channels"]
    input_only_channels = post_conf["model"]["input_only_channels"]
    frames = post_conf["model"]["frames"]
    sp_index = post_conf["skebs"]["SP_ind"]

    in_channels = channels * levels + surface_channels + input_only_channels
    x = torch.randn(2, in_channels, frames, image_height, image_width)
    out_channels = channels * levels + surface_channels + output_only_channels
    y_pred = torch.randn(2, out_channels, frames, image_height, image_width)
    y_pred[:, sp_index] = torch.ones_like(y_pred[:, sp_index]) * 1013

    model = CrossFormer(**conf["model"])
    model.to("cpu")
    model = model.load_model(conf)
    pred = model(x)

    assert pred.shape == y_pred.shape

@pytest.mark.skip(reason="need to have model weights, level_info, and surface area to test on gh")
def test_SKEBS_rand():
    ''' unit test for CPU. testing that values make sense
    '''
    config = os.path.join(CONFIG_FILE_DIR, "example_skebs.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = CREDIT_main_parser(conf) # parser will copy model configs to post_conf
    post_conf = conf['model']['post_conf']
    
    image_height = post_conf["model"]["image_height"]
    image_width = post_conf["model"]["image_width"]
    channels = post_conf["model"]["channels"]
    levels = post_conf["model"]["levels"]
    surface_channels = post_conf["model"]["surface_channels"]
    output_only_channels = post_conf["model"]["output_only_channels"]
    input_only_channels = post_conf["model"]["input_only_channels"]
    frames = post_conf["model"]["frames"]
    sp_index = post_conf["skebs"]["SP_ind"]

    in_channels = channels * levels + surface_channels + input_only_channels
    x = torch.randn(2, in_channels, frames, image_height, image_width)
    out_channels = channels * levels + surface_channels + output_only_channels
    y_pred = torch.randn(2, out_channels, frames, image_height, image_width)
    y_pred[:, sp_index] = torch.ones_like(y_pred[:, sp_index]) * 1013

    postblock = PostBlock(post_conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    input_dict = {"x": x,
                  "y_pred": y_pred}

    skebs_pred = postblock(input_dict)

    assert skebs_pred.shape == y_pred.shape
    assert not torch.isnan(skebs_pred).any()

def test_SKEBS_backscatter():
    config = os.path.join(CONFIG_FILE_DIR, "example_skebs.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = CREDIT_main_parser(conf) # parser will copy model configs to post_conf
    post_conf = conf['model']['post_conf']
    
    image_height = post_conf["model"]["image_height"]
    image_width = post_conf["model"]["image_width"]
    channels = post_conf["model"]["channels"]
    levels = post_conf["model"]["levels"]
    surface_channels = post_conf["model"]["surface_channels"]
    output_only_channels = post_conf["model"]["output_only_channels"]
    input_only_channels = post_conf["model"]["input_only_channels"]
    frames = post_conf["model"]["frames"]
    sp_index = post_conf["skebs"]["SP_ind"]

    in_channels = channels * levels + surface_channels + input_only_channels
    x = torch.randn(2, in_channels, frames, image_height, image_width)
    out_channels = channels * levels + surface_channels + output_only_channels
    y_pred = torch.randn(2, out_channels, frames, image_height, image_width)
    y_pred[:, sp_index] = torch.ones_like(y_pred[:, sp_index]) * 1013

    model = Backscatter_FCNN(out_channels, levels)

    pred = model(y_pred)

    # assert pred.shape == y_pred.shape
    # assert not torch.isnan(skebs_pred).any()

def test_TracerFixer_rand():
    '''
    This function provides a functionality test on 
    TracerFixer at credit.postblock
    '''
    
    # initialize post_conf, turn-off other blocks
    conf = {"post_conf": {"skebs": {'activate': False}}}
    conf['post_conf']['global_mass_fixer'] = {'activate': False}
    conf['post_conf']['global_energy_fixer'] = {'activate': False}

    # tracer fixer specs
    conf['post_conf']['tracer_fixer'] = {'activate': True, 'denorm': False}
    conf['post_conf']['tracer_fixer']['tracer_inds'] = [0,]
    conf['post_conf']['tracer_fixer']['tracer_thres'] = [0,]

    # a random tensor with neg values
    input_tensor = -999*torch.randn((1, 1, 10, 10))

    # initialize postblock for 'TracerFixer' only
    postblock = PostBlock(**conf)

    # verify that TracerFixer is registered in the postblock
    assert any([isinstance(module, TracerFixer) for module in postblock.modules()])
    
    input_dict = {'y_pred': input_tensor}
    output_tensor = postblock(input_dict)

    # verify negative values
    assert output_tensor.min() >= 0

def test_GlobalMassFixer_rand():
    '''
    This function provides a I/O size test on 
    GlobalMassFixer at credit.postblock
    '''
    # initialize post_conf, turn-off other blocks
    conf = {'post_conf': {'skebs': {'activate': False}}}
    conf['post_conf']['tracer_fixer'] = {'activate': False}
    conf['post_conf']['global_energy_fixer'] = {'activate': False}
    
    # global mass fixer specs
    conf['post_conf']['global_mass_fixer'] = {
        'activate': True, 
        'denorm': False, 
        'midpoint': False,
        'simple_demo': True, 
        'fix_level_num': 3,
        'q_inds': [0, 1, 2, 3, 4, 5, 6],
        'precip_ind': 7,
        'evapor_ind': 8
    }
    
    # data specs
    conf['post_conf']['data'] = {'lead_time_periods': 6}
    
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


def test_GlobalEnergyFixer_rand():
    '''
    This function provides a I/O size test on 
    GlobalEnergyFixer at credit.postblock
    '''
    # turn-off other blocks
    conf = {'post_conf': {'skebs': {'activate': False}}}
    conf['post_conf']['tracer_fixer'] = {'activate': False}
    conf['post_conf']['global_mass_fixer'] = {'activate': False}
    
    # global energy fixer specs
    conf['post_conf']['global_energy_fixer'] = {
        'activate': True,
        'simple_demo': True,
        'denorm': False,
        'midpoint': False,
        'T_inds': [0, 1, 2, 3, 4, 5, 6],
        'q_inds': [0, 1, 2, 3, 4, 5, 6],
        'U_inds': [0, 1, 2, 3, 4, 5, 6],
        'V_inds': [0, 1, 2, 3, 4, 5, 6],
        'TOA_rad_inds': [7, 8],
        'surf_rad_inds': [7, 8],
        'surf_flux_inds': [7, 8]}
    
    conf['post_conf']['data'] = {'lead_time_periods': 6}
    
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


def test_SKEBS_era5():
    """
    todo after implementation
    """
    pass

if __name__ == "__main__":
    # test_SKEBS_integration()
    test_SKEBS_rand()