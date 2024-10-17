import os
import yaml

import torch

from credit.models import load_model
from credit.models.unet import SegmentationModel
from credit.models.crossformer import CrossFormer
from credit.parser import CREDIT_main_parser

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]),
                      "config")

def test_unet():
    #load config
    config = os.path.join(CONFIG_FILE_DIR, "unet_test.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    
    conf = CREDIT_main_parser(conf)
    model = load_model(conf)

    assert isinstance(model, SegmentationModel)

    image_height = conf["model"]["image_height"]
    image_width = conf["model"]["image_width"]
    variables = len(conf["data"]["variables"])
    levels = conf["model"]["levels"]
    frames = conf["model"]["frames"]
    surface_variables = len(conf["data"]["surface_variables"])
    static_variables = len(conf["data"]["static_variables"])

    in_channels = int(variables*levels + surface_variables + static_variables)
    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width)

    y_pred = model(input_tensor)

    assert y_pred.shape == torch.Size([1, in_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()

def test_crossformer(): 
    #load config
    config = os.path.join(CONFIG_FILE_DIR, "wxformer_1dg_test.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    image_height = conf["model"]["image_height"]
    image_width = conf["model"]["image_width"]

    channels = conf["model"]["channels"]
    levels = conf["model"]["levels"]
    surface_channels = conf["model"]["surface_channels"]
    input_only_channels = conf["model"]["input_only_channels"]
    frames = conf["model"]["frames"]

    in_channels = channels * levels + surface_channels + input_only_channels
    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width)

    model = load_model(conf)
    assert isinstance(model, CrossFormer)


    y_pred = model(input_tensor)
    assert y_pred.shape == torch.Size([1, in_channels - input_only_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()

# if __name__ == "__main__":
#     # test_unet()
#     test_crossformer()



