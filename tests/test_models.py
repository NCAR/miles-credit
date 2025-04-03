"""Tests for models.py."""

import os
import yaml

import torch

from credit.models import load_model
from credit.models.unet import SegmentationModel
from credit.models.crossformer import CrossFormer
from credit.models.fuxi import Fuxi
from credit.parser import credit_main_parser

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-2]), "config"
)


def test_unet():
    """Test the unet model."""
    # load config
    config = os.path.join(CONFIG_FILE_DIR, "unet_1dg_test.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf)
    model = load_model(conf)

    assert isinstance(model, SegmentationModel)

    image_height = conf["model"]["image_height"]
    image_width = conf["model"]["image_width"]
    variables = len(conf["data"]["variables"])
    levels = conf["model"]["levels"]
    frames = conf["model"]["frames"]
    surface_variables = len(conf["data"]["surface_variables"])
    input_only_variables = len(conf["data"]["static_variables"]) + len(
        conf["data"]["dynamic_forcing_variables"]
    )
    output_only_variables = conf["model"]["output_only_channels"]

    in_channels = int(variables * levels + surface_variables + input_only_variables)
    out_channels = int(variables * levels + surface_variables + output_only_variables)

    assert in_channels != out_channels

    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width)

    y_pred = model(input_tensor)

    assert y_pred.shape == torch.Size([1, out_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()


def test_crossformer():
    """Test the crossformer model."""
    # load config
    config = os.path.join(CONFIG_FILE_DIR, "wxformer_1dg_test.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf)
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
    assert y_pred.shape == torch.Size(
        [1, in_channels - input_only_channels, 1, image_height, image_width]
    )
    assert not torch.isnan(y_pred).any()


def test_fuxi():
    """Test the I/O size of the Fuxi torch model to ensure that the input/output dimensions match the expected configuration.

    This test verifies the following:
    1. Correct loading and parsing of the model configuration file.
    2. Construction of the input tensor with the appropriate number of channels, frames, and spatial dimensions.
    3. Successful instantiation of the Fuxi model.
    4. The output tensor produced by the model has the expected shape, including the correct number of channels, height, width, and no NaN values.

    Test steps:
    -----------
    1. Load the model configuration from a YAML file.
    2. Parse the configuration to extract model-related parameters such as image dimensions, channels, and levels.
    3. Calculate the number of input and output channels based on the configuration.
    4. Create a random input tensor with the specified size and transfer it to the appropriate device (GPU or CPU).
    5. Load the Fuxi model and ensure it is an instance of the `Fuxi` class.
    6. Perform a forward pass with the input tensor and check the output tensor's shape.
    7. Assert that the output tensor has the correct size and contains no NaN values.

    Assertions:
    -----------
    - The model is an instance of the Fuxi class.
    - The output tensor has the correct shape: [batch_size, out_channels, 1, image_height, image_width].
    - The output tensor contains no NaN values.

    Raises
    ------
    AssertionError if any of the checks fail.

    """
    config = os.path.join(CONFIG_FILE_DIR, "fuxi_1deg_test.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    # handle config args
    conf = credit_main_parser(conf)

    image_height = conf["model"]["image_height"]
    image_width = conf["model"]["image_width"]
    channels = conf["model"]["channels"]
    levels = conf["model"]["levels"]
    surface_channels = conf["model"]["surface_channels"]
    input_only_channels = conf["model"]["input_only_channels"]
    output_only_channels = conf["model"]["output_only_channels"]
    frames = conf["model"]["frames"]

    in_channels = channels * levels + surface_channels + input_only_channels
    out_channels = channels * levels + surface_channels + output_only_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width).to(
        device
    )

    model = load_model(conf).to(device)
    assert isinstance(model, Fuxi)

    y_pred = model(input_tensor)
    assert y_pred.shape == torch.Size([1, out_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()


if __name__ == "__main__":
    test_unet()
    # test_crossformer()
