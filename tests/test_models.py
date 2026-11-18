"""Tests for models.py."""

import os
import yaml
import pytest

import torch

from credit.models import load_model, register_model
from credit.parser import load_custom_model_modules
from credit.models.unet import SegmentationModel
from credit.models.wxformer.crossformer import CrossFormer
from credit.models.fuxi import Fuxi
from credit.models.swin import SwinTransformerV2Cr
from credit.distributed import ignore_static_attn_mask_buffers
from credit.parser import credit_main_parser

TEST_FILE_DIR = "/".join(os.path.abspath(__file__).split("/")[:-1])
CONFIG_FILE_DIR = os.path.join(
    "/".join(os.path.abspath(__file__).split("/")[:-2]), "config/gen_1/applications/other_models/"
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
    input_only_variables = len(conf["data"]["static_variables"]) + len(conf["data"]["dynamic_forcing_variables"])
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
    assert y_pred.shape == torch.Size([1, in_channels - input_only_channels, 1, image_height, image_width])
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
    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width).to(device)

    model = load_model(conf).to(device)
    assert isinstance(model, Fuxi)

    y_pred = model(input_tensor)
    assert y_pred.shape == torch.Size([1, out_channels, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()


def test_register_model():
    """Test that register_model decorator adds a custom model to the registry."""
    from credit.models.base_model import BaseModel

    @register_model("test_custom_model", "Loading test custom model ...")
    class CustomModel(BaseModel):
        def __init__(self, hidden_dim=32):
            super().__init__()
            self.fc = torch.nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return self.fc(x)

    conf = {"model": {"type": "test_custom_model", "hidden_dim": 16}, "save_loc": "/tmp"}
    model = load_model(conf)

    assert isinstance(model, CustomModel)
    assert model.fc.in_features == 16


def test_load_custom_model_modules(tmp_path):
    """Test that load_custom_model_modules imports a file and registers its model."""
    module_file = tmp_path / "custom_models.py"
    module_file.write_text(
        "import torch.nn as nn\n"
        "from credit.models import register_model\n"
        "from credit.models.base_model import BaseModel\n"
        "\n"
        "@register_model('file_registered_model')\n"
        "class FileRegisteredModel(BaseModel):\n"
        "    def __init__(self, size=8):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(size, size)\n"
    )

    load_custom_model_modules({"custom_models": [str(module_file)]})

    conf = {"model": {"type": "file_registered_model", "size": 4}, "save_loc": "/tmp"}
    model = load_model(conf)
    assert model.fc.in_features == 4


def test_load_custom_model_modules_missing_file():
    """Test that a missing path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="custom_models"):
        load_custom_model_modules({"custom_models": ["/nonexistent/path/models.py"]})


def test_register_model_overwrite(caplog):
    """Test that registering a duplicate key logs a warning and overwrites."""
    import logging
    from credit.models.base_model import BaseModel

    @register_model("test_overwrite_model")
    class ModelV1(BaseModel):
        pass

    with caplog.at_level(logging.WARNING, logger="credit.models"):

        @register_model("test_overwrite_model")
        class ModelV2(BaseModel):
            pass

    assert any("test_overwrite_model" in msg for msg in caplog.messages)

    conf = {"model": {"type": "test_overwrite_model"}, "save_loc": "/tmp"}
    model = load_model(conf)
    assert isinstance(model, ModelV2)


def _small_swin(img_size, padding_conf=None):
    """Build a CPU-sized SwinTransformerV2Cr.

    The shipped swin config is 640x1280 / embed_dim 768 / depth 12, which is far too
    large for CI, so these tests construct the model directly. ``patch_size=4`` with
    ``window_size=5`` requires the *padded* image to be a multiple of 20 on both axes.

    Args:
        img_size (Tuple[int, int]): Unpadded (height, width).
        padding_conf (dict): Padding configuration, or None to disable.

    Returns:
        SwinTransformerV2Cr: The constructed model.
    """
    return SwinTransformerV2Cr(
        img_size=img_size,
        patch_size=4,
        window_size=5,
        channels=1,
        levels=2,
        surface_channels=1,
        input_only_channels=0,
        output_only_channels=0,
        frames=1,
        embed_dim=32,
        depths=(2,),
        num_heads=(2,),
        padding_conf=padding_conf,
    )


@pytest.mark.parametrize("pad_lat, pad_lon", [(4, 8), ([3, 5], [8, 8])], ids=["symmetric_int", "asymmetric_list"])
def test_swin_boundary_padding(pad_lat, pad_lon):
    """Swin with padding active must build, and round-trip back to the input grid.

    Regression test for two bugs: ``forward`` called ``self.padding_opt.pad``/``unpad``
    but ``__init__`` never constructed it, raising AttributeError on the first batch;
    and the padded image size was computed as ``int(2 * pad_lat)``, which assumed a
    symmetric scalar and raised TypeError on the ``[before, after]`` form that
    ``credit_main_parser`` always produces.
    """
    image_height, image_width = 32, 64
    padding_conf = {"activate": True, "mode": "earth", "pad_lat": pad_lat, "pad_lon": pad_lon}

    model = _small_swin((image_height, image_width), padding_conf)

    # both spellings add 8 rows and 16 columns, and the interior is restored on unpad
    assert model.img_size == (40, 80)

    input_tensor = torch.randn(1, model.in_chans, 1, image_height, image_width)
    y_pred = model(input_tensor)

    assert y_pred.shape == torch.Size([1, model.out_chans, 1, image_height, image_width])
    assert not torch.isnan(y_pred).any()


def test_swin_attn_mask_excluded_from_ddp_buffer_sync():
    """Static attention masks must land on the DDP ignore list by fully-qualified name.

    They derive from the window geometry alone, so every rank builds an identical
    copy. Left in, DDP re-broadcasts them on every forward pass — roughly 300 MiB of
    constants per iteration at the production 640x1280 depth-12 size.
    """
    model = _small_swin((40, 80))

    masks = [name for name, _ in model.named_buffers() if name.rsplit(".", 1)[-1] == "attn_mask"]
    assert masks, "expected at least one shifted block to register an attn_mask"

    added = ignore_static_attn_mask_buffers(model)
    assert sorted(added) == sorted(masks)
    assert sorted(model._ddp_params_and_buffers_to_ignore) == sorted(masks)

    # idempotent, and it must not discard entries an earlier caller added
    assert ignore_static_attn_mask_buffers(model) == []
    assert sorted(model._ddp_params_and_buffers_to_ignore) == sorted(masks)

    # every other buffer keeps synchronizing
    others = [name for name, _ in model.named_buffers() if name.rsplit(".", 1)[-1] != "attn_mask"]
    assert not set(others) & set(model._ddp_params_and_buffers_to_ignore)


if __name__ == "__main__":
    test_unet()
    # test_crossformer()
