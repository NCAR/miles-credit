import os
import copy
import logging
from importlib.metadata import version

# Import model classes
from credit.models.crossformer import CrossFormer
from credit.models.crossformer_may1 import CrossFormer as CrossFormerDep
from credit.models.simple_vit import SimpleViT
from credit.models.cube_vit import CubeViT
from credit.models.vit2d import ViT2D
from credit.models.vit3d import ViT3D
from credit.models.rvt import RViT
from credit.models.unet import SegmentationModel
from credit.models.unet404 import SegmentationModel404
from credit.models.fuxi import Fuxi
from credit.models.swin import SwinTransformerV2Cr

logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    "vit": (ViT2D, "Loading a Vision transformer architecture ..."),
    "vit3d": (ViT3D, "Loading a Vision transformer architecture ..."),
    "rvt": (RViT, "Loading a custom rotary transformer architecture with conv attention ..."),
    "simple-vit": (SimpleViT, "Loading a simplified vit rotary transformer architecture ..."),
    "cube-vit": (CubeViT, "Loading a simplified vit rotary transformer architecture with a 3D conv tokenizer ..."),
    "crossformer": (CrossFormer, "Loading the CrossFormer model with a conv decoder head and skip connections ..."),
    "crossformer-deprecated": (CrossFormerDep, "Loading the CrossFormer model with a conv decoder head and skip connections ..."),
    "unet": (SegmentationModel, "Loading a unet model"),
    "unet404": (SegmentationModel404, "Loading unet404 model"),
    "fuxi": (Fuxi, "Loading Fuxi model"),
    "swin": (SwinTransformerV2Cr, "Loading the minimal Swin model")
} 


def load_model(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    #if model_type == 'unet':
    if model_type in ('unet', 'unet404'):
        import torch
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            model = model(conf)
            save_loc = conf['save_loc']
            ckpt = os.path.join(save_loc, "checkpoint.pt")

            if not os.path.isfile(ckpt):
                raise ValueError(
                    "No saved checkpoint exists. You must train a model first. Exiting."
                )

            logging.info(
                f"Loading a model with pre-trained weights from path {ckpt}"
            )

            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
            
        return model(conf)

    if model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            return model.load_model(conf)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)


# dont need an old timm version anymore https://github.com/qubvel/segmentation_models.pytorch/releases/tag/v0.3.3
# def check_timm_version(model_type):
#     if model_type == "unet":
#         try:
#             assert (version('timm') == '0.6.12')
#         except AssertionError as e:
#             msg = """timm version 0.6 is required for using pytorch-segmentation-models. Please use environment-unet.yml env or pip install timm==0.6.12."""
#             raise Exception(msg) from e
#     elif model_type == "fuxi":
#         try:
#             assert (version('timm') >= '0.9.12')
#         except AssertionError as e:
#             msg = """timm version 0.9.12 or greater is required for FuXi model. Please use environment.yml env or pip install timm==0.9.12."""
#             raise Exception(msg) from e
