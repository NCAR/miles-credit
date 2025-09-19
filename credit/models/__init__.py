import os
import sys
import copy
import logging

# Import model classes
from credit.models.crossformer import CrossFormer
from credit.models.unet import SegmentationModel
from credit.models.unet404 import SegmentationModel404
from credit.models.fuxi import Fuxi
from credit.models.swin import SwinTransformerV2Cr
from credit.models.graph import GraphResTransfGRU
from credit.models.debugger_model import DebuggerModel
from credit.models.crossformer_ensemble import CrossFormerWithNoise
from credit.models.crossformer_diffusion import CrossFormerDiffusion
from credit.models.unet_diffusion import UnetDiffusion
from credit.diffusion import ModifiedGaussianDiffusion

from credit.models.swin_wrf import WRF_Tansformer
from credit.models.dscale_wrf import Dscale_Tansformer

logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    "crossformer": (
        CrossFormer,
        "Loading the CrossFormer model with a conv decoder head and skip connections ...",
    ),
    "crossformer-diffusion": (
        CrossFormerDiffusion,
        "Loading A DDPM model with CrossFormer Backbone ....",
    ),
    "unet-diffusion": (
        UnetDiffusion,
        "Loading A DDPM model with UNET Backbone ....",
    ),
    "crossformer-style": (
        CrossFormerWithNoise,
        "Loading the ensemble CrossFormer model with a Style-GAN-like noise injection scheme ...",
    ),
    "unet": (SegmentationModel, "Loading a unet model"),
    "unet404": (SegmentationModel404, "Loading unet404 model"),
    "fuxi": (Fuxi, "Loading Fuxi model"),
    "swin": (SwinTransformerV2Cr, "Loading the minimal Swin model"),
    "graph": (GraphResTransfGRU, "Loading Graph Residual Transformer GRU model"),
    "debugger": (DebuggerModel, "Loading the debugger model"),
    'wrf': (WRF_Tansformer, 'Loading WRF Transformer'),
    'dscale': (Dscale_Tansformer, 'Loading downscaling Transformer'),
}


# Define FSDP sharding and/or checkpointing policy
def load_fsdp_or_checkpoint_policy(conf):
    # crossformer
    if "crossformer" in conf["model"]["type"]:
        from credit.models.crossformer import (
            Attention,
            DynamicPositionBias,
            FeedForward,
            CrossEmbedLayer,
        )

        transformer_layers_cls = {
            Attention,
            DynamicPositionBias,
            FeedForward,
            CrossEmbedLayer,
        }
    elif "unet" in conf["model"]["type"]:
        from credit.models.crossformer import (
            Attention,
            DynamicPositionBias,
            FeedForward,
            CrossEmbedLayer,
        )

        transformer_layers_cls = {
            Attention,
            DynamicPositionBias,
            FeedForward,
            CrossEmbedLayer,
        }
    # FuXi
    # FuXi supports "spectral_norm = True" only
    elif "fuxi" in conf["model"]["type"] or ("wrf" in conf["model"]["type"]) or ("dscale" in conf["model"]["type"]):
        from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

        transformer_layers_cls = {SwinTransformerV2Stage}

    # Swin by itself
    elif "swin" in conf["model"]["type"]:
        from credit.models.swin import (
            SwinTransformerV2CrBlock,
            WindowMultiHeadAttentionNoPos,
            WindowMultiHeadAttention,
        )

        transformer_layers_cls = {
            SwinTransformerV2CrBlock,
            WindowMultiHeadAttentionNoPos,
            WindowMultiHeadAttention,
        }

    # other models not supported
    else:
        raise OSError(
            "You asked for FSDP but only crossformer, swin, and fuxi are currently supported.",
            "See credit/models/__init__.py for examples on adding new models",
        )

    return transformer_layers_cls


def load_model(conf, load_weights=False, model_name=False):
    conf = copy.deepcopy(conf)

    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type in ("unet", "unet404"):
        import torch

        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            model = model(**model_conf)
            save_loc = os.path.expandvars(conf["save_loc"])

            if model_name:
                ckpt = os.path.join(save_loc, model_name)
            else:
                if os.path.isfile(os.path.join(save_loc, "model_checkpoint.pt")):
                    ckpt = os.path.join(save_loc, "model_checkpoint.pt")
                else:
                    ckpt = os.path.join(save_loc, "checkpoint.pt")

            if not os.path.isfile(ckpt):
                raise ValueError(
                    "No saved checkpoint exists. You must train a model first. Exiting."
                )

            logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

            checkpoint = torch.load(ckpt)
            if "model_state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            return model

        return model(**model_conf)

    elif model_type == "crossformer-diffusion":
        model, message = model_types[model_type]
        logger.info(message)
        diffusion_config = conf.get("model", {}).get("diffusion")
        if diffusion_config is not None:
            diffusion_config = diffusion_config.copy()
            self_condition = diffusion_config.pop("self_condition", False)
            condition = diffusion_config.pop("condition", True)
        else:
            logger.warning(
                "The diffusion details were not specified as model:diffusion, exiting"
            )
            sys.exit(0)

        if load_weights:
            if model_name:
                return model.load_model_name(conf, model_name=model_name)
            else:
                return model.load_model(conf)

        return ModifiedGaussianDiffusion(
            model(**model_conf, self_condition=self_condition, condition=condition),
            **diffusion_config,
        )

    elif model_type == "unet-diffusion":
        model, message = model_types[model_type]
        logger.info(message)
        diffusion_config = conf.get("model", {}).get("diffusion")
        if diffusion_config is not None:
            diffusion_config = diffusion_config.copy()
            self_condition = diffusion_config.pop("self_condition", False)
            condition = diffusion_config.pop("condition", True)
        else:
            logger.warning(
                "The diffusion details were not specified as model:diffusion, exiting"
            )
            sys.exit(0)

        if load_weights:
            if model_name:
                return model.load_model_name(conf, model_name=model_name)
            else:
                return model.load_model(conf)

        return ModifiedGaussianDiffusion(
            model(**model_conf, self_condition=self_condition, condition=condition),
            **diffusion_config,
        )
    elif model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            if model_name:
                return model.load_model_name(conf, model_name=model_name)
            else:
                return model.load_model(conf)
        return model(**model_conf)
    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)


def load_model_name(conf, model_name, load_weights=False):
    conf = copy.deepcopy(conf)
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type in ("unet", "unet404"):
        import torch

        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            model = model(**model_conf)
            save_loc = conf["save_loc"]
            ckpt = os.path.join(save_loc, model_name)

            if not os.path.isfile(ckpt):
                raise ValueError(
                    "No saved checkpoint exists. You must train a model first. Exiting."
                )

            logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model

        return model(**model_conf)

    if model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            return model.load_model_name(conf, model_name)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
