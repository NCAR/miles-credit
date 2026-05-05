import os
import sys
import copy
import logging
import importlib

logger = logging.getLogger(__name__)

# Registry entries are either:
#   (module_path: str, class_name: str, log_message: str)  — internal lazy entries
#   (cls: type,        log_message: str)                   — externally registered classes
_MODEL_REGISTRY = {
    "crossformer": (
        "credit.models.crossformer",
        "CrossFormer",
        "Loading the CrossFormer model with a conv decoder head and skip connections ...",
    ),
    "camulator": (
        "credit.models.camulator",
        "Camulator",
        "Loading the CAMulator model with a conv decoder head and skip connections ...",
    ),
    "crossformer-diffusion": (
        "credit.models.wxformer.crossformer_diffusion",
        "CrossFormerDiffusion",
        "Loading A DDPM model with CrossFormer Backbone ...",
    ),
    "unet-diffusion": (
        "credit.models.unet_diffusion",
        "UnetDiffusion",
        "Loading A DDPM model with UNET Backbone ...",
    ),
    "wxformer": (
        "credit.models.wxformer.crossformer",
        "CrossFormer",
        "Loading the WXFormer deterministic model ...",
    ),
    "crossformer-ensemble": (
        "credit.models.wxformer.crossformer_ensemble",
        "CrossFormerWithNoise",
        "Loading the ensemble CrossFormer model with a noise injection scheme ...",
    ),
    "crossformer-style": (
        "credit.models.wxformer.crossformer_ensemble",
        "CrossFormerWithNoise",
        "Loading the ensemble CrossFormer model with a Style-GAN-like noise injection scheme ...",
    ),
    "unet": ("credit.models.unet", "SegmentationModel", "Loading a unet model"),
    "fuxi": ("credit.models.fuxi", "Fuxi", "Loading Fuxi model"),
    "swin": ("credit.models.swin", "SwinTransformerV2Cr", "Loading the minimal Swin model"),
    "graph": (
        "credit.models.graph",
        "GraphResTransfGRU",
        "Loading Graph Residual Transformer GRU model",
    ),
    "debugger": ("credit.models.debugger_model", "DebuggerModel", "Loading the debugger model"),
    "wrf": ("credit.models.swin_wrf", "WRFTransformer", "Loading WRF Transformer"),
    "dscale": ("credit.models.dscale_wrf", "DscaleTransformer", "Loading downscaling Transformer"),
    "crossformer_downscaling": (
        "credit.models.wxformer.crossformer_downscaling",
        "DownscalingCrossFormer",
        "Loading downscaling crossformer model",
    ),
    "unet_downscaling": (
        "credit.models.unet_downscaling",
        "DownscalingSegmentationModel",
        "Loading downscaling U-net",
    ),
}

# Backward-compatible name -> (module_path, class_name) for direct attribute access
_CLASS_SOURCES = {
    "CrossFormer": ("credit.models.crossformer", "CrossFormer"),
    "Camulator": ("credit.models.camulator", "Camulator"),
    "SegmentationModel": ("credit.models.unet", "SegmentationModel"),
    "Fuxi": ("credit.models.fuxi", "Fuxi"),
    "SwinTransformerV2Cr": ("credit.models.swin", "SwinTransformerV2Cr"),
    "GraphResTransfGRU": ("credit.models.graph", "GraphResTransfGRU"),
    "DebuggerModel": ("credit.models.debugger_model", "DebuggerModel"),
    "WXFormer": ("credit.models.wxformer.crossformer", "CrossFormer"),
    "CrossFormerWithNoise": ("credit.models.wxformer.crossformer_ensemble", "CrossFormerWithNoise"),
    "DownscalingCrossFormer": ("credit.models.wxformer.crossformer_downscaling", "DownscalingCrossFormer"),
    "DownscalingSegmentationModel": ("credit.models.unet_downscaling", "DownscalingSegmentationModel"),
    "CrossFormerDiffusion": ("credit.models.wxformer.crossformer_diffusion", "CrossFormerDiffusion"),
    "UnetDiffusion": ("credit.models.unet_diffusion", "UnetDiffusion"),
    "ModifiedGaussianDiffusion": ("credit.diffusion", "ModifiedGaussianDiffusion"),
    "WRFTransformer": ("credit.models.swin_wrf", "WRFTransformer"),
    "DscaleTransformer": ("credit.models.dscale_wrf", "DscaleTransformer"),
}


def __getattr__(name):
    if name in _CLASS_SOURCES:
        module_path, class_name = _CLASS_SOURCES[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as exc:
            raise AttributeError(f"Cannot import {name!r}: optional dependencies missing.") from exc
    raise AttributeError(f"module 'credit.models' has no attribute {name!r}")


def register_model(model_type, message=None):
    """Decorator that adds an external PyTorch model class to the model registry.

    Args:
        model_type: Key used in the config ``model.type`` field.
        message: Optional log message shown when the model is loaded.

    Example::

        @register_model("my_model", "Loading my custom model ...")
        class MyModel(torch.nn.Module):
            ...
    """

    def decorator(cls):
        if model_type in _MODEL_REGISTRY:
            logger.warning(f"register_model: overwriting existing registry entry for '{model_type}'")
        _MODEL_REGISTRY[model_type] = (cls, message or f"Loading {model_type} model ...")
        return cls

    return decorator


def _load_model_entry(model_type):
    """Lazily import and return (model_class, log_message) for a registered model type."""
    if model_type not in _MODEL_REGISTRY:
        return None
    entry = _MODEL_REGISTRY[model_type]
    # External registration stores the class directly as the first element.
    if not isinstance(entry[0], str):
        cls, message = entry
        return cls, message
    module_path, class_name, message = entry
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls, message
    except ImportError as exc:
        raise ImportError(
            f"Model type '{model_type}' requires optional dependencies that are not installed. Original error: {exc}"
        ) from exc


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
    elif "wxformer" in conf["model"]["type"]:
        from credit.models.wxformer.crossformer import (
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
            "You asked for FSDP but only crossformer, wxformer, swin, and fuxi are currently supported.",
            "See credit/models/__init__.py for examples on adding new models",
        )

    return transformer_layers_cls


def load_custom_model_modules(conf):
    """Import every file listed under ``custom_models`` in the config.

    Each file is executed as a standalone module.  The expected use-case is
    that each file contains one or more classes decorated with
    ``@register_model``, so the import triggers registration as a side-effect.

    Args:
        conf (dict): Top-level config dict.  If ``custom_models`` is absent or
            empty this function is a no-op.

    Raises:
        FileNotFoundError: If a listed path does not exist on disk.
    """
    for raw_path in conf.get("custom_models", []):
        path = os.path.expandvars(raw_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"custom_models: file not found: {path!r}")
        spec = importlib.util.spec_from_file_location("_credit_custom_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def load_model(conf, load_weights=False, model_name=False):
    conf = copy.deepcopy(conf)
    load_custom_model_modules(conf)
    model_conf = conf["model"]

    if "type" not in model_conf:
        msg = "You need to specify a model type in the config file. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    model_type = model_conf.pop("type")

    if model_type in ("unet", "unet404"):
        import torch

        model, message = _load_model_entry(model_type)
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
                raise ValueError("No saved checkpoint exists. You must train a model first. Exiting.")

            logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

            checkpoint = torch.load(ckpt)
            if "model_state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            return model

        return model(**model_conf)

    elif model_type in ("crossformer-diffusion", "unet-diffusion"):
        from credit.diffusion import ModifiedGaussianDiffusion

        model, message = _load_model_entry(model_type)
        logger.info(message)
        diffusion_config = conf.get("model", {}).get("diffusion")
        if diffusion_config is not None:
            diffusion_config = diffusion_config.copy()
            self_condition = diffusion_config.pop("self_condition", False)
            condition = diffusion_config.pop("condition", True)
        else:
            logger.warning("The diffusion details were not specified as model:diffusion, exiting")
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

    elif model_type in _MODEL_REGISTRY:
        model, message = _load_model_entry(model_type)
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

        model, message = _load_model_entry(model_type)
        logger.info(message)
        if load_weights:
            model = model(**model_conf)
            save_loc = conf["save_loc"]
            ckpt = os.path.join(save_loc, model_name)

            if not os.path.isfile(ckpt):
                raise ValueError("No saved checkpoint exists. You must train a model first. Exiting.")

            logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model

        return model(**model_conf)

    if model_type in _MODEL_REGISTRY:
        model, message = _load_model_entry(model_type)
        logger.info(message)
        if load_weights:
            return model.load_model_name(conf, model_name)
        return model(**model_conf)

    else:
        msg = f"Model type {model_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
