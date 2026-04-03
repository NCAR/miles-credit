import os
import sys
import copy
import inspect
import logging

# Legacy model imports — wrapped in try/except because their transitive dependencies
# (bridgescaler → numba) may conflict with newer NumPy (≥2.3) environments.
# Zoo models (below) have no such dependency and always load cleanly.
try:
    from credit.models.crossformer.crossformer import CrossFormer
    from credit.models.camulator import Camulator
    from credit.models.unet import SegmentationModel
    from credit.models.fuxi import Fuxi
    from credit.models.swin import SwinTransformerV2Cr
    from credit.models.graph import GraphResTransfGRU
    from credit.models.debugger_model import DebuggerModel
    from credit.models.wxformer.crossformer import CrossFormer as WXFormer
    from credit.models.wxformer.crossformer_ensemble import CrossFormerWithNoise
    from credit.models.wxformer.crossformer_downscaling import DownscalingCrossFormer
    from credit.models.unet_downscaling import DownscalingSegmentationModel
    from credit.models.wxformer.crossformer_diffusion import CrossFormerDiffusion
    from credit.models.unet_diffusion import UnetDiffusion
    from credit.diffusion import ModifiedGaussianDiffusion
    from credit.models.swin_wrf import WRFTransformer
    from credit.models.dscale_wrf import DscaleTransformer

    _LEGACY_MODELS_AVAILABLE = True
except ImportError as _legacy_import_err:
    logging.warning(
        f"Legacy model imports unavailable (numba/NumPy conflict or missing dep): {_legacy_import_err}. "
        "Crossformer/WXFormer/FuXi/UNet family will not be loadable in this environment."
    )
    _LEGACY_MODELS_AVAILABLE = False

from credit.models.aurora.model import CREDITAurora
from credit.models.pangu.pangu import CREDITPangu
from credit.models.aifs.aifs import CREDITAifs
from credit.models.stormer.stormer import CREDITStormer
from credit.models.climax.climax import CREDITClimaX
from credit.models.fourcastnet.afno import CREDITFourCastNet
from credit.models.sfno.sfno import CREDITSfno
from credit.models.swinrnn.swinrnn import CREDITSwinRNN
from credit.models.fengwu.fengwu import CREDITFengWu
from credit.models.graphcast.graphcast import CREDITGraphCast
from credit.models.healpix.healpix import CREDITHEALPix
from credit.models.fourcastnet3.fcn3 import CREDITFourCastNetV3
from credit.models.itransformer.itransformer import CREDITiTransformer
from credit.models.fuxi_ens.fuxi_ens import CREDITFuXiENS
from credit.models.arches.arches import CREDITArchesWeather
from credit.models.mambavision.mambavision import CREDITMambaVision
from credit.models.corrdiff.corrdiff import CREDITCorrDiff


logger = logging.getLogger(__name__)

# Define model types and their corresponding classes
model_types = {
    # ── Model zoo ──────────────────────────────────────────────────────────
    "aurora": (CREDITAurora, "Loading Aurora (Perceiver3D + Swin3D backbone) ..."),
    "pangu": (CREDITPangu, "Loading Pangu-Weather (3D Earth Transformer) ..."),
    "aifs": (CREDITAifs, "Loading AIFS lat/lon Transformer processor ..."),
    "stormer": (CREDITStormer, "Loading Stormer (plain ViT weather model) ..."),
    "climax": (CREDITClimaX, "Loading ClimaX (per-variable tokenization ViT) ..."),
    "fourcastnet": (CREDITFourCastNet, "Loading FourCastNet v1 (AFNO ViT) ..."),
    "sfno": (CREDITSfno, "Loading SFNO (Spherical Fourier Neural Operator) ..."),
    "swinrnn": (CREDITSwinRNN, "Loading SwinRNN (Swin encoder-decoder) ..."),
    "fengwu": (CREDITFengWu, "Loading FengWu (multi-group cross-attention ViT) ..."),
    "graphcast": (CREDITGraphCast, "Loading GraphCast (kNN GNN encoder-processor-decoder) ..."),
    "healpix": (CREDITHEALPix, "Loading DLWP-HEALPix (HEALPix U-Net with lat/lon reprojection) ..."),
    "fourcastnet3": (CREDITFourCastNetV3, "Loading FourCastNet3 (spherical neural operator U-Net) ..."),
    "itransformer": (CREDITiTransformer, "Loading iTransformer (inverted attention across variables) ..."),
    "fuxi_ens": (CREDITFuXiENS, "Loading FuXi-ENS (ViT + VAE ensemble perturbation head) ..."),
    "arches": (CREDITArchesWeather, "Loading ArchesWeather (window + column attention) ..."),
    "mambavision": (CREDITMambaVision, "Loading MambaVision (hybrid Mamba + attention U-Net) ..."),
    "corrdiff": (CREDITCorrDiff, "Loading CorrDiff (score-based conditional diffusion) ..."),
}

# ── Legacy models (crossformer family, fuxi, unet, swin, graph) ────────────
# Only registered when their imports succeeded (numba/NumPy compat required).
if _LEGACY_MODELS_AVAILABLE:
    model_types.update(
        {
            "crossformer": (
                CrossFormer,
                "Loading the CrossFormer model with a conv decoder head and skip connections ...",
            ),
            "camulator": (
                Camulator,
                "Loading the CAMulator model with a conv decoder head and skip connections ...",
            ),
            "crossformer-diffusion": (
                CrossFormerDiffusion,
                "Loading A DDPM model with CrossFormer Backbone ...",
            ),
            "unet-diffusion": (
                UnetDiffusion,
                "Loading A DDPM model with UNET Backbone ...",
            ),
            "wxformer": (WXFormer, "Loading the WXFormer deterministic model ..."),
            "crossformer-ensemble": (
                CrossFormerWithNoise,
                "Loading the WXFormer v1 SDL ensemble model (noise injection at each transformer scale) ...",
            ),
            "wxformer-sdl": (
                CrossFormerWithNoise,
                "Loading the WXFormer SDL ensemble model (noise injection at each transformer scale) ...",
            ),
            "crossformer-style": (
                CrossFormerWithNoise,
                "Loading the WXFormer SDL ensemble model (legacy alias for wxformer-sdl) ...",
            ),
            "unet": (SegmentationModel, "Loading a unet model"),
            "fuxi": (Fuxi, "Loading Fuxi model"),
            "swin": (SwinTransformerV2Cr, "Loading the minimal Swin model"),
            "graph": (GraphResTransfGRU, "Loading Graph Residual Transformer GRU model"),
            "debugger": (DebuggerModel, "Loading the debugger model"),
            "wrf": (WRFTransformer, "Loading WRF Transformer"),
            "dscale": (DscaleTransformer, "Loading downscaling Transformer"),
            "crossformer_downscaling": (
                DownscalingCrossFormer,
                "Loading downscaling crossformer model",
            ),
            "unet_downscaling": (DownscalingSegmentationModel, "Loading downscaling U-net"),
        }
    )


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
                raise ValueError("No saved checkpoint exists. You must train a model first. Exiting.")

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

    elif model_type == "unet-diffusion":
        model, message = model_types[model_type]
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
    elif model_type in model_types:
        model, message = model_types[model_type]
        logger.info(message)
        if load_weights:
            if model_name:
                return model.load_model_name(conf, model_name=model_name)
            else:
                return model.load_model(conf)
        # Pop pretrained_weights before filtering so it isn't passed to __init__
        pretrained_weights = model_conf.pop("pretrained_weights", None)
        # Filter kwargs to only those accepted by the constructor (handles models
        # that don't accept parser-only keys like 'levels').
        sig = inspect.signature(model.__init__)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            filtered_conf = model_conf  # accepts **kwargs — pass everything
        else:
            filtered_conf = {k: v for k, v in model_conf.items() if k in params}
        model_instance = model(**filtered_conf)
        if pretrained_weights:
            import torch

            ckpt_path = os.path.expandvars(pretrained_weights)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model_instance.load_state_dict(state, strict=False)
            logger.info(
                f"Loaded pretrained weights from {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys"
            )
        return model_instance
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
                raise ValueError("No saved checkpoint exists. You must train a model first. Exiting.")

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
