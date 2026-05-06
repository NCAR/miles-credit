import os
import copy
import torch
from torch import nn
import logging

from credit.models.checkpoint import load_state_dict_error_handler

logger = logging.getLogger(__name__)


class V1InputAdapter(nn.Module):
    """Wraps a V1-trained model to accept V2-ordered input channels.

    ERA5Dataset inserts channels as [dynfrc | static | prog]. V1-trained models
    expect [prog | static | dynfrc]. This wrapper permutes x before the model
    forward so x stays in V2 order everywhere else (trainer, rollout update).

    Enable with  model.v1_channel_order: true  in the YAML config. Only
    supports history_len=1 (T=1) inputs.
    """

    def __init__(self, model: nn.Module, n_prog: int, n_static: int, n_dyn: int):
        super().__init__()
        self.model = model
        perm = torch.cat(
            [
                torch.arange(n_dyn + n_static, n_dyn + n_static + n_prog),
                torch.arange(n_dyn, n_dyn + n_static),
                torch.arange(n_dyn),
            ]
        )
        self.register_buffer("_v1_perm", perm)

    def forward(self, x, **kwargs):
        return self.model(x[:, self._v1_perm, ...], **kwargs)


def _maybe_wrap_v1(model: nn.Module, conf: dict, v1_channel_order: bool) -> nn.Module:
    if not v1_channel_order:
        return model
    src_conf = next(iter(conf["data"]["source"].values()))
    v = src_conf["variables"]
    prog = v.get("prognostic") or {}
    dyn = v.get("dynamic_forcing") or {}
    sta = v.get("static") or {}
    n_levels = len(src_conf["levels"])
    n_prog = len(prog.get("vars_3D", [])) * n_levels + len(prog.get("vars_2D", []))
    n_dyn = len(dyn.get("vars_2D", []))
    n_static = len(sta.get("vars_2D", []))
    return V1InputAdapter(model, n_prog, n_static, n_dyn)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def concat_and_reshape(self, x1, x2):
        """
        x1: upper-air variables with level dimensions.
        x2: surface variables.
        """
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def reshape_only(self, x1):
        """
        As in "concat_and_reshape", but for upper-air variables only.
        """
        x1 = x1.view(
            x1.shape[0],
            x1.shape[1],
            x1.shape[2] * x1.shape[3],
            x1.shape[4],
            x1.shape[5],
        )
        return x1.permute(0, 2, 1, 3, 4)

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, : int(self.channels * self.levels), :, :, :]
        tensor2 = tensor[:, -int(self.surface_channels) :, :, :, :]
        tensor1 = tensor1.view(
            tensor1.shape[0],
            self.channels,
            self.levels,
            tensor1.shape[2],
            tensor1.shape[3],
            tensor1.shape[4],
        )
        return tensor1, tensor2

    @classmethod
    def load_model(cls, conf):
        conf = copy.deepcopy(conf)
        save_loc = os.path.expandvars(conf["save_loc"])

        if os.path.isfile(os.path.join(save_loc, "model_checkpoint.pt")):
            ckpt = os.path.join(save_loc, "model_checkpoint.pt")
        else:
            ckpt = os.path.join(save_loc, "checkpoint.pt")

        if not os.path.isfile(ckpt):
            raise ValueError("No saved checkpoint exists. You must train a model first. Exiting.")

        logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

        checkpoint = torch.load(
            ckpt,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )

        if "type" in conf["model"]:
            del conf["model"]["type"]
        v1 = conf["model"].pop("v1_channel_order", False)

        model_class = cls(**conf["model"])
        if "model_state_dict" in checkpoint.keys():
            load_msg = model_class.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            load_msg = model_class.load_state_dict(checkpoint, strict=False)
        load_state_dict_error_handler(load_msg)

        return _maybe_wrap_v1(model_class, conf, v1)

    @classmethod
    def load_model_name(cls, conf, model_name):
        conf = copy.deepcopy(conf)
        save_loc = os.path.expandvars(conf["save_loc"])

        if conf["trainer"]["mode"] == "fsdp":
            fsdp = True
        else:
            fsdp = False

        ckpt = os.path.join(save_loc, model_name)

        if not os.path.isfile(ckpt):
            raise ValueError(f"No saved checkpoint {ckpt} exists. You must train a model first. Exiting.")

        logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

        checkpoint = torch.load(
            ckpt,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )

        if "type" in conf["model"]:
            del conf["model"]["type"]
        v1 = conf["model"].pop("v1_channel_order", False)

        model_class = cls(**conf["model"])

        load_msg = model_class.load_state_dict(checkpoint if fsdp else checkpoint["model_state_dict"], strict=False)
        load_state_dict_error_handler(load_msg)

        return _maybe_wrap_v1(model_class, conf, v1)

    def save_model(self, conf):
        save_loc = os.path.expandvars(conf["save_loc"])
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, os.path.join(f"{save_loc}", "checkpoint.pt"))
