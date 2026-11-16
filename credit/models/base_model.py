import os
import copy
import torch
from torch import nn
import logging

from credit.models.checkpoint import load_state_dict_error_handler

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    # Models that build their data-dependent layers from the first batch (e.g.
    # wxformer_simple) override this to report True until they are built. The gen2
    # trainer then materializes them before compiling/wrapping the model.
    needs_materialize = False

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
    def _build_for_loading(cls, conf):
        """Instantiate the model that checkpoint weights are loaded into.

        Subclasses whose layers depend on the data override this to rebuild them
        from saved hyperparameters instead of from ``conf["model"]`` alone.
        """
        model_conf = {k: v for k, v in conf["model"].items() if k != "type"}
        return cls(**model_conf)

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

        logger.info(f"Loading a model with pre-trained weights from path {ckpt}")

        checkpoint = torch.load(
            ckpt,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )

        model_class = cls._build_for_loading(conf)
        if "model_state_dict" in checkpoint.keys():
            load_msg = model_class.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            load_msg = model_class.load_state_dict(checkpoint, strict=False)
        load_state_dict_error_handler(load_msg)

        return model_class

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

        logger.info(f"Loading a model with pre-trained weights from path {ckpt}")

        checkpoint = torch.load(
            ckpt,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )

        model_class = cls._build_for_loading(conf)

        load_msg = model_class.load_state_dict(checkpoint if fsdp else checkpoint["model_state_dict"], strict=False)
        load_state_dict_error_handler(load_msg)

        return model_class

    def save_model(self, conf):
        save_loc = os.path.expandvars(conf["save_loc"])
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, os.path.join(f"{save_loc}", "checkpoint.pt"))
