import os
import copy
import torch
from torch import nn
import logging

from credit.models.checkpoint import load_state_dict_error_handler

logger = logging.getLogger(__name__)


class WrapperModel(nn.Module):
    def __init__(self, conf):
        super().__init__()



    def forward(self, x):

        return x

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

        model_class = cls(**conf["model"])
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


        if conf['trainer']['mode'] == 'fsdp':
            fsdp = True 
        else: 
            fsdp = False

        ckpt = os.path.join(save_loc, model_name)
        
        if not os.path.isfile(ckpt):
            raise ValueError(
                f"No saved checkpoint {ckpt} exists. You must train a model first. Exiting."
            )

        logging.info(f"Loading a model with pre-trained weights from path {ckpt}")

        checkpoint = torch.load(
            ckpt,
            map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
        )

        if "type" in conf["model"]:
            del conf["model"]["type"]

        model_class = cls(**conf["model"])

        load_msg = model_class.load_state_dict(
            checkpoint if fsdp else checkpoint["model_state_dict"],
            strict=False
        )
        load_state_dict_error_handler(load_msg)

        return model_class

    def save_model(self, conf):
        save_loc = os.path.expandvars(conf["save_loc"])
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, os.path.join(f"{save_loc}", "checkpoint.pt"))