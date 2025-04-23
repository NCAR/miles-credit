import torch
from torch import nn

import numpy as np

from credit.transforms import load_transforms
from credit.skebs import SKEBS

import logging
from math import pi
PI = pi
logger = logging.getLogger(__name__)


class ICPerturb(nn.Module):
    def __init__(self, ic_conf):
        """
        ic_conf: dictionary with config options for ICPerturb.


        """
        super().__init__()

        module_list = [
            "white_noise",
            "skebs",
        ]
        for module in module_list: # set defaults
            ic_conf[module] = ic_conf.get(module,  {"activate": False})


        self.operations_dict = nn.ModuleDict()

        if ic_conf["white_noise"]["activate"]:
            self.operations_dict["white_noise"] = WhiteNoisePerturb(ic_conf)

        if ic_conf["skebs"]["activate"]:
            self.operations_dict["skebs"] = SKEBSPrescribed(ic_conf)


    def forward(self, x_dict):

        if x_dict["forecast_step"] == 1:
            logger.debug("perturbing ICs")
            for op in self.operations_dict.values():
                x_dict = op(x_dict)

        return x_dict["x"] # return a tensor


class WhiteNoisePerturb(nn.Module):
    def __init__(self, ic_conf):
        """
        ic_conf: dictionary with config options for ICPerturb.

        """
        super().__init__()


    def forward(self, x_dict):
        raise RuntimeError("not implemented")
        return x_dict
    
class SKEBSPrescribed(nn.Module):
    def __init__(self, ic_conf):
        """
        ic_conf: dictionary with config options for ICPerturb.

        """
        super().__init__()

        ic_conf["data"]["retain_graph"] = True # doesn't matter, but doing this for safety
        
        ic_conf["skebs"]["use_statics"] = False
        # x always has statics
        
        self.skebs = SKEBS(ic_conf, ic_mode=True)

    
    
    def forward(self, x_dict):
        
        x_dict["y_pred"] = x_dict["x"] # add this field for skebs

        x_dict["x"] = self.skebs(x_dict)["y_pred"]
        
        return x_dict