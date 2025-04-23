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

        self.operations_dict = nn.ModuleDict()
        
        if ic_conf["white_noise"].get("activate", False):
            self.operations_dict["white_noise"] = WhiteNoisePerturb(ic_conf)

        if ic_conf["skebs"].get("activate", False):
            self.operations_dict["skebs"] = SKEBSPrescribed(ic_conf)


    def forward(self, x):
        for op in self.operations_dict.items():
            x = op(x)

        return x


class WhiteNoisePerturb(nn.Module):
    def __init__(self, ic_conf):
        """
        ic_conf: dictionary with config options for ICPerturb.

        """
        super().__init__()


    def forward(self, x):
        return x
    
class SKEBSPrescribed(nn.Module):
    def __init__(self, ic_conf):
        """
        ic_conf: dictionary with config options for ICPerturb.

        """
        super().__init__()
        

    
    
    def forward(self, x):
        
        input_dict = {"x": None,
                      "y_pred": x}

        x = self.skebs(input_dict)["y_pred"]
        
        return x