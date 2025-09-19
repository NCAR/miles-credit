import os
import sys
import glob
import logging
import numpy as np

import logging
from typing import Dict

import torch
from torchvision import transforms as tforms

from credit.transforms.transforms_les import Normalize_LES, ToTensor_LES
from credit.transforms.transforms_wrf import Normalize_WRF, ToTensor_WRF
from credit.transforms.transforms_global import Normalize_ERA5_and_Forcing, ToTensor_ERA5_and_Forcing
from credit.transforms.transforms_quantile import NormalizeState_Quantile_Bridgescalar, ToTensor_BridgeScaler

logger = logging.getLogger(__name__)

def device_compatible_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely move tensor to device, with float32 casting on MPS (Metal Performance Shaders). Addresses runtime error in OSX about MPS not supporting float64. 

    Args:
        tensor (torch.Tensor): Input tensor to move.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Tensor moved to device (cast to float32 if device is MPS).
    """

    if device.type == "mps":
        return tensor.to(dtype=torch.float32, device=device)
    else:
        return tensor.to(device) 


def load_transforms(conf, scaler_only=False):
    """Load transforms.

    Args:
        conf (str): path to config
        scaler_only (bool): True --> retrun scaler; False --> return scaler and ToTensor

    Returns:
        tf.tensor: transform

    """
    # ------------------------------------------------------------------- #
    # transform class
    
    if conf["data"]["scaler_type"] == "std_new":
        transform_scaler = Normalize_ERA5_and_Forcing(conf)
        
    elif conf["data"]["scaler_type"] == "std_cached":
        transform_scaler = None

    elif conf["data"]["scaler_type"] == "quantile-cached":
        transform_scaler = NormalizeState_Quantile_Bridgescalar(conf)
    
    elif conf["data"]["scaler_type"] == "std-les":
        transform_scaler = Normalize_LES(conf)

    elif conf["data"]["scaler_type"] == "std-wrf":
        transform_scaler = Normalize_WRF(conf)
        
    else:
        logger.log("scaler type not supported check data: scaler_type in config file")
        raise

    if scaler_only:
        return transform_scaler
        
    # ------------------------------------------------------------------- #
    # ToTensor class
    
    if conf["data"]["scaler_type"] == "std_new" or conf["data"]["scaler_type"] == "std_cached":
        to_tensor_scaler = ToTensor_ERA5_and_Forcing(conf)
        
    elif conf["data"]["scaler_type"] == "quantile-cached":
        # beidge scaler ToTensor
        to_tensor_scaler = ToTensor_BridgeScaler(conf)
        
    elif conf["data"]["scaler_type"] == "std-les":
        to_tensor_scaler = ToTensor_LES(conf)
        
    elif conf["data"]["scaler_type"] == "std-wrf":
        to_tensor_scaler = ToTensor_WRF(conf)
        
    else:
        # the old ToTensor
        to_tensor_scaler = ToTensor(conf=conf)

    # ------------------------------------------------------------------- #
    # combine transform and ToTensor
    
    if transform_scaler is not None:
        transforms = [transform_scaler, to_tensor_scaler]
        
    else:
        transforms = [to_tensor_scaler]

    return tforms.Compose(transforms)



