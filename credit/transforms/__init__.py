import logging
from torchvision import transforms as tforms
from credit.transforms.deprecated._transforms import ToTensor
from credit.transforms.transforms_les import NormalizeLES, ToTensorLES
from credit.transforms.transforms_wrf import NormalizeWRF, ToTensorWRF
from credit.transforms.transforms_global import (
    Normalize_ERA5_and_Forcing,
    ToTensor_ERA5_and_Forcing,
)
from credit.transforms.transforms_quantile import (
    BridgescalerScaleState,
    NormalizeState_Quantile_Bridgescalar,
    ToTensor_BridgeScaler,
)

logger = logging.getLogger(__name__)


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

    elif conf["data"]["scaler_type"] == "bridgescaler":
        transform_scaler = BridgescalerScaleState(conf)

    elif conf["data"]["scaler_type"] == "std-les":
        transform_scaler = NormalizeLES(conf)

    elif conf["data"]["scaler_type"] == "std-wrf":
        transform_scaler = NormalizeWRF(conf)

    else:
        logger.log("scaler type not supported check data: scaler_type in config file")
        raise

    if scaler_only:
        return transform_scaler

    # ------------------------------------------------------------------- #
    # ToTensor class

    if (
        conf["data"]["scaler_type"] == "std_new"
        or conf["data"]["scaler_type"] == "std_cached"
    ):
        to_tensor_scaler = ToTensor_ERA5_and_Forcing(conf)

    elif conf["data"]["scaler_type"] == "quantile-cached":
        # beidge scaler ToTensor
        to_tensor_scaler = ToTensor_BridgeScaler(conf)

    elif conf["data"]["scaler_type"] == "std-les":
        to_tensor_scaler = ToTensorLES(conf)

    elif conf["data"]["scaler_type"] == "std-wrf":
        to_tensor_scaler = ToTensorWRF(conf)

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
