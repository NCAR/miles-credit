import copy
import logging

# Import trainer classes
from credit.trainers.trainerERA5 import TrainerERA5
from credit.trainers.trainerERA5gen2 import TrainerERA5Gen2
from credit.trainers.trainerERA5_Diffusion import TrainerERA5Diffusion
from credit.trainers.trainerERA5_ensemble import TrainerERA5Ensemble
from credit.trainers.trainer_downscaling import TrainerDownscaling

try:
    from credit.trainers.ic_optimization import TrainerIC

    _IC_OPT_AVAILABLE = True
except (ImportError, Exception):
    TrainerIC = None
    _IC_OPT_AVAILABLE = False
from credit.trainers.trainer_om4_samudra import TrainerSamudra

from credit.trainers.trainerLES import TrainerLES
from credit.trainers.trainerWRF import TrainerWRF
from credit.trainers.trainerWRF_multi import TrainerWRFMulti

logger = logging.getLogger(__name__)


# Define trainer types and their corresponding classes
trainer_types = {
    "era5": (
        TrainerERA5,
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-gen2": (
        TrainerERA5Gen2,
        "ERA5 Gen 2 trainer for the new nested data schema with preblock-assembled batches. forecast_len=1 means 1 step.",
    ),
    "era5-diffusion": (
        TrainerERA5Diffusion,
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-ensemble": (
        TrainerERA5Ensemble,
        "Loading a single or multi-step trainer for the ERA5 dataset for parallel computation of the CRPS loss.",
    ),
    "cam": (
        TrainerERA5,
        "Loading a single or multi-step trainer for the CAM dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    **({"ic-opt": (TrainerIC, "Loading an initial condition optimizer training class")} if _IC_OPT_AVAILABLE else {}),
    "conus404": (TrainerDownscaling, "Loading a standard trainer for the CONUS404 dataset."),
    "standard-les": (TrainerLES, "Loading a single-step LES trainer"),
    "standard-wrf": (TrainerWRF, "Loading a single-step WRF trainer"),
    "multi-step-wrf": (TrainerWRFMulti, "Loading a multi-step WRF trainer"),
    "samudra": (
        TrainerSamudra,
        "Loading a single or multi-step trainer for the Samudra OM4 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
}


def load_trainer(conf):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(trainer_types.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    if trainer_type in trainer_types:
        trainer, message = trainer_types[trainer_type]
        logger.info(message)
        return trainer

    else:
        msg = f"Trainer type {trainer_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
