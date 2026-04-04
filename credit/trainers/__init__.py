import copy
import logging

# Import trainer classes
from credit.trainers.trainerERA5 import Trainer as TrainerERA5
from credit.trainers.trainerERA5_Diffusion import Trainer as TrainerERA5_Diffusion
from credit.trainers.trainerERA5_ensemble import Trainer as TrainerEnsemble
from credit.trainers.trainer_downscaling import Trainer as Trainer404
from credit.trainers.trainer_om4_samudra import Trainer as TrainerSamudra

from credit.trainers.trainerWRF import Trainer as TrainerWRF
from credit.trainers.trainerWRF_multi import Trainer as TrainerWRFMulti

# ic_optimization → credit.output → credit.interp → numba (needs NumPy ≤ 2.2)
# trainerLES → optuna (optional dep)
# Guard these so environments with NumPy ≥ 2.3 or without optuna still boot.
try:
    from credit.trainers.ic_optimization import Trainer as TrainerIC

    _IC_OPT_AVAILABLE = True
except (ImportError, Exception) as _ic_err:
    logging.warning(f"ic_optimization trainer unavailable: {_ic_err}.")
    TrainerIC = None
    _IC_OPT_AVAILABLE = False

try:
    from credit.trainers.trainerLES import Trainer as TrainerLES

    _LES_TRAINER_AVAILABLE = True
except (ImportError, Exception) as _les_err:
    logging.warning(f"trainerLES unavailable: {_les_err}.")
    TrainerLES = None
    _LES_TRAINER_AVAILABLE = False

logger = logging.getLogger(__name__)


# Define trainer types and their corresponding classes
trainer_types = {
    "era5": (
        TrainerERA5,
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-diffusion": (
        TrainerERA5_Diffusion,
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-ensemble": (
        TrainerEnsemble,
        "Loading a single or multi-step trainer for the ERA5 dataset for parallel computation of the CRPS loss.",
    ),
    "cam": (
        TrainerERA5,
        "Loading a single or multi-step trainer for the CAM dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "conus404": (Trainer404, "Loading a standard trainer for the CONUS404 dataset."),
    "standard-wrf": (TrainerWRF, "Loading a single-step WRF trainer"),
    "multi-step-wrf": (TrainerWRFMulti, "Loading a multi-step WRF trainer"),
    "samudra": (
        TrainerSamudra,
        "Loading a single or multi-step trainer for the Samudra OM4 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
}

if _IC_OPT_AVAILABLE:
    trainer_types["ic-opt"] = (TrainerIC, "Loading an initial condition optimizer training class")

if _LES_TRAINER_AVAILABLE:
    trainer_types["standard-les"] = (TrainerLES, "Loading a single-step LES trainer")


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
