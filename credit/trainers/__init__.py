import copy
import importlib
import logging

logger = logging.getLogger(__name__)

# Registry: trainer_type -> (module_path, class_name, description)
_TRAINER_REGISTRY = {
    "era5-gen1": (
        "credit.trainers.trainerERA5gen1",
        "TrainerERA5Gen1",
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5": (  # backward-compat alias for era5-gen1
        "credit.trainers.trainerERA5gen1",
        "TrainerERA5Gen1",
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-gen2": (
        "credit.trainers.trainerERA5gen2",
        "TrainerERA5Gen2",
        "ERA5 Gen 2 trainer for the new nested data schema with preblock-assembled batches. forecast_len=1 means 1 step.",
    ),
    "era5-gen2-legacy": (
        "credit.trainers.trainerERA5gen2_legacy",
        "TrainerERA5Gen2",
        "ERA5 Gen 2 legacy trainer (update_x/build_channel_layout rollout, no postblocks). For comparison against era5-gen2.",
    ),
    "era5-diffusion": (
        "credit.trainers.trainerERA5_Diffusion",
        "TrainerERA5Diffusion",
        "Loading a single or multi-step trainer for the ERA5 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "era5-ensemble": (
        "credit.trainers.trainerERA5_ensemble",
        "TrainerERA5Ensemble",
        "Loading a single or multi-step trainer for the ERA5 dataset for parallel computation of the CRPS loss.",
    ),
    "cam": (
        "credit.trainers.trainerERA5gen1",
        "TrainerERA5Gen1",
        "Loading a single or multi-step trainer for the CAM dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
    "ic-opt": (
        "credit.trainers.ic_optimization",
        "TrainerIC",
        "Loading an initial condition optimizer training class",
    ),
    "conus404": (
        "credit.trainers.trainer_downscaling",
        "TrainerDownscaling",
        "Loading a standard trainer for the CONUS404 dataset.",
    ),
    "standard-les": (
        "credit.trainers.trainerLES",
        "TrainerLES",
        "Loading a single-step LES trainer",
    ),
    "standard-wrf": (
        "credit.trainers.trainerWRF",
        "TrainerWRF",
        "Loading a single-step WRF trainer",
    ),
    "multi-step-wrf": (
        "credit.trainers.trainerWRF_multi",
        "TrainerWRFMulti",
        "Loading a multi-step WRF trainer",
    ),
    "samudra": (
        "credit.trainers.trainer_om4_samudra",
        "TrainerSamudra",
        "Loading a single or multi-step trainer for the Samudra OM4 dataset that uses gradient accumulation on forecast lengths > 1.",
    ),
}


# Public alias for backward compatibility and test introspection
trainer_types = _TRAINER_REGISTRY


def load_trainer(conf):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(_TRAINER_REGISTRY.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    if trainer_type not in _TRAINER_REGISTRY:
        msg = f"Trainer type {trainer_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)

    module_path, class_name, message = _TRAINER_REGISTRY[trainer_type]
    logger.info(message)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, Exception) as e:
        msg = f"Could not import trainer '{class_name}' from '{module_path}': {e}"
        logger.warning(msg)
        raise ImportError(msg) from e
