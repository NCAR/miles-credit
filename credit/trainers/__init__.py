import copy
import importlib
import logging

logger = logging.getLogger(__name__)

# Registry: trainer_type -> entry. Entries are either:
#   (module_path: str, class_name: str, message: str)  — internal lazy entries
#   (cls: type,        message: str)                    — externally registered classes
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
    "gen2": (
        "credit.trainers.trainer_gen2",
        "TrainerERA5Gen2",
        "Gen2 trainer for the new nested data schema with preblock-assembled batches. forecast_len=1 means 1 step.",
    ),
    "era5-gen2": (  # backward-compat alias for gen2
        "credit.trainers.trainer_gen2",
        "TrainerERA5Gen2",
        "Gen2 trainer for the new nested data schema with preblock-assembled batches. forecast_len=1 means 1 step.",
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


def register_trainer(trainer_type, message=None):
    """Decorator that adds an external trainer class to the trainer registry.

    The class must inherit from :class:`credit.trainers.base_trainer.BaseTrainer`.
    Mirrors ``credit.models.register_model`` — see that docstring for the full
    rationale; this is the same pattern applied to trainers.

    Args:
        trainer_type: Key used in the config ``trainer.type`` field.
        message: Optional log message shown when the trainer is loaded.

    Example::

        from credit.trainers import register_trainer
        from credit.trainers.base_trainer import BaseTrainer

        @register_trainer("my_trainer", "Loading my custom trainer ...")
        class MyTrainer(BaseTrainer):
            ...
    """

    def decorator(cls):
        from credit.trainers.base_trainer import BaseTrainer  # imported here to avoid loading it at module import time

        if not (isinstance(cls, type) and issubclass(cls, BaseTrainer)):
            raise TypeError(
                f"register_trainer: '{cls.__name__}' must inherit from credit.trainers.base_trainer.BaseTrainer."
            )
        if trainer_type in _TRAINER_REGISTRY:  # warn instead of silently overwriting
            logger.warning(f"register_trainer: overwriting existing registry entry for '{trainer_type}'")
        _TRAINER_REGISTRY[trainer_type] = (cls, message or f"Loading {trainer_type} trainer ...")
        return cls  # must return the class, otherwise it becomes None after decoration

    return decorator


def _load_trainer_entry(trainer_type):
    """Lazily import and return (trainer_class, message) for a registered trainer type.

    Raises:
        ValueError: If trainer_type is not in _TRAINER_REGISTRY.
        ImportError: If the trainer's module cannot be imported.
    """
    if trainer_type not in _TRAINER_REGISTRY:
        msg = (
            f"Trainer type '{trainer_type}' not supported. Available types: {sorted(_TRAINER_REGISTRY)}. "
            "Register a custom trainer with @register_trainer or via custom_objects in your config."
        )
        logger.warning(msg)
        raise ValueError(msg)
    entry = _TRAINER_REGISTRY[trainer_type]
    # Unlike a plain lazy entry, an externally registered trainer stores (cls, message)
    # rather than (module_path, class_name, message) -- same disambiguation credit.models
    # uses: check whether the first element is a string (module path) or not (class).
    if not isinstance(entry[0], str):
        cls, message = entry
        return cls, message
    module_path, class_name, message = entry
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name), message
    except (ImportError, Exception) as e:
        msg = f"Could not import trainer '{class_name}' from '{module_path}': {e}"
        logger.warning(msg)
        raise ImportError(msg) from e


def load_trainer(conf):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(_TRAINER_REGISTRY.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    from credit.registry import load_custom_objects  # imported here to avoid a circular import at module load time

    load_custom_objects(conf)  # register any custom classes listed under custom_objects in the config

    cls, message = _load_trainer_entry(trainer_type)
    logger.info(message)
    return cls
