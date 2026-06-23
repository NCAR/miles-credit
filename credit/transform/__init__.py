import logging

from credit.transform.era5_transforms import ERA5StandardTransform, ERA5FieldTransform

logger = logging.getLogger(__name__)


transform_types = {
    "ERA5": (ERA5StandardTransform, "loading an ERA5 transformer class")
}


def load_transform(conf, source, device):
    transform, message = transform_types[source]
    logger.info(message)
    return transform(conf, device)


def load_era5_transforms(conf, device):
    """Return a dict of ERA5FieldTransform objects keyed by field type.

    Only creates a transform for field types that have a 'transform' block
    with mean_path/std_path in the config, e.g.:

        data.source.ERA5.dynamic_forcing.transform.mean_path: ...
    """
    transforms = {}
    source_conf = conf["data"]["source"].get("ERA5", {})
    for field_type, field_conf in source_conf.items():
        if not isinstance(field_conf, dict):
            continue
        transform_conf = field_conf.get("transform")
        if not transform_conf or transform_conf == "None":
            continue
        logger.info(f"loading ERA5FieldTransform for {field_type}")
        transforms[field_type] = ERA5FieldTransform(
            mean_path=transform_conf["mean_path"],
            std_path=transform_conf["std_path"],
            vars_3D=field_conf.get("vars_3D", []),
            vars_2D=field_conf.get("vars_2D", []),
            num_levels=field_conf.get("num_levels", 0),
            device=device,
        )
    return transforms