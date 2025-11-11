import logging

from credit.transform.era5_transforms import ERA5StandardTransform

logger = logging.getLogger(__name__)


transform_types = {
    "ERA5": (ERA5StandardTransform, "loading an ERA5 transformer class")
}


def load_transform(conf, source, device):
    
    # try:
    transform, message = transform_types[source]
    logger.info(message)
    return transform(conf, device)
    # except:
    #     raise ValueError(f"a transform for source {source} does not exist")