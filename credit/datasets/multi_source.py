"""
multi_source.py
---------------
MultiSourceDataset: primary entry point for all data loading.

Wraps one or more registered source datasets and returns a dict nested by
source name.  Only sources whose keys appear under ``config["source"]`` are
instantiated; absent sources are silently skipped.

Sample structure returned by __getitem__::

    {
        "era5": {
            "input":    {"era5/prognostic/3d/T": tensor, ...},
            "target":   {"era5/prognostic/3d/T": tensor, ...},  # return_target only
            "metadata": {"input_datetime": int, "target_datetime": int},
        },
    }

Usage::

    from credit.datasets.multi_source import MultiSourceDataset
    from credit.samplers import DistributedMultiStepBatchSampler
    from torch.utils.data import DataLoader

    dataset = MultiSourceDataset(config["data"], return_target=True)
    sampler = DistributedMultiStepBatchSampler(dataset, batch_size=4,
                  shuffle=True, num_replicas=1, rank=0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)

Extending with a new source::

    # In _SOURCE_REGISTRY, add:
    "NewSource": NewSourceDataset,

    # The dataset class must accept (config, return_target) and expose
    # a ``datetimes`` attribute (pd.DatetimeIndex).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from credit.datasets.base_dataset import AbstractBaseDataset

from credit.datasets.base_dataset import BaseDataset
from credit.datasets.era5 import ERA5Dataset, ARCOERA5Dataset
from credit.datasets.mrms import MRMSDataset
from credit.datasets.goes import GOESDataset
from credit.datasets.hrrr import HRRRDataset

logger = logging.getLogger(__name__)

# Maps config["source"] keys to Dataset classes.
# Add entries here to register new data sources.
_SOURCE_REGISTRY: dict[str, type] = {
    "BASE": BaseDataset,  # for placeholders, testing, and examples
    "ERA5": ERA5Dataset,
    "ARCO_ERA5": ARCOERA5Dataset,
    "MRMS": MRMSDataset,
    "GOES": GOESDataset,
    "HRRR": HRRRDataset,
    "HRRR_NAT": HRRRDataset,
    "HRRR_SUBH": HRRRDataset,
}


def make_single_source_subconfig(config: dict[str, Any], user_dataset_name: str) -> dict[str, Any]:
    """
    Return a modified config dict containing only the specified source.

    This is used internally to instantiate each sub-dataset with a config
    containing just its own source config block, to avoid confusion with
    multisource config fields (e.g. HRRR vs HRRR_NAT vs HRRR_SUBH).

    Args:
        config: Original multisource config dict.
        user_dataset_name: Unique dataset name specified by the user in config["source"] (e.g. "Example_ERA5").

    Returns:
        New config dict containing only the specified source's config block.
    """
    return {
        "source": {user_dataset_name: config["source"][user_dataset_name]},
        **{k: v for k, v in config.items() if k != "source"},
    }


def route_to_dataset_class(source_cfg: dict[str, Any]) -> type:
    """
    Return the appropriate Dataset class based on the "dataset_type" field in the source config.

    Args:
        source_cfg: Config dict for a single source (e.g. config["source"]["Example_ERA5"]).

    Returns:
        Dataset class corresponding to the "dataset_type" field.

    Raises:
        ValueError: If the "dataset_type" field is missing or does not correspond to a registered dataset.
    """
    dataset_type = source_cfg.get("dataset_type", "").upper()
    if not dataset_type:
        raise ValueError("Source config must contain a 'dataset_type' field.")
    cls = _SOURCE_REGISTRY.get(dataset_type)
    if not cls:
        raise ValueError(
            f"Unrecognized dataset_type '{dataset_type}' in source config. Must be one of: {list(_SOURCE_REGISTRY.keys())}"
        )
    return cls


class MultiSourceDataset(AbstractBaseDataset):
    """CREDIT Dataset that combines multiple source datasets.

    Instantiates one sub-dataset per source key found in ``config["source"]``,
    computes the intersection of their valid timestamps, and delegates each
    ``__getitem__`` call to all active sub-datasets.

    See module docstring for full output structure and usage examples.

    Note that we inherit from AbstractBaseDataset _rather_ than BaseDataset.

    Attributes:
        datasets: Ordered mapping of lowercase source name to its Dataset
            instance (e.g. ``{"era5": ERA5Dataset, "mrms": MRMSDataset}``).
        datetimes: DatetimeIndex of timestamps valid for *all* active sources
            (intersection of each source's own ``datetimes``).
        static_metadata: Per-source static metadata aggregated from each
            sub-dataset's ``static_metadata`` attribute.
    """

    def __init__(self, config: dict[str, Any], return_target: bool = False) -> None:
        self.datasets: dict[str, AbstractBaseDataset] = {}
        source_cfg = config.get("source", {})

        # Loop through all the options in the source config. These will have user
        # provided (unique) names.
        for user_dataset_name in source_cfg.keys():
            # Pass in just the sub-config for this source to avoid confusion
            # with multisource datasets (e.g., HRRR and HRRR_NAT)
            sub_config = make_single_source_subconfig(config, user_dataset_name)
            # Route to the appropriate dataset class based on the "dataset_name" field in the sub-config
            cls = route_to_dataset_class(sub_config["source"][user_dataset_name])
            self.datasets[user_dataset_name] = cls(sub_config, return_target)
            logger.info(f"MultiSourceDataset: registered dataset '{user_dataset_name}' with class '{cls.__name__}'")

        self.datetimes: pd.DatetimeIndex = self._intersect_timestamps()

        self.static_metadata: dict[str, dict[str, Any]] = {
            name: ds.static_metadata for name, ds in self.datasets.items()
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple[pd.Timestamp, int]) -> dict[str, dict[str, Any]]:
        """Return a dict of per-source sample dicts.

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler.

        Returns:
            Dict keyed by lowercase source name, each value being the
            sub-dataset's own sample dict::

                {"input": {...}, "target": {...}, "metadata": {...}}
        """
        return {name: ds[args] for name, ds in self.datasets.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _intersect_timestamps(self) -> pd.DatetimeIndex:
        """Return timestamps common to all active source datasets."""
        if not self.datasets:
            return pd.DatetimeIndex([])

        sets: list[set[pd.Timestamp]] = [set(ds.datetimes) for ds in self.datasets.values()]
        common = sets[0].intersection(*sets[1:])
        if not common:
            logger.warning(
                "MultiSourceDataset: timestamp intersection across sources is empty. "
                "Check that start_datetime/end_datetime overlap for all configured sources."
            )
        return pd.DatetimeIndex(sorted(common))
