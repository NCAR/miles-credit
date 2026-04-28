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

import pandas as pd
from torch.utils.data import Dataset

from credit.datasets.era5 import ERA5Dataset
from credit.datasets.MRMS import MRMSDataset
from credit.datasets.goes import GOESDataset

logger = logging.getLogger(__name__)

# Maps config["source"] keys to Dataset classes.
# Add entries here to register new data sources.
_SOURCE_REGISTRY: dict[str, type] = {
    "ERA5": ERA5Dataset,
    "MRMS": MRMSDataset,
    "GOES": GOESDataset,
}


class MultiSourceDataset(Dataset):
    """PyTorch Dataset that combines multiple source datasets.

    Instantiates one sub-dataset per source key found in ``config["source"]``,
    computes the intersection of their valid timestamps, and delegates each
    ``__getitem__`` call to all active sub-datasets.

    See module docstring for full output structure and usage examples.

    Attributes:
        datasets: Ordered mapping of lowercase source name to its Dataset
            instance (e.g. ``{"era5": ERA5Dataset, "mrms": MRMSDataset}``).
        datetimes: DatetimeIndex of timestamps valid for *all* active sources
            (intersection of each source's own ``datetimes``).
        static_metadata: Per-source static metadata aggregated from each
            sub-dataset's ``static_metadata`` attribute.
    """

    def __init__(self, config: dict, return_target: bool = False) -> None:
        self.datasets: dict[str, Dataset] = {}
        source_cfg = config.get("source", {})

        for key, cls in _SOURCE_REGISTRY.items():
            if key in source_cfg:
                self.datasets[key.lower()] = cls(config, return_target)
                logger.info("MultiSourceDataset: registered source '%s'", key.lower())
            else:
                logger.debug("MultiSourceDataset: source '%s' not in config, skipping", key)

        self.datetimes: pd.DatetimeIndex = self._intersect_timestamps()
        self.static_metadata: dict[str, dict] = {name: ds.static_metadata for name, ds in self.datasets.items()}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datetimes)

    def __getitem__(self, args: tuple) -> dict:
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
        sets = [set(ds.datetimes) for ds in self.datasets.values()]
        common = set.intersection(*sets)
        if not common:
            logger.warning(
                "MultiSourceDataset: timestamp intersection across sources is empty. "
                "Check that start_datetime/end_datetime overlap for all configured sources."
            )
        return pd.DatetimeIndex(sorted(common))
