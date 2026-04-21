"""
multi_source.py
---------------
MultiSourceDataset: primary entry point for all data loading.

Wraps one or more registered source datasets (ERA5, MRMS, …) and returns a
dict nested by source name.  Only sources whose keys appear under
``config["source"]`` are instantiated; absent sources are silently skipped.
The wrapper is always used as the entry point — even when only one source
is configured.

Sample structure returned by __getitem__::

    {
        "era5": {
            "input":    {"era5/prognostic/3d/T": tensor, ...},
            "target":   {"era5/prognostic/3d/T": tensor, ...},  # return_target only
            "metadata": {"input_datetime": int, "target_datetime": int},
        },
        "mrms": {
            "input":    {"mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor, ...},
            "target":   {"mrms/prognostic/2d/MultiSensor_QPE_01H_Pass2_00.00": tensor, ...},
            "metadata": {"input_datetime": int, "target_datetime": int},
        },
    }

Pre-block pipeline (applied after the DataLoader)::

    MultiSourceDataset
        → ERA5ScalePreBlock   (scale on native ERA5 grid)
        → MRMSScalePreBlock   (scale on native MRMS grid)
        → MRMSRegridPreBlock  (regrid MRMS → ERA5 grid)
        → MergePreBlock       (flatten nested dict → single input/target dict)
        → Model

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
from credit.datasets.hrrr import HRRRDataset

logger = logging.getLogger(__name__)

# Maps config["source"] keys to Dataset classes.
# Add entries here to register new data sources.
_SOURCE_REGISTRY: dict[str, type] = {
    "ERA5": ERA5Dataset,
    "MRMS": MRMSDataset,
    "HRRR": HRRRDataset,
    "HRRR_NAT": HRRRDataset,
    "HRRR_SUBH": HRRRDataset,
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
            sub-dataset's ``static_metadata`` attribute.  Example::

                {
                    "era5": {"levels": [1000, 850, 500, 300], "datetime_fmt": "unix_ns"},
                    "mrms": {"datetime_fmt": "unix_ns"},
                }
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

        The ``(t, i)`` tuple is passed unchanged to every active sub-dataset.
        Each sub-dataset applies its own field-type / step-index logic
        (e.g. ERA5 and MRMS both skip prognostic disk reads at ``i > 0``).

        Args:
            args: ``(t, i)`` where *t* is the current timestamp (nanoseconds
                or pd.Timestamp) and *i* is the within-sequence step index
                produced by the sampler.

        Returns:
            Dict keyed by lowercase source name.  Each value is the
            sub-dataset's own sample dict::

                {"input": {...}, "target": {...}, "metadata": {...}}
        """
        return {name: ds[args] for name, ds in self.datasets.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _intersect_timestamps(self) -> pd.DatetimeIndex:
        """Return timestamps common to all active source datasets.

        Returns:
            Sorted DatetimeIndex containing only timestamps present in every
            source's ``datetimes`` index.  Returns an empty DatetimeIndex
            when no sources are configured.
        """
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
