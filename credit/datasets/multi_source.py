"""
multi_source.py
---------------
MultiSourceDataset: primary entry point for all data loading.

Wraps one or more registered source datasets and returns a dict nested by
source name.  Only sources whose keys appear under ``config["source"]`` are
instantiated; absent sources are silently skipped.

Sample structure returned by __getitem__::

    {
        "input":    {<user_provided_name>: {"<user_provided_name>/prognostic/3d/T": tensor, ...}, ...},
        "target":   {<user_provided_name>: {"<user_provided_name>/prognostic/3d/T": tensor, ...}, ...},  # return_target only
        "metadata": {<user_provided_name>: {"input_datetime": int, "target_datetime": int}, ...},
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
    "new_source": ("credit.datasets.new_source", "NewSourceDataset"),

    # The dataset class must accept (config, return_target) and expose
    # a ``datetimes`` attribute (pd.DatetimeIndex).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import pandas as pd

from credit.datasets._utils import (  # pyright: ignore[reportPrivateUsage]
    build_time_index,
    filter_index_by_labels,
    is_standard_calendar,
    most_restrictive_calendar,
    normalize_calendar,
    to_calendar,
)
from credit.datasets.base_dataset import AbstractBaseDataset, BaseDataset

logger = logging.getLogger(__name__)

# Maps config["source"] dataset_type keys to (module_path, class_name) pairs.
# Modules are imported on first use so optional heavy dependencies (gcsfs,
# herbie, s3fs, …) are never loaded unless that source type is actually
# requested.  Add entries here to register new built-in data sources; custom
# datasets should instead be registered via custom_objects (see credit.registry),
# which populates credit.datasets._DATASET_REGISTRY.
_SOURCE_REGISTRY: dict[str, tuple[str, str]] = {
    "base": ("credit.datasets.base_dataset", "BaseDataset"),  # placeholders / testing
    "local": ("credit.datasets.local", "LocalDataset"),
    "arco_era5": ("credit.datasets.era5", "ARCOERA5Dataset"),
    "weatherbench2_era5": ("credit.datasets.era5", "WeatherBench2ERA5Dataset"),
    "mrms": ("credit.datasets.mrms", "MRMSDataset"),
    "goes": ("credit.datasets.goes", "GOESDataset"),
    "hrrr": ("credit.datasets.hrrr", "HRRRDataset"),
    "hrrr_nat": ("credit.datasets.hrrr", "HRRRDataset"),
    "hrrr_subh": ("credit.datasets.hrrr", "HRRRDataset"),
    "tisr": ("credit.datasets.tisr", "TISRDataset"),
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

    The module containing the class is imported lazily on first call so that
    optional heavy dependencies are not loaded unless this source type is used.

    Built-in source types (e.g. "local", "arco_era5") are matched against
    ``_SOURCE_REGISTRY``. If no built-in matches, falls back to
    ``credit.datasets._DATASET_REGISTRY`` so that datasets registered via
    ``custom_objects`` in the config can be used as a source's ``dataset_type``.
    Both registries are matched case-sensitively.

    Args:
        source_cfg: Config dict for a single source (e.g. config["source"]["Example_ERA5"]).

    Returns:
        Dataset class corresponding to the "dataset_type" field.

    Raises:
        ValueError: If the "dataset_type" field is missing or does not correspond to a registered dataset.
    """
    dataset_type = source_cfg.get("dataset_type", "")
    if not dataset_type:
        raise ValueError("Source config must contain a 'dataset_type' field.")

    entry = _SOURCE_REGISTRY.get(dataset_type)
    if entry is not None:
        module_path, class_name = entry
        return getattr(importlib.import_module(module_path), class_name)

    from credit.datasets import _DATASET_REGISTRY, _load_dataset_entry  # avoid loading credit.datasets eagerly

    if dataset_type in _DATASET_REGISTRY:
        return _load_dataset_entry(dataset_type)

    raise ValueError(
        f"Unrecognized dataset_type '{dataset_type}' in source config. "
        f"Must be one of the built-in sources {list(_SOURCE_REGISTRY.keys())} "
        "or a dataset registered via custom_objects."
    )


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

    def __init__(self, config: dict[str, Any], return_target: bool = False, label: str | None = None) -> None:
        super().__init__(config, return_target)
        self.datasets: dict[str, BaseDataset] = {}
        source_cfg = config.get("source", {})

        # Loop through all the options in the source config. These will have user
        # provided (unique) names.
        for user_dataset_name in source_cfg.keys():
            # Pass in just the sub-config for this source to avoid confusion
            # with multisource datasets (e.g., HRRR and HRRR_NAT)
            sub_config = make_single_source_subconfig(config, user_dataset_name)
            # Route to the appropriate dataset class based on the "dataset_name" field in the sub-config
            cls = route_to_dataset_class(sub_config["source"][user_dataset_name])
            self.datasets[user_dataset_name] = cls(sub_config, return_target=return_target)
            prefix = f"MultiSourceDataset ({label})" if label else "MultiSourceDataset"
            logger.info(f"{prefix}: registered dataset '{user_dataset_name}' with class '{cls.__name__}'")

        self.dt: pd.Timedelta = pd.Timedelta(config["timestep"])
        self.datetimes: pd.Index = self._build_master_clock(config)  # also sets self.calendar

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
            Dict keyed by data type, each value being a dict of source name
            to that source's data::

                {"input": {"era5": {...}, ...}, "target": {...}, "metadata": {...}}
        """
        raw = {name: ds[args] for name, ds in self.datasets.items()}
        result: dict[str, dict[str, Any]] = {}
        for source_name, source_dict in raw.items():
            for data_type, data in source_dict.items():
                result.setdefault(data_type, {})[source_name] = data
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_master_calendar(self, config: dict[str, Any]) -> str:
        """Resolve the master-clock calendar and validate it against the sources.

        An explicit data-level ``calendar:`` key wins; otherwise the most
        restrictive calendar across sources is used (a non-standard calendar's
        dates are a strict subset of the standard dates we support, so every
        source can serve the master clock's labels). A non-standard source
        under a less restrictive master clock is an init-time error — the
        sampler's step arithmetic would generate ticks (e.g. Feb 29) that the
        source cannot represent, and intersection cannot prevent that.
        """
        source_calendars = {name: getattr(ds, "calendar", "standard") for name, ds in self.datasets.items()}
        explicit = config.get("calendar")
        master_calendar = (
            normalize_calendar(explicit) if explicit else most_restrictive_calendar(source_calendars.values())
        )

        for name, cal in source_calendars.items():
            if not is_standard_calendar(cal) and normalize_calendar(cal) != master_calendar:
                raise ValueError(
                    f"MultiSourceDataset: source '{name}' uses calendar '{cal}' but the master clock "
                    f"calendar is '{master_calendar}'. The master clock must use the most restrictive "
                    "source calendar; remove or fix the data-level `calendar:` override, or align the sources."
                )
        if not is_standard_calendar(master_calendar):
            logger.info("MultiSourceDataset: master clock calendar is '%s'", master_calendar)
        return master_calendar

    def _build_master_clock(self, config: dict[str, Any]) -> pd.Index:
        """Build the master sampling clock from the global config.

        The clock is anchored to the global ``start_datetime``, ``end_datetime``,
        and ``timestep``, and built on the master calendar (see
        ``_resolve_master_calendar``): a ``pd.DatetimeIndex`` for the standard
        family, an ``xr.CFTimeIndex`` otherwise.  For each source:

        - **Normal sources** (no ``temporal_mode``): the clock is filtered to
          timestamps that exist exactly in that source's native datetimes.  A
          warning is emitted when the source's native timestep differs from the
          master clock timestep and ``temporal_mode`` is not set.
        - **Persist sources** (``temporal_mode: persist``): the clock is only
          clipped to the source's coverage range.  Fine-resolution master-clock
          ticks are snapped to the last native timestamp inside
          ``BaseDataset.__getitem__``.
        """
        self.calendar: str = self._resolve_master_calendar(config)

        if "datetimes" in config:
            # Injected init times (e.g. rollout_gen2 inference): treat as date
            # labels and convert into the master calendar. A label that does not
            # exist there (e.g. Feb 29 init for noleap sources) raises loudly.
            converted = [to_calendar(t, self.calendar) for t in config["datetimes"]]
            if is_standard_calendar(self.calendar):
                return pd.DatetimeIndex(converted)
            import xarray as xr  # deferred: standard path never needs it

            return xr.CFTimeIndex(converted)

        if not self.datasets:
            return pd.DatetimeIndex([])

        master_dt = pd.Timedelta(config["timestep"])
        num_history_steps = config.get("history_len", 1)
        num_forecast_steps = config.get("forecast_len", 1)
        # Convert bounds to the master calendar *before* the horizon/history
        # arithmetic so it is calendar-correct near leap days.
        master_start = (
            to_calendar(pd.Timestamp(config["start_datetime"]), self.calendar) + (num_history_steps - 1) * master_dt
        )
        master_end = to_calendar(pd.Timestamp(config["end_datetime"]), self.calendar)

        master = build_time_index(master_start, master_end - num_forecast_steps * master_dt, master_dt, self.calendar)

        source_cfg = config.get("source", {})
        for name, ds in self.datasets.items():
            if not len(ds.datetimes):
                continue
            temporal_mode = source_cfg.get(name, {}).get("temporal_mode")

            if temporal_mode == "persist":
                # Clip master clock to the source's coverage window only.
                # Bounds are converted to the master calendar so the comparison
                # is same-type (mixed pd/cftime comparisons raise TypeError).
                ds_start = to_calendar(ds.datetimes[0], self.calendar)
                ds_end = to_calendar(ds.datetimes[-1], self.calendar)
                master = master[(master >= ds_start) & (master <= ds_end + ds.dt)]
            else:
                # Exact match required — warn if resolutions differ
                if hasattr(ds, "dt") and ds.dt != master_dt:
                    logger.warning(
                        f"MultiSourceDataset: source '{name}' has timestep {ds.dt} which differs "
                        f"from the master clock timestep {master_dt}, but 'temporal_mode' is not set. "
                        "Consider setting temporal_mode: persist in the source config if this source "
                        "should persist its last sample between master-clock ticks."
                    )
                if isinstance(master, pd.DatetimeIndex) and isinstance(ds.datetimes, pd.DatetimeIndex):
                    master = master[master.isin(ds.datetimes)]
                else:
                    # Mixed or cftime index types: isin compares element-wise
                    # across types and would silently empty the clock, so
                    # intersect by calendar-agnostic date labels instead.
                    master = filter_index_by_labels(master, ds.datetimes)

        if not len(master):
            logger.warning(
                "MultiSourceDataset: master clock is empty after applying source constraints. "
                "Check that start_datetime/end_datetime and timestep are consistent across sources."
            )
        return master
