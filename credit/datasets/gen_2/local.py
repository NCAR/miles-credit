"""
local.py
--------
LocalDataset: generic PyTorch Dataset for loading atmospheric data from local
NetCDF/Zarr files. Supports any combination of prognostic, dynamic_forcing,
static, and diagnostic field types with optional 3D (multi-level) and 2D
(surface/single-level) variables.

Sample structure returned by __getitem__::

    {
        "input": {
            "{source_name}/prognostic/3d/T":        tensor,  # (n_levels, 1, lat, lon)
            "{source_name}/prognostic/2d/SP":       tensor,  # (1,        1, lat, lon)
            "{source_name}/dynamic_forcing/2d/tsi": tensor,
            "{source_name}/static/2d/LSM":          tensor,
            ...
        },
        "target": {                                  # only when return_target=True
            "{source_name}/prognostic/3d/T":        tensor,
            "{source_name}/prognostic/2d/SP":       tensor,
            ...
        },
        "metadata": {
            "input_datetime":  int,                  # nanoseconds since epoch
            "target_datetime": int,                  # only when return_target=True
        },
    }

Output key format (flat, slash-delimited)::

    "{source_name}/{field_type}/{dim}/{varname}"

    field_type: "prognostic" | "dynamic_forcing" | "static" | "diagnostic"
    dim       : "2d"  (surface / single-level)
                "3d"  (multi-level upper-air; requires level_coord in config;
                       if levels is omitted all levels in the file are used)
    varname   : variable name as given in config (e.g. "T", "SP", "tsi")

Tensor shapes (no batch dimension)::

    3D variable : (n_levels, 1, lat, lon)   — n_levels = len(config levels)
    2D variable : (1,        1, lat, lon)   — singleton level dim

After DataLoader collation the batch dimension is prepended::

    (batch, n_levels, 1, lat, lon)

File naming:
    Each field type supports an optional ``filename_time_format`` config key
    that specifies a strftime format string describing how the datetime appears
    in the file name.  Defaults to ``"%Y"`` (annual files).

    Examples::

        filename_time_format: "%Y"       # data_2021.zarr
        filename_time_format: "%Y_%m"    # data_2021_06.nc
        filename_time_format: "%Y%m%d"   # data_20210601.nc

    If only a single file matches the glob pattern, ``filename_time_format`` is
    ignored and that file is used for all timestamps.
"""

from __future__ import annotations

import logging
from glob import glob
from os.path import expandvars
from typing import Any

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.datasets.gen_2._utils import (  # pyright: ignore[reportPrivateUsage]
    _find_file,
    _path_template_to_glob,
    _to_cftime,
    is_standard_calendar,
    normalize_calendar,
    to_calendar,
)
from credit.datasets.gen_2.base_dataset import BaseDataset
from credit.datasets.gen_2.grid_utils import (
    expected_spatial_shape,
    resolve_source_grid,
    write_source_grid_schema_if_missing,
)

logger = logging.getLogger(__name__)


def first_data_file(source_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return ``(path, field_cfg)`` for the first data file a local source can open.

    Field types are tried in the order ``prognostic``, ``dynamic_forcing``,
    ``diagnostic``, ``static``; a field with no ``path``, or whose glob matches
    nothing, is skipped. Returns None when no field yields a file.

    Module-level rather than a method so ``credit check`` can probe a source's
    data without constructing a dataset — keeping the two in agreement about
    which file "the source's data" means.
    """
    variables = source_cfg.get("variables") or {}
    for field_type in ("prognostic", "dynamic_forcing", "diagnostic", "static"):
        field_cfg = variables.get(field_type)
        if not isinstance(field_cfg, dict) or not field_cfg.get("path"):
            continue
        files = sorted(glob(_path_template_to_glob(field_cfg["path"])))
        if files:
            return files[0], field_cfg
    return None


class LocalDataset(BaseDataset):
    """Generic PyTorch Dataset for local NetCDF/Zarr atmospheric data files.

    See module docstring for full description of output format and file naming.

    Example YAML configuration::

        data:
          source:
            My_Surface_Data:  # User-provided name (arbitrary key)
              dataset_type: "local"
              grid_type: "unstructured"         # Recommended: explicit override (auto-detection is a
                                                # size-based heuristic and can misfire -- see Assumptions)
              coordinate_file: "/data/grid.nc"  # Optional: read lat/lon from here instead of the data
                                                # files, for data that carries no coordinates of its own
              lat_name: "latCell"               # Optional: name the coordinate variables directly when
              lon_name: "lonCell"               # the built-in name table doesn't recognise them
              level_coord: "level"
              levels: [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
              variables:
                prognostic:
                  vars_3D: ['T', 'U', 'V', 'Q']
                  vars_2D: ['SP', 't2m']
                  path: "/data/era5_*.zarr"
                  filename_time_format: "%Y"        # annual (default)
                dynamic_forcing:
                  vars_2D: ['tsi']
                  path: "/data/solar_*.nc"
                  filename_time_format: "%Y_%m"     # monthly
                static:
                  vars_2D: ['Z_GDS4_SFC', 'LSM']
                  path: "/data/lsm.nc"
                  # single file — filename_time_format not needed
                diagnostic: null

          start_datetime: "2017-01-01"
          end_datetime: "2019-12-31"
          timestep: "6h"
          forecast_len: 1

    Assumptions:
        1. A "time" dimension / coordinate is present for non-static fields (defaults to
           "time", configurable via `time_coord`).
        2. A level coordinate (name given by ``level_coord``) represents the
           vertical axis of 3D variables.
        3. Dimension order for Structured: (time, level, latitude, longitude) for 3D;
           (time, latitude, longitude) for 2D.
        4. Dimension order for Unstructured: (time, level, ncol) for 3D; (time, ncol) for 2D.
        5. Static fields are automatically replicated along the time axis. If a static
           file contains a dummy time dimension, it is safely ignored.
        6. Finding and classifying lat/lon arrays (regardless of naming convention)
           is delegated to `credit.datasets.gen_2.grid_utils.resolve_source_grid`.
           There is no required name for the flattened spatial dimension itself
           (e.g. "ncol" above is illustrative, not enforced) -- auto-detection
           instead prefers lat/lon sharing one real dimension in the file, falling
           back to a weaker same-length heuristic; set `grid_type:` explicitly to
           bypass both. Coordinate *variable* names are matched against a table of
           known conventions (longitude/latitude, lon/lat, XLONG/XLAT, ...); set
           `lon_name:`/`lat_name:` in the source config to name them directly.
           Note that `x`/`y` are never treated as geographic coordinates -- on a
           projected grid they carry projection units, not degrees.
        7. An unstructured source resolves and writes on its native mesh: output
           uses a single `ncol` dimension with per-cell `latitude`/`longitude` as
           CF auxiliary coordinates. A `Regridder` preblock onto a structured
           destination grid is therefore an option for such a source, not a
           requirement (it was required previously, when `GridSchema` represented
           only rectilinear/curvilinear grids).
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        """Initialize LocalDataset with config parsing, timestamp generation, and file mapping.

        Args:
            data_config (dict[str, Any]): Data configuration dictionary from YAML config.
            return_target (bool, optional): Whether to return target variables. Defaults to False.
        """
        # Must exist before super().__init__() -- it calls _load_dt/_load_start_datetime/
        # _load_end_datetime/_load_cycle_year, whose overrides below memoize onto this.
        self._cyclic_time_info_cache: dict[str, Any] | None = None
        super().__init__(data_config, return_target)
        assert self.curr_source_cfg["dataset_type"] == "local", (
            f"Expected dataset_type 'local' in config for LocalDataset, got '{self.curr_source_cfg['dataset_type']}'"
        )

        self.dataset_type = "local"
        self.level_coord: str | None = self.curr_source_cfg.get("level_coord")
        self.levels: list | None = self.curr_source_cfg.get("levels")
        grid = self._find_grid(self.curr_source_cfg)  # also fills self.levels from the same open file, if absent
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "calendar": self.calendar,
            "datetime_fmt": "unix_ns" if is_standard_calendar(self.calendar) else f"cf_ns:{self.calendar}",
            "grid": grid,
        }
        self.mode = "local"
        self.time_coord = self.curr_source_cfg.get("time_coord", "time")
        self.init_register_all_fields()

    def _resolve_calendar(self, data_config: dict[str, Any], curr_source_config: dict[str, Any]) -> str | None:
        """Resolve the calendar from config, falling back to finding it in the data.

        Config (source-level then data-level ``calendar:`` key) wins; otherwise
        the first time-bearing data file's time coordinate is inspected once at
        init. Returns None (→ "standard") when neither yields an answer.
        """
        cal = super()._resolve_calendar(data_config, curr_source_config)
        if cal:
            return cal
        return self._find_calendar(curr_source_config)

    def _find_calendar(self, source_cfg: dict[str, Any]) -> str | None:
        """Read the CF calendar from the first available time-bearing file.

        xarray decodes non-standard-calendar time coordinates to cftime objects
        automatically, so the coordinate's element type is the discriminator.
        Failures are non-fatal: warn and fall back to the standard default.
        """
        time_coord = source_cfg.get("time_coord", "time")
        engine = source_cfg.get("engine")
        variables = source_cfg.get("variables") or {}
        for field_type in ("prognostic", "dynamic_forcing", "diagnostic"):
            field_cfg = variables.get(field_type)
            if not isinstance(field_cfg, dict) or not field_cfg.get("path"):
                continue
            files = sorted(glob(_path_template_to_glob(field_cfg["path"])))
            if not files:
                continue
            try:
                with xr.open_dataset(files[0], engine=engine) as ds:
                    if time_coord not in ds:
                        continue
                    t0 = ds[time_coord].values.ravel()[0]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LocalDataset '%s': could not find calendar in %s (%s); assuming 'standard'. "
                    "Set an explicit `calendar:` key in the source config to silence this.",
                    self.curr_source_name,
                    files[0],
                    exc,
                )
                return None
            calendar = normalize_calendar(t0.calendar) if isinstance(t0, cftime.datetime) else "standard"
            if not is_standard_calendar(calendar):
                logger.info("LocalDataset '%s': found calendar '%s' in %s", self.curr_source_name, calendar, files[0])
            return calendar
        return None

    def _infer_cyclic_time_info(self, source_cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Infer dt/start_datetime/end_datetime/cycle_year for a cyclic source
        from its own data, when the config doesn't declare them explicitly.

        Opens the first available file (same file-finding as ``_find_calendar``/
        ``_find_grid``) and reads its full time coordinate: ``dt`` is the
        spacing between its first two timestamps, start/end are its min/max,
        and ``cycle_year`` is the single calendar year they all share -- a
        cyclic source's file should only ever contain one representative
        cycle, so this raises loudly if it doesn't (rather than silently
        picking one). Memoized on ``self._cyclic_time_info_cache`` since
        ``_load_dt``/``_load_start_datetime``/``_load_end_datetime``/
        ``_load_cycle_year`` may all need it. Failures (missing time
        coordinate, unreadable file, a single-timestamp file with no
        inferrable spacing) are non-fatal: warn and return None, in which case
        the ordinary required-config-key error surfaces instead.

        Raises:
            ValueError: if the file's timestamps span more than one calendar
                year (ambiguous cycle_year -- set it explicitly to disambiguate).
        """
        if self._cyclic_time_info_cache is not None:
            return self._cyclic_time_info_cache

        time_coord = source_cfg.get("time_coord", "time")
        engine = source_cfg.get("engine")
        variables = source_cfg.get("variables") or {}
        for field_type in ("prognostic", "dynamic_forcing", "diagnostic", "static"):
            field_cfg = variables.get(field_type)
            if not isinstance(field_cfg, dict) or not field_cfg.get("path"):
                continue
            files = sorted(glob(_path_template_to_glob(field_cfg["path"])))
            if not files:
                continue
            try:
                with xr.open_dataset(files[0], engine=engine) as ds:
                    if time_coord not in ds:
                        continue
                    raw = ds[time_coord].values.ravel()
                    # Nanosecond-precision datetime64 (xarray's usual CF-decoded dtype)
                    # can't round-trip through .tolist() as datetime objects (numpy
                    # gives plain int in that case) -- pd.DatetimeIndex handles it
                    # correctly. Non-standard calendars decode to an object array of
                    # cftime.datetime, which already sorts/compares/has .year natively.
                    if np.issubdtype(raw.dtype, np.datetime64):
                        times = list(pd.DatetimeIndex(raw).sort_values())
                    else:
                        times = sorted(raw.tolist())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LocalDataset '%s': could not infer cyclic time info from %s (%s). Set "
                    "cycle_year/start_datetime/end_datetime/timestep explicitly in the source config.",
                    self.curr_source_name,
                    files[0],
                    exc,
                )
                return None

            if not times:
                continue
            if len(times) < 2:
                logger.warning(
                    "LocalDataset '%s': %s has only one timestamp; cannot infer a timestep. "
                    "Set timestep explicitly in the source config.",
                    self.curr_source_name,
                    files[0],
                )
                return None

            years = {t.year for t in times}
            if len(years) > 1:
                raise ValueError(
                    f"LocalDataset '{self.curr_source_name}': cyclic source's data file {files[0]} spans "
                    f"more than one year ({sorted(years)}) -- a cyclic source's data must contain exactly "
                    "one representative cycle. Set cycle_year explicitly to disambiguate, or fix the data."
                )

            start_raw, end_raw = times[0], times[-1]
            if not isinstance(start_raw, cftime.datetime):
                start_raw, end_raw = pd.Timestamp(start_raw), pd.Timestamp(end_raw)

            info = {
                "dt": pd.Timedelta(times[1] - times[0]),
                "start_datetime": start_raw,
                "end_datetime": end_raw,
                "cycle_year": next(iter(years)),
            }
            logger.info(
                "LocalDataset '%s': inferred cyclic time info from %s: %s", self.curr_source_name, files[0], info
            )
            self._cyclic_time_info_cache = info
            return info
        return None

    def _load_dt(
        self, data_config: dict[str, Any], curr_source_config: dict[str, Any], dt_key: str = "timestep"
    ) -> pd.Timedelta:
        """Falls back to inferring dt from the data for a cyclic source with no
        explicit timestep (source or data level); otherwise identical to
        ``BaseDataset._load_dt``. See ``_infer_cyclic_time_info``.
        """
        if (
            curr_source_config.get("temporal_mode") == "cyclic"
            and dt_key not in curr_source_config
            and dt_key not in data_config
        ):
            info = self._infer_cyclic_time_info(curr_source_config)
            if info is not None:
                return info["dt"]
        return super()._load_dt(data_config, curr_source_config, dt_key)

    def _load_start_datetime(
        self,
        data_config: dict[str, Any],
        curr_source_config: dict[str, Any],
        start_datetime_key: str = "start_datetime",
    ) -> pd.Timestamp:
        """Falls back to inferring start_datetime from the data for a cyclic
        source with no explicit start_datetime (source or data level);
        otherwise identical to ``BaseDataset._load_start_datetime``. See
        ``_infer_cyclic_time_info``.
        """
        if (
            curr_source_config.get("temporal_mode") == "cyclic"
            and start_datetime_key not in curr_source_config
            and start_datetime_key not in data_config
        ):
            info = self._infer_cyclic_time_info(curr_source_config)
            if info is not None:
                return info["start_datetime"]
        return super()._load_start_datetime(data_config, curr_source_config, start_datetime_key)

    def _load_end_datetime(
        self, data_config: dict[str, Any], curr_source_config: dict[str, Any], end_datetime_key: str = "end_datetime"
    ) -> pd.Timestamp:
        """Falls back to inferring end_datetime from the data for a cyclic
        source with no explicit end_datetime (source or data level); otherwise
        identical to ``BaseDataset._load_end_datetime``. See
        ``_infer_cyclic_time_info``.
        """
        if (
            curr_source_config.get("temporal_mode") == "cyclic"
            and end_datetime_key not in curr_source_config
            and end_datetime_key not in data_config
        ):
            info = self._infer_cyclic_time_info(curr_source_config)
            if info is not None:
                return info["end_datetime"]
        return super()._load_end_datetime(data_config, curr_source_config, end_datetime_key)

    def _load_cycle_year(self, data_config: dict[str, Any], curr_source_config: dict[str, Any]) -> int | None:
        """Falls back to inferring cycle_year from the data when config gives
        no answer; otherwise identical to ``BaseDataset._load_cycle_year``.
        See ``_infer_cyclic_time_info``.
        """
        cycle_year = super()._load_cycle_year(data_config, curr_source_config)
        if cycle_year is not None:
            return cycle_year
        if curr_source_config.get("temporal_mode") != "cyclic":
            return None
        info = self._infer_cyclic_time_info(curr_source_config)
        return info["cycle_year"] if info is not None else None

    def _find_grid(self, source_cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Read this source's real lat/lon coordinates, once.

        Coordinates come from an explicit ``coordinate_file:`` when the config
        names one, else from the first available data file (the original
        behaviour). The two paths differ in how they treat failure:

        * **Data file** — failures are non-fatal: warn and return None. A file
          that simply has no coordinates is a normal, tolerated situation.
        * **coordinate_file** — failures raise. The user named this file
          specifically to supply the grid, so silently ignoring it (and then
          dying much later in ``GridSchema.resolve`` with an unrelated message)
          would be actively misleading.

        This is a debugging aid (``self.static_metadata["grid"]``) reflecting this
        source's *native* grid. It is not necessarily the grid actually written to
        output — a regridding preblock downstream may change that; see
        ``credit.datasets.gen_2.grid_utils.GridSchema``. Also best-effort persisted
        to ``{save_loc}/{source}_grid_schema.nc`` (see
        ``write_source_grid_schema_if_missing``).

        ``self.levels`` is filled from ``level_coord`` when still unset (absent
        from config) — always from a *data* file, never from the coordinate
        file, since the vertical coordinate is unrelated to horizontal geometry.
        On the data-file path this piggybacks on the same open; on the
        coordinate-file path it shares the open used for grid/data validation.
        If no data file carries the level coordinate (e.g. it happens to be a
        2D-only field type), ``self.levels`` stays None and falls back to
        ``_read_3d_array``'s lazy per-batch resolution.
        """
        engine = source_cfg.get("engine")
        coordinate_file = source_cfg.get("coordinate_file")
        if coordinate_file:
            return self._grid_from_coordinate_file(coordinate_file, source_cfg, engine)

        probe = first_data_file(source_cfg)
        if probe is None:
            return None
        path, _field_cfg = probe
        try:
            with xr.open_dataset(path, engine=engine) as ds:
                grid = resolve_source_grid(ds, source_cfg)
                write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)

                if self.levels is None and self.level_coord in ds.coords:
                    self.levels = ds[self.level_coord].values.tolist()
                return grid

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LocalDataset '%s': could not find grid in %s (%s).",
                self.curr_source_name,
                path,
                exc,
            )
            return None

    def _grid_from_coordinate_file(
        self,
        coordinate_file: str,
        source_cfg: dict[str, Any],
        engine: str | None,
    ) -> dict[str, Any]:
        """Read the grid from an explicit ``coordinate_file:``, then validate it.

        Unlike the data-file path, every failure here raises: an explicitly
        configured coordinate file that cannot be opened, has no recognisable
        coordinate pair, or describes a grid the data does not sit on is a
        config error the user needs to see now, not a warning to scroll past.

        Args:
            coordinate_file: Path from the source config (``$VAR`` expanded).
            source_cfg: This source's config block.
            engine: Optional xarray engine.

        Returns:
            The resolved grid dict.

        Raises:
            ValueError: if the file cannot be opened, yields no coordinate pair,
                or disagrees with the data's spatial shape.
        """
        path = expandvars(coordinate_file)
        try:
            coord_ds = xr.open_dataset(path, engine=engine)
        except Exception as exc:
            raise ValueError(
                f"LocalDataset '{self.curr_source_name}': could not open coordinate_file "
                f"'{path}' ({type(exc).__name__}: {exc})."
            ) from exc

        with coord_ds as ds:
            # resolve_source_grid raises its own descriptive ValueError when the
            # file has no recognisable coordinate pair; let it through unwrapped.
            grid = resolve_source_grid(ds, source_cfg)

        self._fill_levels_and_validate(grid, source_cfg, engine)
        write_source_grid_schema_if_missing(self.curr_source_name, grid, self.save_loc)
        logger.info(
            "LocalDataset '%s': grid read from coordinate_file %s (grid_type=%s, shape=%s).",
            self.curr_source_name,
            path,
            grid["grid_type"],
            expected_spatial_shape(grid),
        )
        return grid

    def _fill_levels_and_validate(
        self,
        grid: dict[str, Any],
        source_cfg: dict[str, Any],
        engine: str | None,
    ) -> None:
        """Open a data file once to fill ``self.levels`` and cross-check *grid*.

        Only used on the ``coordinate_file`` path. Being unable to open a data
        file here is not fatal — the grid itself is already resolved, and the
        ordinary read path will report a missing file far more clearly.
        """
        probe = first_data_file(source_cfg)
        if probe is None:
            return
        path, field_cfg = probe
        try:
            data_ds = xr.open_dataset(path, engine=engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LocalDataset '%s': could not open %s to validate coordinate_file against (%s).",
                self.curr_source_name,
                path,
                exc,
            )
            return

        with data_ds as ds:
            if self.levels is None and self.level_coord in ds.coords:
                self.levels = ds[self.level_coord].values.tolist()
            self._check_grid_matches_data(grid, ds, path, source_cfg, field_cfg)

    def _check_grid_matches_data(
        self,
        grid: dict[str, Any],
        ds: xr.Dataset,
        path: str,
        source_cfg: dict[str, Any],
        field_cfg: dict[str, Any],
    ) -> None:
        """Raise if *grid* cannot describe the variables in *ds*.

        A coordinate file that silently disagrees with the data is the failure
        mode this whole feature introduces: nothing downstream would notice,
        and the run would produce output on plausible-looking but wrong
        coordinates. Every configured variable present in *ds* is measured by
        stripping its time and level dimensions; the grid passes if any one of
        them matches the shape it implies.

        Note ``time_coord`` is read from *source_cfg* rather than ``self``:
        ``_find_grid`` runs before ``self.time_coord`` is assigned in ``__init__``.
        """
        time_coord = source_cfg.get("time_coord", "time")
        expected = expected_spatial_shape(grid)

        seen: dict[str, tuple[int, ...]] = {}
        for vname in (field_cfg.get("vars_3D") or []) + (field_cfg.get("vars_2D") or []):
            if vname not in ds:
                continue
            da = ds[vname]
            spatial = tuple(
                size for dim, size in zip(da.dims, da.shape) if dim not in (time_coord, self.level_coord)
            )
            if spatial == expected:
                return
            seen[vname] = spatial

        if not seen:
            return  # nothing comparable in this file; not evidence of a mismatch

        raise ValueError(
            f"LocalDataset '{self.curr_source_name}': coordinate_file describes a "
            f"{grid['grid_type']} grid of shape {expected}, but no configured variable in "
            f"{path} has that spatial shape (found {seen}). The coordinate file and the data "
            "are on different grids — check that coordinate_file matches this source's data, "
            "or set grid_type/lat_name/lon_name if the grid was classified wrongly."
        )

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the dataset for *field_type* at time *t* and populate *sample*.

        Keys written are ``"{source_name}/{field_type}/3d/{varname}"`` for 3D variables
        and ``"{source_name}/{field_type}/2d/{varname}"`` for 2D variables.

        This is the single-step reader (one timestamp). Multi-step history
        (history_len > 1) is handled by :meth:`_extract_field_window`, which
        opens each file at most once across the window.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t: Timestamp to select.
            sample: Dict to write variable tensors into (modified in place).
                Tensor shapes (no batch dimension):

                - 3D variable: ``(n_levels, 1, spatial_dims...)``
                - 2D variable: ``(1, 1, spatial_dims...)``
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        with xr.open_dataset(_find_file(file_intervals, t)) as ds:
            # Bulletproof static fields: drop dummy time dimensions completely so `.sel` doesn't crash
            if field_type == "static" and self.time_coord in ds.dims:
                ds = ds.isel({self.time_coord: 0}, drop=True)

            ds_t = self._select_at_time(ds, t)
            self._write_field_tensors(ds_t, vars_3D, vars_2D, field_type, sample)

    def _extract_field_window(
        self,
        field_type: str,
        t_history: pd.DatetimeIndex,
        sample: dict[str, Any],
    ) -> None:
        """Load *field_type* over the history window and stack along time.

        Overrides the generic per-step reader in
        :meth:`BaseDataset._extract_field_window` to open each underlying file at
        most once. Timestamps in ``t_history`` may span multiple data files (e.g.
        yearly zarrs around a year boundary), so they are grouped by their
        resolved file and each file is opened a single time.

        For fields without a time dimension in the source dataset (typical
        ``"static"``), the single available slice is replicated along the time
        axis ``len(t_history)`` times.

        Produces the same output as the generic reader: 3D →
        ``(n_levels, len(t_history), spatial_dims...)``, 2D → ``(1, len(t_history), spatial_dims...)``.

        Args:
            field_type: One of ``"prognostic"``, ``"dynamic_forcing"``,
                ``"static"``, ``"diagnostic"``.
            t_history: Chronological timestamps of the history window.
            sample: Dict to write the stacked variable tensors into (modified in place).
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        # History path: load each timestamp in t_history, stack along time.
        # Group by file so each file is opened at most once.
        n_t = len(t_history)
        per_var_3D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_3D}
        per_var_2D: dict[str, list[Any]] = {v: [None] * n_t for v in vars_2D}

        # Resolve each timestamp to a file path and group indices by file.
        groups: dict[str, list[int]] = {}
        for k, tk in enumerate(t_history):
            # No pd.Timestamp(...) wrap: tk may be cftime.datetime (non-standard
            # calendar), which pd.Timestamp cannot consume; _find_file handles
            # both natively.
            path = _find_file(file_intervals, tk)
            groups.setdefault(path, []).append(k)

        for path, indices in groups.items():
            with xr.open_dataset(path) as ds:
                # Bulletproof static fields: drop dummy time dimensions completely
                if field_type == "static" and self.time_coord in ds.dims:
                    ds = ds.isel({self.time_coord: 0}, drop=True)

                has_time = self.time_coord in ds.dims
                if not has_time:
                    # Static-style field: load once, replicate at each index.
                    for k in indices:
                        for v in vars_3D:
                            arr = self._read_3d_array(ds, v)
                            per_var_3D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                        for v in vars_2D:
                            arr = ds[v].values
                            per_var_2D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                else:
                    # Time-indexed field: select each timestamp in this file.
                    for k in indices:
                        ds_t = self._select_at_time(ds, t_history[k])
                        for v in vars_3D:
                            arr = self._read_3d_array(ds_t, v)
                            per_var_3D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
                        for v in vars_2D:
                            arr = ds_t[v].values
                            per_var_2D[v][k] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Concatenate along time dim (dim=1): list of (n_lev, 1, spatial)
        # -> (n_lev, n_t, spatial); 2D analogously.
        for v in vars_3D:
            key = self._get_field_name(field_type, "3d", v)
            sample[key] = torch.cat(per_var_3D[v], dim=1)
        for v in vars_2D:
            key = self._get_field_name(field_type, "2d", v)
            sample[key] = torch.cat(per_var_2D[v], dim=1)

    # ------------------------------------------------------------------
    # Internal helpers for _extract_field
    # ------------------------------------------------------------------

    def _select_at_time(self, ds: xr.Dataset, t: pd.Timestamp | cftime.datetime) -> xr.Dataset:
        """Select a single time slice from *ds* at timestamp *t*.

        Handles both numpy datetime64 and cftime calendars, in both directions:
        *t* may be a plain ``pd.Timestamp`` or a ``cftime.datetime`` (e.g. this
        source's own calendar differs from the master clock's, or a mixed-source
        run pairs a noleap source with a standard-calendar one). If the dataset
        has no time dimension (e.g. static fields), returns the dataset
        unchanged.

        Args:
            ds: The open dataset (or a per-time slice of it).
            t: Timestamp to select.

        Returns:
            The dataset selected at *t*, or unchanged if it has no time dim.
        """
        if self.time_coord not in ds.dims:
            return ds
        file_time0 = ds[self.time_coord].values[0]
        if isinstance(file_time0, cftime.datetime):
            t_sel = to_calendar(t, file_time0.calendar)
            if not isinstance(t_sel, cftime.datetime):
                # standard-named calendar stored as cftime objects (e.g. dates
                # outside pandas' nanosecond range): selection still needs a
                # cftime key.
                t_sel = _to_cftime(t_sel, file_time0.calendar)
        else:
            t_sel = to_calendar(t, "standard")
        return ds.sel({self.time_coord: t_sel})

    def _read_3d_array(self, ds_t: xr.Dataset, vname: str) -> np.ndarray:
        """Read a 3D variable from a per-time-slice dataset, applying level
        selection if configured. Lazily caches ``self.levels`` on first use,
        matching the original single-step logic.

        ``.sel()`` preserves the file's native dimension order, and callers
        (``_write_field_tensors``/``_extract_field_window``) treat the result
        positionally as ``(n_levels, spatial...)`` -- so the level dimension
        must genuinely be first among the remaining (post-time-selection) dims,
        or the level/spatial axes get silently swapped. Checked explicitly here
        rather than assumed, since a shape/size mismatch wouldn't otherwise
        surface until much later (or not at all, for e.g. a square grid).

        Args:
            ds_t: A single-time-slice dataset.
            vname: 3D variable name.

        Returns:
            numpy array of the 3D variable, level-selected if configured.

        Raises:
            ValueError: if ``level_coord`` is present but isn't the first
                dimension of ``vname``.
        """
        da = ds_t[vname]
        if self.level_coord in da.dims and da.dims[0] != self.level_coord:
            raise ValueError(
                f"LocalDataset '{self.curr_source_name}': variable '{vname}' has dims {da.dims}, "
                f"but '{self.level_coord}' must be first (after time selection) -- 3D reading "
                "assumes (level, spatial...) order and treats the result positionally."
            )
        if self.levels is None:
            arr = da.values
            if self.level_coord in ds_t.coords:
                self.levels = ds_t[self.level_coord].values.tolist()
                self.static_metadata["levels"] = self.levels
        else:
            arr = da.sel({self.level_coord: self.levels}, method="nearest").values
        return arr

    def _write_field_tensors(
        self,
        ds_t: xr.Dataset,
        vars_3D: list[str],
        vars_2D: list[str],
        field_type: str,
        sample: dict[str, Any],
    ) -> None:
        """Write single-time-slice tensors for vars_3D/vars_2D into sample.

        Used by the legacy single-step path (history_len == 1).

        Args:
            ds_t: A single-time-slice dataset.
            vars_3D: 3D variable names.
            vars_2D: 2D variable names.
            field_type: The field type being written.
            sample: Dict to write tensors into (modified in place).
        """
        # 3D variables: (n_levels, spatial) → (n_levels, 1, spatial)
        for vname in vars_3D:
            arr = self._read_3d_array(ds_t, vname)
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
            key = self._get_field_name(field_type, "3d", vname)
            sample[key] = tensor

        # 2D variables: (spatial) → (1, 1, spatial)
        for vname in vars_2D:
            arr = ds_t[vname].values
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            key = self._get_field_name(field_type, "2d", vname)
            sample[key] = tensor
