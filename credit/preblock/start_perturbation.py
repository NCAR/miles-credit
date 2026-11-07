import datetime
import math
import os
from numbers import Real

import cftime
import numpy as np
import pandas as pd
import torch

from credit.datasets.gen_2._utils import decode_time
from credit.preblock.base import BasePreblock


def _positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value > 0


def _src(var_key: str) -> str:
    return var_key.split("/")[0]


def start_perturbation_errors(path, fields, amplitude, years, window_days=15, nonnegative=None) -> list[str]:
    """Problems with a ``start_perturbation`` block's arguments (empty list if they are valid).

    Shared by :class:`StartPerturbation` (which raises on them) and ``credit check``.
    The store itself is not opened here.
    """
    errors = []
    if not isinstance(path, str) or not path:
        errors.append(f"path must be the prognostic data store (with %Y for the year), got {path!r}")
    if (
        not isinstance(fields, (list, tuple))
        or not fields
        or not all(isinstance(var, str) and "/" in var for var in fields)
    ):
        errors.append(f"fields must be a non-empty list of variable keys such as CESM/prognostic/3d/T, got {fields!r}")
        fields = []
    elif len(set(fields)) != len(fields):
        errors.append("fields lists a variable twice")
    if isinstance(amplitude, dict):
        if set(amplitude) != set(fields):
            errors.append(f"amplitude as a map needs exactly one entry per field; got keys {sorted(amplitude)}")
        bad = {k: v for k, v in amplitude.items() if not _positive(v)}
        if bad:
            errors.append(f"amplitude values must be positive numbers, got {bad}")
    elif not _positive(amplitude):
        errors.append(f"amplitude must be a positive number or a {{field: number}} map, got {amplitude!r}")
    if (
        not isinstance(years, (list, tuple))
        or len(years) != 2
        or not all(isinstance(y, int) and not isinstance(y, bool) for y in years)
        or years[0] > years[1]
    ):
        errors.append(f"years must be [first, last] donor years, got {years!r}")
    if not _positive(window_days):
        errors.append(f"window_days must be a positive number, got {window_days!r}")
    if nonnegative is not None and (
        not isinstance(nonnegative, (list, tuple)) or not all(var in fields for var in nonnegative)
    ):
        errors.append(f"nonnegative must list keys from fields, got {nonnegative!r}")
    return errors


class StartPerturbation(BasePreblock):
    """Perturb the initial state with a scaled difference of two other states from the same time of year.

    For each batch row (ensemble member) it draws two donor states from other
    times of the data record, at the start's hour of day and within
    ``window_days`` of its calendar day in randomly chosen ``years``, and adds

        delta = amplitude * (s1 - s2) / sqrt(2)

    to every listed field. Two independent states differ by about sqrt(2)
    times the field's anomaly standard deviation, so ``delta`` has roughly
    ``amplitude`` times that standard deviation, and its structure across
    space, levels and variables is that of real weather differences. That
    keeps the perturbation balanced, so it grows like a real error instead of
    partly decaying in the first steps.

    Put it first in ``preblocks.ic_only``: it works on the raw initial state in
    physical units, before any regridding or normalization, and only at the
    start of a forecast. Use donor ``years`` that exclude the forecast period.

    Args:
        path: the prognostic data store, as in the data config, with ``%Y`` for
            the year (one store per year) or without (one store for all years).
        fields: variable keys to perturb, e.g. ``CESM/prognostic/3d/T``; the
            store's variable is the key's last part.
        amplitude: a number, or ``{field: number}``.
        years: ``[first, last]`` donor years, inclusive.
        window_days: donor dates lie within this many days of the start's
            calendar day. Near 1 January and 31 December the window is one-sided
            (donors stay in their year).
        nonnegative: fields clamped at 0 after perturbing (tracers such as Qtot).
        seed: with a seed, the draws depend only on (seed, start time, batch
            row), so every experiment arm that shares the seed gets the same
            perturbation. Without one they come from the global torch RNG.
        time_coord: the store's time coordinate.
        data_type: batch entry to perturb (default ``input``).
    """

    def __init__(
        self,
        path: str,
        fields: list,
        amplitude,
        years: list,
        window_days: float = 15,
        nonnegative: list | None = None,
        seed: int | None = None,
        time_coord: str = "time",
        data_type: str = "input",
    ):
        super().__init__()
        errors = start_perturbation_errors(path, fields, amplitude, years, window_days, nonnegative)
        if errors:
            raise ValueError("start_perturbation: " + "; ".join(errors))
        self.path = os.path.expandvars(path)
        self.fields = list(fields)
        self.amplitude = amplitude
        self.years = list(range(years[0], years[1] + 1))
        self.window = datetime.timedelta(days=window_days)
        self.nonnegative = set(nonnegative or [])
        self.seed = seed
        self.time_coord = time_coord
        self.data_type = data_type
        self._stores = {}
        self._calendar = None

    def _open(self, year: int):
        """The store holding ``year`` (cached), opened lazily."""
        path = self.path.replace("%Y", f"{year:04d}") if "%Y" in self.path else self.path
        if path not in self._stores:
            import xarray as xr

            if not os.path.exists(path):
                raise FileNotFoundError(f"start_perturbation: no data store for {year} at {path}")
            opener = xr.open_zarr if path.rstrip("/").endswith(".zarr") else xr.open_dataset
            self._stores[path] = opener(path)
        return self._stores[path]

    def _calendar_of(self, ds) -> str:
        first = ds[self.time_coord].values[0]
        return first.calendar if isinstance(first, cftime.datetime) else "standard"

    def _candidates(self, year: int, start) -> np.ndarray:
        """Time indices in ``year`` at the start's hour and within the window of its calendar day."""
        ds = self._open(year)
        index = ds.indexes[self.time_coord]
        day = start.day
        if start.month == 2 and day == 29:
            day = 28  # no 29 February in this year (or calendar)
        if isinstance(index, pd.DatetimeIndex):
            target = pd.Timestamp(year, start.month, day, start.hour)
        else:
            target = cftime.datetime(year, start.month, day, start.hour, calendar=self._calendar)
        hours = np.asarray(index.hour)
        years = np.asarray(index.year)
        offset = np.abs(np.asarray((index - target).total_seconds()))
        mask = (hours == start.hour) & (years == year) & (offset <= self.window.total_seconds())
        return np.flatnonzero(mask)

    def _donors(self, rng: np.random.Generator, start) -> tuple:
        """Two different (year, time index) pairs."""
        for _ in range(100):
            picks = []
            for _ in range(2):
                year = int(rng.choice(self.years))
                candidates = self._candidates(year, start)
                if len(candidates):
                    picks.append((year, int(rng.choice(candidates))))
            if len(picks) == 2 and picks[0] != picks[1]:
                return tuple(picks)
        raise RuntimeError(
            f"start_perturbation: found no two donor states within {self.window.days} days of "
            f"{start} at hour {start.hour} in years {self.years[0]}-{self.years[-1]}."
        )

    def _read(self, year: int, index: int, var_key: str) -> torch.Tensor:
        """One donor field, shaped like a batch row: (levels, 1, H, W) or (1, 1, H, W)."""
        name = var_key.split("/")[-1]
        values = torch.as_tensor(self._open(year)[name].isel({self.time_coord: index}).values, dtype=torch.float32)
        return values.unsqueeze(1) if values.dim() == 3 else values[None, None]

    def _amplitude(self, var_key: str) -> float:
        return self.amplitude[var_key] if isinstance(self.amplitude, dict) else self.amplitude

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)
        data = batch[self.data_type]
        metadata = batch["metadata"]
        times = metadata.get(_src(self.fields[0]), next(iter(metadata.values())))["input_datetime"]
        times = [int(t) for t in torch.as_tensor(times).reshape(-1).tolist()]
        if self._calendar is None:
            self._calendar = self._calendar_of(self._open(self.years[0]))

        n_rows = data[_src(self.fields[0])][self.fields[0]].shape[0]
        if len(times) != n_rows:
            times = times[:1] * n_rows  # a batch-axis ensemble repeats one start
        deltas = {var: [] for var in self.fields}
        for row, encoded in enumerate(times):
            start = decode_time(encoded, self._calendar)
            base = self.seed if self.seed is not None else int(torch.randint(0, 2**31 - 1, ()).item())
            rng = np.random.default_rng([base, (encoded // 10**9) & 0xFFFFFFFFFFFF, row])
            (y1, i1), (y2, i2) = self._donors(rng, start)
            for var in self.fields:
                diff = self._read(y1, i1, var) - self._read(y2, i2, var)
                deltas[var].append(self._amplitude(var) * diff / math.sqrt(2.0))

        for var in self.fields:
            values = data[_src(var)][var]
            delta = torch.stack(deltas[var]).to(values.device, values.dtype)
            if delta.shape[1] != values.shape[1] or delta.shape[-2:] != values.shape[-2:]:
                raise ValueError(
                    f"start_perturbation: {var} donor field {tuple(delta.shape[1:])} does not match the batch "
                    f"{tuple(values.shape[1:])}; is the block before any level selection or regridding?"
                )
            perturbed = values + delta  # broadcasts over the time (history) dimension
            if var in self.nonnegative:
                perturbed = perturbed.clamp(min=0.0)
            data[_src(var)][var] = perturbed
        return batch
