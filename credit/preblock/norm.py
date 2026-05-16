"""
norm.py
-------
ERA5Normalizer: normalizes per-variable ERA5 tensors using mean/std NC files.

Operates on the raw batch structure from MultiSourceDataset::

    batch["era5"]["input"]["era5/field_type/3d/varname"]  = (B, n_levels, T, H, W)
    batch["era5"]["input"]["era5/field_type/2d/varname"]  = (B, 1, T, H, W)

Variables absent from the mean/std file are passed through unchanged.

Registered in the preblock registry as ``"era5_normalizer"`` so it can be
included via the config's ``preblocks:`` section::

    preblocks:
      norm:
        type: era5_normalizer
        args:
          mean_path: /path/to/mean.nc
          std_path:  /path/to/std.nc
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

logger = logging.getLogger(__name__)


class ERA5Normalizer(nn.Module):
    """Normalizes per-variable ERA5 tensors using pre-computed mean/std files.

    Normalization: ``(x - mean) / std`` applied per variable. Variables not
    found in the statistics file are passed through unchanged.

    Args:
        mean_path: Path to NetCDF file containing per-variable means.
        std_path:  Path to NetCDF file containing per-variable standard deviations.
        levels:          Optional list of 1-indexed model levels to select from
                         the stats (e.g. [60, 90, 120, 137]).  Mutually exclusive
                         with ``pressure_levels``.
        pressure_levels: Optional list of pressure values in hPa to select from
                         the stats file's ``level`` coordinate (e.g. [500, 850,
                         1000]).  Takes priority over ``levels``.
    """

    def __init__(
        self,
        mean_path: str,
        std_path: str,
        levels: list[int] | None = None,
        pressure_levels: list[int] | None = None,
    ) -> None:
        super().__init__()

        ds_mean = xr.open_dataset(mean_path)
        ds_std = xr.open_dataset(std_path)

        # Determine which rows of the level dimension to keep.
        if pressure_levels is not None:
            # Select by matching hPa values against the file's level coordinate.
            if "level" not in ds_mean.coords:
                raise ValueError("pressure_levels requires a 'level' coordinate in the stats file")
            file_levels = np.array(ds_mean["level"].values)
            level_idx = []
            for p in pressure_levels:
                matches = np.where(file_levels == p)[0]
                if len(matches) == 0:
                    raise ValueError(
                        f"pressure level {p} hPa not found in stats file; available: {file_levels.tolist()}"
                    )
                level_idx.append(int(matches[0]))
        elif levels is not None:
            # Legacy: 1-indexed model-level positions → 0-indexed array indices.
            level_idx = [lv - 1 for lv in levels]
        else:
            level_idx = None

        # Build {varname: tensor} lookup. Tensors are scalar or 1-D (levels).
        # Use only variables present in both files; extras in either are skipped.
        self._mean: dict[str, torch.Tensor] = {}
        self._std: dict[str, torch.Tensor] = {}
        for var in set(ds_mean.data_vars) & set(ds_std.data_vars):
            m = torch.tensor(np.array(ds_mean[var].values), dtype=torch.float32)
            s = torch.tensor(np.array(ds_std[var].values), dtype=torch.float32)
            if level_idx is not None and m.dim() == 1 and m.shape[0] > 1:
                m = m[level_idx]
                s = s[level_idx]
            self._mean[var] = m
            self._std[var] = s

        logger.info(
            "ERA5Normalizer: loaded stats for %d variables%s",
            len(self._mean),
            f" (levels={levels})" if levels is not None else "",
        )

    def _normalize_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize *tensor* using the variable name extracted from *key*."""
        # key format: "era5/{field_type}/{dim}/{varname}"
        parts = key.split("/")
        varname = parts[-1] if len(parts) >= 1 else key

        if varname not in self._mean:
            return tensor  # pass through unchanged

        mean = self._mean[varname].to(device=tensor.device, dtype=tensor.dtype)
        std = self._std[varname].to(device=tensor.device, dtype=tensor.dtype)

        # mean/std may be scalar (2D var) or 1-D levels vector.
        # tensor shape: (B, levels_or_1, T, H, W)
        if mean.dim() == 1 and mean.shape[0] > 1:
            # Per-level stats: reshape to (1, levels, 1, 1, 1)
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)
        # else: scalar — broadcasts naturally

        return (tensor - mean) / std.clamp(min=1e-12)

    def forward(self, batch: dict) -> dict:
        """Normalize all input/target tensors in-place (returns same dict)."""
        for data_type in ("input", "target"):
            if data_type not in batch:
                continue
            for field_dict in batch[data_type].values():
                for key in list(field_dict.keys()):
                    field_dict[key] = self._normalize_tensor(key, field_dict[key])
        return batch
