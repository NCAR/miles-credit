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
          xi_path:   /path/to/xi.nc     # optional; applies residual normalization
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

logger = logging.getLogger(__name__)


class ERA5Normalizer(nn.Module):
    """Normalizes per-variable ERA5 tensors using pre-computed mean/std files.

    Normalization: ``(x - mean) / (xi * std)`` applied per variable, where xi
    is the per-variable residual normalization factor (Watt-Meyer et al. 2023,
    Schreck et al. 2025).  When ``xi_path`` is not provided, xi=1 and the
    formula reduces to the standard ``(x - mean) / std``.

    Variables not found in the statistics file are passed through unchanged.

    Args:
        mean_path:       Path to NetCDF file containing per-variable means.
        std_path:        Path to NetCDF file containing per-variable standard deviations.
        xi_path:         Optional path to NetCDF file containing per-variable xi values
                         (residual normalization factors).  When provided, the effective
                         std used for normalization is ``xi * std``.
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
        xi_path: str | None = None,
        levels: list[int] | None = None,
        pressure_levels: list[int] | None = None,
    ) -> None:
        super().__init__()

        ds_mean = xr.open_dataset(mean_path)
        ds_std = xr.open_dataset(std_path)
        ds_xi = xr.open_dataset(xi_path) if xi_path is not None else None

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
        # Variables whose mean or std contain any NaN are also skipped — applying
        # NaN stats would silently poison the entire tensor with NaN values.
        self._mean: dict[str, torch.Tensor] = {}
        self._std: dict[str, torch.Tensor] = {}
        self._xi: dict[str, torch.Tensor] = {}
        skipped_nan: list[str] = []

        common_vars = set(ds_mean.data_vars) & set(ds_std.data_vars)
        for var in common_vars:
            m = torch.tensor(np.array(ds_mean[var].values), dtype=torch.float32)
            s = torch.tensor(np.array(ds_std[var].values), dtype=torch.float32)
            if level_idx is not None and m.dim() == 1 and m.shape[0] > 1:
                m = m[level_idx]
                s = s[level_idx]
            if torch.isnan(m).any() or torch.isnan(s).any():
                skipped_nan.append(var)
                continue
            self._mean[var] = m
            self._std[var] = s

            # Load xi if provided and variable is present in xi file.
            if ds_xi is not None and var in ds_xi.data_vars:
                x = torch.tensor(np.array(ds_xi[var].values), dtype=torch.float32)
                if level_idx is not None and x.dim() == 1 and x.shape[0] > 1:
                    x = x[level_idx]
                if not torch.isnan(x).any():
                    self._xi[var] = x

        if skipped_nan:
            logger.warning(
                "ERA5Normalizer: skipping %d variables with NaN stats (pass-through): %s",
                len(skipped_nan),
                skipped_nan,
            )
        xi_loaded = len(self._xi)
        logger.info(
            "ERA5Normalizer: loaded stats for %d variables, xi for %d variables%s",
            len(self._mean),
            xi_loaded,
            f" (levels={levels})" if levels is not None else "",
        )
        if xi_path is not None and xi_loaded == 0:
            logger.warning("ERA5Normalizer: xi_path provided but no xi values loaded — check file")

    def _normalize_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize *tensor* using the variable name extracted from *key*."""
        # key format: "era5/{field_type}/{dim}/{varname}"
        parts = key.split("/")
        varname = parts[-1] if len(parts) >= 1 else key

        if varname not in self._mean:
            return tensor  # pass through unchanged

        mean = self._mean[varname].to(device=tensor.device, dtype=tensor.dtype)
        std = self._std[varname].to(device=tensor.device, dtype=tensor.dtype)

        # Apply xi if available: effective_std = xi * std
        if varname in self._xi:
            xi = self._xi[varname].to(device=tensor.device, dtype=tensor.dtype)
        else:
            xi = None

        # mean/std/xi may be scalar (2D var) or 1-D levels vector.
        # tensor shape: (B, levels_or_1, T, H, W)
        if mean.dim() == 1 and mean.shape[0] > 1:
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)
            if xi is not None:
                xi = xi.view(1, -1, 1, 1, 1)
        # else: scalar — broadcasts naturally

        effective_std = (xi * std if xi is not None else std).clamp(min=1e-12)
        return (tensor - mean) / effective_std

    def forward(self, batch: dict) -> dict:
        """Normalize all input/target tensors in-place (returns same dict)."""
        for data_type in ("input", "target"):
            if data_type not in batch:
                continue
            for field_dict in batch[data_type].values():
                for key in list(field_dict.keys()):
                    field_dict[key] = self._normalize_tensor(key, field_dict[key])
        return batch
