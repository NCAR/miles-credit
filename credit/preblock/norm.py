"""
norm.py
-------
ERA5Normalizer: normalizes per-variable ERA5 tensors (before ConcatPreblock)
using mean/std NC files (scaler_type: std_new).

Operates on the raw batch structure from MultiSourceDataset:
    batch["era5"]["input"]["era5/field_type/3d/varname"] = (B, n_levels, T, H, W)
    batch["era5"]["input"]["era5/field_type/2d/varname"] = (B, 1, T, H, W)

Variables absent from the mean/std file (e.g. static masks, land_sea_CI_mask)
are passed through unchanged.
"""

import torch
import torch.nn as nn
import xarray as xr


class ERA5Normalizer(nn.Module):
    """Normalize ERA5 per-variable tensors using precomputed mean and std NC files.

    Must be applied *before* ConcatPreblock so variable names are still available
    as dict keys.

    Args:
        conf: Full configuration dict. Reads ``data.mean_path``, ``data.std_path``,
              ``data.source.ERA5.level_coord``, and ``data.source.ERA5.levels``.
    """

    def __init__(self, conf: dict) -> None:
        super().__init__()
        data_conf = conf["data"]
        src_cfg = data_conf["source"]["ERA5"]

        self.level_coord: str = src_cfg["level_coord"]
        self.levels = src_cfg["levels"]

        self.mean_ds = xr.open_dataset(data_conf["mean_path"]).load()
        self.std_ds = xr.open_dataset(data_conf["std_path"]).load()

    def _stats(
        self, varname: str, dim: str, device: torch.device, dtype: torch.dtype
    ):
        """Return (mean, std) broadcast-ready tensors, or (None, None) if variable not in file."""
        if varname not in self.mean_ds:
            return None, None

        if dim == "3d":
            # mean/std shape: (n_levels,) → broadcast to (1, n_levels, 1, 1, 1)
            m = self.mean_ds[varname].sel({self.level_coord: self.levels}).values
            s = self.std_ds[varname].sel({self.level_coord: self.levels}).values
            mean = torch.tensor(m, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
            std = torch.tensor(s, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        else:
            # 2D variable — scalar mean/std
            mean = torch.tensor(float(self.mean_ds[varname].values), dtype=dtype, device=device)
            std = torch.tensor(float(self.std_ds[varname].values), dtype=dtype, device=device)

        return mean, std

    def forward(self, batch: dict) -> dict:
        for source_data in batch.values():
            if not isinstance(source_data, dict):
                continue
            for split in ("input", "target"):
                if split not in source_data:
                    continue
                for key, tensor in source_data[split].items():
                    # key format: "{source}/{field_type}/{3d|2d}/{varname}"
                    parts = key.split("/")
                    if len(parts) < 4:
                        continue
                    dim, varname = parts[2], parts[3]
                    mean, std = self._stats(varname, dim, tensor.device, tensor.dtype)
                    if mean is not None:
                        source_data[split][key] = (tensor - mean) / std
        return batch
