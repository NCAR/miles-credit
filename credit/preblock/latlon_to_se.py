"""
latlon_to_se.py
---------------
TripoleToSEPreBlock: regrid any (nlat×nlon) lat-lon field to the
ne120 spectral-element grid (ncol=777602) using a pre-computed ESMF sparse
weight matrix.

Despite the class name, the implementation is grid-agnostic — it works for
any source grid (lat-lon, tripole, Gaussian reduced) given the right weight
file.  For 0.25° lat-lon (721×1440) → ne120 SE use the pre-built weights at:
  /glade/work/schreck/repos/credit-mesaclip/mesaclip/static/latlon721x1440_to_se_ne120.nc

Registered in the preblock registry as ``"tripole_to_se"`` so configs can use::

    preblocks:
      latlon_to_se:
        type: tripole_to_se
        args:
          weight_file: "/glade/work/schreck/repos/credit-mesaclip/mesaclip/static/latlon721x1440_to_se_ne120.nc"
          source_key: "era5"
          out_key:    "era5"
          fill_value: 0.0

Weight file format
~~~~~~~~~~~~~~~~~~
NetCDF produced by mesaclip/preprocessing/build_tripole_weights.py:

    row  (n_s,) int32   1-indexed destination node
    col  (n_s,) int32   1-indexed source cell (row-major)
    S    (n_s,) float32 weight
    attrs: n_src, n_dst
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TripoleToSEPreBlock(nn.Module):
    """
    Regrid a tripole (B, C, nlat, nlon) tensor to SE (B, C, ncol).

    Parameters
    ----------
    weight_file : str | Path
        Path to ESMF-format NetCDF produced by build_tripole_weights.py.
    source_key : str
        Key in the batch dict whose value is the tripole tensor.
    out_key : str
        Key under which the regridded SE tensor is written back.  Defaults
        to ``source_key``.
    fill_value : float
        Value substituted for NaN land/fill cells before the matmul.
    """

    def __init__(
        self,
        weight_file: str | Path,
        source_key: str = "dynamic_forcing",
        out_key: str | None = None,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.source_key = source_key
        self.out_key = out_key if out_key is not None else source_key
        self.fill_value = fill_value

        weight_file = Path(weight_file)
        if not weight_file.exists():
            raise FileNotFoundError(
                f"Tripole→SE weight file not found: {weight_file}\n"
                "Run mesaclip/preprocessing/build_tripole_weights.py to generate it."
            )

        import xarray as xr

        with xr.open_dataset(weight_file) as ds:
            row = ds["row"].values.astype(np.int64) - 1  # 0-indexed
            col = ds["col"].values.astype(np.int64) - 1
            S = ds["S"].values.astype(np.float32)
            # Support both tripole-style (n_src/n_dst attrs) and ESMF-style (n_a/n_b dims)
            if "n_src" in ds.attrs:
                self.n_src = int(ds.attrs["n_src"])
                self.n_dst = int(ds.attrs["n_dst"])
            else:
                self.n_src = int(ds.sizes.get("n_a", int(col.max()) + 1))
                self.n_dst = int(ds.sizes.get("n_b", int(row.max()) + 1))

        logger.info(
            "TripoleToSEPreBlock: loaded %d weights, src=%d, dst=%d  [%s]",
            len(S),
            self.n_src,
            self.n_dst,
            weight_file.name,
        )

        # Build sparse CSR matrix and register indices/values as buffers so
        # .to(device) moves them with the module.
        indices = torch.from_numpy(np.vstack([row, col])).long()
        values = torch.from_numpy(S)

        # Store COO components as buffers; build CSR lazily on first forward
        # call (CSR construction requires knowing the device).
        self.register_buffer("_coo_row", indices[0])
        self.register_buffer("_coo_col", indices[1])
        self.register_buffer("_coo_val", values)
        self._W: torch.Tensor | None = None
        self._W_device: str | None = None

    # ------------------------------------------------------------------

    def _get_W(self, device: torch.device, n_src: int) -> torch.Tensor:
        dev_str = str(device)
        if self._W is not None and self._W_device == dev_str:
            return self._W
        idx = torch.stack([self._coo_row, self._coo_col], dim=0).to(device)
        val = self._coo_val.to(device)
        W = torch.sparse_coo_tensor(idx, val, size=(self.n_dst, n_src), device=device).coalesce().to_sparse_csr()
        self._W = W
        self._W_device = dev_str
        return W

    # ------------------------------------------------------------------

    def _regrid_4d(self, x: torch.Tensor) -> torch.Tensor:
        """Core sparse-matmul regrid: ``(B, C, H, W)`` → ``(B, C, ncol)``."""
        B, C, nlat, nlon = x.shape
        n_src_actual = nlat * nlon
        if n_src_actual < self.n_src:
            raise ValueError(
                f"Input spatial size {nlat}×{nlon}={n_src_actual} < max column index {self.n_src} in weight file"
            )
        W = self._get_W(x.device, n_src_actual)
        x = torch.nan_to_num(x, nan=self.fill_value)
        x2 = x.reshape(B, C, -1).permute(2, 0, 1).reshape(n_src_actual, B * C)
        return (W @ x2).reshape(self.n_dst, B, C).permute(1, 2, 0).contiguous()

    def _regrid_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Regrid a per-variable tensor from lat-lon to SE.

        Accepts either:
          * ``(B, C, H, W)``       → ``(B, C, ncol)``
          * ``(B, C, T, H, W)``    → ``(B, C, T, ncol)``  (gen2 time dimension)

        During autoregressive Gen2 rollout, previous-step prognostic fields
        have already been reconstructed on the SE grid with shape
        ``(B, C, T, 1, ncol)``.  Those tensors are converted to the same
        ``(B, C, T, ncol)`` shape produced by the source-grid regrid while
        current-step forcing/target tensors are still regridded from lat-lon.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            if H * W == self.n_dst:
                return x.reshape(B, C, T, self.n_dst)
            out = self._regrid_4d(x.reshape(B, C * T, H, W))  # (B, C*T, ncol)
            return out.reshape(B, C, T, self.n_dst)
        if x.dim() == 4 and x.shape[-2] * x.shape[-1] == self.n_dst:
            return x
        return self._regrid_4d(x)  # 4-D: (B, C, H, W) → (B, C, ncol)

    def forward(self, batch: dict) -> dict:
        """Regrid lat-lon tensors to the SE unstructured grid.

        Handles two batch formats:

        **Gen2 nested format** (``batch["input"]`` is a dict of source dicts):
            Iterates over all variable tensors under ``batch["input"]`` and
            ``batch["target"]``, regridding each in-place.  The
            ``source_key`` / ``out_key`` attributes are ignored in this mode.

        **Legacy flat format** (``batch[source_key]`` is a 4-D tensor):
            Regrids ``batch[source_key]`` and stores the result as
            ``batch[out_key]``.

        Parameters
        ----------
        batch : dict
            Nested batch dict or flat dict with ``source_key`` tensor.

        Returns
        -------
        dict
            Shallow copy of *batch* with SE tensors substituted.
        """
        # -- Gen2 nested format --
        if isinstance(batch.get("input"), dict):
            batch = dict(batch)  # shallow copy at top level
            for data_type in ("input", "target"):
                if data_type not in batch or not isinstance(batch[data_type], dict):
                    continue
                new_dt: dict = {}
                for source_name, source_dict in batch[data_type].items():
                    new_dt[source_name] = {k: self._regrid_tensor(v) for k, v in source_dict.items()}
                batch[data_type] = new_dt
            return batch

        # -- Legacy flat format: batch[source_key] = (B, C, H, W) --
        x = batch[self.source_key]
        out = self._regrid_tensor(x)
        batch = dict(batch)
        batch[self.out_key] = out
        return batch

    def extra_repr(self) -> str:
        return (
            f"source_key={self.source_key!r}, out_key={self.out_key!r}, "
            f"n_src={self.n_src}, n_dst={self.n_dst}, nnz={self._coo_val.numel()}"
        )
