"""
regrid_se_to_latlon.py
----------------------
SEToLatLonPostBlock: regrid model predictions from the ne120 spectral-element
grid (ncol=777602) back to a regular lat-lon grid (e.g. 721x1440) using a
pre-computed ESMF sparse weight matrix.

This is the inverse of ``TripoleToSEPreBlock`` (credit/preblock/latlon_to_se.py).
The cube-sphere wxformer trains and predicts on the SE grid; verification
targets (WeatherBench2, ERA5) live on a regular lat-lon grid.  Running this
postblock once after rollout maps each predicted variable back to lat-lon with
a single fixed sparse mat-mul, so the reported skill does not depend on a
re-derived regrid at eval time.

Use the bilinear reverse weights for eval parity, or the conservative reverse
weights when area-integral budgets matter::

    postblocks:
      reconstruct:
        type: reconstruct
      to_latlon:
        type: se_to_latlon
        args:
          weight_file: ".../se_ne120_to_latlon721x1440.nc"
          keys: ["prediction"]        # optionally also "target"

Weight file format (ESMF, slimmed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NetCDF produced by ESMF_RegridWeightGen + slimming:

    row  (n_s,) int32  1-indexed destination (lat-lon) node, row-major
    col  (n_s,) int32  1-indexed source (SE) column
    S    (n_s,) float32 weight
    dst_grid_dims (2,)  [nlon, nlat]  -> reshaped output is (nlat, nlon)
    attrs: n_src, n_dst
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from credit.postblock.base import BasePostblock

logger = logging.getLogger(__name__)


class SEToLatLonPostBlock(BasePostblock):
    """Regrid SE predictions ``(..., ncol)`` to lat-lon ``(..., nlat, nlon)``.

    Operates on the nested prediction dict produced by ``Reconstruct``
    (``batch_dict[key][source][var_key]``) or, if that key holds a bare tensor,
    on the tensor directly.  Runs once after the full rollout, so
    ``per_step`` is ``False``.

    Parameters
    ----------
    weight_file : str | Path
        ESMF-format NetCDF holding the SE -> lat-lon sparse weights.
    keys : list[str] | None
        Which ``batch_dict`` entries to regrid (default ``["prediction"]``).
        Add ``"target"`` to also map verification targets back to lat-lon.
    """

    per_step: bool = False

    def __init__(
        self,
        weight_file: str | Path,
        keys: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.keys = keys or ["prediction"]

        weight_file = Path(weight_file)
        if not weight_file.exists():
            raise FileNotFoundError(
                f"SE->lat-lon weight file not found: {weight_file}\n"
                "Build it with ESMF_RegridWeightGen (see "
                "mesaclip/preprocessing/build_latlon_se_weights.py)."
            )

        import xarray as xr

        with xr.open_dataset(weight_file) as ds:
            row = ds["row"].values.astype(np.int64) - 1  # 0-indexed
            col = ds["col"].values.astype(np.int64) - 1
            S = ds["S"].values.astype(np.float32)

            if "n_src" in ds.attrs:
                self.n_src = int(ds.attrs["n_src"])
                self.n_dst = int(ds.attrs["n_dst"])
            else:
                self.n_src = int(ds.sizes.get("n_a", int(col.max()) + 1))
                self.n_dst = int(ds.sizes.get("n_b", int(row.max()) + 1))

            # dst_grid_dims is [nlon, nlat] (ESMF/SCRIP order) -> reverse to (nlat, nlon)
            if "dst_grid_dims" in ds:
                nlon, nlat = (int(v) for v in ds["dst_grid_dims"].values)
            else:
                raise KeyError(
                    f"{weight_file.name} has no 'dst_grid_dims'; cannot infer the lat-lon output shape for reshaping."
                )

        self.nlat = nlat
        self.nlon = nlon
        if self.nlat * self.nlon != self.n_dst:
            raise ValueError(
                f"dst_grid_dims {self.nlat}x{self.nlon}={self.nlat * self.nlon} "
                f"does not match n_dst={self.n_dst} in {weight_file.name}"
            )

        logger.info(
            "SEToLatLonPostBlock: loaded %d weights, src=%d -> dst=%dx%d  [%s]",
            len(S),
            self.n_src,
            self.nlat,
            self.nlon,
            weight_file.name,
        )

        indices = torch.from_numpy(np.vstack([row, col])).long()
        self.register_buffer("_coo_row", indices[0])
        self.register_buffer("_coo_col", indices[1])
        self.register_buffer("_coo_val", torch.from_numpy(S))
        self._W: torch.Tensor | None = None
        self._W_device: str | None = None

    # ------------------------------------------------------------------

    def _get_W(self, device: torch.device) -> torch.Tensor:
        dev_str = str(device)
        if self._W is not None and self._W_device == dev_str:
            return self._W
        idx = torch.stack([self._coo_row, self._coo_col], dim=0).to(device)
        val = self._coo_val.to(device)
        W = torch.sparse_coo_tensor(idx, val, size=(self.n_dst, self.n_src), device=device).coalesce().to_sparse_csr()
        self._W = W
        self._W_device = dev_str
        return W

    def _regrid_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., ncol)`` -> ``(..., nlat, nlon)`` via sparse mat-mul."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.shape[-1] != self.n_src:
            raise ValueError(
                f"SEToLatLonPostBlock: trailing dim {x.shape[-1]} != n_src={self.n_src}; is this tensor on the SE grid?"
            )
        W = self._get_W(x.device)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.n_src).T  # (n_src, N)
        y = (W @ x2).T  # (N, n_dst)
        return y.reshape(*lead, self.nlat, self.nlon).contiguous()

    # ------------------------------------------------------------------

    def forward(self, batch_dict: dict) -> dict:
        """Regrid every SE tensor under each configured key to lat-lon.

        Handles two shapes per key:

        * nested dict ``{source: {var_key: tensor}}`` (output of ``Reconstruct``)
        * a bare tensor ``(..., ncol)``
        """
        for key in self.keys:
            if key not in batch_dict:
                continue
            value = batch_dict[key]

            if isinstance(value, dict):
                for source, var_dict in value.items():
                    value[source] = {vk: self._regrid_tensor(v) for vk, v in var_dict.items()}
                batch_dict[key] = value
            else:
                batch_dict[key] = self._regrid_tensor(value)

        return batch_dict

    def extra_repr(self) -> str:
        return f"keys={self.keys}, n_src={self.n_src}, dst=({self.nlat}, {self.nlon}), nnz={self._coo_val.numel()}"
