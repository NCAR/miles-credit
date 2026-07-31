import torch
import numpy as np
import xarray as xr
from os.path import expandvars
from credit.preblock.base import BasePreblock


class Regridder(BasePreblock):
    """Regridding preblock using a sparse weight matrix from an ESMF weights file.

    Applies conservative (or bilinear, depending on the weight file) regridding
    to selected variables in ``batch[data_type][source][var_key]``. The sparse
    weight matrix ``W`` (shape ``n_b × n_a``) is assembled once and cached
    per device; subsequent calls on the same device are free of data movement.

    Args:
        weight_file: Path to an ESMF-format NetCDF weights file. Supports
            ``$ENV`` variable expansion.
        variables: Variable keys to regrid (e.g. ``["era5/prognostic/3d/T"]``).
        data_types: Batch splits to process. Defaults to ``["input", "target"]``.
        reshape_to_xy: If ``True`` (default), reshape the flat output back to
            ``(ny, nx)`` using ``dst_grid_dims`` from the weights file, or from
            the unique destination coordinates for unstructured grids.
        flip_axis: Axes to flip on the input before regridding (e.g. ``[-1, -2]``
            to flip both spatial axes). ``None`` skips flipping.

    Config example::

        type: "regrid"
        args:
            weight_file: "$SCRATCH/weights/era5_to_1deg.nc"
            variables:
                - "era5/prognostic/3d/T"
            reshape_to_xy: true
    """

    def __init__(
        self,
        weight_file: str,
        variables: list[str],
        data_types: list[str] = None,
        reshape_to_xy: bool = True,
        flip_axis: list[int] = None,
    ):
        super().__init__()
        dst_lat = None
        dst_lon = None
        grid_type = None
        with xr.open_dataset(expandvars(weight_file)) as grid_weights:
            rows = grid_weights["row"].values - 1  # ESMF indices are 1-based
            cols = grid_weights["col"].values - 1  # ESMF indices are 1-based
            weights = grid_weights["S"].values  # "S" is ESMF's variable name for the sparse regrid weights
            n_a = grid_weights.sizes["n_a"]  # number of source grid points
            n_b = grid_weights.sizes["n_b"]  # number of destination grid points
            raw_dst = grid_weights["dst_grid_dims"].values
            if len(raw_dst) == 2:
                # Structured weight file: dst_grid_dims = [nlon, nlat]; reverse to [nlat, nlon]
                dst_shape = raw_dst[::-1]
            elif reshape_to_xy:
                # Regular unstructured: infer 2D shape from unique destination center coords
                n_lat = np.unique(grid_weights["yc_b"].values).size
                n_lon = np.unique(grid_weights["xc_b"].values).size
                dst_shape = np.array([n_lat, n_lon])
            else:
                # Irregular unstructured: output stays flat, no 2D shape needed
                dst_shape = None

            # Real destination-grid coordinates, for GridSchema/output purposes — the
            # writer needs the *actual* post-regrid grid, not just its shape. ESMF
            # always stores centers flat, so a genuinely rectilinear destination grid
            # still reshapes to a 2D array; collapse it back to 1D lat/lon when the
            # rows/columns are constant (separable), else keep it as a true 2D
            # curvilinear grid (e.g. regridding onto HRRR's native Lambert grid).
            if dst_shape is not None and "xc_b" in grid_weights and "yc_b" in grid_weights:
                lat2d = grid_weights["yc_b"].values.reshape(dst_shape)
                lon2d = grid_weights["xc_b"].values.reshape(dst_shape)
                if np.allclose(lat2d, lat2d[:, :1]) and np.allclose(lon2d, lon2d[:1, :]):
                    grid_type = "rectilinear"
                    dst_lat = lat2d[:, 0]
                    dst_lon = lon2d[0, :]
                else:
                    grid_type = "curvilinear"
                    dst_lat = lat2d
                    dst_lon = lon2d

        self.variables = variables
        self.data_types = data_types or ["input", "target"]
        self.reshape_to_xy = reshape_to_xy
        self.flip_axis = flip_axis
        # Real destination-grid coordinates (None when reshape_to_xy=False or the
        # weight file lacks xc_b/yc_b) — consumed by GridSchema.resolve to determine
        # the actual output grid when this preblock is active. Not a torch buffer:
        # plain numpy, only used at grid-resolution time, not in forward().
        self.dst_grid_type = grid_type
        self.dst_lat = dst_lat
        self.dst_lon = dst_lon

        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(f"Invalid data_types {invalid}. Valid options are {self.VALID_DATA_TYPES}.")
        # Register as persistent buffers so they are saved in state_dict and move with the model.
        self.register_buffer("row", torch.from_numpy(rows.astype(np.int64)), persistent=True)
        self.register_buffer("col", torch.from_numpy(cols.astype(np.int64)), persistent=True)
        self.register_buffer("weights", torch.from_numpy(weights.astype(np.float32)), persistent=True)

        self.n_a = int(n_a)
        self.n_b = int(n_b)
        self.dst_shape = dst_shape
        # Lazy sparse weight cache: built on first use and reused while the device stays the same.
        self._W = None
        self._W_device = None

    def _get_W(self, device):
        """Return the sparse weight matrix on *device*, building and caching it on first call."""
        if self._W is not None and self._W_device == device:
            return self._W

        idx = torch.stack([self.row, self.col], dim=0).to(device)
        val = self.weights.to(device=device)

        W = torch.sparse_coo_tensor(
            idx, val, size=(self.n_b, self.n_a), device=device
        ).coalesce()  # coalesce merges duplicate indices, required for efficient sparse mm

        self._W = W
        self._W_device = device
        return W

    def _regrid(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        W = self._get_W(device)
        if self.flip_axis is not None:
            x = torch.flip(x, dims=self.flip_axis)
        lead_shape = x.shape[:-2]  # all dims except the last two spatial dims (lat, lon)
        # Flatten leading dims and transpose: sparse mm expects (n_a, N) input, returns (n_b, N).
        x_flat = x.reshape(-1, self.n_a).T
        y_flat = torch.sparse.mm(W, x_flat).T

        if self.reshape_to_xy and self.dst_shape is not None:
            ny, nx = self.dst_shape
            return y_flat.reshape(*lead_shape, ny, nx)
        else:
            return y_flat

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        for var_key in self.variables:
            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/T"

            for data_type in self.data_types:
                if data_type not in batch:
                    continue  # data type absent in this batch (e.g. no "target" during inference)
                if source not in batch[data_type]:
                    raise KeyError(f"Regridder: source '{source}' not found in batch['{data_type}'].")
                if var_key not in batch[data_type][source]:
                    continue  # variable absent in this data type (e.g. statics only exist in "input")
                batch[data_type][source][var_key] = self._regrid(batch[data_type][source][var_key])

        return batch
