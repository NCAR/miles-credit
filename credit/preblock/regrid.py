import torch
import torch.nn as nn
import numpy as np
import xarray as xr


class Regrid(nn.Module):
    """
    Regridding layer using weights file provide by the ESMF library.
    Args:
        weight_file: path to weights file
        reshape_to_xy: whether to reshape the flattened array back to xy coordinates
        flip_axis (list, tuple, or None): whether to flip any axis of the input data. If flipping is desired set a list
                                          of axis to flip (e.g. [-1, -2])
    """

    def __init__(self, weight_file, reshape_to_xy=True, flip_axis=None):

        super().__init__()
        with xr.open_dataset(weight_file) as grid_weights:
            rows = grid_weights["row"].values - 1  # ESMF indices are 1-based
            cols = grid_weights["col"].values - 1  # ESMF indices are 1-based
            weights = grid_weights["S"].values
            n_a = grid_weights.sizes["n_a"]
            n_b = grid_weights.sizes["n_b"]
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

        self.reshape_to_xy = reshape_to_xy
        self.flip_axis = flip_axis
        # store as buffers (CPU tensors)
        self.register_buffer("row", torch.from_numpy(rows.astype(np.int64)), persistent=True)
        self.register_buffer("col", torch.from_numpy(cols.astype(np.int64)), persistent=True)
        self.register_buffer("weights", torch.from_numpy(weights.astype(np.float32)), persistent=True)

        self.n_a = int(n_a)
        self.n_b = int(n_b)
        self.dst_shape = dst_shape
        self._W = None
        self._W_device = None

    def _get_W(self, device):

        if self._W is not None and self._W_device == device:
            return self._W

        idx = torch.stack([self.row, self.col], dim=0).to(device)
        val = self.weights.to(device=device)

        W = torch.sparse_coo_tensor(idx, val, size=(self.n_b, self.n_a), device=device).coalesce()

        self._W = W
        self._W_device = device
        return W

    def forward(self, x):

        device = x.device
        W = self._get_W(device)
        if self.flip_axis is not None:
            x = torch.flip(x, dims=self.flip_axis)
        lead_shape = x.shape[:-2]
        x_flat = x.reshape(-1, self.n_a).T
        y_flat = torch.sparse.mm(W, x_flat).T

        if self.reshape_to_xy and self.dst_shape is not None:
            ny, nx = self.dst_shape
            return y_flat.reshape(*lead_shape, ny, nx)
        else:
            return y_flat
