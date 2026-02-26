import xarray as xr
import numpy as np
import torch
import torch.nn as nn


class Scaler(nn.Module):
    """
    Scaling layer using a bridgescaler object. Supports transform and its inverse.
    """
    def __init__(self, scaler, inverse=False):
        super().__init__()
        self.scaler = scaler
        self.inverse = inverse

    def forward(self, x):
        if self.inverse:
            return self.scaler.inverse_transform(x)
        else:
            return self.scaler.transform(x)


class Regrid(nn.Module):
    """
    Regridding layer using weights file provide by the ESMF library.
    """
    def __init__(self, weight_file):

        super().__init__()
        with xr.open_dataset(weight_file) as grid_weights:
            rows = grid_weights['row'].values - 1  # ESMF indices are 1-based
            cols = grid_weights['col'].values - 1  # ESMF indices are 1-based
            weights = grid_weights['S'].values
            n_a = grid_weights.sizes["n_a"]
            n_b = grid_weights.sizes["n_b"]
            dst_shape = grid_weights['dst_grid_dims'].values[::-1] ## should probably chek to see if this is necessary

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
        val = self.val.to(device=device)

        W = torch.sparse_coo_tensor(idx, val, size=(self.n_b, self.n_a), device=device).coalesce()

        self._W = W
        self._W_device = device
        return W

    def forward(self, x):

        device = x.device
        W = self._get_W(device)
        x = torch.flip(x, dims=[-2])
        lead_shape = x.shape[:-2]
        x_flat = x.reshape(-1, self.n_a).T
        y_flat = torch.sparse.mm(W, x_flat).T

        ny, nx = self.dst_shape

        return y_flat.reshape(*lead_shape, ny, nx)