"""Shared vertical-interpolation utilities for pre/postblocks.

Lives in ``credit.postblock`` but is imported by both pre- and postblocks
(the same convention as ``credit.preblock._utils``).
"""

import numpy as np
import torch
import xarray as xr

from credit.metadata import get_meta_file_path


def loglinear_interp_columns(stacked: torch.Tensor, log_x: torch.Tensor, log_xq: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of one column in log-pressure with constant extrapolation.

    Equivalent to ``np.interp`` per row of ``stacked``: targets outside the
    ``log_x`` range take the boundary values. ``log_x`` must be sorted
    ascending; ``log_xq`` may be in any order. vmap-safe (uses gather, no
    data-dependent branching).

    Args:
        stacked: values to interpolate, shape (n_fields, n_x).
        log_x: ascending interpolation coordinate, shape (n_x,).
        log_xq: target coordinates, shape (n_q,).

    Returns:
        Interpolated values, shape (n_fields, n_q).
    """
    n_x = log_x.shape[0]
    # idx_hi is the index of the first x >= target; the weight clamp pins
    # out-of-range targets to the boundary values.
    idx_hi = (log_xq.unsqueeze(-1) >= log_x.unsqueeze(0)).sum(dim=-1).clamp(min=1, max=n_x - 1)
    idx_lo = idx_hi - 1
    x_lo = torch.gather(log_x, 0, idx_lo)
    x_hi = torch.gather(log_x, 0, idx_hi)
    weight = ((log_xq - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    y_lo = torch.gather(stacked, 1, idx_lo.unsqueeze(0).expand(stacked.shape[0], -1))
    y_hi = torch.gather(stacked, 1, idx_hi.unsqueeze(0).expand(stacked.shape[0], -1))
    return y_lo + weight * (y_hi - y_lo)


def load_hybrid_level_coefficients(
    level_info_file: str,
    a_var: str,
    b_var: str,
    on_interfaces: bool = True,
    levels: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load hybrid sigma-pressure coefficients at level midpoints from a netCDF file.

    Args:
        level_info_file: coefficient file; bare filenames resolve to
            ``credit.metadata`` (see ``get_meta_file_path``).
        a_var: name of the `a` (pressure, Pa) coefficient variable. If the
            variable is 2D and ``a_var == b_var`` (e.g. the GFS ``vcoord``
            convention), row 0 is used for `a` and row 1 for `b`.
        b_var: name of the `b` (sigma, unitless) coefficient variable.
        on_interfaces: if True, coefficients are defined on level interfaces
            (half levels) and midpoint values are computed as the mean of
            adjacent interfaces, matching ``credit.interp.create_pressure_grid``.
            If False, coefficients are used directly as midpoint values.
        levels: optional subset of 1-based midpoint level numbers, applied
            after any interface-to-midpoint reduction.

    Returns:
        Tuple of (a, b) float32 tensors at level midpoints.
    """
    with xr.open_dataset(get_meta_file_path(level_info_file)) as level_info:
        a = np.asarray(level_info[a_var].values, dtype=np.float64)
        b = np.asarray(level_info[b_var].values, dtype=np.float64)
    if a.ndim == 2 and a_var == b_var:
        a, b = a[0], b[1]
    if on_interfaces:
        a = 0.5 * (a[:-1] + a[1:])
        b = 0.5 * (b[:-1] + b[1:])
    if levels is not None:
        idx = [lv - 1 for lv in levels]
        a, b = a[idx], b[idx]
    return torch.Tensor(a), torch.Tensor(b)
