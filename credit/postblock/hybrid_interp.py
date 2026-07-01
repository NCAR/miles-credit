"""
HybridLevelInterp block
-----------------------
Interpolates 3D variables from one set of hybrid sigma-pressure levels to
another, following ``credit.interp.interp_hybrid_to_hybrid_levels`` /
``credit.gefs.interpolate_vertical_levels``: source and destination pressure
columns are both built from the same surface pressure field
(``p = a + b * SP``), and each variable is interpolated linearly in
log(pressure), column by column, parallelized with ``torch.vmap``. Values
outside the source pressure range use constant extrapolation (``np.interp``
behavior).

The primary use case is inference with a model trained on one vertical grid
(e.g. ERA5 hybrid levels) driven by initial conditions from another (e.g. GFS):
run this as a preblock (``credit.preblock.hybrid_interp.HybridLevelInterp``)
to move the IC onto the model's levels, or as a postblock to move model output
onto another grid. This module holds the shared engine and the postblock; the
preblock is a thin wrapper around the same engine.
"""

import torch
import torch.nn as nn

from credit.postblock._interp_utils import load_hybrid_level_coefficients, loglinear_interp_columns
from credit.postblock.base import BasePostblock

# floor applied to hybrid pressures before taking log, matching the model-top
# fallback in credit.postblock.geopotential.pressure_on_interfaces
_MIN_PRESSURE_PA = 0.57


def interp_column_hybrid_to_hybrid(
    fields: torch.Tensor,
    surface_pressure: torch.Tensor,
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    dest_a: torch.Tensor,
    dest_b: torch.Tensor,
) -> torch.Tensor:
    """Interpolate one column between two hybrid sigma-pressure level sets.

    Both pressure columns are built from the same surface pressure
    (``p = a + b * SP``); interpolation is linear in log(pressure) with
    constant extrapolation outside the source range. Source levels must be
    ordered top-of-atmosphere to surface (pressure increasing with index);
    destination levels may be in any order. Designed to be wrapped in
    ``torch.vmap`` over columns: ``fields`` and ``surface_pressure`` carry a
    batch dimension in the vmapped call, the coefficients do not.

    Args:
        fields: variables to interpolate, shape (n_fields, n_source_levels).
        surface_pressure: surface pressure in Pa, scalar tensor.
        source_a: source `a` coefficient at level midpoints in Pa, shape (n_source_levels,).
        source_b: source `b` coefficient at level midpoints (unitless), shape (n_source_levels,).
        dest_a: destination `a` coefficient at level midpoints in Pa, shape (n_dest_levels,).
        dest_b: destination `b` coefficient at level midpoints (unitless), shape (n_dest_levels,).

    Returns:
        Interpolated fields, shape (n_fields, n_dest_levels).
    """
    log_p_source = torch.log((source_a + source_b * surface_pressure).clamp(min=_MIN_PRESSURE_PA))
    log_p_dest = torch.log((dest_a + dest_b * surface_pressure).clamp(min=_MIN_PRESSURE_PA))
    return loglinear_interp_columns(fields, log_p_source, log_p_dest)


class _HybridLevelInterpEngine(nn.Module):
    """Shared engine for the hybrid-to-hybrid pre- and postblocks.

    Holds the source/destination coefficients and applies the vmapped column
    interpolation to a nested ``{source: {var_key: tensor}}`` dict, replacing
    each configured variable with its interpolated counterpart in place.
    """

    def __init__(
        self,
        variables: list[str],
        surface_pressure_var: str,
        source_level_info_file: str,
        dest_level_info_file: str = "ERA5_Lev_Info.nc",
        source_a_var: str = "a_half",
        source_b_var: str = "b_half",
        source_on_interfaces: bool = True,
        source_levels: list[int] | None = None,
        dest_a_var: str = "a_half",
        dest_b_var: str = "b_half",
        dest_on_interfaces: bool = True,
        dest_levels: list[int] | None = None,
        chunk_size: int = 1000,
    ):
        super().__init__()
        self.variables = list(variables)
        self.surface_pressure_var = surface_pressure_var
        self.chunk_size = chunk_size
        self.source_a, self.source_b = load_hybrid_level_coefficients(
            source_level_info_file, source_a_var, source_b_var, source_on_interfaces, source_levels
        )
        self.dest_a, self.dest_b = load_hybrid_level_coefficients(
            dest_level_info_file, dest_a_var, dest_b_var, dest_on_interfaces, dest_levels
        )
        # Interpolation needs source pressure increasing with level index
        # (top→surface); detect orientation once from the coefficients.
        ref_pressure = self.source_a + self.source_b * 101325.0
        self._flip_source = bool(ref_pressure[0] > ref_pressure[-1])
        if self._flip_source:
            self.source_a = torch.flip(self.source_a, (0,))
            self.source_b = torch.flip(self.source_b, (0,))

    def interp_nested(self, nested: dict) -> None:
        """Replace each configured variable in ``nested`` with its interpolated version.

        Args:
            nested: ``{source: {var_key: tensor}}`` dict. 3D variables have
                shape ``(B, n_source_levels, n_time, H, W)`` and are replaced
                by ``(B, n_dest_levels, n_time, H, W)`` tensors; variables
                absent from the dict are skipped silently.
        """
        present = [v for v in self.variables if v in nested.get(v.split("/")[0], {})]
        if not present:
            return
        surface_pressure = nested[self.surface_pressure_var.split("/")[0]][self.surface_pressure_var]
        device = surface_pressure.device
        dtype = surface_pressure.dtype
        batch, _, n_time, height, width = surface_pressure.shape

        tensors = [nested[v.split("/")[0]][v] for v in present]
        n_source_levels = self.source_a.shape[0]
        for var_key, tensor in zip(present, tensors):
            if tensor.shape[1] != n_source_levels:
                raise ValueError(
                    f"HybridLevelInterp: {var_key!r} has {tensor.shape[1]} levels but the source "
                    f"coefficients define {n_source_levels} midpoint levels."
                )
        if self._flip_source:
            tensors = [torch.flip(t, (1,)) for t in tensors]

        def flatten_columns(t: torch.Tensor) -> torch.Tensor:
            per = torch.permute(t.to(device), (0, 2, 3, 4, 1))  # (B, n_time, H, W, n_levels)
            return per.reshape(-1, per.shape[-1])

        fields_flat = torch.stack([flatten_columns(t) for t in tensors], dim=1)  # (N, n_fields, n_src)
        sp_flat = flatten_columns(surface_pressure).squeeze(-1)  # (N,)

        vinterp = torch.vmap(
            interp_column_hybrid_to_hybrid,
            in_dims=(0, 0, None, None, None, None),
            chunk_size=self.chunk_size,
        )
        interped = vinterp(
            fields_flat,
            sp_flat,
            self.source_a.to(device=device, dtype=dtype),
            self.source_b.to(device=device, dtype=dtype),
            self.dest_a.to(device=device, dtype=dtype),
            self.dest_b.to(device=device, dtype=dtype),
        )  # (N, n_fields, n_dest)

        n_dest = self.dest_a.shape[0]
        interped = interped.reshape(batch, n_time, height, width, -1, n_dest)
        for k, var_key in enumerate(present):
            # (B, n_time, H, W, n_dest) → (B, n_dest, n_time, H, W)
            nested[var_key.split("/")[0]][var_key] = torch.permute(interped[..., k, :], (0, 4, 1, 2, 3))


class HybridLevelInterp(BasePostblock):
    """Postblock that interpolates 3D variables between hybrid sigma-pressure level sets.

    Follows the same batch-dict protocol as ``Reconstruct`` and
    ``BridgeScalerTransform``: it operates on the nested output dict at
    ``batch_dict[key]`` (default ``"y_processed"``), which has the form
    ``{source: {var_key: tensor}}``, replacing each variable in ``variables``
    with its interpolated ``(B, n_dest_levels, n_time, H, W)`` counterpart in
    place. A preblock with the same engine and arguments is available as
    ``credit.preblock.hybrid_interp.HybridLevelInterp`` for regridding initial
    conditions (e.g. GFS → ERA5 levels) before the model runs.

    Coefficient files: bare filenames resolve to ``credit.metadata``; paths
    with directories (env vars allowed) are used as-is. Coefficients on level
    interfaces (``*_on_interfaces: true``, e.g. ERA5 ``a_half``/``b_half`` or
    CAM ``hyai``/``hybi``) are averaged to midpoints as in
    ``credit.interp.create_pressure_grid``; set false for midpoint coefficients
    (e.g. ``a_model``/``b_model``). If the `a` and `b` variable names are equal
    and the variable is 2D (the GFS ``vcoord`` convention), row 0 is `a` and
    row 1 is `b`.

    Config example (GFS-level output → ERA5 levels)::

        type: "hybrid_level_interp"
        args:
            variables:
                - "GFS/prognostic/3d/temperature"
                - "GFS/prognostic/3d/specific_humidity"
            surface_pressure_var: "GFS/prognostic/2d/surface_pressure"
            source_level_info_file: "/path/to/gfs_ctrl.nc"
            source_a_var: "vcoord"
            source_b_var: "vcoord"
            dest_level_info_file: "ERA5_Lev_Info.nc"

    Args:
        variables: 3D var_keys to interpolate; variables absent from the batch
            are skipped silently.
        surface_pressure_var: variable name for surface pressure (Pa).
        source_level_info_file: file with the levels the data is currently on.
        dest_level_info_file: file with the levels to interpolate to.
        source_a_var: name of the source `a` coefficient variable (Pa).
        source_b_var: name of the source `b` coefficient variable (unitless).
        source_on_interfaces: whether source coefficients are on level interfaces.
        source_levels: optional subset of 1-based source midpoint level numbers.
        dest_a_var: name of the destination `a` coefficient variable (Pa).
        dest_b_var: name of the destination `b` coefficient variable (unitless).
        dest_on_interfaces: whether destination coefficients are on level interfaces.
        dest_levels: optional subset of 1-based destination midpoint level numbers.
        chunk_size: vmap chunk size to bound memory usage.
        key: entry in ``batch_dict`` holding the nested output dict written
            by ``Reconstruct`` (default: ``"y_processed"``).
    """

    def __init__(self, key: str = "y_processed", **engine_kwargs):
        super().__init__()
        self.key = key
        self.engine = _HybridLevelInterpEngine(**engine_kwargs)

    def forward(self, batch_dict: dict) -> dict:
        if self.key not in batch_dict:
            raise ValueError(f"Key {self.key!r} not found in batch_dict.")
        self.engine.interp_nested(batch_dict[self.key])
        return batch_dict
