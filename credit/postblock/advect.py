"""
advect.py
---------
``SemiLagrangianAdvection``: a postblock that performs one explicit
semi-Lagrangian 3D advection step on one or more scalar tracer fields, in place,
on the reconstructed prediction dict (``batch_dict["y_processed"]``).

It is designed to run **after** ``Reconstruct`` (and after any inverse-scaling
block, so that fields are in physical units). For each rollout step it:

1. reads the predicted horizontal winds (``u``, ``v``) and surface pressure from
   ``y_processed``;
2. derives the pressure vertical velocity ``omega`` (Pa s**-1) internally from
   the mass-continuity equation (vertical integral of the horizontal mass
   divergence, with ``omega = 0`` at the model top);
3. traces a back-trajectory of length ``timestep_seconds`` from every grid point
   using an iterative-midpoint scheme; and
4. overwrites each configured tracer with the tracer interpolated at the
   trajectory departure point.

Coordinate / grid assumptions
-----------------------------
* Regular latitude/longitude grid, **periodic in longitude** (global), with hard
  boundaries at the poles and at the model top/bottom.
* Hybrid sigma-pressure model levels; level-interface pressure is
  ``p_half = a_half + b_half * SP`` read from a level-info file (same convention
  and file as :class:`~credit.postblock.geopotential.GeopotentialDiagnostic`).
* Tensors are shaped ``(B, n_levels, n_time, H, W)`` as written by
  ``Reconstruct``; the level axis is ordered top-of-atmosphere -> surface by
  default (``level_order="top_to_surface"``).

Numerics
--------
* Back-trajectory: iterative midpoint (``n_iterations`` fixed-point iterations,
  default 2), evaluated in **grid-index space** so the spherical metric terms
  (``a cos(phi)``, latitude spacing, level pressure thickness) only enter once,
  when the index-space velocity field is built.
* Interpolation: trilinear via :func:`torch.nn.functional.grid_sample`.
  Longitude is wrapped periodically with a one-column circular halo; latitude
  and level are clamped at the boundaries.

Limitations
-----------
* Pole handling is approximate (``cos(phi)`` is floored and trajectories are
  clamped at the polar rows); strong cross-pole flow is not represented.
* The kinematic ``omega`` is not corrected for the surface-pressure tendency.
"""

import logging

import torch
import torch.nn.functional as F
import xarray as xr

from credit.metadata import get_meta_file_path
from credit.physics_constants import RAD_EARTH
from credit.postblock.base import BasePostblock

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# tensor-layout helpers
# --------------------------------------------------------------------------- #
def _to_nlhw(field: torch.Tensor) -> torch.Tensor:
    """Fold ``(B, L, T, H, W)`` -> ``(B*T, L, H, W)`` for per-snapshot ops."""
    b, lev, t, h, w = field.shape
    return field.permute(0, 2, 1, 3, 4).reshape(b * t, lev, h, w)


def _from_nlhw(field: torch.Tensor, like_shape: tuple) -> torch.Tensor:
    """Inverse of :func:`_to_nlhw`, restoring ``(B, L, T, H, W)``."""
    b, lev, t, h, w = like_shape
    return field.reshape(b, t, lev, h, w).permute(0, 2, 1, 3, 4)


# --------------------------------------------------------------------------- #
# physics helpers (operate on (N, L, H, W) tensors, level ordered top->surface)
# --------------------------------------------------------------------------- #
def horizontal_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    lat_rad: torch.Tensor,
    dlon_rad: torch.Tensor,
    radius: float,
    coslat_floor: float,
) -> torch.Tensor:
    """Spherical horizontal divergence ``div = 1/(a cos@) (@u/@lon + @(v cos@)/@lat)``.

    Args:
        u: eastward wind (m s**-1), shape ``(N, L, H, W)``.
        v: northward wind (m s**-1), shape ``(N, L, H, W)``.
        lat_rad: latitude of each row in radians, shape ``(H,)``.
        dlon_rad: longitude grid spacing in radians (scalar tensor).
        radius: Earth radius in metres.
        coslat_floor: minimum value applied to ``cos(lat)`` to keep the
            ``1/(a cos@)`` prefactor finite at the poles.

    Returns:
        Horizontal divergence (s**-1), shape ``(N, L, H, W)``.
    """
    coslat = torch.cos(lat_rad).view(1, 1, -1, 1)
    coslat_safe = coslat.clamp(min=coslat_floor)

    # @u/@lon: centred difference, periodic in longitude.
    dudlon = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dlon_rad)

    # @(v cos@)/@lat: coordinate-aware centred difference (handles Gaussian grids),
    # one-sided at the polar rows.
    vcos = v * coslat
    dvcosdlat = torch.gradient(vcos, spacing=(lat_rad,), dim=(-2,), edge_order=1)[0]

    return (dudlon + dvcosdlat) / (radius * coslat_safe)


def omega_from_continuity(
    u: torch.Tensor,
    v: torch.Tensor,
    p_half: torch.Tensor,
    lat_rad: torch.Tensor,
    dlon_rad: torch.Tensor,
    radius: float,
    coslat_floor: float,
) -> torch.Tensor:
    """Kinematic pressure vertical velocity ``omega = dp/dt`` (Pa s**-1).

    Integrates the horizontal mass divergence downward from the model top
    (``omega = 0`` at ``p = 0``)::

        omega(p) = -@ integral_0^p (div . V_h) dp'

    evaluated at level interfaces via a cumulative sum, then averaged to level
    centres.

    Args:
        u, v: horizontal winds (m s**-1), shape ``(N, L, H, W)``, level axis
            ordered top -> surface.
        p_half: interface pressures (Pa), shape ``(N, L+1, H, W)``, increasing
            top -> surface.
        lat_rad, dlon_rad, radius, coslat_floor: see :func:`horizontal_divergence`.

    Returns:
        ``omega`` at level centres (Pa s**-1), shape ``(N, L, H, W)``.
    """
    div = horizontal_divergence(u, v, lat_rad, dlon_rad, radius, coslat_floor)
    dp = p_half[:, 1:] - p_half[:, :-1]  # layer thickness (Pa), > 0

    flux = torch.cumsum(div * dp, dim=1)  # cumulative integral down to each lower interface
    omega_lower = -flux  # omega at the lower interface of each level
    omega_upper = torch.cat([torch.zeros_like(flux[:, :1]), -flux[:, :-1]], dim=1)
    return 0.5 * (omega_upper + omega_lower)


# --------------------------------------------------------------------------- #
# interpolation helpers
# --------------------------------------------------------------------------- #
def _circular_pad_lon(vol: torch.Tensor, pad: int) -> torch.Tensor:
    """Add a circular halo of ``pad`` columns on each side of the longitude axis."""
    return torch.cat([vol[..., -pad:], vol, vol[..., :pad]], dim=-1)


def _sample(
    vol_padded: torch.Tensor,
    col: torch.Tensor,
    row: torch.Tensor,
    lev: torch.Tensor,
    n_lon: int,
    n_lat: int,
    n_lev: int,
    pad: int,
) -> torch.Tensor:
    """Trilinearly sample a longitude-padded volume at fractional index coordinates.

    Longitude is wrapped periodically (modulo ``n_lon`` + halo offset); latitude
    and level are clamped to their valid ranges (boundary values are held).

    Args:
        vol_padded: ``(N, C, n_lev, n_lat, n_lon + 2*pad)`` circular-padded volume.
        col, row, lev: fractional arrival/departure indices, each broadcastable
            to ``(N, n_lev, n_lat, n_lon)`` in the **unpadded** longitude frame.
        n_lon, n_lat, n_lev: unpadded volume sizes.
        pad: longitude halo width used to build ``vol_padded``.

    Returns:
        Sampled values, shape ``(N, C, n_lev, n_lat, n_lon)``.
    """
    col_w = torch.remainder(col, n_lon) + pad
    x = 2.0 * col_w / (n_lon + 2 * pad - 1) - 1.0
    y = 2.0 * row.clamp(0.0, n_lat - 1) / (n_lat - 1) - 1.0
    if n_lev > 1:
        z = 2.0 * lev.clamp(0.0, n_lev - 1) / (n_lev - 1) - 1.0
    else:
        # single level: any in-range z maps to the lone slice under align_corners.
        z = torch.zeros_like(x)

    # grid_sample expects the last axis ordered (x=lon, y=lat, z=lev).
    grid = torch.stack([x, y, z], dim=-1)
    return F.grid_sample(vol_padded, grid, mode="bilinear", padding_mode="border", align_corners=True)


class SemiLagrangianAdvection(BasePostblock):
    """Semi-Lagrangian 3D tracer advection postblock (operates on ``y_processed``).

    Overwrites each tracer in ``batch_dict[self.key]`` with its value advected one
    step of ``timestep_seconds`` by the concurrently predicted winds, with the
    vertical component driven by a continuity-derived ``omega``. All fields are
    read from, and written back to, the nested ``{source: {var_key: tensor}}``
    dict produced by ``Reconstruct``; the ``source`` for each variable is taken
    from the leading segment of its ``var_key``.

    Example config::

        postblocks:
          per_step:
            reconstruct:
              type: reconstruct
            inverse_scale:
              type: bridgescaler_transform
              args: {method: inverse_transform, scaler_path: /path/scaler.json, variables: [...]}
            advect:
              type: semilagrangian_advection
              args:
                tracer_vars:
                  - "ERA5/prognostic/3d/specific_humidity"
                u_var: "ERA5/prognostic/3d/u_component_of_wind"
                v_var: "ERA5/prognostic/3d/v_component_of_wind"
                surface_pressure_var: "ERA5/prognostic/2d/surface_pressure"
                timestep_seconds: 21600.0
                levels: null          # or e.g. [1, 2, ..., 137] to subselect model levels

    Args:
        tracer_vars: list of ``var_key`` strings to advect in place.
        u_var: ``var_key`` of the eastward wind (m s**-1).
        v_var: ``var_key`` of the northward wind (m s**-1).
        surface_pressure_var: ``var_key`` of surface pressure (Pa).
        timestep_seconds: advection step length (s); should equal the rollout dt.
        n_iterations: number of midpoint back-trajectory iterations (>=1).
        omega_var: optional ``var_key`` of a precomputed ``omega`` (Pa s**-1). If
            given, it is read instead of deriving ``omega`` from continuity.
        level_info_file: NetCDF file with hybrid-coefficient variables.
        model_a_half_var: name of the ``a`` (pressure) half-level coefficient.
        model_b_half_var: name of the ``b`` (sigma) half-level coefficient.
        levels: 1-based model levels present in the data, used to slice the
            half-level coefficients (mirrors ``GeopotentialDiagnostic``); ``None``
            uses every half level in the file.
        level_order: ``"top_to_surface"`` (default) or ``"surface_to_top"`` — the
            ordering of the data's level axis. Internally everything is computed
            top -> surface and flipped back on write.
        grid_info_file: NetCDF file providing 1-D ``latitude``/``longitude`` coords
            (defaults to ``level_info_file``). If its lengths do not match the data
            grid, a uniform global grid is constructed instead.
        latitude_var, longitude_var: coordinate names in ``grid_info_file``.
        key: top-level ``batch_dict`` key holding the nested prediction dict.
        coslat_floor: floor applied to ``cos(lat)`` near the poles.
        dp_dlevel_floor: floor (Pa) on the per-level pressure thickness used to
            convert ``omega`` into an index-space vertical velocity.
        lon_halo: circular longitude halo width for periodic interpolation.
    """

    def __init__(
        self,
        tracer_vars: list[str] | None = None,
        u_var: str = "ERA5/prognostic/3d/u_component_of_wind",
        v_var: str = "ERA5/prognostic/3d/v_component_of_wind",
        surface_pressure_var: str = "ERA5/prognostic/2d/surface_pressure",
        timestep_seconds: float = 21600.0,
        n_iterations: int = 2,
        omega_var: str | None = None,
        level_info_file: str = "ERA5_Lev_Info.nc",
        model_a_half_var: str = "a_half",
        model_b_half_var: str = "b_half",
        levels: list[int] | None = None,
        level_order: str = "top_to_surface",
        grid_info_file: str | None = None,
        latitude_var: str = "latitude",
        longitude_var: str = "longitude",
        key: str = "y_processed",
        coslat_floor: float = 1e-4,
        dp_dlevel_floor: float = 1.0,
        lon_halo: int = 1,
    ):
        super().__init__()
        if level_order not in ("top_to_surface", "surface_to_top"):
            raise ValueError(f"level_order must be 'top_to_surface' or 'surface_to_top', got {level_order!r}.")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")

        self.tracer_vars = list(tracer_vars) if tracer_vars else ["ERA5/prognostic/3d/specific_humidity"]
        self.u_var = u_var
        self.v_var = v_var
        self.surface_pressure_var = surface_pressure_var
        self.timestep_seconds = float(timestep_seconds)
        self.n_iterations = int(n_iterations)
        self.omega_var = omega_var
        self.levels = levels
        self.level_order = level_order
        self.latitude_var = latitude_var
        self.longitude_var = longitude_var
        self.key = key
        self.coslat_floor = float(coslat_floor)
        self.dp_dlevel_floor = float(dp_dlevel_floor)
        self.lon_halo = int(lon_halo)

        # Hybrid sigma-pressure half-level coefficients (mirrors GeopotentialDiagnostic).
        with xr.open_dataset(get_meta_file_path(level_info_file)) as level_info:
            a_all = torch.as_tensor(level_info[model_a_half_var].values, dtype=torch.float32)
            b_all = torch.as_tensor(level_info[model_b_half_var].values, dtype=torch.float32)
        if levels is not None:
            half_idx = [lv - 1 for lv in levels] + [levels[-1]]
            a_half, b_half = a_all[half_idx], b_all[half_idx]
        else:
            a_half, b_half = a_all, b_all
        self.register_buffer("a_half", a_half, persistent=False)
        self.register_buffer("b_half", b_half, persistent=False)

        # Latitude/longitude grid (used to build the spherical metric terms).
        with xr.open_dataset(get_meta_file_path(grid_info_file or level_info_file)) as grid_info:
            lat_deg = torch.as_tensor(grid_info[latitude_var].values, dtype=torch.float32)
            lon_deg = torch.as_tensor(grid_info[longitude_var].values, dtype=torch.float32)
        self.register_buffer("lat_deg", lat_deg, persistent=False)
        self.register_buffer("lon_deg", lon_deg, persistent=False)

    # ------------------------------------------------------------------ #
    # nested-dict accessors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get(y_processed: dict, var_key: str) -> torch.Tensor:
        source = var_key.split("/")[0]
        try:
            return y_processed[source][var_key]
        except KeyError as exc:
            raise KeyError(f"SemiLagrangianAdvection: '{var_key}' not found in y_processed[{source!r}].") from exc

    @staticmethod
    def _set(y_processed: dict, var_key: str, value: torch.Tensor) -> None:
        y_processed[var_key.split("/")[0]][var_key] = value

    # ------------------------------------------------------------------ #
    # grid construction
    # ------------------------------------------------------------------ #
    def _grid_coords(self, n_lat: int, n_lon: int, device, dtype):
        """Return ``(lat_rad (H,), dlat_rad_row (H,), dlon_rad scalar)`` for the data grid.

        Uses the latitude/longitude buffers when their lengths match the data
        grid; otherwise constructs a uniform global grid and warns.
        """
        if self.lat_deg.numel() == n_lat and self.lon_deg.numel() == n_lon:
            lat_deg = self.lat_deg.to(device=device, dtype=dtype)
            lon_deg = self.lon_deg.to(device=device, dtype=dtype)
        else:
            logger.warning(
                "SemiLagrangianAdvection: grid-info coords (%d lat, %d lon) do not match data grid "
                "(%d lat, %d lon); falling back to a uniform global lat/lon grid.",
                self.lat_deg.numel(),
                self.lon_deg.numel(),
                n_lat,
                n_lon,
            )
            lat_deg = torch.linspace(90.0, -90.0, n_lat, device=device, dtype=dtype)
            lon_deg = torch.arange(n_lon, device=device, dtype=dtype) * (360.0 / n_lon)

        lat_rad = torch.deg2rad(lat_deg)
        # per-row latitude spacing (signed) handles non-uniform / N->S-ordered grids
        dlat_rad_row = torch.gradient(lat_rad, edge_order=1)[0]
        dlon_rad = torch.deg2rad(lon_deg[1] - lon_deg[0])
        return lat_rad, dlat_rad_row, dlon_rad

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, batch_dict: dict) -> dict:
        y_processed = batch_dict[self.key]

        u5 = self._get(y_processed, self.u_var)  # (B, L, T, H, W)
        v5 = self._get(y_processed, self.v_var)
        sp5 = self._get(y_processed, self.surface_pressure_var)  # (B, 1, T, H, W)
        like_shape = u5.shape
        b, n_lev, t, n_lat, n_lon = like_shape
        device, dtype = u5.device, u5.dtype

        flip = self.level_order == "surface_to_top"

        def _prep(field5: torch.Tensor) -> torch.Tensor:
            """(B, L, T, H, W) -> (B*T, L, H, W), oriented top->surface."""
            f = _to_nlhw(field5)
            return torch.flip(f, dims=(1,)) if flip else f

        u = _prep(u5)
        v = _prep(v5)
        sp = _to_nlhw(sp5).squeeze(1)  # (B*T, H, W); surface field is orientation-independent

        lat_rad, dlat_rad_row, dlon_rad = self._grid_coords(n_lat, n_lon, device, dtype)
        radius = float(RAD_EARTH)

        # ---- interface / centre pressure (Pa), top -> surface ---------------
        a_half = self.a_half.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        b_half = self.b_half.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        p_half = a_half + b_half * sp.unsqueeze(1)  # (N, L+1, H, W)
        if p_half.shape[1] != n_lev + 1:
            raise ValueError(
                f"SemiLagrangianAdvection: built {p_half.shape[1]} interface pressures for {n_lev} levels; "
                f"expected {n_lev + 1}. Set `levels` to the model levels present in the data."
            )
        p_center = 0.5 * (p_half[:, :-1] + p_half[:, 1:])  # (N, L, H, W)

        # ---- omega (Pa/s): read precomputed, or derive from continuity ------
        if n_lev == 1:
            omega = torch.zeros_like(u)  # no vertical advection for a single level
        elif self.omega_var is not None:
            omega = _prep(self._get(y_processed, self.omega_var))
        else:
            omega = omega_from_continuity(u, v, p_half, lat_rad, dlon_rad, radius, self.coslat_floor)

        # ---- index-space velocities (grid indices per second) ---------------
        coslat_safe = torch.cos(lat_rad).clamp(min=self.coslat_floor).view(1, 1, -1, 1)
        vel_col = u / (radius * coslat_safe) / dlon_rad  # columns / s
        vel_row = v / radius / dlat_rad_row.view(1, 1, -1, 1)  # rows / s

        dp_dlevel = torch.gradient(p_center, dim=1)[0]  # Pa per level index (> 0 top->surface)
        dp_dlevel = dp_dlevel.clamp(min=self.dp_dlevel_floor)
        vel_lev = omega / dp_dlevel  # levels / s

        # ---- back-trajectory in index space (iterative midpoint) ------------
        n = b * t
        dt = self.timestep_seconds
        pad = self.lon_halo
        vel_padded = _circular_pad_lon(torch.stack([vel_col, vel_row, vel_lev], dim=1), pad)  # (N, 3, L, H, W)

        col0 = torch.arange(n_lon, device=device, dtype=dtype).view(1, 1, 1, n_lon).expand(n, n_lev, n_lat, n_lon)
        row0 = torch.arange(n_lat, device=device, dtype=dtype).view(1, 1, n_lat, 1).expand(n, n_lev, n_lat, n_lon)
        lev0 = torch.arange(n_lev, device=device, dtype=dtype).view(1, n_lev, 1, 1).expand(n, n_lev, n_lat, n_lon)

        disp_col = torch.zeros_like(col0)
        disp_row = torch.zeros_like(row0)
        disp_lev = torch.zeros_like(lev0)
        for _ in range(self.n_iterations):
            mid_vel = _sample(
                vel_padded,
                col0 - 0.5 * disp_col,
                row0 - 0.5 * disp_row,
                lev0 - 0.5 * disp_lev,
                n_lon,
                n_lat,
                n_lev,
                pad,
            )  # (N, 3, L, H, W)
            disp_col = dt * mid_vel[:, 0]
            disp_row = dt * mid_vel[:, 1]
            disp_lev = dt * mid_vel[:, 2]

        dep_col = col0 - disp_col
        dep_row = row0 - disp_row
        dep_lev = lev0 - disp_lev

        # ---- advect each tracer by sampling at the departure point ----------
        for tracer_var in self.tracer_vars:
            tracer5 = self._get(y_processed, tracer_var)
            tracer = _prep(tracer5)  # (N, L, H, W), top->surface
            tracer_padded = _circular_pad_lon(tracer.unsqueeze(1), pad)  # (N, 1, L, H, W)
            advected = _sample(tracer_padded, dep_col, dep_row, dep_lev, n_lon, n_lat, n_lev, pad).squeeze(1)

            if flip:
                advected = torch.flip(advected, dims=(1,))
            self._set(y_processed, tracer_var, _from_nlhw(advected, like_shape))

        batch_dict[self.key] = y_processed
        return batch_dict
