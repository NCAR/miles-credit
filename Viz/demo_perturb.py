import sys
import time
import asyncio
import numpy as np
import pandas as pd
import xarray as xr
import pyvista as pv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkRenderer as VtkRenderer

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as vtk_widgets

from shared_utils import (
    parse_level,
    has_level,
    get_2d_field,
    make_star_mesh,
    make_cartopy_coastline_texture,
    make_coast_plane,
    make_contours,
    MapPanel,
)

from credit.applications import rollout_gen2

# ============================================================
# User settings
# ============================================================
FILE = "/Users/cbecker/PycharmProjects/CREDIT/era5_local_testing_data_onedeg_2021.nc"
# FILE = "/glade/work/cbecker/era5_local_testing_data_onedeg_2021.nc"

# ============================================================
# Load data
# ============================================================
ds = xr.open_dataset(FILE)
variable_items = list(ds.data_vars)
lon = ds["longitude"].values.astype(np.float32)
lat = ds["latitude"].values.astype(np.float32)
times = ds["time"].values
nt = ds.sizes["time"]
time_strings = pd.to_datetime(times).strftime("%Y-%m-%d %H:%M").tolist()
level_items = ["default"] + [str(int(v)) for v in ds["level"].values]

DEFAULT_VIEW_VAR = "t2m"
DEFAULT_PERTURB_VAR = "t2m"


# ============================================================
# Global range cache
# ============================================================
@lru_cache(maxsize=32)
def get_global_range(var, level):
    da = ds[var]
    if "level" in da.dims:
        da = da.isel(level=-1) if level is None else da.sel(level=level)
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


@lru_cache(maxsize=128)
def get_cached_contour(ds_key, var, time_idx, level, interval, stride=1):
    """ds_key distinguishes A vs B datasets for cache correctness."""
    _ds = _ds_store[ds_key]
    return make_contours(
        ds=_ds, var=var, time_idx=time_idx, level=level, interval=interval, stride=stride, lon=lon, lat=lat
    )


# Storage for the two forecast datasets (populated after model run)
_ds_store = {"A": ds, "B": ds}  # demo: both start as the same file


# ============================================================
# Perturbation helpers
# ============================================================
def add_wave_perturbation(
    field,
    lat,
    lon,
    amp,
    lat0=40,
    lon0=250,
    sigma_lat=8,
    sigma_lon=15,
    zonal_wavenumber=6,
    vertical_tilt=False,
):
    lon2d, lat2d = np.meshgrid(lon, lat)

    envelope = np.exp(-((lat2d - lat0) ** 2) / (2 * sigma_lat**2) - ((lon2d - lon0) ** 2) / (2 * sigma_lon**2))

    phase = zonal_wavenumber * np.deg2rad(lon2d - lon0)

    for k in range(field.shape[0]):
        # Negative tilt = westward phase shift with height (growing baroclinic wave)
        tilt = -0.5 * (k / max(field.shape[0] - 1, 1)) * np.pi if vertical_tilt else 0.0
        z_weight = np.sin(np.pi * (k + 1) / (field.shape[0] + 1))
        field[k] += amp * z_weight * envelope * np.cos(phase + tilt)

    return field


def build_perturbed_ds(
    base_ds, perturb_var, lon_center, lat_center, amplitude, sigma_lat, sigma_lon, wavenumber, vertical_tilt
):
    """
    Return a new Dataset with a baroclinic wave perturbation applied to perturb_var at t=0.
    Only the target variable is copied; all others are shared references.
    Requires a 3D variable (level dimension). In production, replace with CREDIT inference.
    """
    da_orig = base_ds[perturb_var]
    if "level" not in da_orig.dims:
        raise ValueError(f"{perturb_var} has no level dimension; wave perturbation requires a 3D variable")

    arr = da_orig.values.copy()  # (time, level, lat, lon)
    arr[0] = add_wave_perturbation(
        arr[0].copy(),
        lat,
        lon,
        amp=amplitude,
        lat0=lat_center,
        lon0=lon_center,
        sigma_lat=sigma_lat,
        sigma_lon=sigma_lon,
        zonal_wavenumber=wavenumber,
        vertical_tilt=vertical_tilt,
    )

    da_perturbed = xr.DataArray(arr, dims=da_orig.dims, coords=da_orig.coords)
    return base_ds.assign({perturb_var: da_perturbed})


# ============================================================
# Diagnostic computations
# ============================================================
DIAGNOSTIC_ITEMS = ["Absolute Difference |A-B|", "Normalized Anomaly (A-B)/σ_A"]
CMAP_ITEMS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "turbo",
    "RdYlBu_r",
    "BrBG",
    "PuOr_r",
    "rainbow",
    "Blues_r",
    "Reds",
]

_temporal_std_cache = {}


def get_temporal_std(var, level):
    key = (var, level)
    if key not in _temporal_std_cache:
        da = ds[var]
        lev = parse_level(level)
        if "level" in da.dims:
            da = da.isel(level=-1) if lev is None else da.sel(level=lev)
        std = da.std(dim="time").values.astype(np.float32)  # (lat, lon)
        std = std.T  # → (lon, lat)
        std = np.where(std < 1e-6, 1e-6, std)
        _temporal_std_cache[key] = std
    return _temporal_std_cache[key]


def compute_diagnostic(arr_a, arr_b, diagnostic, var, level):
    diff = arr_a - arr_b
    if diagnostic == "Absolute Difference |A-B|":
        return np.abs(diff)
    elif diagnostic == "Normalized Anomaly (A-B)/σ_A":
        std = get_temporal_std(var, level)
        return diff / std
    return np.abs(diff)


# ============================================================
# Coastline texture (shared across all panels)
# ============================================================
coast_img, extent = make_cartopy_coastline_texture(lon, lat, resolution="50m", dpi=400, linewidth=0.8)
lon_min, lon_max, lat_min, lat_max = extent
coast_texture = pv.Texture(coast_img)
coast_plane = make_coast_plane(extent, z_value=4.0)

# ============================================================
# PyVista scene: 2x2 layout
# ============================================================
plotter = pv.Plotter(
    off_screen=False,
    window_size=[80, 60],
    border=True,
    border_width=3,
    polygon_smoothing=True,
    line_smoothing=True,
    shape=(2, 2),
)

# Panel layout:
#   [0,0] Forecast A   [0,1] Forecast B
#   [1,0] Diff A-B     [1,1] Diagnostic

_PANEL_KEYS = ["fc_a", "fc_b", "diff", "diag"]
_PANEL_TITLES = ["Forecast A (Control)", "Forecast B (Perturbed)", "Difference A − B", "Diagnostic"]
_PANEL_CMAPS = ["viridis", "viridis", "coolwarm", "viridis"]
_PANEL_RC = [(0, 0), (0, 1), (1, 0), (1, 1)]

panels = {}
for key, title, cmap, (r, c) in zip(_PANEL_KEYS, _PANEL_TITLES, _PANEL_CMAPS, _PANEL_RC):
    panels[key] = MapPanel(
        plotter, r, c, lon, lat, coast_texture, coast_plane, title=title, cmap=cmap, show_scalar_bar=False
    )

# Initial field
vmin0, vmax0 = get_global_range(DEFAULT_VIEW_VAR, None)
base0 = get_2d_field(ds, DEFAULT_VIEW_VAR, time_idx=0)

for key in ["fc_a", "fc_b"]:
    panels[key].add_base(base0, clim=(vmin0, vmax0))

# Diff and diag start as zeros
zero = np.zeros_like(base0)
panels["diff"].add_base(zero, clim=(-1.0, 1.0))
panels["diag"].add_base(zero, clim=(0.0, 1.0))

# ============================================================
# Camera: all 4 panels share one camera
# ============================================================
for r, c in _PANEL_RC:
    plotter.subplot(r, c)
    plotter.view_xy()
    plotter.camera.parallel_projection = True

_shared_cam = plotter.renderers[0].GetActiveCamera()
for i in range(1, 4):
    plotter.renderers[i].SetActiveCamera(_shared_cam)

plotter.enable_custom_trackball_style(
    left="pan",
    shift_left="pan",
    control_left="pan",
    middle="pan",
    shift_middle="pan",
    control_middle="pan",
    right="pan",
    shift_right="pan",
    control_right="pan",
)

_ZOOM_FACTOR = 1.15


def _clamp_camera():
    camera = plotter.camera
    w, h = plotter.renderers[0].GetSize()
    aspect = (w / h) if h > 1 else ((lon_max - lon_min) / (lat_max - lat_min))
    dom_hw = 0.5 * (lon_max - lon_min)
    dom_hh = 0.5 * (lat_max - lat_min)
    dom_cx = lon_min + dom_hw
    dom_cy = lat_min + dom_hh
    max_ps = max(dom_hh, dom_hw / aspect)
    ps = min(camera.parallel_scale, max_ps)
    camera.parallel_scale = ps
    view_hw = ps * aspect
    view_hh = ps
    px, py, pz = camera.position
    _fx, _fy, fz = camera.focal_point
    px = dom_cx if view_hw >= dom_hw else max(lon_min + view_hw, min(lon_max - view_hw, px))
    py = dom_cy if view_hh >= dom_hh else max(lat_min + view_hh, min(lat_max - view_hh, py))
    camera.position = (px, py, pz)
    camera.focal_point = (px, py, fz)


def _zoom_at_cursor(direction):
    camera = plotter.camera
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my) or plotter.renderers[0]
    renderer.SetDisplayPoint(mx, my, 0)
    renderer.DisplayToWorld()
    w = renderer.GetWorldPoint()
    wx0, wy0 = w[0] / w[3], w[1] / w[3]
    factor = _ZOOM_FACTOR if direction > 0 else 1.0 / _ZOOM_FACTOR
    camera.parallel_scale /= factor
    renderer.SetDisplayPoint(mx, my, 0)
    renderer.DisplayToWorld()
    w = renderer.GetWorldPoint()
    wx1, wy1 = w[0] / w[3], w[1] / w[3]
    dx, dy = wx0 - wx1, wy0 - wy1
    px, py, pz = camera.position
    fx, fy, fz = camera.focal_point
    camera.position = (px + dx, py + dy, pz)
    camera.focal_point = (fx + dx, fy + dy, fz)
    _clamp_camera()
    ctrl.view_update()


plotter.iren.interactor.AddObserver("MouseWheelForwardEvent", lambda o, e: _zoom_at_cursor(1), 1.0)
plotter.iren.interactor.AddObserver("MouseWheelBackwardEvent", lambda o, e: _zoom_at_cursor(-1), 1.0)
plotter.iren.interactor.AddObserver("EndInteractionEvent", lambda o, e: (_clamp_camera(), ctrl.view_update()))

# ============================================================
# Colorbar renderer strips — one per panel
# ============================================================
CBAR_STRIP = 0.06

# Data viewports shrink vertically to leave room for colorbar strips
_data_vp = [
    (0.0, 0.5 + CBAR_STRIP, 0.5, 1.0),  # [0,0] Forecast A
    (0.5, 0.5 + CBAR_STRIP, 1.0, 1.0),  # [0,1] Forecast B
    (0.0, 0.0 + CBAR_STRIP, 0.5, 0.5),  # [1,0] Diff
    (0.5, 0.0 + CBAR_STRIP, 1.0, 0.5),  # [1,1] Diagnostic
]
_cbar_vp = [
    (0.0, 0.5, 0.5, 0.5 + CBAR_STRIP),
    (0.5, 0.5, 1.0, 0.5 + CBAR_STRIP),
    (0.0, 0.0, 0.5, 0.0 + CBAR_STRIP),
    (0.5, 0.0, 1.0, 0.0 + CBAR_STRIP),
]

for i, vp in enumerate(_data_vp):
    plotter.renderers[i].SetViewport(*vp)

cbar_luts = []
cbar_renderers = []
cbar_actors = []

for cmap_name, cvp in zip(_PANEL_CMAPS, _cbar_vp):
    lut = pv.LookupTable(cmap=cmap_name, n_values=256)
    lut.scalar_range = (vmin0, vmax0)
    cbar_luts.append(lut)

    sb = vtkScalarBarActor()
    sb.UnconstrainedFontSizeOn()
    sb.SetLookupTable(lut)
    sb.SetOrientationToHorizontal()
    sb.SetNumberOfLabels(8)
    sb.SetTitle("")
    sb.SetPosition(0.175, 0.1)
    sb.SetPosition2(0.65, 0.78)
    sb.SetBarRatio(0.4)
    sb.SetLabelFormat("%.1f")
    ltp = sb.GetLabelTextProperty()
    ltp.SetFontSize(15)
    ltp.SetColor(0.0, 0.0, 0.0)
    ltp.SetShadow(False)
    sb.GetFrameProperty().SetColor(0.0, 0.0, 0.0)
    sb.DrawTickLabelsOn()

    r = VtkRenderer()
    r.SetViewport(*cvp)
    r.SetBackground(1.0, 1.0, 1.0)
    r.SetInteractive(False)
    r.AddActor(sb)
    plotter.ren_win.AddRenderer(r)
    cbar_renderers.append(r)
    cbar_actors.append(sb)

# ============================================================
# Perturbation marker
# ============================================================
_perturb_marker_actors = {k: None for k in _PANEL_KEYS}


def _update_perturb_markers():
    for key, (r, c) in zip(_PANEL_KEYS, _PANEL_RC):
        plotter.subplot(r, c)
        if _perturb_marker_actors[key] is not None:
            plotter.remove_actor(_perturb_marker_actors[key])
            _perturb_marker_actors[key] = None

        if state.perturb_placed:
            marker = make_star_mesh(state.perturb_cx, state.perturb_cy, radius=1.5, z=5.0)
            _perturb_marker_actors[key] = plotter.add_mesh(marker, color="red", lighting=False, pickable=False)


def _on_left_click(obj, event):
    if not state.perturb_picking_enabled:
        return
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my)
    if renderer not in plotter.renderers[:4]:
        return
    renderer.SetDisplayPoint(mx, my, 0)
    renderer.DisplayToWorld()
    w = renderer.GetWorldPoint()
    wx, wy = w[0] / w[3], w[1] / w[3]
    state.perturb_cx = float(np.clip(wx, float(lon.min()), float(lon.max())))
    state.perturb_cy = float(np.clip(wy, float(lat.min()), float(lat.max())))
    state.perturb_placed = True
    state.flush()
    _update_perturb_markers()
    ctrl.view_update()


plotter.iren.interactor.AddObserver("LeftButtonPressEvent", _on_left_click, 1.0)

# ============================================================
# Trame server
# ============================================================
server = get_server()
state, ctrl = server.state, server.controller

state.variable_items = variable_items
state.level_items = level_items
state.diagnostic_items = DIAGNOSTIC_ITEMS
state.cmap_items = CMAP_ITEMS

state.view_var = DEFAULT_VIEW_VAR
state.view_level = "default"
state.view_has_level = has_level(ds, DEFAULT_VIEW_VAR)
state.view_cmap = "viridis"
state.t_index = 0
state.time_text = time_strings[0]
state.latency_text = "Ready"

# Perturbation state
state.perturb_var = DEFAULT_PERTURB_VAR
state.perturb_level = "default"
state.perturb_has_level = has_level(ds, DEFAULT_PERTURB_VAR)
state.perturb_amplitude = 2.0
state.perturb_sigma_lat = 8.0
state.perturb_sigma_lon = 15.0
state.perturb_wavenumber = 6
state.perturb_vertical_tilt = True
state.perturb_picking_enabled = False
state.perturb_placed = False
state.perturb_applied = False
state.perturb_cx = float(lon.mean())
state.perturb_cy = float(lat.mean())

# Contour state (two slots, shared across all panels)
state.c0_enabled = True
state.c0_var = "Z500" if "Z500" in variable_items else variable_items[0]
state.c0_level = "default"
state.c0_interval = 600.0
state.c0_color = "black"
state.c0_line_width = 1.2

state.c1_enabled = False
state.c1_var = variable_items[0]
state.c1_level = "default"
state.c1_interval = 5.0
state.c1_color = "white"
state.c1_line_width = 1.0

# Forecast / diagnostic state
state.diagnostic = DIAGNOSTIC_ITEMS[0]
state.forecast_status = "Ready — place perturbation and click Apply"
state.forecasts_ready = False

# Colorbar range state (user-editable)
state.base_vmin = vmin0
state.base_vmax = vmax0

# ============================================================
# Forecast execution (background thread)
# ============================================================
_executor = ThreadPoolExecutor(max_workers=1)
_event_loop = None  # captured on the asyncio thread in run_forecasts()


def _on_forecast_complete(status, ready):
    """Runs on the asyncio event loop thread via call_soon_threadsafe."""
    state.forecast_status = status
    state.forecasts_ready = ready
    state.flush()  # safe: we're on the asyncio thread, not a background thread


@ctrl.add("apply_perturbation")
def apply_perturbation():
    if not state.perturb_placed or not state.perturb_has_level:
        return
    try:
        state.forecast_status = "Applying perturbation to IC..."
        ds_b = build_perturbed_ds(
            ds,
            perturb_var=state.perturb_var,
            lon_center=state.perturb_cx,
            lat_center=state.perturb_cy,
            amplitude=float(state.perturb_amplitude),
            sigma_lat=float(state.perturb_sigma_lat),
            sigma_lon=float(state.perturb_sigma_lon),
            wavenumber=int(state.perturb_wavenumber),
            vertical_tilt=bool(state.perturb_vertical_tilt),
        )
        _ds_store["A"] = ds
        _ds_store["B"] = ds_b
        get_cached_contour.cache_clear()
        state.perturb_applied = True
        state.forecasts_ready = False
        state.forecast_status = "IC perturbation applied — inspect, then launch forecast"
        _update_perturb_markers()
        _refresh_all_panels()
        ctrl.view_update()
    except Exception as e:
        state.forecast_status = f"Error applying perturbation: {e}"


CREDIT_CONFIG = "/glade/work/cbecker/CREDIT-visit/miles-credit/config/smoke/smoke_gen2_casper.yml"


def _run_forecasts_background():
    try:
        # Patch sys.argv so argparse inside main() sees our config flag.
        # Restored in finally regardless of outcome.
        _orig_argv = sys.argv
        try:
            sys.argv = ["credit-rollout", "-c", CREDIT_CONFIG]
            rollout_gen2.main()
        finally:
            sys.argv = _orig_argv

        # TODO: load CREDIT output into _ds_store["B"] here, e.g.:
        # output_path = "..."   # from conf["predict"]["save_forecast"]
        # _ds_store["B"] = xr.open_dataset(output_path)
        get_cached_contour.cache_clear()
        _event_loop.call_soon_threadsafe(lambda: _on_forecast_complete("Forecasts ready", True))

    except SystemExit as e:
        # sys.exit() inside main() raises SystemExit — treat non-zero as an error
        if e.code:
            msg = f"CREDIT exited with code {e.code}"
            _event_loop.call_soon_threadsafe(lambda m=msg: _on_forecast_complete(m, False))
        else:
            _event_loop.call_soon_threadsafe(lambda: _on_forecast_complete("Forecasts ready", True))
    except Exception as e:
        msg = f"Error: {e}"
        _event_loop.call_soon_threadsafe(lambda m=msg: _on_forecast_complete(m, False))


@ctrl.add("run_forecasts")
def run_forecasts():
    global _event_loop
    if not state.perturb_applied or state.forecast_status.startswith("Running"):
        return
    _event_loop = asyncio.get_running_loop()  # safe: we're on the asyncio thread here
    state.forecast_status = "Running CREDIT forecast from perturbed IC..."
    _executor.submit(_run_forecasts_background)


@ctrl.add("reset_perturbation")
def reset_perturbation():
    _ds_store["B"] = ds
    state.perturb_placed = False
    state.perturb_applied = False
    state.forecasts_ready = False
    state.forecast_status = "Ready — place perturbation and click Apply"
    get_cached_contour.cache_clear()
    _update_perturb_markers()
    _refresh_all_panels()
    ctrl.view_update()


# ============================================================
# Panel update logic
# ============================================================
def _get_view_clim():
    return (float(state.base_vmin), float(state.base_vmax))


def _get_diff_clim(diff_arr):
    mx = float(np.nanmax(np.abs(diff_arr)))
    mx = mx if mx > 1e-6 else 1.0
    return (-mx, mx)


def _get_contour_preset(slot):
    return {
        "enabled": bool(getattr(state, f"c{slot}_enabled")),
        "var": getattr(state, f"c{slot}_var"),
        "level": parse_level(getattr(state, f"c{slot}_level")),
        "interval": float(getattr(state, f"c{slot}_interval")),
        "color": getattr(state, f"c{slot}_color"),
        "line_width": float(getattr(state, f"c{slot}_line_width")),
    }


def _apply_contours(t_idx):
    for slot in range(2):
        p = _get_contour_preset(slot)
        if not p["enabled"] or p["interval"] <= 0:
            for panel in panels.values():
                panel.remove_contour(slot)
            continue
        for ds_key, panel_key in [("A", "fc_a"), ("A", "diff"), ("A", "diag"), ("B", "fc_b")]:
            mesh = get_cached_contour(
                ds_key=ds_key,
                var=p["var"],
                time_idx=t_idx,
                level=p["level"],
                interval=p["interval"],
            )
            panels[panel_key].set_contour(slot, mesh, p["color"], p["line_width"])


def _refresh_all_panels():
    t = int(state.t_index)
    var = state.view_var
    lev = parse_level(state.view_level)
    clim = _get_view_clim()

    arr_a = get_2d_field(_ds_store["A"], var, t, lev)
    arr_b = get_2d_field(_ds_store["B"], var, t, lev)

    panels["fc_a"].update_base(arr_a, clim=clim)
    panels["fc_b"].update_base(arr_b, clim=clim)

    diff = arr_a - arr_b
    diff_clim = _get_diff_clim(diff)
    panels["diff"].update_base(diff, clim=diff_clim)
    cbar_luts[2].scalar_range = diff_clim

    diag = compute_diagnostic(arr_a, arr_b, state.diagnostic, var, state.view_level)
    diag_max = float(np.nanmax(np.abs(diag)))
    diag_max = diag_max if diag_max > 1e-6 else 1.0
    diag_clim = (0.0, diag_max) if state.diagnostic == "Absolute Difference |A-B|" else (-diag_max, diag_max)
    panels["diag"].update_base(diag, clim=diag_clim)
    cbar_luts[3].scalar_range = diag_clim

    for idx in [0, 1]:
        cbar_luts[idx].scalar_range = clim

    _apply_contours(t)


# ============================================================
# State change callbacks
# ============================================================
@state.change("t_index")
def on_time_change(t_index, **kwargs):
    t0 = time.perf_counter()
    state.t_index = int(round(t_index))
    state.time_text = time_strings[state.t_index]
    _refresh_all_panels()
    ctrl.view_update()
    state.latency_text = f"t={state.t_index:03d} | render={(time.perf_counter() - t0) * 1000:.1f} ms"


@state.change("view_var")
def on_view_var_change(view_var, **kwargs):
    has_lev = has_level(ds, view_var)
    state.view_has_level = has_lev
    if not has_lev:
        state.view_level = "default"
    new_vmin, new_vmax = get_global_range(view_var, parse_level(state.view_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    _refresh_all_panels()
    ctrl.view_update()


@state.change("view_level")
def on_view_level_change(view_level, **kwargs):
    new_vmin, new_vmax = get_global_range(state.view_var, parse_level(view_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    _refresh_all_panels()
    ctrl.view_update()


@state.change("view_cmap")
def on_view_cmap_change(view_cmap, **kwargs):
    clim = _get_view_clim()
    for idx, key in enumerate(["fc_a", "fc_b"]):
        lut = pv.LookupTable(cmap=view_cmap, n_values=256)
        lut.scalar_range = clim
        panels[key].base_actor.mapper.lookup_table = lut
        cbar_luts[idx].cmap = view_cmap
        cbar_luts[idx].scalar_range = clim
    ctrl.view_update()


@state.change("base_vmin", "base_vmax")
def on_clim_change(**kwargs):
    _refresh_all_panels()
    ctrl.view_update()


@state.change("forecasts_ready")
def on_forecasts_ready(forecasts_ready, **kwargs):
    # Intentionally empty: VTK calls from a background-thread state.flush() would
    # use VTK's OpenGL context on the wrong thread and crash. Panel refresh happens
    # via on_time_change / on_view_var_change when the user next interacts, or when
    # CREDIT output is loaded into _ds_store["B"] and forecasts_ready is set on the
    # main thread (e.g. from a future "Load output" button).
    pass


@state.change("diagnostic")
def on_diagnostic_change(**kwargs):
    _refresh_all_panels()
    ctrl.view_update()


@state.change("perturb_var")
def on_perturb_var_change(perturb_var, **kwargs):
    state.perturb_has_level = has_level(ds, perturb_var)
    if not state.perturb_has_level:
        state.perturb_level = "default"


@state.change(
    "c0_enabled",
    "c0_var",
    "c0_level",
    "c0_interval",
    "c0_color",
    "c0_line_width",
    "c1_enabled",
    "c1_var",
    "c1_level",
    "c1_interval",
    "c1_color",
    "c1_line_width",
)
def on_contour_change(**kwargs):
    _apply_contours(int(state.t_index))
    ctrl.view_update()


# ============================================================
# UI helpers
# ============================================================
def contour_controls(slot):
    vuetify.VCheckbox(v_model=(f"c{slot}_enabled", True), label="Enabled", density="compact", hide_details=True)
    with vuetify.VRow(no_gutters=True, classes="mt-1"):
        with vuetify.VCol(cols=7, classes="pr-1"):
            vuetify.VSelect(
                v_model=(f"c{slot}_var", getattr(state, f"c{slot}_var")),
                items=("variable_items",),
                label="Variable",
                density="compact",
                hide_details=True,
            )
        with vuetify.VCol(cols=5):
            vuetify.VSelect(
                v_model=(f"c{slot}_level", "default"),
                items=("level_items",),
                label="Level",
                density="compact",
                hide_details=True,
            )
    with vuetify.VRow(no_gutters=True, classes="mt-1"):
        with vuetify.VCol(cols=5, classes="pr-1"):
            vuetify.VTextField(
                v_model=(f"c{slot}_interval", getattr(state, f"c{slot}_interval")),
                label="Interval",
                type="number",
                density="compact",
                hide_details=True,
            )
        with vuetify.VCol(cols=7):
            vuetify.VTextField(
                v_model=(f"c{slot}_color", getattr(state, f"c{slot}_color")),
                label="Color",
                density="compact",
                hide_details=True,
            )
    vuetify.VSlider(
        v_model=(f"c{slot}_line_width", getattr(state, f"c{slot}_line_width")),
        min=0.5,
        max=4.0,
        step=0.25,
        label="Line width",
        density="compact",
        hide_details=True,
        classes="mt-1",
    )


# ============================================================
# UI Build
# ============================================================
with SinglePageLayout(server) as layout:
    layout.title.set_text("CREDIT Sensitivity Steering")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VChip("{{ time_text }}", classes="mr-2")
        vuetify.VSlider(
            v_model=("t_index", 0),
            min=0,
            max=nt - 1,
            step=1,
            density="compact",
            hide_details=True,
            style="max-width: 350px;",
        )
        vuetify.VChip("{{ latency_text }}", classes="ml-2")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="overflow: hidden; height: 100%;"):
            with vuetify.VRow(classes="fill-height", no_gutters=True, style="overflow: hidden; height: 100%;"):
                # ---- Sidebar ----
                with vuetify.VCol(cols=3, classes="pa-2", style="overflow-y: auto; height: 100%;"):
                    with vuetify.VExpansionPanels(multiple=True, variant="accordion", v_model=("open_panels", [])):
                        # -- View Variable --
                        with vuetify.VExpansionPanel(title="View Variable"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VSelect(
                                    v_model=("view_var", DEFAULT_VIEW_VAR),
                                    items=("variable_items",),
                                    label="Variable",
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify.VSelect(
                                    v_model=("view_level", "default"),
                                    items=("level_items",),
                                    label="Level",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("view_has_level",),
                                )
                                vuetify.VSelect(
                                    v_model=("view_cmap", "viridis"),
                                    items=("cmap_items",),
                                    label="Colormap",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                with vuetify.VRow(no_gutters=True, classes="mt-2"):
                                    with vuetify.VCol(cols=6, classes="pr-1"):
                                        vuetify.VTextField(
                                            v_model=("base_vmin",),
                                            label="Color min",
                                            type="number",
                                            density="compact",
                                            hide_details=True,
                                        )
                                    with vuetify.VCol(cols=6):
                                        vuetify.VTextField(
                                            v_model=("base_vmax",),
                                            label="Color max",
                                            type="number",
                                            density="compact",
                                            hide_details=True,
                                        )

                        # -- Perturbation --
                        with vuetify.VExpansionPanel(title="Perturbation"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VSelect(
                                    v_model=("perturb_var", DEFAULT_PERTURB_VAR),
                                    items=("variable_items",),
                                    label="Perturb Variable",
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify.VSelect(
                                    v_model=("perturb_level", "default"),
                                    items=("level_items",),
                                    label="Level",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("perturb_has_level",),
                                )
                                vuetify.VSlider(
                                    v_model=("perturb_amplitude", 2.0),
                                    min=0.1,
                                    max=20.0,
                                    step=0.1,
                                    label="Amplitude",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-2",
                                    thumb_label=True,
                                )
                                vuetify.VSlider(
                                    v_model=("perturb_sigma_lat", 8.0),
                                    min=1.0,
                                    max=30.0,
                                    step=0.5,
                                    label="σ lat (°)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    thumb_label=True,
                                )
                                vuetify.VSlider(
                                    v_model=("perturb_sigma_lon", 15.0),
                                    min=1.0,
                                    max=60.0,
                                    step=1.0,
                                    label="σ lon (°)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    thumb_label=True,
                                )
                                vuetify.VSlider(
                                    v_model=("perturb_wavenumber", 6),
                                    min=1,
                                    max=15,
                                    step=1,
                                    label="Zonal wavenumber",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    thumb_label=True,
                                )
                                vuetify.VCheckbox(
                                    v_model=("perturb_vertical_tilt", True),
                                    label="Westward vertical tilt",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VCheckbox(
                                    v_model=("perturb_picking_enabled", False),
                                    label="Click map to place perturbation",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VChip(
                                    "Center: {{ perturb_cy.toFixed(1) }}°N  {{ perturb_cx.toFixed(1) }}°E",
                                    size="small",
                                    color="primary",
                                    classes="mt-1",
                                    v_show=("perturb_placed",),
                                )
                                vuetify.VBtn(
                                    "Apply Perturbation",
                                    color="primary",
                                    block=True,
                                    classes="mt-3",
                                    click=ctrl.apply_perturbation,
                                    disabled=("!perturb_placed || !perturb_has_level",),
                                )
                                vuetify.VAlert(
                                    "Wave perturbation requires a 3D (leveled) variable",
                                    density="compact",
                                    variant="tonal",
                                    type="warning",
                                    classes="mt-2 text-caption",
                                    v_show=("perturb_placed && !perturb_has_level",),
                                )

                        # -- Forecast --
                        with vuetify.VExpansionPanel(title="Forecast"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VAlert(
                                    "{{ forecast_status }}",
                                    density="compact",
                                    variant="tonal",
                                    color="secondary",
                                    classes="mb-2 text-caption",
                                )
                                vuetify.VBtn(
                                    "Launch CREDIT Forecast",
                                    color="success",
                                    block=True,
                                    click=ctrl.run_forecasts,
                                    disabled=("!perturb_applied",),
                                )
                                vuetify.VBtn(
                                    "Reset",
                                    color="warning",
                                    block=True,
                                    classes="mt-2",
                                    click=ctrl.reset_perturbation,
                                    variant="outlined",
                                )

                        # -- Contours --
                        with vuetify.VExpansionPanel(title="Contour 1"):
                            with vuetify.VExpansionPanelText():
                                contour_controls(0)
                        with vuetify.VExpansionPanel(title="Contour 2"):
                            with vuetify.VExpansionPanelText():
                                contour_controls(1)

                        # -- Diagnostic --
                        with vuetify.VExpansionPanel(title="Diagnostic Panel"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VSelect(
                                    v_model=("diagnostic", DIAGNOSTIC_ITEMS[0]),
                                    items=("diagnostic_items",),
                                    label="Metric",
                                    density="compact",
                                    hide_details=True,
                                )

                # ---- VTK Viewport ----
                with vuetify.VCol(cols=9, classes="pa-0 fill-height"):
                    view = vtk_widgets.VtkRemoteView(
                        plotter.ren_win,
                        style="width: 100%; height: 100%;",
                        interactive_ratio=1,
                    )
                    ctrl.view_update = view.update

# ============================================================
# Fit camera to viewport on first connect
# ============================================================
_view_fitted = [False]


def _on_window_configure(obj, event):
    w, h = obj.GetSize()
    if w < 2 or h < 2:
        return
    if not _view_fitted[0]:
        aspect = w / h
        dom_hw = 0.5 * (lon_max - lon_min)
        dom_hh = 0.5 * (lat_max - lat_min)
        dom_cx = lon_min + dom_hw
        dom_cy = lat_min + dom_hh
        plotter.camera.parallel_scale = max(dom_hh, dom_hw / aspect)
        pz = plotter.camera.position[2]
        fz = plotter.camera.focal_point[2]
        plotter.camera.position = (dom_cx, dom_cy, pz)
        plotter.camera.focal_point = (dom_cx, dom_cy, fz)
        _view_fitted[0] = True
    else:
        _clamp_camera()
    ctrl.view_update()


plotter.ren_win.AddObserver("ConfigureEvent", _on_window_configure)

# ============================================================
# Initial render
# ============================================================
_refresh_all_panels()
state.latency_text = "Ready"

if __name__ == "__main__":
    server.start(port=8081, host="0.0.0.0", open_browser=False)
    pl = pv.Plotter(off_screen=True)
    print(pl.ren_win.ReportCapabilities())
