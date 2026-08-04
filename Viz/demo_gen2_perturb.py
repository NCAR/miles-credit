import asyncio
import os
import time
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyvista as pv

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

import gen2_data
import gen2_rollout as gr

# ============================================================
# User settings
# ============================================================
CONFIG_PATH = "../config/gen_2/camulator/camulator_cesm_tutorial_casper.yml"

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

# ============================================================
# Load config + initial-condition SST (for click placement, before any rollout has run)
# ============================================================
conf = gen2_data.load_config(CONFIG_PATH)
output_variables = gen2_data.bare_output_variables(conf)
DEFAULT_VIEW_VAR = "TREFHT" if "TREFHT" in output_variables else output_variables[0]

ic_ds = gen2_data.load_ic_snapshot(conf, ["SST"])
lon = ic_ds["longitude"].values.astype(np.float32)
lat = ic_ds["latitude"].values.astype(np.float32)
ic_sst = get_2d_field(ic_ds, "SST", time_idx=0)
ic_vmin, ic_vmax = float(np.nanquantile(ic_sst, 0.01)), float(np.nanquantile(ic_sst, 0.99))

# Populated once a rollout has actually run (see run_forecasts / _on_run_complete).
_ds_store = {}


# ============================================================
# Global range + contour caches (keyed on the ds_key so A/B don't collide)
# ============================================================
@lru_cache(maxsize=32)
def get_global_range(ds_key, var, level):
    da = _ds_store[ds_key][var]
    if "level" in da.dims:
        da = da.isel(level=-1) if level is None else da.sel(level=level)
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


@lru_cache(maxsize=128)
def get_cached_contour(ds_key, var, time_idx, level, interval, stride=1):
    _ds = _ds_store[ds_key]
    return make_contours(
        ds=_ds, var=var, time_idx=time_idx, level=level, interval=interval, stride=stride, lon=lon, lat=lat
    )


_temporal_std_cache = {}


def get_temporal_std(var, level):
    key = (var, level)
    if key not in _temporal_std_cache:
        da = _ds_store["A"][var]
        lev = parse_level(level)
        if "level" in da.dims:
            da = da.isel(level=-1) if lev is None else da.sel(level=lev)
        std = da.std(dim="time").values.astype(np.float32)  # (lat, lon)
        std = std.T  # -> (lon, lat)
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

# Initial content: real IC SST on panels A/B (something to click on); diff/diag start as zeros.
for key in ["fc_a", "fc_b"]:
    panels[key].add_base(ic_sst, clim=(ic_vmin, ic_vmax))

zero = np.zeros_like(ic_sst)
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
    lut.scalar_range = (ic_vmin, ic_vmax)
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
    # %.2g (2 significant figures, auto-switches to scientific notation) instead of a fixed
    # %.1f -- a flat 1-decimal format rounds any small-magnitude variable (e.g. Qtot ~1e-3) to
    # "0.0" on every tick.
    sb.SetLabelFormat("%.2g")
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
# Perturbation marker (ENSO blob center)
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

state.variable_items = output_variables
state.level_items = ["default"]
state.diagnostic_items = DIAGNOSTIC_ITEMS
state.cmap_items = CMAP_ITEMS

state.view_var = DEFAULT_VIEW_VAR
state.view_level = "default"
state.view_has_level = False
state.view_cmap = "viridis"
state.t_index = 0
state.t_max = 0
state.time_text = "Initial condition"
state.latency_text = "Ready"

# Perturbation state (SST-only Gaussian "El Niño" blob; defaults match the tutorial notebook's
# idealized-ENSO example: notebooks/camulator_tutorial_experiments.ipynb).
state.perturb_amplitude = 3.0
state.perturb_sigma_lat = 7.5
state.perturb_sigma_lon = 30.0
state.perturb_picking_enabled = False
state.perturb_placed = False
state.perturb_applied = False
state.perturb_cx = float(lon.mean())
state.perturb_cy = 0.0

# Contour state (two slots, shared across all panels) — only meaningful once forecasts_ready.
state.c0_enabled = False
state.c0_var = output_variables[0]
state.c0_level = "default"
state.c0_interval = 5.0
state.c0_color = "black"
state.c0_line_width = 1.2

state.c1_enabled = False
state.c1_var = output_variables[0]
state.c1_level = "default"
state.c1_interval = 5.0
state.c1_color = "white"
state.c1_line_width = 1.0

# Forecast / diagnostic state
state.diagnostic = DIAGNOSTIC_ITEMS[0]
state.forecast_status = "Ready — click the ocean to place the ENSO SST anomaly, then Apply"
state.forecasts_ready = False

# Colorbar range state (user-editable)
state.base_vmin = ic_vmin
state.base_vmax = ic_vmax

# ============================================================
# Run-path staging (Apply Perturbation) — writes a perturbed forcing zarr + config copies.
# Actually launching `credit rollout` is wired in a follow-up step.
# ============================================================
_run_paths = {}


def _ensure_run_paths():
    if _run_paths:
        return _run_paths
    rollout_root = Path(os.path.expandvars(gr.get_by_path(conf, gr.OUTPUT_KEY)))
    run_root = rollout_root.parent
    _run_paths.update(
        forcing_root=run_root / "forcing",
        # Control uses unmodified forcing, so it just writes to the config's own default
        # save_forecast location -- the same place a plain `credit rollout -c <this config>`
        # run (including one you launch by hand outside this app) would land, so it's
        # automatically picked up as a cache hit either way.
        control_cfg=Path(CONFIG_PATH).resolve(),
        perturbed_out=rollout_root / "interactive_perturbed",
    )
    return _run_paths


_executor = ThreadPoolExecutor(max_workers=1)


@ctrl.add("apply_perturbation")
def apply_perturbation():
    if not state.perturb_placed:
        return
    try:
        state.forecast_status = "Staging perturbed SST forcing..."
        state.flush()
        paths = _ensure_run_paths()

        start_dt = conf["inference"]["single_forecast"]["start_datetime"]
        length = conf["inference"]["single_forecast"]["forecast_length"]
        years = gr.forecast_years(start_dt, length)

        edit_fn = gr.add_enso_blob(
            amp=float(state.perturb_amplitude),
            lat0=float(state.perturb_cy),
            lon0=float(state.perturb_cx),
            sig_lat=float(state.perturb_sigma_lat),
            sig_lon=float(state.perturb_sigma_lon),
        )
        src_template = gr.get_by_path(conf, gr.DYN_FORCING_KEY)
        perturbed_template = gr.perturb_forcing(
            src_template=src_template,
            dst_dir=paths["forcing_root"] / "interactive_perturbed",
            years=years,
            edit_fn=edit_fn,
        )

        perturbed_cfg = gr.write_config(
            conf,
            {gr.DYN_FORCING_KEY: perturbed_template, gr.OUTPUT_KEY: str(paths["perturbed_out"]) + "/"},
            paths["perturbed_out"] / "config.yml",
        )
        _run_paths["perturbed_cfg"] = perturbed_cfg

        state.perturb_applied = True
        state.forecast_status = f"Perturbation staged for {years} — ready to run"
    except Exception as e:
        state.forecast_status = f"Error staging perturbation: {e}"
    state.flush()


_event_loop = None  # captured on the asyncio thread in run_forecasts()
_last_status_post = [0.0]


def _post_status_immediate(text):
    """Thread-safe: marshals onto the asyncio (main/VTK) thread via call_soon_threadsafe."""
    if _event_loop is None:
        print(text)
        return

    def _apply():
        state.forecast_status = text
        state.flush()

    _event_loop.call_soon_threadsafe(_apply)


def _post_status_throttled(text):
    """Same as above but rate-limited, for run_credit's high-frequency tqdm-refresh callback."""
    now = time.perf_counter()
    if now - _last_status_post[0] < 0.25:
        return
    _last_status_post[0] = now
    _post_status_immediate(text)


def _on_run_complete(pred_a, pred_b):
    """Runs on the asyncio thread (via call_soon_threadsafe) -- safe to touch VTK/state here."""
    _ds_store["A"] = pred_a
    _ds_store["B"] = pred_b
    get_global_range.cache_clear()
    get_cached_contour.cache_clear()
    _temporal_std_cache.clear()

    nt = pred_a.sizes["time"]
    state.t_max = nt - 1
    state.t_index = 0
    state.time_text = pred_a["time"].values[0].strftime("%Y-%m-%d %H:%M")

    state.view_has_level = has_level(pred_a, state.view_var)
    levels = pred_a["level"].values if "level" in pred_a.dims else []
    state.level_items = ["default"] + [str(int(v)) for v in levels]

    vmin, vmax = get_global_range("A", state.view_var, parse_level(state.view_level))
    state.base_vmin, state.base_vmax = vmin, vmax

    state.forecasts_ready = True
    state.forecast_status = "Forecasts ready"
    state.flush()

    _refresh_all_panels()
    ctrl.view_update()


def _on_run_error(msg):
    state.forecast_status = msg
    state.flush()


def _run_forecasts_background():
    try:
        control_cfg = _run_paths["control_cfg"]
        perturbed_cfg = _run_paths["perturbed_cfg"]
        control_conf = gen2_data.load_config(str(control_cfg))
        perturbed_conf = gen2_data.load_config(str(perturbed_cfg))

        try:
            pred_a = gen2_data.load_prediction_dataset(control_conf, variables=output_variables)
            _post_status_immediate("Control run already cached — skipping to perturbed run")
        except FileNotFoundError:
            _post_status_immediate("Running control forecast...")
            gr.run_credit(control_cfg, mode="none", procs=4, on_line=_post_status_throttled)
            pred_a = gen2_data.load_prediction_dataset(control_conf, variables=output_variables)

        _post_status_immediate("Running perturbed forecast...")
        gr.run_credit(perturbed_cfg, mode="none", procs=4, on_line=_post_status_throttled)
        pred_b = gen2_data.load_prediction_dataset(perturbed_conf, variables=output_variables)

        _post_status_immediate("Loading rollout output...")
        _event_loop.call_soon_threadsafe(lambda: _on_run_complete(pred_a, pred_b))
    except Exception as e:
        msg = f"Error running forecast: {e}"
        if _event_loop is not None:
            _event_loop.call_soon_threadsafe(lambda m=msg: _on_run_error(m))
        else:
            print(msg)


@ctrl.add("run_forecasts")
def run_forecasts():
    global _event_loop
    if not state.perturb_applied or state.forecast_status.startswith("Running"):
        return
    _event_loop = asyncio.get_running_loop()  # safe: we're on the asyncio thread here
    state.forecast_status = "Running forecast..."
    _executor.submit(_run_forecasts_background)


@ctrl.add("reset_perturbation")
def reset_perturbation():
    state.perturb_placed = False
    state.perturb_applied = False
    state.forecast_status = "Ready — click the ocean to place the ENSO SST anomaly, then Apply"
    _update_perturb_markers()
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
    if not state.forecasts_ready:
        for panel in panels.values():
            panel.remove_contour(0)
            panel.remove_contour(1)
        return
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
    if not state.forecasts_ready:
        clim = (ic_vmin, ic_vmax)
        for key in ["fc_a", "fc_b"]:
            panels[key].update_base(ic_sst, clim=clim)
        zero = np.zeros_like(ic_sst)
        panels["diff"].update_base(zero, clim=(-1.0, 1.0))
        panels["diag"].update_base(zero, clim=(0.0, 1.0))
        for idx in [0, 1]:
            cbar_luts[idx].scalar_range = clim
        return

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
    if state.forecasts_ready:
        # cftime.datetime (noleap calendar) has its own .strftime -- don't wrap in pd.Timestamp,
        # which doesn't accept cftime objects.
        state.time_text = _ds_store["A"]["time"].values[state.t_index].strftime("%Y-%m-%d %H:%M")
    _refresh_all_panels()
    ctrl.view_update()
    state.latency_text = f"t={state.t_index:03d} | render={(time.perf_counter() - t0) * 1000:.1f} ms"


@state.change("view_var")
def on_view_var_change(view_var, **kwargs):
    if not state.forecasts_ready:
        return
    has_lev = has_level(_ds_store["A"], view_var)
    state.view_has_level = has_lev
    if not has_lev:
        state.view_level = "default"
    new_vmin, new_vmax = get_global_range("A", view_var, parse_level(state.view_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    _refresh_all_panels()
    ctrl.view_update()


@state.change("view_level")
def on_view_level_change(view_level, **kwargs):
    if not state.forecasts_ready:
        return
    new_vmin, new_vmax = get_global_range("A", state.view_var, parse_level(view_level))
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


@state.change("diagnostic")
def on_diagnostic_change(**kwargs):
    _refresh_all_panels()
    ctrl.view_update()


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
    vuetify.VCheckbox(v_model=(f"c{slot}_enabled", False), label="Enabled", density="compact", hide_details=True)
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
    layout.title.set_text("CREDIT ENSO Perturbation")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VChip("{{ time_text }}", classes="mr-2")
        vuetify.VSlider(
            v_model=("t_index", 0),
            min=0,
            max=("t_max",),
            step=1,
            density="compact",
            hide_details=True,
            style="max-width: 350px;",
            disabled=("!forecasts_ready",),
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
                                vuetify.VAlert(
                                    "Showing the initial-condition SST — panels switch to rollout "
                                    "output once a forecast has run.",
                                    density="compact",
                                    variant="tonal",
                                    type="info",
                                    classes="mb-2 text-caption",
                                    v_show=("!forecasts_ready",),
                                )
                                vuetify.VSelect(
                                    v_model=("view_var", DEFAULT_VIEW_VAR),
                                    items=("variable_items",),
                                    label="Variable",
                                    density="compact",
                                    hide_details=True,
                                    disabled=("!forecasts_ready",),
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
                        with vuetify.VExpansionPanel(title="Perturbation (ENSO SST)"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VSlider(
                                    v_model=("perturb_amplitude", 3.0),
                                    min=0.1,
                                    max=10.0,
                                    step=0.1,
                                    label="Amplitude (K)",
                                    density="compact",
                                    hide_details=True,
                                    thumb_label=True,
                                )
                                vuetify.VSlider(
                                    v_model=("perturb_sigma_lat", 7.5),
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
                                    v_model=("perturb_sigma_lon", 30.0),
                                    min=1.0,
                                    max=90.0,
                                    step=1.0,
                                    label="σ lon (°)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    thumb_label=True,
                                )
                                vuetify.VCheckbox(
                                    v_model=("perturb_picking_enabled", False),
                                    label="Click map to place ENSO center",
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
                                    disabled=("!perturb_placed",),
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
                                    "Run Forecast",
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
