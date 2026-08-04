import sys
import time
from functools import lru_cache

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
    make_cartopy_coastline_texture,
    make_coast_plane,
    make_contours,
    make_dashed_line_mesh,
    make_star_mesh,
    get_vertical_slice,
    pick_colorbar_label_format,
    MapPanel,
    TimeSeriesPanel,
    VerticalSlicePanel,
)
import gen2_data

# ============================================================
# User settings
# ============================================================
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "../config/gen_2/camulator/camulator_cesm_tutorial_casper.yml"
DEFAULT_VAR = "TREFHT"

# ============================================================
# Load config + truth/prediction datasets
# ============================================================
conf = gen2_data.load_config(CONFIG_PATH)
variable_items = gen2_data.bare_output_variables(conf)
truth, pred = gen2_data.load_truth_and_prediction(conf, variables=variable_items)

lon = pred["longitude"].values.astype(np.float32)
lat = pred["latitude"].values.astype(np.float32)

times = pred["time"].values
nt = pred.sizes["time"]
time_strings = [t.strftime("%Y-%m-%d %H:%M") for t in times]
ts_x_data = np.arange(nt, dtype=np.float32)

# level_items values are the plain 0..31 index (what .sel(level=...) expects on
# both datasets, see gen2_data.align_truth_to_prediction); titles show the real
# CESM hybrid-sigma level value (truth["level_value"]) for readability.
if "level" in pred.dims:
    real_levels = truth["level_value"].values if "level_value" in truth.coords else pred["level"].values
    level_value_labels = [f"{float(v):.2f}" for v in real_levels]
    level_items = [{"title": "default", "value": "default"}] + [
        {"title": label, "value": str(i)} for i, label in enumerate(level_value_labels)
    ]
else:
    level_value_labels = []
    level_items = [{"title": "default", "value": "default"}]

if DEFAULT_VAR not in variable_items:
    DEFAULT_VAR = variable_items[0]

base0 = get_2d_field(pred, DEFAULT_VAR, time_idx=0, level=parse_level("default"))


# ============================================================
# Global range cache
# ============================================================
@lru_cache(maxsize=32)
def get_global_range(var, level):
    da = pred[var]
    if "level" in da.dims:
        da = da.isel(level=-1) if level is None else da.sel(level=level)
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


def get_diff_clim(diff_arr):
    mx = float(np.nanmax(np.abs(diff_arr)))
    mx = mx if mx > 1e-6 else 1.0
    return (-mx, mx)


@lru_cache(maxsize=128)
def get_cached_contour(var, time_idx, level, interval, stride=1):
    return make_contours(
        ds=pred, var=var, time_idx=time_idx, level=level, interval=interval, stride=stride, lon=lon, lat=lat
    )


def get_contour_preset_from_state(state):
    return {
        "enabled": bool(state.c0_enabled),
        "var": state.c0_var,
        "level": parse_level(state.c0_level),
        "interval": float(state.c0_interval),
        "color": state.c0_color,
        "line_width": float(state.c0_line_width),
    }


def update_contours_for_panels(time_idx, state, target_panels):
    preset = get_contour_preset_from_state(state)

    if (not preset["enabled"]) or preset["interval"] <= 0:
        for panel in target_panels:
            panel.remove_contour(0)
        return

    contour_mesh = get_cached_contour(
        var=preset["var"], time_idx=time_idx, level=preset["level"], interval=preset["interval"], stride=1
    )
    for panel in target_panels:
        panel.set_contour(slot=0, contour_mesh=contour_mesh, color=preset["color"], line_width=preset["line_width"])


vmin, vmax = get_global_range(DEFAULT_VAR, parse_level("default"))
base_clim = (vmin, vmax)

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
# Vertical slice (forecast field only, for now)
# ============================================================
slice_variable_items = [v for v in variable_items if "level" in pred[v].dims]


@lru_cache(maxsize=32)
def get_slice_global_range(var):
    da = pred[var]
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


# ============================================================
# Coastline texture
# ============================================================
coast_img, extent = make_cartopy_coastline_texture(lon, lat, resolution="50m", dpi=400, linewidth=0.8)
lon_min, lon_max, lat_min, lat_max = extent
coast_texture = pv.Texture(coast_img)
coast_plane = make_coast_plane(extent, z_value=4.0)

# ============================================================
# PyVista scene setup
# ============================================================
plotter = pv.Plotter(
    off_screen=False,
    window_size=[80, 60],
    border=False,
    polygon_smoothing=True,
    line_smoothing=True,
    shape=(2, 2),
)

border_actors = [r.add_border(color="black", width=10.0) for r in plotter.renderers]

# Title strips for all 4 panels: separate, non-interactive, fixed-viewport renderers above
# each data panel (viewport set below alongside the colorbar strips). Keeping the label outside
# the data renderer means panning/zooming that renderer's camera can never scroll the field
# behind the label or shrink the data window below it.
title_renderers = [VtkRenderer(), VtkRenderer(), VtkRenderer(), VtkRenderer()]
for _r in title_renderers:
    _r.SetBackground(1.0, 1.0, 1.0)
    _r.SetInteractive(False)
    plotter.ren_win.AddRenderer(_r)

panels = {
    "forecast": MapPanel(
        plotter,
        0,
        0,
        lon,
        lat,
        coast_texture,
        coast_plane,
        title="Forecast",
        cmap="viridis",
        show_scalar_bar=False,
        title_renderer=title_renderers[0],
    ),
    "diff": MapPanel(
        plotter,
        0,
        1,
        lon,
        lat,
        coast_texture,
        coast_plane,
        title="Difference (Forecast − Truth)",
        cmap="coolwarm",
        show_scalar_bar=False,
        title_renderer=title_renderers[1],
    ),
    "slice": VerticalSlicePanel(
        plotter,
        1,
        0,
        title="Vertical Slice",
        cmap="viridis",
        border_actor=border_actors[2],
        title_renderer=title_renderers[2],
    ),
    "ts": TimeSeriesPanel(plotter, 1, 1, border_actor=border_actors[3], title_renderer=title_renderers[3]),
}

map_panels = [panels["forecast"], panels["diff"]]

for panel in map_panels:
    panel.add_base(base0, clim=base_clim, opacity=0.95)
panels["diff"].update_base(np.zeros_like(base0), clim=(-1.0, 1.0))

for r, c in [(0, 0), (0, 1), (1, 0)]:
    plotter.subplot(r, c)
    plotter.view_xy()
    plotter.camera.parallel_projection = True

_shared_map_cam = plotter.renderers[0].GetActiveCamera()
plotter.renderers[1].SetActiveCamera(_shared_map_cam)

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

# ============================================================
# Colorbar renderer strips
# ============================================================
CBAR_STRIP = 0.06
TITLE_STRIP = 0.05

_data_vp = [
    (0.0, 0.5 + CBAR_STRIP, 0.5, 1.0 - TITLE_STRIP),  # Forecast
    (0.5, 0.5 + CBAR_STRIP, 1.0, 1.0 - TITLE_STRIP),  # Diff
    (0.0, 0.0 + CBAR_STRIP, 0.5, 0.5 - TITLE_STRIP),  # Vertical slice
]
_cbar_vp = [
    (0.0, 0.5, 0.5, 0.5 + CBAR_STRIP),
    (0.5, 0.5, 1.0, 0.5 + CBAR_STRIP),
    (0.0, 0.0, 0.5, 0.0 + CBAR_STRIP),
]
_title_vp = [
    (0.0, 1.0 - TITLE_STRIP, 0.5, 1.0),  # Forecast
    (0.5, 1.0 - TITLE_STRIP, 1.0, 1.0),  # Diff
    (0.0, 0.5 - TITLE_STRIP, 0.5, 0.5),  # Vertical slice
    (0.5, 0.5 - TITLE_STRIP, 1.0, 0.5),  # Time series
]
for _r, vp in zip(title_renderers, _title_vp):
    _r.SetViewport(*vp)

plotter.renderers[3].SetViewport(0.5, 0.0, 1.0, 0.5 - TITLE_STRIP)

_panel_cmaps = [panels["forecast"].cmap, panels["diff"].cmap, panels["slice"].cmap]

for i, vp in enumerate(_data_vp):
    plotter.renderers[i].SetViewport(*vp)

cbar_luts = []
cbar_renderers = []
cbar_actors = []

for cmap_name, cvp in zip(_panel_cmaps, _cbar_vp):
    lut = pv.LookupTable(cmap=cmap_name, n_values=256)
    lut.scalar_range = (vmin, vmax)
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
    sb.SetLabelFormat(pick_colorbar_label_format(vmin, vmax))
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

# Vertical slice panel starts hidden until toggled on in the sidebar.
cbar_renderers[2].SetBackground(0.9, 0.9, 0.9)
cbar_actors[2].SetVisibility(False)


def _set_cbar_range(idx, clim):
    """Update a colorbar's range and its tick-label format together, so the format always
    matches whatever variable/range is currently shown instead of a single fixed format
    string compromising between very different variable magnitudes."""
    cbar_luts[idx].scalar_range = clim
    cbar_actors[idx].SetLabelFormat(pick_colorbar_label_format(*clim))


# ============================================================
# Interactor / Camera helpers
# ============================================================
_ZOOM_FACTOR = 1.15


def _clamp_camera():
    # plotter.camera follows whatever subplot was last made active (e.g. by the
    # slice panel's set_slice()), not necessarily the shared map camera -- grab
    # it explicitly from renderer 0 instead.
    camera = plotter.renderers[0].GetActiveCamera()
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

    if view_hw >= dom_hw:
        px = dom_cx
    else:
        px = max(lon_min + view_hw, min(lon_max - view_hw, px))

    if view_hh >= dom_hh:
        py = dom_cy
    else:
        py = max(lat_min + view_hh, min(lat_max - view_hh, py))

    camera.position = (px, py, pz)
    camera.focal_point = (px, py, fz)


def _zoom_at_cursor(direction):
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my) or plotter.renderers[0]
    # Camera of the renderer actually under the cursor, not plotter.camera
    # (which follows last-active-subplot state -- see _clamp_camera).
    camera = renderer.GetActiveCamera()

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

    if renderer in (plotter.renderers[0], plotter.renderers[1]):
        _user_interacted[0] = True
    _clamp_camera()
    ctrl.view_update()


def event_is_over_vertical_slice():
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my)
    return renderer == plotter.renderers[2]


def event_is_over_ts_chart():
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my)
    return renderer == plotter.renderers[3]


def _update_slice_from_camera():
    fp = _shared_map_cam.GetFocalPoint()
    if state.slice_orientation == "latitude":
        state.slice_lat_value = float(np.clip(fp[1], float(lat.min()), float(lat.max())))
    else:
        state.slice_lon_value = float(np.clip(fp[0], float(lon.min()), float(lon.max())))


def _on_scroll_forward(obj, event):
    if event_is_over_ts_chart():
        return
    if event_is_over_vertical_slice():
        panels["slice"].restore_camera_state()
        ctrl.view_update()
        return
    _zoom_at_cursor(1)
    _update_slice_from_camera()


def _on_scroll_backward(obj, event):
    if event_is_over_ts_chart():
        return
    if event_is_over_vertical_slice():
        panels["slice"].restore_camera_state()
        ctrl.view_update()
        return
    _zoom_at_cursor(-1)
    _update_slice_from_camera()


def _on_pan_end(obj, event):
    if event_is_over_vertical_slice():
        panels["slice"].restore_camera_state()
    elif not event_is_over_ts_chart():
        _user_interacted[0] = True
        _clamp_camera()
        _update_slice_from_camera()
    ctrl.view_update()


def _on_interaction(obj, event):
    if event_is_over_vertical_slice():
        panels["slice"].restore_camera_state()


plotter.iren.interactor.AddObserver("MouseWheelForwardEvent", _on_scroll_forward, 1.0)
plotter.iren.interactor.AddObserver("MouseWheelBackwardEvent", _on_scroll_backward, 1.0)
plotter.iren.interactor.AddObserver("InteractionEvent", _on_interaction, 1.0)
plotter.iren.interactor.AddObserver("EndInteractionEvent", _on_pan_end)

# ============================================================
# Trame app initialization
# ============================================================
server = get_server()
state, ctrl = server.state, server.controller

state.variable_items = variable_items
state.level_items = level_items
state.cmap_items = CMAP_ITEMS

state.base_var = DEFAULT_VAR
state.base_has_level = has_level(pred, DEFAULT_VAR)
state.base_level = "default"
state.base_cmap = "viridis"
state.panel2_mode = "Difference"
state.panel2_mode_items = ["Difference", "Truth"]
state.t_index = 0
state.time_text = time_strings[0]
state.latency_text = "Ready"

state.base_vmin = vmin
state.base_vmax = vmax

state.c0_enabled = True
state.c0_var = DEFAULT_VAR
state.c0_level = "default"
state.c0_interval = 5.0
state.c0_color = "black"
state.c0_line_width = 1.2

state.slice_variable_items = slice_variable_items
state.slice_var = "T" if "T" in slice_variable_items else (slice_variable_items[0] if slice_variable_items else "")
state.slice_cmap = "viridis"
state.slice_n_contours = 10
state.slice_orientation = "latitude"
state.slice_lat_value = 0.0
state.slice_lon_value = 260.0
state.slice_orientation_items = ["latitude", "longitude"]
state.slice_panel_visible = False
# x-axis sub-range for the slice: longitude range when orientation="latitude"
# (x=longitude), latitude range when orientation="longitude" (x=latitude).
state.slice_lon_range = [float(lon.min()), float(lon.max())]
state.slice_lat_range = [float(lat.min()), float(lat.max())]

state.ts_picking_enabled = False
state.ts_point_picked = False
state.ts_lon = float(lon.min() + (lon.max() - lon.min()) / 2)
state.ts_lat = float(lat.min() + (lat.max() - lat.min()) / 2)
state.ts_ymin = 0.0
state.ts_ymax = 1.0


# ============================================================
# Panel update logic
# ============================================================
def refresh_panels():
    t = int(state.t_index)
    var = state.base_var
    lev = parse_level(state.base_level)

    pred_arr = get_2d_field(pred, var, t, lev)
    truth_arr = get_2d_field(truth, var, t, lev)

    clim = (state.base_vmin, state.base_vmax)
    panels["forecast"].update_base(pred_arr, clim=clim)

    if state.panel2_mode == "Truth":
        panels["diff"].update_base(truth_arr, clim=clim)
        _set_cbar_range(1, clim)
    else:
        diff_arr = pred_arr - truth_arr
        diff_clim = get_diff_clim(diff_arr)
        panels["diff"].update_base(diff_arr, clim=diff_clim)
        _set_cbar_range(1, diff_clim)

    update_contours_for_panels(t, state, map_panels)


def update_vertical_slice():
    if not slice_variable_items:
        return
    if not state.slice_panel_visible:
        # Panel hidden -- skip the dask read entirely instead of computing a slice nobody sees.
        return
    x, levels, arr, title = get_vertical_slice(
        ds=pred,
        var=state.slice_var,
        time_idx=int(state.t_index),
        orientation=state.slice_orientation,
        lat_value=float(state.slice_lat_value),
        lon_value=float(state.slice_lon_value),
    )

    x_range = state.slice_lon_range if state.slice_orientation == "latitude" else state.slice_lat_range
    mask = (x >= x_range[0]) & (x <= x_range[1])
    if mask.sum() >= 2:
        x = x[mask]
        arr = arr[mask, :]

    vmin_s, vmax_s = get_slice_global_range(state.slice_var)
    panels["slice"].set_slice(
        x=x,
        levels=levels,
        arr=arr,
        clim=(vmin_s, vmax_s),
        title=title,
        level_labels=level_value_labels,
        show_axis_labels=True,
        n_contours=int(state.slice_n_contours),
    )
    _set_cbar_range(2, (vmin_s, vmax_s))


_slice_line_actors = {"forecast": None, "diff": None}


def _update_slice_lines():
    for key in ["forecast", "diff"]:
        panel = panels[key]
        if _slice_line_actors[key] is not None:
            plotter.subplot(panel.row, panel.col)
            plotter.remove_actor(_slice_line_actors[key])
            _slice_line_actors[key] = None

        if not state.slice_panel_visible:
            continue

        plotter.subplot(panel.row, panel.col)
        if state.slice_orientation == "latitude":
            lat_val = float(state.slice_lat_value)
            x0, x1 = state.slice_lon_range
            mesh = make_dashed_line_mesh(x0, lat_val, x1, lat_val, z=5.0)
        else:
            lon_val = float(state.slice_lon_value)
            y0, y1 = state.slice_lat_range
            mesh = make_dashed_line_mesh(lon_val, y0, lon_val, y1, z=5.0)

        actor = plotter.add_mesh(mesh, color="dimgray", line_width=2.0, lighting=False, pickable=False)
        _slice_line_actors[key] = actor


def update_timeseries_chart():
    """Truth vs. forecast at the clicked point, for the currently-displayed variable/level."""
    if not state.ts_point_picked:
        return

    var = state.base_var
    lev = parse_level(state.base_level)

    pred_da = pred[var]
    truth_da = truth[var]
    if "level" in pred_da.dims:
        pred_da = pred_da.isel(level=-1) if lev is None else pred_da.sel(level=lev)
        truth_da = truth_da.isel(level=-1) if lev is None else truth_da.sel(level=lev)

    pred_y = pred_da.sel(longitude=state.ts_lon, latitude=state.ts_lat, method="nearest").values.astype(np.float32)
    truth_y = truth_da.sel(longitude=state.ts_lon, latitude=state.ts_lat, method="nearest").values.astype(np.float32)

    y_min = float(min(np.nanmin(pred_y), np.nanmin(truth_y)))
    y_max = float(max(np.nanmax(pred_y), np.nanmax(truth_y)))
    state.ts_ymin, state.ts_ymax = y_min, y_max

    tick_locs = np.linspace(0, nt - 1, min(5, nt), dtype=int)
    tick_labels = [time_strings[i] for i in tick_locs]

    lvl_str = f" (Level: {lev})" if lev is not None else ""
    title = f"{var}{lvl_str} at {state.ts_lat:.2f}°N, {state.ts_lon:.2f}°E"

    panels["ts"].update_multi_chart(
        ts_x_data,
        series=[
            {"label": "Truth", "y": truth_y, "color": "black"},
            {"label": "Forecast", "y": pred_y, "color": "blue"},
        ],
        tick_locs=tick_locs,
        tick_labels=tick_labels,
        title=title,
        y_label=var,
    )
    panels["ts"].update_time_indicator(int(state.t_index), state.ts_ymin, state.ts_ymax)


_ts_marker_actors = {"forecast": None, "diff": None}


def _update_ts_markers():
    for key in ["forecast", "diff"]:
        panel = panels[key]
        plotter.subplot(panel.row, panel.col)
        if _ts_marker_actors[key] is not None:
            plotter.remove_actor(_ts_marker_actors[key])
            _ts_marker_actors[key] = None

        if state.ts_picking_enabled:
            marker = make_star_mesh(state.ts_lon, state.ts_lat, radius=1.5, z=5.0)
            _ts_marker_actors[key] = plotter.add_mesh(marker, color="black", lighting=False, pickable=False)


def _on_left_click(obj, event):
    if not state.ts_picking_enabled:
        return
    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my)
    if renderer not in [plotter.renderers[0], plotter.renderers[1]]:
        return

    renderer.SetDisplayPoint(mx, my, 0)
    renderer.DisplayToWorld()
    w = renderer.GetWorldPoint()
    wx, wy = w[0] / w[3], w[1] / w[3]

    state.ts_lon = float(np.clip(wx, float(lon.min()), float(lon.max())))
    state.ts_lat = float(np.clip(wy, float(lat.min()), float(lat.max())))
    state.ts_point_picked = True
    state.flush()
    ctrl.view_update()


plotter.iren.interactor.AddObserver("LeftButtonPressEvent", _on_left_click, 1.0)


@state.change("t_index")
def on_time_change(t_index, **kwargs):
    t0 = time.perf_counter()
    state.t_index = int(round(t_index))
    state.time_text = time_strings[state.t_index]
    refresh_panels()
    update_vertical_slice()
    panels["ts"].update_time_indicator(state.t_index, state.ts_ymin, state.ts_ymax)
    ctrl.view_update()
    state.latency_text = f"t={state.t_index:03d} | render={(time.perf_counter() - t0) * 1000:.1f} ms"


@state.change("ts_lon", "ts_lat", "ts_picking_enabled")
def on_ts_marker_change(**kwargs):
    _update_ts_markers()
    ctrl.view_update()


@state.change("ts_lon", "ts_lat")
def on_ts_point_change(**kwargs):
    update_timeseries_chart()
    ctrl.view_update()


@state.change("c0_enabled", "c0_var", "c0_level", "c0_interval", "c0_color", "c0_line_width")
def on_contour_change(**kwargs):
    update_contours_for_panels(int(state.t_index), state, map_panels)
    ctrl.view_update()


@state.change(
    "slice_var",
    "slice_orientation",
    "slice_lat_value",
    "slice_lon_value",
    "slice_lon_range",
    "slice_lat_range",
    "slice_n_contours",
)
def on_slice_selection_change(**kwargs):
    update_vertical_slice()
    _update_slice_lines()
    ctrl.view_update()


@state.change("slice_panel_visible")
def on_slice_panel_visible(slice_panel_visible, **kwargs):
    panels["slice"].toggle_visibility(slice_panel_visible)

    if slice_panel_visible:
        cbar_renderers[2].SetBackground(1.0, 1.0, 1.0)
        cbar_actors[2].SetVisibility(True)
        # toggle_visibility() only flips SetVisibility() -- it doesn't rebuild the
        # grid or camera. Whatever fit happened at module-load time (before the
        # browser was connected, at the placeholder window_size) is what's being
        # revealed. Force a fresh fit now, when the real viewport size is
        # guaranteed correct since the app is actively running.
        panels["slice"].view_initialized = False
        update_vertical_slice()
    else:
        cbar_renderers[2].SetBackground(0.9, 0.9, 0.9)
        cbar_actors[2].SetVisibility(False)

    _update_slice_lines()
    ctrl.view_update()


@state.change("base_var")
def on_base_var_change(base_var, **kwargs):
    has_lev = has_level(pred, base_var)
    state.base_has_level = has_lev
    if not has_lev:
        state.base_level = "default"
    elif state.base_level in (None, "default") and len(level_items) > 1:
        state.base_level = level_items[1]["value"]
    new_vmin, new_vmax = get_global_range(base_var, parse_level(state.base_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    _set_cbar_range(0, (new_vmin, new_vmax))
    refresh_panels()
    update_timeseries_chart()
    ctrl.view_update()


@state.change("base_level")
def on_base_level_change(base_level, **kwargs):
    new_vmin, new_vmax = get_global_range(state.base_var, parse_level(base_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    _set_cbar_range(0, (new_vmin, new_vmax))
    refresh_panels()
    update_timeseries_chart()
    ctrl.view_update()


@state.change("base_cmap")
def on_base_cmap_change(base_cmap, **kwargs):
    panels["forecast"].set_cmap(base_cmap)
    cbar_luts[0].cmap = base_cmap
    # In "Truth" mode, panel 2 shows the same kind of physical field as panel 1
    # (not a diverging difference), so keep it visually consistent.
    if state.panel2_mode == "Truth":
        panels["diff"].set_cmap(base_cmap)
        cbar_luts[1].cmap = base_cmap
    ctrl.view_update()


@state.change("panel2_mode")
def on_panel2_mode_change(panel2_mode, **kwargs):
    if panel2_mode == "Truth":
        panels["diff"].set_title("Truth")
        panels["diff"].set_cmap(state.base_cmap)
        cbar_luts[1].cmap = state.base_cmap
    else:
        panels["diff"].set_title("Difference (Forecast − Truth)")
        panels["diff"].set_cmap("coolwarm")
        cbar_luts[1].cmap = "coolwarm"
    refresh_panels()
    ctrl.view_update()


@state.change("slice_cmap")
def on_slice_cmap_change(slice_cmap, **kwargs):
    panels["slice"].cmap = slice_cmap
    cbar_luts[2].cmap = slice_cmap
    update_vertical_slice()
    ctrl.view_update()


# ============================================================
# UI helpers
# ============================================================
def contour_controls():
    vuetify.VCheckbox(v_model=("c0_enabled", True), label="Enabled", density="compact", hide_details=True)
    with vuetify.VRow(no_gutters=True, classes="mt-1"):
        with vuetify.VCol(cols=7, classes="pr-1"):
            vuetify.VSelect(
                v_model=("c0_var", state.c0_var),
                items=("variable_items",),
                label="Variable",
                density="compact",
                hide_details=True,
            )
        with vuetify.VCol(cols=5):
            vuetify.VSelect(
                v_model=("c0_level", "default"),
                items=("level_items",),
                label="Level",
                density="compact",
                hide_details=True,
            )
    with vuetify.VRow(no_gutters=True, classes="mt-1"):
        with vuetify.VCol(cols=5, classes="pr-1"):
            vuetify.VTextField(
                v_model=("c0_interval", state.c0_interval),
                label="Interval",
                type="number",
                density="compact",
                hide_details=True,
            )
        with vuetify.VCol(cols=7):
            vuetify.VTextField(
                v_model=("c0_color", state.c0_color),
                label="Color",
                density="compact",
                hide_details=True,
            )
    vuetify.VSlider(
        v_model=("c0_line_width", state.c0_line_width),
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
    layout.title.set_text("Gen2 Forecast vs Truth")

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
                with vuetify.VCol(cols=3, classes="pa-2", style="overflow-y: auto; height: 100%;"):
                    with vuetify.VExpansionPanels(multiple=True, variant="accordion", v_model=("sf_panels", [0])):
                        with vuetify.VExpansionPanel(title="Variable"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VSelect(
                                    v_model=("base_var", DEFAULT_VAR),
                                    items=("variable_items",),
                                    label="Variable",
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify.VSelect(
                                    v_model=("base_level", "default"),
                                    items=("level_items",),
                                    label="Level",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("base_has_level",),
                                )
                                vuetify.VSelect(
                                    v_model=("base_cmap", "viridis"),
                                    items=("cmap_items",),
                                    label="Colormap",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VSelect(
                                    v_model=("panel2_mode", "Difference"),
                                    items=("panel2_mode_items",),
                                    label="Panel 2 shows",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )

                        with vuetify.VExpansionPanel(title="Contour"):
                            with vuetify.VExpansionPanelText():
                                contour_controls()

                        with vuetify.VExpansionPanel(title="Vertical Slice"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VCheckbox(
                                    v_model=("slice_panel_visible", False),
                                    label="Show vertical slice panel",
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify.VSelect(
                                    v_model=("slice_var", state.slice_var),
                                    items=("slice_variable_items",),
                                    label="Variable",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VSelect(
                                    v_model=("slice_cmap", "viridis"),
                                    items=("cmap_items",),
                                    label="Colormap",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VSlider(
                                    v_model=("slice_n_contours", 10),
                                    min=3,
                                    max=25,
                                    step=1,
                                    label="Number of contours",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    thumb_label=True,
                                )
                                vuetify.VSelect(
                                    v_model=("slice_orientation", "latitude"),
                                    items=("slice_orientation_items",),
                                    label="Slice orientation",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                )
                                vuetify.VSlider(
                                    v_model=("slice_lat_value", 0.0),
                                    min=float(lat.min()),
                                    max=float(lat.max()),
                                    step=1.0,
                                    label="Latitude",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("slice_orientation === 'latitude'",),
                                )
                                vuetify.VRangeSlider(
                                    v_model=("slice_lon_range", [float(lon.min()), float(lon.max())]),
                                    min=float(lon.min()),
                                    max=float(lon.max()),
                                    step=1.25,
                                    label="Longitude range",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("slice_orientation === 'latitude'",),
                                )
                                vuetify.VSlider(
                                    v_model=("slice_lon_value", 260.0),
                                    min=float(lon.min()),
                                    max=float(lon.max()),
                                    step=1.25,
                                    label="Longitude",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("slice_orientation === 'longitude'",),
                                )
                                vuetify.VRangeSlider(
                                    v_model=("slice_lat_range", [float(lat.min()), float(lat.max())]),
                                    min=float(lat.min()),
                                    max=float(lat.max()),
                                    step=1.0,
                                    label="Latitude range",
                                    density="compact",
                                    hide_details=True,
                                    classes="mt-1",
                                    v_show=("slice_orientation === 'longitude'",),
                                )

                        with vuetify.VExpansionPanel(title="Time Series (Panel 1 Variable)"):
                            with vuetify.VExpansionPanelText():
                                vuetify.VCheckbox(
                                    v_model=("ts_picking_enabled", False),
                                    label="Enable Map Clicking",
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify.VAlert(
                                    "Always plots truth vs. forecast for the Variable/Level selected "
                                    "above -- there is no separate selector for this panel.",
                                    density="compact",
                                    variant="tonal",
                                    type="info",
                                    classes="mb-1 text-caption",
                                )
                                vuetify.VChip(
                                    "Lat: {{ ts_lat.toFixed(2) }} | Lon: {{ ts_lon.toFixed(2) }}",
                                    classes="mt-2",
                                    size="small",
                                    color="primary",
                                )

                with vuetify.VCol(cols=9, classes="pa-0 fill-height"):
                    view = vtk_widgets.VtkRemoteView(
                        plotter.ren_win, style="width: 100%; height: 100%;", interactive_ratio=1
                    )
                    ctrl.view_update = view.update

# ============================================================
# Fit view to browser viewport on first connect
# ============================================================
# True once the user has manually panned/zoomed the map -- until then, every
# resize event re-fits from scratch (see _fit_map_camera_to_domain), since
# remote-view windows can report an intermediate size before settling on the
# browser's real viewport across several ConfigureEvents, and a one-shot fit
# (the previous approach) locks in whichever size arrived first. Once the user
# has taken over the camera, we stop re-fitting and just clamp their view to
# stay in bounds on further resizes.
_user_interacted = [False]


# <1.0 deliberately zooms in tighter than the strict "show the entire domain,
# no cropping" fit below -- the exact fit leaves a visibly loose margin in
# practice (likely device-pixel-ratio / remote-view scaling not reflected in
# GetSize()), so this trades a small, fixed amount of edge cropping for a
# panel that actually looks filled. Tune to taste.
_MAP_ZOOM_FACTOR = 2.2


def _fit_map_camera_to_domain():
    # Aspect of this renderer's own viewport, not the whole render window --
    # the colorbar strip eats vertical space asymmetrically, so per-quadrant
    # aspect differs slightly from the whole window's.
    rw, rh = plotter.renderers[0].GetSize()
    if rh <= 0:
        return
    aspect = rw / rh
    dom_hw = 0.5 * (lon_max - lon_min)
    dom_hh = 0.5 * (lat_max - lat_min)
    dom_cx = lon_min + dom_hw
    dom_cy = lat_min + dom_hh
    # Explicit renderer-0 camera, not plotter.camera (see _clamp_camera).
    camera = plotter.renderers[0].GetActiveCamera()
    camera.parallel_scale = max(dom_hh, dom_hw / aspect) * _MAP_ZOOM_FACTOR
    pz = camera.position[2]
    fz = camera.focal_point[2]
    camera.position = (dom_cx, dom_cy, pz)
    camera.focal_point = (dom_cx, dom_cy, fz)


def _on_window_configure(obj, event):
    w, h = obj.GetSize()
    if w < 2 or h < 2:
        return
    if not _user_interacted[0]:
        _fit_map_camera_to_domain()
    else:
        _clamp_camera()

    # Remote-view windows can report an intermediate size before settling on the
    # browser's real viewport across several ConfigureEvents, so refit on every
    # one (not just the first) -- both the slice panel's aspect-matching stretch
    # (_make_grid) and its reset_camera() need the final size to frame correctly.
    panels["slice"].view_initialized = False
    update_vertical_slice()

    ctrl.view_update()


plotter.ren_win.AddObserver("ConfigureEvent", _on_window_configure)

# Initial render
refresh_panels()
update_vertical_slice()
state.latency_text = "Ready"

if __name__ == "__main__":
    # server.start(port=8082, open_browser=True)
    server.start(port=8082, host="0.0.0.0", open_browser=False)
