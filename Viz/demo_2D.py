import time
import numpy as np
import pandas as pd
import xarray as xr
import pyvista as pv
from functools import lru_cache

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
    make_dashed_line_mesh,
    get_vertical_slice,
    MapPanel,
)

# ============================================================
# User settings
# ============================================================
FILE = "/Users/cbecker/PycharmProjects/CREDIT/era5_local_testing_data_onedeg_2021.nc"


# ============================================================
# MapPanel abstractions (app-specific panels below)
# ============================================================
class TimeSeriesPanel:
    def __init__(self, plotter, row, col, border_actor=None):  # <-- Accept border_actor
        self.plotter = plotter
        self.row = row
        self.col = col
        self.border_actor = border_actor

        self.plotter.subplot(row, col)
        self.renderer = self.plotter.renderer
        self.renderer.SetBackground(0.9, 0.9, 0.9)

        # Hide the border initially
        if self.border_actor:
            self.border_actor.SetVisibility(False)

        self.chart = pv.Chart2D()
        self.chart.visible = False
        self.plotter.add_chart(self.chart)
        self._time_line = None

        self.placeholder_actor = self.plotter.add_text(
            "Time Series Panel\n(Enable Map Clicking and select a point)",
            position=(0.5, 0.5),
            viewport=True,
            font_size=12,
            color="#444444",
        )
        self.placeholder_actor.GetTextProperty().SetJustificationToCentered()
        self.placeholder_actor.GetTextProperty().SetVerticalJustificationToCentered()
        self.has_data = False

    def update_chart(self, x_data, y_data, tick_locs, tick_labels, title, y_label):
        if not self.has_data:
            self.has_data = True
            self.renderer.SetBackground(1.0, 1.0, 1.0)
            self.placeholder_actor.SetVisibility(False)
            self.chart.visible = True

            # Show the border once data loads
            if self.border_actor:
                self.border_actor.SetVisibility(True)

        self.chart.clear()
        self.chart.line(x_data, y_data, color="blue", width=2.0)

        self.chart.x_axis.tick_locations = tick_locs
        self.chart.x_axis.tick_labels = tick_labels

        self.chart.title = title
        self.chart.y_axis.label = y_label
        self._time_line = None

    def update_time_indicator(self, t_idx, ymin, ymax):
        # Do not attempt to draw the line if the chart hasn't been initialized
        if not self.has_data:
            return

        if self._time_line is not None:
            try:
                self.chart.remove_plot(self._time_line)
            except ValueError:
                pass

        if ymin is not None and ymax is not None:
            self._time_line = self.chart.line([t_idx, t_idx], [ymin, ymax], color="red", width=2.0, style="--")


class VerticalSlicePanel:
    def __init__(
        self, plotter, row, col, title="Vertical Slice", cmap="viridis", border_actor=None
    ):  # <-- Accept border_actor
        self.plotter = plotter
        self.row = row
        self.col = col
        self.title = title
        self.cmap = cmap
        self.is_active = False
        self.border_actor = border_actor

        self.plotter.subplot(row, col)
        self.renderer = self.plotter.renderer
        self.renderer.SetBackground(0.9, 0.9, 0.9)

        # Hide the border initially
        if self.border_actor:
            self.border_actor.SetVisibility(False)

        self.grid = None
        self.actor = None
        self.view_initialized = False

        self.title_actor = self.plotter.add_text(
            title, position="upper_left", font_size=9, color="black", name=f"title_{row}_{col}"
        )
        self.title_actor.SetVisibility(False)

        self.placeholder_actor = self.plotter.add_text(
            "Vertical Slice Panel\n(Enable in sidebar)",
            position=(0.5, 0.5),
            viewport=True,
            font_size=12,
            color="#444444",
        )
        self.placeholder_actor.GetTextProperty().SetJustificationToCentered()
        self.placeholder_actor.GetTextProperty().SetVerticalJustificationToCentered()

    def toggle_visibility(self, visible):
        self.is_active = visible

        # Toggle border visibility
        if self.border_actor:
            self.border_actor.SetVisibility(visible)

        if visible:
            self.renderer.SetBackground(1.0, 1.0, 1.0)
            self.placeholder_actor.SetVisibility(False)
            self.title_actor.SetVisibility(True)
            if self.actor is not None:
                self.actor.SetVisibility(True)
        else:
            self.renderer.SetBackground(0.9, 0.9, 0.9)
            self.placeholder_actor.SetVisibility(True)
            self.title_actor.SetVisibility(False)
            if self.actor is not None:
                self.actor.SetVisibility(False)

    def _make_grid(self, x, levels):
        X, Y = np.meshgrid(x, levels, indexing="ij")
        Z = np.zeros_like(X, dtype=np.float32)
        return pv.StructuredGrid(X.astype(np.float32), Y.astype(np.float32), Z)

    def set_slice(self, x, levels, arr, clim=None, title=None):
        self.plotter.subplot(self.row, self.col)
        if clim is None:
            clim = (
                float(np.nanpercentile(arr, 1)),
                float(np.nanpercentile(arr, 99)),
            )

        self.grid = self._make_grid(x, levels)
        self.grid.point_data["slice_field"] = arr.ravel(order="F")

        if self.actor is not None:
            self.plotter.remove_actor(self.actor)

        self.actor = self.plotter.add_mesh(
            self.grid,
            scalars="slice_field",
            cmap=self.cmap,
            clim=clim,
            show_edges=False,
            lighting=False,
            show_scalar_bar=False,
        )
        self.actor.mapper.scalar_range = clim

        # Ensure new mesh respects current panel visibility
        self.actor.SetVisibility(self.is_active)

        if title is not None:
            self.title_actor = self.plotter.add_text(
                title, position="upper_left", font_size=9, color="black", name=f"title_{self.row}_{self.col}"
            )
            self.title_actor.SetVisibility(self.is_active)

        if not self.view_initialized:
            self.plotter.view_xy()
            self.plotter.camera.parallel_projection = True
            self.plotter.reset_camera()
            self.view_initialized = True
        self.save_camera_state()

    def save_camera_state(self):
        self.plotter.subplot(self.row, self.col)
        cam = self.plotter.camera
        self._locked_camera_state = {
            "position": cam.position,
            "focal_point": cam.focal_point,
            "up": cam.up,
            "parallel_scale": cam.parallel_scale,
        }

    def restore_camera_state(self):
        if not hasattr(self, "_locked_camera_state"):
            return
        self.plotter.subplot(self.row, self.col)
        cam = self.plotter.camera
        s = self._locked_camera_state
        cam.position = s["position"]
        cam.focal_point = s["focal_point"]
        cam.up = s["up"]
        cam.parallel_scale = s["parallel_scale"]


# ============================================================
# Load data & format times
# ============================================================
ds = xr.open_dataset(FILE)
variable_items = list(ds.data_vars)
slice_variable_items = [v for v in ds.data_vars if "level" in ds[v].dims]
lon = ds["longitude"].values.astype(np.float32)
lat = ds["latitude"].values.astype(np.float32)

times = ds["time"].values
nt = ds.sizes["time"]
time_strings = pd.to_datetime(times).strftime("%Y-%m-%d %H:%M").tolist()
ts_x_data = np.arange(nt, dtype=np.float32)

level_items = ["default"] + [str(int(v)) for v in ds["level"].values]

DEFAULT_BASE_VAR = "t2m"
DEFAULT_BASE_LEVEL = "default"

base0 = get_2d_field(
    ds,
    DEFAULT_BASE_VAR,
    time_idx=0,
    level=parse_level(DEFAULT_BASE_LEVEL),
)


# ============================================================
# Global range cache (Optimized)
# ============================================================
@lru_cache(maxsize=32)
def get_global_range(var, level):
    da = ds[var]
    if "level" in da.dims:
        da = da.isel(level=-1) if level is None else da.sel(level=level)
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


@lru_cache(maxsize=32)
def get_slice_global_range(var):
    da = ds[var]
    return (float(da.quantile(0.01).values), float(da.quantile(0.99).values))


vmin, vmax = get_global_range(DEFAULT_BASE_VAR, parse_level(DEFAULT_BASE_LEVEL))
base_clim = (vmin, vmax)

# ============================================================
# Coastline texture
# ============================================================
coast_img, extent = make_cartopy_coastline_texture(
    lon,
    lat,
    resolution="50m",
    dpi=400,
    linewidth=0.8,
)
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

# Explicitly create borders for all 4 panels so we can control them dynamically
border_actors = [r.add_border(color="black", width=10.0) for r in plotter.renderers]

panels = {
    "single_main": MapPanel(
        plotter,
        0,
        0,
        lon,
        lat,
        coast_texture,
        coast_plane,
        title="Single Forecast",
        cmap="viridis",
        show_scalar_bar=False,
    ),
    "single_aux": MapPanel(
        plotter,
        0,
        1,
        lon,
        lat,
        coast_texture,
        coast_plane,
        title="Aux / Future Panel",
        cmap="coolwarm",
        show_scalar_bar=False,
    ),
    "single_bottom_left": VerticalSlicePanel(
        plotter,
        1,
        0,
        title="Vertical Slice",
        cmap="viridis",
        border_actor=border_actors[2],  # <-- Passed here
    ),
    "ts_panel": TimeSeriesPanel(
        plotter,
        1,
        1,
        border_actor=border_actors[3],  # <-- Passed here
    ),
}

visible_single_panels = [panels["single_main"], panels["single_aux"]]
slice_panel = panels["single_bottom_left"]
ts_panel = panels["ts_panel"]


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


for panel in visible_single_panels:
    panel.add_base(base0, clim=base_clim, opacity=0.95)

# Setup initial 2D views for maps and slice
for r, c in [(0, 0), (0, 1), (1, 0)]:
    plotter.subplot(r, c)
    plotter.view_xy()
    plotter.camera.parallel_projection = True

# Share cameras only among map panels.
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
# Colorbar renderer strips (Updated for 3 data panels)
# ============================================================
# ============================================================
# Colorbar renderer strips (Updated for 3 data panels)
# ============================================================
CBAR_STRIP = 0.06

_data_vp = [
    (0.0, 0.5 + CBAR_STRIP, 0.5, 1.0),  # Top Left
    (0.5, 0.5 + CBAR_STRIP, 1.0, 1.0),  # Top Right
    (0.0, 0.0 + CBAR_STRIP, 0.5, 0.5),  # Bottom Left (Slice)
]
_cbar_vp = [
    (0.0, 0.5, 0.5, 0.5 + CBAR_STRIP),
    (0.5, 0.5, 1.0, 0.5 + CBAR_STRIP),
    (0.0, 0.0, 0.5, 0.0 + CBAR_STRIP),  # Make sure this 3rd line is here!
]

# Ensure Renderer 3 (TimeSeries Chart) takes full quadrant 4 space
plotter.renderers[3].SetViewport(0.5, 0.0, 1.0, 0.5)

# Make sure slice_panel.cmap is the 3rd item here!
_panel_cmaps = [panels["single_main"].cmap, panels["single_aux"].cmap, slice_panel.cmap]

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
    sb.SetLabelFormat("%.0f")
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
# plotter.renderers[2].DrawOff()
# cbar_renderers[2].DrawOff()
cbar_renderers[2].SetBackground(0.9, 0.9, 0.9)
cbar_actors[2].SetVisibility(False)


# ============================================================
# Contour cache (Optimized)
# ============================================================
def get_contour_preset_from_state(state, slot):
    return {
        "enabled": bool(getattr(state, f"c{slot}_enabled")),
        "var": getattr(state, f"c{slot}_var"),
        "level": parse_level(getattr(state, f"c{slot}_level")),
        "interval": float(getattr(state, f"c{slot}_interval")),
        "color": getattr(state, f"c{slot}_color"),
        "line_width": float(getattr(state, f"c{slot}_line_width")),
    }


@lru_cache(maxsize=128)
def get_cached_contour(var, time_idx, level, interval, stride=1):
    return make_contours(
        ds=ds,
        var=var,
        time_idx=time_idx,
        level=level,
        interval=interval,
        stride=stride,
        lon=lon,
        lat=lat,
    )


def update_contours_for_panels(time_idx, state, target_panels):
    t0 = time.perf_counter()

    for slot in range(2):
        preset = get_contour_preset_from_state(state, slot)

        if (not preset["enabled"]) or preset["interval"] <= 0:
            for panel in target_panels:
                panel.remove_contour(slot)
            continue

        contour_mesh = get_cached_contour(
            var=preset["var"], time_idx=time_idx, level=preset["level"], interval=preset["interval"], stride=1
        )

        for panel in target_panels:
            panel.set_contour(
                slot=slot, contour_mesh=contour_mesh, color=preset["color"], line_width=preset["line_width"]
            )

    return 1000 * (time.perf_counter() - t0)


# ============================================================
# Trame app initialization
# ============================================================
server = get_server()
state, ctrl = server.state, server.controller

state.variable_items = variable_items
state.level_items = level_items

state.base_var = DEFAULT_BASE_VAR
state.base_has_level = has_level(ds, DEFAULT_BASE_VAR)
state.base_level = DEFAULT_BASE_LEVEL
state.opacity = 0.95
state.t_index = 0
state.active_tab = "single_forecast"

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

state.sf_panels = []  # Keep some panels open by default
state.time_text = str(times[0])
state.latency_text = "Ready"

state.base_vmin = vmin
state.base_vmax = vmax

state.slice_variable_items = slice_variable_items
state.slice_var = "T" if "T" in slice_variable_items else (slice_variable_items[0] if slice_variable_items else "")
state.slice_orientation = "latitude"
state.slice_lat_value = 0.0
state.slice_lon_value = 260.0
state.slice_orientation_items = ["latitude", "longitude"]
state.slice_panel_visible = False

# Time Series States
state.ts_picking_enabled = False
state.ts_point_picked = False
state.ts_lon = float(lon.min() + (lon.max() - lon.min()) / 2)
state.ts_lat = float(lat.min() + (lat.max() - lat.min()) / 2)
state.ts_var = "t2m"

state.ts_has_level = has_level(ds, state.ts_var)
state.ts_level = "default"
state.ts_ymin = 0.0
state.ts_ymax = 1.0

# ============================================================
# Interactor / Camera helpers
# ============================================================
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


def _update_slice_from_camera():
    fp = _shared_map_cam.GetFocalPoint()
    if state.slice_orientation == "latitude":
        state.slice_lat_value = float(np.clip(fp[1], float(lat.min()), float(lat.max())))
    else:
        state.slice_lon_value = float(np.clip(fp[0], float(lon.min()), float(lon.max())))


_slice_line_actors = {k: None for k in ["single_main", "single_aux"]}


def _update_slice_lines():
    for key in ["single_main", "single_aux"]:
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
            mesh = make_dashed_line_mesh(lon_min, lat_val, lon_max, lat_val, z=5.0)
        else:
            lon_val = float(state.slice_lon_value)
            mesh = make_dashed_line_mesh(lon_val, lat_min, lon_val, lat_max, z=5.0)

        actor = plotter.add_mesh(mesh, color="black", line_width=2.0, lighting=False, pickable=False)
        _slice_line_actors[key] = actor


# --- Interactor Observers ---
def _on_scroll_forward(obj, event):
    if event_is_over_ts_chart():
        return
    if event_is_over_vertical_slice():
        slice_panel.restore_camera_state()
        ctrl.view_update()
        return
    _zoom_at_cursor(1)
    _update_slice_from_camera()


def _on_scroll_backward(obj, event):
    if event_is_over_ts_chart():
        return
    if event_is_over_vertical_slice():
        slice_panel.restore_camera_state()
        ctrl.view_update()
        return
    _zoom_at_cursor(-1)
    _update_slice_from_camera()


def _on_pan_end(obj, event):
    if event_is_over_vertical_slice():
        slice_panel.restore_camera_state()
    elif not event_is_over_ts_chart():
        _clamp_camera()
        _update_slice_from_camera()
    ctrl.view_update()


def _on_interaction(obj, event):
    if event_is_over_vertical_slice():
        slice_panel.restore_camera_state()


def _on_left_click(obj, event):
    if not state.ts_picking_enabled:
        return

    interactor = plotter.iren.interactor
    mx, my = interactor.GetEventPosition()
    renderer = interactor.FindPokedRenderer(mx, my)

    # Make sure we only click on map panels
    if renderer not in [plotter.renderers[0], plotter.renderers[1]]:
        return

    renderer.SetDisplayPoint(mx, my, 0)
    renderer.DisplayToWorld()
    w = renderer.GetWorldPoint()
    wx, wy = w[0] / w[3], w[1] / w[3]

    state.ts_lon = float(np.clip(wx, float(lon.min()), float(lon.max())))
    state.ts_lat = float(np.clip(wy, float(lat.min()), float(lat.max())))
    state.ts_point_picked = True
    # Force Trame to immediately execute any @state.change decorators
    # tied to ts_lon and ts_lat without waiting for a UI interaction.
    state.flush()

    ctrl.view_update()


plotter.iren.interactor.AddObserver("MouseWheelForwardEvent", _on_scroll_forward, 1.0)
plotter.iren.interactor.AddObserver("MouseWheelBackwardEvent", _on_scroll_backward, 1.0)
plotter.iren.interactor.AddObserver("InteractionEvent", _on_interaction, 1.0)
plotter.iren.interactor.AddObserver("EndInteractionEvent", _on_pan_end)
plotter.iren.interactor.AddObserver("LeftButtonPressEvent", _on_left_click, 1.0)


# ============================================================
# Update callbacks
# ============================================================
def update_vertical_slice():
    if not slice_variable_items:
        return
    x, levels, arr, title = get_vertical_slice(
        ds=ds,
        var=state.slice_var,
        time_idx=int(state.t_index),
        orientation=state.slice_orientation,
        lat_value=float(state.slice_lat_value),
        lon_value=float(state.slice_lon_value),
    )
    vmin_s, vmax_s = get_slice_global_range(state.slice_var)
    slice_panel.set_slice(x=x, levels=levels, arr=arr, clim=(vmin_s, vmax_s), title=title)
    cbar_luts[2].scalar_range = (vmin_s, vmax_s)


@state.change("slice_var", "slice_orientation", "slice_lat_value", "slice_lon_value")
def update_slice_selection(**kwargs):
    update_vertical_slice()
    _update_slice_lines()
    ctrl.view_update()


@state.change("slice_panel_visible")
def on_slice_panel_visible(slice_panel_visible, **kwargs):
    slice_panel.toggle_visibility(slice_panel_visible)

    if slice_panel_visible:
        cbar_renderers[2].SetBackground(1.0, 1.0, 1.0)
        cbar_actors[2].SetVisibility(True)
    else:
        cbar_renderers[2].SetBackground(0.9, 0.9, 0.9)
        cbar_actors[2].SetVisibility(False)

    _update_slice_lines()
    ctrl.view_update()


# --- Time Series Updates ---
_marker_actors = {0: None, 1: None}


@state.change("ts_lon", "ts_lat", "ts_picking_enabled")
def update_map_markers(**kwargs):
    for idx, key in enumerate(["single_main", "single_aux"]):
        panel = panels[key]
        plotter.subplot(panel.row, panel.col)
        if _marker_actors[idx] is not None:
            plotter.remove_actor(_marker_actors[idx])
            _marker_actors[idx] = None

        if state.ts_picking_enabled:
            # marker = pv.Sphere(center=(state.ts_lon, state.ts_lat, 5.0), radius=1.5)
            marker = make_star_mesh(state.ts_lon, state.ts_lat, radius=1.5, z=5.0)
            actor = plotter.add_mesh(marker, color="black", lighting=False, pickable=False)
            _marker_actors[idx] = actor
    ctrl.view_update()


@state.change("ts_lon", "ts_lat", "ts_var", "ts_level")  # <-- Added ts_level here
def update_timeseries_chart(**kwargs):
    if not hasattr(state, "ts_var") or state.ts_var not in ds:
        return
    if getattr(state, "ts_point_picked", False) is False:
        return
    da = ds[state.ts_var]

    # Slice by level if the variable has a level dimension
    lvl = None
    if "level" in da.dims:
        lvl = parse_level(getattr(state, "ts_level", "default"))
        if lvl is None:
            da = da.isel(level=-1)
        else:
            da = da.sel(level=lvl)

    # Extract the 1D time series for the selected lat/lon
    da_point = da.sel(longitude=state.ts_lon, latitude=state.ts_lat, method="nearest")
    y_data = da_point.values.astype(np.float32)

    y_min, y_max = float(np.nanmin(y_data)), float(np.nanmax(y_data))
    state.ts_ymin, state.ts_ymax = y_min, y_max

    tick_locs = np.linspace(0, nt - 1, 5, dtype=int)
    tick_labels = [time_strings[i] for i in tick_locs]

    # Update title to include level info if applicable
    lvl_str = f" (Level: {lvl})" if lvl is not None else ""
    title = f"{state.ts_var}{lvl_str} at {state.ts_lat:.2f}°N, {state.ts_lon:.2f}°E"

    ts_panel.update_chart(ts_x_data, y_data, tick_locs, tick_labels, title, state.ts_var)
    ts_panel.update_time_indicator(int(getattr(state, "t_index", 0)), state.ts_ymin, state.ts_ymax)
    ctrl.view_update()


def update_base_field():
    arr = get_2d_field(ds, state.base_var, time_idx=int(state.t_index), level=parse_level(state.base_level))
    clim = (state.base_vmin, state.base_vmax)
    for panel in visible_single_panels:
        panel.update_base(arr, clim=clim)


@state.change("t_index")
def update_time(t_index, **kwargs):
    i = int(round(t_index))
    state.t_index = i

    t0 = time.perf_counter()
    update_base_field()
    update_vertical_slice()
    ts_panel.update_time_indicator(i, state.ts_ymin, state.ts_ymax)
    t1 = time.perf_counter()

    contour_ms = update_contours_for_panels(i, state, visible_single_panels)
    t2 = time.perf_counter()

    ctrl.view_update()
    t3 = time.perf_counter()

    state.time_text = str(time_strings[i])
    state.latency_text = (
        f"t={i:03d} | "
        f"base={(t1 - t0) * 1000:.1f} ms | "
        f"contours={contour_ms:.1f} ms | "
        f"view={(t3 - t2) * 1000:.1f} ms | "
        f"cache={get_cached_contour.cache_info().currsize}"
    )


@state.change("base_var")
def on_base_var_change(base_var, **kwargs):
    has_lev = has_level(ds, base_var)
    state.base_has_level = has_lev
    if not has_lev:
        state.base_level = "default"
    elif state.base_level in (None, "default") and level_items:
        state.base_level = level_items[1]
    new_vmin, new_vmax = get_global_range(base_var, parse_level(state.base_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    for idx in [0, 1]:
        cbar_luts[idx].scalar_range = (new_vmin, new_vmax)
    update_base_field()
    ctrl.view_update()


@state.change("base_level")
def on_base_level_change(base_level, **kwargs):
    new_vmin, new_vmax = get_global_range(state.base_var, parse_level(base_level))
    state.base_vmin, state.base_vmax = new_vmin, new_vmax
    for idx in [0, 1]:
        cbar_luts[idx].scalar_range = (new_vmin, new_vmax)
    update_base_field()
    ctrl.view_update()


@state.change("ts_var")
def on_ts_var_change(ts_var, **kwargs):
    has_lev = has_level(ds, ts_var)
    state.ts_has_level = has_lev

    if not has_lev:
        state.ts_level = "default"
    elif state.ts_level in (None, "default") and level_items:
        state.ts_level = level_items[1]


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
def update_contour_selection(**kwargs):
    contour_ms = update_contours_for_panels(int(state.t_index), state, visible_single_panels)
    state.latency_text = f"Contour update: {contour_ms:.1f} ms | cache={get_cached_contour.cache_info().currsize}"
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
    layout.title.set_text("2D Earth-system viewer")

    with layout.toolbar:
        with vuetify.VTabs(v_model=("active_tab", "single_forecast"), density="compact", classes="mr-4"):
            vuetify.VTab("Single Forecast", value="single_forecast")
            vuetify.VTab("Compare Forecasts", value="compare_forecasts")
            vuetify.VTab("Perturb & Run", value="perturb_and_run")

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
                    with vuetify.VContainer(v_show=("active_tab === 'single_forecast'",), fluid=True, classes="pa-0"):
                        with vuetify.VExpansionPanels(multiple=True, variant="accordion", v_model=("sf_panels",)):
                            with vuetify.VExpansionPanel(title="Base Field"):
                                with vuetify.VExpansionPanelText():
                                    vuetify.VSelect(
                                        v_model=("base_var", DEFAULT_BASE_VAR),
                                        items=("variable_items",),
                                        label="Variable",
                                        density="compact",
                                        hide_details=True,
                                    )
                                    vuetify.VSelect(
                                        v_model=("base_level", DEFAULT_BASE_LEVEL),
                                        items=("level_items",),
                                        label="Level",
                                        density="compact",
                                        hide_details=True,
                                        classes="mt-1",
                                        v_show=("base_has_level",),
                                    )
                                    # vuetify.VSlider(v_model=("opacity", 0.95), min=0.0, max=1.0, step=0.05,
                                    #                 label="Opacity", density="compact", hide_details=True,
                                    #                 classes="mt-1")

                            with vuetify.VExpansionPanel(title="Contour 1"):
                                with vuetify.VExpansionPanelText():
                                    contour_controls(0)
                            with vuetify.VExpansionPanel(title="Contour 2"):
                                with vuetify.VExpansionPanelText():
                                    contour_controls(1)

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

                            with vuetify.VExpansionPanel(title="Time Series"):
                                with vuetify.VExpansionPanelText():
                                    vuetify.VCheckbox(
                                        v_model=("ts_picking_enabled", False),
                                        label="Enable Map Clicking",
                                        density="compact",
                                        hide_details=True,
                                    )
                                    vuetify.VSelect(
                                        v_model=("ts_var", state.ts_var),
                                        items=("variable_items",),
                                        label="Variable",
                                        density="compact",
                                        hide_details=True,
                                        classes="mt-1",
                                    )
                                    vuetify.VSelect(
                                        v_model=("ts_level", "default"),
                                        items=("level_items",),
                                        label="Level",
                                        density="compact",
                                        hide_details=True,
                                        classes="mt-1",
                                        v_show=("ts_has_level",),
                                    )
                                    vuetify.VChip(
                                        "Lat: {{ ts_lat.toFixed(2) }} | Lon: {{ ts_lon.toFixed(2) }}",
                                        classes="mt-2",
                                        size="small",
                                        color="primary",
                                    )

                    with vuetify.VContainer(v_show=("active_tab === 'compare_forecasts'",), fluid=True, classes="pa-0"):
                        vuetify.VLabel("Compare Forecasts — coming soon", classes="text-caption")
                    with vuetify.VContainer(v_show=("active_tab === 'perturb_and_run'",), fluid=True, classes="pa-0"):
                        vuetify.VLabel("Perturb & Run — coming soon", classes="text-caption")

                with vuetify.VCol(cols=9, classes="pa-0 fill-height"):
                    view = vtk_widgets.VtkRemoteView(
                        plotter.ren_win, style="width: 100%; height: 100%;", interactive_ratio=1
                    )
                    ctrl.view_update = view.update

# ============================================================
# Fit view to browser viewport on first connect
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

# Initial computations
initial_contour_ms = update_contours_for_panels(0, state, visible_single_panels)
update_vertical_slice()
# update_timeseries_chart()
state.latency_text = f"Initial contour build: {initial_contour_ms:.1f} ms"

if __name__ == "__main__":
    server.start(port=8080, open_browser=True)
