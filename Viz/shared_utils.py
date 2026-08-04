import math

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, MultiLineString
from vtkmodules.vtkRenderingCore import vtkTextActor

BASE_ARRAY_NAME = "base_field"


def pick_colorbar_label_format(vmin, vmax):
    """Choose a printf-style tick-label format (for vtkScalarBarActor.SetLabelFormat) sized to
    the value range, so ticks stay readable across wildly different variable magnitudes.

    A fixed decimal count either rounds small-magnitude variables to "0.0" (e.g. Qtot ~1e-3) or,
    if given enough decimals to show those, forces scientific notation on everyday-sized ones
    (TREFHT ~300, PS ~1e5) -- %g's rule is "scientific once the exponent reaches the requested
    precision," which normal-sized numbers hit immediately at low precision. Instead: fixed
    decimal notation with a magnitude-derived decimal count for normal ranges, and scientific
    notation only once decimals alone would need more than ~5 places to show anything.
    """
    span = abs(vmax - vmin)
    scale = span if span > 0 else max(abs(vmin), abs(vmax))
    if not np.isfinite(scale) or scale == 0:
        return "%.2f"

    magnitude = math.floor(math.log10(scale))
    if magnitude < -3:
        return "%.1e"
    decimals = min(max(0, 2 - magnitude), 6)
    return f"%.{decimals}f"


# ============================================================
# Data helpers
# ============================================================
def parse_level(level_value):
    if level_value in (None, "default", "__default__"):
        return None
    return int(level_value)


def has_level(ds, var):
    return "level" in ds[var].dims


def get_2d_field(ds, var, time_idx, level=None):
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=time_idx)
    if "level" in da.dims:
        if level is None:
            da = da.isel(level=-1)
        else:
            da = da.sel(level=level)
    return da.transpose("longitude", "latitude").values.astype(np.float32)


# ============================================================
# Geometry helpers
# ============================================================
def make_surface_grid(lon, lat, z_value=0.0):
    x, y = np.meshgrid(lon, lat, indexing="ij")
    z = np.full_like(x, z_value, dtype=np.float32)
    return pv.StructuredGrid(x.astype(np.float32), y.astype(np.float32), z)


def make_star_mesh(cx, cy, radius=1.5, z=5.0):
    pts = [[cx, cy, z]]
    inner_radius = radius * 0.382
    for i in range(10):
        angle = (i * np.pi / 5) + (np.pi / 2)
        r = radius if i % 2 == 0 else inner_radius
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle), z])
    pts = np.array(pts, dtype=np.float32)
    faces = []
    for i in range(1, 11):
        next_i = i + 1 if i < 10 else 1
        faces.extend([3, 0, i, next_i])
    return pv.PolyData(pts, np.array(faces, dtype=np.int32))


def make_cartopy_coastline_texture(lon, lat, resolution="50m", dpi=400, linewidth=0.4):
    dx = float(lon[1] - lon[0])
    dy = abs(float(lat[1] - lat[0]))
    lon_min = float(lon.min() - dx / 2)
    lon_max = float(lon.max() + dx / 2)
    lat_min = float(lat.min() - dy / 2)
    lat_max = float(lat.max() + dy / 2)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    shp = shpreader.natural_earth(resolution=resolution, category="physical", name="coastline")
    reader = shpreader.Reader(shp)

    for geom in reader.geometries():
        if isinstance(geom, LineString):
            line_geoms = [geom]
        elif isinstance(geom, MultiLineString):
            line_geoms = list(geom.geoms)
        else:
            continue
        for line in line_geoms:
            coords = np.asarray(line.coords, dtype=np.float32)
            if coords.shape[0] < 2:
                continue
            x = np.mod(coords[:, 0], 360.0)
            y = coords[:, 1]
            jumps = np.where(np.abs(np.diff(x)) > 180)[0] + 1
            segments = np.split(np.arange(len(x)), jumps)
            for seg in segments:
                if len(seg) < 2:
                    continue
                ax.plot(
                    x[seg],
                    y[seg],
                    color="black",
                    linewidth=linewidth,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba, (lon_min, lon_max, lat_min, lat_max)


def make_coast_plane(extent, z_value=4.0):
    lon_min, lon_max, lat_min, lat_max = extent
    return pv.Plane(
        center=(0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max), z_value),
        direction=(0, 0, 1),
        i_size=(lon_max - lon_min),
        j_size=(lat_max - lat_min),
        i_resolution=1,
        j_resolution=1,
    )


def make_contours(ds, var, time_idx, level, interval, stride, lon, lat):
    arr = get_2d_field(ds, var, time_idx, level=level)
    if stride > 1:
        arr = arr[::stride, ::stride]
        lon_use = lon[::stride]
        lat_use = lat[::stride]
    else:
        lon_use = lon
        lat_use = lat

    cgrid = make_surface_grid(lon_use, lat_use, z_value=2.0)
    cgrid.point_data["contour_field"] = arr.ravel(order="F")

    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    if not np.isfinite(amin) or not np.isfinite(amax) or amin == amax:
        return pv.PolyData()

    start = np.floor(amin / interval) * interval
    stop = np.ceil(amax / interval) * interval
    levels = np.arange(start, stop + interval, interval, dtype=np.float32)
    return cgrid.contour(isosurfaces=levels, scalars="contour_field")


def make_dashed_line_mesh(x0, y0, x1, y1, z=5.0, n_dashes=40):
    pts = []
    for i in range(n_dashes):
        t0 = i / n_dashes
        t1 = (i + 0.5) / n_dashes
        pts.append([x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0), z])
        pts.append([x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0), z])
    pts = np.array(pts, dtype=np.float32)
    cells = np.zeros(n_dashes * 3, dtype=np.int32)
    cells[0::3] = 2
    cells[1::3] = np.arange(0, n_dashes * 2, 2)
    cells[2::3] = np.arange(1, n_dashes * 2, 2)
    mesh = pv.PolyData(pts)
    mesh.lines = cells
    return mesh


def get_vertical_slice(ds, var, time_idx, orientation, lat_value, lon_value):
    da = ds[var]
    if "level" not in da.dims:
        raise ValueError(f"{var} does not have a level dimension")
    da = da.isel(time=time_idx)
    if orientation == "latitude":
        da_slice = da.sel(latitude=lat_value, method="nearest")
        arr = da_slice.transpose("longitude", "level").values.astype(np.float32)
        x = ds["longitude"].values.astype(np.float32)
        title = f"{var} vertical slice at lat={float(da_slice.latitude.values):.2f}"
    elif orientation == "longitude":
        da_slice = da.sel(longitude=lon_value, method="nearest")
        arr = da_slice.transpose("latitude", "level").values.astype(np.float32)
        x = ds["latitude"].values.astype(np.float32)
        title = f"{var} vertical slice at lon={float(da_slice.longitude.values):.2f}"
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    levels = ds["level"].values.astype(np.float32)
    return x, levels, arr, title


def _make_title_actor2d(renderer, title, font_size=16):
    """Build a vtkTextActor pinned to a fixed-viewport renderer's left-center.

    Shared by MapPanel and VerticalSlicePanel's title_renderer option: a title drawn in its own
    separate, non-interactive renderer (set up by the app alongside the colorbar strips) instead
    of overlaid inside the data renderer, so panning/zooming the data camera can never scroll
    the field behind the label or shrink the data window below it.
    """
    actor = vtkTextActor()
    tp = actor.GetTextProperty()
    tp.SetFontSize(font_size)
    tp.SetColor(0.0, 0.0, 0.0)
    tp.SetJustificationToLeft()
    tp.SetVerticalJustificationToCentered()
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.02, 0.5)
    actor.SetInput(title)
    renderer.AddActor2D(actor)
    return actor


# ============================================================
# MapPanel
# ============================================================
class MapPanel:
    def __init__(
        self,
        plotter,
        row,
        col,
        lon,
        lat,
        coast_texture,
        coast_plane,
        title,
        cmap="viridis",
        show_scalar_bar=True,
        title_renderer=None,
    ):
        self.plotter = plotter
        self.row = row
        self.col = col
        self.lon = lon
        self.lat = lat
        self.title = title
        self.cmap = cmap
        self.show_scalar_bar = show_scalar_bar
        # Optional separate, fixed-viewport renderer for the title (set up by the app alongside
        # the colorbar strips -- see demo_gen2_compare.py). When given, the label lives outside
        # the data renderer entirely, so panning/zooming the data camera can never scroll the
        # field behind it or shrink the data window below it. When None (the default), falls
        # back to the original in-panel overlay text, unchanged.
        self.title_renderer = title_renderer
        self._title_actor2d = None

        self.grid = make_surface_grid(lon, lat, z_value=0.0)
        self.base_actor = None
        self.coast_actor = None
        self.contour_actors = {}

        self.plotter.subplot(row, col)
        self.coast_actor = self.plotter.add_mesh(coast_plane, texture=coast_texture, lighting=False, pickable=False)
        self.set_title(title)

    def set_title(self, title):
        self.title = title
        if self.title_renderer is not None:
            if self._title_actor2d is None:
                self._title_actor2d = _make_title_actor2d(self.title_renderer, title)
            else:
                self._title_actor2d.SetInput(title)
        else:
            self.plotter.subplot(self.row, self.col)
            # Same name= every call -- replaces the existing title actor in place.
            self.plotter.add_text(
                title, position="upper_left", font_size=9, color="black", name=f"title_{self.row}_{self.col}"
            )

    def add_base(self, arr, clim, opacity=0.95):
        self.plotter.subplot(self.row, self.col)
        self.grid.point_data[BASE_ARRAY_NAME] = arr.ravel(order="F")
        scalar_bar_args = {"title": self.title if self.show_scalar_bar else "", "vertical": True}
        self.base_actor = self.plotter.add_mesh(
            self.grid,
            scalars=BASE_ARRAY_NAME,
            cmap=self.cmap,
            clim=clim,
            opacity=opacity,
            show_edges=False,
            lighting=False,
            show_scalar_bar=self.show_scalar_bar,
            scalar_bar_args=scalar_bar_args if self.show_scalar_bar else None,
        )

    def update_base(self, arr, clim=None):
        arr_flat = arr.ravel(order="F")
        self.grid.point_data[BASE_ARRAY_NAME][:] = arr_flat
        if self.base_actor is not None and clim is not None:
            self.base_actor.mapper.scalar_range = clim

    def set_cmap(self, cmap):
        self.cmap = cmap
        if self.base_actor is not None:
            lut = pv.LookupTable(cmap=cmap, n_values=256)
            lut.scalar_range = self.base_actor.mapper.scalar_range
            self.base_actor.mapper.lookup_table = lut

    def remove_contour(self, slot):
        actor = self.contour_actors.get(slot)
        if actor is not None:
            self.plotter.remove_actor(actor)
        self.contour_actors[slot] = None

    def set_contour(self, slot, contour_mesh, color, line_width):
        old_actor = self.contour_actors.get(slot)
        if old_actor is not None:
            self.plotter.remove_actor(old_actor)
        self.plotter.subplot(self.row, self.col)
        actor = self.plotter.add_mesh(
            contour_mesh,
            color=color,
            line_width=line_width,
            render_lines_as_tubes=False,
            lighting=False,
            pickable=False,
        )
        self.contour_actors[slot] = actor


# ============================================================
# TimeSeriesPanel
# ============================================================
class TimeSeriesPanel:
    def __init__(self, plotter, row, col, border_actor=None, title_renderer=None):  # <-- Accept border_actor
        self.plotter = plotter
        self.row = row
        self.col = col
        self.border_actor = border_actor
        # See MapPanel's title_renderer. pv.Chart2D's own .title normally reserves its own
        # layout space (unlike MapPanel/VerticalSlicePanel's raw text overlay), but when a
        # title_renderer is given we still route through it for visual consistency with the
        # other panels.
        self.title_renderer = title_renderer
        self.title = None
        self._title_actor2d = None

        self.plotter.subplot(row, col)
        self.renderer = self.plotter.renderer
        self.renderer.SetBackground(0.9, 0.9, 0.9)

        # Hide the border initially
        if self.border_actor:
            self.border_actor.SetVisibility(False)

        if self.title_renderer is not None:
            self.title_renderer.SetBackground(0.9, 0.9, 0.9)

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

    def _draw_title(self, title):
        self.title = title
        if self.title_renderer is not None:
            if self._title_actor2d is None:
                self._title_actor2d = _make_title_actor2d(self.title_renderer, title)
            else:
                self._title_actor2d.SetInput(title)
        else:
            self.chart.title = title

    def _on_first_data(self):
        self.has_data = True
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.placeholder_actor.SetVisibility(False)
        self.chart.visible = True
        if self.title_renderer is not None:
            self.title_renderer.SetBackground(1.0, 1.0, 1.0)
        if self.border_actor:
            self.border_actor.SetVisibility(True)

    def update_chart(self, x_data, y_data, tick_locs, tick_labels, title, y_label):
        if not self.has_data:
            self._on_first_data()

        self.chart.clear()
        self.chart.line(x_data, y_data, color="blue", width=2.0)

        self.chart.x_axis.tick_locations = tick_locs
        self.chart.x_axis.tick_labels = tick_labels

        self._draw_title(title)
        self.chart.y_axis.label = y_label
        self._time_line = None

    def update_multi_chart(self, x_data, series, tick_locs, tick_labels, title, y_label):
        """Like update_chart, but draws several named/colored lines with a legend."""
        if not self.has_data:
            self._on_first_data()

        self.chart.clear()
        for s in series:
            self.chart.line(
                x_data, s["y"], color=s.get("color", "blue"), width=s.get("width", 2.0), label=s.get("label", "")
            )
        self.chart.legend_visible = True

        self.chart.x_axis.tick_locations = tick_locs
        self.chart.x_axis.tick_labels = tick_labels

        self._draw_title(title)
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


# ============================================================
# VerticalSlicePanel
# ============================================================
class VerticalSlicePanel:
    def __init__(
        self, plotter, row, col, title="Vertical Slice", cmap="viridis", border_actor=None, title_renderer=None
    ):  # <-- Accept border_actor
        self.plotter = plotter
        self.row = row
        self.col = col
        self.title = title
        self.cmap = cmap
        self.is_active = False
        self.border_actor = border_actor
        # See MapPanel's title_renderer -- keeps the label outside the data renderer, which
        # matters even more here since the slice fills its whole panel by design (the
        # aspect-stretch in _make_grid), so an in-panel title would sit right on top of data.
        self.title_renderer = title_renderer
        self.title_actor = None
        self._title_actor2d = None

        self.plotter.subplot(row, col)
        self.renderer = self.plotter.renderer
        self.renderer.SetBackground(0.9, 0.9, 0.9)

        # Hide the border initially
        if self.border_actor:
            self.border_actor.SetVisibility(False)

        self.grid = None
        self.actor = None
        self.contour_line_actor = None
        self.view_initialized = False
        self._domain = None
        self._axis_label_actor = None

        self._draw_title(title)
        self._set_title_visible(False)

        self.placeholder_actor = self.plotter.add_text(
            "Vertical Slice Panel\n(Enable in sidebar)",
            position=(0.5, 0.5),
            viewport=True,
            font_size=12,
            color="#444444",
        )
        self.placeholder_actor.GetTextProperty().SetJustificationToCentered()
        self.placeholder_actor.GetTextProperty().SetVerticalJustificationToCentered()

    def _draw_title(self, title):
        self.title = title
        if self.title_renderer is not None:
            if self._title_actor2d is None:
                self._title_actor2d = _make_title_actor2d(self.title_renderer, title)
            else:
                self._title_actor2d.SetInput(title)
        else:
            self.plotter.subplot(self.row, self.col)
            self.title_actor = self.plotter.add_text(
                title, position="upper_left", font_size=9, color="black", name=f"title_{self.row}_{self.col}"
            )

    def _set_title_visible(self, visible):
        if self.title_renderer is not None:
            if self._title_actor2d is not None:
                self._title_actor2d.SetVisibility(visible)
            self.title_renderer.SetBackground(*((1.0, 1.0, 1.0) if visible else (0.9, 0.9, 0.9)))
        elif self.title_actor is not None:
            self.title_actor.SetVisibility(visible)

    def toggle_visibility(self, visible):
        self.is_active = visible

        # Toggle border visibility
        if self.border_actor:
            self.border_actor.SetVisibility(visible)

        self._set_title_visible(visible)

        if visible:
            self.renderer.SetBackground(1.0, 1.0, 1.0)
            self.placeholder_actor.SetVisibility(False)
            if self.actor is not None:
                self.actor.SetVisibility(True)
            if self.contour_line_actor is not None:
                self.contour_line_actor.SetVisibility(True)
            if self._axis_label_actor is not None:
                self._axis_label_actor.SetVisibility(True)
        else:
            self.renderer.SetBackground(0.9, 0.9, 0.9)
            self.placeholder_actor.SetVisibility(True)
            if self.actor is not None:
                self.actor.SetVisibility(False)
            if self.contour_line_actor is not None:
                self.contour_line_actor.SetVisibility(False)
            if self._axis_label_actor is not None:
                self._axis_label_actor.SetVisibility(False)

    def _make_grid(self, x, levels):
        x = np.asarray(x, dtype=np.float32)
        levels = np.asarray(levels, dtype=np.float32)

        # Level index/pressure has no spatial correspondence to the horizontal
        # degree axis (and no ticks are drawn for it), so under a true-aspect
        # parallel projection it collapses into a thin band. Vertically exaggerate
        # to match this panel's actual current viewport aspect ratio (not a fixed
        # 1:1 square) so the slice fills the whole panel rectangle, same as the
        # map panels, rather than being pillarboxed inside a square.
        x_span = float(x.max() - x.min()) if len(x) > 1 else 1.0
        level_span = float(levels.max() - levels.min()) if len(levels) > 1 else 1.0
        w, h = self.renderer.GetSize()
        viewport_aspect = (w / h) if h > 0 else 1.0
        target_level_span = (x_span / viewport_aspect) if viewport_aspect > 0 else x_span
        stretch = (target_level_span / level_span) if level_span > 0 else 1.0
        levels_scaled = (levels - levels.min()) * stretch + levels.min()

        X, Y = np.meshgrid(x, levels_scaled, indexing="ij")
        Z = np.zeros_like(X, dtype=np.float32)
        bounds = (float(x.min()), float(x.max()), float(levels_scaled.min()), float(levels_scaled.max()))
        return pv.StructuredGrid(X.astype(np.float32), Y.astype(np.float32), Z), bounds

    def _clear_axis_labels(self):
        if self._axis_label_actor is not None:
            self.plotter.remove_actor(self._axis_label_actor)
            self._axis_label_actor = None

    def _update_axis_labels(self, x, levels, level_labels, grid_bounds):
        self._clear_axis_labels()
        x = np.asarray(x, dtype=np.float32)
        levels = np.asarray(levels, dtype=np.float32)
        x_min, x_max, y_min, y_max = grid_bounds
        level_min, level_max = float(levels.min()), float(levels.max())

        points = []
        labels = []

        # X-axis ticks along the bottom edge: a handful of evenly spaced values.
        for xv in np.linspace(float(x.min()), float(x.max()), 5):
            points.append([xv, y_min, 0.5])
            labels.append(f"{xv:.0f}")

        # Level ticks along the left edge: map each chosen level index through
        # the same linear stretch _make_grid used (backed out from grid_bounds,
        # since levels here are the raw 0..31 index, not the stretched values).
        n_ticks = min(6, len(levels))
        for idx in np.linspace(0, len(levels) - 1, n_ticks).astype(int):
            frac = (float(levels[idx]) - level_min) / (level_max - level_min) if level_max > level_min else 0.0
            yv = y_min + frac * (y_max - y_min)
            label = level_labels[idx] if level_labels is not None else f"{int(levels[idx])}"
            points.append([x_min, yv, 0.5])
            labels.append(str(label))

        self._axis_label_actor = self.plotter.add_point_labels(
            np.array(points, dtype=np.float32),
            labels,
            font_size=11,
            text_color="black",
            shape=None,
            show_points=False,
            always_visible=True,
            reset_camera=False,
            pickable=False,
        )
        self._axis_label_actor.SetVisibility(self.is_active)

    def set_slice(
        self, x, levels, arr, clim=None, title=None, level_labels=None, show_axis_labels=False, n_contours=10
    ):
        self.plotter.subplot(self.row, self.col)
        if clim is None:
            clim = (
                float(np.nanpercentile(arr, 1)),
                float(np.nanpercentile(arr, 99)),
            )

        self.grid, grid_bounds = self._make_grid(x, levels)
        self.grid.point_data["slice_field"] = arr.ravel(order="F")

        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        if self.contour_line_actor is not None:
            self.plotter.remove_actor(self.contour_line_actor)

        # Filled + line contours at matching levels (like matplotlib's contourf + contour on
        # the same `levels`), both from one vtkBandedPolyDataContourFilter call:
        # `bands` carries a per-region CELL scalar (one flat value per band -- cell data
        # renders as a solid color per polygon, unlike point data which would interpolate
        # smoothly across a band boundary) and `edges` is the polyline geometry running
        # exactly along each band boundary.
        surface = self.grid.extract_surface(algorithm="dataset_surface")
        # clipping=False: clim is typically a 1st/99th-percentile range (robust to outliers),
        # so a few points always fall outside it. The default clipping=True *drops* those
        # cells from the output entirely, leaving gaps of bare background -- clamp them into
        # the nearest band instead, matching how a plain VTK lookup table clamps out-of-range
        # scalars to the end colors rather than hiding them.
        bands, edges = surface.contour_banded(
            max(int(n_contours), 2), rng=clim, scalars="slice_field", scalar_mode="value", clipping=False
        )
        band_scalar_name = bands.cell_data.keys()[0]

        self.actor = self.plotter.add_mesh(
            bands,
            scalars=band_scalar_name,
            cmap=self.cmap,
            clim=clim,
            show_edges=False,
            lighting=False,
            show_scalar_bar=False,
        )
        self.actor.mapper.scalar_range = clim

        self.contour_line_actor = self.plotter.add_mesh(
            edges,
            color="black",
            line_width=1.0,
            render_lines_as_tubes=False,
            lighting=False,
            pickable=False,
        )

        # Ensure new meshes respect current panel visibility
        self.actor.SetVisibility(self.is_active)
        self.contour_line_actor.SetVisibility(self.is_active)

        if title is not None:
            self._draw_title(title)
            self._set_title_visible(self.is_active)

        # x's meaning (and range) changes with slice orientation -- latitude vs.
        # longitude span completely different domains -- so a camera framing fit
        # for one is wrong for the other. Re-fit whenever the actual coordinate
        # domain changes, not just on the very first call, but otherwise leave
        # the camera alone so manual pan/zoom survives ordinary data updates.
        domain = (float(x.min()), float(x.max()), float(levels.min()), float(levels.max()))
        if not self.view_initialized or domain != self._domain:
            self.plotter.view_xy()
            cam = self.plotter.camera
            cam.parallel_projection = True
            # Not reset_camera(): VTK sizes parallel_scale off the bounding box's
            # diagonal, not its height, which badly under-fills a wide/short
            # panel (~45% for a 2:1 box). Set it explicitly from the known grid
            # bounds instead -- _make_grid already matched their aspect ratio to
            # this panel's viewport, so half-height fills it exactly.
            x_min, x_max, y_min, y_max = grid_bounds
            cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
            pz = cam.position[2] or 1.0
            cam.position = (cx, cy, pz)
            cam.focal_point = (cx, cy, 0.0)
            cam.parallel_scale = max((y_max - y_min) / 2.0, 1e-6)
            self.view_initialized = True
            self._domain = domain

        if show_axis_labels:
            self._update_axis_labels(x, levels, level_labels, grid_bounds)
        else:
            self._clear_axis_labels()

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
