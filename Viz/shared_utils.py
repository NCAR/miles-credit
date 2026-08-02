import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, MultiLineString

BASE_ARRAY_NAME = "base_field"


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


# ============================================================
# MapPanel
# ============================================================
class MapPanel:
    def __init__(
        self, plotter, row, col, lon, lat, coast_texture, coast_plane, title, cmap="viridis", show_scalar_bar=True
    ):
        self.plotter = plotter
        self.row = row
        self.col = col
        self.lon = lon
        self.lat = lat
        self.title = title
        self.cmap = cmap
        self.show_scalar_bar = show_scalar_bar

        self.grid = make_surface_grid(lon, lat, z_value=0.0)
        self.base_actor = None
        self.coast_actor = None
        self.contour_actors = {}

        self.plotter.subplot(row, col)
        self.coast_actor = self.plotter.add_mesh(coast_plane, texture=coast_texture, lighting=False, pickable=False)
        self.plotter.add_text(title, position="upper_left", font_size=9, color="black", name=f"title_{row}_{col}")

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
