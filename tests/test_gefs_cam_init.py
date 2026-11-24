"""Tests for the gen1 GEFS -> CAM initial-condition helpers in ``credit.gefs``."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import credit.gefs as gefs
from credit.gefs import (
    GRAVITY,
    RD,
    STD_LAPSE_RATE,
    adjust_surface_pressure,
    download_gefs_run,
    extrapolate_temperature_below,
    interpolate_vertical_levels,
    load_member_tiles,
    load_target_surface_geopotential,
    regrid_member,
    unstagger_winds,
)

INIT = "2026-11-19 0000"


def _staggered_tile(n_lev=2, n_lat=2, n_lon=2):
    """One cube-sphere tile whose staggered winds increase by 1 per grid edge."""
    u_s = np.tile(np.arange(n_lat + 1, dtype=float)[None, :, None], (n_lev, 1, n_lon))
    v_w = np.tile(np.arange(n_lon + 1, dtype=float)[None, None, :], (n_lev, n_lat, 1))
    return xr.Dataset(
        {"u_s": (("lev", "latp", "lon"), u_s), "v_w": (("lev", "lat", "lonp"), v_w)},
        coords={
            "lev": np.arange(n_lev),
            "lat": np.arange(n_lat),
            "lon": np.arange(n_lon),
            "latp": np.arange(n_lat + 1),
            "lonp": np.arange(n_lon + 1),
            "geolat": (("lat", "lon"), np.zeros((n_lat, n_lon))),
            "geolon": (("lat", "lon"), np.zeros((n_lat, n_lon))),
        },
    )


def test_unstagger_winds_averages_the_two_cell_edges():
    out = unstagger_winds(_staggered_tile())
    # Cell centres sit halfway between edges 0,1 and 1,2.
    np.testing.assert_allclose(out["u_a"].values[0, :, 0], [0.5, 1.5])
    np.testing.assert_allclose(out["v_a"].values[0, 0, :], [0.5, 1.5])
    assert "u_s" not in out and "v_w" not in out


def _write_member_tiles(root, member="c00", n_lev=3, n_lat=2, n_lon=2):
    """Write six fake cold-start tiles (gfs_data/sfc_data) for one member and return the download root."""
    member_dir = root / pd.Timestamp(INIT).strftime("gefs.%Y%m%d/%H") / member
    member_dir.mkdir(parents=True)
    for tile in range(1, 7):
        xr.Dataset(
            {
                "t": (("lev", "lat", "lon"), np.full((n_lev, n_lat, n_lon), 250.0 + tile)),
                "ps": (("lat", "lon"), np.full((n_lat, n_lon), 1.0e5)),
                "zh": (("levp", "lat", "lon"), np.zeros((n_lev + 1, n_lat, n_lon))),
            },
            coords={
                "lev": np.arange(n_lev),
                "levp": np.arange(n_lev + 1),
                "lat": np.arange(n_lat),
                "lon": np.arange(n_lon),
            },
        ).to_netcdf(member_dir / f"gfs_data.tile{tile}.nc")
        xr.Dataset(
            {
                "t2m": (("Time", "yaxis_1", "xaxis_1"), np.full((1, n_lat, n_lon), 280.0)),
                "smc": (("Time", "zaxis_1", "yaxis_1", "xaxis_1"), np.full((1, 4, n_lat, n_lon), 0.3)),
            },
            coords={"Time": [1]},
        ).to_netcdf(member_dir / f"sfc_data.tile{tile}.nc")
    return root


@pytest.mark.parametrize("variables", [["t", "ps"], ["t", "t2m"], ["t2m"], ["t", "t2m", "smc"]])
def test_load_member_tiles_without_soil_moisture(tmp_path, variables):
    root = _write_member_tiles(tmp_path)
    tiles = load_member_tiles(str(root), INIT, "c00", variables)
    assert len(tiles) == 6
    assert set(variables) <= set(tiles[0].data_vars)
    if "smc" in variables:
        np.testing.assert_allclose(tiles[0]["smc"].values, 30.0)  # top layer only, x100


def test_load_member_tiles_rejects_unknown_variables(tmp_path):
    root = _write_member_tiles(tmp_path)
    with pytest.raises(ValueError, match="valid GEFS variables"):
        load_member_tiles(str(root), INIT, "c00", ["not_a_gefs_variable"])


class _FakeGCS:
    """Stands in for gcsfs.GCSFileSystem: ls returns bucket paths without the gs:// prefix, like gcsfs."""

    def __init__(self, files):
        self.files = files  # remote path -> bytes
        self.got = []

    def exists(self, path):
        prefix = path.removeprefix("gs://").rstrip("/") + "/"
        return any(f.startswith(prefix) for f in self.files)

    def ls(self, path):
        prefix = path.removeprefix("gs://").rstrip("/") + "/"
        return sorted(f for f in self.files if f.startswith(prefix))

    def du(self, path):
        return len(self.files[path])

    def get(self, path, local_path):
        self.got.append(path)
        with open(local_path, "wb") as fobj:
            fobj.write(self.files[path])


def test_download_gefs_run_fetches_full_remote_paths_and_resumes(tmp_path, monkeypatch):
    remote = "gfs-ensemble-forecast-system/gefs.20261119/00/atmos/init/c00"
    files = {f"{remote}/gfs_ctrl.nc": b"ctrl", f"{remote}/gfs_data.tile1.nc": b"0123456789"}
    fake = _FakeGCS(files)
    monkeypatch.setattr(gefs.gcsfs, "GCSFileSystem", lambda **kwargs: fake)

    download_gefs_run(INIT, str(tmp_path), n_pert_members=0)
    local = tmp_path / "gefs.20261119" / "00" / "c00"
    assert sorted(fake.got) == sorted(files)
    assert (local / "gfs_data.tile1.nc").read_bytes() == b"0123456789"

    # A truncated file is fetched again; a complete one is left alone.
    (local / "gfs_data.tile1.nc").write_bytes(b"012")
    fake.got.clear()
    download_gefs_run(INIT, str(tmp_path), n_pert_members=0)
    assert fake.got == [f"{remote}/gfs_data.tile1.nc"]
    assert (local / "gfs_data.tile1.nc").read_bytes() == b"0123456789"


def _tile_average_weights(path, n_lat=2, n_lon=2):
    """ESMF-style weights mapping each of the 6 tiles to one destination point (the tile mean), on a 2 x 3 grid."""
    per_tile = n_lat * n_lon
    rows, cols = np.repeat(np.arange(6), per_tile) + 1, np.arange(6 * per_tile) + 1
    lat, lon = np.array([-45.0, 45.0]), np.array([0.0, 120.0, 240.0])
    xr.Dataset(
        {
            "S": ("n_s", np.full(rows.size, 1.0 / per_tile)),
            "row": ("n_s", rows),
            "col": ("n_s", cols),
            "dst_grid_dims": ("dst_grid_rank", np.array([lon.size, lat.size])),
            "xc_b": ("n_b", np.tile(lon, lat.size)),
            "yc_b": ("n_b", np.repeat(lat, lon.size)),
            "xc_a": ("n_a", np.zeros(6 * per_tile)),
        }
    ).to_netcdf(path)
    return path


def test_regrid_member_fills_every_interface_level(tmp_path):
    n_lev, n_lat, n_lon = 3, 2, 2
    tiles = [
        xr.Dataset(
            {
                "t": (("lev", "lat", "lon"), np.full((n_lev, n_lat, n_lon), 250.0 + tile)),
                "zh": (
                    ("levp", "lat", "lon"),
                    np.broadcast_to(
                        (10.0 * np.arange(n_lev + 1))[:, None, None] + tile, (n_lev + 1, n_lat, n_lon)
                    ).copy(),
                ),
                "ps": (("lat", "lon"), np.full((n_lat, n_lon), 1.0e5)),
            },
            coords={
                "lev": np.arange(n_lev),
                "levp": np.arange(n_lev + 1),
                "lat": np.arange(n_lat),
                "lon": np.arange(n_lon),
            },
        )
        for tile in range(6)
    ]
    out = regrid_member(tiles, str(_tile_average_weights(tmp_path / "weights.nc")))
    expected_zh = 10.0 * np.arange(n_lev + 1)[:, None] + np.arange(6)[None, :]  # (levp, destination point)
    np.testing.assert_allclose(out["zh"].values.reshape(n_lev + 1, -1), expected_zh)
    np.testing.assert_allclose(out["ps"].values, 1.0e5)


def test_adjust_surface_pressure_follows_the_hypsometric_equation():
    ps, t_low = np.array([95000.0]), np.array([288.0])
    same = adjust_surface_pressure(ps, np.array([500.0]), np.array([500.0]), t_low)
    np.testing.assert_allclose(same, ps)
    lower = adjust_surface_pressure(ps, np.array([500.0]), np.array([0.0]), t_low)
    t_layer = 288.0 + 0.5 * STD_LAPSE_RATE * 500.0
    np.testing.assert_allclose(lower, ps * np.exp(GRAVITY * 500.0 / (RD * t_layer)))
    higher = adjust_surface_pressure(ps, np.array([500.0]), np.array([1000.0]), t_low)
    assert higher[0] < ps[0] < lower[0]


def test_extrapolate_temperature_below_only_changes_levels_under_the_source():
    temperature = np.full((2, 1, 1), 280.0)
    pressure = np.array([90000.0, 101000.0]).reshape(2, 1, 1)
    out = extrapolate_temperature_below(temperature, pressure, np.array([[95000.0]]), np.array([[280.0]]))
    assert out[0, 0, 0] == 280.0
    expected = 280.0 * (101000.0 / 95000.0) ** (RD * STD_LAPSE_RATE / GRAVITY)
    np.testing.assert_allclose(out[1, 0, 0], expected)
    assert out[1, 0, 0] > 280.0


def _vertical_case(tmp_path, gefs_surface_height=500.0):
    """A 4-level GEFS column on 500 m terrain plus a 3-level destination grid that reaches closer to the ground."""
    member_dir = tmp_path / pd.Timestamp(INIT).strftime("gefs.%Y%m%d/%H") / "c00"
    member_dir.mkdir(parents=True)
    gefs_a = np.array([0.0, 5000.0, 10000.0, 5000.0, 0.0])
    gefs_b = np.array([0.0, 0.0, 0.2, 0.6, 1.0])
    xr.Dataset({"vcoord": (("nvcoord", "levsp"), np.stack([gefs_a, gefs_b]))}).to_netcdf(member_dir / "gfs_ctrl.nc")
    dest_file = tmp_path / "dest_levels.nc"
    xr.Dataset({"hyai": ("ilev", [0.0, 2000.0, 500.0, 0.0]), "hybi": ("ilev", [0.0, 0.1, 0.9, 1.0])}).to_netcdf(
        dest_file
    )
    shape = (2, 3)
    ps = np.full(shape, 95000.0)
    p_mid = 0.5 * ((gefs_a[:-1] + gefs_b[:-1] * 95000.0) + (gefs_a[1:] + gefs_b[1:] * 95000.0))
    t = 288.0 * (p_mid / 95000.0) ** (RD * STD_LAPSE_RATE / GRAVITY)
    zh = np.array([20000.0, 12000.0, 6000.0, 2000.0, gefs_surface_height])
    regrid_ds = xr.Dataset(
        {
            "ps": (("lat", "lon"), ps),
            "t": (("lev", "lat", "lon"), np.broadcast_to(t[:, None, None], (4, *shape)).copy()),
            "sphum": (("lev", "lat", "lon"), np.full((4, *shape), 0.005)),
            "zh": (("levp", "lat", "lon"), np.broadcast_to(zh[:, None, None], (5, *shape)).copy()),
        },
        coords={"lev": np.arange(4), "levp": np.arange(5), "lat": [-45.0, 45.0], "lon": [0.0, 120.0, 240.0]},
    )
    return regrid_ds, str(tmp_path), str(dest_file), t


def test_interpolate_vertical_levels_without_target_terrain_is_unchanged(tmp_path):
    regrid_ds, member_path, dest_file, t = _vertical_case(tmp_path)
    out = interpolate_vertical_levels(regrid_ds, member_path, INIT, "c00", dest_file)
    np.testing.assert_allclose(out["ps"].values, 95000.0)
    np.testing.assert_allclose(out["t"].values[-1], t[-1])  # held constant below the lowest GEFS level


def test_interpolate_vertical_levels_moves_to_the_target_terrain(tmp_path):
    regrid_ds, member_path, dest_file, t = _vertical_case(tmp_path)
    phis = np.zeros((2, 3))  # destination terrain at sea level, 500 m below GEFS
    out = interpolate_vertical_levels(regrid_ds, member_path, INIT, "c00", dest_file, target_surface_geopotential=phis)
    expected_ps = adjust_surface_pressure(
        np.full((2, 3), 95000.0), np.full((2, 3), 500.0), phis / GRAVITY, t[-1], 0.005
    )
    np.testing.assert_allclose(out["ps"].values, expected_ps)
    assert (out["ps"].values > 95000.0).all()
    np.testing.assert_allclose(out["P"].values[-1], 0.5 * ((500.0 + 0.9 * expected_ps) + expected_ps))
    assert (out["t"].values[-1] > t[-1]).all()  # warmer than the lowest GEFS level, not clamped


def test_interpolate_vertical_levels_needs_surface_height(tmp_path):
    regrid_ds, member_path, dest_file, _ = _vertical_case(tmp_path)
    with pytest.raises(ValueError, match="zh"):
        interpolate_vertical_levels(
            regrid_ds.drop_vars("zh"), member_path, INIT, "c00", dest_file, target_surface_geopotential=np.zeros((2, 3))
        )


def test_load_target_surface_geopotential_checks_the_grid(tmp_path):
    lat, lon = np.array([-45.0, 45.0]), np.array([0.0, 120.0, 240.0])
    good = tmp_path / "phis.nc"
    xr.Dataset(
        {"PHIS": (("latitude", "longitude"), np.ones((2, 3)))}, coords={"latitude": lat, "longitude": lon}
    ).to_netcdf(good)
    np.testing.assert_allclose(load_target_surface_geopotential(str(good), "PHIS", lat, lon), 1.0)
    flipped = tmp_path / "phis_flipped.nc"
    xr.Dataset(
        {"PHIS": (("latitude", "longitude"), np.ones((2, 3)))}, coords={"latitude": lat[::-1], "longitude": lon}
    ).to_netcdf(flipped)
    with pytest.raises(ValueError, match="latitude"):
        load_target_surface_geopotential(str(flipped), "PHIS", lat, lon)
