"""Tests for the gen1 GEFS -> CAM initial-condition helpers in ``credit.gefs``."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import credit.gefs as gefs
from credit.gefs import download_gefs_run, load_member_tiles, unstagger_winds

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
