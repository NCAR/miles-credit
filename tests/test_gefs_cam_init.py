"""Tests for the gen1 GEFS -> CAM initial-condition helpers in ``credit.gefs``."""

import numpy as np
import xarray as xr

from credit.gefs import unstagger_winds


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
