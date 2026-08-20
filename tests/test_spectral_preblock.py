"""Tests for the spectral_to_grid preblock (credit/preblock/spectral.py),
the reduced Gaussian grid utilities (credit/reduced_gaussian.py), and the
arco_era5_co dataset plumbing.

All tests here run offline on small synthetic truncations. The one test that
streams real ARCO ERA5 data is gated behind SKIP_REMOTE (mirrors the HRRR
remote-test convention).
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from credit.preblock.spectral import SpectralToGrid, pack_spectral, spectral_packing_index
from credit.reduced_gaussian import ReducedGaussianGrid, reduced_gaussian_to_latlon_bilinear_weights

SKIP_REMOTE = bool(os.getenv("SKIP_REMOTE")) or bool(os.getenv("SKIP_ARCO_REMOTE"))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def gaussian_reduced_grid(n_rings: int, ring_nlon=None) -> ReducedGaussianGrid:
    """Small synthetic reduced Gaussian grid: rings at Gaussian latitudes."""
    cost, _ = np.polynomial.legendre.leggauss(n_rings)
    ring_lats = np.degrees(np.arcsin(cost))[::-1]  # N -> S
    if ring_nlon is None:
        # crudely reduced: fewer points near the poles, even counts
        # reduced toward the poles, but keep enough points per ring for the
        # linear lon interpolation to resolve low wavenumbers (real N320 pole
        # rings have 20 points)
        ring_nlon = np.maximum(16, (2 * n_rings * np.cos(np.radians(ring_lats))).astype(int) // 2 * 2)
    else:
        ring_nlon = np.asarray(ring_nlon)
    return ReducedGaussianGrid(ring_lats, ring_nlon)


def make_grid_file(tmp_path, grid: ReducedGaussianGrid) -> str:
    path = str(tmp_path / "grid.nc")
    grid.save(path)
    return path


def packed_single_coef(truncation: int, l: int, m: int, value: complex) -> torch.Tensor:  # noqa: E741
    """Packed ECMWF coefficient vector with a single nonzero (l, m) entry."""
    L = truncation + 1
    coef = torch.zeros(L, L, dtype=torch.complex64)
    coef[l, m] = value
    return pack_spectral(coef, truncation).to(torch.float32)


def batchify(packed: torch.Tensor) -> torch.Tensor:
    """(values,) -> (B=1, n_levels=1, T=1, values) as produced after collate."""
    return packed.view(1, 1, 1, -1)


def make_batch(key: str, tensor: torch.Tensor) -> dict:
    source = key.split("/")[0]
    return {"input": {source: {key: tensor}}}


# ----------------------------------------------------------------------
# ReducedGaussianGrid
# ----------------------------------------------------------------------


def test_reduced_grid_from_points_roundtrip(tmp_path):
    grid = gaussian_reduced_grid(8)
    lat, lon = grid.lats_flat(), grid.lons_flat()
    grid2 = ReducedGaussianGrid.from_points(lat, lon)
    np.testing.assert_allclose(grid2.ring_lats, grid.ring_lats)
    np.testing.assert_array_equal(grid2.ring_nlon, grid.ring_nlon)

    path = make_grid_file(tmp_path, grid)
    grid3 = ReducedGaussianGrid.from_file(path)
    np.testing.assert_allclose(grid3.ring_lats, grid.ring_lats)
    np.testing.assert_array_equal(grid3.ring_nlon, grid.ring_nlon)
    assert grid3.n_points == grid.ring_nlon.sum()


def test_reduced_grid_rejects_unsorted():
    with pytest.raises(ValueError, match="strictly decreasing"):
        ReducedGaussianGrid(np.array([0.0, 10.0]), np.array([4, 4]))


# ----------------------------------------------------------------------
# Scalar synthesis
# ----------------------------------------------------------------------


def test_packing_index_order():
    # ECMWF m-major: (0,0),(1,0)...(T,0),(1,1),(2,1)...
    T = 3
    l_idx, m_idx = spectral_packing_index(T)
    assert l_idx.tolist() == [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]
    assert m_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]


def test_scalar_constant_field(tmp_path):
    """IFS convention: the (0,0) coefficient is the global mean -> constant field."""
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    block = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    key = "ERA5/prognostic/3d/temperature"
    x = batchify(packed_single_coef(T, 0, 0, 288.0 + 0j))
    out = block(make_batch(key, x))["input"]["ERA5"][key]
    assert out.shape == (1, 1, 1, grid.n_points)
    assert torch.allclose(out, torch.full_like(out, 288.0), atol=1e-4)


def test_scalar_y10_field(tmp_path):
    """(1,0) coefficient c -> field c * sqrt(3) * sin(lat) (ECMWF normalization)."""
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    block = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    key = "ERA5/prognostic/3d/temperature"
    c = 2.5
    x = batchify(packed_single_coef(T, 1, 0, c + 0j))
    out = block(make_batch(key, x))["input"]["ERA5"][key][0, 0, 0]
    expected = c * math.sqrt(3.0) * np.sin(np.radians(grid.lats_flat()))
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)


def test_scalar_m2_mode_longitude_dependence(tmp_path):
    """A single (l=2, m=2) coefficient must synthesize the correct zonal wave-2
    longitude structure: 2*Re[c * Ybar_22 * exp(2 i lambda)] * sqrt(4 pi)."""
    T = 6
    grid = gaussian_reduced_grid(T + 1, ring_nlon=[16] * (T + 1))
    block = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    key = "ERA5/prognostic/3d/temperature"
    c = 1.0 + 0.5j
    x = batchify(packed_single_coef(T, 2, 2, c))
    out = block(make_batch(key, x))["input"]["ERA5"][key][0, 0, 0].numpy()

    # Orthonormal (no CS phase) Ybar_22 latitude part: sqrt(15/(32 pi)) cos^2(lat)
    lats = np.radians(grid.lats_flat())
    lons = np.radians(grid.lons_flat())
    ybar = np.sqrt(15.0 / (32.0 * np.pi)) * np.cos(lats) ** 2
    expected = math.sqrt(4 * math.pi) * 2.0 * np.real(c * ybar * np.exp(2j * lons))
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_scalar_matches_torch_harmonics_full_grid(tmp_path):
    """On a reduced grid whose rings all share one nlon (i.e. a full Gaussian
    grid), the ring synthesis must equal torch-harmonics InverseRealSHT."""
    from torch_harmonics import InverseRealSHT

    T = 15
    nlat, nlon = T + 1, 2 * (T + 1)
    grid = gaussian_reduced_grid(nlat, ring_nlon=[nlon] * nlat)
    block = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    torch.manual_seed(0)
    coef = torch.randn(T + 1, T + 1, 2)
    tri = torch.zeros(T + 1, T + 1, dtype=torch.bool)
    l_idx, m_idx = spectral_packing_index(T)
    tri[l_idx, m_idx] = True
    coef[~tri] = 0.0
    coef[:, 0, 1] = 0.0  # m=0 coefficients are real
    coef_c = torch.view_as_complex(coef.contiguous())

    key = "ERA5/prognostic/3d/temperature"
    x = batchify(pack_spectral(coef_c, T))
    out = block(make_batch(key, x))["input"]["ERA5"][key][0, 0, 0]

    isht = InverseRealSHT(nlat, nlon, lmax=T + 1, mmax=T + 1, grid="legendre-gauss", norm="ortho", csphase=False)
    ref = isht(coef_c * math.sqrt(4 * math.pi))
    np.testing.assert_allclose(out.view(nlat, nlon).numpy(), ref.numpy(), atol=1e-4)


def test_scalar_equiangular_target():
    """Equiangular synthesis matches torch-harmonics on the same grid."""
    from torch_harmonics import InverseRealSHT

    T = 10
    nlat, nlon = 19, 36
    block = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        truncation=T,
        target_grid="equiangular",
        nlat=nlat,
        nlon=nlon,
    )
    key = "ERA5/prognostic/3d/temperature"
    c = 3.0
    x = batchify(packed_single_coef(T, 1, 0, c + 0j))
    out = block(make_batch(key, x))["input"]["ERA5"][key]
    assert out.shape == (1, 1, 1, nlat, nlon)
    isht = InverseRealSHT(nlat, nlon, lmax=T + 2, mmax=T + 1, grid="equiangular", norm="ortho", csphase=False)
    coef = torch.zeros(T + 2, T + 1, dtype=torch.complex64)
    coef[1, 0] = c * math.sqrt(4 * math.pi)
    np.testing.assert_allclose(out[0, 0, 0].numpy(), isht(coef).numpy(), atol=1e-5)


def test_exp_vars_and_rename(tmp_path):
    """log_surface_pressure -> surface_pressure via out-key rename + exp."""
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    raw = "ERA5/prognostic/2d/log_surface_pressure"
    out_key = "ERA5/prognostic/2d/surface_pressure"
    block = SpectralToGrid(
        scalar_vars={raw: out_key},
        exp_vars=[out_key],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    lnsp = 11.5
    x = batchify(packed_single_coef(T, 0, 0, lnsp + 0j))
    result = block(make_batch(raw, x))["input"]["ERA5"]
    assert raw not in result  # consumed
    assert torch.allclose(result[out_key], torch.full_like(result[out_key], math.exp(lnsp)), rtol=1e-4)


# ----------------------------------------------------------------------
# Vector (wind) synthesis
# ----------------------------------------------------------------------


def test_solid_body_rotation_winds(tmp_path):
    """zeta ~ Y10, D = 0 (solid-body rotation) -> u = U cos(lat), v = 0.

    With the ECMWF convention (coef c at (1,0), x sqrt(4pi) to orthonormal),
    zeta = c*sqrt(3)*mu and u = (a c sqrt(3) / 2) cos(lat).
    """
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    radius = 6371229.0
    vo_key, d_key = "ERA5/prognostic/3d/vorticity", "ERA5/prognostic/3d/divergence"
    u_key, v_key = "ERA5/prognostic/3d/u_component_of_wind", "ERA5/prognostic/3d/v_component_of_wind"
    block = SpectralToGrid(
        vector_vars=[{"vorticity": vo_key, "divergence": d_key, "u": u_key, "v": v_key}],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
        radius=radius,
    )
    c = 2.0e-5
    batch = make_batch(vo_key, batchify(packed_single_coef(T, 1, 0, c + 0j)))
    batch["input"]["ERA5"][d_key] = batchify(packed_single_coef(T, 0, 0, 0j))
    out = block(batch)["input"]["ERA5"]
    assert vo_key not in out and d_key not in out  # consumed
    u = out[u_key][0, 0, 0].numpy()
    v = out[v_key][0, 0, 0].numpy()
    expected_u = radius * c * math.sqrt(3.0) / 2.0 * np.cos(np.radians(grid.lats_flat()))
    np.testing.assert_allclose(u, expected_u, rtol=1e-4)
    np.testing.assert_allclose(v, np.zeros_like(v), atol=1e-9 * radius * c)


def test_divergent_flow_winds(tmp_path):
    """Pure divergence: u = (1/(a cos)) dchi/dlon, v = (1/a) dchi/dlat.

    With D ~ single (l=2, m=1) coefficient the flow is irrotational; verify
    against a dense finite-difference of the synthesized velocity potential.
    """
    T = 20
    n_rings = T + 1
    grid = gaussian_reduced_grid(n_rings, ring_nlon=[64] * n_rings)
    radius = 6371229.0
    vo_key, d_key = "ERA5/prognostic/3d/vorticity", "ERA5/prognostic/3d/divergence"
    u_key, v_key = "ERA5/prognostic/3d/u_component_of_wind", "ERA5/prognostic/3d/v_component_of_wind"
    grid_file = make_grid_file(tmp_path, grid)
    block = SpectralToGrid(
        vector_vars=[{"vorticity": vo_key, "divergence": d_key, "u": u_key, "v": v_key}],
        truncation=T,
        grid_file=grid_file,
        radius=radius,
    )
    c = 1.0e-5 + 0.7e-5j
    batch = make_batch(vo_key, batchify(packed_single_coef(T, 0, 0, 0j)))
    batch["input"]["ERA5"][d_key] = batchify(packed_single_coef(T, 2, 1, c))
    out = block(batch)["input"]["ERA5"]
    u = out[u_key][0, 0, 0].numpy()
    v = out[v_key][0, 0, 0].numpy()

    # chi = -a^2 D_lm / (l(l+1)); synthesize chi as a scalar, then finite-diff.
    chi_block = SpectralToGrid(scalar_vars=["ERA5/prognostic/3d/chi"], truncation=T, grid_file=grid_file)
    chi_packed = packed_single_coef(T, 2, 1, c * (-(radius**2) / 6.0))
    chi = chi_block(make_batch("ERA5/prognostic/3d/chi", batchify(chi_packed)))["input"]["ERA5"][
        "ERA5/prognostic/3d/chi"
    ][0, 0, 0].numpy()

    # u check: spectral d/dlon of chi on each ring (exact for band-limited fields)
    lats = np.radians(grid.ring_lats)
    for j in range(grid.n_rings):
        sl = slice(grid.ring_offset[j], grid.ring_offset[j] + grid.ring_nlon[j])
        ring_chi = chi[sl]
        dchi_dlon = np.fft.irfft(np.fft.rfft(ring_chi) * 1j * np.arange(len(ring_chi) // 2 + 1), n=len(ring_chi))
        np.testing.assert_allclose(u[sl], dchi_dlon / (radius * np.cos(lats[j])), rtol=1e-3, atol=1e-8)

    # v check: centered finite difference of chi across rings (same lon 0 column)
    ring0 = grid.ring_offset  # index of lon=0 point on each ring
    dlat = lats[:-2] - lats[2:]  # decreasing lats
    v_fd = (chi[ring0[:-2]] - chi[ring0[2:]]) / (radius * dlat)
    np.testing.assert_allclose(v[ring0[1:-1]], v_fd, rtol=5e-2, atol=1e-8)


def test_vector_and_scalar_share_key(tmp_path):
    """vorticity requested both as wind input and as a scalar channel:
    the scalar output must be the synthesized vorticity, not deleted."""
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    vo_key, d_key = "ERA5/prognostic/3d/vorticity", "ERA5/prognostic/3d/divergence"
    block = SpectralToGrid(
        scalar_vars=[vo_key],
        vector_vars=[
            {
                "vorticity": vo_key,
                "divergence": d_key,
                "u": "ERA5/prognostic/3d/u_component_of_wind",
                "v": "ERA5/prognostic/3d/v_component_of_wind",
            }
        ],
        truncation=T,
        grid_file=make_grid_file(tmp_path, grid),
    )
    c = 3.0e-5
    batch = make_batch(vo_key, batchify(packed_single_coef(T, 1, 0, c + 0j)))
    batch["input"]["ERA5"][d_key] = batchify(packed_single_coef(T, 0, 0, 0j))
    out = block(batch)["input"]["ERA5"]
    assert vo_key in out and d_key not in out
    expected = c * math.sqrt(3.0) * np.sin(np.radians(grid.lats_flat()))
    np.testing.assert_allclose(out[vo_key][0, 0, 0].numpy(), expected, atol=1e-9)


def test_noop_when_inputs_absent(tmp_path):
    """At rollout steps t>1 the spectral keys are gone — block must be a no-op."""
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    key = "ERA5/prognostic/3d/temperature"
    block = SpectralToGrid(scalar_vars=[key], truncation=T, grid_file=make_grid_file(tmp_path, grid))
    gridded = torch.randn(1, 1, 1, 30, 60)
    batch = {"input": {"ERA5": {"ERA5/prognostic/3d/u_component_of_wind": gridded}}}
    out = block(batch)
    assert set(out["input"]["ERA5"]) == {"ERA5/prognostic/3d/u_component_of_wind"}
    assert torch.equal(out["input"]["ERA5"]["ERA5/prognostic/3d/u_component_of_wind"], gridded)


def test_wrong_length_raises(tmp_path):
    T = 10
    grid = gaussian_reduced_grid(T + 1)
    key = "ERA5/prognostic/3d/temperature"
    block = SpectralToGrid(scalar_vars=[key], truncation=T, grid_file=make_grid_file(tmp_path, grid))
    with pytest.raises(ValueError, match="spectral values"):
        block(make_batch(key, torch.randn(1, 1, 1, 999)))


# ----------------------------------------------------------------------
# Bilinear weights + Regridder round trip
# ----------------------------------------------------------------------


def test_bilinear_weights_regrid_smooth_field(tmp_path):
    from credit.preblock.regrid import Regridder

    grid = gaussian_reduced_grid(64)
    # stay away from the sparse polar rings; rows poleward of the first/last
    # ring are clamped by design (tested separately below)
    dst_lat = np.linspace(80.0, -80.0, 45)
    dst_lon = np.arange(0.0, 360.0, 4.0)
    wfile = str(tmp_path / "weights.nc")
    ds = reduced_gaussian_to_latlon_bilinear_weights(grid, dst_lat, dst_lon, wfile)
    # partition of unity: every destination row's weights sum to 1
    row_sums = np.zeros(len(dst_lat) * len(dst_lon))
    np.add.at(row_sums, ds["row"].values - 1, ds["S"].values)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def field(lat_deg, lon_deg):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        return np.sin(lat) ** 2 + 0.5 * np.cos(lat) * np.cos(2 * lon)

    src = torch.tensor(field(grid.lats_flat(), grid.lons_flat()), dtype=torch.float32)
    key = "ERA5/prognostic/2d/temperature"
    regrid = Regridder(weight_file=wfile, variables=[key])
    out = regrid({"input": {"ERA5": {key: src.view(1, 1, 1, -1)}}})["input"]["ERA5"][key]
    assert out.shape == (1, 1, 1, len(dst_lat), len(dst_lon))
    expected = field(dst_lat[:, None], dst_lon[None, :])
    # tolerance is bilinear truncation error at this ring spacing (~2.8 deg)
    np.testing.assert_allclose(out[0, 0, 0].numpy(), expected, atol=1e-2)
    # Regridder must expose the destination grid for GridSchema.resolve
    assert regrid.dst_grid_type == "rectilinear"
    np.testing.assert_allclose(regrid.dst_lat, dst_lat)


def test_bilinear_weights_pole_clamp(tmp_path):
    """Destination rows poleward of the first/last ring clamp to that ring."""
    grid = gaussian_reduced_grid(8)
    dst_lat = np.array([90.0, -90.0])
    dst_lon = np.array([0.0, 180.0])
    ds = reduced_gaussian_to_latlon_bilinear_weights(grid, dst_lat, dst_lon)
    rows = ds["row"].values - 1
    cols = ds["col"].values - 1
    first_ring = set(range(grid.ring_nlon[0]))
    last_ring = set(range(grid.ring_offset[-1], grid.n_points))
    assert set(cols[rows < 2]) <= first_ring
    assert set(cols[rows >= 2]) <= last_ring


# ----------------------------------------------------------------------
# Full synthetic chain: spectral -> reduced grid -> lat/lon
# ----------------------------------------------------------------------


def test_full_chain_spectral_to_latlon(tmp_path):
    """Synthesize a smooth band-limited field and regrid it to lat/lon; the
    result must match the field evaluated analytically on the lat/lon grid."""
    from credit.preblock.regrid import Regridder

    T = 20
    grid = gaussian_reduced_grid(64)  # synthesis is exact on any rings; finer rings shrink bilinear error
    key = "ERA5/prognostic/3d/temperature"
    synth = SpectralToGrid(scalar_vars=[key], truncation=T, grid_file=make_grid_file(tmp_path, grid))
    # Y10-like + constant: T = 280 + 10 sqrt(3) sin(lat)
    L = T + 1
    coef = torch.zeros(L, L, dtype=torch.complex64)
    coef[0, 0] = 280.0
    coef[1, 0] = 10.0
    x = batchify(pack_spectral(coef, T))

    dst_lat = np.linspace(87.0, -87.0, 30)
    dst_lon = np.arange(0.0, 360.0, 6.0)
    wfile = str(tmp_path / "w.nc")
    reduced_gaussian_to_latlon_bilinear_weights(grid, dst_lat, dst_lon, wfile)
    regrid = Regridder(weight_file=wfile, variables=[key])

    out = regrid(synth(make_batch(key, x)))["input"]["ERA5"][key][0, 0, 0].numpy()
    expected = 280.0 + 10.0 * math.sqrt(3.0) * np.sin(np.radians(dst_lat))[:, None] * np.ones(len(dst_lon))
    np.testing.assert_allclose(out, expected, atol=1e-2)


# ----------------------------------------------------------------------
# arco_era5_co dataset plumbing (offline parts)
# ----------------------------------------------------------------------


def _co_config(tmp_path, variables):
    return {
        "source": {
            "ERA5": {
                "dataset_type": "arco_era5_co",
                "levels": [137],
                "variables": variables,
            }
        },
        "save_loc": str(tmp_path),
        "start_datetime": "2010-07-01",
        "end_datetime": "2010-07-02",
        "timestep": "6h",
        "history_len": 1,
        "forecast_len": 1,
    }


def test_co_dataset_fetch_plan(tmp_path):
    from credit.datasets.gen_2.era5_co import ARCOERA5CODataset

    ds = ARCOERA5CODataset(
        _co_config(
            tmp_path,
            {
                "prognostic": {
                    "vars_3D": [
                        "temperature",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "specific_humidity",
                    ],
                    "vars_2D": ["surface_pressure"],
                },
                "static": {"vars_2D": ["geopotential_at_surface"]},
            },
        )
    )
    plan3d = ds._fetch_plans["prognostic"]["3d"]
    emitted = [name for name, *_ in plan3d]
    # u/v collapse to a single vorticity+divergence fetch
    assert emitted == ["temperature", "vorticity", "divergence", "specific_humidity"]
    stores = {name: store for name, store, _, _ in plan3d}
    assert stores["vorticity"] == "wind" and stores["specific_humidity"] == "moisture"
    assert ds._fetch_plans["prognostic"]["2d"] == [("surface_pressure", "reanalysis", "sp", "grid")]
    assert ds._fetch_plans["static"]["2d"] == [("geopotential_at_surface", "reanalysis", "z", "grid")]


def test_co_dataset_short_alias_and_unknown(tmp_path):
    from credit.datasets.gen_2.era5_co import ARCOERA5CODataset

    ds = ARCOERA5CODataset(_co_config(tmp_path, {"prognostic": {"vars_2D": ["t2m"]}}))
    assert ds._fetch_plans["prognostic"]["2d"] == [("2m_temperature", "reanalysis", "t2m", "grid")]

    with pytest.raises(KeyError, match="unknown variable 'not_a_var'"):
        ARCOERA5CODataset(_co_config(tmp_path, {"prognostic": {"vars_2D": ["not_a_var"]}}))


def test_co_dataset_rejects_pressure_levels(tmp_path):
    from credit.datasets.gen_2.era5_co import ARCOERA5CODataset

    cfg = _co_config(tmp_path, {"prognostic": {"vars_3D": ["temperature"]}})
    cfg["source"]["ERA5"]["level_coord"] = "level"
    with pytest.raises(ValueError, match="hybrid model levels"):
        ARCOERA5CODataset(cfg)


@pytest.mark.skipif(SKIP_REMOTE, reason="Set SKIP_REMOTE=1 to skip remote ARCO tests")
def test_co_dataset_remote_sample_and_synthesis(tmp_path):
    """End-to-end against real ARCO data: read one time/level, synthesize t/u/v
    on the N320 grid, regrid to 0.25°, and compare with the analysis-ready store."""
    import gcsfs
    import xarray as xr

    from credit.datasets.gen_2.era5_co import ARCOERA5CODataset

    cfg = _co_config(
        tmp_path,
        {
            "prognostic": {
                # specific_humidity sits between the spectral vars on purpose: it comes
                # from the moisture store while t/u/v come from the wind store, so a
                # store-grouped read order would reshuffle the channels (see the
                # emission-order assertion below).
                "vars_3D": [
                    "temperature",
                    "specific_humidity",
                    "u_component_of_wind",
                    "v_component_of_wind",
                ],
                "vars_2D": ["surface_pressure"],
            }
        },
    )
    ds = ARCOERA5CODataset(cfg)
    t0 = pd.Timestamp("2010-07-01T00:00")
    sample = ds[(t0, 0)]

    # ChannelSchema.from_config builds the expected layout from the declared order and
    # the concat preblock rejects a batch that disagrees, so the dataset must emit
    # variables in declared order even though it reads one zarr store at a time.
    assert [k for k in sample["input"] if k.startswith("ERA5/prognostic/3d/")] == [
        "ERA5/prognostic/3d/temperature",
        "ERA5/prognostic/3d/specific_humidity",
        "ERA5/prognostic/3d/vorticity",
        "ERA5/prognostic/3d/divergence",
    ]

    x_t = sample["input"]["ERA5/prognostic/3d/temperature"]
    x_vo = sample["input"]["ERA5/prognostic/3d/vorticity"]
    x_d = sample["input"]["ERA5/prognostic/3d/divergence"]
    x_sp = sample["input"]["ERA5/prognostic/2d/surface_pressure"]
    assert x_t.shape == (1, 1, 410240)
    assert x_sp.shape == (1, 1, 542080)

    grid_file = os.path.join(str(tmp_path), "ERA5_reduced_gaussian_grid.nc")
    assert os.path.isfile(grid_file)

    synth = SpectralToGrid(
        scalar_vars=["ERA5/prognostic/3d/temperature"],
        vector_vars=[
            {
                "vorticity": "ERA5/prognostic/3d/vorticity",
                "divergence": "ERA5/prognostic/3d/divergence",
                "u": "ERA5/prognostic/3d/u_component_of_wind",
                "v": "ERA5/prognostic/3d/v_component_of_wind",
            }
        ],
        truncation=639,
        grid_file=grid_file,
    )
    batch = {
        "input": {
            "ERA5": {
                "ERA5/prognostic/3d/temperature": x_t.unsqueeze(0),
                "ERA5/prognostic/3d/vorticity": x_vo.unsqueeze(0),
                "ERA5/prognostic/3d/divergence": x_d.unsqueeze(0),
            }
        }
    }
    out = synth(batch)["input"]["ERA5"]

    from credit.preblock.regrid import Regridder

    grid = ReducedGaussianGrid.from_file(grid_file)
    dst_lat = 90.0 - 0.25 * np.arange(721)
    dst_lon = 0.25 * np.arange(1440)
    wfile = str(tmp_path / "n320_to_0p25.nc")
    reduced_gaussian_to_latlon_bilinear_weights(grid, dst_lat, dst_lon, wfile)
    keys = [f"ERA5/prognostic/3d/{v}" for v in ("temperature", "u_component_of_wind", "v_component_of_wind")]
    regrid = Regridder(weight_file=wfile, variables=keys)
    fields = regrid({"input": {"ERA5": out}})["input"]["ERA5"]

    ar = xr.open_zarr(
        gcsfs.GCSFileSystem(token="anon").get_mapper(
            "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
        ),
        chunks=None,
    )
    for var, tol in (("temperature", 0.25), ("u_component_of_wind", 0.5), ("v_component_of_wind", 0.5)):
        ref = ar[var].sel(time=t0, hybrid=137).values
        got = fields[f"ERA5/prognostic/3d/{var}"][0, 0, 0].numpy()
        rmse = float(np.sqrt(np.mean((got - ref) ** 2)))
        assert rmse < tol, f"{var}: RMSE {rmse} vs ARCO analysis-ready"
