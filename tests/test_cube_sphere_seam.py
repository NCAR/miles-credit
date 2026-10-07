"""Regression tests for the cubed-sphere halo on real CESM SE grids.

``CubedWXFormer.se_to_cube`` scatters ``ncol`` SE nodes into a dense
``6 x E x E`` cube.  The SE grid de-duplicates cells shared between faces, so
some cube cells have no owning SE node and stay at the ``new_zeros`` fill.  On
ne120 that is 4,324 cells per channel, all of them on face seams (faces 2/3
lose two edge rings each, the two polar faces 4/5 lose all four).

``HaloExchange`` is what is supposed to repair them: it reprojects every padded
coordinate back to a physically equivalent SE-owned cell.  The first test pins
the property that matters for training -- after the pad step, nothing inside
the native face window is still sitting at the scatter's zero fill.

Left broken, those cells feed a ring of "zero" (i.e. climatological mean, in
normalized units) into the encoder along every face seam at every step, which
is a compounding artifact source under autoregressive rollout.

The accuracy test pins the ghost-cell geometry: the grids are equiangular with
GLL nodes inside each element, so a linear index <-> gnomonic-coordinate map
places ghost cells up to ~2 cells away from the face's continued grid lines.

ne120 uses the shipped ``se_index_ne120.npy``; ne30 and ne16 build their index
from the CESM SCRIP file with ``_build_se_index`` (a parameterized port of
credit-mesaclip's ``build_se_index.py``, checked against the shipped ne120 one).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from credit.models.wxformer.cubed_wxformer import NFACE
from credit.models.wxformer.halo import HaloExchange

CROP = 11
SCRIP_FILES = {
    120: "ne120np4_pentagons_100310.nc",
    30: "ne30np4_091226_pentagons.nc",
    16: "ne16np4_110512_pentagons.nc",
}


def _static_dir():
    return Path(
        os.environ.get(
            "MESACLIP_STATIC",
            Path(__file__).resolve().parents[2] / "credit-mesaclip" / "mesaclip" / "static",
        )
    )


def _scrip_path(ne):
    scripgrids = Path(os.environ.get("CESM_SCRIPGRIDS", "/glade/campaign/cesm/cesmdata/inputdata/share/scripgrids"))
    return scripgrids / SCRIP_FILES[ne]


def _scrip_xyz(scrip_path):
    import xarray as xr

    with xr.open_dataset(scrip_path) as ds:
        lat = np.deg2rad(ds["grid_center_lat"].values.astype(np.float64))
        lon = np.deg2rad(ds["grid_center_lon"].values.astype(np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _build_se_index(scrip_path, ne):
    """credit-mesaclip build_se_index.py for any ne (np=4): dominant-axis face,
    rank-ordered (alpha, beta) per face, shared edges owned by lower faces."""
    xyz = _scrip_xyz(scrip_path)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    face = HaloExchange._assign_faces(x, y, z)
    alpha, beta = HaloExchange._xyz_to_face_alpha_beta(face, x, y, z)
    edge = 3 * ne + 1
    offsets = [(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1)]  # (col, row) offset per face
    row = np.empty(face.size, dtype=np.int64)
    col = np.empty(face.size, dtype=np.int64)
    for f in range(NFACE):
        m = face == f
        a = np.round(alpha[m], 8)
        b = np.round(beta[m], 8)
        col[m] = np.searchsorted(np.unique(a), a) + offsets[f][0]
        row[m] = np.searchsorted(np.unique(b), b) + offsets[f][1]
    se_index = face * edge * edge + row * edge + col
    assert np.unique(se_index).size == se_index.size, "index is not a bijection"
    return se_index


def _grid(ne, tmp_path):
    """(se_index_path, adjacency_path, scrip_path, face edge) for one grid, or skip."""
    scrip_path = _scrip_path(ne)
    if not scrip_path.exists():
        pytest.skip(f"CESM SCRIP file for ne{ne} is not available")
    if ne == 120:
        se_index_path = _static_dir() / "se_index_ne120.npy"
        adjacency_path = _static_dir() / "se_face_adjacency_ne120.npz"
        if not se_index_path.exists() or not adjacency_path.exists():
            pytest.skip("ne120 cubed-sphere static files are not available")
    else:
        se_index_path = tmp_path / f"se_index_ne{ne}.npy"
        np.save(se_index_path, _build_se_index(scrip_path, ne))
        adjacency_path = tmp_path / f"se_face_adjacency_ne{ne}.npz"
        np.savez(adjacency_path)
    return se_index_path, adjacency_path, scrip_path, 3 * ne + 1


def _halo(grid, geometry):
    se_index_path, adjacency_path, scrip_path, edge = grid
    return HaloExchange(
        adjacency_path=adjacency_path,
        se_index_path=se_index_path,
        scrip_path=scrip_path,
        geometry=geometry,
        padded_size=edge + 2 * CROP + 1,
        crop_top=CROP,
        crop_left=CROP,
    )


def test_index_builder_matches_shipped_ne120_index(tmp_path):
    se_index_path, _, scrip_path, _ = _grid(120, tmp_path)
    np.testing.assert_array_equal(_build_se_index(scrip_path, 120), np.load(se_index_path))


@pytest.mark.parametrize("geometry", ["scrip", "linear"])
@pytest.mark.parametrize("ne", [120, 30, 16])
def test_halo_leaves_no_unfilled_cells_in_the_native_face(tmp_path, ne, geometry):
    """Every SE-unowned cell inside the native face is filled from a neighbour."""
    grid = _grid(ne, tmp_path)
    se_index_path, _, _, edge = grid
    halo = _halo(grid, geometry)

    # Mirror CubedWXFormer.se_to_cube: scatter a constant field onto the owned
    # SE nodes and leave every unowned cube cell at the zero fill, so any cell
    # the halo fails to repair shows up as an exact 0.0 against a constant 1.0.
    se_index = torch.from_numpy(np.load(se_index_path).astype(np.int64))
    cube = torch.zeros(1, 1, NFACE * edge * edge)
    cube[:, :, se_index] = 1.0
    x6 = cube.reshape(1, 1, NFACE, edge, edge).permute(0, 2, 1, 3, 4).reshape(NFACE, 1, edge, edge)

    padded = halo(x6)
    native = padded[:, :, CROP : CROP + edge, CROP : CROP + edge]

    unfilled = int((native == 0.0).sum())
    assert unfilled == 0, (
        f"{unfilled} cells inside the native face window are still at the "
        f"scatter's zero fill after HaloExchange "
        f"(per face: {[int((native[f] == 0.0).sum()) for f in range(NFACE)]}). "
        "These are the SE-deduplicated face-seam cells; they must be gathered "
        "from the owning neighbour face, not left at zero."
    )
    torch.testing.assert_close(native, torch.ones_like(native))


def _analytic_padded_xyz(ne, padded):
    """Unit vectors of every padded cell, from the analytic neXXnp4 node layout.

    ne equiangular elements per face edge, 4 GLL nodes per element (shared
    endpoints), so node angles are known in closed form.  Ghost cells continue
    each face's grid lines past the edge by mirroring the node angles about it.
    Built independently of the SCRIP file the module reads.
    """
    gll = np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)])
    theta = (-np.pi / 4 + (np.pi / 2 / ne) * (np.arange(ne)[:, None] + (1.0 + gll) / 2.0)).ravel()
    theta = np.append(theta, np.pi / 4)

    j = np.arange(padded) - CROP
    last = 3 * ne
    ext = np.where(
        j < 0,
        2 * theta[0] - theta[np.clip(-j, 0, last)],
        np.where(j > last, 2 * theta[last] - theta[np.clip(2 * last - j, 0, last)], theta[np.clip(j, 0, last)]),
    )
    beta, alpha = np.meshgrid(np.tan(ext), np.tan(ext), indexing="ij")
    xyz = np.stack(
        [np.stack(HaloExchange._face_alpha_beta_to_xyz(np.full(alpha.shape, f), alpha, beta), -1) for f in range(NFACE)]
    )
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


def _ghost_ring_errors(halo, se_index_path, scrip_path, ne, k):
    """RMS halo error per ghost ring for a smooth field, in units of the field's
    largest possible change across one cell (k * mean node spacing)."""
    edge = 3 * ne + 1
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)

    def field(p):
        return np.sin(k * (p @ direction) + 0.4)

    cube = np.zeros(NFACE * edge * edge)
    cube[np.load(se_index_path).astype(np.int64)] = field(_scrip_xyz(scrip_path))
    x6 = torch.from_numpy(cube.reshape(NFACE, 1, edge, edge))
    with torch.no_grad():
        padded = halo(x6)[:, 0].numpy()

    one_cell = k * (np.pi / 2) / (edge - 1)
    err = (padded - field(_analytic_padded_xyz(ne, halo.padded_size))) / one_cell

    p = halo.padded_size
    rr, cc = np.meshgrid(np.arange(p) - CROP, np.arange(p) - CROP, indexing="ij")
    last = edge - 1
    ring = np.maximum(np.maximum(np.maximum(-rr, rr - last), np.maximum(-cc, cc - last)), 0)
    return {d: float(np.sqrt(np.mean(err[:, ring == d] ** 2))) for d in range(int(ring.max()) + 1)}


@pytest.mark.parametrize("ne", [120, 30, 16])
def test_scrip_halo_ghost_cells_follow_the_true_grid_geometry(tmp_path, ne):
    """With geometry="scrip", ghost cells hold the field value at the continued
    equiangular/GLL grid location, on every resolution."""
    grid = _grid(ne, tmp_path)
    se_index_path, _, scrip_path, edge = grid
    # Same ~35 cells per wavelength on every grid (k=64 on ne120).
    k = 64 * (edge - 1) / 360
    errors = _ghost_ring_errors(_halo(grid, "scrip"), se_index_path, scrip_path, ne, k)

    # Native window (owned cells exact, seam cells copied from their owner).
    assert errors[0] < 1e-6, errors
    # Bilinear interpolation alone leaves <= 0.007 on ne120; the linear-alpha
    # geometry gave 0.054 at ring 1 rising to 0.62 at ring 12.
    worst = max(v for d, v in errors.items() if d > 0)
    assert worst < 0.03, {d: round(v, 4) for d, v in errors.items()}


def test_scrip_geometry_requires_scrip_path(tmp_path):
    se_index_path, adjacency_path, _, edge = _grid(30, tmp_path)
    with pytest.raises(ValueError, match="scrip_path"):
        HaloExchange(adjacency_path=adjacency_path, se_index_path=se_index_path, padded_size=edge + 24, crop_top=11)


def test_halo_wider_than_half_a_face_is_rejected(tmp_path):
    se_index_path, adjacency_path, scrip_path, edge = _grid(16, tmp_path)
    with pytest.raises(ValueError, match="half"):
        HaloExchange(
            adjacency_path=adjacency_path,
            se_index_path=se_index_path,
            scrip_path=scrip_path,
            padded_size=edge + 2 * 30,
            crop_top=30,
            crop_left=30,
        )
