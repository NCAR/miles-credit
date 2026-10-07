"""Regression tests for the cubed-sphere halo on the ne120 grid.

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

The second test pins the ghost-cell geometry: the ne120 grid is equiangular
with GLL nodes inside each element, so a linear index <-> gnomonic-coordinate
map places ghost cells up to ~2 cells away from the face's continued grid lines.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from credit.models.wxformer.cubed_wxformer import NFACE
from credit.models.wxformer.halo import HaloExchange, NFACE_EDGE

CROP = 11
PADDED = 384


def _static_dir():
    return Path(
        os.environ.get(
            "MESACLIP_STATIC",
            Path(__file__).resolve().parents[2] / "credit-mesaclip" / "mesaclip" / "static",
        )
    )


def _scrip_path():
    return Path(
        os.environ.get(
            "NE120_SCRIP",
            "/glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc",
        )
    )


def _ne120_halo():
    static = _static_dir()
    se_index_path = static / "se_index_ne120.npy"
    adjacency_path = static / "se_face_adjacency_ne120.npz"
    scrip_path = _scrip_path()
    if not se_index_path.exists() or not adjacency_path.exists() or not scrip_path.exists():
        pytest.skip("ne120 cubed-sphere static files or SCRIP grid file are not available")
    halo = HaloExchange(
        adjacency_path=adjacency_path,
        se_index_path=se_index_path,
        scrip_path=scrip_path,
        padded_size=PADDED,
        crop_top=CROP,
        crop_left=CROP,
    )
    return halo, se_index_path, scrip_path


def test_halo_leaves_no_unfilled_cells_in_the_native_face():
    """Every SE-unowned cell inside the native face is filled from a neighbour."""
    halo, se_index_path, _ = _ne120_halo()

    # Mirror CubedWXFormer.se_to_cube: scatter a constant field onto the owned
    # SE nodes and leave every unowned cube cell at the zero fill, so any cell
    # the halo fails to repair shows up as an exact 0.0 against a constant 1.0.
    se_index = torch.from_numpy(np.load(se_index_path).astype(np.int64))
    cube = torch.zeros(1, 1, NFACE * NFACE_EDGE * NFACE_EDGE)
    cube[:, :, se_index] = 1.0
    x6 = (
        cube.reshape(1, 1, NFACE, NFACE_EDGE, NFACE_EDGE)
        .permute(0, 2, 1, 3, 4)
        .reshape(NFACE, 1, NFACE_EDGE, NFACE_EDGE)
    )

    padded = halo(x6)
    native = padded[:, :, CROP : CROP + NFACE_EDGE, CROP : CROP + NFACE_EDGE]

    unfilled = int((native == 0.0).sum())
    assert unfilled == 0, (
        f"{unfilled} cells inside the native face window are still at the "
        f"scatter's zero fill after HaloExchange "
        f"(per face: {[int((native[f] == 0.0).sum()) for f in range(NFACE)]}). "
        "These are the SE-deduplicated face-seam cells; they must be gathered "
        "from the owning neighbour face, not left at zero."
    )
    torch.testing.assert_close(native, torch.ones_like(native))


def _analytic_padded_xyz():
    """Unit vectors of every padded cell, from the analytic ne120 node layout.

    120 equiangular elements per face edge, 4 GLL nodes per element (shared
    endpoints), so node angles are known in closed form.  Ghost cells continue
    each face's grid lines past the edge by mirroring the node angles about it.
    Built independently of the SCRIP file the module reads.
    """
    ne = 120
    gll = np.array([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)])
    theta = (-np.pi / 4 + (np.pi / 2 / ne) * (np.arange(ne)[:, None] + (1.0 + gll) / 2.0)).ravel()
    theta = np.append(theta, np.pi / 4)

    j = np.arange(PADDED) - CROP
    last = NFACE_EDGE - 1
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


def _ghost_ring_errors(halo, se_index_path, scrip_path, k=64):
    """RMS halo error per ghost ring for a smooth field, in units of the field's
    largest possible change across one cell (k * mean node spacing)."""
    import xarray as xr

    with xr.open_dataset(scrip_path) as ds:
        lat = np.deg2rad(ds["grid_center_lat"].values.astype(np.float64))
        lon = np.deg2rad(ds["grid_center_lon"].values.astype(np.float64))
    nodes = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)

    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)

    def field(p):
        return np.sin(k * (p @ direction) + 0.4)

    cube = np.zeros(NFACE * NFACE_EDGE * NFACE_EDGE)
    cube[np.load(se_index_path).astype(np.int64)] = field(nodes)
    x6 = torch.from_numpy(cube.reshape(NFACE, 1, NFACE_EDGE, NFACE_EDGE))
    with torch.no_grad():
        padded = halo(x6)[:, 0].numpy()

    one_cell = k * (np.pi / 2) / (NFACE_EDGE - 1)
    err = (padded - field(_analytic_padded_xyz())) / one_cell

    rr, cc = np.meshgrid(np.arange(PADDED) - CROP, np.arange(PADDED) - CROP, indexing="ij")
    last = NFACE_EDGE - 1
    ring = np.maximum(np.maximum(np.maximum(-rr, rr - last), np.maximum(-cc, cc - last)), 0)
    return {d: float(np.sqrt(np.mean(err[:, ring == d] ** 2))) for d in range(int(ring.max()) + 1)}


def test_halo_ghost_cells_follow_the_true_grid_geometry():
    """Ghost cells hold the field value at the continued equiangular/GLL grid
    location, not at a linear-in-gnomonic-coordinate approximation of it."""
    halo, se_index_path, scrip_path = _ne120_halo()
    errors = _ghost_ring_errors(halo, se_index_path, scrip_path)

    # Native window (owned cells exact, seam cells copied from their owner).
    assert errors[0] < 1e-6, errors
    # Bilinear interpolation alone leaves <= 0.007 here; the linear-alpha
    # geometry gave 0.054 at ring 1 rising to 0.62 at ring 12.
    worst = max(v for d, v in errors.items() if d > 0)
    assert worst < 0.03, {d: round(v, 4) for d, v in errors.items()}
