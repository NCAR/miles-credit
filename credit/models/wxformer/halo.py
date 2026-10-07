"""
halo.py
-------
HaloExchange: populate cubed-sphere ghost cells with physically equivalent
data from adjacent faces before the encoder stack.

Background
~~~~~~~~~~
A cubed-sphere SE grid (e.g. ne120: 6 faces of 361x361) is padded per face to
the encoder tile size (384x384 for ne120).  Zero padding would put fake values
along every face edge, even though physically adjacent nodes on neighboring
faces carry real atmospheric data; under autoregressive rollout this leads to
edge artifacts that compound over time.

HaloExchange builds a full padded face with no scratch edge padding.  Every
logical padded coordinate is mapped back to the owning SE cell using the same
dominant-axis cubed-sphere ownership convention used to build ``se_index``.
This fills duplicated face edges, missing native face cells, and corner/vertex
ghost regions from physically equivalent owned cells.

Ghost cells are filled by **bilinear interpolation** of the owning face's four
bracketing grid cells (the continuous reprojection is only rounded to the
nearest cell for the legacy ``source_flat_index`` map, kept for
validation/tests).  Nearest-neighbor rounding would leave a small but
systematic discretization mismatch at every ghost cell; compounded over long
autoregressive rollouts this shows up as blur at the face seams.  Every
SE-owned cell is passed through exactly (never interpolated), so real input
data is never touched.  The SE-unowned cells inside the native face window --
the de-duplicated shared face edges, which reach this module as zeros from
``se_to_cube``'s scatter -- are filled like any other ghost cell.

Ghost-cell geometry (``geometry``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``se_index`` assigns rows/cols by *rank* of the gnomonic coordinate, so where
a grid index sits on the sphere depends on the grid:

``"scrip"`` (default)
    The angle of every grid row/col is read from the SE grid's SCRIP file (the
    one ``se_index`` was built from).  Ghost cells continue each face's grid
    lines past the edge by mirroring those angles about it -- the next
    equiangular element with the same symmetric GLL spacing.
``"linear"``
    Grid index is taken as linear in the gnomonic coordinate
    (``alpha = 2*col/(E-1) - 1``), extended linearly past the edge.  No SCRIP
    file needed.  This is the pre-SCRIP behavior and reproduces its tables
    exactly.  On the equiangular/GLL ne120 grid it samples ghost cells 0.1
    (ring 1) to ~2.3 (ring 12) cells away from the continued grid lines.

Supported grids
~~~~~~~~~~~~~~~
The face edge length E is read from ``se_index``, so any resolution works as
long as ALL of the following hold:

- ``se_index`` was built with credit-mesaclip's ``build_se_index.py``
  conventions: dominant-axis face assignment with ties to the lower face,
  per-face (alpha, beta) orientation as in ``_face_alpha_beta_to_xyz``, and
  faces 0/1 own all their edges, 2/3 their beta edges, 4/5 none.  (That script
  hard-codes ne120 -- ``NE``, ``NCOL_EXPECTED`` -- and must be edited for
  other resolutions.)
- The grid is a uniform equiangular cubed sphere whose rows/cols are lines of
  constant gnomonic coordinate (CESM ``neXXnp4`` SE grids).  ``"scrip"``
  checks this and raises otherwise.  Stretched (Schmidt) cubed spheres,
  regionally refined meshes (RRM) and physics grids (``pgN``) are NOT
  supported.
- Each side of the halo padding is narrower than half a face,
  ``pad < (E - 1) / 2`` (checked at construction).  Large encoder tiles on
  coarse grids (e.g. 192 for ne30's E=91) violate this; use smaller windows.

Implementation
~~~~~~~~~~~~~~
The module works on (B*6, C, E, E) tensors where the 6 faces are packed
into the batch dimension in face order (face 0 at indices [::6][0], etc.).
It returns (B*6, C, padded_size, padded_size).  The native face is located at
``[crop_top:crop_top+E, crop_left:crop_left+E]``; its SE-owned cells are
exact, never interpolated. Every other cell -- the ghost region outside that
window, plus the SE-unowned seam cells inside it -- is bilinearly
interpolated from physically equivalent owned SE cells on the owning face.

All index buffers are registered as nn.Buffers so .to(device) moves them.

Usage
-----
    halo = HaloExchange(adj_path, se_index_path, scrip_path, padded_size=384, crop_top=11, crop_left=11)
    x_pad = halo(x6)   # (B*6, C, 361, 361) -> (B*6, C, 384, 384)  for ne120
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A cube always has 6 faces; the face edge length is read from se_index.
NFACE = 6


class HaloExchange(nn.Module):
    """Populate padded cubed-sphere faces from physically equivalent cells.

    Parameters
    ----------
    adjacency_path : str | Path
        Path to the face adjacency file (e.g. ``se_face_adjacency_ne120.npz``).
        Kept as the feature gate for boundary-aware cube-sphere behavior.
    se_index_path : str | Path
        Path to the SE index (e.g. ``se_index_ne120.npy``).  Gives the face
        edge length and which logical cube cells are SE-owned.
    scrip_path : str | Path | None
        SCRIP grid file of the SE grid ``se_index`` was built from (e.g.
        ``ne120np4_pentagons_100310.nc``).  Required for ``geometry="scrip"``,
        ignored for ``"linear"``.
    geometry : {"scrip", "linear"}
        How grid indices map to positions on the cube; see the module
        docstring.  Default ``"scrip"``.
    padded_size : int
        Final per-face encoder size.
    crop_top, crop_left : int
        Offset of the native face inside the padded face.
    halo_size : int
        Deprecated compatibility argument.  If crop offsets are not supplied,
        it is used as the symmetric crop offset.
    """

    def __init__(
        self,
        adjacency_path: str | Path,
        se_index_path: str | Path | None = None,
        scrip_path: str | Path | None = None,
        padded_size: int | None = None,
        crop_top: int | None = None,
        crop_left: int | None = None,
        halo_size: int = 6,
        geometry: str = "scrip",
    ) -> None:
        super().__init__()
        # Touch the file so missing adjacency paths fail at construction.  The
        # full ghost map is derived from se_index ownership and cube geometry.
        np.load(str(adjacency_path))

        if se_index_path is None:
            raise ValueError("HaloExchange requires se_index_path for full ghost-cell exchange")
        if geometry not in ("scrip", "linear"):
            raise ValueError(f"HaloExchange geometry must be 'scrip' or 'linear', got {geometry!r}")
        if geometry == "scrip" and scrip_path is None:
            raise ValueError(
                "HaloExchange geometry='scrip' requires scrip_path (the SE grid's SCRIP file); "
                "pass it, or use geometry='linear'"
            )
        self.geometry = geometry

        # Face edge length: the smallest E whose 6*E*E cube holds every index
        # (same inference as CubedWXFormer).
        se_max = int(np.load(str(se_index_path)).max())
        self.nface_edge = int(np.ceil(np.sqrt((se_max + 1) / NFACE)))
        E = self.nface_edge

        self.halo_size = halo_size
        self.padded_size = int(padded_size or (E + 2 * halo_size))
        self.crop_top = int(halo_size if crop_top is None else crop_top)
        self.crop_left = int(halo_size if crop_left is None else crop_left)
        pads = (
            self.crop_top,
            self.crop_left,
            self.padded_size - self.crop_top - E,
            self.padded_size - self.crop_left - E,
        )
        if min(pads) < 0 or 2 * max(pads) >= E - 1:
            raise ValueError(
                f"HaloExchange padding {pads} (top, left, bottom, right) must be >= 0 and narrower than half "
                f"a face (E={E}); use a smaller padded_size (smaller attention windows) for this grid"
            )
        if geometry == "scrip":
            self.theta_col, self.theta_row = self._build_angle_tables(se_index_path, scrip_path)

        # Legacy nearest-neighbor map: kept only so existing ghost-map validation
        # (tests/test_cubed_wxformer.py) can still check every ghost cell
        # resolves to a real owned SE cell. forward() does not use this map.
        source_flat = self._build_source_flat_index(se_index_path)
        self.register_buffer("source_flat_index", torch.from_numpy(source_flat.astype(np.int64)))

        idx00, idx01, idx10, idx11, w00, w01, w10, w11 = self._build_interp_source(se_index_path)
        self.register_buffer("idx00", torch.from_numpy(idx00))
        self.register_buffer("idx01", torch.from_numpy(idx01))
        self.register_buffer("idx10", torch.from_numpy(idx10))
        self.register_buffer("idx11", torch.from_numpy(idx11))
        self.register_buffer("w00", torch.from_numpy(w00))
        self.register_buffer("w01", torch.from_numpy(w01))
        self.register_buffer("w10", torch.from_numpy(w10))
        self.register_buffer("w11", torch.from_numpy(w11))

        # Pass-through mask: a cell is taken verbatim from the input only if it
        # is inside the native window AND actually owned by an SE node. The SE
        # grid de-duplicates cells shared between faces, so a face's native
        # window contains cells with no owning node (on ne120: 4,324 cells --
        # two edge rings on faces 2/3, all four on the polar faces 4/5). Those
        # arrive from se_to_cube's scatter as zeros; they must be gathered from
        # the owning neighbour face like any other ghost cell, not passed
        # through. A fully-populated cube (every cell owned) is unaffected.
        se_owned = np.zeros(NFACE * E * E, dtype=bool)
        se_owned[np.load(str(se_index_path)).astype(np.int64)] = True
        native_mask = np.zeros((NFACE, self.padded_size, self.padded_size), dtype=bool)
        native_mask[:, self.crop_top : self.crop_top + E, self.crop_left : self.crop_left + E] = se_owned.reshape(
            NFACE, E, E
        )
        self.register_buffer(
            "native_mask", torch.from_numpy(native_mask).view(1, NFACE, 1, self.padded_size, self.padded_size)
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _face_alpha_beta_to_xyz(face: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, ...]:
        """Invert build_se_index.local_coords for logical cube coordinates."""
        x = np.empty_like(alpha, dtype=np.float64)
        y = np.empty_like(alpha, dtype=np.float64)
        z = np.empty_like(alpha, dtype=np.float64)

        m = face == 0
        x[m], y[m], z[m] = 1.0, alpha[m], beta[m]
        m = face == 1
        x[m], y[m], z[m] = -1.0, -alpha[m], beta[m]
        m = face == 2
        x[m], y[m], z[m] = -alpha[m], 1.0, beta[m]
        m = face == 3
        x[m], y[m], z[m] = alpha[m], -1.0, beta[m]
        m = face == 4
        x[m], y[m], z[m] = alpha[m], beta[m], 1.0
        m = face == 5
        x[m], y[m], z[m] = alpha[m], -beta[m], -1.0
        return x, y, z

    @staticmethod
    def _assign_faces(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Same dominant-axis ownership convention as build_se_index.assign_faces."""
        absx, absy, absz = np.abs(x), np.abs(y), np.abs(z)
        eps = 1e-12
        axis = np.argmax(np.stack([absx, absy - eps, absz - 2 * eps], axis=1), axis=1)
        face = np.empty_like(axis, dtype=np.int64)
        m = axis == 0
        face[m] = np.where(x[m] > 0, 0, 1)
        m = axis == 1
        face[m] = np.where(y[m] > 0, 2, 3)
        m = axis == 2
        face[m] = np.where(z[m] > 0, 4, 5)
        return face

    @staticmethod
    def _xyz_to_face_alpha_beta(
        face: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        alpha = np.empty_like(x, dtype=np.float64)
        beta = np.empty_like(x, dtype=np.float64)

        m = face == 0
        alpha[m], beta[m] = y[m] / x[m], z[m] / x[m]
        m = face == 1
        alpha[m], beta[m] = y[m] / x[m], -z[m] / x[m]
        m = face == 2
        alpha[m], beta[m] = -x[m] / y[m], z[m] / y[m]
        m = face == 3
        # Inverts face==3's (x,y,z) = (alpha, -1, beta): dividing by y=-1 flips
        # sign, so both terms need an extra negation to recover (alpha, beta).
        alpha[m], beta[m] = -x[m] / y[m], -z[m] / y[m]
        m = face == 4
        alpha[m], beta[m] = x[m] / z[m], y[m] / z[m]
        m = face == 5
        # Inverts face==5's (x,y,z) = (alpha, -beta, -1): dividing by z=-1
        # flips sign on the alpha term; the beta term already has the
        # compensating negation from y=-beta baked in, so it does not.
        alpha[m], beta[m] = -x[m] / z[m], y[m] / z[m]
        return alpha, beta

    def _build_angle_tables(self, se_index_path: str | Path, scrip_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Angle (arctan of the gnomonic coordinate) of every grid column and
        row on every face, read from the SCRIP node coordinates.

        Returns
        -------
        theta_col, theta_row : each (NFACE, nface_edge)
        """
        import xarray as xr

        E = self.nface_edge
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        with xr.open_dataset(scrip_path) as ds:
            lat = ds["grid_center_lat"].values.astype(np.float64)
            lon = ds["grid_center_lon"].values.astype(np.float64)
            if not ds["grid_center_lat"].attrs.get("units", "degrees").startswith("rad"):
                lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        if lat.size != se_idx.size:
            raise ValueError(f"SCRIP file {scrip_path} has {lat.size} nodes but se_index has {se_idx.size}")

        x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
        face = se_idx // (E * E)
        row = (se_idx // E) % E
        col = se_idx % E
        alpha, beta = self._xyz_to_face_alpha_beta(face, x, y, z)

        tables = []
        for line, angle in ((face * E + col, np.arctan(alpha)), (face * E + row, np.arctan(beta))):
            count = np.bincount(line, minlength=NFACE * E)
            theta = np.bincount(line, weights=angle, minlength=NFACE * E) / np.maximum(count, 1)
            if np.abs(angle - theta[line]).max() > 1e-6:
                raise ValueError(
                    f"SCRIP file {scrip_path} does not match se_index {se_index_path}: "
                    "nodes on the same grid line disagree on its angle (not a uniform equiangular cubed sphere?)"
                )
            theta = theta.reshape(NFACE, E)
            # Edge lines with no owned node (faces 2-5) lie exactly on the cube edge.
            empty = count.reshape(NFACE, E) == 0
            theta[:, 0] = np.where(empty[:, 0], -np.pi / 4, theta[:, 0])
            theta[:, -1] = np.where(empty[:, -1], np.pi / 4, theta[:, -1])
            tables.append(theta)
        return tables[0], tables[1]

    def _owner_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every padded coordinate on every face, find the owning face and
        the continuous (unrounded) row/col position on it.

        With ``geometry="scrip"``, cells inside the native window sit at their
        grid line's true angle and ghost cells continue the face's grid lines
        past each edge by mirroring the node angles about it (the next
        equiangular element, with the same symmetric GLL spacing).  With
        ``"linear"``, grid index is linear in the gnomonic coordinate.  Shared
        by the legacy nearest-neighbor map and the bilinear interpolation map
        below -- they only differ in how (owner_row, owner_col) is turned into
        cube indices.

        Returns
        -------
        owner_face, owner_row, owner_col : each (NFACE, padded_size, padded_size)
        """
        p = self.padded_size
        last = self.nface_edge - 1
        grid_index = np.arange(self.nface_edge, dtype=np.float64)

        def extend(theta: np.ndarray, crop: int) -> np.ndarray:
            j = np.arange(p) - crop
            return np.where(
                j < 0,
                2 * theta[0] - theta[np.clip(-j, 0, last)],
                np.where(j > last, 2 * theta[last] - theta[np.clip(2 * last - j, 0, last)], theta[np.clip(j, 0, last)]),
            )

        owner_face = np.empty((NFACE, p * p), dtype=np.int64)
        owner_row = np.empty((NFACE, p * p), dtype=np.float64)
        owner_col = np.empty((NFACE, p * p), dtype=np.float64)

        for f in range(NFACE):
            if self.geometry == "linear":
                alpha_line = 2.0 * (np.arange(p) - self.crop_left) / last - 1.0
                beta_line = 2.0 * (np.arange(p) - self.crop_top) / last - 1.0
            else:
                alpha_line = np.tan(extend(self.theta_col[f], self.crop_left))
                beta_line = np.tan(extend(self.theta_row[f], self.crop_top))
            beta, alpha = np.meshgrid(beta_line, alpha_line, indexing="ij")
            face = np.full(p * p, f, dtype=np.int64)
            x, y, z = self._face_alpha_beta_to_xyz(face, alpha.reshape(-1), beta.reshape(-1))
            owner = self._assign_faces(x, y, z)
            oa, ob = self._xyz_to_face_alpha_beta(owner, x, y, z)
            if self.geometry == "linear":
                owner_col[f] = np.clip((oa + 1.0) * last / 2.0, 0.0, last)
                owner_row[f] = np.clip((ob + 1.0) * last / 2.0, 0.0, last)
            else:
                for g in range(NFACE):
                    m = owner == g
                    owner_col[f, m] = np.interp(np.arctan(oa[m]), self.theta_col[g], grid_index)
                    owner_row[f, m] = np.interp(np.arctan(ob[m]), self.theta_row[g], grid_index)
            owner_face[f] = owner

        return (
            owner_face.reshape(NFACE, p, p),
            owner_row.reshape(NFACE, p, p),
            owner_col.reshape(NFACE, p, p),
        )

    def _build_source_flat_index(self, se_index_path: str | Path) -> np.ndarray:
        """Nearest-owned-cell ghost map. Kept only for the existing ghost-map
        ownership validation test; ``forward`` uses ``_build_interp_source`` instead.
        """
        E = self.nface_edge
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        cube_flat = NFACE * E * E
        owned = np.zeros(cube_flat, dtype=bool)
        owned[se_idx] = True
        owned_cube = owned.reshape(NFACE, E, E)
        identity = np.arange(cube_flat, dtype=np.int64).reshape(NFACE, E, E)

        owner_face, owner_row_f, owner_col_f = self._owner_geometry()
        owner_col = np.rint(owner_col_f).astype(np.int64)
        owner_row = np.rint(owner_row_f).astype(np.int64)
        owner_row = np.clip(owner_row, 0, E - 1)
        owner_col = np.clip(owner_col, 0, E - 1)

        flat = owner_face * (E * E) + owner_row * E + owner_col
        if not np.all(owned[flat]):
            bad = flat[~owned[flat]][:10]
            raise RuntimeError(f"ghost map produced non-owned cube cells: {bad.tolist()}")

        face_chunks = []
        for f in range(NFACE):
            face_map = flat[f]
            native = face_map[self.crop_top : self.crop_top + E, self.crop_left : self.crop_left + E]
            native[owned_cube[f]] = identity[f][owned_cube[f]]
            face_chunks.append(face_map)

        return np.stack(face_chunks, axis=0)

    def _build_interp_source(
        self, se_index_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear ghost map: 4 bracketing owned cells + weights per padded coordinate.

        Same reprojection as ``_build_source_flat_index``, but the continuous
        (owner_row, owner_col) is bracketed by its 4 neighboring grid cells
        instead of rounded to the nearest one, so ghost cells get an interpolated
        value rather than a discretized nearest-neighbor one.

        Some cube faces don't own all four of their own edge rings under the SE
        dedup convention -- e.g. the two polar faces own none of their edges at
        all, always sourced from the surrounding equatorial faces instead. The
        continuous dominant-axis owner assignment (``_owner_geometry``) doesn't
        know about that discrete convention, so a bracket corner can legitimately
        land exactly on such an unowned ring. When that happens, that single
        corner falls back to the already-validated nearest-owned cell (see
        ``_build_source_flat_index``) instead of raising -- this only affects the
        innermost ring of the halo immediately adjacent to a face's true
        boundary; every other ghost cell still gets full bilinear interpolation.
        """
        E = self.nface_edge
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        cube_flat = NFACE * E * E
        owned = np.zeros(cube_flat, dtype=bool)
        owned[se_idx] = True

        owner_face, row_f, col_f = self._owner_geometry()

        col0 = np.floor(col_f).astype(np.int64)
        row0 = np.floor(row_f).astype(np.int64)
        col1 = np.minimum(col0 + 1, E - 1)
        row1 = np.minimum(row0 + 1, E - 1)
        frac_col = col_f - col0
        frac_row = row_f - row0

        # Nearest-owned fallback for corners that land on an unowned edge ring.
        nearest_row = np.clip(np.rint(row_f).astype(np.int64), 0, E - 1)
        nearest_col = np.clip(np.rint(col_f).astype(np.int64), 0, E - 1)
        nearest_flat = owner_face * (E * E) + nearest_row * E + nearest_col
        if not np.all(owned[nearest_flat]):
            bad = nearest_flat[~owned[nearest_flat]][:10]
            raise RuntimeError(f"nearest-owned fallback produced non-owned cube cells: {bad.tolist()}")

        def flat_index(row, col):
            return owner_face * (E * E) + row * E + col

        idx = {
            "00": flat_index(row0, col0),
            "01": flat_index(row0, col1),
            "10": flat_index(row1, col0),
            "11": flat_index(row1, col1),
        }
        for name, flat in idx.items():
            unowned = ~owned[flat]
            if unowned.any():
                flat[unowned] = nearest_flat[unowned]

        w00 = (1.0 - frac_row) * (1.0 - frac_col)
        w01 = (1.0 - frac_row) * frac_col
        w10 = frac_row * (1.0 - frac_col)
        w11 = frac_row * frac_col

        return (
            idx["00"].astype(np.int64),
            idx["01"].astype(np.int64),
            idx["10"].astype(np.int64),
            idx["11"].astype(np.int64),
            w00.astype(np.float32),
            w01.astype(np.float32),
            w10.astype(np.float32),
            w11.astype(np.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad faces with a full ghost-cell exchange.

        Ghost cells are bilinearly interpolated from the owning face's 4
        bracketing cells. SE-owned cells are always the exact input, never
        interpolated; SE-unowned cells inside the native face window are
        gathered like ghost cells, since the scatter leaves them at zero.

        Parameters
        ----------
        x : (B*6, C, H, W)  where H=W=nface_edge

        Returns
        -------
        (B*6, C, padded_size, padded_size)
        """
        B6, C, H, W = x.shape
        B = B6 // NFACE
        if H != self.nface_edge or W != self.nface_edge:
            raise ValueError(f"HaloExchange expects {self.nface_edge}x{self.nface_edge} faces, got {H}x{W}")

        p = self.padded_size
        cube = x.reshape(B, NFACE, C, H * W).permute(0, 2, 1, 3).reshape(B, C, NFACE * H * W)

        def gather(idx: torch.Tensor) -> torch.Tensor:
            return cube[:, :, idx.reshape(-1)].reshape(B, C, NFACE, p, p)

        w00, w01, w10, w11 = (w.to(dtype=cube.dtype) for w in (self.w00, self.w01, self.w10, self.w11))
        interpolated = (
            gather(self.idx00) * w00 + gather(self.idx01) * w01 + gather(self.idx10) * w10 + gather(self.idx11) * w11
        )
        interpolated = interpolated.permute(0, 2, 1, 3, 4)  # (B, NFACE, C, p, p)

        pad_top, pad_left = self.crop_top, self.crop_left
        pad_bottom, pad_right = p - self.crop_top - H, p - self.crop_left - W
        x_native = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom)).reshape(B, NFACE, C, p, p)
        return torch.where(self.native_mask, x_native, interpolated).reshape(B6, C, p, p)
