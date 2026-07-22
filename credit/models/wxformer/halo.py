"""
halo.py
-------
HaloExchange: populate cubed-sphere ghost cells with physically equivalent
data from adjacent faces before the encoder stack.

Background
~~~~~~~~~~
The ne120 cubed-sphere SE grid has 6 faces (361x361).  Before encoding,
each face is padded to 384x384 with zeros.  The padding along the face
edges is therefore zero-filled, even though physically adjacent nodes on
neighboring faces carry real atmospheric data.  During autoregressive
rollout this leads to edge artifacts that compound over time.

HaloExchange builds a full padded face with no scratch edge padding.  Every
logical padded coordinate is mapped back to the owning SE cell using the same
dominant-axis cubed-sphere ownership convention used to build ``se_index``.
This fills duplicated face edges, missing native face cells, and corner/vertex
ghost regions from physically equivalent owned cells.

Ghost cells are filled by **bilinear interpolation** of the owning face's four
bracketing grid cells (the continuous alpha/beta reprojection is only rounded
to the nearest cell for the legacy ``source_flat_index`` map, kept for
validation/tests). Nearest-neighbor rounding leaves a small but systematic
discretization mismatch at every ghost cell relative to the true value the
neighboring face would report at that exact location; compounded over long
autoregressive rollouts this shows up as blur at the face seams. The native
face window is always passed through exactly (never interpolated), so real
input data is never touched.

Geometry
~~~~~~~~
The 12 directed face-edge pairs are (derived from ne120 topology):

  Equatorial <-> equatorial (col strips, 1:1 forward):
    F0 right   (col=360) <- F2 col=1..h
    F0 left    (col=0)   -> F3 col=359..359-h+1  (written to F0 left halo)
    F1 left    (col=0)   <- F2 col=359..359-h+1
    F1 right   (col=360) <- F3 col=1..h

  Equatorial top/bottom <-> polar col strips (359 nodes, reversed for some):
    F0 top     (row=0)   <- F5 col=359..359-h+1  (row-reversed)
    F0 bottom  (row=360) <- F4 col=359..359-h+1  (forward)
    F1 top     (row=0)   <- F5 col=1..h           (forward)
    F1 bottom  (row=360) <- F4 col=1..h           (row-reversed)

  F2/F3 top/bottom <-> polar row strips (359 nodes):
    F2 top     (row=0)   <- F5 row=1..h           (col-reversed)
    F2 bottom  (row=360) <- F4 row=359..359-h+1   (col-reversed)
    F3 top     (row=0)   <- F5 row=359..359-h+1   (col-forward)
    F3 bottom  (row=360) <- F4 row=1..h           (col-forward)

Implementation
~~~~~~~~~~~~~~
The module works on (B*6, C, H, W) tensors where the 6 faces are packed
into the batch dimension in face order (face 0 at indices [::6][0], etc.).
It returns (B*6, C, padded_size, padded_size).  The native face is located at
``[crop_top:crop_top+H, crop_left:crop_left+W]`` (always exact, never
interpolated); all other cells are bilinearly interpolated from physically
equivalent owned SE cells on the owning face.

All index buffers are registered as nn.Buffers so .to(device) moves them.

Usage
-----
    halo = HaloExchange(adj_path, se_index_path, padded_size=384, crop_top=11, crop_left=11)
    x_pad = halo(x6)   # (B*6, C, 361, 361) -> (B*6, C, 384, 384)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ne120 constants
NFACE = 6
NFACE_EDGE = 361  # 0..360


class HaloExchange(nn.Module):
    """Populate padded cubed-sphere faces from physically equivalent cells.

    Parameters
    ----------
    adjacency_path : str | Path
        Path to ``se_face_adjacency_ne120.npz``.  Kept as the feature gate for
        boundary-aware cube-sphere behavior.
    se_index_path : str | Path
        Path to ``se_index_ne120.npy``.  Used to determine which logical cube
        cells are SE-owned.
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
        padded_size: int | None = None,
        crop_top: int | None = None,
        crop_left: int | None = None,
        halo_size: int = 6,
    ) -> None:
        super().__init__()
        # Touch the file so missing adjacency paths fail at construction.  The
        # full ghost map is derived from se_index ownership and cube geometry.
        np.load(str(adjacency_path))

        if se_index_path is None:
            raise ValueError("HaloExchange requires se_index_path for full ghost-cell exchange")

        self.halo_size = halo_size
        self.padded_size = int(padded_size or (NFACE_EDGE + 2 * halo_size))
        self.crop_top = int(halo_size if crop_top is None else crop_top)
        self.crop_left = int(halo_size if crop_left is None else crop_left)

        # Legacy nearest-neighbor map: kept only so existing ghost-map validation
        # (tests/test_cube_sphere_wxformer.py) can still check every ghost cell
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

        native_mask = np.zeros((self.padded_size, self.padded_size), dtype=bool)
        native_mask[self.crop_top : self.crop_top + NFACE_EDGE, self.crop_left : self.crop_left + NFACE_EDGE] = True
        self.register_buffer(
            "native_mask", torch.from_numpy(native_mask).view(1, 1, self.padded_size, self.padded_size)
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

    def _owner_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every padded coordinate on every face, find the owning face and
        the continuous (unrounded) local (alpha, beta) coordinate on it.

        Shared by the legacy nearest-neighbor map and the bilinear interpolation
        map below -- both need the same reprojection, they only differ in how
        the continuous (owner_alpha, owner_beta) is turned into cube indices.

        Returns
        -------
        owner_face, owner_alpha, owner_beta : each (NFACE, padded_size, padded_size)
        """
        p = self.padded_size
        rr, cc = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        rr = rr.reshape(-1)
        cc = cc.reshape(-1)

        owner_face = np.empty((NFACE, p * p), dtype=np.int64)
        owner_alpha = np.empty((NFACE, p * p), dtype=np.float64)
        owner_beta = np.empty((NFACE, p * p), dtype=np.float64)

        for f in range(NFACE):
            face = np.full(rr.shape, f, dtype=np.int64)
            row = rr.astype(np.float64) - self.crop_top
            col = cc.astype(np.float64) - self.crop_left
            alpha = 2.0 * col / (NFACE_EDGE - 1) - 1.0
            beta = 2.0 * row / (NFACE_EDGE - 1) - 1.0

            x, y, z = self._face_alpha_beta_to_xyz(face, alpha, beta)
            owner = self._assign_faces(x, y, z)
            oa, ob = self._xyz_to_face_alpha_beta(owner, x, y, z)
            owner_face[f], owner_alpha[f], owner_beta[f] = owner, oa, ob

        return (
            owner_face.reshape(NFACE, p, p),
            owner_alpha.reshape(NFACE, p, p),
            owner_beta.reshape(NFACE, p, p),
        )

    def _build_source_flat_index(self, se_index_path: str | Path) -> np.ndarray:
        """Nearest-owned-cell ghost map. Kept only for the existing ghost-map
        ownership validation test; ``forward`` uses ``_build_interp_source`` instead.
        """
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        cube_flat = NFACE * NFACE_EDGE * NFACE_EDGE
        owned = np.zeros(cube_flat, dtype=bool)
        owned[se_idx] = True
        owned_cube = owned.reshape(NFACE, NFACE_EDGE, NFACE_EDGE)
        identity = np.arange(cube_flat, dtype=np.int64).reshape(NFACE, NFACE_EDGE, NFACE_EDGE)

        owner_face, owner_alpha, owner_beta = self._owner_geometry()
        owner_col = np.rint((owner_alpha + 1.0) * (NFACE_EDGE - 1) / 2.0).astype(np.int64)
        owner_row = np.rint((owner_beta + 1.0) * (NFACE_EDGE - 1) / 2.0).astype(np.int64)
        owner_row = np.clip(owner_row, 0, NFACE_EDGE - 1)
        owner_col = np.clip(owner_col, 0, NFACE_EDGE - 1)

        flat = owner_face * (NFACE_EDGE * NFACE_EDGE) + owner_row * NFACE_EDGE + owner_col
        if not np.all(owned[flat]):
            bad = flat[~owned[flat]][:10]
            raise RuntimeError(f"ghost map produced non-owned cube cells: {bad.tolist()}")

        face_chunks = []
        for f in range(NFACE):
            face_map = flat[f]
            native = face_map[self.crop_top : self.crop_top + NFACE_EDGE, self.crop_left : self.crop_left + NFACE_EDGE]
            native[owned_cube[f]] = identity[f][owned_cube[f]]
            face_chunks.append(face_map)

        return np.stack(face_chunks, axis=0)

    def _build_interp_source(
        self, se_index_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear ghost map: 4 bracketing owned cells + weights per padded coordinate.

        Same reprojection as ``_build_source_flat_index``, but the continuous
        (owner_alpha, owner_beta) is bracketed by its 4 neighboring grid cells
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
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        cube_flat = NFACE * NFACE_EDGE * NFACE_EDGE
        owned = np.zeros(cube_flat, dtype=bool)
        owned[se_idx] = True

        owner_face, owner_alpha, owner_beta = self._owner_geometry()

        col_f = np.clip((owner_alpha + 1.0) * (NFACE_EDGE - 1) / 2.0, 0.0, NFACE_EDGE - 1)
        row_f = np.clip((owner_beta + 1.0) * (NFACE_EDGE - 1) / 2.0, 0.0, NFACE_EDGE - 1)

        col0 = np.floor(col_f).astype(np.int64)
        row0 = np.floor(row_f).astype(np.int64)
        col1 = np.minimum(col0 + 1, NFACE_EDGE - 1)
        row1 = np.minimum(row0 + 1, NFACE_EDGE - 1)
        frac_col = col_f - col0
        frac_row = row_f - row0

        # Nearest-owned fallback for corners that land on an unowned edge ring.
        nearest_row = np.clip(np.rint(row_f).astype(np.int64), 0, NFACE_EDGE - 1)
        nearest_col = np.clip(np.rint(col_f).astype(np.int64), 0, NFACE_EDGE - 1)
        nearest_flat = owner_face * (NFACE_EDGE * NFACE_EDGE) + nearest_row * NFACE_EDGE + nearest_col
        if not np.all(owned[nearest_flat]):
            bad = nearest_flat[~owned[nearest_flat]][:10]
            raise RuntimeError(f"nearest-owned fallback produced non-owned cube cells: {bad.tolist()}")

        def flat_index(row, col):
            return owner_face * (NFACE_EDGE * NFACE_EDGE) + row * NFACE_EDGE + col

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
        bracketing cells; the native face window is always the exact input,
        never interpolated.

        Parameters
        ----------
        x : (B*6, C, H, W)  where H=W=361=NFACE_EDGE

        Returns
        -------
        (B*6, C, padded_size, padded_size)
        """
        B6, C, H, W = x.shape
        B = B6 // NFACE
        if H != NFACE_EDGE or W != NFACE_EDGE:
            raise ValueError(f"HaloExchange expects {NFACE_EDGE}x{NFACE_EDGE} faces, got {H}x{W}")

        p = self.padded_size
        cube = x.reshape(B, NFACE, C, H * W).permute(0, 2, 1, 3).reshape(B, C, NFACE * H * W)

        def gather(idx: torch.Tensor) -> torch.Tensor:
            return cube[:, :, idx.reshape(-1)].reshape(B, C, NFACE, p, p)

        w00, w01, w10, w11 = (w.to(dtype=cube.dtype) for w in (self.w00, self.w01, self.w10, self.w11))
        interpolated = (
            gather(self.idx00) * w00 + gather(self.idx01) * w01 + gather(self.idx10) * w10 + gather(self.idx11) * w11
        )
        interpolated = interpolated.permute(0, 2, 1, 3, 4).reshape(B6, C, p, p)

        pad_top, pad_left = self.crop_top, self.crop_left
        pad_bottom, pad_right = p - self.crop_top - H, p - self.crop_left - W
        x_native = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return torch.where(self.native_mask, x_native, interpolated)
