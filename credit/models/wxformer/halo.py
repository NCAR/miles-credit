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
``[crop_top:crop_top+H, crop_left:crop_left+W]``; all other cells are gathered
from physically equivalent owned SE cells.

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

        source_flat = self._build_source_flat_index(se_index_path)
        self.register_buffer("source_flat_index", torch.from_numpy(source_flat.astype(np.int64)))

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
        alpha[m], beta[m] = x[m] / y[m], z[m] / y[m]
        m = face == 4
        alpha[m], beta[m] = x[m] / z[m], y[m] / z[m]
        m = face == 5
        alpha[m], beta[m] = x[m] / z[m], -y[m] / z[m]
        return alpha, beta

    def _build_source_flat_index(self, se_index_path: str | Path) -> np.ndarray:
        se_idx = np.load(str(se_index_path)).astype(np.int64)
        cube_flat = NFACE * NFACE_EDGE * NFACE_EDGE
        owned = np.zeros(cube_flat, dtype=bool)
        owned[se_idx] = True

        p = self.padded_size
        rr, cc = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        rr = rr.reshape(-1)
        cc = cc.reshape(-1)

        face_chunks = []
        owned_cube = owned.reshape(NFACE, NFACE_EDGE, NFACE_EDGE)
        identity = np.arange(cube_flat, dtype=np.int64).reshape(NFACE, NFACE_EDGE, NFACE_EDGE)

        for f in range(NFACE):
            face = np.full(rr.shape, f, dtype=np.int64)
            row = rr.astype(np.float64) - self.crop_top
            col = cc.astype(np.float64) - self.crop_left
            alpha = 2.0 * col / (NFACE_EDGE - 1) - 1.0
            beta = 2.0 * row / (NFACE_EDGE - 1) - 1.0

            x, y, z = self._face_alpha_beta_to_xyz(face, alpha, beta)
            owner = self._assign_faces(x, y, z)
            owner_alpha, owner_beta = self._xyz_to_face_alpha_beta(owner, x, y, z)
            owner_col = np.rint((owner_alpha + 1.0) * (NFACE_EDGE - 1) / 2.0).astype(np.int64)
            owner_row = np.rint((owner_beta + 1.0) * (NFACE_EDGE - 1) / 2.0).astype(np.int64)
            owner_row = np.clip(owner_row, 0, NFACE_EDGE - 1)
            owner_col = np.clip(owner_col, 0, NFACE_EDGE - 1)

            flat = owner * (NFACE_EDGE * NFACE_EDGE) + owner_row * NFACE_EDGE + owner_col
            if not np.all(owned[flat]):
                bad = flat[~owned[flat]][:10]
                raise RuntimeError(f"ghost map produced non-owned cube cells: {bad.tolist()}")

            face_map = flat.reshape(p, p)
            native = face_map[self.crop_top : self.crop_top + NFACE_EDGE, self.crop_left : self.crop_left + NFACE_EDGE]
            native[owned_cube[f]] = identity[f][owned_cube[f]]
            face_chunks.append(face_map)

        return np.stack(face_chunks, axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad faces with a full ghost-cell exchange.

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

        cube = x.reshape(B, NFACE, C, H * W).permute(0, 2, 1, 3).reshape(B, C, NFACE * H * W)
        src = self.source_flat_index.reshape(-1)
        out = cube[:, :, src]
        out = out.reshape(B, C, NFACE, self.padded_size, self.padded_size)
        return out.permute(0, 2, 1, 3, 4).reshape(B6, C, self.padded_size, self.padded_size)
