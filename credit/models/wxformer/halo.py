"""
halo.py
-------
HaloExchange: populate face-edge padding with real data from physically
adjacent cube-sphere faces before the zero-pad + encoder stack.

Background
~~~~~~~~~~
The ne120 cubed-sphere SE grid has 6 faces (361x361).  Before encoding,
each face is padded to 384x384 with zeros.  The padding along the face
edges is therefore zero-filled, even though physically adjacent nodes on
neighboring faces carry real atmospheric data.  During autoregressive
rollout this leads to edge artifacts that compound over time.

HaloExchange replaces the first halo_size rows/cols of each face's zero
padding with actual feature values copied from the physically nearest
strip of the appropriate neighbor face.

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
It returns (B*6, C, H+2*halo_size, W+2*halo_size) where:
  - The center (H x W) region contains the original face data.
  - The halo borders contain real neighbor data where available.
  - Remaining padding (corners and faces with no neighbor) stays zero.

All index buffers are registered as nn.Buffers so .to(device) moves them.

Usage
-----
    halo = HaloExchange(adj_path, halo_size=6)
    x_halo = halo(x6)   # (B*6, C, 361, 361) -> (B*6, C, 373, 373)
    x_pad  = F.pad(x_halo, (0, 384-373, 0, 384-373))  # -> (B*6, C, 384, 384)
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
    """Populate face-edge padding from physically adjacent face strips.

    Parameters
    ----------
    adjacency_path : str | Path
        Path to ``se_face_adjacency_ne120.npz``.
    halo_size : int
        Number of rows/cols to copy from each neighbor (default 6).
        The output spatial size is (NFACE_EDGE + 2*halo_size) x same.
    """

    def __init__(
        self,
        adjacency_path: str | Path,
        halo_size: int = 6,
    ) -> None:
        super().__init__()
        self.halo_size = halo_size
        h = halo_size
        H = NFACE_EDGE

        adj = np.load(str(adjacency_path))

        # ── Build per-face-edge copy recipes ─────────────────────────────
        # Each recipe says: for face fi, edge 'top'/'bottom'/'left'/'right',
        # copy from face f_nb, what slice of f_nb's 2D feature map, to what
        # slice of the destination padded feature map.
        #
        # We store everything as flat integer arrays so the forward pass is
        # pure tensor indexing (no Python if-branches on tensor values).
        #
        # Recipe format (for a single edge copy):
        #   src_face   : int  (face index 0-5)
        #   src_row_start, src_row_stop, src_row_step : row slice in src face
        #   src_col_start, src_col_stop, src_col_step : col slice in src face
        #   dst_row_start, dst_row_stop               : row slice in dst padded tensor
        #   dst_col_start, dst_col_stop               : col slice in dst padded tensor
        #
        # Because some strips are reversed or transposed, we pre-build explicit
        # index arrays (gather-style) rather than trying to use slices everywhere.

        # We'll register for each face fi, each edge:
        #   src_face (scalar), src_rows (int64), src_cols (int64),
        #   dst_rows (int64), dst_cols (int64)

        self._recipes: list[tuple[int, int, int, int, int, int]] = []
        # list of (fi, src_face, src_rows_buf_name, src_cols_buf_name,
        #          dst_rows_buf_name, dst_cols_buf_name)

        # Known topology: (fi, edge, f_nb, nb_col_or_row, axis, reversed_order)
        # axis: 'col' = neighbor strip is a fixed column, rows vary
        #        'row' = neighbor strip is a fixed row, cols vary
        # reversed: neighbor indices run in reverse order relative to edge order
        topology = [
            # Equatorial <-> equatorial, 361 nodes, forward
            (0, "right", 2, 1, "col", False),
            (0, "left", 3, 359, "col", False),
            (1, "right", 3, 1, "col", False),
            (1, "left", 2, 359, "col", False),
            # Equatorial top/bottom <-> polar col strips, 359 nodes
            (0, "top", 5, 359, "col", True),
            (0, "bottom", 4, 359, "col", False),
            (1, "top", 5, 1, "col", False),
            (1, "bottom", 4, 1, "col", True),
            # F2/F3 top/bottom <-> polar row strips, 359 nodes
            (2, "top", 5, 1, "row", True),
            (2, "bottom", 4, 359, "row", True),
            (3, "top", 5, 359, "row", False),
            (3, "bottom", 4, 1, "row", False),
        ]

        for fi, edge, f_nb, nb_fixed_val, nb_axis, is_reversed in topology:
            for layer in range(h):
                # Determine neighbor strip position (goes inward from boundary)
                if nb_axis == "col":
                    direction = +1 if nb_fixed_val <= NFACE_EDGE // 2 else -1
                    nb_c = nb_fixed_val + layer * direction
                    # Neighbor: rows vary, col fixed at nb_c
                    # Available rows: for polar faces 1..359, for equatorial 0..360
                    if f_nb in (4, 5):
                        nb_rows_arr = np.arange(1, 360, dtype=np.int64)  # 359 rows
                    else:
                        nb_rows_arr = np.arange(0, 361, dtype=np.int64)  # 361 rows
                    if is_reversed:
                        nb_rows_arr = nb_rows_arr[::-1].copy()
                    nb_cols_arr = np.full(len(nb_rows_arr), nb_c, dtype=np.int64)
                else:  # nb_axis == 'row'
                    direction = +1 if nb_fixed_val <= NFACE_EDGE // 2 else -1
                    nb_r = nb_fixed_val + layer * direction
                    # Neighbor: cols vary, row fixed at nb_r
                    if f_nb in (4, 5):
                        nb_cols_arr = np.arange(1, 360, dtype=np.int64)  # 359 cols
                    else:
                        nb_cols_arr = np.arange(0, 361, dtype=np.int64)  # 361 cols
                    if is_reversed:
                        nb_cols_arr = nb_cols_arr[::-1].copy()
                    nb_rows_arr = np.full(len(nb_cols_arr), nb_r, dtype=np.int64)

                n = len(nb_rows_arr)  # 361 or 359

                # Destination positions in the PADDED output tensor (H+2h x W+2h)
                # Center region: [h : h+H, h : h+W]
                # Top halo:    row layer -> rows [h-1-layer, h-1-layer] cols [h : h+n_dst]
                # Bottom halo: row layer -> rows [h+H+layer] cols [h : h+n_dst]
                # Left halo:   col layer -> rows [h : h+n_dst] cols [h-1-layer]
                # Right halo:  col layer -> rows [h : h+n_dst] cols [h+H+layer]
                #
                # For 359-node polar edges, the nodes span cols/rows 1..359 of the face.
                # In the padded tensor, position 1 in the face corresponds to h+1.
                # So the 359 nodes go to dst positions h+1 .. h+359.

                if n == NFACE_EDGE:  # 361 nodes: covers full edge 0..360 -> h..h+360
                    dst_along_start = h
                elif n == NFACE_EDGE - 2:  # 359 nodes: inner edge 1..359 -> h+1..h+359
                    dst_along_start = h + 1
                else:
                    raise RuntimeError(f"Unexpected strip length {n} for F{fi} {edge} layer {layer}")

                # Build dst_rows and dst_cols arrays
                if edge == "top":
                    dst_row_val = h - 1 - layer
                    dst_rows_arr = np.full(n, dst_row_val, dtype=np.int64)
                    dst_cols_arr = np.arange(dst_along_start, dst_along_start + n, dtype=np.int64)
                elif edge == "bottom":
                    dst_row_val = h + H + layer
                    dst_rows_arr = np.full(n, dst_row_val, dtype=np.int64)
                    dst_cols_arr = np.arange(dst_along_start, dst_along_start + n, dtype=np.int64)
                elif edge == "left":
                    dst_col_val = h - 1 - layer
                    dst_cols_arr = np.full(n, dst_col_val, dtype=np.int64)
                    dst_rows_arr = np.arange(dst_along_start, dst_along_start + n, dtype=np.int64)
                elif edge == "right":
                    dst_col_val = h + H + layer
                    dst_cols_arr = np.full(n, dst_col_val, dtype=np.int64)
                    dst_rows_arr = np.arange(dst_along_start, dst_along_start + n, dtype=np.int64)

                # Register buffers
                buf_prefix = f"_{fi}_{edge}_l{layer}"
                self.register_buffer(f"{buf_prefix}_sr", torch.from_numpy(nb_rows_arr))
                self.register_buffer(f"{buf_prefix}_sc", torch.from_numpy(nb_cols_arr))
                self.register_buffer(f"{buf_prefix}_dr", torch.from_numpy(dst_rows_arr))
                self.register_buffer(f"{buf_prefix}_dc", torch.from_numpy(dst_cols_arr))

        # Store topology for forward lookup
        self._topology = topology

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad faces with neighbor halos.

        Parameters
        ----------
        x : (B*6, C, H, W)  where H=W=361=NFACE_EDGE

        Returns
        -------
        (B*6, C, H+2*halo_size, W+2*halo_size)
        """
        B6, C, H, W = x.shape
        B = B6 // NFACE
        h = self.halo_size
        Ho = H + 2 * h
        Wo = W + 2 * h

        # Allocate output, zero-filled
        out = x.new_zeros(B6, C, Ho, Wo)

        # Copy original face data to center region
        out[:, :, h : h + H, h : h + W] = x

        # Reshape for per-face access: (B, 6, C, H, W)
        x_f = x.reshape(B, NFACE, C, H, W)
        out_f = out.reshape(B, NFACE, C, Ho, Wo)

        for fi, edge, f_nb, nb_fixed_val, nb_axis, is_reversed in self._topology:
            for layer in range(self.halo_size):
                buf_prefix = f"_{fi}_{edge}_l{layer}"
                sr = getattr(self, f"{buf_prefix}_sr")  # src rows in f_nb
                sc = getattr(self, f"{buf_prefix}_sc")  # src cols in f_nb
                dr = getattr(self, f"{buf_prefix}_dr")  # dst rows in padded fi
                dc = getattr(self, f"{buf_prefix}_dc")  # dst cols in padded fi

                # Gather from neighbor face: (B, C, n)
                src_vals = x_f[:, f_nb, :, sr, sc]  # (B, C, n)

                # Scatter to destination face: (B, C, n)
                out_f[:, fi, :, dr, dc] = src_vals

        return out_f.reshape(B6, C, Ho, Wo)
