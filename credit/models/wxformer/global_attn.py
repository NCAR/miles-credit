"""
global_attn.py
--------------
Two global / cross-face attention modules for CubeSphereWxFormer.

BottleneckGlobalAttention
~~~~~~~~~~~~~~~~~~~~~~~~~
Full global attention over all 6*h*w tokens from all cube faces.  Used at
stage 3 only (bottleneck, 12×12 spatial), replacing FaceAttention there.

CrossFaceTileAttention
~~~~~~~~~~~~~~~~~~~~~~
For each of the 12 face-edge pairs in the cube-sphere topology, gathers
the single boundary tile column/row from each side, concatenates, runs MHA,
and scatters back.  Uses the CrossFormer tile size (global_window_size=6).
Applied after each encoder stage (stages 0-3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ne120 constants
NFACE = 6
NFACE_EDGE = 361

# 12 directed face-edge pairs (same topology as HaloExchange / FaceEdgeAttention)
TOPOLOGY = [
    # (fi, edge, f_nb, nb_fixed_val, nb_axis, is_reversed)
    (0, "right", 2, 1, "col", False),
    (0, "left", 3, 359, "col", False),
    (1, "right", 3, 1, "col", False),
    (1, "left", 2, 359, "col", False),
    (0, "top", 5, 359, "col", True),
    (0, "bottom", 4, 359, "col", False),
    (1, "top", 5, 1, "col", False),
    (1, "bottom", 4, 1, "col", True),
    (2, "top", 5, 1, "row", True),
    (2, "bottom", 4, 359, "row", True),
    (3, "top", 5, 359, "row", False),
    (3, "bottom", 4, 1, "row", False),
]


# ---------------------------------------------------------------------------
# Module 1: BottleneckGlobalAttention
# ---------------------------------------------------------------------------


class BottleneckGlobalAttention(nn.Module):
    """Global MHA over all 6*h*w tokens from all cube faces (bottleneck only).

    At stage 3 the spatial size is 12×12, giving 6*144=864 tokens per sample.
    This replaces FaceAttention at stage 3 with true spatial global attention.

    Parameters
    ----------
    dim : int
        Feature channels (1024 at stage 3).
    heads : int
        Number of attention heads (must divide dim).
    dropout : float
        Attention dropout.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B*6, D, h, w)

        Returns
        -------
        (B*6, D, h, w)  with global cross-face residual update.
        """
        B6, D, h, w = x.shape
        B = B6 // NFACE

        # (B*6, D, h, w) → (B, 6, D, h, w) → (B, 6*h*w, D)
        faces = x.reshape(B, NFACE, D, h, w)
        tokens = faces.permute(0, 1, 3, 4, 2).reshape(B, NFACE * h * w, D)

        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out  # residual

        # (B, 6*h*w, D) → (B, 6, h, w, D) → (B, 6, D, h, w) → (B*6, D, h, w)
        faces = tokens.reshape(B, NFACE, h, w, D).permute(0, 1, 4, 2, 3)
        return faces.reshape(B6, D, h, w)


# ---------------------------------------------------------------------------
# Module 2: CrossFaceTileAttention
# ---------------------------------------------------------------------------


def _tile_boundary_indices(
    fi: int,
    edge: str,
    f_nb: int,
    nb_fixed_val: int,
    nb_axis: str,
    is_reversed: bool,
    tile_size: int,
    halo_size: int,
    cum_stride: int,
    padded_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute tile-level boundary indices for one face-edge pair at one stage.

    For the face side (fi/edge) we take the single boundary tile column/row
    (the tile touching the edge).  For the neighbor side (f_nb) we take the
    tile touching its corresponding boundary.

    Each tile contains tile_size × tile_size positions in the downsampled map.
    We return T tile pairs, each described by tile_size² (row, col) pairs for
    face fi and tile_size² pairs for face f_nb.

    Returns
    -------
    fi_rows  : (T*tile_size², )  int64
    fi_cols  : (T*tile_size², )  int64
    fnb_rows : (T*tile_size², )  int64
    fnb_cols : (T*tile_size², )  int64

    T = number of tile positions along the edge at this stage.
    Returns None if T == 0.
    """
    h = halo_size
    s = cum_stride
    ts = tile_size
    ds = padded_size // s  # downsampled spatial size (e.g. 96, 48, 24, 12)

    # Face start and extent in downsampled coordinates
    face_start = h // s  # first pixel of face fi in ds map
    face_end = (h + NFACE_EDGE) // s  # one past last pixel
    face_size_ds = face_end - face_start  # number of tiles fit into face extent

    # Number of complete tile positions along the edge
    # Tiles are laid out: tile index k covers pixels [k*ts, (k+1)*ts) in the ds map
    # We want tiles that overlap [face_start, face_end)
    # First tile touching face start:
    first_tile = face_start // ts
    # Last tile touching face end - 1:
    last_tile = (face_end - 1) // ts
    T = last_tile - first_tile + 1  # number of tile positions along edge

    if T <= 0:
        return None

    # ── Face A (fi) boundary tile row/col ────────────────────────────────
    # At the boundary the tile is at the first or last tile index along the
    # perpendicular dimension (row for top/bottom, col for left/right).
    # Along the edge the tiles are indexed first_tile .. last_tile.

    # Tile indices along the edge for face fi (always first_tile..last_tile)
    tile_along_a = np.arange(first_tile, last_tile + 1, dtype=np.int64)  # (T,)

    if edge == "top":
        # Boundary tile: row index = first_tile (topmost row of tiles)
        tile_perp_a = first_tile  # scalar tile row index
        # grid: (T, ts, ts) positions in ds map
        # rows: tile_perp_a * ts .. tile_perp_a * ts + ts - 1   (same for all along)
        # cols: tile_along_a[k] * ts .. tile_along_a[k] * ts + ts - 1
        a_row_tiles = np.full(T, tile_perp_a, dtype=np.int64)  # (T,)
        a_col_tiles = tile_along_a  # (T,)
    elif edge == "bottom":
        tile_perp_a = last_tile
        a_row_tiles = np.full(T, tile_perp_a, dtype=np.int64)
        a_col_tiles = tile_along_a
    elif edge == "left":
        tile_perp_a = first_tile
        a_row_tiles = tile_along_a
        a_col_tiles = np.full(T, tile_perp_a, dtype=np.int64)
    else:  # right
        tile_perp_a = last_tile
        a_row_tiles = tile_along_a
        a_col_tiles = np.full(T, tile_perp_a, dtype=np.int64)

    # Expand each tile index to tile_size² pixel positions
    # For tile index k along row dim: pixels k*ts .. k*ts+ts-1
    def tile_to_pixels(tile_idxs: np.ndarray) -> np.ndarray:
        """(T,) tile indices → (T, ts) pixel indices."""
        offsets = np.arange(ts, dtype=np.int64)  # (ts,)
        return tile_idxs[:, None] * ts + offsets[None, :]  # (T, ts)

    # a_rows_2d[k, i] = row of pixel i in tile k (for face A)
    a_rows_2d = tile_to_pixels(a_row_tiles)  # (T, ts)
    a_cols_2d = tile_to_pixels(a_col_tiles)  # (T, ts)

    # Meshgrid within each tile: (T, ts, ts)
    # For each tile k: rows repeat along col dim, cols repeat along row dim
    a_r_grid = np.repeat(a_rows_2d[:, :, None], ts, axis=2)  # (T, ts, ts)
    a_c_grid = np.repeat(a_cols_2d[:, None, :], ts, axis=1)  # (T, ts, ts)

    # Clip to valid ds range
    a_r_grid = np.clip(a_r_grid, 0, ds - 1)
    a_c_grid = np.clip(a_c_grid, 0, ds - 1)

    # Flatten: (T*ts², )
    fi_rows = a_r_grid.reshape(-1).astype(np.int64)
    fi_cols = a_c_grid.reshape(-1).astype(np.int64)

    # ── Face B (f_nb) boundary tile row/col ──────────────────────────────
    # The neighbor face has its own face_start in the ds map (same geometry
    # since all faces are embedded in the same padded space after halo).
    nb_face_start = h // s
    nb_face_end = (h + NFACE_EDGE) // s
    nb_first_tile = nb_face_start // ts
    nb_last_tile = (nb_face_end - 1) // ts
    Tnb = nb_last_tile - nb_first_tile + 1

    # Tile positions along neighbor edge (boundary layer 0 inward)
    if nb_axis == "col":
        # nb_fixed_val is a pixel column in the ORIGINAL face (0..360)
        # direction: inward from the boundary col
        direction = +1 if nb_fixed_val <= NFACE_EDGE // 2 else -1
        # The boundary col in original face coords; tile index = (h + nb_fixed_val) // (s * ts)
        nb_bnd_px = nb_fixed_val
        nb_bnd_ds = (h + nb_bnd_px) // s  # pixel in ds map
        nb_bnd_tile = nb_bnd_ds // ts  # tile containing boundary

        # Along-edge tiles for neighbor: same count T, ordered to match fi ordering
        nb_along_tiles = np.arange(nb_first_tile, nb_last_tile + 1, dtype=np.int64)  # (Tnb,)
        if is_reversed:
            nb_along_tiles = nb_along_tiles[::-1].copy()

        # Truncate or pad to T (ideally T == Tnb)
        T_use = min(T, Tnb)
        tile_along_a_use = tile_along_a[:T_use]
        tile_along_a_final = tile_along_a_use
        nb_along_use = nb_along_tiles[:T_use]

        b_col_tiles = np.full(T_use, nb_bnd_tile, dtype=np.int64)
        b_row_tiles = nb_along_use

    else:  # nb_axis == 'row'
        nb_bnd_px = nb_fixed_val
        nb_bnd_ds = (h + nb_bnd_px) // s
        nb_bnd_tile = nb_bnd_ds // ts

        nb_along_tiles = np.arange(nb_first_tile, nb_last_tile + 1, dtype=np.int64)
        if is_reversed:
            nb_along_tiles = nb_along_tiles[::-1].copy()

        T_use = min(T, Tnb)
        tile_along_a_final = tile_along_a[:T_use]
        nb_along_use = nb_along_tiles[:T_use]

        b_row_tiles = np.full(T_use, nb_bnd_tile, dtype=np.int64)
        b_col_tiles = nb_along_use

    if T_use == 0:
        return None

    # Rebuild face-A indices with T_use tiles
    if edge == "top":
        a_r_use = np.full(T_use, tile_perp_a, dtype=np.int64)
        a_c_use = tile_along_a_final
    elif edge == "bottom":
        a_r_use = np.full(T_use, tile_perp_a, dtype=np.int64)
        a_c_use = tile_along_a_final
    elif edge == "left":
        a_r_use = tile_along_a_final
        a_c_use = np.full(T_use, tile_perp_a, dtype=np.int64)
    else:  # right
        a_r_use = tile_along_a_final
        a_c_use = np.full(T_use, tile_perp_a, dtype=np.int64)

    a_rows_2d_use = tile_to_pixels(a_r_use)  # (T_use, ts)
    a_cols_2d_use = tile_to_pixels(a_c_use)  # (T_use, ts)
    b_rows_2d = tile_to_pixels(b_row_tiles)  # (T_use, ts)
    b_cols_2d = tile_to_pixels(b_col_tiles)  # (T_use, ts)

    # Meshgrid within tiles: (T_use, ts, ts)
    a_r_g = np.repeat(a_rows_2d_use[:, :, None], ts, axis=2)
    a_c_g = np.repeat(a_cols_2d_use[:, None, :], ts, axis=1)
    b_r_g = np.repeat(b_rows_2d[:, :, None], ts, axis=2)
    b_c_g = np.repeat(b_cols_2d[:, None, :], ts, axis=1)

    # Clip
    a_r_g = np.clip(a_r_g, 0, ds - 1)
    a_c_g = np.clip(a_c_g, 0, ds - 1)
    b_r_g = np.clip(b_r_g, 0, ds - 1)
    b_c_g = np.clip(b_c_g, 0, ds - 1)

    fi_rows_out = a_r_g.reshape(-1).astype(np.int64)
    fi_cols_out = a_c_g.reshape(-1).astype(np.int64)
    fnb_rows_out = b_r_g.reshape(-1).astype(np.int64)
    fnb_cols_out = b_c_g.reshape(-1).astype(np.int64)

    return fi_rows_out, fi_cols_out, fnb_rows_out, fnb_cols_out


class CrossFaceTileAttention(nn.Module):
    """Tile-level cross-face MHA at each of the 12 cube-sphere face edges.

    After each encoder stage, for each of the 12 face-edge pairs, gathers the
    boundary tile column/row from each side, concatenates, runs MHA, and scatters
    back.  Uses the CrossFormer tile_size (global_window_size).

    Parameters
    ----------
    dims : tuple[int, ...]  (length 4)
        Feature dimension at each encoder stage (d0, d1, d2, d3).
    tile_size : int
        Tile size matching global_window_size (default 6).
    num_heads : int
        Attention heads for MHA (must divide each dim).
    halo_size : int
        Halo padding used by HaloExchange (needed for pixel offset).
    strides : tuple[int, ...]  (length 4)
        Cumulative downsampling stride at each stage (4, 8, 16, 32).
    padded_size : int
        Padded spatial size (384).
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        dims: tuple,
        tile_size: int = 6,
        num_heads: int = 4,
        halo_size: int = 6,
        strides: tuple = (4, 8, 16, 32),
        padded_size: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_stages = len(dims)
        assert len(strides) == n_stages

        self.dims = dims
        self.tile_size = tile_size
        self.num_heads = num_heads
        self.n_stages = n_stages

        # One LayerNorm + MHA per stage
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for d in dims
            ]
        )

        ts2 = tile_size * tile_size  # tokens per tile

        # Precompute index buffers for each (stage, topology pair)
        # Buffer _s{si}_t{ti}_valid  : scalar bool — whether this pair is active
        # Buffer _s{si}_t{ti}_fi     : scalar int  — face index fi
        # Buffer _s{si}_t{ti}_fnb    : scalar int  — face index f_nb
        # Buffer _s{si}_t{ti}_T      : scalar int  — number of tile pairs
        # Buffer _s{si}_t{ti}_ar/ac  : (T*ts²,) int64 — pixel indices in fi
        # Buffer _s{si}_t{ti}_br/bc  : (T*ts²,) int64 — pixel indices in f_nb
        for si, (cum_stride, dim) in enumerate(zip(strides, dims)):
            for ti, (fi, edge, f_nb, nb_fixed_val, nb_axis, is_reversed) in enumerate(TOPOLOGY):
                pfx = f"_s{si}_t{ti}"
                result = _tile_boundary_indices(
                    fi,
                    edge,
                    f_nb,
                    nb_fixed_val,
                    nb_axis,
                    is_reversed,
                    tile_size,
                    halo_size,
                    cum_stride,
                    padded_size,
                )
                if result is None:
                    self.register_buffer(f"{pfx}_valid", torch.tensor(False))
                    self.register_buffer(f"{pfx}_fi", torch.tensor(fi, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_fnb", torch.tensor(f_nb, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_T", torch.tensor(0, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_ar", torch.zeros(0, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_ac", torch.zeros(0, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_br", torch.zeros(0, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_bc", torch.zeros(0, dtype=torch.int64))
                else:
                    ar, ac, br, bc = result
                    T_use = len(ar) // ts2
                    self.register_buffer(f"{pfx}_valid", torch.tensor(True))
                    self.register_buffer(f"{pfx}_fi", torch.tensor(fi, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_fnb", torch.tensor(f_nb, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_T", torch.tensor(T_use, dtype=torch.int64))
                    self.register_buffer(f"{pfx}_ar", torch.from_numpy(ar))
                    self.register_buffer(f"{pfx}_ac", torch.from_numpy(ac))
                    self.register_buffer(f"{pfx}_br", torch.from_numpy(br))
                    self.register_buffer(f"{pfx}_bc", torch.from_numpy(bc))

    def forward(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply tile-level cross-face attention for one encoder stage.

        Parameters
        ----------
        x     : (B*6, D, h, w)
        stage : int  (0-indexed encoder stage)

        Returns
        -------
        (B*6, D, h, w)  with boundary tiles updated.
        """
        B6, D, h_map, w_map = x.shape
        B = B6 // NFACE
        si = stage
        ts = self.tile_size
        ts2 = ts * ts

        norm = self.norms[si]
        attn = self.attns[si]

        # Reshape to (B, 6, D, h_map, w_map) for per-face access
        x_f = x.reshape(B, NFACE, D, h_map, w_map)
        out_f = x_f.clone()

        for ti in range(len(TOPOLOGY)):
            pfx = f"_s{si}_t{ti}"
            valid = bool(getattr(self, f"{pfx}_valid"))
            if not valid:
                continue

            fi = int(getattr(self, f"{pfx}_fi"))
            f_nb = int(getattr(self, f"{pfx}_fnb"))
            T = int(getattr(self, f"{pfx}_T"))
            ar = getattr(self, f"{pfx}_ar")  # (T*ts²,)
            ac = getattr(self, f"{pfx}_ac")
            br = getattr(self, f"{pfx}_br")
            bc = getattr(self, f"{pfx}_bc")

            if T == 0 or len(ar) == 0:
                continue

            # Gather face-fi tokens: x_f[:, fi, :, ar, ac] → (B, D, T*ts²)
            # Reshape to (B, T, ts², D) then to (B*T, ts², D)
            toks_a = x_f[:, fi, :, ar, ac].permute(0, 2, 1)  # (B, T*ts², D)
            toks_b = x_f[:, f_nb, :, br, bc].permute(0, 2, 1)  # (B, T*ts², D)

            # Reshape to (B, T, ts², D)
            toks_a = toks_a.reshape(B, T, ts2, D)
            toks_b = toks_b.reshape(B, T, ts2, D)

            # Flatten batch+tile: (B*T, 2*ts², D)
            toks_a = toks_a.reshape(B * T, ts2, D)
            toks_b = toks_b.reshape(B * T, ts2, D)
            tokens = torch.cat([toks_a, toks_b], dim=1)  # (B*T, 2*ts², D)

            # LayerNorm + MHA + residual
            normed = norm(tokens)
            attn_out, _ = attn(normed, normed, normed)
            tokens = tokens + attn_out  # (B*T, 2*ts², D)

            # Split back and reshape to (B, T*ts², D)
            toks_a_out = tokens[:, :ts2, :].reshape(B, T * ts2, D)
            toks_b_out = tokens[:, ts2:, :].reshape(B, T * ts2, D)

            # Scatter back: (B, D, T*ts²)
            out_f[:, fi, :, ar, ac] = toks_a_out.permute(0, 2, 1)
            out_f[:, f_nb, :, br, bc] = toks_b_out.permute(0, 2, 1)

        return out_f.reshape(B6, D, h_map, w_map)
