"""
edge_attn.py
------------
FaceEdgeAttention: sparse MHA over physically adjacent cross-face node pairs.

After each encoder stage the feature map is downsampled (384→96→48→24→12).
At each resolution, face-boundary nodes that are physically adjacent should
be allowed to communicate directly.  This module gathers the border strips
from both sides of each face edge, runs a single MHA layer over the 2*M
tokens, and scatters the updates back.

This is applied AFTER each FaceTransformerBlock, operating on the
downsampled feature maps at resolutions 96, 48, 24, 12.

Edge nodes at downsampled resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original 361x361 faces are padded to 384x384 then downsampled by
strides 4, 2, 2, 2 to give 96, 48, 24, 12.  After the CrossEmbedLayer
the feature map has spatial size (384/stride) = 96, 48, 24, 12.

For halo exchange we pad before encoding, so the padded face starts at
pixel h (halo_size) in the padded output.  After downsampling by stride s,
face pixel p maps to padded pixel (p + h), which maps to downsampled
position (p + h) // s.

For the edge_attn we gather a strip of edge_width rows/cols from each side
of each face edge.  At the original (361x361) resolution, the face
occupies positions h:h+361 in the padded image.  After stride-s downsampling
the face occupies positions h//s : (h+361)//s (approximately).

Usage
-----
    edge_attn = FaceEdgeAttention(
        adj_npz_path = "mesaclip/static/se_face_adjacency_ne120.npz",
        dims         = (64, 128, 256, 512),
        num_heads    = 4,
        edge_width   = 2,
        halo_size    = 6,
        strides      = (4, 2, 2, 2),
        padded_size  = 384,
    )
    # In the encoder loop after stage i:
    z = edge_attn(z, stage=i)
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

# ne120 constants
NFACE = 6
NFACE_EDGE = 361

# 12 face-edge pairs, same topology as HaloExchange
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


def _edge_indices_at_resolution(
    fi: int,
    edge: str,
    f_nb: int,
    nb_fixed_val: int,
    nb_axis: str,
    is_reversed: bool,
    edge_width: int,
    halo_size: int,
    stride: int,
    padded_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rows, cols) index arrays into the downsampled feature map.

    Returns
    -------
    src_a_rows, src_a_cols : (M,) int64 — border strip from face fi
    src_b_rows, src_b_cols : (M,) int64 — border strip from face f_nb
    """
    h = halo_size
    # At the downsampled resolution, the padded image has size padded_size // stride
    # Face fi occupies [h:h+NFACE_EDGE] in the original padded space,
    # which maps to [h//stride : (h+NFACE_EDGE)//stride] after downsampling (approx)
    # But CrossEmbedLayer uses strided conv; simpler to just compute the
    # face start pixel at each resolution.
    face_start = h // stride  # where face pixel 0 appears in downsampled map
    face_size = NFACE_EDGE // stride  # approx face size in downsampled map
    # Use floor division; face_size may be 90, 45, 22, 11 (not exactly 361/stride)
    # Actually 361//4=90, 361//8=45, 361//16=22, 361//32=11
    # The padded size is 384, so 384//4=96, 384//8=48, etc.
    down_size = padded_size // stride  # 96, 48, 24, 12

    # FACE A border strip: the edge_width rows/cols just inside the face edge
    # In the ORIGINAL face coordinate system (0..360):
    #   top edge: rows 0..edge_width-1
    #   bottom edge: rows (NFACE_EDGE-edge_width)..(NFACE_EDGE-1)
    #   left edge: cols 0..edge_width-1
    #   right edge: cols (NFACE_EDGE-edge_width)..(NFACE_EDGE-1)
    # After downsampling by stride, these roughly correspond to
    #   face pixel row r -> padded row (h + r) -> downsampled row (h + r) // stride
    # We compute the unique downsampled positions.

    def face_to_down(p: np.ndarray) -> np.ndarray:
        """Convert face pixel positions to downsampled pixel positions."""
        return (h + p) // stride

    if edge == "top":
        face_rows = np.arange(0, min(edge_width, NFACE_EDGE))
        # Determine along-edge length (359 or 361 depending on face)
        along = 361 if fi in (0, 1) else 359
        along_offset = 0 if fi in (0, 1) else 1  # col offset for face interior
        face_cols = np.arange(along_offset, along_offset + along)
        a_rows = face_to_down(face_rows)
        a_cols = face_to_down(face_cols)
        # Unique downsampled positions
        a_r_unique = np.unique(a_rows)
        a_c_unique = np.unique(a_cols)
        src_a_r, src_a_c = np.meshgrid(a_r_unique, a_c_unique, indexing="ij")
        src_a_rows = src_a_r.ravel()
        src_a_cols = src_a_c.ravel()

    elif edge == "bottom":
        face_rows = np.arange(NFACE_EDGE - edge_width, NFACE_EDGE)
        along = 361 if fi in (0, 1) else 359
        along_offset = 0 if fi in (0, 1) else 1
        face_cols = np.arange(along_offset, along_offset + along)
        a_rows = face_to_down(face_rows)
        a_cols = face_to_down(face_cols)
        a_r_unique = np.unique(a_rows)
        a_c_unique = np.unique(a_cols)
        src_a_r, src_a_c = np.meshgrid(a_r_unique, a_c_unique, indexing="ij")
        src_a_rows = src_a_r.ravel()
        src_a_cols = src_a_c.ravel()

    elif edge == "left":
        face_cols = np.arange(0, min(edge_width, NFACE_EDGE))
        # Left/right edges only appear on faces 0 and 1 (row full range)
        face_rows = np.arange(0, NFACE_EDGE)
        a_rows = face_to_down(face_rows)
        a_cols = face_to_down(face_cols)
        a_r_unique = np.unique(a_rows)
        a_c_unique = np.unique(a_cols)
        src_a_r, src_a_c = np.meshgrid(a_r_unique, a_c_unique, indexing="ij")
        src_a_rows = src_a_r.ravel()
        src_a_cols = src_a_c.ravel()

    elif edge == "right":
        face_cols = np.arange(NFACE_EDGE - edge_width, NFACE_EDGE)
        face_rows = np.arange(0, NFACE_EDGE)
        a_rows = face_to_down(face_rows)
        a_cols = face_to_down(face_cols)
        a_r_unique = np.unique(a_rows)
        a_c_unique = np.unique(a_cols)
        src_a_r, src_a_c = np.meshgrid(a_r_unique, a_c_unique, indexing="ij")
        src_a_rows = src_a_r.ravel()
        src_a_cols = src_a_c.ravel()

    else:
        raise ValueError(f"Unknown edge: {edge!r}")

    # FACE B border strip: edge_width layers from neighbor face
    # The neighbor strip for layer 0 is at nb_fixed_val (col or row),
    # going inward for edge_width layers.
    if nb_axis == "col":
        direction = +1 if nb_fixed_val <= NFACE_EDGE // 2 else -1
        nb_cols_range = [nb_fixed_val + k * direction for k in range(edge_width)]
        along = 359 if f_nb in (4, 5) else 361
        along_offset = 1 if f_nb in (4, 5) else 0
        nb_rows_range = list(range(along_offset, along_offset + along))
        if is_reversed:
            nb_rows_range = nb_rows_range[::-1]
        b_rows = face_to_down(np.array(nb_rows_range))
        b_cols = face_to_down(np.array(nb_cols_range))
    else:  # nb_axis == 'row'
        direction = +1 if nb_fixed_val <= NFACE_EDGE // 2 else -1
        nb_rows_range = [nb_fixed_val + k * direction for k in range(edge_width)]
        along = 359 if f_nb in (4, 5) else 361
        along_offset = 1 if f_nb in (4, 5) else 0
        nb_cols_range = list(range(along_offset, along_offset + along))
        if is_reversed:
            nb_cols_range = nb_cols_range[::-1]
        b_rows = face_to_down(np.array(nb_rows_range))
        b_cols = face_to_down(np.array(nb_cols_range))

    b_r_unique = np.unique(b_rows)
    b_c_unique = np.unique(b_cols)
    src_b_r, src_b_c = np.meshgrid(b_r_unique, b_c_unique, indexing="ij")
    src_b_rows = src_b_r.ravel()
    src_b_cols = src_b_c.ravel()

    # Clip to valid range [0, down_size)
    def clip(arr):
        return np.clip(arr, 0, down_size - 1)

    return (
        clip(src_a_rows).astype(np.int64),
        clip(src_a_cols).astype(np.int64),
        clip(src_b_rows).astype(np.int64),
        clip(src_b_cols).astype(np.int64),
    )


class FaceEdgeAttention(nn.Module):
    """Sparse MHA between physically adjacent cross-face node pairs.

    Operates on (B*6, D, h, w) feature maps at downsampled encoder resolution.
    For each of the 12 face-edge pairs, gathers a border strip from each side,
    runs MHA over the combined 2*M tokens, and scatters updates back.

    Parameters
    ----------
    dims : tuple[int, ...]  (length n_stages)
        Feature dimension at each encoder stage.
    num_heads : int
        Attention heads (must divide each dim).
    edge_width : int
        Number of node rows/cols included on each side of the edge (default 2).
    halo_size : int
        Halo padding used by HaloExchange (needed for pixel offset calculation).
    strides : tuple[int, ...]  (length n_stages)
        Cumulative downsampling stride at each stage (4, 8, 16, 32).
    padded_size : int
        Size of the padded face (384).
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        dims: tuple,
        num_heads: int = 4,
        edge_width: int = 2,
        halo_size: int = 6,
        strides: tuple = (4, 8, 16, 32),
        padded_size: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_stages = len(dims)
        assert len(strides) == n_stages

        self.dims = dims
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

        # Precompute edge index buffers for each stage and each face-edge pair
        for si, (stride, dim) in enumerate(zip(strides, dims)):
            for ti, (fi, edge, f_nb, nb_fixed_val, nb_axis, is_reversed) in enumerate(TOPOLOGY):
                a_rows, a_cols, b_rows, b_cols = _edge_indices_at_resolution(
                    fi,
                    edge,
                    f_nb,
                    nb_fixed_val,
                    nb_axis,
                    is_reversed,
                    edge_width,
                    halo_size,
                    stride,
                    padded_size,
                )
                pfx = f"_s{si}_t{ti}"
                self.register_buffer(f"{pfx}_fi", torch.tensor(fi, dtype=torch.int64))
                self.register_buffer(f"{pfx}_fnb", torch.tensor(f_nb, dtype=torch.int64))
                self.register_buffer(f"{pfx}_ar", torch.from_numpy(a_rows))
                self.register_buffer(f"{pfx}_ac", torch.from_numpy(a_cols))
                self.register_buffer(f"{pfx}_br", torch.from_numpy(b_rows))
                self.register_buffer(f"{pfx}_bc", torch.from_numpy(b_cols))

    def forward(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply edge attention for one encoder stage.

        Parameters
        ----------
        x     : (B*6, D, h, w)
        stage : int  (0-indexed encoder stage)

        Returns
        -------
        (B*6, D, h, w)  with edge-adjacent tokens updated.
        """
        B6, D, h_map, w_map = x.shape
        B = B6 // NFACE
        si = stage

        norm = self.norms[si]
        attn = self.attns[si]

        # Reshape to (B, 6, D, h, w)
        x_f = x.reshape(B, NFACE, D, h_map, w_map)
        # We'll accumulate updates into a clone
        out_f = x_f.clone()

        n_pairs = len(TOPOLOGY)
        for ti in range(n_pairs):
            pfx = f"_s{si}_t{ti}"
            fi = int(getattr(self, f"{pfx}_fi"))
            f_nb = int(getattr(self, f"{pfx}_fnb"))
            ar = getattr(self, f"{pfx}_ar")  # (Ma,)
            ac = getattr(self, f"{pfx}_ac")  # (Ma,)
            br = getattr(self, f"{pfx}_br")  # (Mb,)
            bc = getattr(self, f"{pfx}_bc")  # (Mb,)

            if len(ar) == 0 or len(br) == 0:
                continue

            # Gather tokens: (B, Ma, D) and (B, Mb, D)
            tokens_a = x_f[:, fi, :, ar, ac].permute(0, 2, 1)  # (B, Ma, D)
            tokens_b = x_f[:, f_nb, :, br, bc].permute(0, 2, 1)  # (B, Mb, D)

            # Concatenate: (B, Ma+Mb, D)
            tokens = torch.cat([tokens_a, tokens_b], dim=1)
            Ma = tokens_a.shape[1]
            Mb = tokens_b.shape[1]

            # LayerNorm + MHA with residual
            normed = norm(tokens)
            attn_out, _ = attn(normed, normed, normed)
            tokens = tokens + attn_out

            # Scatter back
            out_f[:, fi, :, ar, ac] = tokens[:, :Ma, :].permute(0, 2, 1)
            out_f[:, f_nb, :, br, bc] = tokens[:, Ma:, :].permute(0, 2, 1)

        return out_f.reshape(B6, D, h_map, w_map)
