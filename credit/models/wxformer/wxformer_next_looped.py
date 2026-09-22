"""
WXFormerNextLooped: NextGenWXFormer with input-conditioned recurrent bottleneck refinement.

Experiment A from docs/looped_transformer_claude_handoff.md: instead of running the
bottleneck (spatial attention + global spectral mixing) once, run a single SHARED
block K times, refining the bottleneck latent before decoding. Each pass:

  - re-injects the original encoded bottleneck features (not just the evolving
    latent), so later passes don't drift from what the encoder actually saw
  - conditions on a learned per-iteration embedding, so the shared weights can
    behave differently pass-to-pass
  - is scaled by a small, learned, per-iteration gate before being added back,
    so extra passes start out close to a no-op and the model has to learn to
    make each one useful

This only touches the bottleneck. The encoder/decoder stages are unchanged from
NextGenWXFormer (their CrossEmbedLayer changes spatial resolution, so they
cannot be looped the way the fixed-resolution bottleneck can), and the outer
input/output persistence residual is applied exactly once, same as the base
model -- the recurrent refinement never touches it.

K (``bottleneck_loops``) is a fixed constructor argument for now. Per the
handoff, randomized loop counts and adaptive halting are later experiments,
once a fixed-depth loop is shown to help at all.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from credit.models.base_model import BaseModel
from credit.models.wxformer.crossformer import (
    CrossEmbedLayer,
    LayerNorm,
    UpBlockPS,
    apply_spectral_norm,
    cast_tuple,
)
from credit.models.wxformer.wxformer_next import (
    ColumnAttention,
    LevelEmbedding,
    SpectralGNNBottleneck,
    Transformer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared recurrent bottleneck refiner
# ---------------------------------------------------------------------------


class LoopedBottleneckRefiner(nn.Module):
    """Shared block applied K times to refine the bottleneck latent.

    Combines local/global spatial attention (reusing ``Transformer``) with
    global spectral mixing (reusing ``SpectralGNNBottleneck``), conditioned on
    the original encoded bottleneck features and the current iteration index.

    Returns an additive delta -- callers apply their own (typically small,
    learned) gate before adding it back into the running latent. This module
    never adds the running latent ``z`` back to its own output itself, so a
    caller doing ``z = z + gate * refiner(z, c, k)`` never double-adds ``z``.

    Args:
        dim: Bottleneck channel dimension.
        nlat: Bottleneck height (spatial nodes = nlat x nlon).
        nlon: Bottleneck width.
        local_window_size: Short-range attention window (as in ``Transformer``).
        global_window_size: Long-range attention stride (as in ``Transformer``).
        dim_head: Attention head dimension.
        depth: Transformer sublayer depth *inside one pass* of the shared
            block. Kept small (default 1) since the outer loop count is what
            provides the extra depth -- that's the parameter-efficiency point.
        num_spectral_nodes: Virtual spectral nodes K in the global mixing step.
        loop_count: Number of passes this instance will be used for (only
            needed to size the per-iteration embedding table).
    """

    def __init__(
        self,
        dim: int,
        nlat: int,
        nlon: int,
        local_window_size: int,
        global_window_size: int,
        dim_head: int = 32,
        depth: int = 1,
        num_spectral_nodes: int = 64,
        loop_count: int = 2,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.loop_count = loop_count

        # Per-iteration conditioning, broadcast-added like LevelEmbedding.
        self.iter_embedding = nn.Parameter(torch.zeros(loop_count, dim))

        self.norm = LayerNorm(dim)
        # Projects the original encoded bottleneck features before combining
        # with the (normed, iteration-conditioned) evolving latent -- the
        # "reinject the original input via a learned projection" step.
        self.reinject = nn.Conv2d(dim, dim, kernel_size=1)
        self.combine = nn.Conv2d(2 * dim, dim, kernel_size=1)

        self.spatial_mix = Transformer(
            dim=dim,
            local_window_size=local_window_size,
            global_window_size=global_window_size,
            depth=depth,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        # SpectralGNNBottleneck already returns `delta + residual` against
        # whatever it's given -- here that residual is against the
        # attention-refined intermediate `h`, not against `z`, so this does
        # not re-add the outer loop's `z`.
        self.global_mix = SpectralGNNBottleneck(dim, nlat, nlon, num_spectral_nodes=num_spectral_nodes)

    def forward(self, z: torch.Tensor, c: torch.Tensor, k: int) -> torch.Tensor:
        """One refinement pass.

        Args:
            z: Current bottleneck latent, (B, dim, H, W).
            c: Original encoded bottleneck features (fixed across passes), (B, dim, H, W).
            k: Iteration index (0-indexed), selects the conditioning embedding.

        Returns:
            Additive delta, (B, dim, H, W).
        """
        cond = z + self.iter_embedding[k][None, :, None, None]
        cond = self.norm(cond)
        reinj = self.reinject(c)
        h = self.combine(torch.cat([cond, reinj], dim=1))
        h = self.spatial_mix(h)
        delta = self.global_mix(h)
        return delta


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class WXFormerNextLooped(BaseModel):
    """NextGenWXFormer with input-conditioned recurrent bottleneck refinement.

    Identical to ``NextGenWXFormer`` (encoder, level embedding, column
    attention, decoder, persistence residual) except at the bottleneck: instead
    of a single ``SpectralGNNBottleneck`` call, a single shared
    ``LoopedBottleneckRefiner`` block is applied ``bottleneck_loops`` times,
    each pass re-injecting the original encoded features and conditioning on
    the iteration index, gated by a small learned per-iteration scale.

    Args:
        image_height: Grid cells in south-north direction.
        image_width: Grid cells in west-east direction.
        frames: Number of input time steps.
        channels: Number of 3D (pressure-level) variables.
        surface_channels: Number of surface (single-level) variables.
        input_only_channels: Forcing/static variables that are input-only.
        output_only_channels: Diagnostic variables that are output-only.
        levels: Number of vertical pressure levels.
        dim: Hidden dimension at each of the 4 encoder stages.
        depth: Transformer block depth at each encoder stage.
        dim_head: Attention head dimension.
        global_window_size: Long-range attention stride at each stage.
        local_window_size: Short-range window size (scalar, shared across stages).
        cross_embed_kernel_sizes: CrossEmbedLayer kernel sizes at each stage.
        cross_embed_strides: CrossEmbedLayer strides at each stage.
        col_attn_heads: Number of heads in column attention.
        col_attn_stride: Spatial pooling stride before column attention (1 = full resolution).
        decoder_col_attn: Apply column attention to atmospheric output channels before the
            residual add.
        num_spectral_nodes: Virtual spectral nodes K in the bottleneck's global mixing step.
        bottleneck_loops: Number of shared-weight refinement passes at the bottleneck (K).
            2 or 4 per the handoff's initial experiment; 1 reduces to (approximately) a
            single spectral-mixing pass, structurally close to NextGenWXFormer's bottleneck.
        bottleneck_loop_depth: Transformer sublayer depth *inside one pass* of the shared
            refiner (kept small -- the loop count is what provides the extra depth).
        loop_gate_init: Initial value of the per-iteration residual gate. Small and non-zero
            (not exactly 0.0): a gate of exactly zero would give the refiner zero gradient at
            init, since d(z + 0*delta)/d(refiner params) = 0.
        use_spectral_norm: Apply spectral normalization to conv/linear layers.
    """

    def __init__(
        self,
        image_height: int = 640,
        image_width: int = 1280,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size: tuple = (5, 5, 2, 1),
        local_window_size: int = 10,
        cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides: tuple = (4, 2, 2, 2),
        col_attn_heads: int = 4,
        col_attn_stride: int = 1,
        decoder_col_attn: bool = False,
        num_spectral_nodes: int = 64,
        bottleneck_loops: int = 2,
        bottleneck_loop_depth: int = 1,
        loop_gate_init: float = 0.1,
        use_spectral_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        if bottleneck_loops < 1:
            raise ValueError(f"bottleneck_loops must be >= 1, got {bottleneck_loops}")

        dim = cast_tuple(tuple(dim), 4)
        depth = cast_tuple(tuple(depth), 4)
        global_window_size = cast_tuple(tuple(global_window_size), 4)
        local_window_size = cast_tuple(
            tuple(local_window_size) if isinstance(local_window_size, (list, tuple)) else local_window_size, 4
        )
        cross_embed_kernel_sizes = cast_tuple(tuple([tuple(k) for k in cross_embed_kernel_sizes]), 4)
        cross_embed_strides = cast_tuple(tuple(cross_embed_strides), 4)

        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.input_only_channels = input_only_channels
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        self.decoder_col_attn = decoder_col_attn
        self.bottleneck_loops = bottleneck_loops

        atmos_channels = channels * levels
        input_channels = (atmos_channels + surface_channels + input_only_channels) * frames
        last_dim = dim[-1]

        self.prognostic_channels = channels * levels + surface_channels
        self.output_channels = self.prognostic_channels + output_only_channels

        # ── Input processing ─────────────────────────────────────────────
        self.level_embedding = LevelEmbedding(channels, levels)
        self.col_attn = ColumnAttention(channels, levels, num_heads=col_attn_heads, spatial_stride=col_attn_stride)

        # ── Encoder ──────────────────────────────────────────────────────
        # Unchanged from NextGenWXFormer: each stage's CrossEmbedLayer changes
        # spatial resolution, so it cannot be looped the way the fixed-shape
        # bottleneck below can.
        dims = [input_channels, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList()
        for (dim_in, dim_out), n_layers, g_wsz, l_wsz, kernels, stride in zip(
            dim_pairs,
            depth,
            global_window_size,
            local_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            self.layers.append(
                nn.ModuleList(
                    [
                        CrossEmbedLayer(dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernels, stride=stride),
                        Transformer(
                            dim=dim_out,
                            local_window_size=l_wsz,
                            global_window_size=g_wsz,
                            depth=n_layers,
                            dim_head=dim_head,
                        ),
                    ]
                )
            )

        # ── Looped bottleneck refinement (Experiment A) ───────────────────
        bn_h, bn_w = image_height, image_width
        for s in cross_embed_strides:
            bn_h //= s
            bn_w //= s
        self.bottleneck_refiner = LoopedBottleneckRefiner(
            last_dim,
            bn_h,
            bn_w,
            local_window_size=local_window_size[-1],
            global_window_size=global_window_size[-1],
            dim_head=dim_head,
            depth=bottleneck_loop_depth,
            num_spectral_nodes=num_spectral_nodes,
            loop_count=bottleneck_loops,
        )
        # Small, non-zero init: a gate of exactly 0.0 would give the refiner
        # zero gradient at init (d(z + 0*delta)/d(theta) = 0), so extra passes
        # could never learn to become useful.
        self.loop_gate = nn.Parameter(torch.full((bottleneck_loops,), float(loop_gate_init)))

        # ── Decoder ──────────────────────────────────────────────────────
        # Unchanged from NextGenWXFormer.
        scale = 2
        self.up_block1 = UpBlockPS(last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlockPS(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlockPS(2 * (last_dim // 4), last_dim // 8, dim[0])
        self.up_block4 = nn.Sequential(
            nn.Conv2d(2 * (last_dim // 8), self.output_channels * scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
        )

        if decoder_col_attn:
            self.dec_col_attn = ColumnAttention(
                channels, levels, num_heads=col_attn_heads, spatial_stride=col_attn_stride
            )

        if use_spectral_norm:
            apply_spectral_norm(self.layers)
            apply_spectral_norm(self.bottleneck_refiner.spatial_mix)
            apply_spectral_norm(self.bottleneck_refiner.global_mix)
            apply_spectral_norm(self.bottleneck_refiner.reinject)
            apply_spectral_norm(self.bottleneck_refiner.combine)
            apply_spectral_norm(self.up_block1)
            apply_spectral_norm(self.up_block2)
            apply_spectral_norm(self.up_block3)
            apply_spectral_norm(self.up_block4)
            apply_spectral_norm(self.level_embedding)
            # col_attn excluded: MHA internal linear buffers conflict with spectral norm device placement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        # x: (B, C_in, T, H, W)
        B, C, T, H, W = x.shape

        if T > 1:
            x = x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        else:
            x = x.squeeze(2)

        # Residual base: prognostic channels of the last input frame, zero-padded
        # for output-only (diagnostic) channels, which have no matching input to
        # persist. Model predicts a delta; adding the base means output ≈
        # persistence at random init for prognostic channels, which gives
        # non-zero ACC from the very first batch. Applied exactly once, at the
        # very end -- the bottleneck refinement loop below never touches it.
        total_per_frame = self.channels * self.levels + self.surface_channels + self.input_only_channels
        last_frame_offset = (T - 1) * total_per_frame
        x_res = x[:, last_frame_offset : last_frame_offset + self.prognostic_channels]
        if self.output_channels > self.prognostic_channels:
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, self.output_channels - self.prognostic_channels))

        # Apply level embedding and column attention to each frame's atmos channels
        atmos_size = self.channels * self.levels

        frame_slices = []
        for t in range(T):
            offset = t * total_per_frame
            x_atmos = x[:, offset : offset + atmos_size]
            x_rest = x[:, offset + atmos_size : offset + total_per_frame]
            x_atmos = self.level_embedding(x_atmos)
            x_atmos = self.col_attn(x_atmos)
            frame_slices.extend([x_atmos, x_rest])

        x = torch.cat(frame_slices, dim=1)

        # Encode
        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Looped bottleneck refinement: a single shared block, applied
        # bottleneck_loops times, each pass re-injecting the original encoded
        # features `c` and conditioning on the iteration index. Gradients flow
        # through every pass (no detaching between iterations).
        c = x
        z = x
        for k in range(self.bottleneck_loops):
            delta = self.bottleneck_refiner(z, c, k)
            z = z + self.loop_gate[k] * delta
        x = z

        # Decode
        x = self.up_block1(x)
        x = F.interpolate(x, size=encodings[2].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        x = F.interpolate(x, size=encodings[1].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        x = F.interpolate(x, size=encodings[0].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False)

        # Decoder column attention: refine vertical coupling in prediction space.
        if self.decoder_col_attn:
            atmos_size = self.channels * self.levels
            x_atmos = self.dec_col_attn(x[:, :atmos_size])
            x = torch.cat([x_atmos, x[:, atmos_size:]], dim=1)

        # Add residual: output is last-frame prognostic state + predicted delta
        x = x + x_res

        return x.unsqueeze(2)

    @classmethod
    def load_model(cls, conf):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "model_checkpoint.pt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def load_model_name(cls, conf, model_name):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        ckpt = os.path.join(os.path.expandvars(conf["save_loc"]), model_name)
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, H, W = 1, 2, 64, 128
    channels, levels, surface_channels = 4, 5, 7
    input_only_channels, output_only_channels = 3, 2

    C_in = channels * levels + surface_channels + input_only_channels
    C_out = channels * levels + surface_channels + output_only_channels

    for bottleneck_loops in (1, 2, 4):
        model = WXFormerNextLooped(
            image_height=H,
            image_width=W,
            frames=T,
            channels=channels,
            surface_channels=surface_channels,
            input_only_channels=input_only_channels,
            output_only_channels=output_only_channels,
            levels=levels,
            dim=(32, 64, 128, 256),
            depth=(2, 2, 2, 2),
            dim_head=8,
            global_window_size=(4, 2, 2, 1),
            local_window_size=4,
            cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
            cross_embed_strides=(2, 2, 2, 2),
            col_attn_heads=4,
            bottleneck_loops=bottleneck_loops,
            use_spectral_norm=True,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"bottleneck_loops={bottleneck_loops}  Parameters: {n_params:.3f}M  Device: {device}")

        x = torch.randn(B, C_in, T, H, W, device=device)
        y = model(x)

        expected = (B, C_out, 1, H, W)
        assert tuple(y.shape) == expected, f"shape mismatch: {tuple(y.shape)} != {expected}"
        assert not torch.isnan(y).any(), "NaN in output"

        y.mean().backward()
        assert model.loop_gate.grad is not None and torch.isfinite(model.loop_gate.grad).all(), (
            "loop_gate got no/non-finite gradient -- refiner is not receiving gradient signal"
        )
        print(f"  Output: {tuple(y.shape)}  Backward: OK  loop_gate.grad: {model.loop_gate.grad.tolist()}")

    print("All bottleneck_loops configurations OK")
