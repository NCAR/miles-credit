"""
Standalone PyTorch implementation of WXFormer (CrossFormer).
Credit/external dependencies removed; core model only.
"""

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def norm_window_sizes(ws, num_levels=4):
    """
    Normalise a window-size spec to a tuple of num_levels (H, W) pairs.

    Accepts:
      int          → same square window at every level
      (H, W)       → same anisotropic window at every level  (2-tuple of ints)
      (s0,s1,s2,s3)→ per-level square windows                (4-tuple of ints)
      ((H0,W0),…)  → per-level anisotropic windows           (4-tuple of 2-tuples)
    """
    # scalar → square window for all levels
    if isinstance(ws, int):
        return tuple((ws, ws) for _ in range(num_levels))
    ws = tuple(ws)
    # 2-tuple of ints → single anisotropic spec for all levels
    if len(ws) == 2 and all(isinstance(w, int) for w in ws):
        return tuple(ws for _ in range(num_levels))   # each element is (H,W)
    # otherwise must be a per-level spec of length num_levels
    assert len(ws) == num_levels, \
        f"window_size must be int, (H,W), or {num_levels}-tuple; got {ws}"
    return tuple((w, w) if isinstance(w, int) else tuple(w) for w in ws)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-embed layer
# ──────────────────────────────────────────────────────────────────────────────

class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([
            nn.Conv2d(dim_in, ds, k, stride=stride, padding=(k - stride) // 2)
            for k, ds in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x):
        return torch.cat([c(x) for c in self.convs], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Global register tokens
# ──────────────────────────────────────────────────────────────────────────────

class GlobalRegister(nn.Module):
    """
    Appended at the encoder bottleneck to provide truly global context.

    A small pool of learnable tokens cross-attends to every spatial position,
    then broadcasts a summary back as an additive bias — giving the model
    planetary-scale (teleconnection) awareness that windowed attention cannot
    provide on its own.
    """
    def __init__(self, dim: int, num_tokens: int = 8, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.tokens     = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.norm_tok   = nn.LayerNorm(dim)
        self.norm_x     = LayerNorm(dim)         # channel-first (BCHW)
        self.to_q       = nn.Linear(dim, dim, bias=False)
        self.to_kv      = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims; keep x_norm channel-last for attention
        x_flat = self.norm_x(x).flatten(2).transpose(1, 2)   # (B, H*W, C)
        tok = self.norm_tok(self.tokens.expand(B, -1, -1))    # (B, T, C)

        # Multi-head cross-attention: tokens (Q) attend to spatial (K, V)
        def split_heads(t):
            b, n, _ = t.shape
            return t.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.to_q(tok))                       # (B, heads, T, d)
        k, v = self.to_kv(x_flat).chunk(2, dim=-1)
        k, v = split_heads(k), split_heads(v)                 # (B, heads, H*W, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, heads, T, H*W)
        ctx  = (attn.softmax(dim=-1) @ v)                     # (B, heads, T, d)
        ctx  = ctx.transpose(1, 2).reshape(B, -1, C)          # (B, T, C)
        ctx  = self.out_proj(ctx)                              # (B, T, C)

        # Mean-pool tokens → broadcast as additive spatial bias
        summary = ctx.mean(dim=1)                              # (B, C)
        return x + summary[:, :, None, None]


# ──────────────────────────────────────────────────────────────────────────────
# Temporal aggregation
# ──────────────────────────────────────────────────────────────────────────────

class TemporalAggregator(nn.Module):
    """
    Replaces the naive reshape(b, c*t, h, w) with a learned inter-frame mixer.

    Groups = num_channels so each physical variable mixes across its T frames
    independently, letting the network learn tendencies and accelerations rather
    than just concatenating raw values.  The output shape is identical to the
    naive reshape — a drop-in replacement with zero interface change.
    """
    def __init__(self, num_channels: int, frames: int):
        super().__init__()
        in_ch = num_channels * frames
        self.norm  = LayerNorm(in_ch)
        # Each group = one physical variable; kernel mixes its T time-steps
        self.mixer = nn.Conv2d(in_ch, in_ch, 1, groups=num_channels)
        self.gate  = nn.Conv2d(in_ch, in_ch, 1, groups=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C*T, H, W)  — already reshaped
        h = self.norm(x)
        return F.gelu(self.mixer(h)) * torch.sigmoid(self.gate(h)) + x


# ──────────────────────────────────────────────────────────────────────────────
# Transformer components
# ──────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Channel-first (BCHW) layer norm."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, inner, 1),                           # channel expand
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner), # depthwise spatial mix
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner, dim, 1),                           # project back
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Mixture-of-Experts FeedForward
# ──────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single MoE expert — same ConvFFN structure as FeedForward."""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, inner, 1),
            nn.Conv2d(inner, inner, 3, padding=1, groups=inner),  # depthwise spatial mix
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts drop-in replacement for FeedForward.

    Each spatial position independently routes to top_k of num_experts experts
    via a 1×1 conv gate (preserves H×W structure).  Gaussian noise on logits
    during training encourages load balancing (Switch Transformer style).

    Call load_balancing_loss() after forward() to obtain the Switch Transformer
    auxiliary loss: num_experts × Σ_i f_i · p_i.
    """
    def __init__(self, dim, mult=4, dropout=0.0,
                 num_experts=4, top_k=2, noise_std=0.1):
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.experts = nn.ModuleList(
            [Expert(dim, mult, dropout) for _ in range(num_experts)]
        )
        # Spatial gate: (B, dim, H, W) → (B, num_experts, H, W)
        self.gate = nn.Conv2d(dim, num_experts, kernel_size=1)
        self._last_probs = None
        self._last_topk_idx = None

    def forward(self, x):
        B, C, H, W = x.shape
        logits = self.gate(x)                                    # (B, E, H, W)
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = logits.softmax(dim=1)                            # (B, E, H, W)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=1)     # (B, k, H, W)
        topk_weights = topk_vals / topk_vals.sum(dim=1, keepdim=True)

        # Cache routing stats; keep probs in-graph for auxiliary loss gradient
        self._last_probs = probs
        self._last_topk_idx = topk_idx

        # Run all experts over the full input, then select & weight top-k
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, C, H, W)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            idx = topk_idx[:, i:i+1, :, :]                        # (B, 1, H, W)
            w   = topk_weights[:, i:i+1, :, :]                    # (B, 1, H, W)
            idx_exp = idx.unsqueeze(2).expand(-1, -1, C, -1, -1)  # (B, 1, C, H, W)
            gathered = expert_outs.gather(1, idx_exp).squeeze(1)   # (B, C, H, W)
            out = out + w * gathered
        return out

    def load_balancing_loss(self):
        """
        Switch Transformer auxiliary loss: num_experts × Σ_i f_i · p_i.

        f_i = fraction of tokens dispatched to expert i (hard top-k routing).
        p_i = mean soft routing probability for expert i.
        """
        if self._last_probs is None:
            raise RuntimeError("Call forward() before load_balancing_loss().")
        probs    = self._last_probs      # (B, E, H, W)
        topk_idx = self._last_topk_idx  # (B, k, H, W)
        dispatch = torch.zeros_like(probs)
        ones = torch.ones(topk_idx.shape, dtype=probs.dtype, device=probs.device)
        dispatch.scatter_add_(1, topk_idx, ones)         # (B, E, H, W)
        f_i = dispatch.mean(dim=(0, 2, 3))               # (E,)
        p_i = probs.mean(dim=(0, 2, 3))                  # (E,)
        return self.num_experts * (f_i * p_i).sum()


class DynamicPositionBias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            Rearrange("... () -> ..."),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0):
        super().__init__()
        assert attn_type in {"short", "long"}
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.attn_type = attn_type
        # window_size may be int (square) or (H, W) tuple (anisotropic)
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dpb = DynamicPositionBias(dim // 4)
        self._dpb_cache = None   # populated on first eval forward, cleared on train()

        wsz_h, wsz_w = self.window_size
        pos_h = torch.arange(wsz_h)
        pos_w = torch.arange(wsz_w)
        grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"))  # (2, H, W)
        grid = rearrange(grid, "c i j -> (i j) c")                        # (H*W, 2)
        rel_pos = grid[:, None] - grid[None, :]                            # (H*W, H*W, 2)
        rel_pos[..., 0] += wsz_h - 1                                      # shift to [0, 2H-2]
        rel_pos[..., 1] += wsz_w - 1                                      # shift to [0, 2W-2]
        rel_pos_indices = rel_pos[..., 0] * (2 * wsz_w - 1) + rel_pos[..., 1]
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def train(self, mode=True):
        if mode:
            self._dpb_cache = None   # DPB weights will change; cached bias is stale
        return super().train(mode)

    def forward(self, x):
        *_, height, width, heads, device = (*x.shape, self.heads, x.device)
        wsz_h, wsz_w = self.window_size
        x = self.norm(x)

        if self.attn_type == "short":
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz_h, s2=wsz_w)
        else:
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz_h, l2=wsz_w)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        if self.training or self._dpb_cache is None:
            pos_h = torch.arange(-wsz_h, wsz_h + 1, device=device)
            pos_w = torch.arange(-wsz_w, wsz_w + 1, device=device)
            grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"))
            rel_pos = rearrange(grid, "c i j -> (i j) c").to(x.dtype)
            dpb_bias = self.dpb(rel_pos)[self.rel_pos_indices]
            if not self.training:
                self._dpb_cache = dpb_bias.detach()
        else:
            dpb_bias = self._dpb_cache
        sim = sim + dpb_bias

        attn = self.dropout(sim.softmax(dim=-1))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz_h, y=wsz_w)
        out = self.to_out(out)

        if self.attn_type == "short":
            out = rearrange(out, "(b h w) d s1 s2 -> b d (h s1) (w s2)",
                            h=height // wsz_h, w=width // wsz_w)
        else:
            out = rearrange(out, "(b h w) d l1 l2 -> b d (l1 h) (l2 w)",
                            h=height // wsz_h, w=width // wsz_w)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, *, local_window_size, global_window_size,
                 depth=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, "short", local_window_size, dim_head, attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
                Attention(dim, "long", global_window_size, dim_head, attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x


class TransformerWithMoE(nn.Module):
    """
    Transformer block where both FeedForward sublayers are replaced by
    MoEFeedForward.  Interface is a strict superset of Transformer.

    After a forward pass, call get_load_balancing_loss() to retrieve the
    summed Switch Transformer auxiliary loss across all MoE layers.
    """
    def __init__(self, dim, *, local_window_size, global_window_size,
                 depth=4, dim_head=32, attn_dropout=0.0, ff_dropout=0.0,
                 num_experts=4, top_k=2, moe_noise_std=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, "short", local_window_size, dim_head, attn_dropout),
                MoEFeedForward(dim, dropout=ff_dropout,
                               num_experts=num_experts, top_k=top_k,
                               noise_std=moe_noise_std),
                Attention(dim, "long", global_window_size, dim_head, attn_dropout),
                MoEFeedForward(dim, dropout=ff_dropout,
                               num_experts=num_experts, top_k=top_k,
                               noise_std=moe_noise_std),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x

    def get_load_balancing_loss(self):
        """Sum Switch Transformer aux loss across all MoE feedforward layers."""
        total = None
        for _, short_ff, _, long_ff in self.layers:
            for ff in (short_ff, long_ff):
                loss = ff.load_balancing_loss()
                total = loss if total is None else total + loss
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Decoder transformer (lightweight single-layer short-window attention block)
# ──────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One short-window attention + ConvFFN sublayer applied after each UpBlock.
    Uses a smaller FF mult (2 vs 4) to keep decoder compute modest.
    window_size must evenly divide the spatial dims at that decoder level.
    """
    def __init__(self, dim: int, window_size, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, "short", window_size, dim_head, dropout)
        self.ff   = FeedForward(dim, mult=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x) + x
        x = self.ff(x)   + x
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Decoder blocks
# ──────────────────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        blk = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_chans, out_chans, 3, padding=1),
                nn.GroupNorm(num_groups, out_chans),
                nn.SiLU(),
            ]
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        return self.b(x) + x


class UpBlockPS(nn.Module):
    """Pixel-shuffle upsampling block (sub-pixel convolution)."""
    def __init__(self, in_ch, out_ch, num_groups, scale=2, num_residuals=2):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch, out_ch * scale ** 2, 3, padding=1)
        self.ps    = nn.PixelShuffle(scale)
        self.sharp = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.xavier_normal_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        blk = []
        for _ in range(num_residuals):
            blk += [nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(num_groups, out_ch),
                    nn.SiLU()]
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x  = self.ps(self.conv(x))   # upsample via sub-pixel conv
        x  = x + self.sharp(x)       # sharpening residual
        sc = x
        x  = self.b(x)
        return x + sc


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class CrossFormer(nn.Module):
    def __init__(
        self,
        image_height: int = 640,
        image_width: int = 1280,
        frames: int = 2,
        output_frames: int = 1,
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
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        upsample_with_ps: bool = False,
        global_register_tokens: int = 8,   # 0 to disable
        dec_window_size: int = 4,           # short-window size for decoder attention blocks
        # MoE — scalars or 4-tuples (one value per encoder level, shallow→deep)
        num_experts: int = 4,
        top_k: int = 2,
        moe_noise_std: float = 0.1,
    ):
        super().__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = norm_window_sizes(global_window_size)   # → 4×(H,W)
        local_window_size  = norm_window_sizes(local_window_size)    # → 4×(H,W)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)
        num_experts = cast_tuple(num_experts, 4)
        top_k       = cast_tuple(top_k, 4)

        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.output_frames = output_frames

        self.base_input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = self.base_input_channels * frames
        self.base_output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = self.base_output_channels * output_frames
        # Variable groups for split output heads
        self.atm_channels = channels * levels                               # pressure-level vars
        self.sfc_channels = surface_channels + output_only_channels         # surface vars

        # Temporal aggregator: learned inter-frame mixing before encoder
        self.temporal_agg = TemporalAggregator(self.base_input_channels, frames)

        last_dim = dim[-1]
        dims = [self.input_channels, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossEmbedLayer(di, do, ks, stride=s),
                TransformerWithMoE(do, local_window_size=lws, global_window_size=gws,
                                   depth=dep, dim_head=dim_head,
                                   attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                   num_experts=ne, top_k=tk,
                                   moe_noise_std=moe_noise_std),
            ])
            for (di, do), dep, gws, lws, ks, s, ne, tk in zip(
                dim_pairs, depth, global_window_size, local_window_size,
                cross_embed_kernel_sizes, cross_embed_strides, num_experts, top_k,
            )
        ])

        # Decoder transformer blocks (one per UpBlock, operating on upsampled features)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dim[-1] // 2, dec_window_size, dim_head),
            DecoderBlock(dim[-1] // 4, dec_window_size, dim_head),
            DecoderBlock(dim[-1] // 8, dec_window_size, dim_head),
        ])

        # Global register tokens at the encoder bottleneck
        last_dim_heads = max(1, dim[-1] // 32)   # same ratio as Attention default
        self.global_register = (
            GlobalRegister(dim[-1], num_tokens=global_register_tokens,
                           num_heads=last_dim_heads)
            if global_register_tokens > 0 else None
        )

        # Learned sigmoid gates for encoder→decoder skip connections.
        # Each gate is a 1×1 conv applied to the encoder features before cat.
        self.skip_gates = nn.ModuleList([
            nn.Conv2d(d, d, 1) for d in (dim[0], dim[1], dim[2])
        ])

        num_groups = dim[0]
        self.upsample_with_ps = upsample_with_ps

        # Split final output into atmospheric (pressure-level) and surface heads
        # so each group gets a dedicated projection rather than sharing one.
        atm_out = self.atm_channels * output_frames
        sfc_out = self.sfc_channels * output_frames
        dec_in  = 2 * (last_dim // 8)

        if upsample_with_ps:
            scale = 2
            self.up_block1 = UpBlockPS(last_dim,               last_dim // 2, num_groups)
            self.up_block2 = UpBlockPS(2 * (last_dim // 2),   last_dim // 4, num_groups)
            self.up_block3 = UpBlockPS(2 * (last_dim // 4),   last_dim // 8, num_groups)
            self.out_head_atm = nn.Sequential(
                nn.Conv2d(dec_in, atm_out * scale ** 2, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(atm_out, atm_out, 3, padding=1),
            )
            self.out_head_sfc = nn.Sequential(
                nn.Conv2d(dec_in, sfc_out * scale ** 2, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(sfc_out, sfc_out, 3, padding=1),
            )
        else:
            self.up_block1 = UpBlock(last_dim,               last_dim // 2, num_groups)
            self.up_block2 = UpBlock(2 * (last_dim // 2),   last_dim // 4, num_groups)
            self.up_block3 = UpBlock(2 * (last_dim // 4),   last_dim // 8, num_groups)
            self.out_head_atm = nn.ConvTranspose2d(dec_in, atm_out, kernel_size=4, stride=2, padding=1)
            self.out_head_sfc = nn.ConvTranspose2d(dec_in, sfc_out, kernel_size=4, stride=2, padding=1)

    def get_load_balancing_loss(self):
        """Sum Switch Transformer aux loss across all encoder MoE transformers."""
        total = None
        for _, transformer in self.layers:
            loss = transformer.get_load_balancing_loss()
            total = loss if total is None else total + loss
        return total

    def forward(self, x):
        # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        x = x.reshape(b, c * t, h, w)          # (B, C*T, H, W)
        x = self.temporal_agg(x)               # learned inter-frame mixing

        encodings = []
        for i, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if i == len(self.layers) - 1 and self.global_register is not None:
                x = self.global_register(x)   # inject global context at bottleneck
            encodings.append(x)

        x = self.dec_blocks[0](self.up_block1(x))
        x = torch.cat([x, torch.sigmoid(self.skip_gates[2](encodings[2])) * encodings[2]], dim=1)

        x = self.dec_blocks[1](self.up_block2(x))
        x = torch.cat([x, torch.sigmoid(self.skip_gates[1](encodings[1])) * encodings[1]], dim=1)

        x = self.dec_blocks[2](self.up_block3(x))
        x = torch.cat([x, torch.sigmoid(self.skip_gates[0](encodings[0])) * encodings[0]], dim=1)

        x = torch.cat([self.out_head_atm(x), self.out_head_sfc(x)], dim=1)

        x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear",
                          align_corners=False)

        b, _, h, w = x.shape
        x = x.view(b, self.base_output_channels, self.output_frames, h, w)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test  (same tiny config as test_wxformer.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Same tiny config as test_wxformer.py
    # base_input_channels = 2*2+1+1 = 6 → input (1,6,2,64,128)
    # base_output_channels = 2*2+1   = 5 → output (1,5,1,64,128)
    CFG = dict(
        image_height=64, image_width=128,
        frames=2, output_frames=1,
        channels=2, surface_channels=1,
        input_only_channels=1, output_only_channels=0,
        levels=2,
        dim=(16, 32, 64, 128), depth=(1, 1, 2, 1), dim_head=16,
        global_window_size=(2, 2, 2, 2), local_window_size=4,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        attn_dropout=0.0, ff_dropout=0.0,
        # MoE: per-level tuples — deeper levels get more experts
        num_experts=(4, 4, 8, 8),
        top_k=(2, 2, 2, 1),
        moe_noise_std=0.1,
    )

    x = torch.randn(1, 6, 2, 64, 128)

    # ── Eval (shape check) ────────────────────────────────────────────────────
    model = CrossFormer(**CFG)
    model.eval()
    with torch.no_grad():
        out = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"params={n:,}  in={tuple(x.shape)}  out={tuple(out.shape)}")
    assert out.shape == (1, 5, 1, 64, 128), f"unexpected shape {out.shape}"
    print("PASS")

    # ── Training-step demo with MoE aux loss ──────────────────────────────────
    print("\n── Training-step demo (MoE aux loss) ──")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    optimizer.zero_grad()
    y_pred = model(x)                          # forward — populates routing cache
    y_true = torch.zeros_like(y_pred)

    task_loss = criterion(y_pred, y_true)
    aux_loss  = model.get_load_balancing_loss()
    loss      = task_loss + 1e-2 * aux_loss    # small weight keeps it from dominating
    loss.backward()
    optimizer.step()

    print(f"Task loss : {task_loss.item():.6f}")
    print(f"Aux loss  : {aux_loss.item():.6f}")
    print(f"Combined  : {loss.item():.6f}")
    print("OK — backward pass completed.")
