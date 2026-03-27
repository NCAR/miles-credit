import torch
import logging
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from credit.models.base_model import BaseModel
from credit.postblock import PostBlock
from credit.boundary_padding import TensorPadding
from credit.models.unet_attention_modules import load_unet_attention

logger = logging.getLogger(__name__)

# helpers


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def norm_window_sizes(ws, num_levels=4):
    """Normalise a window-size spec to a tuple of num_levels (H, W) pairs.

    Accepts:
      int           → same square window at every level
      (H, W)        → same anisotropic window at every level
      (s0,s1,s2,s3) → per-level square windows
      ((H0,W0),…)  → per-level anisotropic windows
    """
    if isinstance(ws, int):
        return tuple((ws, ws) for _ in range(num_levels))
    ws = tuple(ws)
    if len(ws) == 2 and all(isinstance(w, int) for w in ws):
        return tuple(ws for _ in range(num_levels))
    assert len(ws) == num_levels, f"window_size must be int, (H,W), or {num_levels}-tuple; got {ws}"
    return tuple((w, w) if isinstance(w, int) else tuple(w) for w in ws)


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


# cube embedding
class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, T, C, Lat, Lon = x.shape
        x = self.proj(x)

        # ----------------------------------- #
        # Layer norm on T*lat*lon
        x = x.reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)

        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        num_groups,
        num_residuals=2,
        attention_type=None,
        reduction=32,
        spatial_kernel=7,
    ):
        super().__init__()

        # Always use ConvTranspose2d for upsampling
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.output_channels = out_chans

        # Residual stack
        blk = []
        for _ in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

        # Optional attention
        self.attention = load_unet_attention(attention_type, out_chans, reduction, spatial_kernel)

    def forward(self, x):
        x = self.conv(x)
        shortcut = x

        x = self.b(x)
        x = x + shortcut

        if self.attention is not None:
            x = self.attention(x)

        return x


class UpBlockPS(nn.Module):
    """Pixel-shuffle upsampling — avoids ConvTranspose2d checkerboard artifacts."""

    def __init__(self, in_ch, out_ch, num_groups, scale=2, num_residuals=2):
        super().__init__()
        # sub-pixel conv at low res
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale)
        # sharpening branch (identity init)
        self.sharp = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.xavier_normal_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        # residual stack
        blk = []
        for _ in range(num_residuals):
            blk += [nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(num_groups, out_ch), nn.SiLU()]
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.ps(self.conv(x))  # upsample+conv at low res
        x = x + self.sharp(x)  # sharpen residual
        sc = x
        x = self.b(x)
        return x + sc


# cross embed layer


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class CrossExpandLayer(nn.Module):
    """Exact inverse of CrossEmbedLayer.

    Parallel ConvTranspose2d branches with the same kernel sizes and channel
    split formula as CrossEmbedLayer, so the two are perfectly symmetric.
    For every (kernel, stride) pair, padding=(kernel-stride)//2 gives the
    same formula as in the forward path, and output_padding=0 is exact for
    spatial dims that are divisible by stride (which they must be for the
    encoder to function at all).
    """

    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
                for kernel, dim_scale in zip(kernel_sizes, dim_scales)
            ]
        )

    def forward(self, x):
        fmaps = tuple(conv(x) for conv in self.convs)
        return torch.cat(fmaps, dim=1)


class TransformerDecodeLevel(nn.Module):
    """One decoder stage that mirrors one encoder stage.

    1. CrossExpandLayer  — upsample by `stride` (dim_in → dim_out)
    2. cat skip          — (B, 2*dim_out, H, W)
    3. 1×1 proj          — (B, dim_out, H, W)
    4. Transformer block — local + global attention at the upsampled resolution
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride,
        local_window_size,
        global_window_size,
        depth=2,
        dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.expand = CrossExpandLayer(dim_in, dim_out, kernel_sizes, stride)
        self.skip_proj = nn.Conv2d(dim_out * 2, dim_out, 1)
        self.transformer = Transformer(
            dim_out,
            local_window_size=local_window_size,
            global_window_size=global_window_size,
            depth=depth,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def forward(self, x, skip):
        x = self.expand(x)
        x = self.skip_proj(torch.cat([x, skip], dim=1))
        x = self.transformer(x)
        return x


class CrossAttention(nn.Module):
    """Local windowed cross-attention.

    Query from the upsampled stream (x), keys/values from the encoder skip.
    Both streams are partitioned into local windows of size window_size before
    computing attention, making it O(H*W * wsz^2) rather than O((H*W)^2).

    No positional bias is added here; the skip features already carry strong
    spatial structure from the encoder.
    """

    def __init__(self, dim, window_size, dim_head=32, dropout=0.0):
        super().__init__()
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        *_, height, width = x.shape
        wsz_h, wsz_w = self.window_size
        heads = self.heads

        xn = rearrange(self.norm_q(x), "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz_h, s2=wsz_w)
        skn = rearrange(self.norm_kv(skip), "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz_h, s2=wsz_w)

        q = self.to_q(xn)
        k, v = self.to_kv(skn).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz_h, y=wsz_w)
        out = self.to_out(out)
        out = rearrange(out, "(b h w) d s1 s2 -> b d (h s1) (w s2)", h=height // wsz_h, w=width // wsz_w)
        return out


class CrossAttentionDecodeLevel(nn.Module):
    """Decoder stage using PixelShuffle upsampling + cross-attention skip fusion.

    1. PixelShuffle ×2  — artifact-free 2× upsample (dim_in → dim_out)
    2. CrossAttention   — Q from upsampled, K/V from encoder skip
    3. FeedForward      — pointwise refinement
    """

    def __init__(self, dim_in, dim_out, local_window_size, dim_head=32, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.cross_attn = CrossAttention(dim_out, local_window_size, dim_head, attn_dropout)
        self.ff = FeedForward(dim_out, dropout=ff_dropout)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.cross_attn(x, skip) + x
        x = self.ff(x) + x
        return x


# dynamic positional bias


class DynamicPositionBias(nn.Module):
    def __init__(self, dim):
        super(DynamicPositionBias, self).__init__()
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


# ── 2D RoPE helpers ──────────────────────────────────────────────────────────


def _precompute_window_rope(wsz_h, wsz_w, dim_head, device, dtype, base=10000):
    """Within-window 2D RoPE.  Returns (cos, sin) each of shape (wsz_h*wsz_w, dim_head).

    The head dimension is split in half: the first half encodes the row index (lat),
    the second half encodes the column index (lon).  Each half uses dim_head//4
    frequency bands.  Tokens within the window therefore carry relative (row, col)
    positional information without any learned parameters.
    """
    quarter = dim_head // 4
    freq = 1.0 / (base ** (torch.arange(quarter, device=device, dtype=dtype) / quarter))

    pos_h = torch.arange(wsz_h, device=device, dtype=dtype)
    pos_w = torch.arange(wsz_w, device=device, dtype=dtype)
    theta_h = torch.outer(pos_h, freq)  # (wsz_h, quarter)
    theta_w = torch.outer(pos_w, freq)  # (wsz_w, quarter)

    # Expand to (wsz_h, wsz_w, quarter) then flatten to (L, quarter)
    theta_h = theta_h[:, None, :].expand(wsz_h, wsz_w, quarter).reshape(wsz_h * wsz_w, quarter)
    theta_w = theta_w[None, :, :].expand(wsz_h, wsz_w, quarter).reshape(wsz_h * wsz_w, quarter)

    # Concatenate lat + lon frequencies → (L, dim_head//2)
    theta = torch.cat([theta_h, theta_w], dim=-1)  # (L, dim_head//2)

    # Build (L, dim_head) cos/sin for the rotate_half convention
    # cat([θ.cos(), θ.cos()]) so that pair (x[k], x[k+half]) rotates by θ[k]
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)  # (L, dim_head)
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    return cos, sin


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    """Apply 2D RoPE to Q and K.  cos/sin: (L, d) → broadcast over (B, heads, L, d)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# transformer classes


class LayerNorm(nn.Module):
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
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU feed-forward: SiLU(xW₁) ⊙ (xW₂), then project back.

    Hidden dimension is scaled to 2/3 * mult * dim so the total parameter
    count matches the original GELU-MLP with the same mult.
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.norm = LayerNorm(dim)
        self.proj_up = nn.Conv2d(dim, hidden * 2, 1)  # produces gate + value
        self.proj_out = nn.Conv2d(hidden, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        gate, val = self.proj_up(x).chunk(2, dim=1)
        x = F.silu(gate) * val
        x = self.dropout(x)
        return self.proj_out(x)


class Attention(nn.Module):
    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0, use_rope=False):
        super().__init__()
        assert attn_type in {"short", "long", "shifted"}
        heads = dim // dim_head
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.attn_type = attn_type
        self.use_rope = use_rope
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        if use_rope:
            # RoPE: parameter-free, cached at eval time
            self._rope_cache = None  # (cos, sin) each (L, dim_head)
        else:
            # 2D Dynamic Position Bias (MLP over relative offsets)
            self.dpb = DynamicPositionBias(dim // 4)
            self._dpb_cache = None
            wsz_h, wsz_w = self.window_size
            pos_h = torch.arange(wsz_h)
            pos_w = torch.arange(wsz_w)
            grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = grid[:, None] - grid[None, :]
            rel_pos[..., 0] += wsz_h - 1
            rel_pos[..., 1] += wsz_w - 1
            rel_pos_indices = rel_pos[..., 0] * (2 * wsz_w - 1) + rel_pos[..., 1]
            self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def train(self, mode=True):
        if mode:
            if self.use_rope:
                self._rope_cache = None
            else:
                self._dpb_cache = None
        return super().train(mode)

    def forward(self, x):
        *_, height, width, heads, device = (*x.shape, self.heads, x.device)
        wsz_h, wsz_w = self.window_size
        x = self.norm(x)

        if self.attn_type == "shifted":
            shift_h, shift_w = wsz_h // 2, wsz_w // 2
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(-2, -1))

        if self.attn_type in {"short", "shifted"}:
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz_h, s2=wsz_w)
        else:
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz_h, l2=wsz_w)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))

        dropout_p = self.dropout.p if self.training else 0.0

        if self.use_rope:
            # 2D RoPE: parameter-free, encodes within-window (row, col) position
            # No attn_mask → FlashAttention backend
            if self.training or self._rope_cache is None:
                cos, sin = _precompute_window_rope(wsz_h, wsz_w, self.dim_head, device, q.dtype)
                if not self.training:
                    self._rope_cache = (cos.detach(), sin.detach())
            else:
                cos, sin = self._rope_cache
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q, k = _apply_rope(q, k, cos, sin)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # 2D Dynamic Position Bias — additive bias → memory-efficient backend
            if self.training or self._dpb_cache is None:
                pos_h = torch.arange(-wsz_h, wsz_h + 1, device=device)
                pos_w = torch.arange(-wsz_w, wsz_w + 1, device=device)
                grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing="ij"))
                rel_pos = rearrange(grid, "c i j -> (i j) c").to(q.dtype)
                dpb_bias = self.dpb(rel_pos)[self.rel_pos_indices]
                if not self.training:
                    self._dpb_cache = dpb_bias.detach()
            else:
                dpb_bias = self._dpb_cache.to(q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=dpb_bias, dropout_p=dropout_p)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz_h, y=wsz_w)
        out = self.to_out(out)

        if self.attn_type in {"short", "shifted"}:
            out = rearrange(out, "(b h w) d s1 s2 -> b d (h s1) (w s2)", h=height // wsz_h, w=width // wsz_w)
            if self.attn_type == "shifted":
                out = torch.roll(out, shifts=(shift_h, shift_w), dims=(-2, -1))
        else:
            out = rearrange(out, "(b h w) d l1 l2 -> b d (l1 h) (l2 w)", h=height // wsz_h, w=width // wsz_w)
        return out


class AxialAttention(nn.Module):
    """Full-sequence axial attention along one spatial axis.

    axis='lon': each latitude row attends to all W longitude tokens.
                Exploits ERA5's periodic longitude; no positional masking needed.
    axis='lat': each longitude column attends to all H latitude tokens.
                Captures meridional teleconnections (jets, Hadley cell, etc.)

    Uses Conv1d projections so the implementation is axis-agnostic and compact.
    No positional bias: the shifted-window DPB already covers local structure;
    axial attention is meant to capture global patterns.
    """

    def __init__(self, dim, axis, dim_head=32, dropout=0.0):
        super().__init__()
        assert axis in {"lon", "lat"}
        self.axis = axis
        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        inner_dim = dim_head * self.heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        n = self.heads
        xn = self.norm(x)

        dropout_p = self.dropout.p if self.training else 0.0

        if self.axis == "lon":
            s = rearrange(xn, "b c h w -> (b h) c w")
            q, k, v = self.to_qkv(s).chunk(3, dim=1)
            q, k, v = map(lambda t: rearrange(t, "bh (n d) w -> bh n w d", n=n), (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            out = self.to_out(rearrange(out, "bh n w d -> bh (n d) w"))
            out = rearrange(out, "(b h) c w -> b c h w", b=B)
        else:  # lat
            s = rearrange(xn, "b c h w -> (b w) c h")
            q, k, v = self.to_qkv(s).chunk(3, dim=1)
            q, k, v = map(lambda t: rearrange(t, "bw (n d) h -> bw n h d", n=n), (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            out = self.to_out(rearrange(out, "bw n h d -> bw (n d) h"))
            out = rearrange(out, "(b w) c h -> b c h w", b=B)

        return out


class GridAttention(nn.Module):
    """MaxViT-style dilated grid attention with 2D sin/cos positional encoding.

    For each within-window position (i, j), all windows' token at that same
    position are collected into a sequence and attended jointly.  This gives
    every token a global receptive field at cost O(nh*nw) per grid cell
    rather than O((H*W)²) for full self-attention.

    Positional encoding: 2D sin/cos PE encodes each window's (row, col) index
    in the window grid so the model knows relative window positions.  The row
    half uses frequencies tuned to nh, the column half to nw, concatenated to
    form the full C-dim PE vector added to each token before Q/K/V projection.

    window_size : (wsz_h, wsz_w) — same partition as local attention.
    """

    def __init__(self, dim, window_size, dim_head=32, dropout=0.0):
        super().__init__()
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        heads = dim // dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._pe_cache = None  # (nh*nw, C) — invalidated on train()

    def train(self, mode=True):
        if mode:
            self._pe_cache = None
        return super().train(mode)

    @staticmethod
    def _make_grid_pe(nh, nw, dim, device, dtype):
        """2D sin/cos PE for window-grid positions, shape (nh*nw, dim).

        First dim//2 channels encode the row index; last dim//2 encode the col
        index.  Each half uses dim//4 sin + dim//4 cos frequencies.
        """
        half = dim // 2
        n_freq = half // 2  # sin + cos each
        freq = torch.arange(n_freq, device=device, dtype=dtype)
        freq = 1.0 / (10000 ** (2 * freq / half))  # (n_freq,)

        pos_h = torch.arange(nh, device=device, dtype=dtype)  # (nh,)
        pos_w = torch.arange(nw, device=device, dtype=dtype)  # (nw,)

        ang_h = pos_h.unsqueeze(1) * freq.unsqueeze(0)  # (nh, n_freq)
        ang_w = pos_w.unsqueeze(1) * freq.unsqueeze(0)  # (nw, n_freq)

        pe_h = torch.cat([torch.sin(ang_h), torch.cos(ang_h)], dim=1)  # (nh, half)
        pe_w = torch.cat([torch.sin(ang_w), torch.cos(ang_w)], dim=1)  # (nw, half)

        # Broadcast: every (k, l) pair gets [pe_h[k] || pe_w[l]]
        pe_h = pe_h.unsqueeze(1).expand(nh, nw, half)  # (nh, nw, half)
        pe_w = pe_w.unsqueeze(0).expand(nh, nw, half)  # (nh, nw, half)
        pe = torch.cat([pe_h, pe_w], dim=-1)  # (nh, nw, dim)
        return pe.reshape(nh * nw, dim)  # (L, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        wsz_h, wsz_w = self.window_size
        nh, nw = H // wsz_h, W // wsz_w
        heads = self.heads

        xn = self.norm(x)  # (B, C, H, W) — channel-wise norm

        # Dilated grid partition: group tokens by within-window position
        # Each "sequence" has length nh*nw (one token per window at position (i,j))
        xg = rearrange(xn, "b c (nh sh) (nw sw) -> (b sh sw) (nh nw) c", sh=wsz_h, sw=wsz_w)

        # 2D sin/cos PE — tells each token its window's (row, col) position
        if self.training or self._pe_cache is None:
            pe = self._make_grid_pe(nh, nw, C, x.device, xg.dtype)
            if not self.training:
                self._pe_cache = pe.detach()
        else:
            pe = self._pe_cache.to(dtype=xg.dtype)
        xg = xg + pe.unsqueeze(0)  # (N, L, C)

        qkv = self.to_qkv(xg)  # (N, L, 3*inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "n l (h d) -> n h l d", h=heads), (q, k, v))

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = rearrange(out, "n h l d -> n l (h d)")
        out = self.to_out(out)  # (N, L, C)

        # Reassemble into spatial feature map
        return rearrange(out, "(b sh sw) (nh nw) c -> b c (nh sh) (nw sw)", b=B, sh=wsz_h, sw=wsz_w, nh=nh, nw=nw)


class DeformableAttention(nn.Module):
    """Global deformable attention (Xia et al. 2022, adapted for weather models).

    Q attends over the full H×W grid; K/V are sampled at H/r × W/r positions
    learned via a lightweight offset network — giving each position a global
    but adaptive receptive field at O(H*W * H/r*W/r) cost.

    Periodic longitude boundary: the x-coordinate of the sampling grid is
    wrapped modulo 2 (in normalised [-1,1] space) so offsets can freely cross
    the antimeridian.  Latitude uses border clamping (poles are hard boundaries).

    downsample_factor : r — controls K/V resolution; larger → cheaper, coarser.
    offset_scale      : max offset in offset-grid pixels (default = r).
    """

    def __init__(self, dim, downsample_factor=4, offset_scale=None, dim_head=32, dropout=0.0):
        super().__init__()
        r = downsample_factor
        offset_scale = offset_scale if offset_scale is not None else r
        # kernel >= stride and (kernel-stride) even for symmetric padding
        offset_kernel_size = r + 2
        offset_padding = (offset_kernel_size - r) // 2

        heads = dim // dim_head
        inner_dim = dim_head * heads
        self.heads = heads
        self.downsample_factor = r
        self.norm = LayerNorm(dim)

        # Lightweight offset network: input features → (2,) offset map at 1/r res
        self.to_offsets = nn.Sequential(
            nn.Conv2d(dim, dim, offset_kernel_size, groups=dim, stride=r, padding=offset_padding),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.offset_scale = offset_scale

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.downsample_factor
        xn = self.norm(x)

        # ── Compute sampling offsets ─────────────────────────────────────────
        offsets = self.to_offsets(xn) * self.offset_scale  # (B, 2, Hd, Wd)
        Hd, Wd = offsets.shape[-2:]

        # Base grid: uniformly covers [-1,1]×[-1,1] in normalised coords
        # (x/lon axis first, then y/lat axis — matches grid_sample convention)
        gy = torch.linspace(-1, 1, Hd, device=x.device, dtype=x.dtype)
        gx = torch.linspace(-1, 1, Wd, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # (Hd, Wd)
        base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,Hd,Wd)

        # Offsets are in offset-grid pixels; convert to normalised coords
        off_norm_x = offsets[:, 0] / max(Wd - 1, 1) * 2  # (B, Hd, Wd)
        off_norm_y = offsets[:, 1] / max(Hd - 1, 1) * 2

        vgrid_x = base[:, 0] + off_norm_x  # (B, Hd, Wd)
        vgrid_y = base[:, 1] + off_norm_y

        # Periodic lon (x) — wrap to [-1,1]; clamped lat (y)
        vgrid_x = (vgrid_x + 1) % 2 - 1
        vgrid_y = vgrid_y.clamp(-1.0, 1.0)

        vgrid = torch.stack([vgrid_x, vgrid_y], dim=-1)  # (B, Hd, Wd, 2)

        # ── Sample K/V features at learned positions ─────────────────────────
        kv_feats = F.grid_sample(
            xn, vgrid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B, C, Hd, Wd)

        # ── Q/K/V projections ────────────────────────────────────────────────
        q = self.to_q(xn)  # (B, inner_dim, H,  W)
        k = self.to_k(kv_feats)  # (B, inner_dim, Hd, Wd)
        v = self.to_v(kv_feats)

        n = self.heads
        q = rearrange(q, "b (n d) h w -> b n (h w) d", n=n)
        k = rearrange(k, "b (n d) h w -> b n (h w) d", n=n)
        v = rearrange(v, "b (n d) h w -> b n (h w) d", n=n)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = rearrange(out, "b n (h w) d -> b (n d) h w", h=H, w=W)
        return self.to_out(out)


class BlockAttentionResidual(nn.Module):
    """Inter-block attention residual (Moonshot AI, Attention Residuals, 2025).

    Before each sublayer, gates over all previous depth-block representations
    plus the current feature map to form the sublayer input. Adapted from the
    original sequence formulation to (B, C, H, W) spatial features.

    When blocks list is empty (first depth iteration) returns x unchanged.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, blocks: list, x: torch.Tensor) -> torch.Tensor:
        if not blocks:
            return x
        V = torch.stack(blocks + [x])  # (N+1, B, C, H, W)
        N1, B, C, H, W = V.shape
        K = self.norm(V.reshape(N1 * B, C, H, W)).reshape(N1, B, C, H, W)
        logits = torch.einsum("c,nbchw->nbhw", self.w, K)  # (N+1, B, H, W)
        h = torch.einsum("nbhw,nbchw->bchw", logits.softmax(0), V)
        return h


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth=4,
        dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_shifted_windows=False,
        use_swiglu=False,
        use_axial=False,
        use_grid=False,
        use_deformable=False,
        use_rope=False,
        use_attn_res=False,
    ):
        super().__init__()
        FF = FeedForwardSwiGLU if use_swiglu else FeedForward
        self.use_axial = use_axial
        self.use_grid = use_grid
        self.use_deformable = use_deformable

        def A(t, w):
            return Attention(dim, t, w, dim_head, attn_dropout, use_rope=use_rope)

        if use_axial:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            AxialAttention(dim, "lon", dim_head, attn_dropout),
                            FF(dim, dropout=ff_dropout),
                            AxialAttention(dim, "lat", dim_head, attn_dropout),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        elif use_deformable and use_shifted_windows:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            A("shifted", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            DeformableAttention(dim, dim_head=dim_head, dropout=attn_dropout),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        elif use_deformable:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            DeformableAttention(dim, dim_head=dim_head, dropout=attn_dropout),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        elif use_grid and use_shifted_windows:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            A("shifted", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            GridAttention(dim, local_window_size, dim_head, attn_dropout),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        elif use_grid:
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            GridAttention(dim, local_window_size, dim_head, attn_dropout),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        else:
            second_attn_type = "shifted" if use_shifted_windows else "long"
            second_window = local_window_size if use_shifted_windows else global_window_size
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            A("short", local_window_size),
                            FF(dim, dropout=ff_dropout),
                            A(second_attn_type, second_window),
                            FF(dim, dropout=ff_dropout),
                        ]
                    )
                    for _ in range(depth)
                ]
            )
        self.use_grid_and_shift = use_grid and use_shifted_windows
        self.use_deformable_and_shift = use_deformable and use_shifted_windows
        self.use_attn_res = use_attn_res
        six_sublayer = use_axial or (use_grid and use_shifted_windows) or (use_deformable and use_shifted_windows)
        n_sublayers = 6 if six_sublayer else 4
        if use_attn_res:
            self.attn_res = nn.ModuleList([BlockAttentionResidual(dim) for _ in range(n_sublayers)])

    def forward(self, x):
        six = self.use_axial or self.use_grid_and_shift or self.use_deformable_and_shift
        if self.use_attn_res:
            blocks = []
            rs = self.attn_res
            if six:
                for a, fa, b, fb, c, fc in self.layers:
                    x = a(rs[0](blocks, x)) + x
                    x = fa(rs[1](blocks, x)) + x
                    x = b(rs[2](blocks, x)) + x
                    x = fb(rs[3](blocks, x)) + x
                    x = c(rs[4](blocks, x)) + x
                    x = fc(rs[5](blocks, x)) + x
                    blocks.append(x)
            else:
                for a, fa, b, fb in self.layers:
                    x = a(rs[0](blocks, x)) + x
                    x = fa(rs[1](blocks, x)) + x
                    x = b(rs[2](blocks, x)) + x
                    x = fb(rs[3](blocks, x)) + x
                    blocks.append(x)
        elif six:
            for a, fa, b, fb, c, fc in self.layers:
                x = a(x) + x
                x = fa(x) + x
                x = b(x) + x
                x = fb(x) + x
                x = c(x) + x
                x = fc(x) + x
        else:
            for a, fa, b, fb in self.layers:
                x = a(x) + x
                x = fa(x) + x
                x = b(x) + x
                x = fb(x) + x
        return x


# classes


class CrossFormer(BaseModel):
    def __init__(
        self,
        image_height: int = 640,
        patch_height: int = 1,
        image_width: int = 1280,
        patch_width: int = 1,
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
        use_spectral_norm: bool = True,
        attention_type: str = None,
        interp: bool = True,
        upsample_with_ps: bool = True,
        upsample_with_transformer: bool = False,
        upsample_with_cross_attn: bool = False,
        use_shifted_windows: bool = True,
        use_swiglu: bool = True,
        use_axial: bool = False,
        use_grid: bool = False,
        use_deformable: bool = True,
        use_rope: bool = False,
        use_attn_res: bool = False,
        num_residuals: int = 4,
        lat_file: str = None,
        padding_conf: dict = None,
        post_conf: dict = None,
        **kwargs,
    ):
        """
        WXFormer v2 (promoted from v3) — CrossFormer with ablation-validated improvements.

        Defaults are the ablation-validated best configuration (-12.6% vs v1 at 5000 steps):
          upsample_with_ps=True        : PixelShuffle decoder (eliminates checkerboard artifacts)
          num_residuals=4              : residual conv blocks per decoder UpBlock
          use_swiglu=True              : SwiGLU feedforward
          use_shifted_windows=True     : Swin-style cyclic-shift attention
          use_deformable=True          : deformable attention over H/4 x W/4 K/V grid (new best)

        To recover OG wxformer + PixelShuffle behaviour, set all flags to False:
          CrossFormer(use_shifted_windows=False, use_swiglu=False, use_deformable=False)
        """
        super().__init__()

        dim = tuple(dim)
        depth = tuple(depth)
        global_window_size = tuple(global_window_size)
        cross_embed_kernel_sizes = tuple([tuple(_) for _ in cross_embed_kernel_sizes])
        cross_embed_strides = tuple(cross_embed_strides)

        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.upsample_with_ps = upsample_with_ps
        self.upsample_with_transformer = upsample_with_transformer
        self.upsample_with_cross_attn = upsample_with_cross_attn
        self.frames = frames
        self.output_frames = output_frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        self.use_interp = interp
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = post_conf["activate"]

        # input channels
        self.input_only_channels = input_only_channels
        self.base_input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = self.base_input_channels * frames

        # sin/cos latitude embedding — loaded from the same file as latitude_weights.
        # Stored as a (1, 2, H, 1) buffer so it broadcasts across batch and longitude.
        # Injected into the 5D input tensor before padding, so it receives the same
        # circular-lon and pole-reflect treatment as all other channels.
        self.use_lat_embed = False
        if lat_file is not None:
            import xarray as xr

            ds = xr.open_dataset(lat_file)
            lat_key = [k for k in ds.coords if "lat" in k.lower()][0]
            lat_deg = torch.tensor(ds[lat_key].values, dtype=torch.float32)  # (H,)
            lat_rad = torch.deg2rad(lat_deg)
            lat_embed = torch.stack([torch.sin(lat_rad), torch.cos(lat_rad)], dim=0)  # (2, H)
            lat_embed = lat_embed.unsqueeze(0).unsqueeze(-1)  # (1, 2, H, 1)
            self.register_buffer("lat_embed", lat_embed)
            self.use_lat_embed = True
            self.input_channels += 2  # 2 extra channels per frame
            logger.info(f"Loaded sin/cos lat embedding from {lat_file} ({lat_embed.shape[2]} lat points)")

        # output channels
        self.base_output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = self.base_output_channels * output_frames

        if kwargs.get("diffusion"):
            self.input_channels = self.input_channels + self.output_channels

        dim = tuple(dim)
        num_levels = len(dim)
        depth = cast_tuple(depth, num_levels)
        global_window_size = norm_window_sizes(global_window_size, num_levels)
        local_window_size = norm_window_sizes(local_window_size, num_levels)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, num_levels)
        cross_embed_strides = cast_tuple(cross_embed_strides, num_levels)

        assert len(depth) == num_levels
        assert len(cross_embed_kernel_sizes) == num_levels
        assert len(cross_embed_strides) == num_levels

        # dimensions
        last_dim = dim[-1]
        first_dim = self.input_channels if (patch_height == 1 and patch_width == 1) else dim[0]
        dims = [first_dim, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # allocate cross embed layers
        self.layers = nn.ModuleList([])

        for (
            dim_in,
            dim_out,
        ), num_layers, global_wsize, local_wsize, kernel_sizes, stride in zip(
            dim_in_and_out,
            depth,
            global_window_size,
            local_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            cross_embed_layer = CrossEmbedLayer(
                dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernel_sizes, stride=stride
            )
            transformer_layer = Transformer(
                dim=dim_out,
                local_window_size=local_wsize,
                global_window_size=global_wsize,
                depth=num_layers,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_shifted_windows=use_shifted_windows,
                use_swiglu=use_swiglu,
                use_axial=use_axial,
                use_grid=use_grid,
                use_deformable=use_deformable,
                use_rope=use_rope,
                use_attn_res=use_attn_res,
            )
            self.layers.append(nn.ModuleList([cross_embed_layer, transformer_layer]))

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        self.cube_embedding = CubeEmbedding(
            (frames, image_height, image_width),
            (frames, patch_height, patch_width),
            self.input_channels,
            dim[0],
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.num_levels = num_levels

        if self.upsample_with_cross_attn:
            # ── Cross-attention decoder ────────────────────────────────────────
            # dec_layers[i]: PixelShuffle ×2 upsample + cross-attention skip fusion
            #   Q = upsampled features, K/V = encoder skip at that level
            #   Window size uses the local_window_size at the spatial level reached.
            # dec_final: two chained PixelShuffle ×2 to handle the stride-4 final step
            #            (no skip connection at full resolution).
            self.dec_layers = nn.ModuleList()
            for i in range(num_levels - 1):
                enc_level = num_levels - 1 - i
                spatial_lvl = enc_level - 1
                self.dec_layers.append(
                    CrossAttentionDecodeLevel(
                        dim_in=dim[enc_level],
                        dim_out=dim[spatial_lvl],
                        local_window_size=local_window_size[spatial_lvl],
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    )
                )
            # Final: stride-4 upsample via two chained PixelShuffle ×2, no skip
            mid_ch = dim[0]
            self.dec_final = nn.Sequential(
                nn.Conv2d(dim[0], mid_ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(mid_ch, self.output_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
            )

        elif self.upsample_with_transformer:
            # ── Transformer decoder (symmetric inverse of the encoder) ─────────
            # dec_layers[i] inverts encoder level (num_levels-1-i):
            #   CrossExpandLayer  — same kernel_sizes/stride as that encoder level
            #   skip cat + 1×1   — fuse encoder skip connection
            #   Transformer       — local + global attention at the upsampled res
            #                       using the same window sizes as that encoder level
            # dec_final: CrossExpandLayer only, inverts encoder level 0 (stride-4
            #            patch embedding), no skip and no transformer here.
            self.dec_layers = nn.ModuleList()
            for i in range(num_levels - 1):
                enc_level = num_levels - 1 - i  # encoder level being inverted
                spatial_lvl = enc_level - 1  # resolution after expanding
                self.dec_layers.append(
                    TransformerDecodeLevel(
                        dim_in=dim[enc_level],
                        dim_out=dim[spatial_lvl],
                        kernel_sizes=cross_embed_kernel_sizes[enc_level],
                        stride=cross_embed_strides[enc_level],
                        local_window_size=local_window_size[spatial_lvl],
                        global_window_size=global_window_size[spatial_lvl],
                        depth=depth[spatial_lvl],
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    )
                )
            self.dec_final = CrossExpandLayer(
                dim_in=dim[0],
                dim_out=self.output_channels,
                kernel_sizes=cross_embed_kernel_sizes[0],
                stride=cross_embed_strides[0],
            )
        else:
            # ── Conv decoder (UpBlock / UpBlockPS) ────────────────────────────
            dec_dims = [last_dim // (2**i) for i in range(num_levels + 1)]
            self.dec_dims = dec_dims  # exposed for ensemble subclass

            self.up_blocks = nn.ModuleList()
            for i in range(num_levels - 1):
                in_ch = dec_dims[i] if i == 0 else 2 * dec_dims[i]
                out_ch = dec_dims[i + 1]
                num_grp = max(1, min(dim[0], out_ch))
                while out_ch % num_grp != 0:
                    num_grp //= 2
                if self.upsample_with_ps:
                    self.up_blocks.append(UpBlockPS(in_ch, out_ch, num_grp, num_residuals=num_residuals))
                else:
                    self.up_blocks.append(
                        UpBlock(in_ch, out_ch, num_grp, num_residuals=num_residuals, attention_type=attention_type)
                    )

            in_ch_final = 2 * dec_dims[num_levels - 1]
            if self.upsample_with_ps:
                scale = 2
                self.up_block_out = nn.Sequential(
                    nn.Conv2d(in_ch_final, self.output_channels * (scale**2), kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(upscale_factor=scale),
                    nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
                )
            else:
                self.up_block_out = nn.ConvTranspose2d(
                    in_ch_final, self.output_channels, kernel_size=4, stride=2, padding=1
                )

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        if self.use_post_block:
            if "skebs" in post_conf.keys():
                if post_conf["skebs"].get("activate", False) and post_conf["skebs"].get(
                    "freeze_base_model_weights", False
                ):
                    logger.warning("freezing all base model weights due to skebs config")
                    for param in self.parameters():
                        param.requires_grad = False

            logger.info("using postblock")
            self.postblock = PostBlock(post_conf)

    def forward(self, x):
        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        # Inject sin(lat)/cos(lat) channels BEFORE padding so they receive the
        # same circular-lon and pole-reflect treatment as all other channels.
        # lat_embed is (1, 2, H, 1); x is (B, C, T, H, W).
        if self.use_lat_embed:
            B, C, T, H, W = x.shape
            # lat_embed is (1, 2, H, 1); unsqueeze(2) → (1, 2, 1, H, 1) then expand to (B, 2, T, H, W)
            lat = self.lat_embed.unsqueeze(2).expand(B, -1, T, -1, W)
            x = torch.cat([x, lat], dim=1)

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)

        if self.frames > 1:
            b, c, t, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
        else:
            x = x.squeeze(2)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Decoder
        if self.upsample_with_cross_attn or self.upsample_with_transformer:
            # dec_layers[i] consumes encodings[num_levels-2-i] as its skip connection.
            for i, dec in enumerate(self.dec_layers):
                skip = encodings[self.num_levels - 2 - i]
                x = dec(x, skip)
            x = self.dec_final(x)
        else:
            # Conv decoder: first upsample from bottleneck, then interleave skip-cats
            x = self.up_blocks[0](x)
            for i in range(1, self.num_levels):
                x = torch.cat([x, encodings[self.num_levels - 1 - i]], dim=1)
                if i < self.num_levels - 1:
                    x = self.up_blocks[i](x)
                else:
                    x = self.up_block_out(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        b, _, h, w = x.shape
        x = x.view(b, self.base_output_channels, self.output_frames, h, w)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x

    def rk4(self, x):
        def integrate_step(x, k, factor):
            return self.forward(x + k * factor)

        k1 = self.forward(x)
        k1 = torch.cat([x[:, :, -2:-1], k1], dim=2)
        k2 = integrate_step(x, k1, 0.5)
        k2 = torch.cat([x[:, :, -2:-1], k2], dim=2)
        k3 = integrate_step(x, k2, 0.5)
        k3 = torch.cat([x[:, :, -2:-1], k3], dim=2)
        k4 = integrate_step(x, k3, 1.0)

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6
