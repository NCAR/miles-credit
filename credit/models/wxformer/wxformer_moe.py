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
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Mixture-of-Experts FeedForward
# ──────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single MoE expert — identical structure to FeedForward."""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts drop-in replacement for FeedForward.

    Each spatial position independently routes to top_k of num_experts experts
    via a 1×1 conv gate (preserves H×W structure rather than collapsing to a
    single token vector).  Gaussian noise on logits during training encourages
    load balancing (Switch Transformer style).

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
        # Routing statistics cached during forward() for aux loss
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

        # f_i: fraction of spatial positions that select expert i
        dispatch = torch.zeros_like(probs)
        ones = torch.ones(topk_idx.shape, dtype=probs.dtype, device=probs.device)
        dispatch.scatter_add_(1, topk_idx, ones)         # (B, E, H, W)
        f_i = dispatch.mean(dim=(0, 2, 3))               # (E,)
        # p_i: mean routing probability per expert
        p_i = probs.mean(dim=(0, 2, 3))                  # (E,)
        return self.num_experts * (f_i * p_i).sum()


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────

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
        self.window_size = window_size
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dpb = DynamicPositionBias(dim // 4)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        *_, height, width, heads, wsz, device = (
            *x.shape, self.heads, self.window_size, x.device,
        )
        x = self.norm(x)

        if self.attn_type == "short":
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz, s2=wsz)
        else:
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz, l2=wsz)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rearrange(rel_pos, "c i j -> (i j) c").to(x.dtype)
        biases = self.dpb(rel_pos)
        sim = sim + biases[self.rel_pos_indices]

        attn = self.dropout(sim.softmax(dim=-1))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz, y=wsz)
        out = self.to_out(out)

        if self.attn_type == "short":
            out = rearrange(out, "(b h w) d s1 s2 -> b d (h s1) (w s2)",
                            h=height // wsz, w=width // wsz)
        else:
            out = rearrange(out, "(b h w) d l1 l2 -> b d (l1 h) (l2 w)",
                            h=height // wsz, w=width // wsz)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Transformer blocks
# ──────────────────────────────────────────────────────────────────────────────

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

    def get_load_balancing_loss(self, x=None):
        """Sum Switch Transformer aux loss across all MoE feedforward layers."""
        total = None
        for _, short_ff, _, long_ff in self.layers:
            for ff in (short_ff, long_ff):
                loss = ff.load_balancing_loss()
                total = loss if total is None else total + loss
        return total


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
        # MoE arguments — may be scalars (applied to all levels) or 4-tuples
        # (one value per encoder depth, shallow→deep).
        num_experts: int = 4,
        top_k: int = 2,
        moe_noise_std: float = 0.1,
    ):
        super().__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)
        num_experts = cast_tuple(num_experts, 4)
        top_k = cast_tuple(top_k, 4)

        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.output_frames = output_frames

        self.base_input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = self.base_input_channels * frames
        self.base_output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = self.base_output_channels * output_frames

        last_dim = dim[-1]
        dims = [self.input_channels, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossEmbedLayer(di, do, ks, stride=s),
                TransformerWithMoE(
                    do,
                    local_window_size=lws,
                    global_window_size=gws,
                    depth=dep,
                    dim_head=dim_head,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    num_experts=ne,
                    top_k=tk,
                    moe_noise_std=moe_noise_std,
                ),
            ])
            for (di, do), dep, gws, lws, ks, s, ne, tk in zip(
                dim_pairs, depth, global_window_size, local_window_size,
                cross_embed_kernel_sizes, cross_embed_strides, num_experts, top_k,
            )
        ])

        num_groups = dim[0]
        self.upsample_with_ps = upsample_with_ps

        if upsample_with_ps:
            scale = 2
            self.up_block1 = UpBlockPS(last_dim,               last_dim // 2, num_groups)
            self.up_block2 = UpBlockPS(2 * (last_dim // 2),   last_dim // 4, num_groups)
            self.up_block3 = UpBlockPS(2 * (last_dim // 4),   last_dim // 8, num_groups)
            self.up_block4 = nn.Sequential(
                nn.Conv2d(2 * (last_dim // 8), self.output_channels * scale ** 2,
                          kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=scale),
                nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
            )
        else:
            self.up_block1 = UpBlock(last_dim,               last_dim // 2, num_groups)
            self.up_block2 = UpBlock(2 * (last_dim // 2),   last_dim // 4, num_groups)
            self.up_block3 = UpBlock(2 * (last_dim // 4),   last_dim // 8, num_groups)
            self.up_block4 = nn.ConvTranspose2d(
                2 * (last_dim // 8), self.output_channels, kernel_size=4, stride=2, padding=1
            )

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

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear",
                          align_corners=False)

        b, _, h, w = x.shape
        x = x.view(b, self.base_output_channels, self.output_frames, h, w)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Minimal smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Same spatial/channel config as the test suite (64×128, strides 2×2×2×2)
    model = CrossFormer(
        image_height=64,
        image_width=128,
        frames=2,
        output_frames=1,
        channels=2,
        surface_channels=1,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(16, 32, 64, 128),
        depth=(1, 1, 2, 1),
        dim_head=16,
        global_window_size=(2, 2, 2, 2),
        local_window_size=4,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        # MoE: per-level tuples — deeper levels get more experts & harder routing
        num_experts=(4, 4, 8, 8),
        top_k=(2, 2, 1, 1),
        moe_noise_std=0.1,
    )
    model.train()  # enable noise for load-balancing stats

    # base_input_channels = 2*2 + 1 + 1 = 6  →  input: (1, 6, 2, 64, 128)
    x = torch.randn(1, 6, 2, 64, 128)

    out = model(x)
    lb_loss = model.get_load_balancing_loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Load-balancing loss: {lb_loss.item():.6f}")
    assert lb_loss.item() > 0.0, "Load-balancing loss should be nonzero"
    print("OK — MoE smoke test passed.")

    # ── Training-step pattern ─────────────────────────────────────────────────
    # Demonstrates how to combine the task loss with the MoE auxiliary loss.
    # The routing stats (f_i, p_i) are already cached from the forward pass
    # above, so get_load_balancing_loss() reads them without a second pass.
    # encoder_output is accepted by the method signature for interface symmetry
    # but is unused — cached stats from forward() are sufficient.
    print("\n── Training-step demo ──")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    y_pred = model(x)                              # forward (populates routing cache)
    y_true = torch.zeros_like(y_pred)              # dummy target

    loss = criterion(y_pred, y_true)

    # Collect aux loss from all transformer layers
    encoder_output = x                             # passed for API consistency; unused internally
    aux_loss = 0.0
    for cel, transformer in model.layers:
        aux_loss = aux_loss + transformer.get_load_balancing_loss(encoder_output)

    loss = loss + 1e-2 * aux_loss                  # small weight — don't let it dominate
    loss.backward()
    optimizer.step()

    print(f"Task loss   : {criterion(y_pred, y_true).item():.6f}")
    print(f"Aux loss    : {aux_loss.item():.6f}")
    print(f"Combined    : {loss.item():.6f}")
    print("OK — backward pass completed.")
