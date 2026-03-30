"""
ArchesWeather — Hétu et al., INRIA 2024.
https://arxiv.org/abs/2412.12971   Apache-2.0
Original code: https://github.com/INRIA/geoarches

Key ideas:
  1. Local window self-attention (spatial) for mesoscale interactions.
  2. Column attention: at each spatial position, attention across the
     channel dimension — approximates vertical coupling across pressure
     levels without requiring the level structure to be explicit.
  3. The two attention types alternate block-by-block.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


def window_partition(x, ws):
    """(B, H, W, C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_unpartition(x, ws, H, W):
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    B = x.shape[0] // (H // ws * W // ws)
    x = x.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window attention block (spatial)
# ---------------------------------------------------------------------------


class WindowAttentionBlock(nn.Module):
    """
    Local window self-attention on a 2D (H, W) feature map.
    Input/output: (B, H, W, d_model).
    """

    def __init__(self, d_model, num_heads, window_size=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim**-0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, drop=drop)

    def _attn(self, x):
        # x: (B*nW, ws*ws, d_model)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, N, D)

    def forward(self, x):
        B, H, W, C = x.shape
        ws = self.ws
        # pad to multiple of ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        # window attention
        wins = window_partition(x, ws)  # (B*nW, ws, ws, C)
        wins_flat = wins.view(-1, ws * ws, C)
        wins_flat = wins_flat + self.proj(self._attn(self.norm1(wins_flat)))
        wins = wins_flat.view(-1, ws, ws, C)
        x = window_unpartition(wins, ws, Hp, Wp)

        # trim padding
        x = x[:, :H, :W, :].contiguous()

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Column attention block (across channels at each spatial position)
# ---------------------------------------------------------------------------


class ColumnAttentionBlock(nn.Module):
    """
    Attention applied across the channel dimension at each (h, w) position.
    This approximates ArchesWeather's vertical column attention: information
    flows between all d_model features at each grid point, modelling
    cross-level and cross-variable coupling.

    Input/output: (B, H, W, d_model).
    """

    def __init__(self, d_model, num_heads, n_col_tokens=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        # Split d_model into n_col_tokens "column tokens" of dim col_dim
        assert d_model % n_col_tokens == 0, f"d_model ({d_model}) must be divisible by n_col_tokens ({n_col_tokens})"
        self.n_col = n_col_tokens
        self.col_dim = d_model // n_col_tokens
        self.num_heads = max(1, num_heads // n_col_tokens)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(self.col_dim, 3 * self.col_dim)
        self.proj = nn.Linear(self.col_dim, self.col_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, drop=drop)

        head_dim = self.col_dim // self.num_heads
        self.scale = head_dim**-0.5

    def _col_attn(self, x):
        # x: (B*H*W, n_col, col_dim)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, N, D)

    def forward(self, x):
        B, H, W, C = x.shape
        # Reshape channel dim → n_col tokens of col_dim each
        residual = x
        x_norm = self.norm1(x)  # (B, H, W, C)
        x_tok = x_norm.view(B * H * W, self.n_col, self.col_dim)
        x_tok = x_tok + self.proj(self._col_attn(x_tok))
        x = residual + x_tok.view(B, H, W, C)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ArchesWeatherBackbone(nn.Module):
    """
    ArchesWeather backbone: alternating window attention (spatial) and
    column attention (channel/vertical).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    img_size : tuple[int, int]
    patch_size : int
        Spatial patch size for initial embedding.
    d_model : int
        Embedding dimension (must be divisible by n_col_tokens).
    depth : int
        Total number of blocks (alternates window/column).
    num_heads : int
    window_size : int
        Window size for local window attention (in patch units).
    n_col_tokens : int
        Number of "column tokens" d_model is split into for column attention.
    mlp_ratio : float
    drop : float
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(192, 288),
        patch_size=4,
        d_model=256,
        depth=8,
        num_heads=8,
        window_size=8,
        n_col_tokens=8,
        mlp_ratio=4.0,
        drop=0.0,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Pad to multiple of patch_size
        self.pad_h = (patch_size - H % patch_size) % patch_size
        self.pad_w = (patch_size - W % patch_size) % patch_size
        Hp = H + self.pad_h
        Wp = W + self.pad_w
        self.n_h = Hp // patch_size
        self.n_w = Wp // patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm_in = nn.LayerNorm(d_model)

        # Alternating blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:
                self.blocks.append(WindowAttentionBlock(d_model, num_heads, window_size, mlp_ratio, drop))
            else:
                self.blocks.append(ColumnAttentionBlock(d_model, num_heads, n_col_tokens, mlp_ratio, drop))

        self.norm_out = nn.LayerNorm(d_model)

        # Pixel-shuffle decoder
        self.head = nn.Sequential(
            nn.Linear(d_model, out_channels * patch_size * patch_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

        # Patch embed: (B, C, H, W) → (B, d, n_h, n_w)
        x = self.patch_embed(x)
        # Rearrange to (B, n_h, n_w, d) for window ops
        x = rearrange(x, "b d h w -> b h w d")
        x = self.norm_in(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_out(x)  # (B, n_h, n_w, d)

        # Decode back to spatial
        x = self.head(x)  # (B, n_h, n_w, C_out * p^2)
        x = rearrange(
            x,
            "b nh nw (c ph pw) -> b c (nh ph) (nw pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        # Trim padding
        x = x[:, :, :H, :W].contiguous()
        return x


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITArchesWeather(nn.Module):
    """CREDIT wrapper around ArchesWeatherBackbone."""

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(192, 288),
        patch_size=4,
        d_model=256,
        depth=8,
        num_heads=8,
        window_size=8,
        n_col_tokens=8,
        mlp_ratio=4.0,
        drop=0.0,
    ):
        super().__init__()
        self.model = ArchesWeatherBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            n_col_tokens=n_col_tokens,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        return self.model(x)

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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C_in, C_out, H, W = 2, 70, 69, 64, 128
    model = CREDITArchesWeather(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        d_model=128,
        depth=4,
        num_heads=4,
        window_size=4,
        n_col_tokens=4,
    ).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"shape mismatch: {y.shape}"
    y.mean().backward()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITArchesWeather OK — {tuple(y.shape)}, {n:.1f}M params, {device}")
