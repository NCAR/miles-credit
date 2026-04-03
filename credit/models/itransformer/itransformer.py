"""
iTransformer — Liu et al., ICLR 2024. https://arxiv.org/abs/2310.06625

Key idea: instead of attention across spatial tokens, each channel/variable
gets ONE token (its full spatial field projected to a single embedding vector),
then attention is applied ACROSS VARIABLES. This directly models cross-variable
interactions at every layer, with memory that scales as O(C²) rather than O(N²)
in the spatial sequence length.

Architecture:
    (B, C_in, H, W)
        → flatten spatial: (B, C_in, H*W)
        → shared Linear(H*W → d_model): (B, C_in, d_model)   # one token per variable
        → N × iTransformerBlock (attention over C_in tokens)
        → shared Linear(d_model → H*W): (B, C_in, H*W)
        → Conv2d(C_in, C_out, 1): channel mix → (B, C_out, H, W)
"""

import os
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class VariateAttention(nn.Module):
    """
    Multi-head self-attention applied over the variate (channel) dimension.

    Input:  (B, C, d_model)  — C tokens of dimension d_model
    Output: (B, C, d_model)

    Identical to standard SDPA; the only difference is that N = C (variables)
    rather than the number of spatial patches.

    Parameters
    ----------
    dim : int
        Token dimension (d_model).
    num_heads : int
        Number of attention heads. Must divide dim evenly.
    attn_drop : float
        Dropout applied to attention weights.
    proj_drop : float
        Dropout applied after the output projection.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape  # C == number of variates (tokens)
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, C, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, C, C)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, D)  # (B, C, D)
        return self.proj_drop(self.proj(x))


class VariateMLP(nn.Module):
    """
    Feed-forward network applied identically to every variate token.

    Shared weights across variates — analogous to a pointwise MLP in a
    standard transformer (no information exchange between variates here;
    that happens in VariateAttention).

    Parameters
    ----------
    dim : int
        Token dimension.
    mlp_ratio : float
        Hidden-to-input width ratio.
    drop : float
        Dropout after each linear.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class iTransformerBlock(nn.Module):
    """
    Pre-norm transformer block over variate tokens.

    Residual stream:
        x = x + VariateAttention(LayerNorm(x))
        x = x + VariateMLP(LayerNorm(x))

    Parameters
    ----------
    dim : int
        Token dimension (d_model).
    num_heads : int
        Attention heads.
    mlp_ratio : float
        MLP hidden ratio.
    drop : float
        Dropout (attention + MLP).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VariateAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = VariateMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# iTransformer backbone
# ---------------------------------------------------------------------------


class iTransformerBackbone(nn.Module):
    """
    iTransformer backbone (Liu et al., ICLR 2024).

    Applies self-attention across the variate (channel) dimension rather than
    the spatial dimension, so memory scales as O(C²) instead of O(N²).

    Input:  (B, in_channels, H, W)
    Output: (B, out_channels, H, W)

    Processing pipeline
    -------------------
    1. Flatten spatial:  (B, C_in, H, W) → (B, C_in, H*W)
    2. Input projection (shared): Linear(H*W → d_model) → (B, C_in, d_model)
    3. N × iTransformerBlock (attention over C_in variate tokens)
    4. Output projection (shared): Linear(d_model → H*W) → (B, C_in, H*W)
    5. Reshape: → (B, C_in, H, W)
    6. Channel mix: Conv2d(C_in, C_out, kernel=1) → (B, C_out, H, W)

    Parameters
    ----------
    in_channels : int
        Number of input channels (variables).
    out_channels : int
        Number of output channels.
    img_size : tuple[int, int]
        (H, W) spatial dimensions.
    d_model : int
        Per-variate embedding dimension.
    depth : int
        Number of iTransformerBlocks.
    num_heads : int
        Attention heads (must divide d_model).
    mlp_ratio : float
        MLP hidden-to-d_model ratio.
    drop : float
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 69,
        img_size: tuple = (192, 288),
        d_model: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.d_model = d_model

        H, W = img_size
        self.spatial_size = H * W

        # Step 1 — project each variable's spatial field to a d_model vector
        self.input_proj = nn.Linear(self.spatial_size, d_model)

        # Step 2 — variate-attention transformer blocks
        self.blocks = nn.ModuleList(
            [iTransformerBlock(d_model, num_heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Step 3 — decode each variate token back to spatial field
        self.output_proj = nn.Linear(d_model, self.spatial_size)

        # Step 4 — mix C_in decoded maps to C_out output maps (pointwise)
        self.channel_mix = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, out_channels, H, W).
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input spatial size ({H}, {W}) does not match model img_size {self.img_size}."
        )

        # (B, C, H, W) → (B, C, H*W)
        x = x.reshape(B, C, H * W)

        # Project each variable to d_model: (B, C, d_model)
        x = self.input_proj(x)

        # Variate-attention blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, C, d_model)

        # Decode each variate back to spatial: (B, C, H*W)
        x = self.output_proj(x)

        # Reshape to spatial: (B, C_in, H, W)
        x = x.reshape(B, C, H, W)

        # Mix channels: (B, C_out, H, W)
        x = self.channel_mix(x)
        return x


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITiTransformer(nn.Module):
    """
    Thin CREDIT wrapper around iTransformerBackbone.

    Accepts (B, C, T, H, W) 5-D tensors from the CREDIT trainer (squeezes
    the T dimension), and returns (B, C_out, H, W).

    Parameters
    ----------
    in_channels : int
        Total input channels (surface + atmospheric + static).
    out_channels : int
        Output channels (surface + atmospheric, no static).
    img_size : tuple[int, int]
        (H, W). Default (192, 288).
    d_model : int
        Per-variate embedding dimension. Default 256.
    depth : int
        Number of iTransformerBlocks. Default 6.
    num_heads : int
        Attention heads. Default 8.
    mlp_ratio : float
        MLP hidden ratio. Default 4.0.
    drop : float
        Dropout. Default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 69,
        img_size: tuple = (192, 288),
        d_model: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.model = iTransformerBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            # (B, C, T, H, W) — squeeze T (take first time step or reshape)
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
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, _root)

    B, C_in, C_out, H, W = 2, 70, 69, 192, 288
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITiTransformer(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        d_model=128,  # small for smoke test
        depth=3,
        num_heads=4,
    ).to(device)

    # 4-D input
    x4 = torch.randn(B, C_in, H, W, device=device)
    y4 = model(x4)
    assert y4.shape == (B, C_out, H, W), f"4-D: unexpected shape {y4.shape}"

    # 5-D input (trainer format)
    x5 = torch.randn(B, C_in, 1, H, W, device=device)
    y5 = model(x5)
    assert y5.shape == (B, C_out, H, W), f"5-D: unexpected shape {y5.shape}"

    loss = y4.mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITiTransformer OK — output {y4.shape}, params {n_params:.1f}M, device {device}")
