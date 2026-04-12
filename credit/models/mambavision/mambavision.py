"""
MambaVision — Hatamizadeh & Kautz, CVPR 2025 (NVlabs).
https://arxiv.org/abs/2407.08083
Original code: https://github.com/NVlabs/MambaVision   Apache-2.0

Key idea: hierarchical 4-stage backbone where early stages use residual
conv blocks (fast local feature extraction) and late stages mix
MambaVision SSM blocks with standard transformer attention blocks.

This CREDIT implementation uses a pure-PyTorch approximation of the
MambaVision SSM block (depthwise conv + gating) that does not require
the mamba_ssm CUDA package.  If `mamba_ssm` is installed the selective
SSM will be used instead.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Optionally use the real mamba_ssm selective SSM
try:
    from mamba_ssm import Mamba

    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-PyTorch SSM approximation (used when mamba_ssm is not installed)
# ---------------------------------------------------------------------------


class MambaApprox(nn.Module):
    """
    Lightweight approximation of a Mamba SSM block using depthwise conv
    and gating.  Same inductive bias (sequential mixing + gating) without
    the custom CUDA kernel.
    """

    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, L, d_model)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        # depthwise conv along sequence
        x_branch = x_branch.transpose(1, 2)  # (B, d, L)
        x_branch = self.conv1d(x_branch)[:, :, : x.shape[1]]  # causal trim
        x_branch = x_branch.transpose(1, 2)  # (B, L, d)
        x_branch = self.act(x_branch) * self.act(z)
        return self.out_proj(x_branch)


# ---------------------------------------------------------------------------
# MambaVision building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Residual depthwise-separable conv block (early stages)."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class MambaAttentionBlock(nn.Module):
    """
    Late-stage block: SSM mixing followed by multi-head self-attention.
    Operates on flattened tokens (B, L, d).
    """

    def __init__(self, d_model, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        # SSM path
        self.norm1 = nn.LayerNorm(d_model)
        if _MAMBA_AVAILABLE:
            self.ssm = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.ssm = MambaApprox(d_model=d_model, d_conv=4, expand=2)

        # Attention path
        self.norm2 = nn.LayerNorm(d_model)
        head_dim = d_model // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_proj = nn.Linear(d_model, d_model)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )
        self.num_heads = num_heads

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, N, D)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.attn_proj(self._attn(self.norm2(x)))
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Hierarchical backbone
# ---------------------------------------------------------------------------


class MambaVisionBackbone(nn.Module):
    """
    4-stage hierarchical backbone:
      Stage 0–1: residual conv blocks (local features)
      Stage 2–3: MambaAttentionBlocks (global SSM + attention)

    Between stages, a strided Conv2d downsamples by 2 and doubles channels.

    Parameters
    ----------
    in_channels, out_channels : int
    img_size : tuple[int, int]
    stem_dim : int
        Base channel count (doubles at each downsampling stage).
    stage_depths : list[int]
        Number of blocks per stage (4 stages).
    num_heads : int
        Attention heads in stages 2-3 (must divide stem_dim*4 and stem_dim*8).
    mlp_ratio : float
    drop : float
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(192, 288),
        stem_dim=64,
        stage_depths=(2, 2, 4, 2),
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.out_channels = out_channels

        # Pad to multiple of 8 (3 downsampling stages × 2)
        self.pad_h = (8 - H % 8) % 8
        self.pad_w = (8 - W % 8) % 8

        dims = [stem_dim, stem_dim * 2, stem_dim * 4, stem_dim * 8]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i, (d, depth) in enumerate(zip(dims, stage_depths)):
            if i < 2:
                stage = nn.Sequential(*[ConvBlock(d) for _ in range(depth)])
            else:
                stage = nn.Sequential(*[MambaAttentionBlock(d, num_heads, mlp_ratio, drop) for _ in range(depth)])
            self.stages.append(stage)
            if i < len(dims) - 1:
                self.downsamplers.append(nn.Conv2d(d, dims[i + 1], 2, stride=2))

        # Decoder: progressive upsampling with skip connections
        self.upsamplers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        up_dims = list(reversed(dims))  # [d8, d4, d2, d1]
        for i in range(len(up_dims) - 1):
            self.upsamplers.append(nn.ConvTranspose2d(up_dims[i], up_dims[i + 1], 2, stride=2))
            self.skip_projs.append(nn.Conv2d(up_dims[i + 1] * 2, up_dims[i + 1], 1))
            self.dec_norms.append(nn.BatchNorm2d(up_dims[i + 1]))

        # Final head
        self.head = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))

        x = self.stem(x)

        # Encode
        enc_feats = []
        for i, stage in enumerate(self.stages):
            if i >= 2:
                # MambaAttentionBlocks expect (B, L, d)
                B2, D, Hs, Ws = x.shape
                x_tok = rearrange(x, "b d h w -> b (h w) d")
                x_tok = stage(x_tok)
                x = rearrange(x_tok, "b (h w) d -> b d h w", h=Hs, w=Ws)
            else:
                x = stage(x)
            enc_feats.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        # Decode with skip connections
        enc_feats = list(reversed(enc_feats))  # deepest first
        for i, (up, skip_proj, norm) in enumerate(zip(self.upsamplers, self.skip_projs, self.dec_norms)):
            x = up(x)
            skip = enc_feats[i + 1]
            x = norm(skip_proj(torch.cat([x, skip], dim=1)))

        x = self.head(x)
        return x[:, :, :H, :W].contiguous()


# ---------------------------------------------------------------------------
# CREDIT wrapper
# ---------------------------------------------------------------------------


class CREDITMambaVision(nn.Module):
    """CREDIT wrapper around MambaVisionBackbone."""

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(192, 288),
        stem_dim=64,
        stage_depths=(2, 2, 4, 2),
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
    ):
        super().__init__()
        self.model = MambaVisionBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            stem_dim=stem_dim,
            stage_depths=stage_depths,
            num_heads=num_heads,
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
    ssm_note = "(real mamba_ssm)" if _MAMBA_AVAILABLE else "(pure-PyTorch approx)"
    B, C_in, C_out, H, W = 2, 20, 18, 32, 64
    model = CREDITMambaVision(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        stem_dim=32,
        stage_depths=(1, 1, 2, 1),
        num_heads=4,
    ).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"shape mismatch: {y.shape}"
    y.mean().backward()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITMambaVision OK {ssm_note} — {tuple(y.shape)}, {n:.1f}M params, {device}")
