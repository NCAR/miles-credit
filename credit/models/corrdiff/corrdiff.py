"""
CorrDiff — NVIDIA, 2024.
https://arxiv.org/abs/2309.15214   Apache-2.0
Original code: https://github.com/NVIDIA/physicsnemo  (examples/weather/corrdiff)

Score-based conditional diffusion model for statistical downscaling.
Generates high-resolution output conditioned on a coarse-resolution input.

Architecture:
  - Conditioning encoder: U-Net encoder on coarse input → multi-scale feature maps
  - Score network: SongUNet denoising backbone conditioned on coarse features and noise level
  - EDM noise schedule (Karras et al. 2022): sigma-based preconditioning

For global forecasting use-cases, CorrDiff can also be used as a
probabilistic super-resolution head: coarse = global model output,
fine = analysis or high-res reanalysis.

Smoke-test mode: forward() runs one denoising step at a fixed sigma.
Full sampling requires calling sample() with a noise schedule.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EDM preconditioning (Karras et al. 2022, eq. 7)
# ---------------------------------------------------------------------------


def edm_precond(sigma, sigma_data=0.5):
    """Return (c_skip, c_out, c_in, c_noise) for EDM preconditioning."""
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1.0 / (sigma**2 + sigma_data**2) ** 0.5
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


# ---------------------------------------------------------------------------
# Noise embedding
# ---------------------------------------------------------------------------


class FourierEmbedding(nn.Module):
    """Sinusoidal / Fourier embedding for the noise level log(sigma)."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(dim // 2))

    def forward(self, log_sigma):
        # log_sigma: (B,) scalar
        x = log_sigma.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        return torch.cat([x.sin(), x.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with optional noise-level conditioning."""

    def __init__(self, in_ch, out_ch, emb_dim=None, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if emb_dim is not None:
            self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        else:
            self.emb_proj = None

    def forward(self, x, emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if emb is not None and self.emb_proj is not None:
            scale, shift = self.emb_proj(self.act(emb)).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class AttnBlock(nn.Module):
    """Single-head self-attention on 2D feature map."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        scale = C**-0.5
        attn = (q.transpose(-1, -2) @ k * scale).softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Conditioning encoder (coarse → multi-scale feature maps)
# ---------------------------------------------------------------------------


class CondEncoder(nn.Module):
    """
    Lightweight U-Net encoder that extracts multi-scale conditioning features
    from the coarse-resolution input.
    """

    def __init__(self, in_ch, base_ch=64, n_levels=3):
        super().__init__()
        self.levels = n_levels
        chs = [base_ch * (2**i) for i in range(n_levels)]

        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch_prev = chs[0]
        for ch in chs:
            self.enc_blocks.append(ResBlock(ch_prev, ch))
            self.downsamplers.append(nn.Conv2d(ch, ch, 2, stride=2))
            ch_prev = ch
        self.out_channels = chs  # list of output channel counts per level

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for enc, down in zip(self.enc_blocks, self.downsamplers):
            x = enc(x)
            feats.append(x)
            x = down(x)
        return feats  # list of (B, ch, H/2^i, W/2^i)


# ---------------------------------------------------------------------------
# SongUNet score network
# ---------------------------------------------------------------------------


class SongUNet(nn.Module):
    """
    U-Net score network for EDM-style diffusion.
    Conditioned on noise level (via Fourier embedding) and coarse features
    (injected at matching resolutions via channel concatenation).

    Parameters
    ----------
    in_ch : int
        Channels of the noisy high-res input.
    out_ch : int
        Channels of the denoised prediction.
    cond_ch : int
        Channels of the coarse conditioning input.
    base_ch : int
        Base channel count for the U-Net.
    ch_mult : list[int]
        Channel multipliers per resolution level.
    n_res_per_level : int
        ResBlocks per level.
    emb_dim : int
        Noise embedding dimension.
    attn_levels : list[int]
        Levels (0-indexed, from top) where attention is applied.
    """

    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        cond_ch=4,
        base_ch=64,
        ch_mult=(1, 2, 4),
        n_res_per_level=2,
        emb_dim=256,
        attn_levels=(2,),
    ):
        super().__init__()
        n_levels = len(ch_mult)
        chs = [base_ch * m for m in ch_mult]

        # Noise embedding
        self.fourier = FourierEmbedding(emb_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # Conditioning encoder
        self.cond_enc = CondEncoder(cond_ch, base_ch=base_ch, n_levels=n_levels)
        cond_chs = self.cond_enc.out_channels  # channels injected at each level

        # Encoder
        self.enc_stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        prev_ch = chs[0]
        for i, ch in enumerate(chs):
            in_c = prev_ch + (cond_chs[i] if i > 0 else 0)
            blks = nn.ModuleList([ResBlock(in_c if j == 0 else ch, ch, emb_dim) for j in range(n_res_per_level)])
            attn = AttnBlock(ch) if i in attn_levels else nn.Identity()
            self.enc_blocks.append(blks)
            self.enc_attn.append(attn)
            if i < n_levels - 1:
                self.enc_down.append(nn.Conv2d(ch, ch, 2, stride=2))
            prev_ch = ch

        # Bottleneck
        self.mid_block1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid_attn = AttnBlock(chs[-1])
        self.mid_block2 = ResBlock(chs[-1], chs[-1], emb_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i, ch in enumerate(rev_chs):
            skip_ch = chs[n_levels - 1 - i]
            # h entering this level has rev_chs[i-1] channels (from prev dec level or bottleneck)
            h_ch = chs[-1] if i == 0 else rev_chs[i - 1]
            in_c = h_ch + skip_ch + (cond_chs[n_levels - 1 - i] if i > 0 else 0)
            blks = nn.ModuleList([ResBlock(in_c if j == 0 else ch, ch, emb_dim) for j in range(n_res_per_level)])
            attn = AttnBlock(ch) if (n_levels - 1 - i) in attn_levels else nn.Identity()
            self.dec_blocks.append(blks)
            self.dec_attn.append(attn)
            if i < n_levels - 1:
                self.dec_up.append(nn.ConvTranspose2d(ch, ch, 2, stride=2))

        self.out_norm = nn.GroupNorm(min(8, rev_chs[-1]), rev_chs[-1])
        self.out_conv = nn.Conv2d(rev_chs[-1], out_ch, 3, padding=1)

    def forward(self, x_noisy, sigma, x_cond):
        """
        Parameters
        ----------
        x_noisy : (B, in_ch, H, W)   noisy high-res target
        sigma   : (B,)               noise standard deviation
        x_cond  : (B, cond_ch, H_c, W_c)  coarse-res conditioning

        Returns
        -------
        (B, out_ch, H, W)  denoised prediction (before EDM skip)
        """
        # Noise embedding
        emb = self.emb_mlp(self.fourier(sigma.log()))  # (B, emb_dim)

        # Upsample conditioning to match x_noisy resolution if needed
        if x_cond.shape[-2:] != x_noisy.shape[-2:]:
            x_cond = F.interpolate(x_cond, size=x_noisy.shape[-2:], mode="bilinear", align_corners=False)

        # Get conditioning features at multiple scales
        cond_feats = self.cond_enc(x_cond)  # list of tensors per level

        # Encoder
        h = self.enc_stem(x_noisy)
        skips = [h]
        for i, (blks, attn) in enumerate(zip(self.enc_blocks, self.enc_attn)):
            if i > 0:
                # Upsample/downsample cond feat to match current h spatial size
                cf = F.interpolate(cond_feats[i], size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, cf], dim=1)
            for blk in blks:
                h = blk(h, emb)
            h = attn(h)
            skips.append(h)
            if i < len(self.enc_down):
                h = self.enc_down[i](h)

        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder
        for i, (blks, attn) in enumerate(zip(self.dec_blocks, self.dec_attn)):
            skip = skips[-(i + 1)]
            if i > 0 and i < len(self.dec_up) + 1:
                cf = F.interpolate(cond_feats[-(i + 1)], size=h.shape[-2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, skip, cf], dim=1)
            else:
                h = torch.cat([h, skip], dim=1)
            for blk in blks:
                h = blk(h, emb)
            h = attn(h)
            if i < len(self.dec_up):
                h = self.dec_up[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# CREDIT wrapper (EDM preconditioning)
# ---------------------------------------------------------------------------


class CREDITCorrDiff(nn.Module):
    """
    CREDIT wrapper for CorrDiff score-based downscaling.

    In downscaling mode:
      - ``x_cond`` is the coarse-resolution forecast (channels = in_channels)
      - The model generates the fine-resolution correction/output

    For global single-resolution use (e.g., probabilistic noise injection):
      - coarse and fine resolution are the same; set scale_factor=1

    Parameters
    ----------
    in_channels : int
        Coarse conditioning channels.
    out_channels : int
        Fine output channels.
    img_size : tuple[int, int]
        Fine-resolution (H, W).
    scale_factor : int
        Upsampling factor from coarse to fine (1 = same resolution).
    base_ch : int
    ch_mult : list[int]
    n_res_per_level : int
    emb_dim : int
    sigma_data : float
        EDM sigma_data parameter (controls skip weighting).
    """

    def __init__(
        self,
        in_channels=70,
        out_channels=69,
        img_size=(192, 288),
        scale_factor=1,
        base_ch=64,
        ch_mult=(1, 2, 4),
        n_res_per_level=2,
        emb_dim=256,
        sigma_data=0.5,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.out_channels = out_channels
        H, W = img_size
        self.H, self.W = H, W

        self.score_net = SongUNet(
            in_ch=out_channels,
            out_ch=out_channels,
            cond_ch=in_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
            n_res_per_level=n_res_per_level,
            emb_dim=emb_dim,
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
        """
        Single denoising step (for smoke-testing / training).

        Parameters
        ----------
        x : (B, in_channels, H, W)  coarse input (conditioning)
        sigma : (B,) noise level; defaults to sigma=1.0 if not provided

        Returns
        -------
        (B, out_channels, H, W)  denoised prediction
        """
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)

        B = x.shape[0]
        if sigma is None:
            sigma = torch.ones(B, device=x.device)

        # Sample Gaussian noise to denoise
        x_noisy = torch.randn(B, self.out_channels, self.H, self.W, device=x.device)

        # EDM preconditioning
        c_skip, c_out, c_in, c_noise = edm_precond(sigma.float(), self.sigma_data)

        x_in = c_in[:, None, None, None] * x_noisy
        F_x = self.score_net(x_in, sigma, x)
        D_x = c_skip[:, None, None, None] * x_noisy + c_out[:, None, None, None] * F_x

        # Resize to output if needed
        if D_x.shape[-2:] != (self.H, self.W):
            D_x = F.interpolate(D_x, size=(self.H, self.W), mode="bilinear", align_corners=False)
        return D_x

    @torch.no_grad()
    def sample(
        self,
        x_cond: torch.Tensor,
        n_steps: int = 18,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """
        Full EDM deterministic sampler (Heun's 2nd order method).

        Parameters
        ----------
        x_cond : (B, in_channels, H, W)
        n_steps : int  number of denoising steps
        sigma_max, sigma_min, rho : EDM schedule parameters

        Returns
        -------
        (B, out_channels, H, W)
        """
        B = x_cond.shape[0]
        device = x_cond.device

        # EDM schedule
        steps = torch.arange(n_steps + 1, device=device)
        sigmas = (sigma_max ** (1 / rho) + steps / n_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas[-1] = 0.0

        x = torch.randn(B, self.out_channels, self.H, self.W, device=device) * sigmas[0]

        for i in range(n_steps):
            sigma_i = sigmas[i].expand(B)
            c_skip, c_out, c_in, _ = edm_precond(sigma_i, self.sigma_data)
            F_x = self.score_net(c_in[:, None, None, None] * x, sigma_i, x_cond)
            D_x = c_skip[:, None, None, None] * x + c_out[:, None, None, None] * F_x
            d = (x - D_x) / sigmas[i]

            if sigmas[i + 1] == 0:
                x = x + d * (sigmas[i + 1] - sigmas[i])
            else:
                # Heun step
                x2 = x + d * (sigmas[i + 1] - sigmas[i])
                sigma_i2 = sigmas[i + 1].expand(B)
                c_skip2, c_out2, c_in2, _ = edm_precond(sigma_i2, self.sigma_data)
                F_x2 = self.score_net(c_in2[:, None, None, None] * x2, sigma_i2, x_cond)
                D_x2 = c_skip2[:, None, None, None] * x2 + c_out2[:, None, None, None] * F_x2
                d2 = (x2 - D_x2) / sigmas[i + 1]
                x = x + 0.5 * (d + d2) * (sigmas[i + 1] - sigmas[i])

        return x

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
    B, C_in, C_out, H, W = 2, 20, 18, 32, 64
    model = CREDITCorrDiff(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        base_ch=32,
        ch_mult=(1, 2),
        n_res_per_level=1,
        emb_dim=64,
    ).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    y = model(x)
    assert y.shape == (B, C_out, H, W), f"shape mismatch: {y.shape}"
    y.mean().backward()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITCorrDiff OK — {tuple(y.shape)}, {n:.1f}M params, {device}")
