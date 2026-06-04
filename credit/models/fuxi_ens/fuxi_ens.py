"""
FuXi-ENS — Chen et al., Science Advances 2025. https://arxiv.org/abs/2405.05925.
VAE ensemble perturbation head on ViT backbone.

Core idea (FuXi-ENS): add a VAE bottleneck to the latent space of a
deterministic backbone so that sampling different z vectors produces diverse
but physically plausible ensemble members.

CREDIT implementation notes
---------------------------
- The Swin-based encoder/decoder from the original paper is replaced by a
  plain ViT (same building blocks as CREDITStormer) to keep the code
  self-contained and avoid heavy timm Swin dependencies.
- use_vae=False collapses to a plain ViT identical in spirit to Stormer.
- kl_loss is stored in self.last_kl_loss after each forward pass so that
  trainers can access it without changing the forward signature.
- forward always returns (B, C_out, H, W); the caller decides whether to
  add the KL term to the training loss.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# ViT building blocks  (self-contained; not imported from stormer)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """
    2-D patch embedding via Conv2d.

    Parameters
    ----------
    img_size : tuple[int, int]
        (H, W) of the input.
    patch_size : int
        Square patch size.
    in_channels : int
        Input channel count.
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.patch_h = self.patch_w = patch_size
        self.n_patches = (self.img_h // patch_size) * (self.img_w // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)
        return rearrange(x, "b d h w -> b (h w) d")


class MLP(nn.Module):
    """Point-wise feed-forward network."""

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


class Attention(nn.Module):
    """Standard multi-head self-attention (SDPA)."""

    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual; LN → MLP → residual."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# VAE bottleneck
# ---------------------------------------------------------------------------


class VAEBottleneck(nn.Module):
    """
    Variational bottleneck that perturbs token features for ensemble generation.

    Given encoded features h of shape (B, N, latent_dim):
    1. Project to mu and logvar: Linear(latent_dim → 2 * z_dim).
    2. Reparameterisation: z = mu + eps * exp(0.5 * logvar)  (training)
                           z = mu                             (eval / deterministic)
    3. Project z back: Linear(z_dim → latent_dim).
    4. Add perturbation to h.
    5. KL loss = -0.5 * mean(1 + logvar - mu² - exp(logvar)).

    At inference, draw stochastic samples by calling ``model.train()`` briefly or
    by manually setting ``self.training = True``.  For a deterministic forecast
    use ``model.eval()``.

    Parameters
    ----------
    latent_dim : int
        Dimension of the incoming token features.
    z_dim : int
        Dimension of the latent code z.
    """

    def __init__(self, latent_dim: int, z_dim: int = 64):
        super().__init__()
        self.to_stats = nn.Linear(latent_dim, 2 * z_dim)
        self.from_z = nn.Linear(z_dim, latent_dim)

    def forward(self, h: torch.Tensor):
        """
        Parameters
        ----------
        h : torch.Tensor
            Shape (B, N, latent_dim).

        Returns
        -------
        h_perturbed : torch.Tensor
            Shape (B, N, latent_dim).
        kl_loss : torch.Tensor
            Scalar KL divergence (averaged over all elements).
        """
        stats = self.to_stats(h)  # (B, N, 2*z_dim)
        mu, logvar = stats.chunk(2, dim=-1)  # each (B, N, z_dim)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu

        kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        perturbation = self.from_z(z)  # (B, N, latent_dim)
        return h + perturbation, kl_loss


# ---------------------------------------------------------------------------
# FuXi-ENS model
# ---------------------------------------------------------------------------


class CREDITFuXiENS(nn.Module):
    """
    FuXi-ENS: VAE perturbation head on a plain ViT backbone.

    Architecture
    ------------
    ::

        (B, C_in, H, W)
          → PatchEmbed (Conv2d, patch_size)  → (B, N, embed_dim)
          → pos_embed
          → depth/2 × Block  [encoder]
          → VAEBottleneck (optional)         → (B, N, embed_dim), kl_loss
          → depth/2 × Block  [decoder]
          → LayerNorm
          → Linear(embed_dim → C_out * p²)
          → fold patches                     → (B, C_out, H, W)

    The kl_loss scalar is stored in ``self.last_kl_loss`` after every forward
    call so trainers can retrieve it without modifying the signature.

    Parameters
    ----------
    in_channels : int
        Total input channels (surface + atmos + static).
    out_channels : int
        Output channels (surface + atmos, no static).
    img_size : tuple[int, int]
        (H, W) of the lat/lon grid.
    patch_size : int
        Square ViT patch size.
    embed_dim : int
        ViT embedding dimension.
    depth : int
        Total transformer depth (split evenly between encoder and decoder).
        Must be even.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        MLP hidden ratio.
    drop_rate : float
        Dropout.
    use_vae : bool
        If True, insert a VAEBottleneck between encoder and decoder.
        If False, the model is a plain deterministic ViT.
    z_dim : int
        Latent dimension of the VAE bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 70,
        out_channels: int = 69,
        img_size: tuple = (192, 288),
        patch_size: int = 4,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        use_vae: bool = True,
        z_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_vae = use_vae

        H, W = img_size
        self.H, self.W = H, W
        self.patch_size = patch_size

        # Pad to next multiple of patch_size
        pad_H = (patch_size - H % patch_size) % patch_size
        pad_W = (patch_size - W % patch_size) % patch_size
        self.pad_H, self.pad_W = pad_H, pad_W
        pH, pW = H + pad_H, W + pad_W

        self.n_h = pH // patch_size
        self.n_w = pW // patch_size
        self.n_patches = self.n_h * self.n_w

        # Patch embedding
        self.patch_embed = PatchEmbed((pH, pW), patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Encoder blocks (first half)
        enc_depth = depth // 2
        dec_depth = depth - enc_depth
        self.encoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(enc_depth)]
        )

        # Optional VAE bottleneck
        if use_vae:
            self.vae = VAEBottleneck(latent_dim=embed_dim, z_dim=z_dim)
        else:
            self.vae = None

        # Decoder blocks (second half)
        self.decoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate) for _ in range(dec_depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Unpatchify head: each patch → (C_out * patch_size²) pixels
        self.head = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

        # KL loss from last forward (scalar, zero if use_vae=False)
        self.last_kl_loss: torch.Tensor = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C_in, H, W) or (B, C_in, T, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, C_out, H, W).

        Side effects
        ------------
        self.last_kl_loss is updated with the scalar KL divergence from the
        VAE bottleneck (0.0 when use_vae=False).
        """
        # Handle 5-D trainer input
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)

        B = x.shape[0]

        # Spatial padding
        if self.pad_H > 0 or self.pad_W > 0:
            x = F.pad(x, (0, self.pad_W, 0, self.pad_H))

        # Patch embed + positional encoding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        # Encoder
        for blk in self.encoder_blocks:
            x = blk(x)

        # VAE bottleneck
        if self.vae is not None:
            x, kl_loss = self.vae(x)
            self.last_kl_loss = kl_loss
        else:
            self.last_kl_loss = torch.tensor(0.0, device=x.device)

        # Decoder
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.norm(x)  # (B, N, embed_dim)

        # Unpatchify
        x = self.head(x)  # (B, N, C_out * p²)
        x = rearrange(
            x,
            "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
            nh=self.n_h,
            nw=self.n_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )  # (B, C_out, pH, pW)

        # Remove padding
        if self.pad_H > 0 or self.pad_W > 0:
            x = x[:, :, : self.H, : self.W]

        return x  # (B, C_out, H, W)

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

    # ---- VAE mode ----
    model_vae = CREDITFuXiENS(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        embed_dim=256,  # small for smoke test
        depth=4,
        num_heads=8,
        use_vae=True,
        z_dim=32,
    ).to(device)

    x = torch.randn(B, C_in, H, W, device=device)
    model_vae.train()
    y = model_vae(x)
    assert y.shape == (B, C_out, H, W), f"VAE train: unexpected shape {y.shape}"
    kl = model_vae.last_kl_loss
    loss = y.mean() + 1e-4 * kl
    loss.backward()
    print(f"VAE train mode  — output {y.shape}, KL {kl.item():.4f}")

    model_vae.eval()
    with torch.no_grad():
        y_eval = model_vae(x)
    assert y_eval.shape == (B, C_out, H, W)
    print(f"VAE eval  mode  — output {y_eval.shape}")

    # ---- deterministic mode ----
    model_det = CREDITFuXiENS(
        in_channels=C_in,
        out_channels=C_out,
        img_size=(H, W),
        patch_size=4,
        embed_dim=256,
        depth=4,
        num_heads=8,
        use_vae=False,
    ).to(device)

    y_det = model_det(x)
    assert y_det.shape == (B, C_out, H, W), f"Det: unexpected shape {y_det.shape}"
    print(f"Deterministic   — output {y_det.shape}")

    # ---- 5-D trainer input ----
    x5 = torch.randn(B, C_in, 1, H, W, device=device)
    y5 = model_det(x5)
    assert y5.shape == (B, C_out, H, W), f"5-D: unexpected shape {y5.shape}"
    print(f"5-D input       — output {y5.shape}")

    n_params_vae = sum(p.numel() for p in model_vae.parameters()) / 1e6
    n_params_det = sum(p.numel() for p in model_det.parameters()) / 1e6
    print(f"Params: VAE={n_params_vae:.1f}M  det={n_params_det:.1f}M  device={device}")
