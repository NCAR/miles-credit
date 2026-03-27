"""
WXFormer v2 Ensemble — CrossFormer v2 with SDL noise injection.

Workflow
--------
1. Train a deterministic CrossFormer v2 checkpoint normally.
2. Load it here with `pretrained_weights` in the config.
   The base weights are loaded with strict=False (SDL keys absent → ignored).
3. Freeze the base weights (freeze=True, default).
4. Fine-tune only the SDL layers to learn calibrated spread.

The SDL layers attach at every decoder upsampling stage.  The base forward()
in wxformer_v2.py is overridden to inject noise after each UpBlock.
"""

import logging

import torch
import torch.nn as nn

from credit.models.wxformer.wxformer_v2 import CrossFormer
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer

logger = logging.getLogger(__name__)


class CrossFormerV2WithNoise(CrossFormer):
    """CrossFormer v2 with per-scale SDL noise injection in encoder + decoder."""

    def __init__(
        self,
        noise_latent_dim: int = 128,
        encoder_noise_factor: float = 0.05,
        decoder_noise_factor: float = 0.275,
        encoder_noise: bool = True,
        freeze: bool = True,
        correlated: bool = False,
        pretrained_weights: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert hasattr(self, "dec_dims"), (
            "dec_dims not found — make sure upsample_with_ps=True or "
            "upsample_with_transformer=False so the conv decoder is used."
        )

        self.noise_latent_dim = noise_latent_dim
        self.correlated = correlated
        self.encoder_noise = encoder_noise

        # ── Load pretrained base weights (SDL keys absent → strict=False) ────
        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location="cpu")
            # support both raw state_dict and checkpoint dicts
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.load_state_dict(state, strict=False)
            sdl_missing = [k for k in missing if "noise" not in k and "sdl" not in k.lower()]
            if sdl_missing:
                logger.warning("Unexpected missing keys (non-SDL): %s", sdl_missing)
            logger.info("Loaded pretrained weights from %s", pretrained_weights)

        # ── Freeze base model parameters ──────────────────────────────────────
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("Base model weights frozen.")

        # ── Encoder SDL layers ────────────────────────────────────────────────
        # One layer per intermediate encoder level (all but the bottleneck).
        if encoder_noise:
            enc_channels = [self.dec_dims[self.num_levels - 1 - i] for i in range(self.num_levels - 1)]
            self.encoder_noise_layers = nn.ModuleList(
                [
                    StochasticDecompositionLayer(noise_latent_dim, ch, noise_factor=encoder_noise_factor)
                    for ch in enc_channels
                ]
            )

        # ── Decoder SDL layers ────────────────────────────────────────────────
        # One layer per UpBlock (num_levels - 1 blocks total).
        # dec_dims[i+1] is the output channel count of up_blocks[i].
        dec_sdl_channels = [self.dec_dims[i + 1] for i in range(self.num_levels - 1)]
        self.decoder_noise_layers = nn.ModuleList(
            [
                StochasticDecompositionLayer(noise_latent_dim, ch, noise_factor=decoder_noise_factor)
                for ch in dec_sdl_channels
            ]
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x, noise=None, forecast_step=None):
        import torch.nn.functional as F

        if self.correlated and noise is None:
            noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)

        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:
            x = x.squeeze(2)

        # ── Encoder ──────────────────────────────────────────────────────────
        encodings = []
        for k, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if self.encoder_noise and k < self.num_levels - 1:
                if not self.correlated:
                    noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
                x = self.encoder_noise_layers[k](x, noise)
            encodings.append(x)

        # ── Decoder (mirrors wxformer_v2.py conv decoder) ────────────────────
        x = self.up_blocks[0](x)
        if not self.correlated:
            noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
        x = self.decoder_noise_layers[0](x, noise)

        for i in range(1, self.num_levels):
            x = torch.cat([x, encodings[self.num_levels - 1 - i]], dim=1)
            if i < self.num_levels - 1:
                x = self.up_blocks[i](x)
                if not self.correlated:
                    noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
                x = self.decoder_noise_layers[i](x, noise)
            else:
                x = self.up_block_out(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear", align_corners=False)

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = self.postblock({"y_pred": x, "x": x_copy})

        return x


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CrossFormerV2WithNoise(
        # base model — minimal size for quick test
        image_height=640,
        image_width=1280,
        frames=1,
        channels=4,
        surface_channels=7,
        input_only_channels=3,
        levels=15,
        dim=(64, 128, 256, 512),
        depth=(2, 2, 8, 2),
        local_window_size=10,
        global_window_size=(5, 5, 2, 1),
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(4, 2, 2, 2),
        # SDL
        noise_latent_dim=128,
        encoder_noise=True,
        freeze=False,  # no pretrained weights in smoke test
        pretrained_weights=None,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total params: %.2fM  |  Trainable: %.2fM", total / 1e6, trainable / 1e6)

    B = 2
    x = torch.randn(B, model.input_channels, 1, 640, 1280, device=device)
    y = model(x)
    logger.info("Output shape: %s", y.shape)

    # verify spread across batch (different noise per sample)
    assert y.shape == (B, model.output_channels, 1, 640, 1280)
    assert not torch.allclose(y[0], y[1], atol=1e-4), "No spread — SDL not firing"
    logger.info("Spread check passed.")
