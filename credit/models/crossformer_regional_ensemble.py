from credit.models.crossformer_regional_invertable import RegionalCrossFormerInvertable
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
from credit.models.stochastic_decomposition_layer import StochasticDecompositionLayer

class RegionalCrossFormerWithNoise(RegionalCrossFormerInvertable):
    """
    CrossFormer variant with pixel-wise noise injection in both encoder and decoder stages.

    Attributes:
        noise_latent_dim (int): Dimensionality of the noise vector.
        encoder_noise_factor (float): Initial scaling factor for encoder noise injection.
        decoder_noise_factor (float): Initial scaling factor for decoder noise injection.
        encoder_noise (bool): Whether to apply noise injection in the encoder.
        freeze (bool): Whether to freeze pre-trained model weights.
        correlated (bool): Whether to use the same latent vector for all injection layers.
    """

    def __init__(
        self,
        noise_latent_dim=128,
        encoder_noise_factor=0.235,
        decoder_noise_factor=0.235,
        encoder_noise=False, # Set False by default to match paper's decoder-only focus
        freeze=False,        # False for joint fine-tuning transfer learning
        correlated=False,    # Added correlated toggle
        enable_sdl=True,
        **kwargs,
    ):
        """
        Initializes the RegionalCrossFormerWithNoise model.
        """
        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim
        self.enable_sdl = enable_sdl
        self.correlated = correlated # Save the flag

        # Extract the channel dimensions from the config (defaults to standard sizes)
        dims = kwargs.get('dim', [32, 64, 128, 256])

        # Freeze weights if using pre-trained model
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.encoder_noise = encoder_noise
        
        # If SDL is disabled, force noise factors to 0.0
        if not self.enable_sdl:
            encoder_noise_factor = 0.0
            decoder_noise_factor = 0.0

        if encoder_noise:
            # Encoder noise injection layers
            self.encoder_noise_layers = nn.ModuleList(
                [
                    StochasticDecompositionLayer(self.noise_latent_dim, dims[0], encoder_noise_factor),
                    StochasticDecompositionLayer(self.noise_latent_dim, dims[1], encoder_noise_factor),
                    StochasticDecompositionLayer(self.noise_latent_dim, dims[2], encoder_noise_factor),
                ]
            )

        # Decoder noise injection layers (reverse order: 2, 1, 0)
        self.noise_inject1 = StochasticDecompositionLayer(self.noise_latent_dim, dims[2], decoder_noise_factor)
        self.noise_inject2 = StochasticDecompositionLayer(self.noise_latent_dim, dims[1], decoder_noise_factor)
        self.noise_inject3 = StochasticDecompositionLayer(self.noise_latent_dim, dims[0], decoder_noise_factor)

        # --- THE FIX: FREEZE NOISE LAYERS IF NOT USING THEM ---
        # If SDL is disabled, freeze all noise parameters so DDP ignores them
        if not self.enable_sdl:
            for name, param in self.named_parameters():
                if "noise" in name or "modulation" in name:
                    param.requires_grad = False
        # ------------------------------------------------------

    def forward(self, x, x_era5, forcing_t_delta, noise=None, forecast_step=None):
        """
        Forward pass through the CrossFormer with noise injection.
        Note: The forward pass remains identical regardless of enable_sdl.
        If enable_sdl=False, the injected noise is scaled by 0.0.
        """

        batch_size = x.shape[0]

        # If correlated, sample ONCE for the whole network
        if self.correlated:
            shared_noise = torch.randn(batch_size, self.noise_latent_dim, device=x.device)

        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)


        # Feature-wise Linear Modulation for time embedding
        alpha_beta = self.film(forcing_t_delta.view(-1, 1).expand(batch_size, 1) / 3600.)  # [batch, 2*dim]
        alpha, beta = alpha_beta.chunk(2, dim=1)  # each is [batch, dim]
        alpha = alpha.view(batch_size, self.input_only_channels, 1, 1, 1)  # [batch, dim, 1, 1, 1]
        beta = beta.view(batch_size, self.input_only_channels, 1, 1, 1)  # [batch, dim, 1, 1, 1]
        x_era5 = alpha * x_era5 + beta

        x = torch.concat([x, x_era5], dim=1)

        if self.pad_all:
            x = self.padding_all.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:
            x = x.squeeze(2)

        encodings = []
        for k, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if self.encoder_noise and k < len(self.layers) - 1:
                # Use shared_noise if correlated, else sample fresh
                current_noise = shared_noise if self.correlated else torch.randn(batch_size, self.noise_latent_dim, device=x.device)
                x = self.encoder_noise_layers[k](x, current_noise)
            encodings.append(x)

        x = self.up_block1(x)
        # Use shared_noise if correlated, else sample fresh
        current_noise = shared_noise if self.correlated else torch.randn(batch_size, self.noise_latent_dim, device=x.device)
        x = self.noise_inject1(x, current_noise)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        # Use shared_noise if correlated, else sample fresh
        current_noise = shared_noise if self.correlated else torch.randn(batch_size, self.noise_latent_dim, device=x.device)
        x = self.noise_inject2(x, current_noise)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        # Use shared_noise if correlated, else sample fresh
        current_noise = shared_noise if self.correlated else torch.randn(batch_size, self.noise_latent_dim, device=x.device)
        x = self.noise_inject3(x, current_noise)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.pad_all:
            x = self.padding_all.unpad(x)
        if self.use_padding:
            x = self.padding_opt.unpad(x)

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x

if __name__ == "__main__":
    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    crossformer_config = {
        "type": "crossformer",
        "frames": 1,
        "image_height": 192,
        "image_width": 288,
        "levels": 16,
        "channels": 4,
        "surface_channels": 7,
        "input_only_channels": 3,
        "output_only_channels": 0,
        "patch_width": 1,
        "patch_height": 1,
        "frame_patch_size": 1,
        "dim": [32, 64, 128, 256],
        "depth": [2, 2, 2, 2],
        "global_window_size": [8, 4, 2, 1],
        "local_window_size": 8,
        "cross_embed_kernel_sizes": [
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "use_spectral_norm": True,
        "padding_conf": {
            "activate": True,
            "mode": "regional",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
        "padding_all_conf": {
            "activate": False,
        },
    }

    crossformer_config["noise_latent_dim"] = 128
    crossformer_config["encoder_noise_factor"] = 0.235
    crossformer_config["decoder_noise_factor"] = 0.235
    crossformer_config["encoder_noise"] = False # Defaulting to paper implementation
    crossformer_config["freeze"] = False  
    crossformer_config["correlated"] = False  # Added flag to test block

    logger.info("Testing the regional ensemble model with noise injection")

    ensemble_model = RegionalCrossFormerWithNoise(**crossformer_config).to("cuda")

    x = torch.randn(5, 71, 1, 192, 288).to("cuda")  
    x_era5 = torch.randn(5, 3, 1, 192, 288).to("cuda")  
    forcing_t_delta = torch.randn(5).to("cuda")  

    output = ensemble_model(x, x_era5, forcing_t_delta)
    print("Output shape:", output.shape)
    variance = torch.var(output)
    print("Variance:", variance.item())