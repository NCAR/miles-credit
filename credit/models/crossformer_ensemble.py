from credit.models.crossformer import CrossFormer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch


class CrossFormerWithNoise(CrossFormer):
    """
    CrossFormer variant with pixel-wise noise injection in both encoder and decoder stages.

    Attributes:
        noise_latent_dim (int): Dimensionality of the noise vector.
        encoder_noise_factor (float): Initial scaling factor for encoder noise injection.
        decoder_noise_factor (float): Initial scaling factor for decoder noise injection.
        encoder_noise (bool): Whether to apply noise injection in the encoder.
        freeze (bool): Whether to freeze pre-trained model weights.
    """

    def __init__(
        self,
        noise_latent_dim=128,
        encoder_noise_factor=0.05,
        decoder_noise_factor=0.275,
        encoder_noise=True,
        freeze=True,
        **kwargs,
    ):
        """
        Initializes the CrossFormerWithNoise model.

        Args:
            noise_latent_dim (int, optional): Dimensionality of the noise latent space. Defaults to 128.
            encoder_noise_factor (float, optional): Scaling factor for encoder noise injection. Defaults to 0.05.
            decoder_noise_factor (float, optional): Scaling factor for decoder noise injection. Defaults to 0.275.
            encoder_noise (bool, optional): Whether to inject noise into encoder layers. Defaults to True.
            freeze (bool, optional): Whether to freeze pre-trained model weights. Defaults to True.
            **kwargs: Additional arguments passed to the CrossFormer base class.
        """

        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim

        # Freeze weights if using pre-trained model
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.encoder_noise = encoder_noise
        if encoder_noise:
            # Define separate learnable noise factors for encoder noise layers
            encoder_noise_factors = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(encoder_noise_factor, dtype=torch.float32)
                    ),
                    nn.Parameter(
                        torch.tensor(encoder_noise_factor, dtype=torch.float32)
                    ),
                    nn.Parameter(
                        torch.tensor(encoder_noise_factor, dtype=torch.float32)
                    ),
                ]
            )
            # Encoder noise injection layers
            self.encoder_noise_layers = nn.ModuleList(
                [
                    PixelNoiseInjection(
                        self.noise_latent_dim,
                        self.up_block3.output_channels,
                        encoder_noise_factors[0],
                    ),
                    PixelNoiseInjection(
                        self.noise_latent_dim,
                        self.up_block2.output_channels,
                        encoder_noise_factors[1],
                    ),
                    PixelNoiseInjection(
                        self.noise_latent_dim,
                        self.up_block1.output_channels,
                        encoder_noise_factors[2],
                    ),
                ]
            )

        # Define separate learnable noise factors for decoder noise layers
        decoder_noise_factors = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(decoder_noise_factor, dtype=torch.float32)),
                nn.Parameter(torch.tensor(decoder_noise_factor, dtype=torch.float32)),
                nn.Parameter(torch.tensor(decoder_noise_factor, dtype=torch.float32)),
            ]
        )

        # Decoder noise injection layers
        self.noise_inject1 = PixelNoiseInjection(
            self.noise_latent_dim,
            self.up_block1.output_channels,
            decoder_noise_factors[0],
        )
        self.noise_inject2 = PixelNoiseInjection(
            self.noise_latent_dim,
            self.up_block2.output_channels,
            decoder_noise_factors[1],
        )
        self.noise_inject3 = PixelNoiseInjection(
            self.noise_latent_dim,
            self.up_block3.output_channels,
            decoder_noise_factors[2],
        )

    def forward(self, x, noise=None, forecast_step=None):
        """
        Forward pass through the CrossFormer with noise injection.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
            noise (Tensor, optional): External noise tensor. If None, noise is sampled internally. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through the model.
        """

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

        encodings = []
        for k, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if self.encoder_noise and k < len(self.layers) - 1:
                noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
                x = self.encoder_noise_layers[k](x, noise)
            encodings.append(x)

        x = self.up_block1(x)
        noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
        x = self.noise_inject1(x, noise)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
        x = self.noise_inject2(x, noise)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
        x = self.noise_inject3(x, noise)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(
                x, size=(self.image_height, self.image_width), mode="bilinear"
            )

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x


# class PixelNoiseInjection(nn.Module):
#     """
#     A module that injects noise into feature maps, with a per-pixel and per-channel style modulation.
#     The noise intensity anneals as the forecast step increases.

#     Attributes:
#         noise_transform (nn.Linear): A linear transformation to map latent noise to the feature map's channels.
#         modulation (nn.Parameter): A learnable scaling factor applied to the noise.
#         initial_noise_factor (float): Initial scaling factor for noise injection.
#         noise_scheduler (callable): A function that determines noise scaling based on forecast_step.
#     """

#     def __init__(
#         self, noise_dim, feature_channels, initial_noise_factor=0.1, decay_rate=0.99
#     ):
#         super().__init__()
#         self.noise_transform = nn.Linear(noise_dim, feature_channels)
#         self.modulation = nn.Parameter(torch.ones(1, feature_channels, 1, 1))
#         self.initial_noise_factor = initial_noise_factor
#         self.decay_rate = decay_rate  # Decay rate for noise annealing

#     def forward(self, feature_map, noise, forecast_step=None):
#         """
#         Injects noise into the feature map with an annealing schedule.

#         Args:
#             feature_map (torch.Tensor): The input feature map (batch, channels, height, width).
#             noise (torch.Tensor): The latent noise tensor (batch, noise_dim), used for modulating the injected noise.
#             forecast_step (int, optional): The current forecast step, used to anneal noise over time.

#         Returns:
#             torch.Tensor: The feature map with injected noise.
#         """
#         batch, channels, height, width = feature_map.shape

#         # Compute noise factor with exponential decay
#         noise_factor = (
#             self.initial_noise_factor * (self.decay_rate**forecast_step)
#             if forecast_step is not None
#             else self.initial_noise_factor
#         )

#         # Generate per-pixel, per-channel noise
#         pixel_noise = noise_factor * torch.randn(
#             batch, channels, height, width, device=feature_map.device
#         )

#         # Transform latent noise and reshape
#         style = self.noise_transform(noise).view(batch, channels, 1, 1)

#         # Combine style-modulated per-pixel noise with features
#         return feature_map + pixel_noise * style * self.modulation


class PixelNoiseInjection(nn.Module):
    """
    A module that injects noise into feature maps, with a per-pixel and per-channel style modulation.

    Attributes:
        noise_transform (nn.Linear): A linear transformation to map latent noise to the feature map's channels.
        modulation (nn.Parameter): A learnable scaling factor applied to the noise.
        noise_factor (float): A scaling factor for controlling the intensity of the injected noise.

    Methods:
        forward(feature_map, noise): Adds noise to the feature map, modulated by style and the modulation parameter.
    """

    def __init__(self, noise_dim, feature_channels, noise_factor=0.1):
        super().__init__()
        self.noise_transform = nn.Linear(noise_dim, feature_channels)
        self.modulation = nn.Parameter(torch.ones(1, feature_channels, 1, 1))
        # self.noise_factor = noise_factor
        self.noise_factor = nn.Parameter(
            torch.tensor([noise_factor]), requires_grad=False
        )

    def forward(self, feature_map, noise):
        """
        Injects noise into the feature map.

        Args:
            feature_map (torch.Tensor): The input feature map (batch, channels, height, width).
            noise (torch.Tensor): The latent noise tensor (batch, noise_dim), used for modulating the injected noise.

        Returns:
            torch.Tensor: The feature map with injected noise.
        """

        batch, channels, height, width = feature_map.shape

        # Generate per-pixel, per-channel noise
        pixel_noise = self.noise_factor * torch.randn(
            batch, channels, height, width, device=feature_map.device
        )

        # Transform latent noise and reshape
        style = self.noise_transform(noise).view(batch, channels, 1, 1)

        # Combine style-modulated per-pixel noise with features
        return feature_map + pixel_noise * style * self.modulation


if __name__ == "__main__":
    # Set up the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    crossformer_config = {
        "type": "crossformer",
        "frames": 1,  # Number of input states
        "image_height": 192,  # Number of latitude grids
        "image_width": 288,  # Number of longitude grids
        "levels": 16,  # Number of upper-air variable levels
        "channels": 4,  # Upper-air variable channels
        "surface_channels": 7,  # Surface variable channels
        "input_only_channels": 3,  # Dynamic forcing, forcing, static channels
        "output_only_channels": 0,  # Diagnostic variable channels
        "patch_width": 1,  # Number of latitude grids in each 3D patch
        "patch_height": 1,  # Number of longitude grids in each 3D patch
        "frame_patch_size": 1,  # Number of input states in each 3D patch
        "dim": [32, 64, 128, 256],  # Dimensionality of each layer
        "depth": [2, 2, 2, 2],  # Depth of each layer
        "global_window_size": [8, 4, 2, 1],  # Global window size for each layer
        "local_window_size": 8,  # Local window size
        "cross_embed_kernel_sizes": [  # Kernel sizes for cross-embedding
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],  # Strides for cross-embedding
        "attn_dropout": 0.0,  # Dropout probability for attention layers
        "ff_dropout": 0.0,  # Dropout probability for feed-forward layers
        "use_spectral_norm": True,  # Whether to use spectral normalization
        "padding_conf": {
            "activate": True,
            "mode": "earth",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
        "pretrained_weights": "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/model_levels/single_step/checkpoint.pt",
    }

    crossformer_config["noise_input_dim"] = 16
    crossformer_config["noise_latent_dim"] = 32
    crossformer_config["state_latent_dim"] = 64

    logger.info("Testing the noise model")

    # Final StyleFormer

    ensemble_model = CrossFormerWithNoise(**crossformer_config).to("cuda")

    x = torch.randn(5, 74, 1, 192, 288).to(
        "cuda"
    )  # (batch size * ensemble size, channels, time, height, width)

    output = ensemble_model(x)

    print(output.shape)

    # Compute the variance
    variance = torch.var(output)

    print("Variance:", variance.item())
