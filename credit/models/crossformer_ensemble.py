from credit.models.crossformer import CrossFormer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch


class CrossFormerWithNoise(CrossFormer):
    def __init__(self, noise_latent_dim=128, noise_factor=0.1, freeze=True, **kwargs):
        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim

        # Freeze weights if using pre-trained model
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # Noise injection layers
        self.noise_inject1 = PixelNoiseInjection(self.noise_latent_dim, self.up_block1.output_channels, noise_factor)
        self.noise_inject2 = PixelNoiseInjection(self.noise_latent_dim, self.up_block2.output_channels, noise_factor)
        self.noise_inject3 = PixelNoiseInjection(self.noise_latent_dim, self.up_block3.output_channels, noise_factor)

    def forward(self, x, noise=None):
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

        if noise is None:
            noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        x = self.up_block1(x)
        x = self.noise_inject1(x, noise)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        x = self.noise_inject2(x, noise)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        x = self.noise_inject3(x, noise)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x


class PixelNoiseInjection(nn.Module):
    def __init__(self, noise_dim, feature_channels, noise_factor=0.1):
        super().__init__()
        self.noise_transform = nn.Linear(noise_dim, feature_channels)
        self.modulation = nn.Parameter(torch.ones(1, feature_channels, 1, 1))
        self.noise_factor = noise_factor

    def forward(self, feature_map, noise):
        batch, channels, height, width = feature_map.shape

        # Generate per-pixel, per-channel noise
        pixel_noise = self.noise_factor * torch.randn(batch, channels, height, width, device=feature_map.device)

        # Transform latent noise and reshape
        style = self.noise_transform(noise).view(batch, channels, 1, 1)

        # Combine style-modulated per-pixel noise with features
        return feature_map + pixel_noise * style * self.modulation


class CrossFormerWithNoiseChannel(CrossFormer):
    def __init__(self, noise_latent_dim=128, initial_coeff=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim
        self.initial_coeff = initial_coeff

        # Noise injection layers
        self.noise_inject1 = NoiseInjection(self.noise_latent_dim, self.up_block1.output_channels)
        self.noise_inject2 = NoiseInjection(self.noise_latent_dim, self.up_block2.output_channels)
        self.noise_inject3 = NoiseInjection(self.noise_latent_dim, self.up_block3.output_channels)

    def forward(self, x, noise=None):
        x_copy = None
        if self.use_post_block:  # copy tensor to feed into postBlock later
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:  # case where only using one time-step as input
            x = x.squeeze(2)

        # Generate random noise if none is provided
        if noise is None:
            noise = torch.randn(x.size(0), self.noise_latent_dim, device=x.device)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        # Upsampling with noise injection
        x = self.up_block1(x)
        x = self.noise_inject1(x, noise)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        x = self.noise_inject2(x, noise)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        x = self.noise_inject3(x, noise)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x


class NoiseInjection(nn.Module):
    def __init__(self, noise_dim, feature_channels):
        super().__init__()
        self.noise_transform = nn.Linear(noise_dim, feature_channels)
        self.weight = nn.Parameter(torch.ones(feature_channels))  # Scale parameter for noise

    def forward(self, feature_map, noise):
        # Transform noise to match feature dimensions
        transformed_noise = self.noise_transform(noise)
        transformed_noise = transformed_noise.unsqueeze(-1).unsqueeze(-1)  # Broadcast
        return feature_map + transformed_noise * torch.exp(self.weight)


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

    x = torch.randn(5, 74, 1, 192, 288).to("cuda")  # (batch size * ensemble size, channels, time, height, width)

    output = ensemble_model(x)

    print(output.shape)

    # Compute the variance
    variance = torch.var(output)

    print("Variance:", variance.item())
