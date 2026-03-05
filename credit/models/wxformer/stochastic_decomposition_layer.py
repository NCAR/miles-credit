import torch.nn as nn
import torch


class StochasticDecompositionLayer(nn.Module):
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
        self.noise_factor = nn.Parameter(torch.tensor([noise_factor]), requires_grad=False)

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
        pixel_noise = self.noise_factor * torch.randn(batch, channels, height, width, device=feature_map.device)

        # Transform latent noise and reshape
        style = self.noise_transform(noise).view(batch, channels, 1, 1)

        # Combine style-modulated per-pixel noise with features
        return feature_map + pixel_noise * style * self.modulation
