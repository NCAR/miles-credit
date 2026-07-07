import torch
import torch.nn as nn

class ConditionalStochasticDecompositionLayer(nn.Module):
    """
    A module that injects stochastic noise into feature maps, modulated by a 
    semantically meaningful latent vector (conditional noise injection).

    Attributes:
        style_transform (nn.Linear): A linear transformation to map the latent vector 
                                     to per-channel style scaling factors.
        modulation (nn.Parameter): A learnable scaling factor applied to the noise.
        noise_factor (torch.Tensor): A fixed scaling factor for controlling the base 
                                     intensity of the injected noise.
    """

    def __init__(self, latent_dim, feature_channels, noise_factor=0.235):
        super().__init__()
        # Renamed to clarify it transforms a semantic latent into a style gate
        self.style_transform = nn.Linear(latent_dim, feature_channels)
        self.modulation = nn.Parameter(torch.ones(1, feature_channels, 1, 1))
        self.noise_factor = nn.Parameter(
            torch.tensor([noise_factor]), requires_grad=False
        )

    def forward(self, feature_map, latent):
        """
        Injects stochastic pixel noise into the feature map, conditionally modulated 
        by the latent vector.

        Args:
            feature_map (torch.Tensor): The input feature map (batch, channels, height, width).
            latent (torch.Tensor): The semantic latent vector (batch, latent_dim) used 
                                   as a conditional gate to modulate the noise per-channel.

        Returns:
            torch.Tensor: The feature map with conditionally modulated noise injected.
        """
        batch, channels, height, width = feature_map.shape

        # 1. Generate pure stochastic per-pixel, per-channel noise
        pixel_noise = self.noise_factor * torch.randn(
            batch, channels, height, width, device=feature_map.device
        )

        # 2. Transform the semantic latent into a per-channel style modulator
        style = self.style_transform(latent).view(batch, channels, 1, 1)

        # 3. Combine: feature_map + (stochastic_noise * semantic_style * learned_modulation)
        return feature_map + pixel_noise * style * self.modulation