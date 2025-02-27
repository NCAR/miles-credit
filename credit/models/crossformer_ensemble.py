from credit.models.crossformer import CrossFormer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch


class CrossFormerWithNoiseEmbedding(CrossFormer):
    def __init__(self, noise_latent_dim: int, **kwargs):
        """
        CrossFormer with noise embedding integration.
        Args:
            noise_latent_dim (int): Dimension of the noise embedding.
            **kwargs: Other arguments for the base CrossFormer.
        """
        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim
        self.total_input_channels = (
            self.channels * self.levels
            + self.surface_channels
            + self.input_only_channels
        )

        # A projection layer to incorporate the noise embedding into the model
        self.noise_projector = nn.Linear(noise_latent_dim, self.total_input_channels)

    def forward(self, x, noise_embedding):
        """
        Forward pass with noise embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, levels, height, width).
            noise_embedding (torch.Tensor): Noise embedding tensor of shape (batch_size, noise_latent_dim).
        """
        batch_size, _, levels, height, width = x.shape

        # Get a noise projection
        noise_projection = self.noise_projector(
            noise_embedding
        )  # (batch_size, channels * levels)
        noise_projection = noise_projection.reshape(
            batch_size, self.total_input_channels, levels
        )
        noise_projection = noise_projection.unsqueeze(-1).unsqueeze(-1)  # Add H,W dims
        noise_projection = noise_projection.expand(-1, -1, -1, height, width)

        # Expand noise projection to match the spatial dimensions of `x`
        noise_projection = noise_projection.expand(-1, -1, -1, height, width)

        # Modify input with noise projection
        x = x + 0.00001 * noise_projection

        # Proceed with the standard CrossFormer forward pass
        return super().forward(x)


class NoiseEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dims: tuple = (128, 256)):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims, embed_dim]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, noise):
        return self.mlp(noise)


class WXFormerEnsemble(nn.Module):
    def __init__(self, noise_input_dim, noise_latent_dim, **kwargs):
        super().__init__()

        """
        Ensemble using a single model with noise embedding.
        Args:
            crossformer_config (dict): Configuration dictionary containing model parameters.
                        Required keys in crossformer_config:
                        - noise_input_dim: Dimension of the noise vector
                        - noise_latent_dim: Dimension of the noise embedding
        """
        super().__init__()

        # required_keys = ["noise_input_dim", "noise_latent_dim"]

        # for key in required_keys:
        #     if key not in crossformer_config:
        #         logging.warning(f"Missing required key '{key}' in model config")

        self.base_model = CrossFormerWithNoiseEmbedding(
            noise_latent_dim=noise_latent_dim, **kwargs
        )

        self.noise_input_dim = noise_input_dim
        self.noise_latent_dim = noise_latent_dim

        self.noise_mlp = NoiseEmbeddingMLP(self.noise_input_dim, self.noise_latent_dim)

    def forward(self, x):
        """
        Forward pass with ensemble generation.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Aggregated predictions.
        """
        batch_size = x.size(0)

        # Generate independent noise vectors for each ensemble member
        noise = torch.randn((batch_size, self.noise_input_dim), device=x.device)

        # x.shape = (batch*ens, *)
        return self.base_model(x, self.noise_mlp(noise))


### Start version with latent state coupled with noise


class StateLatentEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_channels: tuple = (32, 64, 128),
    ):
        """
        Encodes the input state into a compact latent representation using convolutions.
        Args:
            input_channels (int): Number of input channels in the state tensor.
            latent_dim (int): Dimension of the final latent bottleneck representation.
            hidden_channels (tuple): Number of channels for intermediate convolutional layers.
        """
        super().__init__()
        layers = []

        # Add convolutional layers
        in_channels = input_channels
        for out_channels in hidden_channels:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Add a global pooling layer to aggregate spatial dimensions
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 feature map
        self.fc = nn.Linear(hidden_channels[-1], latent_dim)  # Final bottleneck layer

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): Input tensor of shape (batch_size, channels, 1, height, width).
        Returns:
            torch.Tensor: Latent state representation of shape (batch_size, latent_dim).
        """
        batch_size = state.shape[0]
        state = state.squeeze(
            2
        )  # Remove singleton dimension -> (batch_size, channels, height, width)

        # Apply convolutional layers
        conv_out = self.conv_layers(
            state
        )  # Shape: (batch_size, hidden_channels[-1], H', W')

        # Global pooling to reduce spatial dimensions
        pooled_out = self.global_pool(
            conv_out
        )  # Shape: (batch_size, hidden_channels[-1], 1, 1)

        # Flatten and pass through final linear layer
        latent_state = self.fc(
            pooled_out.view(batch_size, -1)
        )  # Shape: (batch_size, latent_dim)

        return latent_state


class CrossFormerWithStateNoiseEmbedding(CrossFormer):
    def __init__(
        self,
        state_latent_dim: int,
        noise_latent_dim: int = 64,
        pretrained_weights=False,
        **kwargs,
    ):
        """
        CrossFormer with state-dependent noise embedding integration.
        Args:
            state_embed_dim (int): Dimension of the state bottleneck embedding.
            noise_latent_dim (int): Dimension of the noise embedding.
            **kwargs: Other arguments for the base CrossFormer.
        """
        super().__init__(**kwargs)

        # Load pretrained weights if provided

        if pretrained_weights:
            crossformer_state_dict = torch.load(pretrained_weights)
            current_state = self.state_dict()
            for key in crossformer_state_dict:
                if key in current_state:
                    current_state[key] = crossformer_state_dict[key]
            self.load_state_dict(current_state, strict=False)
            logging.info(f"Loaded CrossFormer weights from {pretrained_weights}")

        self.noise_latent_dim = noise_latent_dim
        self.state_embed_dim = state_latent_dim
        self.total_input_channels = (
            self.channels * self.levels
            + self.surface_channels
            + self.input_only_channels
        )

        # A projection layer to combine state and noise embedding
        self.state_noise_projector = nn.Linear(
            state_latent_dim + noise_latent_dim, self.total_input_channels
        )

    def forward(self, x, state_embedding, noise_embedding):
        """
        Forward pass with state-dependent noise embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, levels, height, width).
            state_embedding (torch.Tensor): Latent state tensor of shape (batch_size, state_embed_dim).
            noise_embedding (torch.Tensor): Noise embedding tensor of shape (batch_size, noise_latent_dim).
        """
        batch_size, _, levels, height, width = x.shape

        # Concatenate state and noise embeddings
        state_noise = torch.cat([state_embedding, noise_embedding], dim=-1)

        # Project combined embeddings
        state_noise_projection = self.state_noise_projector(state_noise)
        state_noise_projection = state_noise_projection.reshape(
            batch_size, self.total_input_channels, levels
        )
        state_noise_projection = state_noise_projection.unsqueeze(-1).unsqueeze(-1)
        state_noise_projection = state_noise_projection.expand(
            -1, -1, -1, height, width
        )

        # Modify input with state-dependent noise projection
        x = x + state_noise_projection

        # Proceed with the standard CrossFormer forward pass (only `x`)
        return super().forward(x)


class WXFormerStateEnsemble(nn.Module):
    def __init__(
        self,
        noise_input_dim: int,
        noise_latent_dim: int,
        state_latent_dim: int,
        pretrained_weights: bool = False,
        **kwargs,
    ):
        """
        Ensemble using a single model with state and noise embeddings.
        Args:
            noise_input_dim (int): Dimension of the noise vector
            noise_latent_dim (int): Dimension of the noise embedding
            state_latent_dim (int): Dimension of the state representation
            **kwargs: Additional arguments passed to CrossFormerWithStateNoiseEmbedding
        """
        super().__init__()

        # Store dimensions
        self.noise_input_dim = noise_input_dim
        self.noise_latent_dim = noise_latent_dim
        self.state_latent_dim = state_latent_dim

        # Initialize the base model
        self.base_model = CrossFormerWithStateNoiseEmbedding(
            state_latent_dim=state_latent_dim,
            noise_latent_dim=noise_latent_dim,
            pretrained_weights=pretrained_weights,
            **kwargs,
        )

        # Initialize encoders
        self.state_encoder = StateLatentEncoder(
            input_channels=self.base_model.input_channels,
            latent_dim=self.state_latent_dim,
        )
        self.noise_mlp = NoiseEmbeddingMLP(self.noise_input_dim, self.noise_latent_dim)

    def forward(self, x):
        """
        Forward pass with state and noise embeddings.
        Args:
            x (torch.Tensor): Input tensor.
            state (torch.Tensor): State tensor.
        Returns:
            torch.Tensor: Aggregated predictions.
        """
        batch_size = x.size(0)

        # Generate latent state representation
        state_embedding = self.state_encoder(x)

        # Generate independent noise vectors for each ensemble member
        noise = torch.randn((batch_size, self.noise_input_dim), device=x.device)

        # x.shape = (batch*ens, *)
        return self.base_model(x, state_embedding, self.noise_mlp(noise))


### Start version with latent state perturbation


class NoiseInjectionLayer(nn.Module):
    def __init__(self, channels):
        """
        Noise injection layer similar to StyleGAN's approach.
        Args:
            channels (int): Number of channels to inject noise into
        """
        super().__init__()
        # Per-channel scaling factors for noise
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        """
        Args:
            x (torch.Tensor): Input feature map
            noise (torch.Tensor, optional): Pre-generated noise
        """
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise


class LatentEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_channels: tuple = (32, 64, 128),
    ):
        """
        Encodes the input state into a compact latent representation using convolutions
        with StyleGAN-like noise injection.
        Args:
            input_channels (int): Number of input channels in the state tensor
            latent_dim (int): Dimension of the final latent bottleneck representation
            hidden_channels (tuple): Number of channels for intermediate layers
        """
        super().__init__()

        self.layers = nn.ModuleList()
        in_channels = input_channels

        # Build layers with noise injection
        for out_channels in hidden_channels:
            block = nn.ModuleDict(
                {
                    "conv": nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=2, padding=1
                    ),
                    "noise": NoiseInjectionLayer(out_channels),
                    "act": nn.ReLU(),
                }
            )
            self.layers.append(block)
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels[-1], latent_dim)

        # Final noise injection for latent space
        self.latent_noise = NoiseInjectionLayer(latent_dim)

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): Input tensor of shape (batch_size, channels, 1, height, width)
        Returns:
            torch.Tensor: Latent state representation with noise of shape (batch_size, latent_dim)
        """
        batch_size = state.shape[0]
        x = state.squeeze(2)  # Remove singleton dimension

        # Process through layers with noise injection
        for layer in self.layers:
            x = layer["conv"](x)
            x = layer["noise"](x)  # Inject noise after conv
            x = layer["act"](x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        # Final latent space noise injection
        if self.training:
            # During training, reshape to 4D for noise injection
            x = x.view(batch_size, -1, 1, 1)
            x = self.latent_noise(x)
            x = x.squeeze(-1).squeeze(-1)  # Back to 2D

        return x


class CrossFormerWithState(CrossFormer):
    def __init__(
        self,
        state_latent_dim: int,
        pretrained_weights=False,
        **kwargs,
    ):
        """
        CrossFormer with state embedding integration.
        Args:
            state_latent_dim (int): Dimension of the state bottleneck embedding.
            **kwargs: Other arguments for the base CrossFormer.
        """
        super().__init__(**kwargs)

        if pretrained_weights:
            crossformer_state_dict = torch.load(pretrained_weights)
            current_state = self.state_dict()
            for key in crossformer_state_dict:
                if key in current_state:
                    current_state[key] = crossformer_state_dict[key]
            self.load_state_dict(current_state, strict=False)
            logging.info(f"Loaded CrossFormer weights from {pretrained_weights}")

        self.state_embed_dim = state_latent_dim
        self.total_input_channels = (
            self.channels * self.levels
            + self.surface_channels
            + self.input_only_channels
        )

        # Project state embedding to input dimensions
        self.state_projector = nn.Linear(state_latent_dim, self.total_input_channels)

    def forward(self, x, state_embedding):
        """
        Forward pass with state embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, levels, height, width).
            state_embedding (torch.Tensor): Latent state tensor of shape (batch_size, state_embed_dim).
        """
        batch_size, _, levels, height, width = x.shape

        # Project state embedding
        state_projection = self.state_projector(state_embedding)
        state_projection = state_projection.reshape(
            batch_size, self.total_input_channels, levels
        )
        state_projection = state_projection.unsqueeze(-1).unsqueeze(-1)
        state_projection = state_projection.expand(-1, -1, -1, height, width)

        # Modify input with state projection
        x = x + state_projection

        return super().forward(x)


class WXFormerStyle(nn.Module):
    def __init__(
        self,
        state_latent_dim: int,
        noise_scale: float = 0.1,
        pretrained_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.state_latent_dim = state_latent_dim
        self.noise_scale = noise_scale

        # Initialize the base model
        self.base_model = CrossFormerWithState(
            state_latent_dim=state_latent_dim,
            pretrained_weights=pretrained_weights,
            **kwargs,
        )

        self.state_encoder = LatentEncoder(
            input_channels=self.base_model.input_channels,
            latent_dim=self.state_latent_dim,
        )

    def forward(self, x):
        """
        Forward pass with feature-statistics-based perturbations.
        The batch dimension represents ensemble members (identical inputs).
        Args:
            x (torch.Tensor): Input tensor [B, C, L, H, W] where B contains identical copies
        """
        # Process through network normally
        state_embedding = self.state_encoder(x)
        output = self.base_model(x, state_embedding)

        return output


# Last one where we directly modify the CrossFormer and add Style-GAN like noise


class CrossFormerWithNoise(CrossFormer):
    def __init__(self, noise_latent_dim=128, noise_factor=0.1, freeze=True, **kwargs):
        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim

        # Freeze weights if using pre-trained model
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # Noise injection layers
        self.noise_inject1 = PixelNoiseInjection(
            self.noise_latent_dim, self.up_block1.output_channels, noise_factor
        )
        self.noise_inject2 = PixelNoiseInjection(
            self.noise_latent_dim, self.up_block2.output_channels, noise_factor
        )
        self.noise_inject3 = PixelNoiseInjection(
            self.noise_latent_dim, self.up_block3.output_channels, noise_factor
        )

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


class PixelNoiseInjection(nn.Module):
    def __init__(self, noise_dim, feature_channels, noise_factor=0.1):
        super().__init__()
        self.noise_transform = nn.Linear(noise_dim, feature_channels)
        self.modulation = nn.Parameter(torch.ones(1, feature_channels, 1, 1))
        self.noise_factor = noise_factor

    def forward(self, feature_map, noise):
        batch, channels, height, width = feature_map.shape

        # Generate per-pixel, per-channel noise
        pixel_noise = self.noise_factor * torch.randn(
            batch, channels, height, width, device=feature_map.device
        )

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
        self.noise_inject1 = NoiseInjection(
            self.noise_latent_dim, self.up_block1.output_channels
        )
        self.noise_inject2 = NoiseInjection(
            self.noise_latent_dim, self.up_block2.output_channels
        )
        self.noise_inject3 = NoiseInjection(
            self.noise_latent_dim, self.up_block3.output_channels
        )

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


class NoiseInjection(nn.Module):
    def __init__(self, noise_dim, feature_channels):
        super().__init__()
        self.noise_transform = nn.Linear(noise_dim, feature_channels)
        self.weight = nn.Parameter(
            torch.ones(feature_channels)
        )  # Scale parameter for noise

    def forward(self, feature_map, noise):
        # Transform noise to match feature dimensions
        transformed_noise = self.noise_transform(noise)
        transformed_noise = transformed_noise.unsqueeze(-1).unsqueeze(-1)  # Broadcast
        return feature_map + transformed_noise * torch.exp(self.weight)


if __name__ == "__main__":
    import gc

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

    # Create the single-model ensemble
    ensemble_model = WXFormerEnsemble(**crossformer_config).to("cuda")

    # Example input
    x = torch.randn(5, 74, 1, 192, 288).to(
        "cuda"
    )  # (batch size * ensemble size, channels, time, height, width)
    output = ensemble_model(x)

    print(output.shape)

    # Compute the variance
    variance = torch.var(output)

    print("Variance:", variance.item())

    del x, ensemble_model
    # clear the cached memory from the gpu
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Testing the stateful noise model")

    crossformer_config["noise_input_dim"] = 16
    crossformer_config["noise_latent_dim"] = 32
    crossformer_config["state_latent_dim"] = 64

    ensemble_model = WXFormerStateEnsemble(
        **crossformer_config,
    ).to("cuda")

    x = torch.randn(5, 74, 1, 192, 288).to(
        "cuda"
    )  # (batch size * ensemble size, channels, time, height, width)

    output = ensemble_model(x)

    print(output.shape)

    # Version 3 with noise added to latent state

    ensemble_model = WXFormerStyle(**crossformer_config).to("cuda")

    x = torch.randn(5, 74, 1, 192, 288).to(
        "cuda"
    )  # (batch size * ensemble size, channels, time, height, width)

    output = ensemble_model(x)

    print(output.shape)

    # Compute the variance
    variance = torch.var(output)

    print("Variance:", variance.item())

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
