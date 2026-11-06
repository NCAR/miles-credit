from credit.models.wxformer.crossformer import CrossFormer
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
from credit.ensemble.red_noise import AR1Noise, ar1_phi, red_noise_errors
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer


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
        correlated=False,
        red_noise=None,
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
            red_noise (dict, optional): Make the noise AR(1) in time instead of white. Keys:
                ``tau_hours`` (one e-folding time for every injection point, or a list of three for the
                decoder points, coarsest first; 0 or null keeps a point white), ``timestep_hours`` (the
                model step) and ``apply_to`` (any of ``latent`` and ``pixel``; default both). Every step
                keeps the white N(0, 1) marginal, so a white-trained checkpoint runs it unchanged.
                None, the default, is white noise. See ``credit.ensemble.red_noise``.
            **kwargs: Additional arguments passed to the CrossFormer base class.
        """

        super().__init__(**kwargs)
        self.noise_latent_dim = noise_latent_dim
        self.correlated = correlated

        # Freeze weights if using pre-trained model
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.encoder_noise = encoder_noise
        if encoder_noise:
            # Define separate learnable noise factors for encoder noise layers
            encoder_noise_factors = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(encoder_noise_factor, dtype=torch.float32)),
                    nn.Parameter(torch.tensor(encoder_noise_factor, dtype=torch.float32)),
                    nn.Parameter(torch.tensor(encoder_noise_factor, dtype=torch.float32)),
                ]
            )
            # Encoder noise injection layers
            self.encoder_noise_layers = nn.ModuleList(
                [
                    StochasticDecompositionLayer(
                        self.noise_latent_dim,
                        self.up_block3.sharp.out_channels,
                        encoder_noise_factors[0],
                    ),
                    StochasticDecompositionLayer(
                        self.noise_latent_dim,
                        self.up_block2.sharp.out_channels,
                        encoder_noise_factors[1],
                    ),
                    StochasticDecompositionLayer(
                        self.noise_latent_dim,
                        self.up_block1.sharp.out_channels,
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
        self.noise_inject1 = StochasticDecompositionLayer(
            self.noise_latent_dim,
            self.up_block1.sharp.out_channels,
            decoder_noise_factors[0],
        )
        self.noise_inject2 = StochasticDecompositionLayer(
            self.noise_latent_dim,
            self.up_block2.sharp.out_channels,
            decoder_noise_factors[1],
        )
        self.noise_inject3 = StochasticDecompositionLayer(
            self.noise_latent_dim,
            self.up_block3.sharp.out_channels,
            decoder_noise_factors[2],
        )

        # Red latent streams by injection point; empty means white latents.
        self._latent_ar1 = {}
        self.red_noise = red_noise
        if red_noise is not None:
            self._init_red_noise(red_noise)

    def _init_red_noise(self, red_noise):
        errors = red_noise_errors(red_noise, encoder_noise=self.encoder_noise, correlated=self.correlated)
        if errors:
            raise ValueError("model.red_noise: " + "; ".join(errors))
        tau = red_noise["tau_hours"]
        dt = red_noise["timestep_hours"]
        apply_to = red_noise.get("apply_to", ["latent", "pixel"])

        decoder_taus = tau if isinstance(tau, (list, tuple)) else [tau] * 3
        points = [
            (("decoder", i), layer, t)
            for i, (layer, t) in enumerate(
                zip([self.noise_inject1, self.noise_inject2, self.noise_inject3], decoder_taus)
            )
        ]
        if self.encoder_noise:
            points += [(("encoder", k), layer, tau) for k, layer in enumerate(self.encoder_noise_layers)]
        for point, layer, t in points:
            phi = ar1_phi(t, dt)
            if not phi:
                continue
            if "pixel" in apply_to:
                layer.pixel_ar1 = AR1Noise(phi)
            if "latent" in apply_to and not self.correlated:
                self._latent_ar1[point] = AR1Noise(phi)
        if "latent" in apply_to and self.correlated and ar1_phi(tau, dt):
            self._latent_ar1["shared"] = AR1Noise(ar1_phi(tau, dt))

    def reset_noise_state(self):
        """Restart the red latent streams (if any) from their stationary distribution.

        The SDL layers' per-pixel streams reset themselves; call
        ``credit.ensemble.red_noise.reset_noise_state(model)`` to reset both.
        """
        for stream in self._latent_ar1.values():
            stream.reset()

    def _latent(self, x, point):
        """The latent z for one injection point: a fresh N(0, 1) draw, or the next AR(1) step if red."""
        stream = self._latent_ar1.get(point)
        if stream is None:
            return torch.randn(x.size(0), self.noise_latent_dim, device=x.device)
        return stream.sample((x.size(0), self.noise_latent_dim), x.device)

    def forward(self, x, noise=None, forecast_step=None):
        """
        Forward pass through the CrossFormer with noise injection.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
            noise (Tensor, optional): External noise tensor. If None, noise is sampled internally. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through the model.
        """

        if self.correlated:
            noise = self._latent(x, "shared")

        x_copy = None
        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)

        if self.frames > 1:
            b, c, t, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
        else:
            x = x.squeeze(2)

        encodings = []
        for k, (cel, transformer) in enumerate(self.layers):
            x = cel(x)
            x = transformer(x)
            if self.encoder_noise and k < len(self.layers) - 1:
                if not self.correlated:
                    noise = self._latent(x, ("encoder", k))
                x = self.encoder_noise_layers[k](x, noise)
            encodings.append(x)

        x = self.up_block1(x)
        if not self.correlated:
            noise = self._latent(x, ("decoder", 0))
        x = self.noise_inject1(x, noise)
        x = torch.cat([x, encodings[2]], dim=1)

        x = self.up_block2(x)
        if not self.correlated:
            noise = self._latent(x, ("decoder", 1))
        x = self.noise_inject2(x, noise)
        x = torch.cat([x, encodings[1]], dim=1)

        x = self.up_block3(x)
        if not self.correlated:
            noise = self._latent(x, ("decoder", 2))
        x = self.noise_inject3(x, noise)
        x = torch.cat([x, encodings[0]], dim=1)

        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        b, _, h, w = x.shape
        x = x.view(b, self.base_output_channels, self.output_frames, h, w)

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
