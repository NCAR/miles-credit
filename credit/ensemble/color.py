import torch
from typing import Optional, Tuple, Union


class ColorNoise:
    """2D spatially correlated noise generator for lat/lon grids.

    Generates colored noise with controllable spatial correlation using power-law
    scaling in the frequency domain. The noise characteristics are determined by
    the reddening parameter:

    - reddening = 0: White noise (uncorrelated, flat power spectrum)
    - reddening = 1: Pink noise (1/f power spectrum)
    - reddening = 2: Brown/Brownian/Red noise (1/f² power spectrum)
    - reddening > 2: Higher-order red noise (1/f^n power spectrum)

    Higher reddening values produce smoother, more spatially coherent patterns,
    which are often more realistic for geophysical applications.

    Parameters
    ----------
    amplitude : float, optional
        Scaling factor for the generated noise, by default 0.05
    reddening : int, optional
        Power-law exponent controlling spatial correlation. Higher values create
        smoother, more correlated noise patterns, by default 2 (Brown noise)
    """

    def __init__(self, amplitude: float = 0.05, reddening: int = 2):
        self.amplitude = amplitude
        self.reddening = reddening

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate spatially correlated noise matching input tensor dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Reference tensor whose shape determines the output noise dimensions.
            The last two dimensions are treated as the spatial lat/lon grid.

        Returns
        -------
        torch.Tensor
            Spatially correlated noise tensor with the same shape as input,
            scaled by the amplitude parameter.
        """
        correlated_noise = self._create_correlated_noise(x.shape, x.device)
        return self.amplitude * correlated_noise

    def _create_correlated_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate spatially correlated noise using frequency domain filtering.

        Creates colored noise by:
        1. Starting with white noise (uniform power spectrum)
        2. Transforming to frequency domain
        3. Applying power-law scaling (1/f^reddening)
        4. Transforming back to spatial domain

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the output noise tensor
        device : torch.device
            Device to generate tensors on

        Returns
        -------
        torch.Tensor
            Spatially correlated noise with unit variance (before amplitude scaling)
        """
        # Generate initial white noise
        white_noise = torch.randn(*shape, device=device)

        # Transform to frequency domain (only last 2 dimensions)
        freq_domain = torch.fft.rfft2(white_noise, dim=(-2, -1))

        # Create frequency grids for power-law scaling
        freq_y = torch.abs(torch.fft.fftfreq(shape[-2], device=device)).reshape(-1, 1)
        freq_x = torch.fft.rfftfreq(shape[-1], device=device)

        # Compute power spectrum weights: 1/f^reddening
        power_spectrum = freq_y**self.reddening + freq_x**self.reddening
        scaling_weights = 1.0 / torch.where(power_spectrum > 0, power_spectrum, 1.0)

        # Remove DC component to ensure zero mean
        scaling_weights[..., 0, 0] = 0

        # Normalize to maintain consistent variance
        scaling_weights = scaling_weights / torch.sqrt(torch.mean(scaling_weights**2))

        # Apply frequency domain filtering
        filtered_freq = freq_domain * scaling_weights

        # Transform back to spatial domain
        correlated_noise = torch.fft.irfft2(filtered_freq, s=shape[-2:], dim=(-2, -1))

        return correlated_noise

    def __repr__(self) -> str:
        return f"ColorNoise(amplitude={self.amplitude},reddening={self.reddening})"


def apply_noise_perturbation_step(
    x: torch.Tensor,
    delta_prev: Optional[torch.Tensor],
    forecast_step: int,
    rho: float = 0.9,
    perturbation_std: Union[float, torch.Tensor] = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Red noise perturbation in physical units, with per-channel control.
    """
    device = x.device
    dynamic_x = x

    if isinstance(perturbation_std, float):
        noise_scale = perturbation_std
    else:
        # broadcast to match shape: [B, C_dyn, T, H, W]
        noise_scale = perturbation_std.view(1, -1, 1, 1, 1).to(device)

    if forecast_step == 1 or delta_prev is None:
        white_noise = noise_scale * torch.randn_like(dynamic_x, device=device)
        delta = white_noise
    else:
        white_noise = noise_scale * torch.randn_like(dynamic_x, device=device)
        delta = rho * delta_prev + white_noise

    x = dynamic_x + delta
    return x, delta
