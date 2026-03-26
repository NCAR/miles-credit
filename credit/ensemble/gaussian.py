import torch


class GaussianNoise:
    """Simple Gaussian white noise generator.

    Generates uncorrelated Gaussian noise with zero mean and controllable standard
    deviation. Each point is independently sampled from a normal distribution.

    Unlike spatially correlated noise (e.g., RedNoise), this produces completely
    independent random values at each grid point, which may be appropriate for:
    * Model parameter uncertainty
    * Observation errors
    * Simple perturbation schemes
    * Baseline comparisons with more sophisticated noise models

    Args:
        amplitude (float, optional): Standard deviation of the Gaussian noise.
            Defaults to 0.05.
    """

    def __init__(self, amplitude: float = 0.05):
        self.amplitude = amplitude

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Gaussian white noise matching input tensor dimensions.

        Args:
            x (torch.Tensor): Reference tensor whose shape determines the output noise
                dimensions.

        Returns:
            torch.Tensor: Gaussian white noise tensor with the same shape as input,
                zero mean, and standard deviation equal to the amplitude parameter.
        """

        return self.amplitude * torch.randn_like(x, device=x.device)

    def __repr__(self) -> str:
        return f"GaussianNoise(amplitude={self.amplitude})"
