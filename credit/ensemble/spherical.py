from typing import Optional, Union

import numpy as np
import torch
from torch_harmonics import InverseRealSHT
from credit.boundary_padding import TensorPadding


class SphericalNoise:
    """Spherical harmonic-based noise generator with Matérn covariance for lat/lon grids.

    Generates spatially correlated noise on the sphere using spherical harmonics and a
    Matérn covariance structure. This method produces realistic geophysical noise
    patterns that respect the spherical geometry of Earth.

    The generated noise has Matérn covariance:

        C = σ² (-Δ + τ²I)^(-α)

    where:
    * Δ is the spherical Laplacian operator.
    * I is the identity operator.
    * σ, τ, α are scalar parameters controlling the covariance structure.

    Warning:
        For grids that don't naturally fit spherical harmonic constraints (N:2N or (N+1):2N),
        this method generates noise at a larger valid grid size and symmetrically crops to
        the target dimensions. This preserves most correlation properties but may introduce
        minor boundary effects.

    Args:
        amplitude (float, optional): Overall scaling factor for the generated noise.
            Defaults to 0.05.
        smoothness (float, optional): Regularity/smoothness parameter (α). Higher values
            produce smoother fields. Must be > 1.0 for well-defined covariance. Defaults to 2.0.
        length_scale (float, optional): Characteristic length scale parameter (τ). Higher
            values include more spatial scales in the noise. Defaults to 3.0.
        variance_scale (float | None, optional): Variance scaling parameter (σ). If None,
            computed as τ^(0.5*(2*α - 2)). Defaults to None.
    """

    def __init__(
        self,
        amplitude: float = 0.05,
        smoothness: float = 2.0,
        length_scale: float = 3.0,
        variance_scale: Union[float, None] = None,
        padding_conf: dict = None,
    ):
        self.amplitude = amplitude
        self.smoothness = smoothness
        self.length_scale = length_scale
        self.variance_scale = variance_scale

        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Generate spherical noise matching input tensor dimensions.

        Args:
            x (torch.Tensor): Reference tensor whose shape determines the output noise
                dimensions. The last two dimensions must correspond to a lat/lon grid.

        Returns:
            torch.Tensor: Spherical noise tensor with the same shape as the input, scaled
            by the amplitude.

        Raises:
            ValueError: If the lat/lon aspect ratio is not in N:2N or (N+1):2N format.
        """

        if self.use_padding:
            x = self.padding_opt.pad(x)

        input_shape = x.shape

        # Remove the old validation logic since we now handle any grid size
        lat_size = input_shape[-2]  # e.g., 192
        lon_size = input_shape[-1]  # e.g., 288

        # Generate noise using appropriate latitude count for spherical harmonics
        # Find the smallest valid spherical harmonics grid that can contain our target
        if lat_size <= 144:
            harmonic_lat_size = 144  # Will generate 144x288
        elif lat_size <= 256:
            harmonic_lat_size = 256  # Will generate 256x512, then we'll crop longitude too
        else:
            harmonic_lat_size = ((lat_size + 31) // 32) * 32  # Round up to nearest 32

        harmonic_lon_size = 2 * harmonic_lat_size

        noise_generator = SphericalRandomField(
            latitude_modes=harmonic_lat_size,
            smoothness=self.smoothness,
            length_scale=self.length_scale,
            variance_scale=self.variance_scale,
            device=x.device,
        )

        # Move generator to correct device
        noise_generator = noise_generator.to(x.device)

        # Calculate number of samples needed (all dimensions except lat/lon)
        batch_size = np.array(input_shape[:-2]).prod()

        # Generate base noise at larger valid grid size
        base_noise = noise_generator(batch_size).reshape(*input_shape[:-2], harmonic_lat_size, harmonic_lon_size)

        # Crop to target size, respecting Earth's spherical geometry
        if (harmonic_lat_size, harmonic_lon_size) != (lat_size, lon_size):
            # Latitude cropping: symmetric removal preserves balanced pole-to-pole coverage
            lat_start = (harmonic_lat_size - lat_size) // 2
            lat_end = lat_start + lat_size

            # Longitude cropping: maintain periodicity by sampling evenly across 360°
            if harmonic_lon_size != lon_size:
                # Calculate stride to sample evenly across the full longitude range
                lon_indices = torch.linspace(0, harmonic_lon_size - 1, lon_size, dtype=torch.long)
                final_noise = base_noise[..., lat_start:lat_end, :]
                final_noise = final_noise[..., lon_indices]
            else:
                final_noise = base_noise[..., lat_start:lat_end, :lon_size]
        else:
            final_noise = base_noise

        if self.use_padding:
            final_noise = self.padding_opt.unpad(final_noise)

        if hasattr(self.amplitude, "__len__"):
            amp_tensor = torch.tensor(self.amplitude, device=final_noise.device, dtype=final_noise.dtype)
            C1 = len(amp_tensor)
            final_noise[:, -C1:, :, :] *= amp_tensor.view(1, -1, 1, 1, 1)
            return final_noise
        else:
            return self.amplitude * final_noise


class SphericalRandomField(torch.nn.Module):
    """Gaussian Random Field generator on the sphere with Matérn covariance.

    Implements a mean-zero Gaussian Random Field on the sphere using spherical
    harmonics with Matérn covariance structure:

        C = σ² (-Δ + τ²I)^(-α)

    where:
    - Δ is the spherical Laplacian operator.
    - I is the identity operator.
    - σ² controls overall variance (variance_scale²).
    - τ² controls characteristic length scale (length_scale²).
    - α controls smoothness/regularity (smoothness).

    The covariance is trace-class (well-defined) if and only if α > 1.

    Args:
        latitude_modes (int): Number of spherical harmonic modes in the latitude
            direction. Longitude modes are automatically set to 2 * latitude_modes.
        smoothness (float, optional): Regularity parameter (α). Higher values produce
            smoother fields. Must be > 1.0. Defaults to 2.0.
        length_scale (float, optional): Characteristic length scale parameter (τ).
            Defaults to 3.0.
        variance_scale (Union[float, None], optional): Variance parameter (σ). If
            None, computed automatically as τ^(0.5*(2*α - 2)). Defaults to None.
        sphere_radius (float, optional): Radius of the sphere for scaling. Defaults
            to 1.0.
        grid_type (str, optional): Grid type for spherical harmonics. Options are
            "equiangular" or "legendre-gauss". Defaults to "equiangular".
        dtype (torch.dtype, optional): Numerical precision for calculations. Defaults
            to torch.float32.
        device (torch.device, optional): PyTorch device for computations. Defaults
            to "cuda:0".
    """

    def __init__(
        self,
        latitude_modes: int,
        smoothness: float = 2.0,
        length_scale: float = 3.0,
        variance_scale: Union[float, None] = None,
        sphere_radius: float = 1.0,
        grid_type: str = "equiangular",
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda:0",
    ):
        super().__init__()

        # Store grid parameters
        self.latitude_modes = latitude_modes
        self.longitude_modes = 2 * latitude_modes

        # Validate smoothness parameter
        if smoothness < 1.0:
            raise ValueError(f"Smoothness parameter must be > 1.0 for well-defined covariance. Got: {smoothness}")

        # Set default variance scale if not provided
        if variance_scale is None:
            variance_scale = length_scale ** (0.5 * (2 * smoothness - 2.0))

        # Initialize inverse spherical harmonic transform
        self.inverse_sht = (
            InverseRealSHT(self.latitude_modes, self.longitude_modes, grid=grid_type, norm="backward")
            .to(dtype=dtype)
            .to(device=device)
        )

        # Compute square root of covariance eigenvalues
        # Eigenvalues of spherical Laplacian: λ_j = j(j+1) for j = 0, 1, 2, ...
        laplacian_eigenvals = torch.tensor([j * (j + 1) for j in range(self.latitude_modes)], device=device)

        # Reshape for broadcasting over all spherical harmonic modes
        laplacian_eigenvals = laplacian_eigenvals.view(self.latitude_modes, 1).repeat(1, self.latitude_modes + 1)

        # Compute covariance eigenvalues: σ² * (λ/R² + τ²)^(-α)
        covariance_eigenvals = variance_scale * (
            (laplacian_eigenvals / sphere_radius**2 + length_scale**2) ** (-smoothness / 2.0)
        )

        # Apply lower triangular mask (spherical harmonics structure)
        covariance_sqrt = torch.tril(covariance_eigenvals)

        # Set DC component to zero (ensures zero mean)
        covariance_sqrt[0, 0] = 0.0

        # Add batch dimension and register as buffer
        covariance_sqrt = covariance_sqrt.unsqueeze(0)
        self.register_buffer("covariance_sqrt", covariance_sqrt)

        # Register Gaussian distribution parameters
        gaussian_mean = torch.tensor([0.0], device=device, dtype=dtype)
        gaussian_std = torch.tensor([1.0], device=device, dtype=dtype)
        self.register_buffer("gaussian_mean", gaussian_mean)
        self.register_buffer("gaussian_std", gaussian_std)

    def forward(self, num_samples: int, noise_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate random field samples on the sphere.

        Uses Karhunen-Loève expansion to generate correlated random fields:
        1. Sample independent Gaussian noise in spherical harmonic space.
        2. Scale by the square root of covariance eigenvalues.
        3. Transform back to physical space via inverse spherical harmonics.

        Args:
            num_samples (int): Number of independent random field realizations to generate.
            noise_input (torch.Tensor, optional): Pre-generated complex Gaussian noise
                with shape (num_samples, latitude_modes, latitude_modes+1). If None,
                new noise is sampled automatically. Defaults to None.

        Returns:
            torch.Tensor: Random field samples with shape
                (num_samples, latitude_modes, longitude_modes) on an equiangular grid
                covering the sphere.
        """

        # Generate or use provided Gaussian noise in spherical harmonic space
        if noise_input is None:
            # Create standard Gaussian distribution
            gaussian_dist = torch.distributions.normal.Normal(self.gaussian_mean, self.gaussian_std)

            # Sample complex Gaussian noise: real and imaginary parts
            noise_shape = torch.Size(
                (
                    num_samples,
                    self.latitude_modes,
                    self.latitude_modes + 1,
                    2,  # Real and imaginary components
                )
            )

            complex_noise_parts = gaussian_dist.sample(noise_shape).squeeze()
            # Convert to complex tensor
            complex_noise = torch.view_as_complex(complex_noise_parts)
        else:
            complex_noise = noise_input

        # Apply Karhunen-Loève expansion: scale noise by covariance square root
        scaled_harmonics = complex_noise * self.covariance_sqrt

        # Transform from spherical harmonic space to physical lat/lon grid
        random_field = self.inverse_sht(scaled_harmonics)

        return random_field
