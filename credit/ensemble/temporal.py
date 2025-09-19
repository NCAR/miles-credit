import os
import torch
import xarray as xr
from typing import Optional, Union
from credit.ensemble.utils import hemispheric_rescale as hemi_rescale


class TemporalNoise:
    """AR(1) temporal noise generator that leverages spatial noise patterns.

    Implements an autoregressive process of order 1 for temporal correlation:
    δ_t = ρ * δ_{t-1} + ε_t

    where ρ is the temporal correlation coefficient and ε_t is generated using
    a Noise instance (allowing for spatially correlated innovations).

    This creates perturbations that evolve smoothly over forecast steps while
    maintaining realistic spatial patterns at each time step.

    Parameters
    ----------
    noise_generator : Noise
        Noise generator instance used to create the white noise innovations (ε_t)
    temporal_correlation : float, optional
        Temporal correlation coefficient (0-1). Higher values create smoother
        temporal evolution, by default 0.9
    perturbation_std : Union[float, torch.Tensor], optional
        Noise standard deviation scaling. Can be either:
        - float: uniform scaling applied to all channels
        - torch.Tensor: per-channel scaling with shape matching channel dimension
        If provided, overrides the amplitude from the noise_generator, by default None
    """

    def __init__(
        self,
        noise_generator: torch.Tensor,
        temporal_correlation: float = 0.9,
        perturbation_std: Optional[Union[float, torch.Tensor]] = None,
        hemispheric_rescale: Optional[bool] = False,
        terrain_file: str = None,
    ):
        self.noise_generator = noise_generator
        self.temporal_correlation = temporal_correlation
        self.perturbation_std = perturbation_std
        self.hemispheric_rescale = hemi_rescale if hemispheric_rescale else False
        if self.hemispheric_rescale is not None:
            if not os.path.exists(terrain_file) or terrain_file is None:
                raise FileNotFoundError(f"Terrain file {terrain_file} not found")
            latlons = xr.open_dataset(terrain_file).load()
            self.latitudes = torch.tensor(latlons.latitude.values)

    def __call__(
        self, x: torch.Tensor, previous_perturbation: Optional[torch.Tensor] = None, forecast_step: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply temporally correlated perturbation for sequential forecasting.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor to perturb
        previous_perturbation : torch.Tensor, optional
            Perturbation from the previous forecast step, by default None.
            If None or forecast_step=1, generates new initial perturbation.
        forecast_step : int, optional
            Current forecast step (1-indexed), by default 1

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - Perturbed state tensor (x + perturbation)
            - Current perturbation tensor (for use in next step)
        """
        # Generate white noise innovation using the spatial noise generator
        if self.perturbation_std is not None:
            # Generate base noise with original amplitude
            white_noise = self.noise_generator(x)
            original_amplitude = self.noise_generator.amplitude

            # Convert both to tensors for consistent handling
            if isinstance(self.perturbation_std, (int, float)):
                perturb_tensor = torch.tensor([self.perturbation_std], device=x.device)
            else:
                perturb_tensor = self.perturbation_std.to(x.device)

            if isinstance(original_amplitude, (int, float)):
                orig_amp_tensor = torch.tensor([original_amplitude], device=x.device)
            else:
                orig_amp_tensor = torch.from_numpy(original_amplitude).to(x.device)

            # Apply scaling with proper broadcasting
            scaling_factor = (perturb_tensor / orig_amp_tensor).view(1, -1, 1, 1, 1)
            white_noise = white_noise * scaling_factor
        else:
            # Use noise generator directly
            white_noise = self.noise_generator(x)

        # Apply temporal correlation
        if forecast_step == 1 or previous_perturbation is None:
            # Initial perturbation: just the white noise innovation
            current_perturbation = white_noise
        else:
            # AR(1) process: δ_t = ρ * δ_{t-1} + ε_t
            current_perturbation = self.temporal_correlation * previous_perturbation + white_noise

        if self.hemispheric_rescale is not None:
            current_perturbation = self.hemispheric_rescale(current_perturbation, self.latitudes.to(current_perturbation.device))

        # Apply perturbation to input
        perturbed_state = x + current_perturbation

        return perturbed_state, current_perturbation
