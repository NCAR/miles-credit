import torch
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

    Args:
        noise_generator (Noise): Noise generator instance used to create the white
            noise innovations (ε_t).
        temporal_correlation (float, optional): Temporal correlation coefficient in
            the range [0, 1]. Higher values create smoother temporal evolution.
            Defaults to 0.9.
        perturbation_std (Union[float, torch.Tensor], optional): Noise standard
            deviation scaling. Can be either:
            * float: uniform scaling applied to all channels
            * torch.Tensor: per-channel scaling with shape matching the channel
                dimension
            If provided, overrides the amplitude from the noise_generator.
            Defaults to None.
    """

    def __init__(
        self,
        noise_generator: torch.Tensor,
        temporal_correlation: float = 0.9,
        perturbation_std: Optional[Union[float, torch.Tensor]] = None,
        hemispheric_rescale: Optional[bool] = False,
    ):
        self.noise_generator = noise_generator
        self.temporal_correlation = temporal_correlation
        self.perturbation_std = perturbation_std
        self.hemispheric_rescale = hemi_rescale if hemispheric_rescale else False

    def __call__(
        self, x: torch.Tensor, previous_perturbation: Optional[torch.Tensor] = None, forecast_step: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply temporally correlated perturbation for sequential forecasting.

        Args:
            x (torch.Tensor): Input state tensor to perturb.
            previous_perturbation (torch.Tensor, optional): Perturbation from the
                previous forecast step. If None or if forecast_step == 1, generates a
                new initial perturbation. Defaults to None.
            forecast_step (int, optional): Current forecast step (1-indexed).
                Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:  
                * Perturbed state tensor (x + perturbation).  
                * Current perturbation tensor (for use in the next step).  
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
            latitudes = torch.linspace(90, -90, current_perturbation.shape[-2], device=current_perturbation.device)
            current_perturbation = self.hemispheric_rescale(current_perturbation, latitudes)

        # Apply perturbation to input
        perturbed_state = x + current_perturbation

        return perturbed_state, current_perturbation
