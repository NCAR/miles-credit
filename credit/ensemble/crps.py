import torch
import numpy as np
from properscoring import crps_ensemble


def calculate_crps_per_channel(
    ensemble_predictions: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Calculate CRPS score for each channel.

    Args:
        ensemble_predictions: Tensor of shape [ensemble_size, 1, channels, 1, height, width]
        y_true: Tensor of shape [1, channels, 1, height, width]

    Returns:
        Tensor of CRPS scores [channels] where each score is averaged over the ensemble
    """
    # Get dimensions
    ensemble_size = ensemble_predictions.shape[0]
    num_channels = y_true.shape[1]  # Number of channels from y_true

    # Remove unnecessary dimensions and rearrange
    # Remove the extra dimension after ensemble_size and the singleton dimensions
    ensemble_predictions = ensemble_predictions.squeeze(1).squeeze(
        3
    )  # Shape: [ensemble_size, channels, height, width]
    y_true = y_true.squeeze(0).squeeze(2)  # Shape: [channels, height, width]

    # Initialize CRPS scores tensor
    crps_scores = torch.zeros(num_channels, device=y_true.device)

    # Calculate CRPS for each channel
    for channel in range(num_channels):
        # Extract data for current channel
        y_true_channel = y_true[channel].cpu().numpy()  # Shape: [height, width]
        ensemble_predictions_channel = ensemble_predictions[:, channel].cpu().numpy()
        # Shape: [ensemble_size, height, width]

        # Reshape to match properscoring requirements:
        # y_true needs shape [n_points]
        # ensemble_predictions needs shape [n_points, n_ensemble]
        y_true_flat = y_true_channel.reshape(-1)  # Shape: [height*width]
        ensemble_predictions_flat = ensemble_predictions_channel.reshape(
            ensemble_size, -1
        ).T
        # Shape: [height*width, ensemble_size]

        # Calculate CRPS for current channel
        channel_crps = crps_ensemble(
            observations=y_true_flat, forecasts=ensemble_predictions_flat
        )

        # Average over spatial dimensions
        crps_scores[channel] = torch.tensor(np.mean(channel_crps), device=y_true.device)

    return crps_scores.unsqueeze(0)
