import torch


def add_gaussian_noise(x, std=0.1, mean=0.0, N=1):
    """
    Adds Gaussian noise to the first portion of channels in a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T, lat, lon).
        std (float): Standard deviation of the Gaussian noise.
        mean (float): Mean of the Gaussian noise.
        N (int): Number of different noisy tensors to return.

    Returns:
        A list of torch.Tensor: Single noisy tensor if N=1, else a list of N noisy tensors.
    """
    num_channels = x.shape[1]  # Modify only first 71 channels or all if fewer
    noisy_tensors = []

    for _ in range(N):
        noise = (
            torch.randn_like(x[:, :num_channels, ...]) * std + mean
        )  # Generate noise for first `num_channels`
        x_noisy = x.clone()
        x_noisy[:, :num_channels, ...] += (
            noise  # Add noise only to first `num_channels`
        )
        noisy_tensors.append(x_noisy)

    return noisy_tensors if N > 1 else noisy_tensors[0]
