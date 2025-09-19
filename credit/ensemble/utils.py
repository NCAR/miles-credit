import torch
import torch.nn.functional as F


def add_spatially_correlated_noise(x, correlation_scale=10):
    """Add spatially correlated Gaussian noise using a convolutional filter.

    Applies correlated noise independently at each time step.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        correlation_scale (float): Standard deviation of the Gaussian kernel in pixels.

    Returns:
        torch.Tensor: Tensor of the same shape as input with added spatially
            correlated noise.
    """

    B, C, T, H, W = x.shape
    noise = torch.randn_like(x)

    # Create 2D Gaussian kernel
    size = int(6 * correlation_scale + 1)
    if size % 2 == 0:  # Ensure odd kernel size
        size += 1
    coords = torch.arange(size, device=x.device) - size // 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * correlation_scale**2))
    kernel /= kernel.sum()

    # Single kernel for all channels
    kernel = kernel.view(1, 1, size, size)

    # Apply convolution for each time step
    noise_out = torch.zeros_like(noise)
    for t in range(T):
        for c in range(C):  # Process each channel separately
            slice_noise = noise[:, c : c + 1, t, :, :].contiguous()  # Shape: (B, 1, H, W)
            smoothed = F.conv2d(slice_noise, kernel, padding=size // 2)
            noise_out[:, c, t, :, :] = smoothed.squeeze(1)

    return noise_out


def hemispheric_rescale(perturbation, latitudes, north_scale=1.0, south_scale=1.0):
    """Linearly interpolate scaling in the tropics between 20S and 20N.

    Args:
        latitudes (torch.Tensor): Tensor of shape (H,) with latitude values
            ranging from +90 to -90.
    """

    device = perturbation.device
    latitudes = latitudes.to(device)
    weights = torch.ones_like(latitudes)

    for i, lat in enumerate(latitudes):
        if lat >= 20:
            weights[i] = north_scale
        elif lat <= -20:
            weights[i] = south_scale
        else:
            frac = (lat + 20) / 40  # Maps [-20, 20] to [0, 1]
            weights[i] = (1 - frac) * south_scale + frac * north_scale

    weights = weights.view(1, 1, 1, -1, 1)
    return perturbation * weights
