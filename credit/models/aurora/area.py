"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Utilities for computing the area of spherical patches.
"""

import torch

__all__ = ["area", "radius_earth"]

radius_earth: float = 6371.0
"""float: Radius of the Earth in km."""


def area(vertices: torch.Tensor) -> torch.Tensor:
    """Compute the area of a spherical polygon in km².

    Uses the Gauss shoelace formula on the sphere:
        A = R² × |Σᵢ (λᵢ₊₁ - λᵢ₋₁) × sin(φᵢ)| / 2

    where φ is latitude and λ is longitude (both in radians).

    Args:
        vertices: ``(N, 2)`` tensor of ``[latitude, longitude]`` in **degrees**,
            ordered counter-clockwise.

    Returns:
        Scalar area in km².
    """
    lat = torch.deg2rad(vertices[:, 0].double())
    lon = torch.deg2rad(vertices[:, 1].double())
    n = lat.shape[0]

    total = torch.zeros(1, dtype=torch.float64, device=vertices.device)
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        total = total + (lon[next_i] - lon[prev_i]) * torch.sin(lat[i])

    return torch.abs(total) / 2.0 * (radius_earth**2)
