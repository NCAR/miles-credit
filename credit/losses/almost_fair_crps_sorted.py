import torch

class AlmostFairKCRPSLoss(torch.nn.Module):
    """
    Calculates CRPS loss that has corrections for very small ensembles.
    Uses O(M log M) sorting to avoid O(M^2) memory explosion.
    """
    def __init__(self, alpha=1.0, reduction="mean", no_autocast=True):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.no_autocast = no_autocast

    def forward(self, target, pred):
        # Compute ensemble size: pred shape = (batch * ensemble, ...)
        # target shape = (batch, ...)
        b = target.shape[0]
        m = pred.shape[0] // b
        
        # Reshape pred to separate batch and ensemble dimensions
        # Shape: (batch, ensemble, c, t, lat, lon)
        pred = pred.view(b, m, *target.shape[1:])
        
        # 1. Skill (Mean Absolute Error of ensemble vs target)
        # Unsqueeze target to broadcast over the ensemble dimension
        target_expanded = target.unsqueeze(1)
        skill = torch.abs(pred - target_expanded).mean(dim=1)
        
        # 2. Spread (Pairwise differences) using the Sorting Trick
        # Sort the ensemble dimension (dim=1)
        pred_sorted, _ = torch.sort(pred, dim=1)
        
        # Create mathematically exact weights: 2i - m + 1 (where i is 0-indexed)
        i = torch.arange(m, device=pred.device, dtype=pred.dtype)
        weights = 2 * i - m + 1
        
        # Reshape weights to strictly match dimensions for high-speed broadcasting
        # This dynamically builds [1, m, 1, 1, 1...] depending on input tensor depth
        view_shape = [1, m] + [1] * (pred.ndim - 2)
        weights = weights.view(*view_shape)
        
        # Calculate sum of absolute pairwise differences
        sum_diffs = 2 * torch.sum(pred_sorted * weights, dim=1)
        
        # Calculate fair spread
        spread = (1.0 / (2 * m * (m - 1))) * sum_diffs
        
        # 3. Final afCRPS calculation
        epsilon = (1.0 - self.alpha) / m
        crps = skill - (1.0 - epsilon) * spread

        # 4. Reduction
        if self.reduction == "mean":
            return torch.mean(crps)
        elif self.reduction == "sum":
            return torch.sum(crps)
        elif self.reduction == "none":
            return crps
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")