import torch
import torch.nn as nn

# Import the pairwise implementation that safely handles complex numbers
from credit.losses.almost_fair_crps import AlmostFairKCRPSLoss

class CombinedSpatialSpectralAFCRPS(nn.Module):
    """
    Computes a combined Almost-Fair CRPS loss in both spatial and spectral domains.
    Loss = CRPS_spatial + (lambda_reg * CRPS_spectral)
    """
    def __init__(self, alpha=0.95, lambda_reg=0.1, reduction="mean"):
        super().__init__()
        self.lambda_reg = lambda_reg
        
        # Let the base class natively handle the reduction so it returns scalars!
        self.base_crps = AlmostFairKCRPSLoss(alpha=alpha, reduction=reduction, no_autocast=True)

    def forward(self, target, pred):
        
        # 1. Compute Spatial afCRPS (Returns a scalar if reduction="mean")
        loss_spatial = self.base_crps(target, pred)

        # 2. Transform to Spectral Space
        # norm="ortho" preserves variance/energy scale between domains
        pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1), norm="ortho")

        # 3. Compute Spectral afCRPS (Returns a scalar)
        loss_spectral = self.base_crps(target_fft, pred_fft)

        # 4. Combine and return
        return loss_spatial + (self.lambda_reg * loss_spectral)