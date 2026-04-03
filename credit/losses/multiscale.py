import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMSELoss(nn.Module):
    """Multi-scale MSE loss.

    Computes MSE between avg-pooled predictions and targets at multiple
    spatial scales. Coarse-scale terms penalise large-scale structure errors
    that single-resolution MSE can ignore; the combination resists the
    smoothing / variance-collapse that pure pixel-wise MSE encourages.

    scales  : spatial downsampling factors, e.g. (2, 4)
    weights : per-scale loss coefficients; defaults to [0.5, 0.25] (geometric)
    """

    def __init__(self, scales=(2, 4), weights=None, reduction="mean"):
        super().__init__()
        self.scales = list(scales)
        self.weights = weights if weights is not None else [1.0 / (2**i) for i in range(1, len(self.scales) + 1)]
        assert len(self.scales) == len(self.weights), "scales and weights must have the same length"

    def forward(self, pred, target):
        # Accept both (B, C, H, W) and (B, C, T, H, W)
        if pred.dim() == 5:
            B, C, T, H, W = pred.shape
            pred = pred.reshape(B, C * T, H, W)
            target = target.reshape(B, C * T, H, W)

        pred = pred.float()
        target = target.float()

        loss = pred.new_zeros(1).squeeze()
        for scale, w in zip(self.scales, self.weights):
            p = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            t = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            loss = loss + w * F.mse_loss(p, t)

        return loss
