import torch


class AlmostFairKCRPSLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, reduction="mean", no_autocast=True):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.no_autocast = no_autocast
        self.batched_forward = torch.vmap(self.single_sample_forward)

    def forward(self, target, pred):
        # Compute ensemble size assuming pred shape = (batch * ensemble, ...)
        ensemble_size = pred.shape[0] // target.shape[0] + pred.shape[0] % target.shape[0]
        pred = pred.view(target.shape[0], ensemble_size, *target.shape[1:])  # b, ensemble, c, t, lat, lon
        target = target.unsqueeze(1)  # b, 1, c, t, lat, lon

        # Apply single_sample_forward to each batch entry
        crps = self.batched_forward(target, pred).squeeze(1)

        if self.reduction == "mean":
            return torch.mean(crps)
        elif self.reduction == "sum":
            return torch.sum(crps)
        elif self.reduction == "none":
            return crps
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

    def single_sample_forward(self, target, pred):
        """
        Args:
            target: shape (1, c, t, lat, lon)
            pred: shape (ensemble, c, t, lat, lon)
        Returns:
            crps: shape (c, t, lat, lon)
        """
        # Move ensemble dim to the end: (c, t, lat, lon, ensemble)
        pred = torch.movedim(pred, 0, -1)
        target = target.squeeze(0)  # remove singleton batch dim

        return self._kernel_crps(pred, target, self.alpha)

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float):
        """
        Args:
            preds: (c, t, lat, lon, ensemble)
            targets: (c, t, lat, lon)
        Returns:
            crps: (c, t, lat, lon)
        """
        m = preds.shape[-1]
        assert m > 1, "Ensemble size must be greater than 1."

        epsilon = (1.0 - alpha) / m

        # |X_i - y|
        skill = torch.abs(preds - targets.unsqueeze(-1)).mean(-1)

        # |X_i - X_j|
        pred1 = preds.unsqueeze(-2)  # (c, t, lat, lon, 1, m)
        pred2 = preds.unsqueeze(-1)  # (c, t, lat, lon, m, 1)
        pairwise_diffs = torch.abs(pred1 - pred2)  # (c, t, lat, lon, m, m)

        # Create diagonal mask to exclude i == j
        eye = torch.eye(m, dtype=torch.bool, device=preds.device)
        mask = ~eye.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,1,m,m)
        pairwise_diffs = pairwise_diffs * mask

        spread = (1.0 / (2 * m * (m - 1))) * torch.sum(pairwise_diffs, dim=(-1, -2))

        return skill - (1 - epsilon) * spread
