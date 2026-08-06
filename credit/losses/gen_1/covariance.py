import torch
from torch import nn


def passthrough(in_val):
    return in_val


reduction_functions = {
    "mean": torch.mean,
    "sum": torch.sum,
    "min": torch.min,
    "max": torch.max,
    "none": passthrough,
}


class CovarianceWeightedMSELoss(nn.Module):
    """
    Mean Squared Error weighted by the error covariance matrix across variables, levels and output times.
    Assumes input Tensors have shape (batch, variable, time, lat, lon).

    Args:
        reduction (str): one of mean, none, sum, min, max
        batch_normalize (bool): If true, normalize each variable by the y_true batch means and standard devs.
    """

    def __init__(
        self,
        reduction: str = "mean",
        batch_normalize: bool = False,
        off_diagonal_scale: float = 1.0,
        **kwargs,
    ):
        self.reduction = reduction
        self.reduction_function = reduction_functions[reduction]
        self.batch_normalize = batch_normalize
        self.off_diagonal_scale = off_diagonal_scale
        super(CovarianceWeightedMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Assumes an initial shape based on (batch, variable, time, lat, lon)
        # First reorder dimensions to (variable, time, batch, lat, lon)
        assert len(y_true.shape) == 5, "y_true # dimensions does not match 5: (batch, var, time, lat, lon)"
        assert len(y_pred.shape) == 5, "y_pred # dimensions does no  match 5: (batch, var, time, lat, lon)"
        new_order = (1, 2, 0, 3, 4)
        y_true_shuff = torch.permute(y_true, new_order)
        y_pred_shuff = torch.permute(y_pred, new_order)
        new_shape = (
            y_true_shuff.shape[0] * y_true_shuff.shape[1],
            y_true_shuff.shape[2] * y_true_shuff.shape[3] * y_true_shuff.shape[4],
        )
        y_true_2d = torch.reshape(y_true_shuff, new_shape)
        y_pred_2d = torch.reshape(y_pred_shuff, new_shape)
        if self.batch_normalize:
            yt_mean = torch.unsqueeze(torch.mean(y_true_2d, 1), 1)
            yt_std = torch.unsqueeze(torch.std(y_true_2d, 1), 1)
            y_true_2d = (y_true_2d - yt_mean) / yt_std
            y_pred_2d = (y_pred_2d - yt_mean) / yt_std
        residual = y_true_2d - y_pred_2d
        cov = torch.cov(residual)
        precision = torch.linalg.inv(cov)
        diag_idx = torch.eye(precision.shape[0], dtype=torch.float32, device=precision.device)
        off_diag_idx = 1.0 - diag_idx
        precision = precision * off_diag_idx * self.off_diagonal_scale + precision * diag_idx

        def point_mse(res_row, prec=precision):
            res_long = torch.unsqueeze(res_row, 1)
            return res_long.T @ prec @ res_long

        col_mse = torch.vmap(point_mse)
        loss_vals = col_mse(residual.T)
        return self.reduction_function(loss_vals)
