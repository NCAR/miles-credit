import torch
import torch.nn.functional as F


class MSLELoss(torch.nn.Module):
    """Mean Squared Logarithmic Error (MSLE) Loss Function.

    This loss function computes the mean squared logarithmic error between the
    predicted and target values. It is useful for handling targets that span
    several orders of magnitude.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'mean' | 'none'. 'mean': the output is averaged; 'none': no reduction is applied.
    """

    def __init__(self, reduction="mean"):
        super(MSLELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        """Forward pass for MSLE loss.

        Args:
            prediction (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: MSLE loss value.
        """
        log_prediction = torch.log(prediction.abs() + 1)  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target, reduction=self.reduction)
        return loss
