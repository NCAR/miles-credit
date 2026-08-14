import torch


class LogCoshLoss(torch.nn.Module):
    """Log-Cosh Loss Function.

    This loss function computes the logarithm of the hyperbolic cosine of the
    prediction error. It is less sensitive to outliers compared to the Mean
    Squared Error (MSE) loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'mean' | 'none'. 'mean': the output is averaged; 'none': no reduction is applied.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        """Forward pass for Log-Cosh loss.

        Args:
            y_t (torch.Tensor): Target tensor.
            y_prime_t (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: Log-Cosh loss value.
        """
        ey_t = y_t - y_prime_t
        return (
            torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
            if self.reduction == "mean"
            else torch.log(torch.cosh(ey_t + 1e-12))
        )
