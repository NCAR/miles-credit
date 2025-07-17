import torch


class XTanhLoss(torch.nn.Module):
    """X-Tanh Loss Function.

    This loss function computes the element-wise product of the prediction error
    and the hyperbolic tangent of the error. This loss function aims to be more
    robust to outliers than traditional MSE.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' | 'none'. 'mean': the output is averaged; 'none': no reduction is applied.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        """Forward pass for X-Tanh loss.

        Args:
            y_t (torch.Tensor): Target tensor.
            y_prime_t (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: X-Tanh loss value.
        """
        ey_t = y_t - y_prime_t + 1e-12
        return torch.mean(ey_t * torch.tanh(ey_t)) if self.reduction == "mean" else ey_t * torch.tanh(ey_t)
