import torch


class XSigmoidLoss(torch.nn.Module):
    """X-Sigmoid Loss Function.

    This loss function computes a modified loss by using a sigmoid function
    transformation. It is designed to handle large errors in a non-linear fashion.

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'mean' | 'none'. 'mean': the output is averaged; 'none': no reduction is applied.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        """Forward pass for X-Sigmoid loss.

        Args:
            y_t (torch.Tensor): Target tensor.
            y_prime_t (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: X-Sigmoid loss value.
        """
        ey_t = y_t - y_prime_t + 1e-12
        return (
            torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
            if self.reduction == "mean"
            else 2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t
        )
