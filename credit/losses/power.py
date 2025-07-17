import torch


class PSDLoss(torch.nn.Module):
    """Power Spectral Density (PSD) Loss Function.

    This loss function calculates the Power Spectral Density (PSD) of the
    predicted and target outputs and compares them to ensure similar frequency
    content in the predictions.

    Args:
        wavenum_init (int): The initial wavenumber to start considering in the loss calculation.
    """

    def __init__(self, wavenum_init=20):
        super(PSDLoss, self).__init__()
        self.wavenum_init = wavenum_init

    def forward(self, target, pred, weights=None):
        """Forward pass for PSD loss.

        Args:
            target (torch.Tensor): Target tensor.
            pred (torch.Tensor): Predicted tensor.
            weights (torch.Tensor, optional): Latitude weights for the loss.

        Returns:
            torch.Tensor: PSD loss value.
        """
        # weights.shape = (1, lat, 1)
        device, dtype = pred.device, pred.dtype
        target = target.float()
        pred = pred.float()

        # Calculate power spectra for true and predicted data
        true_psd = self.get_psd(target, device, dtype)
        pred_psd = self.get_psd(pred, device, dtype)

        # Logarithm transformation to normalize magnitudes
        # Adding epsilon to avoid log(0)
        true_psd_log = torch.log(true_psd + 1e-8)
        pred_psd_log = torch.log(pred_psd + 1e-8)

        # Calculate mean of squared distance weighted by latitude
        lat_shape = pred_psd.shape[-2]
        if weights is None:  # weights for a normal average
            weights = torch.full((1, lat_shape), 1 / lat_shape, dtype=torch.float32).to(device=device, dtype=dtype)
        else:
            weights = weights.permute(0, 2, 1).to(device=device, dtype=dtype) / weights.sum()
            # (1, lat, 1) -> (1, 1, lat)
        # (B, C, t, lat, coeffs)
        sq_diff = (true_psd_log[..., self.wavenum_init :] - pred_psd_log[..., self.wavenum_init :]) ** 2

        loss = torch.mean(torch.matmul(weights, sq_diff))
        # (B, C, t, lat, coeffs) -> (B, C, t, 1, coeffs) -> ()
        return loss.to(device=device, dtype=dtype)

    def get_psd(self, f_x, device, dtype):
        # (B, C, t, lat, lon)
        f_k = torch.fft.rfft(f_x, dim=-1, norm="forward")
        mult_by_two = torch.full(f_k.shape[-1:], 2.0, dtype=torch.float32).to(device=device, dtype=dtype)
        mult_by_two[0] = 1.0  # except first coord
        magnitudes = torch.real(f_k * torch.conj(f_k)) * mult_by_two
        # (B, C, t, lat, coeffs)
        return magnitudes
