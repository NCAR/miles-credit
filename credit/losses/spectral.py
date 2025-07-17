import torch


class SpectralLoss2D(torch.nn.Module):
    """Spectral Loss in 2D.

    This loss function compares the spectral (frequency domain) content of the
    predicted and target outputs using FFT. It is useful for ensuring that the
    predicted output has similar frequency characteristics as the target.

    Args:
        wavenum_init (int): The initial wavenumber to start considering in the loss calculation.
        reduction (str): Specifies the reduction to apply to the output:
            'mean' | 'none'. 'mean': the output is averaged; 'none': no reduction is applied.
    """

    def __init__(self, wavenum_init=20, reduction="none"):
        super(SpectralLoss2D, self).__init__()
        self.wavenum_init = wavenum_init
        self.reduction = reduction

    def forward(self, output, target, weights=None, fft_dim=-1):
        """Forward pass for Spectral Loss 2D.

        Args:
            output (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.
            weights (torch.Tensor, optional): Latitude weights for the loss.
            fft_dim (int): The dimension to apply FFT.

        Returns:
            torch.Tensor: Spectral loss value.
        """
        # code is currently for (..., lat, lon)
        # todo: write for  (... lat, lon, ... )
        device, dtype = output.device, output.dtype
        output = output.float()
        target = target.float()

        # Take FFT over the 'lon' dimension
        # (B, c, lat, lon)
        # reduced fft to save memory, only computes up to nyquist freq. dims will always match
        out_fft = torch.fft.rfft(output, dim=fft_dim)
        target_fft = torch.fft.rfft(target, dim=fft_dim)
        # (B, c, lat, wavenum)

        # Take absolute value
        out_fft_abs = torch.abs(out_fft)
        target_fft_abs = torch.abs(target_fft)

        if weights is not None:
            # weights.shape = (1, lat, 1)
            weights = weights.permute(0, 2, 1).to(device=device, dtype=dtype)
            # (1, 1, lat), matmul will broadcast as long as last dim is lat
            out_fft_abs = torch.matmul(weights, out_fft_abs)
            target_fft_abs = torch.matmul(weights, target_fft_abs)
            # matmul broadcasts over dims except last two, where it does a matrix mult
            # (1, 1, 1, lat) x (B, c, T, lat, wavenum)
            # does multiplication on submatrices (2d tensors) defined by last two dims
            # result: (B, c, T, 1, wavenum), weighted sum over all wavenums
            # would probably be clearer to rewrite this using torch.mean and weight vector

            # to get average, need to normalize by the norm of the lat weights
            # so divide everything by |lat| to get a true average
            # then remove lat dim, since its now averaged
            out_fft_mean = (out_fft_abs / weights.shape[-1]).squeeze(fft_dim - 1)
            target_fft_mean = (target_fft_abs / weights.shape[-1]).squeeze(fft_dim - 1)
            # (B, c, T, wavenum)
        else:  # do regular average over latitudes
            out_fft_mean = torch.mean(out_fft_abs, dim=(fft_dim - 1))
            target_fft_mean = torch.mean(target_fft_abs, dim=(fft_dim - 1))

        # Compute MSE, no sqrt according to FouRKS paper/ repo
        loss = torch.square(out_fft_mean[..., self.wavenum_init :] - target_fft_mean[..., self.wavenum_init :])
        loss = loss.mean()
        return loss.to(device=device, dtype=dtype)
