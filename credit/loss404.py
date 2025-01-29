import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import numpy as np
import logging


logger = logging.getLogger(__name__)


def load_loss(loss_type, reduction="mean"):
    if loss_type == "mse":
        return torch.nn.MSELoss(reduction=reduction)
    if loss_type == "msle":
        return MSLELoss(reduction=reduction)
    if loss_type == "mae":
        return torch.nn.L1Loss(reduction=reduction)
    if loss_type == "huber":
        return torch.nn.SmoothL1Loss(reduction=reduction)
    if loss_type == "logcosh":
        return LogCoshLoss(reduction=reduction)
    if loss_type == "xtanh":
        return XTanhLoss(reduction=reduction)
    if loss_type == "xsigmoid":
        return XSigmoidLoss(reduction=reduction)


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return (
            torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
            if self.reduction == "mean"
            else torch.log(torch.cosh(ey_t + 1e-12))
        )


class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return (
            torch.mean(ey_t * torch.tanh(ey_t))
            if self.reduction == "mean"
            else ey_t * torch.tanh(ey_t)
        )


class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t + 1e-12
        return (
            torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
            if self.reduction == "mean"
            else 2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t
        )


class MSLELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(MSLELoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        log_prediction = torch.log(
            prediction.abs() + 1
        )  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target, reduction=self.reduction)
        return loss


class SpectralLoss2D(torch.nn.Module):
    def __init__(self, wavenum_init=20, reduction="none"):
        super(SpectralLoss2D, self).__init__()
        self.wavenum_init = wavenum_init
        self.reduction = reduction

    def forward(self, output, target, weights=None, fft_dim=-1):
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
        loss = torch.square(
            out_fft_mean[..., self.wavenum_init :]
            - target_fft_mean[..., self.wavenum_init :]
        )
        loss = loss.mean()
        return loss.to(device=device, dtype=dtype)


class PSDLoss(nn.Module):
    def __init__(self, wavenum_init=20):
        super(PSDLoss, self).__init__()
        self.wavenum_init = wavenum_init

    def forward(self, target, pred, weights=None):
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
            weights = torch.full((1, lat_shape), 1 / lat_shape, dtype=torch.float32).to(
                device=device, dtype=dtype
            )
        else:
            weights = (
                weights.permute(0, 2, 1).to(device=device, dtype=dtype) / weights.sum()
            )
            # (1, lat, 1) -> (1, 1, lat)
        # (B, C, t, lat, coeffs)
        sq_diff = (
            true_psd_log[..., self.wavenum_init :]
            - pred_psd_log[..., self.wavenum_init :]
        ) ** 2

        loss = torch.mean(torch.matmul(weights, sq_diff))
        # (B, C, t, lat, coeffs) -> (B, C, t, 1, coeffs) -> ()
        return loss.to(device=device, dtype=dtype)

    def get_psd(self, f_x, device, dtype):
        # (B, C, t, lat, lon)
        f_k = torch.fft.rfft(f_x, dim=-1, norm="forward")
        mult_by_two = torch.full(f_k.shape[-1:], 2.0, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        mult_by_two[0] = 1.0  # except first coord
        magnitudes = torch.real(f_k * torch.conj(f_k)) * mult_by_two
        # (B, C, t, lat, coeffs)
        return magnitudes


def latititude_weights(conf):
    cos_lat = xr.open_dataset(conf["loss"]["latitude_weights"])["coslat"].values
    # Normalize over lat
    cos_lat_sum = cos_lat.sum(axis=0) / cos_lat.shape[0]
    L = cos_lat / cos_lat_sum
    return torch.from_numpy(L).float()


#     # Compute the latitude-weighting factor for each row
#     L = cos_lat / cos_lat_sum
#     L = L / L.sum()

#     min_val = np.min(L) // 2
#     max_val = np.max(L)
#     normalized_L = (L - min_val) / (max_val - min_val)

#     return torch.from_numpy(normalized_L).float()


def variable_weights(conf, channels, surface_channels, frames):
    # Load weights for U, V, T, Q
    weights_UVTQ = torch.tensor(
        [
            conf["loss"]["variable_weights"]["U"],
            conf["loss"]["variable_weights"]["V"],
            conf["loss"]["variable_weights"]["T"],
            conf["loss"]["variable_weights"]["Q"],
        ]
    ).view(1, channels * frames, 1, 1)

    # Load weights for SP, t2m, V500, U500, T500, Z500, Q500
    weights_sfc = torch.tensor(
        [
            conf["loss"]["variable_weights"]["SP"],
            conf["loss"]["variable_weights"]["t2m"],
            conf["loss"]["variable_weights"]["V500"],
            conf["loss"]["variable_weights"]["U500"],
            conf["loss"]["variable_weights"]["T500"],
            conf["loss"]["variable_weights"]["Z500"],
            conf["loss"]["variable_weights"]["Q500"],
        ]
    ).view(1, surface_channels, 1, 1)

    # Combine all weights along the color channel
    variable_weights = torch.cat([weights_UVTQ, weights_sfc], dim=1)

    return variable_weights


class VariableTotalLoss2D(torch.nn.Module):
    def __init__(self, conf, validation=False):
        super(VariableTotalLoss2D, self).__init__()

        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]

        lat_file = conf["loss"]["latitude_weights"]
        atmos_vars = conf["data"]["variables"]

        self.vars = atmos_vars

        self.lat_weights = None
        if conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(lat_file)["latitude"].values
            w_lat = np.cos(np.deg2rad(lat))
            w_lat = w_lat / w_lat.mean()
            self.lat_weights = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1)

        self.var_weights = None
        if conf["loss"]["use_variable_weights"]:
            var_weights = [
                value if isinstance(value, list) else [value]
                for value in conf["loss"]["variable_weights"].values()
            ]
            var_weights = [item for sublist in var_weights for item in sublist]
            self.var_weights = torch.from_numpy(var_weights).unsqueeze(0).unsqueeze(-1)

        self.use_spectral_loss = conf["loss"]["use_spectral_loss"]
        if self.use_spectral_loss:
            self.spectral_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.spectral_loss_surface = SpectralLoss2D(
                wavenum_init=conf["loss"]["spectral_wavenum_init"], reduction="none"
            )

        self.use_power_loss = (
            conf["loss"]["use_power_loss"]
            if "use_power_loss" in conf["loss"]
            else False
        )
        if self.use_power_loss:
            self.power_lambda_reg = conf["loss"]["spectral_lambda_reg"]
            self.power_loss = PSDLoss(
                wavenum_init=conf["loss"]["spectral_wavenum_init"]
            )

        self.validation = validation
        if self.validation:
            self.loss_fn = nn.L1Loss(reduction="none")
        else:
            self.loss_fn = load_loss(self.training_loss, reduction="none")

    def forward(self, target, pred):
        # User defined loss
        loss = self.loss_fn(target, pred)

        loss_dict = {}
        for i, var in enumerate(self.vars):
            loss_dict[f"loss_{var}"] = loss[:, i]

            if self.lat_weights is not None:
                loss_dict[f"loss_{var}"] = torch.mul(
                    loss_dict[f"loss_{var}"], self.lat_weights.to(target.device)
                )
            if self.var_weights is not None:
                loss_dict[f"loss_{var}"] = torch.mul(
                    loss_dict[f"loss_{var}"], self.var_weights.to(target.device)
                )

            loss_dict[f"loss_{var}"] = loss_dict[f"loss_{var}"].mean()

        loss = torch.mean(torch.stack([loss for loss in loss_dict.values()]))

        # Add the spectral loss
        if not self.validation and self.use_power_loss:
            loss += self.power_lambda_reg * self.power_loss(
                target, pred, weights=self.lat_weights
            )

        if not self.validation and self.use_spectral_loss:
            loss += (
                self.spectral_lambda_reg
                * self.spectral_loss_surface(
                    target, pred, weights=self.lat_weights
                ).mean()
            )

        return loss
