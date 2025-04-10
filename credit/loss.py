import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import numpy as np
import logging


logger = logging.getLogger(__name__)


def load_loss(loss_type, reduction="mean"):
    """Load a specified loss function by its type.
    Helper function of VariableTotalLoss2D

    This function returns a loss function based on the specified
    `loss_type`. It supports several common loss functions, including
    MSE, MAE, MSLE, Huber, Log-Cosh, X-Tanh, and X-Sigmoid. The loss
    function can also be customized to use different reduction methods
    (e.g., 'mean', 'sum'). Use reduction=none if using latitude or variable
    weights

    Args:
        loss_type (str): The type of loss function to load. Supported
            values are "mse", "mae", "msle", "huber", "logcosh",
            "xtanh", and "xsigmoid".
        reduction (str, optional): Specifies the reduction to apply to
            the output: 'mean' (default) or 'sum'.

    Returns:
        torch.nn.Module: The corresponding loss function.

    Raises:
        ValueError: If the specified `loss_type` is not supported.

    Example:
        >>> loss_fn = load_loss("mse")
        >>> loss = loss_fn(pred, target)
    """
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "msle": MSLELoss,
        "huber": nn.HuberLoss,
        "logcosh": LogCoshLoss,
        "xtanh": XTanhLoss,
        "xsigmoid": XSigmoidLoss,
        "KCRPS": KCRPSLoss,
    }

    if loss_type in losses:
        logger.info(f"Loaded the {loss_type} loss function")
        return losses[loss_type](reduction=reduction)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported")


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
        return (
            torch.mean(ey_t * torch.tanh(ey_t))
            if self.reduction == "mean"
            else ey_t * torch.tanh(ey_t)
        )


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


class MSLELoss(nn.Module):
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
        log_prediction = torch.log(
            prediction.abs() + 1
        )  # Adding 1 to avoid logarithm of zero
        log_target = torch.log(target.abs() + 1)
        loss = F.mse_loss(log_prediction, log_target, reduction=self.reduction)
        return loss


class KCRPSLoss(nn.Module):
    """Adapted from Nvidia Modulus
    pred : Tensor
        Tensor containing the ensemble predictions. The ensemble dimension
        is assumed to be the leading dimension
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    biased :
        When False, uses the unbiased estimators described in (Zamo and Naveau, 2018)::

            E|X-y|/m - 1/(2m(m-1)) sum_(i,j=1)|x_i - x_j|

        Unlike ``crps`` this is fair for finite ensembles. Non-fair ``crps`` favors less
        dispersive ensembles since it is biased high by E|X- X'|/ m where m is the
        ensemble size.

    Estimate the CRPS from a finite ensemble

    Computes the local Continuous Ranked Probability Score (CRPS) by using
    the kernel version of CRPS. The cost is O(m log m).

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Approximates:

    .. math::
        CRPS(X, y) = E[X - y] - 0.5 E[X-X']

    with

    .. math::
        sum_i=1^m |X_i - y| / m - 1/(2m^2) sum_i,j=1^m |x_i - x_j|

    """

    def __init__(self, reduction, biased: bool = False):
        super().__init__()
        self.biased = biased
        self.batched_forward = torch.vmap(self.single_sample_forward)

    def forward(self, target, pred):
        # integer division but will error out next op if there is a remainder
        ensemble_size = (
            pred.shape[0] // target.shape[0] + pred.shape[0] % target.shape[0]
        )
        pred = pred.view(
            target.shape[0], ensemble_size, *target.shape[1:]
        )  # b, ensemble, c, t, lat, lon
        # apply single_sample_forward to each dim
        target = target.unsqueeze(1)
        return self.batched_forward(target, pred).squeeze(1)

    def single_sample_forward(self, target, pred):
        """
        Forward pass for KCRPS loss for a single sample.

        Args:
            target (torch.Tensor): Target tensor.
            pred (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: CRPS loss values at each lat/lon
        """
        pred = torch.movedim(pred, 0, -1)
        return self._kernel_crps_implementation(pred, target, self.biased)

    def _kernel_crps_implementation(
        self, pred: torch.Tensor, obs: torch.Tensor, biased: bool
    ) -> torch.Tensor:
        """An O(m log m) implementation of the kernel CRPS formulas"""
        skill = torch.abs(pred - obs[..., None]).mean(-1)
        pred, _ = torch.sort(pred)

        # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
        # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
        #                           = 2 sum_(i <= j) |x_i -x_j|
        #                           = 2 sum_(i <= j) (x_j - x_i)
        #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
        #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
        #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
        m = pred.size(-1)
        i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
        denom = m * m if biased else m * (m - 1)
        factor = (2 * i - m - 1) / denom
        spread = torch.sum(factor * pred, dim=-1)
        return skill - spread


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
        loss = torch.square(
            out_fft_mean[..., self.wavenum_init :]
            - target_fft_mean[..., self.wavenum_init :]
        )
        loss = loss.mean()
        return loss.to(device=device, dtype=dtype)


class PSDLoss(nn.Module):
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


def latitude_weights(conf):
    """Calculate latitude-based weights for loss function.
    This function calculates weights based on latitude values
    to be used in loss functions for geospatial data. The weights
    are derived from the cosine of the latitude and normalized
    by their mean.

    Args:
        conf (dict): Configuration dictionary containing the
            path to the latitude weights file.

    Returns:
        torch.Tensor: A 2D tensor of weights with dimensions
            corresponding to latitude and longitude.
    """
    # Open the dataset and extract latitude and longitude information
    ds = xr.open_dataset(conf["loss"]["latitude_weights"])
    lat = torch.from_numpy(ds["latitude"].values).float()

    # Calculate weights using PyTorch operations
    weights = torch.cos(torch.deg2rad(lat))
    weights = weights / weights.mean()

    return weights


def variable_weights(conf, channels, frames):
    """Create variable-specific weights for different atmospheric
    and surface channels.

    This function loads weights for different atmospheric variables
    (e.g., U, V, T, Q) and surface variables (e.g., SP, t2m) from
    the configuration file. It then combines them into a single
    weight tensor for use in loss calculations.

    Args:
        conf (dict): Configuration dictionary containing the
            variable weights.
        channels (int): Number of channels for atmospheric variables.
        frames (int): Number of time frames.

    Returns:
        torch.Tensor: A tensor containing the combined weights for
            all variables.
    """
    # Load weights for U, V, T, Q
    varname_upper_air = conf["data"]["variables"]
    varname_surface = conf["data"]["surface_variables"]
    varname_diagnostics = conf["data"]["diagnostic_variables"]

    # surface + diag channels
    N_channels_single = len(varname_surface) + len(varname_diagnostics)

    weights_upper_air = torch.tensor(
        [conf["loss"]["variable_weights"][var] for var in varname_upper_air]
    ).view(1, channels * frames, 1, 1)

    weights_single = torch.tensor(
        [
            conf["loss"]["variable_weights"][var]
            for var in (varname_surface + varname_diagnostics)
        ]
    ).view(1, N_channels_single, 1, 1)

    # Combine all weights along the color channel
    var_weights = torch.cat([weights_upper_air, weights_single], dim=1)

    return var_weights


class VariableTotalLoss2D(torch.nn.Module):
    """Custom loss function class for 2D geospatial data
    with optional spectral and power loss components.

    This class defines a loss function that combines a base loss
    (e.g., L1, MSE) with optional spectral and power loss components
    for 2D geospatial data. The loss function can incorporate latitude
    and variable-specific weights.

    Args:
        conf (dict): Configuration dictionary containing loss
            function settings and weights.
        validation (bool, optional): If True, the loss function
            is used in validation mode. Defaults to False.
    """

    def __init__(self, conf, validation=False):
        """Initialize the VariableTotalLoss2D.

        Args:
            conf (str): path to config file.
            training_loss (str): Loss equation to use during training.
            vars (list): Atmospheric, surface, and diagnostic variable names.
            lat_weights (str): Path to upper latitude weights file.
            var_weights (bool): Whether to use variable weights during training.
            use_spectral_loss (bool): Whether to use spectral loss during training.
            
            varname_surface (list): List of surface variable names.
            varname_dyn_forcing (list): List of dynamic forcing variable names.
            varname_forcing (list): List of forcing variable names.
            varname_static (list): List of static variable names.
            varname_diagnostic (list): List of diagnostic variable names.
            filenames (list): List of filenames for upper air data.
            filename_surface (list, optional): List of filenames for surface data.
            filename_dyn_forcing (list, optional): List of filenames for dynamic forcing data.
            filename_forcing (str, optional): Filename for forcing data.
            filename_static (str, optional): Filename for static data.
            filename_diagnostic (list, optional): List of filenames for diagnostic data.
            history_len (int, optional): Length of the history sequence. Default is 2.
            forecast_len (int, optional): Length of the forecast sequence. Default is 0.
            transform (callable, optional): Transformation function to apply to the data.
            seed (int, optional): Random seed for reproducibility. Default is 42.
            skip_periods (int, optional): Number of periods to skip between samples.
            one_shot(bool, optional): Whether to return all states or just
                                    the final state of the training target. Default is None
            max_forecast_len (int, optional): Maximum length of the forecast sequence.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.

        Returns:
            sample (dict): A dictionary containing historical_ERA5_images,
                                                 target_ERA5_images,
                                                 datetime index, and additional information.

        """
        super(VariableTotalLoss2D, self).__init__()

        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = (
            conf["model"]["levels"]
            if "levels" in conf["model"]
            else conf["model"]["frames"]
        )

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.lat_weights = torch.tensor(1.0)
        if conf["loss"]["use_latitude_weights"]:
            logger.info("Using latitude weights in loss calculations")
            self.lat_weights = latitude_weights(conf)
            self.lat_weights = self.lat_weights.view(1, 1, 1, len(self.lat_weights), 1)

        # ------------------------------------------------------------- #
        # variable weights
        # order: upper air --> surface --> diagnostics
        self.var_weights = torch.tensor(1.0)
        if conf["loss"]["use_variable_weights"]:
            logger.info("Using variable weights in loss calculations")
            
            var_weights = []
            for key, value in conf["loss"]["variable_weights"].items():
                logger.info(f"extracting {key} weights")
                
                if key in ["U", "V", "T", "Q", "Qtot"] and not isinstance(value, list):
                    weights = [value] * levels
                else:
                    weights = value if isinstance(value, list) else [value] 
                var_weights += weights

            self.var_weights = torch.tensor(var_weights)
            logger.info(self.var_weights)
            
            num_missing_weights = len(self.vars) - len(self.var_weights)
            if num_missing_weights > 0:
                self.var_weights = torch.cat([self.var_weights, 
                                              torch.full((num_missing_weights, ), conf["loss"]["default_var_weight"])])

            assert len(self.var_weights) == len(self.vars)

            self.var_weights = self.var_weights.view(1, len(self.var_weights), 1, 1, 1)
        # ------------------------------------------------------------- #

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
        if conf["loss"]["training_loss"] == "KCRPS":  # for ensembles, load same loss for train and valid
            self.loss_fn = load_loss(self.training_loss, reduction="none")

        elif self.validation:
            self.loss_fn = nn.L1Loss(reduction="none")

        else:
            self.loss_fn = load_loss(self.training_loss, reduction="none")

    def forward(self, target, pred):
        """Calculate the total loss for the given target and prediction.

        This method computes the base loss between the target and prediction,
        applies latitude and variable weights, and optionally adds spectral
        and power loss components.

        Args:
            target (torch.Tensor): Ground truth tensor.
            pred (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: The computed loss value.
        
        """
        # User defined loss
        loss = self.loss_fn(target, pred)

        # Latitutde and variable weights
        loss = (loss 
                * self.lat_weights.to(target.device)
                * self.var_weights.to(target.device))

        loss = loss.mean()

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
