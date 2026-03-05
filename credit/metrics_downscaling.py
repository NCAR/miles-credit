import torch
import numpy as np
# from datetime import datetime
# from credit.loss import latitude_weights


# class LatWeightedMetrics:
class UnWeightedMetrics:
    """Metrics calculation for training & validation; takes the place
    of LatWeightedMetrics, which does not apply for regional datasets
    on projected grids.  Differs in the organization of
    variables; otherwise, reuses the same code.

    Args:
    conf (dict): configuration dictionary

    varnames: list of variable names corresponding to channels of
              output tensor.  Use DownscalingDataset.tnames, which
              constructs it automatically.
    """

    def __init__(self, conf, varnames):
        self.vars = varnames
        # self.conf = conf
        # atmos_vars = conf["data"]["variables"]
        # surface_vars = conf["data"]["surface_variables"]
        # diag_vars = conf["data"]["diagnostic_variables"]
        #
        # levels = (
        #     conf["model"]["levels"]
        #     if "levels" in conf["model"]
        #     else conf["model"]["frames"]
        # )
        #
        # self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        # self.vars += surface_vars
        # self.vars += diag_vars

        # No latitude weighting
        self.w_lat = None
        # if conf["loss"]["use_latitude_weights"]:
        #     self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # variable weighting does not apply to metrics, only training loss (I think?)
        self.w_var = None

        # ensembles not yet supported for downcaling.
        # Todo: change training_mode from boolean to string in [train, test, validate]

        self.ensemble_size = 1
        # if training_mode:
        #     self.ensemble_size = conf["trainer"].get("ensemble_size", 1) # default value of 1 if not set
        # else:
        #     self.ensemble_size = conf["predict"].get("ensemble_size", 1)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # forecast_datetime is passed for interface consistency but not used here

        # pretty sure this never gets used, but we can leave it as-is
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.0
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            # calculate ensemble mean, if ensemble_size=1, does nothing
            if self.ensemble_size > 1:
                pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:])  # b, ensemble, c, t, lat, lon
                std_dev = torch.std(pred, dim=1)  # std dev of ensemble
                pred = pred.mean(dim=1)

            error = pred - y
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = (
                    torch.sqrt(
                        torch.sum(pred_prime**2) * torch.sum(y_prime**2)
                        # torch.sum(w_var * w_lat * pred_prime**2)
                        # * torch.sum(w_var * w_lat * y_prime**2)
                    )
                    + epsilon
                )

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1)))
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()
                # mean of std across all batches
                if self.ensemble_size > 1:
                    loss_dict[f"std_{var}"] = torch.mean(
                        torch.sqrt(torch.mean(std_dev[:, i] ** 2 * w_lat * w_var, dim=(-2, -1)))
                    )

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])
        if self.ensemble_size > 1:
            loss_dict["std"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "std_" in k])

        return loss_dict
