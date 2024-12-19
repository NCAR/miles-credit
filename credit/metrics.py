import torch
import numpy as np
from credit.loss import latitude_weights


class LatWeightedMetrics:
    def __init__(self, conf, predict_mode=False):
        self.conf = conf
        self.predict_mode = predict_mode
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

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None

        self.ensemble_size = conf["trainer"]["ensemble_size"]

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # calculate ensemble mean, if ensemble_size=1, does nothing
        pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:]) #b, ensemble, c, t, lat, lon
        pred = pred.mean(dim=1)

        # Get latitude and variable weights
        w_lat = (
            self.w_lat.to(dtype=pred.dtype, device=pred.device)
            if self.w_lat is not None
            else 1.0
        )
        w_var = (
            self.w_var.to(dtype=pred.dtype, device=pred.device)
            if self.w_var is not None
            else 1.0
        )

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = pred - y
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = (
                    torch.sqrt(
                        torch.sum(w_var * w_lat * pred_prime**2)
                        * torch.sum(w_var * w_lat * y_prime**2)
                    )
                    + epsilon
                )

                loss_dict[f"acc_{var}"] = (
                    torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                )
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(
                        torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))
                    )
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (
                    torch.abs(error[:, i]) * w_lat * w_var
                ).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean(
            [loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k]
        )
        loss_dict["rmse"] = np.mean(
            [loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k]
        )
        loss_dict["mse"] = np.mean(
            [
                loss_dict[k].cpu()
                for k in loss_dict.keys()
                if "mse_" in k and "rmse_" not in k
            ]
        )
        loss_dict["mae"] = np.mean(
            [loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k]
        )

        return loss_dict
