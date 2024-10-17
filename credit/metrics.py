import torch
import numpy as np
from credit.loss import latitude_weights

# from credit.data_conversions import dataConverter
# from weatherbench2.derived_variables import ZonalEnergySpectrum
# WEC limiting spectrum shit cause weather bench2 not installed.


class LatWeightedMetrics:

    def __init__(self, conf, predict_mode=False):
        self.conf = conf
        self.predict_mode = predict_mode
        # lat_file = conf['loss']['latitude_weights']
        atmos_vars = conf['data']['variables']
        surface_vars = conf['data']['surface_variables']
        diag_vars = conf['data']['diagnostic_variables']
        
        levels = conf['model']['levels'] if 'levels' in conf['model'] else conf['model']['frames']

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars
        
        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None
        # if conf["loss"]["use_variable_weights"]:
        #     var_weights = [value if isinstance(value, list) else [value] for value in
        #                    conf["loss"]["variable_weights"].values()]
        #     var_weights = np.array([item for sublist in var_weights for item in sublist])
        #     self.w_var = torch.from_numpy(var_weights).unsqueeze(0).unsqueeze(-1)

        # if self.predict_mode:
        #    self.zonal_metrics = ZonalSpectrumMetric(self.conf)
        # WEC limiting spectrum shit cause weather bench2 not installed.

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self.w_lat.to(dtype=pred.dtype, device=pred.device) if self.w_lat is not None else 1.
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = (pred - y)
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = torch.sqrt(
                    torch.sum(w_var * w_lat * pred_prime ** 2) * torch.sum(w_var * w_lat * y_prime ** 2)
                ) + epsilon

                loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1)))
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (torch.abs(error[:, i]) * w_lat * w_var).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
        loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k])
        loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mse_" in k and "rmse_" not in k])
        loss_dict["mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k])

        # additional metrics where xarray computations are needed
        # put metric configs here
        # convert to xarray:
        # if self.predict_mode:
        #     self.converter = dataConverter(self.conf)
        #     pred_ds = self.converter.tensor_to_dataset(pred, [forecast_datetime])
        #     y_ds = self.converter.tensor_to_dataset(y, [forecast_datetime])
        #     loss_dict = loss_dict | self.zonal_metrics(pred_ds, y_ds)  # merge two dictionaries

        return loss_dict

# class ZonalSpectrumMetric:
#     def __init__(self, conf):
#         '''
#         _variables arguments determine which data vars to compute spectra metric
#         '''
#         self.conf = conf
#         self.x_variables =  self.conf['data']['variables']
#         self.single_level_variables = self.conf['data']['static_variables']
#         self.variables = self.x_variables + self.single_level_variables
#         self.zonal_spectrum_calculator = ZonalEnergySpectrum(self.variables)
#         if conf["loss"]["use_latitude_weights"]:
#             lat = xr.open_dataset(conf['loss']['latitude_weights'])["latitude"]
#             w_lat = np.cos(np.deg2rad(lat))
#             self.w_lat = w_lat / w_lat.mean()

#     def __call__(self, pred_ds, y_ds):
#         '''
#         pred, y can be normalized or unnormalized tensors.
#         trying to achieve minimal interface with LatWeightedMetrics
#         '''
#         # first dim is the batch dim
#         loss_dict = {}

#         # compute spectrum and add epsilon to avoid division by zero
#         epsilon = 1e-7
#         pred_spectrum = self.zonal_spectrum_calculator.compute(pred_ds) + epsilon
#         y_spectrum = self.zonal_spectrum_calculator.compute(y_ds) + epsilon
#         loss = self.lat_weighted_spectrum_diff(pred_spectrum, y_spectrum)
#         loss_dict = self.store_loss(loss, loss_dict, 'spectrum_mse')
#         return loss_dict

#     def store_loss(self, loss, loss_dict, metric_header_str):
#         '''
#         loss: dataset
#             w/ each variable dim must include level if atmos var,
#             ow no need for single level vars
#         sums over remaining dimensions, and writes to loss_dict
#         '''
#         keys = []
#         for v in self.x_variables:
#             for k in range(self.conf['model']['levels']):
#                 label = f"{metric_header_str}_{v}_{k}"
#                 keys.append(label)
#                 loss_dict[label] = loss[v].sel(level=k).sum()  #latitudes already weighted

#         for v in self.single_level_variables:
#             label = f"{metric_header_str}_{v}"
#             keys.append(label)
#             loss_dict[label] = loss[v].sum()  #latitudes already weighted

#         loss_dict[f"{metric_header_str}"] = np.mean([loss_dict[k] for k in keys])
#         return loss_dict

#     def lat_weighted_spectrum_diff(self, pred_spectrum, y_spectrum):
#         # using squared distance
#         # variables to compute spectra on determined by class init
#         sq_diff = np.square(np.log10(pred_spectrum) - np.log10(y_spectrum))
#         sq_diff = sq_diff.sum(dim=['datetime', 'zonal_wavenumber'])
#         return sq_diff * self.w_lat
