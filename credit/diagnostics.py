import os
from os.path import join, expandvars
import typing as t
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch

# from weatherbench2.derived_variables import ZonalEnergySpectrum
# WEC limiting spectrum shit cause weather bench2 not installed.

import logging

logger = logging.getLogger(__name__)


class Diagnostics:
    """
    program flow: this class, sets up necessary pipeline, converts to xarray and/or pressure levels, calls diagnostics
    """

    def __init__(self, conf, init_datetime, data_converter):
        """data_converter is a dataConverter object"""
        self.conf = conf
        self.converter = data_converter
        self.w_lat, self.w_var = self.get_weights()

        self.diagnostics = []
        self.plev_diagnostics = []
        diag_conf = self.conf["diagnostics"]
        if diag_conf["use_spectrum_vis"]:
            logger.info("computing spectrum visualizations")
            # save directory for spectra plots
            plot_save_loc = join(
                expandvars(self.conf["save_loc"]), f"forecasts/spectra_{init_datetime}/"
            )
            os.makedirs(plot_save_loc, exist_ok=True)

            spectrum_vis = ZonalSpectrumVis(
                self.conf, self.w_lat, self.w_var, plot_save_loc
            )
            self.diagnostics.append(spectrum_vis)

        if diag_conf["use_KE_diagnostics"]:  # only plotting summary spectra for KE
            logger.info("computing KE visualizations")
            plot_save_loc = join(
                expandvars(self.conf["save_loc"]), f"forecasts/ke_{init_datetime}/"
            )
            os.makedirs(plot_save_loc, exist_ok=True)
            os.makedirs(join(plot_save_loc, "ke_diff"), exist_ok=True)

            ke_vis = KE_Diagnostic(self.conf, self.w_lat, self.w_var, plot_save_loc)
            self.plev_diagnostics.append(ke_vis)

    def __call__(self, pred_ds, y_ds, fh):
        metric_dict = {}

        for diagnostic in self.diagnostics:  # non-plev diagnostics
            diagnostic(pred_ds, y_ds, fh)

        # guard clauses for plev computation
        if not self.plev_diagnostics:
            return {}
        if (
            self.conf["diagnostics"]["plev_summary_only"]
            and fh not in self.conf["diagnostics"]["summary_plot_fhs"]
        ):
            logger.info(f"skipping plev diagnostics for fh {fh}")
            return {}

        logger.info(f"computing plev diagnostics for fh {fh}")
        pred_pressure = self.converter.dataset_to_pressure_levels(pred_ds).compute()
        y_pressure = self.converter.dataset_to_pressure_levels(y_ds).compute()
        for diagnostic in self.plev_diagnostics:
            metric_dict = metric_dict | diagnostic(pred_pressure, y_pressure, fh)
        # only return metrics if not doing summarys
        return {} if self.conf["diagnostics"]["plev_summary_only"] else metric_dict

    def get_weights(self):
        """
        gets variable and latitude weights from latitude file.
        """
        w_lat = None
        if self.conf["loss"]["use_latitude_weights"]:
            lat = xr.open_dataset(self.conf["loss"]["latitude_weights"])["latitude"]
            w_lat = np.cos(np.deg2rad(lat))
            self.w_lat = w_lat / w_lat.mean()

        w_var = None
        if self.conf["loss"]["use_variable_weights"]:
            var_weights = [
                value if isinstance(value, list) else [value]
                for value in self.conf["loss"]["variable_weights"].values()
            ]
            var_weights = [item for sublist in var_weights for item in sublist]

        return w_lat, w_var


def calculate_KE(dataset):
    wind_squared = dataset.U**2 + dataset.V**2
    return -1 * 0.5 * wind_squared.integrate("plev")
    # negative needed because of doing integration 'backwards' could also reverse the plev coord


class KE_Diagnostic:
    def __init__(self, conf, w_lat, w_var, plot_save_loc):
        self.conf = conf
        self.w_lat = w_lat
        self.plot_save_loc = plot_save_loc

        self.summary_plot_fhs = conf["diagnostics"]["summary_plot_fhs"]
        for k, v in conf["diagnostics"]["ke_vis"].items():
            setattr(self, k, v)

        # if self.use_KE_spectrum_vis:
        #     self.zonal_spectrum_calculator = ZonalEnergySpectrum("KE")
        #     if self.summary_plot_fhs:
        #         self.KE_fig, self.KE_axs = plt.subplots(ncols=1, figsize=(5, 5))
        #         self.KE_axs = [self.KE_axs]

    def __call__(self, pred_ds, y_ds, fh):
        """
        pressure level datasets
        """
        pred_ke = calculate_KE(pred_ds).compute()
        y_ke = calculate_KE(y_ds).compute()

        if self.use_KE_spectrum_vis:
            self.KE_spectrum_vis(pred_ke, y_ke, fh)

        if self.use_KE_difference_vis:
            self.KE_difference_vis(pred_ke, y_ke, fh)

        metric_dict = self.avg_KE_metric(pred_ke, y_ke)
        return metric_dict

    def avg_KE_metric(self, pred_ke, y_ke):
        diff = np.abs(pred_ke - y_ke)
        weighted = diff.weighted(self.w_lat).mean()
        return {"avg_KE_difference": weighted.values}

    def KE_difference_vis(self, pred_ke, y_ke, fh):
        # ke_diff = pred_ke - y_ke

        # Plotting
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())

        # Plot data using colormesh
        # divnorm = colors.TwoSlopeNorm(vcenter=0.0)  # center cmap at 0
        datetime_str = np.datetime_as_string(
            pred_ke.datetime.values[0], unit="h", timezone="UTC"
        )
        ax.set_title(f"pred_ke - y_ke | fh={fh} {datetime_str}")
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines()

        # Add colorbar
        y_ke_total = y_ke.sum().values
        pred_ke_total = pred_ke.sum().values
        text = (
            f"Total ERA5 KE: {y_ke_total:.2e}; "
            f"Difference: {pred_ke_total - y_ke_total:.2e}; "
            f"abs diff: {np.abs(pred_ke - y_ke).sum().values:.2e}"
        )
        ax.annotate(
            text,
            xy=(0.5, 0),
            xytext=(0.5, -0.1),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="top",
        )

        # save figure
        filepath = join(self.plot_save_loc, f"ke_diff/ke_diff_{datetime_str}.pdf")
        fig.savefig(filepath, format="pdf")

    def get_avg_spectrum_ke(self, da):
        ds = xr.Dataset({"KE": da})
        ds_spectrum = self.zonal_spectrum_calculator.compute(ds)
        ds_spectrum = interpolate_spectral_frequencies(ds_spectrum, "zonal_wavenumber")
        ds_spectrum = ds_spectrum.weighted(self.w_lat).mean(dim="latitude")
        return ds_spectrum

    def KE_spectrum_vis(self, pred_ke, y_ke, fh):
        # plot on summary plot

        if int(fh) in self.summary_plot_fhs:  # plot some fhs onto a single plot
            avg_pred_spectrum = self.get_avg_spectrum_ke(pred_ke)
            avg_y_spectrum = self.get_avg_spectrum_ke(y_ke)

            fh_idx = self.summary_plot_fhs.index(fh)
            self.KE_fig, self.KE_axs = self.plot_avg_spectrum(
                avg_pred_spectrum,
                avg_y_spectrum,
                self.KE_fig,
                self.KE_axs,
                alpha=1 - fh_idx / len(self.summary_plot_fhs),
                label=f"fh={fh}",
            )
            # if fh == self.summary_plot_fhs[-1]:
            for ax in self.KE_axs:  # overwrite every time in case of crash
                ax.legend()
            self.KE_fig.savefig(join(self.plot_save_loc, f"ke_spectra_summary{fh}"))
            logger.info(f"saved summary plot to {join(self.plot_save_loc, 'ke_spectra_summary')}")

    def plot_avg_spectrum(self, avg_pred_spectrum, avg_y_spectrum,
                          fig, axs, alpha=1, label=None):
        # copied from spectrum diagnostic function
        for ax in axs:
            ax.set_yscale("log")
            ax.set_xscale("log")
        curr_ax = 0
        avg_pred_spectrum.plot(
            x="wavelength", ax=axs[curr_ax], color="r", alpha=alpha, label=label
        )
        avg_y_spectrum.plot(x="wavelength", ax=axs[curr_ax], color="0")

        axs[curr_ax].set_title("KE Spectrum")
        ticks = axs[curr_ax].get_xticks()  # rescale x axis to km
        axs[curr_ax].set_xticks(ticks, ticks / 1000)
        axs[curr_ax].autoscale_view()
        axs[curr_ax].set_xlabel("Wavelength (km)")
        axs[curr_ax].set_ylabel("Power")

        return fig, axs


class ZonalSpectrumVis:
    def __init__(self, conf, w_lat, w_var, plot_save_loc):
        """ """
        self.conf = conf
        self.w_lat = w_lat
        self.plot_save_loc = plot_save_loc
        self.summary_plot_fhs = conf["diagnostics"]["summary_plot_fhs"]

        # this replaces unpacking the dictionary (below)
        for k, v in conf["diagnostics"]["spectrum_vis"].items():
            setattr(self, k, v)
        # vis_conf = conf['diagnostics']['spectrum_vis']
        # self.atmos_variables = vis_conf['atmos_variables']
        # self.atmos_levels= vis_conf['atmos_levels']
        # self.single_level_variables = vis_conf['single_level_variables']

        self.zonal_spectrum_calculator = None
        # self.zonal_spectrum_calculator = ZonalEnergySpectrum(
        #     self.atmos_variables + self.single_level_variables
        # )

        self.ifs_levels = xr.open_dataset(
            "/glade/derecho/scratch/dkimpara/nwp_files/ifs_levels.nc"
        )
        # self.figsize = vis_conf['figsize']
        if len(self.figsize) == 0:
            self.figsize = self.figsize
        else:
            num_vars = len(self.atmos_variables) * len(self.atmos_levels) + len(
                self.single_level_variables
            )
            self.figsize = (5 * num_vars, 5)

        if self.summary_plot_fhs:
            self.summary_fig, self.summary_axs = plt.subplots(
                ncols=len(self.atmos_variables) * len(self.atmos_levels)
                + len(self.single_level_variables),
                figsize=self.figsize,
            )

    def __call__(self, pred_ds, y_ds, fh):
        """
        pred, y can be normalized or unnormalized tensors.
        """
        # compute spectrum and add epsilon to avoid division by zero
        epsilon = 1e-7
        pred_spectrum = self.zonal_spectrum_calculator.compute(pred_ds) + epsilon
        y_spectrum = self.zonal_spectrum_calculator.compute(y_ds) + epsilon

        # compute average zonal spectrum
        avg_pred_spectrum = self.get_avg_spectrum(pred_spectrum)
        avg_y_spectrum = self.get_avg_spectrum(y_spectrum)

        # visualize
        fig, axs = plt.subplots(
            ncols=len(self.atmos_variables) * len(self.atmos_levels)
            + len(self.single_level_variables),
            figsize=self.figsize,
        )
        datetime_str = np.datetime_as_string(
            pred_ds.datetime.values[0], unit="h", timezone="UTC"
        )

        fig.suptitle(f"t={datetime_str}, fh={fh}")
        fig, axs = self.plot_avg_spectrum(avg_pred_spectrum, avg_y_spectrum, fig, axs)
        fig.savefig(join(self.plot_save_loc, f"spectra_{datetime_str}"))

        # plot on summary plot

        if fh in self.summary_plot_fhs:  # plot some fhs onto a single plot
            fh_idx = self.summary_plot_fhs.index(fh)
            self.summary_fig, self.summary_axs = self.plot_avg_spectrum(
                avg_pred_spectrum,
                avg_y_spectrum,
                self.summary_fig,
                self.summary_axs,
                alpha=1 - fh_idx / len(self.summary_plot_fhs),
                label=f"fh={fh}",
            )
            for ax in self.summary_axs:  # overwrite every time in case of crash
                ax.legend()
            self.summary_fig.savefig(join(self.plot_save_loc, "spectra_summary"))
            logger.info(
                f"saved summary plot to {join(self.plot_save_loc, 'spectra_summary')}"
            )

    def plot_avg_spectrum(
        self, avg_pred_spectrum, avg_y_spectrum, fig, axs, alpha=1, label=None
    ):
        for ax in axs:
            ax.set_yscale("log")
            ax.set_xscale("log")

        curr_ax = 0
        for level in self.atmos_levels:
            for variable in self.atmos_variables:
                avg_pred_spectrum[variable].sel(level=level).plot(
                    x="wavelength", ax=axs[curr_ax], color="r", alpha=alpha, label=label
                )
                avg_y_spectrum[variable].sel(level=level).plot(
                    x="wavelength", ax=axs[curr_ax], color="0"
                )

                axs[curr_ax].set_title(
                    f"{variable} {self.ifs_levels.ref_hPa.sel(level=level).values}"
                )
                ticks = axs[curr_ax].get_xticks()  # rescale x axis to km
                axs[curr_ax].set_xticks(ticks, ticks / 1000)
                axs[curr_ax].autoscale_view()
                axs[curr_ax].set_xlabel("Wavelength (km)")
                axs[curr_ax].set_ylabel("Power")
                curr_ax += 1

        for variable in self.single_level_variables:
            avg_pred_spectrum[variable].plot(
                x="wavelength", ax=axs[curr_ax], color="r", alpha=alpha, label=label
            )
            avg_y_spectrum[variable].plot(x="wavelength", ax=axs[curr_ax], color="0")
            axs[curr_ax].set_title(variable)

            ticks = axs[curr_ax].get_xticks()  # rescale x axis to km
            axs[curr_ax].set_xticks(ticks, ticks / 1000)
            axs[curr_ax].autoscale_view()
            axs[curr_ax].set_xlabel("Wavelength (km)")
            axs[curr_ax].set_ylabel("Power")
            curr_ax += 1

        return fig, axs

    def get_avg_spectrum(self, ds_spectrum):
        ds_spectrum = ds_spectrum.sel(level=self.atmos_levels)
        ds_spectrum = interpolate_spectral_frequencies(ds_spectrum, "zonal_wavenumber")
        ds_spectrum = ds_spectrum.weighted(self.w_lat).mean(dim="latitude")
        return ds_spectrum


# from weatherbench, slightly modified
def interpolate_spectral_frequencies(
    spectrum: xr.DataArray,
    wavenumber_dim: str,
    frequencies: t.Optional[t.Sequence[float]] = None,
    method: str = "linear",
    **interp_kwargs: t.Optional[dict[str, t.Any]],
) -> xr.DataArray:
    """Interpolate frequencies in `spectrum` to common values.

    Args:
    spectrum: Data as produced by ZonalEnergySpectrum.compute.
    wavenumber_dim: Dimension that indexes wavenumber, e.g. 'zonal_wavenumber'
      if `spectrum` is produced by ZonalEnergySpectrum.
    frequencies: Optional 1-D sequence of frequencies to interpolate to. By
      default, use the most narrow range of frequencies in `spectrum`.
    method: Interpolation method passed on to DataArray.interp.
    **interp_kwargs: Additional kwargs passed on to DataArray.interp.

    Returns:
    New DataArray with dimension "frequency" replacing the "wavenumber" dim in
      `spectrum`.
    """

    if set(spectrum.frequency.dims) != set((wavenumber_dim, "latitude")):
        raise ValueError(
            f"{spectrum.frequency.dims=} was not a permutation of "
            f'("{wavenumber_dim}", "latitude")'
        )

    if frequencies is None:
        freq_min = spectrum.frequency.max("latitude").min(wavenumber_dim).data
        freq_max = spectrum.frequency.min("latitude").max(wavenumber_dim).data
        frequencies = np.linspace(
            freq_min, freq_max, num=spectrum.sizes[wavenumber_dim]
        )
        frequencies = np.asarray(frequencies)
    if frequencies.ndim != 1:
        raise ValueError(f"Expected 1-D frequencies, found {frequencies.shape=}")

    def interp_at_one_lat(da: xr.DataArray) -> xr.DataArray:
        if len(da.latitude.values.shape) > 0:  # latitude weirdly not squeezed out by groupby sometimes
            da = da.squeeze("latitude")
        da = (
            da.swap_dims({wavenumber_dim: "frequency"})
            .drop_vars(wavenumber_dim)
            .interp(frequency=frequencies, method=method, **interp_kwargs)
        )
        # Interp didn't deal well with the infinite wavelength, so just reset λ as..
        da["wavelength"] = 1 / da.frequency
        da["wavelength"] = da["wavelength"].assign_attrs(units="m")
        return da

    return spectrum.groupby("latitude", squeeze=True).apply(interp_at_one_lat)


def anomaly_correlation_coefficient(pred, true):
    pred = pred.float()
    true = true.float()

    B, C, H, W = pred.size()

    # Flatten the spatial dimensions
    pred_flat = pred.view(B, C, -1)
    true_flat = true.view(B, C, -1)

    # Mean over spatial dimensions
    pred_mean = torch.mean(pred_flat, dim=-1, keepdim=True)
    true_mean = torch.mean(true_flat, dim=-1, keepdim=True)

    # Anomaly calculation
    pred_anomaly = pred_flat - pred_mean
    true_anomaly = true_flat - true_mean

    # Covariance matrix
    covariance_matrix = torch.bmm(pred_anomaly, true_anomaly.transpose(1, 2)) / (
        H * W - 1
    )

    # Variance terms
    pred_var = torch.bmm(pred_anomaly, pred_anomaly.transpose(1, 2)) / (H * W - 1)
    true_var = torch.bmm(true_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Anomaly Correlation Coefficient
    acc_numerator = torch.einsum("bii->b", covariance_matrix).sum()
    acc_denominator = torch.sqrt(
        torch.einsum("bii->b", pred_var).sum() * torch.einsum("bii->b", true_var).sum()
    )

    # Avoid division by zero
    epsilon = 1e-8
    acc = acc_numerator / (acc_denominator + epsilon)

    return acc.item()


if __name__ == "__main__":
    import yaml
    from credit.data_conversions import dataConverter
    import datetime

    test_dir = (
        "/glade/work/dkimpara/repos/global/miles-credit/results/test_files_quarter"
    )
    config = join(test_dir, "model.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    y_pred = torch.load(join(test_dir, "pred.pt"))
    y = torch.load(join(test_dir, "y.pt"))

    data_converter = dataConverter(conf)
    diagnostic = Diagnostics(conf, 0, data_converter=dataConverter(conf))

    time = datetime.datetime.now()
    pred_ds = data_converter.tensor_to_dataset(y_pred.float(), [time])
    y_ds = data_converter.tensor_to_dataset(y.float(), [time])
    metrics = diagnostic(pred_ds, y_ds, 1)
    metrics = diagnostic(pred_ds, y_ds + 1, 2)
    metrics = diagnostic(y_ds, pred_ds, 3)
    for k, v in metrics.items():
        print(v)
