import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pandas as pd
import numpy as np
from .physics_constants import GRAVITY

projections = {
    "robinson": ccrs.Robinson,
    "lcc": ccrs.LambertConformal,
    "cyl": ccrs.PlateCarree,
    "mercator": ccrs.Mercator,
    "stereographic": ccrs.Stereographic,
    "geostationary": ccrs.Geostationary,
    "nearside": ccrs.NearsidePerspective,
}


def kgkg_to_gkg(q):
    return q * 1000.0


def k_to_c(temperature):
    return temperature - 273.15


def k_to_f(temperature):
    return k_to_c(temperature) * 1.8 + 32


def gp_to_height_dam(gp):
    return gp / GRAVITY / 10.0


variable_transforms = {
    "T": k_to_c,
    "Z": gp_to_height_dam,
    "Q": kgkg_to_gkg,
    "Z500": gp_to_height_dam,
    "Q500": kgkg_to_gkg,
    "T500": k_to_c,
}


def plot_global_animation(
    forecast_dir,
    init_date,
    forecast_step,
    final_forecast_step,
    output_video_file="./credit_prediction.mp4",
    contourf_config=None,
    contour_config=None,
    projection_type="robinson",
    projection_config=None,
    title="CREDIT Prediction",
    date_format="%Y-%m-%d %HZ",
    figure_kwargs=None,
    axes_rect=(0.02, 0.02, 0.96, 0.96),
    fontsize=12,
    coastline_kwargs=None,
    border_kwargs=None,
    colorbar_kwargs=None,
    save_kwargs=None,
):
    """


    Args:
        forecast_dir:
        init_date:
        forecast_step:
        final_forecast_step:
        output_video_file:
        contourf_config:
        contour_config:
        projection_type:
        projection_config:
        title:
        date_format:
        figure_kwargs:
        axes_rect:
        fontsize:
        coastline_kwargs:
        border_kwargs:
        colorbar_kwargs:
        save_kwargs:

    Returns:

    """
    if contourf_config is None:
        contourf_config = dict(
            variable="Q500", contourf_kwargs=dict(levels=[1, 2, 3, 4, 5], cmap="viridis", vmin=1, vmax=5, extend="max")
        )
    if contour_config is None:
        contour_config = dict(
            Z500=dict(levels=np.arange(500, 605, 5), cmap="Purples"),
            T500=dict(levels=np.arange(-40, 10, 5), cmap="RdBu_r"),
        )
    if projection_config is None:
        projection_config = dict()
    if figure_kwargs is None:
        figure_kwargs = dict(figsize=(8, 6), dpi=300)

    init_date = pd.Timestamp(init_date)
    f_dates = pd.date_range(
        start=init_date + pd.Timedelta(hours=forecast_step),
        end=init_date + pd.Timedelta(hours=final_forecast_step),
        freq=forecast_step,
    )
    if save_kwargs is None:
        save_kwargs = dict(writer="ffmpeg", fps=5, dpi=300)
    with xr.open_mfdataset(os.path.join(forecast_dir, "*.nc")) as f_ds:
        fig = plt.figure(**figure_kwargs)
        ax = fig.add_axes(axes_rect, projection=projections[projection_type](**projection_config))
        lon_g, lat_g = np.meshgrid(f_ds["longitude"], f_ds["latitude"])
        ll_proj = ccrs.PlateCarree()

        def plot_step(i):
            f_date = f_dates[i]
            f_date_str = f_date.strftime(date_format)
            ax.clear()
            ax.set_title(f"{title} Valid {f_date_str}", fontsize=fontsize)
            if coastline_kwargs is not None:
                ax.coastlines(**coastline_kwargs)
            if border_kwargs is not None:
                ax.add_feature(cfeature.BORDERS, **border_kwargs)
            c_var = contourf_config["variable"]
            level = None
            if level in contourf_config.keys():
                level = contourf_config["level"]
            if c_var in variable_transforms.keys():
                if level is not None:
                    data_var = variable_transforms[c_var](f_ds[c_var].loc[f_date, level])
                else:
                    data_var = variable_transforms[c_var](f_ds[c_var].loc[f_date])
            else:
                if level is not None:
                    data_var = f_ds[c_var].loc[f_date, level]
                else:
                    data_var = f_ds[c_var].loc[f_date, level]

            filled_cont = ax.contourf(
                lon_g, lat_g, data_var, transform=ll_proj, transform_first=True, **contourf_config["contourf_kwargs"]
            )
            for c_var, c_var_config in contour_config.items():
                if c_var in variable_transforms.keys():
                    data_var = variable_transforms[c_var](f_ds[c_var].loc[f_date])
                else:
                    data_var = f_ds[c_var].loc[f_date]
                reg_cont = ax.contour(lon_g, lat_g, data_var, transform=ll_proj, transform_first=True, **c_var_config)
                ax.clabel(reg_cont)
            plt.colorbar(filled_cont, ax=ax, **colorbar_kwargs)
            return

        ani = animation.FuncAnimation(fig, plot_step, frames=f_dates.size)
        ani.save(output_video_file, **save_kwargs)
    return


def plot_regional_animation():
    return
