import os
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import argparse
from glob import glob
from bridgescaler.distributed import DQuantileScaler, DStandardScaler
from bridgescaler import print_scaler, read_scaler
from os.path import exists, join
from mpi4py import MPI

scalers = dict(standard=DStandardScaler, quantile=DQuantileScaler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--out", help="Path to save scaler files.")
    parser.add_argument("-d", "--dataout", help="Path to save transformed files.")
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="25h",
        help="Difference between times used for fitting.",
    )
    parser.add_argument("-f", "--fit", action="store_true", help="Fit scalers.")
    parser.add_argument(
        "-r", "--transform", action="store_true", help="Transform data with scalers."
    )
    parser.add_argument(
        "-g",
        "--fitdt",
        action="store_true",
        help="Fit scaler to time residuals of scaled data.",
    )
    parser.add_argument(
        "-s", "--scalerfile", help="Path to parquet file containing scalers."
    )

    args = parser.parse_args()
    args_dict = vars(args)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    config = args_dict.pop("config")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    if "scaler" in conf.keys():
        scaler_type = conf["scaler"]["type"]
        scaler_config = conf["scaler"]["options"]
    else:
        scaler_type = "quantile"
        scaler_config = dict(distribution="normal", channels_last=False)

    if rank == 0:
        all_era5_files = sorted(glob(conf["data"]["save_loc"]))
        for e5 in all_era5_files:
            if "_small_" in e5:
                all_era5_files.remove(e5)
        all_era5_filenames = np.array([f.split("/")[-1] for f in all_era5_files])
        era5_dates = []
        for fname in all_era5_filenames:
            start_date_str, end_date_str = fname.split("_")[1:3]
            start_date_str += " 00:00:00"
            end_date_str += " 23:00:00"
            era5_dates.append(
                pd.date_range(
                    start=start_date_str, end=end_date_str, freq=args_dict["time"]
                ).to_series()
            )
        all_era5_dates = pd.concat(era5_dates, ignore_index=True)
        split_indices = np.round(np.linspace(0, all_era5_dates.size, size + 1)).astype(
            int
        )
        split_era5_dates = [
            all_era5_dates.values[split_indices[s] : split_indices[s + 1]]
            for s in range(split_indices.size - 1)
        ]
        scaler_start_dates = pd.DatetimeIndex(
            [split[0] for split in split_era5_dates]
        ).strftime("%Y-%m-%d %H:%M")
        scaler_end_dates = pd.DatetimeIndex(
            [split[-1] for split in split_era5_dates]
        ).strftime("%Y-%m-%d %H:%M")

    else:
        scaler_start_dates = None
        scaler_end_dates = None
        split_era5_dates = None
    era5_subset_times = comm.scatter(split_era5_dates, root=0)
    vars_3d = conf["data"]["variables"]
    vars_surf = conf["data"]["surface_variables"]
    e5_file_dir = "/".join(conf["data"]["save_loc"].split("/")[:-1])
    if args.fit:
        scalers = fit_era5_scaler_times(
            era5_subset_times,
            rank,
            era5_file_dir=e5_file_dir,
            vars_3d=vars_3d,
            vars_surf=vars_surf,
            scaler_type=scaler_type,
            scaler_config=scaler_config,
        )
        all_scalers = np.array(comm.gather(scalers, root=0))
        if rank == 0:
            all_scalers_dict = {
                "start_date": scaler_start_dates,
                "end_date": scaler_end_dates,
                "scaler_3d": [print_scaler(s) for s in all_scalers[:, 0]],
                "scaler_surface": [print_scaler(s) for s in all_scalers[:, 1]],
            }
            all_scalers_df = pd.DataFrame(
                all_scalers_dict,
                columns=["start_date", "end_date", "scaler_3d", "scaler_surface"],
            )
            if not exists(args.out):
                os.makedirs(args.out)
            now = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H:%M")
            all_scalers_df.to_parquet(
                join(args.out, f"era5_{scaler_type}_scalers_{now}.parquet")
            )
    if args.transform:
        if rank == 0:
            if not exists(args.dataout):
                os.makedirs(args.dataout, exist_ok=True)
        print(f"Rank {rank:d}: ", era5_subset_times[0], era5_subset_times[-1])
        transform_era5_times(
            era5_subset_times,
            rank,
            scaler_file=args.scalerfile,
            era5_file_dir=e5_file_dir,
            scaler_type=scaler_type,
            vars_3d=vars_3d,
            vars_surf=vars_surf,
            out_dir=args.dataout,
        )
    if args.fitdt:
        scalers = fit_scaled_era5_time_residuals(
            era5_subset_times,
            rank,
            dt=1,
            era5_file_dir=args.dataout,
            scaler_type=scaler_type,
            scaler_config=scaler_config,
        )
        all_scalers = np.array(comm.gather(scalers, root=0))
        if rank == 0:
            all_scalers_dict = {
                "start_date": scaler_start_dates,
                "end_date": scaler_end_dates,
                "scaler_3d": [print_scaler(s) for s in all_scalers[:, 0]],
                "scaler_surface": [print_scaler(s) for s in all_scalers[:, 1]],
            }
            all_scalers_df = pd.DataFrame(
                all_scalers_dict,
                columns=["start_date", "end_date", "scaler_3d", "scaler_surface"],
            )
            if not exists(args.out):
                os.makedirs(args.out)
            now = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H:%M")
            all_scalers_df.to_parquet(
                join(args.out, f"era5_dt_{scaler_type}_scalers_{now}.parquet")
            )
    return


def fit_era5_scaler_times(
    times,
    rank,
    era5_file_dir=None,
    vars_3d=None,
    vars_surf=None,
    scaler_type="quantile",
    scaler_config=None,
):
    """ """
    if scaler_config is None:
        scaler_config = dict()
    dsc_3d = scalers[scaler_type](**scaler_config)
    dsc_surf = scalers[scaler_type](**scaler_config)
    curr_f_start = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-01-01 00:00")
    curr_f_end = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-12-31 23:00")
    curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
    curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
    eds = xr.open_zarr(
        join(era5_file_dir, f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr")
    )
    levels = eds.level.values
    var_levels = []
    for var in vars_3d:
        for level in levels:
            var_levels.append(f"{var}_{level:d}")
    n_times = times.size
    times_index = pd.DatetimeIndex(times)
    for t, ctime in enumerate(times_index):
        print(f"Rank {rank:d}: {ctime} {t + 1:d}/{n_times:d}")
        if not curr_f_start >= ctime <= curr_f_end:
            eds.close()
            curr_f_start = pd.Timestamp(
                pd.Timestamp(ctime).strftime("%Y") + "-01-01 00:00"
            )
            curr_f_end = pd.Timestamp(
                pd.Timestamp(ctime).strftime("%Y") + "-12-31 23:00"
            )
            curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
            curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
            eds = xr.open_zarr(
                join(
                    era5_file_dir,
                    f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr",
                )
            )
        var_slices = []
        for var in vars_3d:
            for level in levels:
                var_slices.append(eds[var].loc[ctime, level])
        e3d = xr.concat(var_slices, pd.Index(var_levels, name="variable")).load()
        e3d = e3d.expand_dims(dim="time", axis=0)
        dsc_3d.fit(e3d)
        e_surf = xr.concat(
            [eds[v].loc[ctime] for v in vars_surf], pd.Index(vars_surf, name="variable")
        ).load()
        e_surf = e_surf.expand_dims(dim="time", axis=0)
        dsc_surf.fit(e_surf)
    eds.close()
    return dsc_3d, dsc_surf


def transform_era5_times(
    times,
    rank,
    scaler_file=None,
    era5_file_dir=None,
    vars_3d=None,
    vars_surf=None,
    scaler_type=None,
    out_dir="/glade/derecho/scratch/dgagne/era5_quantile/",
    var_encoding=None,
):
    if var_encoding is None:
        var_encoding = {
            "zlib": True,
            "complevel": 3,
            "shuffle": True,
            "significant_digits": 4,
        }
    dqs_df = pd.read_parquet(scaler_file)
    dqs_end_dates = pd.DatetimeIndex(dqs_df["end_date"])
    dqs_3d = dqs_df["scaler_3d"][dqs_end_dates < "2014-01-01"].apply(read_scaler).sum()
    dqs_surf = (
        dqs_df["scaler_surface"][dqs_end_dates < "2014-01-01"].apply(read_scaler).sum()
    )
    curr_f_start = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-01-01 00:00")
    curr_f_end = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-12-31 23:00")
    curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
    curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
    eds = xr.open_zarr(
        join(era5_file_dir, f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr"),
        chunks=None,
    )
    levels = eds.level.values
    var_levels = []
    for var in vars_3d:
        for level in levels:
            var_levels.append(f"{var}_{level:d}")
    n_times = times.size
    times_index = pd.DatetimeIndex(times)
    for t, ctime in enumerate(times_index):
        print(f"Rank {rank:d}: {ctime} {t + 1:d}/{n_times:d}")

        if not curr_f_start >= ctime <= curr_f_end:
            eds.close()
            curr_f_start = pd.Timestamp(
                pd.Timestamp(ctime).strftime("%Y") + "-01-01 00:00"
            )
            curr_f_end = pd.Timestamp(
                pd.Timestamp(ctime).strftime("%Y") + "-12-31 23:00"
            )
            curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
            curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
            eds = xr.open_zarr(
                join(
                    era5_file_dir,
                    f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr",
                ),
                chunks=None,
            )
        var_level_data = []
        for var in vars_3d:
            for level in levels:
                var_level_data.append(eds[var].loc[ctime, level])
        level_data = (
            xr.concat(var_level_data, pd.Index(var_levels, name="variable"))
            .expand_dims("time", axis=0)
            .load()
        )
        surf_data = (
            eds[vars_surf]
            .sel(time=ctime)
            .to_dataarray(dim="surface_variable", name="era5_surface")
            .expand_dims("time", axis=0)
            .load()
        )
        level_scaled = dqs_3d.transform(level_data)
        surf_scaled = dqs_surf.transform(surf_data)
        f_time_now = ctime.strftime("%Y-%m-%dT%H:%M:%S")
        full_out_dir = join(out_dir, ctime.strftime("%Y/%m/%d/"))
        if not exists(full_out_dir):
            os.makedirs(full_out_dir, exist_ok=True)
        out_ds = xr.Dataset({"levels": level_scaled, "surface": surf_scaled})
        full_out_filename = join(full_out_dir, f"TOTAL_{f_time_now}_{scaler_type}.nc")
        out_ds.to_netcdf(
            full_out_filename,
            encoding={"levels": var_encoding, "surface": var_encoding},
        )
    eds.close()
    return


def fit_scaled_era5_time_residuals(
    times, rank, era5_file_dir=None, dt=1, scaler_type="standard", scaler_config=None
):
    """
    Fit scalers to distributions of time differences for each variable.

    Args:
        times: List or Series of times
        rank (int): MPI rank
        era5_file_dir (str): Path to era5 scaled files
        dt (int): number of hours difference
        scaler_type (str): standard or quantile
        scaler_config (dict): kwargs for the scaler obj

    Returns:
        3D scaler, surface scaler
    """
    times_index = pd.DatetimeIndex(times)
    n_times = times.size
    if scaler_config is None:
        scaler_config = dict()
    dsc_3d = scalers[scaler_type](**scaler_config)
    dsc_surf = scalers[scaler_type](**scaler_config)
    for t, ctime in enumerate(times_index):
        ct1 = ctime + pd.Timedelta(dt, "hours")
        print(f"Rank {rank:d}: {ctime} {t + 1:d}/{n_times:d}")
        ct_str = ctime.strftime("%Y-%m-%dT%H:%M:%S")
        ct1_str = ct1.strftime("%Y-%m-%dT%H:%M:%S")
        sds_t_filename = join(
            era5_file_dir,
            ctime.strftime("%Y/%m/%d/"),
            f"TOTAL_{ct_str}_{scaler_type}.nc",
        )
        sds_t1_filename = join(
            era5_file_dir,
            ctime.strftime("%Y/%m/%d/"),
            f"TOTAL_{ct1_str}_{scaler_type}.nc",
        )
        if not (exists(sds_t1_filename) and exists(sds_t_filename)):
            continue
        sds_t = xr.open_dataset(sds_t_filename)
        sds_t1 = xr.open_dataset(sds_t1_filename)
        sds_3d_diff = (
            sds_t1["levels"].squeeze() - sds_t["levels"].squeeze()
        ).expand_dims(dim="time", axis=0)
        dsc_3d.fit(sds_3d_diff)
        sds_surf_diff = (
            sds_t1["surface"].squeeze() - sds_t["surface"].squeeze()
        ).expand_dims(dim="time", axis=0)
        dsc_surf.fit(sds_surf_diff)
        sds_t.close()
        sds_t1.close()
    return dsc_3d, dsc_surf


if __name__ == "__main__":
    main()
