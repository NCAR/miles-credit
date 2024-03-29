import os
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import argparse
from glob import glob
from bridgescaler.distributed import DQuantileScaler
from os.path import exists, join
from mpi4py import MPI
from time import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-o", "--out", help="Path to save scaler files.")
    parser.add_argument("-t", "--time", type=str, default="25h",
                        help="Difference between times used for fitting.")
    args = parser.parse_args()
    args_dict = vars(args)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    config = args_dict.pop("config")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
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
            era5_dates.append(pd.date_range(start=start_date_str, end=end_date_str, freq=args_dict["time"]).to_series())
        all_era5_dates = pd.concat(era5_dates, ignore_index=True)
        split_indices = np.round(np.linspace(0, all_era5_dates.size, size + 1)).astype(int)
        split_era5_dates = [all_era5_dates.values[split_indices[s]:split_indices[s + 1]]
                            for s in range(split_indices.size - 1)]
        scaler_start_dates = pd.DatetimeIndex([split[0] for split in split_era5_dates]).strftime("%Y-%m-%d %H:%M")
        scaler_end_dates = pd.DatetimeIndex([split[-1] for split in split_era5_dates]).strftime("%Y-%m-%d %H:%M")
        print(scaler_start_dates)
        print(scaler_end_dates)
    else:
        scaler_start_dates = None
        scaler_end_dates = None
        all_era5_filenames = None
        split_era5_dates = None
    era5_subset_times = comm.scatter(split_era5_dates, root=0)
    vars_3d = conf["data"]["variables"]
    vars_surf = conf["data"]["surface_variables"]
    e5_file_dir = "/".join(conf["data"]["save_loc"].split("/")[:-1])
    scalers = fit_era5_scaler_times(era5_subset_times, rank, era5_file_dir=e5_file_dir,
                                    vars_3d=vars_3d, vars_surf=vars_surf)
    all_scalers = np.array(comm.gather(scalers, root=0))
    if rank == 0:
        all_scalers_dict = {"start_date": scaler_start_dates, "end_date": scaler_end_dates,
                           "scaler_3d": all_scalers[:, 0], "scaler_surface": all_scalers[:, 1]}
        all_scalers_df = pd.DataFrame(all_scalers_dict,
                                      columns=["start_date", "end_date", "scaler_3d", "scaler_surface"],
                                      index=all_era5_filenames)
        if not exists(args.out):
            os.makedirs(args.out)
        now = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H:%M")
        all_scalers_df.to_parquet(join(args.out, f"era5_quantile_scalers_{now}.parquet"))
    return


def fit_era5_scaler_times(times, rank, era5_file_dir=None, vars_3d=None, vars_surf=None):
    dqs_3d = DQuantileScaler(distribution="normal", channels_last=False)
    dqs_surf = DQuantileScaler(distribution="normal", channels_last=False)
    curr_f_start = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-01-01 00:00")
    curr_f_end = pd.Timestamp(pd.Timestamp(times[0]).strftime("%Y") + "-12-31 23:00")
    curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
    curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
    eds = xr.open_zarr(join(era5_file_dir, f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr"))
    levels = eds.level.values
    var_levels = []
    for var in vars_3d:
        for level in levels:
            var_levels.append(f"{var}_{level:d}")
    n_times = times.size
    times_index = pd.DatetimeIndex(times)
    for t, ctime in enumerate(times_index):
        print(f"Rank {rank:d}: {ctime} {t:d}/{n_times:d}")
        if not curr_f_start >= ctime <= curr_f_end:
            eds.close()
            curr_f_start = pd.Timestamp(pd.Timestamp(ctime).strftime("%Y") + "-01-01 00:00")
            curr_f_end = pd.Timestamp(pd.Timestamp(ctime).strftime("%Y") + "-12-31 23:00")
            curr_f_start_str = curr_f_start.strftime("%Y-%m-%d")
            curr_f_end_str = curr_f_end.strftime("%Y-%m-%d")
            eds = xr.open_zarr(join(era5_file_dir, f"TOTAL_{curr_f_start_str}_{curr_f_end_str}_staged.zarr"))
        var_slices = []
        for var in vars_3d:
            for level in levels:
                var_slices.append(eds[var].loc[ctime, level])
        e3d = xr.concat(var_slices, pd.Index(var_levels, name="variable")).load()
        e3d = e3d.expand_dims(dim="time", axis=0)
        dqs_3d.fit(e3d)
        e_surf = xr.concat([eds[v].loc[ctime] for v in vars_surf], pd.Index(vars_surf, name="variable")
                           ).load()
        e_surf = e_surf.expand_dims(dim="time", axis=0)
        dqs_surf.fit(e_surf)
    eds.close()
    return dqs_3d, dqs_surf


if __name__ == '__main__':
    main()
