import os
import gc
import sys
import yaml
import logging
import warnings
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import cftime

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import csv

# ---------- #
import torch
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything

from credit.data import (
    concat_and_reshape,
    reshape_only,
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class Predict_Dataset(torch.utils.data.IterableDataset):
    """
    Same as ERA5_and_Forcing_Dataset() but work with rollout_to_netcdf_new.py

    *ksha: dynamic forcing has been added to the rollout-only Dataset, but it has
    not been tested. Once the new tsi is ready, this dataset class will be tested
    """

    def __init__(
        self,
        conf,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames,
        filename_surface,
        filename_dyn_forcing,
        filename_forcing,
        filename_static,
        filename_diagnostic,
        fcst_datetime,
        history_len,
        rank,
        world_size,
        transform=None,
        rollout_p=0.0,
        which_forecast=None,
    ):
        # ------------------------------------------------------------------------------ #

        ## no diagnostics because they are output only
        # varname_diagnostic = None

        self.rank = rank
        self.world_size = world_size
        self.transform = transform
        self.history_len = history_len
        self.init_datetime = fcst_datetime

        self.which_forecast = (
            which_forecast  # <-- got from the old roll-out script. Dont know
        )

        # -------------------------------------- #
        # file names
        self.filenames = filenames  # <------------------------ a list of files
        self.filename_surface = filename_surface  # <---------- a list of files
        self.filename_dyn_forcing = filename_dyn_forcing  # <-- a list of files
        self.filename_forcing = filename_forcing  # <-- single file
        self.filename_static = filename_static  # <---- single file
        self.filename_diagnostic = filename_diagnostic  # <---- single file

        # -------------------------------------- #
        # var names
        self.varname_upper_air = varname_upper_air
        self.varname_surface = varname_surface
        self.varname_dyn_forcing = varname_dyn_forcing
        self.varname_forcing = varname_forcing
        self.varname_static = varname_static
        self.varname_diagnostic = varname_diagnostic

        # ====================================== #
        # import all upper air zarr files
        all_files = []
        for fn in self.filenames:
            # drop variables if they are not in the config
            xarray_dataset = get_forward_data(filename=fn)
            xarray_dataset = drop_var_from_dataset(
                xarray_dataset, self.varname_upper_air
            )
            # collect yearly datasets within a list
            all_files.append(xarray_dataset)
        self.all_files = all_files
        # ====================================== #

        # -------------------------------------- #
        # other settings
        self.current_epoch = 0
        self.rollout_p = rollout_p

        self.lead_time_periods = conf["data"]["lead_time_periods"]
        self.skip_periods = conf["data"]["skip_periods"]

    def ds_read_and_subset(self, filename, time_start, time_end, varnames):
        sliced_x = get_forward_data(filename)
        sliced_x = sliced_x.isel(time=slice(time_start, time_end))
        sliced_x = drop_var_from_dataset(sliced_x, varnames)
        return sliced_x

    def load_zarr_as_input(self, i_file, i_init_start, i_init_end, mode="input"):
        # get the needed file from a list of zarr files
        # open the zarr file as xr.dataset and subset based on the needed time

        # sliced_x: the final output, starts with an upper air xr.dataset
        sliced_x = self.ds_read_and_subset(
            self.filenames[i_file], i_init_start, i_init_end + 1, self.varname_upper_air
        )
        # surface variables
        if self.filename_surface is not None:
            sliced_surface = self.ds_read_and_subset(
                self.filename_surface[i_file],
                i_init_start,
                i_init_end + 1,
                self.varname_surface,
            )
            # merge surface to sliced_x
            sliced_surface["time"] = sliced_x["time"]
            sliced_x = sliced_x.merge(sliced_surface)

        if mode == "input":
            # dynamic forcing variables
            if self.filename_dyn_forcing is not None:
                sliced_dyn_forcing = self.ds_read_and_subset(
                    self.filename_dyn_forcing[i_file],
                    i_init_start,
                    i_init_end + 1,
                    self.varname_dyn_forcing,
                )
                # merge surface to sliced_x
                sliced_dyn_forcing["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_dyn_forcing)

            # forcing / static
            if self.filename_forcing is not None:
                sliced_forcing = get_forward_data(self.filename_forcing)
                sliced_forcing = drop_var_from_dataset(
                    sliced_forcing, self.varname_forcing
                )

                # See also `ERA5_and_Forcing_Dataset`
                # =============================================================================== #
                # matching month, day, hour between forcing and upper air [time]
                # this approach handles leap year forcing file and non-leap-year upper air file
                month_day_forcing = extract_month_day_hour(
                    np.array(sliced_forcing["time"])
                )
                month_day_inputs = extract_month_day_hour(np.array(sliced_x["time"]))
                # indices to subset
                ind_forcing, _ = find_common_indices(
                    month_day_forcing, month_day_inputs
                )
                sliced_forcing = sliced_forcing.isel(time=ind_forcing)
                # forcing and upper air have different years but the same mon/day/hour
                # safely replace forcing time with upper air time
                sliced_forcing["time"] = sliced_x["time"]
                # =============================================================================== #

                # merge forcing to sliced_x
                sliced_x = sliced_x.merge(sliced_forcing)

            if self.filename_static is not None:
                sliced_static = get_forward_data(self.filename_static)
                sliced_static = drop_var_from_dataset(
                    sliced_static, self.varname_static
                )
                sliced_static = sliced_static.expand_dims(
                    dim={"time": len(sliced_x["time"])}
                )
                sliced_static["time"] = sliced_x["time"]
                # merge static to sliced_x
                sliced_x = sliced_x.merge(sliced_static)

        elif mode == "target":
            # diagnostic
            if self.filename_diagnostic is not None:
                sliced_diagnostic = self.ds_read_and_subset(
                    self.filename_diagnostic[i_file],
                    i_init_start,
                    i_init_end + 1,
                    self.varname_diagnostic,
                )
                # merge diagnostics to sliced_x
                sliced_diagnostic["time"] = sliced_x["time"]
                sliced_x = sliced_x.merge(sliced_diagnostic)

        return sliced_x

    def find_start_stop_indices(self, index):
        # ============================================================================ #
        # shift hours for history_len > 1, because more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = (
            self.lead_time_periods * self.skip_periods * (self.history_len - 1)
        )
        # ============================================================================ #
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.strptime(
            self.init_datetime[index][0], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)
        self.init_datetime[index][1] = datetime.strptime(
            self.init_datetime[index][1], "%Y-%m-%d %H:%M:%S"
        ) - timedelta(hours=shifted_hours)

        # convert the 1st & last init times to a list of init times
        self.init_datetime[index] = generate_datetime(
            self.init_datetime[index][0],
            self.init_datetime[index][1],
            self.lead_time_periods,
        )
        # convert datetime obj to nanosecondes
        init_time_list_dt = [
            np.datetime64(date.strftime("%Y-%m-%d %H:%M:%S"))
            for date in self.init_datetime[index]
        ]

        # init_time_list_np: a list of python datetime objects, each is a forecast step
        # init_time_list_np[0]: the first initialization time
        # init_time_list_np[t]: the forcasted time of the (t-1)th step; the initialization time of the t-th step
        self.init_time_list_np = [
            np.datetime64(str(dt_obj) + ".000000000").astype(datetime)
            for dt_obj in init_time_list_dt
        ]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.all_files):
                # get the year of the current file

                if isinstance(ds["time"][0].values.item(), cftime.DatetimeNoLeap):
                    datetime64_times = np.array([np.datetime64(time.strftime("%Y-%m-%d %H:%M:%S"), 'ns') for time in ds["time"].values])
                    ds["time"] = datetime64_times
                
                ds_year = int(np.datetime_as_string(ds["time"][0].values, unit="Y"))

                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)

                # found the right yearly file
                if init_year0 == ds_year:
                    N_times = len(ds["time"])
                    # convert ds['time'] to a list of nanosecondes
                    ds_time_list = [
                        np.datetime64(ds_time.values).astype(datetime)
                        for ds_time in ds["time"]
                    ]
                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]

                    init_time_start = init_time
                    # if initalization time is within this (yearly) xr.Dataset
                    if ds_start_time <= init_time_start <= ds_end_time:
                        # try getting the index of the first initalization time
                        i_init_start = ds_time_list.index(init_time_start)

                        # for multiple init time inputs (history_len > 1), init_end is different for init_start
                        init_time_end = init_time_start + hour_to_nanoseconds(
                            shifted_hours
                        )

                        # see if init_time_end is alos in this file
                        if ds_start_time <= init_time_end <= ds_end_time:
                            # try getting the index
                            i_init_end = ds_time_list.index(init_time_end)
                        else:
                            # this set of initalizations have crossed years
                            # get the last element of the current file
                            # we have anthoer section that checks additional input data
                            i_init_end = len(ds_time_list) - 1

                        info.append([i_file, i_init_start, i_init_end, N_times])
        return info

    def __len__(self):
        return len(self.init_datetime)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(
            self,
            num_replicas=num_workers * self.world_size,
            rank=self.rank * num_workers + worker_id,
            shuffle=False,
        )
        for index in sampler:
            # get the init time info for the current sample
            data_lookup = self.find_start_stop_indices(index)

            for k, _ in enumerate(self.init_time_list_np):
                # the first initialization time: get initalization from data
                i_file, i_init_start, i_init_end, N_times = data_lookup[k]

                # allocate output dict
                output_dict = {}

                # get all inputs in one xr.Dataset
                sliced_x = self.load_zarr_as_input(
                    i_file, i_init_start, i_init_end, mode="input"
                )

                # Check if additional data from the next file is needed
                if (len(sliced_x["time"]) < self.history_len) or (
                    i_init_end + 1 >= N_times
                ):
                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[i_file]) + 1

                    if next_file_idx >= len(self.filenames):
                        # not enough input data to support this forecast
                        raise OSError(
                            "You have reached the end of the available data. Exiting."
                        )

                    else:
                        sliced_y = self.load_zarr_as_input(
                            i_file, i_init_end, i_init_end, mode="target"
                        )

                        # i_init_start = 0 because we need the beginning of the next file only
                        sliced_x_next = self.load_zarr_as_input(
                            next_file_idx, 0, self.history_len, mode="input"
                        )
                        sliced_y_next = self.load_zarr_as_input(
                            next_file_idx, 0, 1, mode="target"
                        )
                        # 1 becuase taregt is one step a time

                        # Concatenate excess data from the next file with the current data
                        sliced_x_combine = xr.concat(
                            [sliced_x, sliced_x_next], dim="time"
                        )
                        sliced_y_combine = xr.concat(
                            [sliced_y, sliced_y_next], dim="time"
                        )

                        sliced_x = sliced_x_combine.isel(
                            time=slice(0, self.history_len)
                        )
                        sliced_y = sliced_y_combine.isel(
                            time=slice(self.history_len, self.history_len + 1)
                        )
                else:
                    sliced_y = self.load_zarr_as_input(
                        i_file, i_init_end + 1, i_init_end + 1, mode="target"
                    )

                sample_x = {
                    "historical_ERA5_images": sliced_x,
                    "target_ERA5_images": sliced_y,
                }

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in sample_x.keys():
                    output_dict[key] = sample_x[key]

                # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                output_dict["forecast_hour"] = k + 1
                # Adjust stopping condition
                output_dict["stop_forecast"] = k == (len(self.init_time_list_np) - 1)
                output_dict["datetime"] = sliced_x.time.values.astype(
                    "datetime64[s]"
                ).astype(int)[-1]

                # return output_dict
                yield output_dict

                if output_dict["stop_forecast"]:
                    break


def predict(rank, world_size, conf, p):
    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    # number of input time frames
    history_len = conf["data"]["history_len"]

    # length of forecast steps
    lead_time_periods = conf["data"]["lead_time_periods"]

    # transform and ToTensor class
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == "std_new":
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
        raise
    # ----------------------------------------------------------------- #
    # parse varnames and save_locs from config

    # upper air variables
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))
    varname_upper_air = conf["data"]["variables"]

    # surface variables
    varname_surface = conf["data"]["surface_variables"]

    if conf["data"]["flag_surface"]:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))
    else:
        surface_files = None

    # diagnostic variables
    varname_diagnostic = conf["data"]["diagnostic_variables"]

    if conf["data"]["flag_diagnostic"]:
        diagnostic_files = sorted(glob(conf["data"]["save_loc_diagnostic"]))
    else:
        diagnostic_files = None

    # dynamic forcing variables
    varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]

    if conf["data"]["flag_dyn_forcing"]:
        dyn_forcing_files = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
    else:
        dyn_forcing_files = None

    # forcing variables
    forcing_files = conf["data"]["save_loc_forcing"]
    varname_forcing = conf["data"]["forcing_variables"]

    # static variables
    static_files = conf["data"]["save_loc_static"]
    varname_static = conf["data"]["static_variables"]

    # number of diagnostic variables
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

    # ====================================================== #
    # postblock opts outside of model
    post_conf = conf["model"]["post_conf"]
    flag_mass_conserve = False
    flag_water_conserve = False
    flag_energy_conserve = False

    if post_conf["activate"]:
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalMassFixer outside of model")
                flag_mass_conserve = True
                opt_mass = GlobalMassFixer(post_conf)

        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalWaterFixer outside of model")
                flag_water_conserve = True
                opt_water = GlobalWaterFixer(post_conf)

        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"]:
                logger.info("Activate GlobalEnergyFixer outside of model")
                flag_energy_conserve = True
                opt_energy = GlobalEnergyFixer(post_conf)
    # ====================================================== #

    # ----------------------------------------------------------------- #\
    # get dataset
    dataset = Predict_Dataset(
        conf,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        varname_diagnostic,
        filenames=all_ERA_files,
        filename_surface=surface_files,
        filename_dyn_forcing=dyn_forcing_files,
        filename_forcing=forcing_files,
        filename_static=static_files,
        filename_diagnostic=diagnostic_files,
        fcst_datetime=load_forecasts(conf),
        history_len=history_len,
        rank=rank,
        world_size=world_size,
        transform=transform,
        rollout_p=0.0,
        which_forecast=None,
    )

    # setup the dataloder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # load model
    model = load_model(conf, load_weights=True).to(device)

    # Warning -- see next line
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:  # A new field needs to be added to predict
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp":
            # Load model weights (if any), an optimizer, scheduler, and gradient scaler
            model = load_model_state(conf, model, device)

    model.eval()

    # get lat/lons from x-array
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])

    meta_data = load_metadata(conf)

    # Set up metrics and containers
    metrics = LatWeightedMetrics(conf, predict_mode=True)
    metrics_results = defaultdict(list)

    # Set up the diffusion and pole filters
    if (
        "use_laplace_filter" in conf["predict"]
        and conf["predict"]["use_laplace_filter"]
    ):
        dpf = Diffusion_and_Pole_Filter(
            nlat=conf["model"]["image_height"],
            nlon=conf["model"]["image_width"],
            device=device,
        )

    # Rollout
    #open the forcing file.
    DSforc = xr.open_dataset(conf["predict"]["forcing_file"]).load()
    #grab forcing only .. 

    df_variables = conf["data"]["dynamic_forcing_variables"]
    sf_variables = conf["data"]["static_variables"]

    if conf["data"]["static_first"]:
        df_sf_variables = conf["data"]["static_variables"] + conf["data"]["dynamic_forcing_variables"]
    else:
        df_sf_variables = conf["data"]["dynamic_forcing_variables"] + conf["data"]["static_variables"]

    DSforc_norm = state_transformer.transform_dataset(DSforc)
    DSforc = DSforc[df_sf_variables]
    print('forcing file opened and loaded')

    #transform the forcing file
    
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0

        # y_pred allocation
        results = []

        # model inference loop
        for k, batch in enumerate(data_loader):
            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            print('forecast hour: ', forecast_hour)
            # initialization on the first forecast hour
            if forecast_hour == 1:
                # Initialize x and x_surf with the first time step
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = (
                        concat_and_reshape(batch["x"], batch["x_surf"])
                        .to(device)
                        .float()
                    )
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(device).float()

                init_datetime = datetime.utcfromtimestamp(date_time)
                init_datetime_str = init_datetime.strftime("%Y-%m-%dT%HZ")

            # -------------------------------------------------------------------------------------- #
            # add forcing and static variables (regardless of fcst hours)
            if "x_forcing_static" in batch:
                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = (
                    batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                )
                print('braaaap: ', x_forcing_batch.shape)
                # concat on var dimension
                x = torch.cat((x, x_forcing_batch), dim=1)

            # -------------------------------------------------------------------------------------- #
            # Load y-truth
            if "y_surf" in batch:
                # combine y and y_surf
                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
            else:
                # no y_surf
                y = reshape_only(batch["y"]).to(device).float()

            # -------------------------------------------------------------------------------------- #
            # start prediction
            # Save the current forecast hour data in parallel
            utc_datetime = init_datetime + timedelta(
                hours=lead_time_periods * forecast_hour
            )
            print(utc_datetime)
            
            y_pred = model(x)

            # ============================================= #
            # postblock opts outside of model

            # backup init state
            if flag_mass_conserve:
                if forecast_hour == 1:
                    x_init = x.clone()

            # mass conserve using initialization as reference
            if flag_mass_conserve:
                input_dict = {"y_pred": y_pred, "x": x_init}
                input_dict = opt_mass(input_dict)
                y_pred = input_dict["y_pred"]

            # water conserve use previous step output as reference
            if flag_water_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_water(input_dict)
                y_pred = input_dict["y_pred"]

            # energy conserve use previous step output as reference
            if flag_energy_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_energy(input_dict)
                y_pred = input_dict["y_pred"]
            # ============================================= #
            # WEC
            # # y_pred with unit
            # y_pred = state_transformer.inverse_transform(y_pred.cpu())
            # # y_target with unit
            # y = state_transformer.inverse_transform(y.cpu())

            # if (
            #     "use_laplace_filter" in conf["predict"]
            #     and conf["predict"]["use_laplace_filter"]
            # ):
            #     y_pred = (
            #         dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
            #         .unsqueeze(0)
            #         .unsqueeze(2)
            #         .cpu()
            #     )

            # # Compute metrics
            # metrics_dict = metrics(
            #     y_pred.float(), y.float(), forecast_datetime=forecast_hour
            # )
            # for k, m in metrics_dict.items():
            #     metrics_results[k].append(m.item())
            # metrics_results["forecast_hour"].append(forecast_hour)

            # convert the current step result as x-array
            darray_upper_air, darray_single_level = make_xarray(
                y_pred.cpu(),
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf,
            )

            # Save the current forecast hour data in parallel
            result = p.apply_async(
                save_netcdf_increment,
                (
                    darray_upper_air,
                    darray_single_level,
                    init_datetime_str,
                    lead_time_periods * forecast_hour,
                    meta_data,
                    conf,
                ),
            )
            results.append(result)

            metrics_results["datetime"].append(utc_datetime)

            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "
            # print_str += f"ACC: {metrics_dict['acc']} "

            # Update the input
            # setup for next iteration, transform to z-space and send to device
            # y_pred = state_transformer.transform_array(y_pred).to(device)

            # ============================================================ #
            # use previous step y_pred as the next step input
            if history_len == 1:
                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()

            # multi-step in
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = torch.cat(
                        [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                    )
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)
            # ============================================================ #

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            if batch["stop_forecast"][0]:
                # Wait for all processes to finish in order
                for result in results:
                    result.get()

                # save metrics file
                save_location = os.path.join(
                    os.path.expandvars(conf["save_loc"]), "forecasts", "metrics"
                )
                os.makedirs(
                    save_location, exist_ok=True
                )  # should already be made above
                df = pd.DataFrame(metrics_results)
                df.to_csv(
                    os.path.join(save_location, f"metrics{init_datetime_str}.csv")
                )

                # forecast count = a constant for each run
                forecast_count += 1

                # y_pred allocation
                y_pred = None

                gc.collect()

                if distributed:
                    torch.distributed.barrier()

            break
        add_it = 0
        for k in range(conf["predict"]["climate_timesteps"]):
            forecast_hour = k+2
            utc_datetime = init_datetime + timedelta(
                hours=lead_time_periods * forecast_hour
            )

            # -------------------------------------------------------------------------------------- #
            # add forcing and static variables (regardless of fcst hours)               

                
            if k ==0 and "x_forcing_static" in batch:
                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = (
                    batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
                )
                print('braaaap: ', x_forcing_batch.shape)
                # concat on var dimension
                #x = torch.cat((x, x_forcing_batch), dim=1)
                
                selected_data = DSforc.sel(time=utc_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                print(selected_data['time'])
                time_selection = utc_datetime.strftime('%Y-%m-%d %H:%M:%S')
                
                # Find the index of the specified time
                forc_time_index = DSforc.get_index('time').get_loc(time_selection).start
                print(f"The index for time {time_selection} is {forc_time_index}")

                total_len_forc_file = len(DSforc['time'])
                print(f"The the len of the forcing time ind is {total_len_forc_file}")

            if (forc_time_index+k) > total_len_forc_file:
                forc_time_index = 0
                add_it = 0

            if conf["data"]["static_first"]:
                for bb,sfv in enumerate(sf_variables):
                    x_forcing_batch[:,bb,:,:,:] = torch.tensor(DSforc[sfv].values)
                    
                for bb,dfv in enumerate(df_variables):
                    x_forcing_batch[:,len(sf_variables)+bb,:,:,:] = torch.tensor(DSforc_norm[dfv].isel(time=forc_time_index+add_it).values)
                with open('output_solin.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    valval = DSforc_norm[dfv].isel(time=forc_time_index + add_it).squeeze().values[30, 100]
                    writer.writerow([valval])
        
            else:
                for bb,dfv in enumerate(df_variables):
                    x_forcing_batch[:,bb,:,:,:] = torch.tensor(DSforc_norm[dfv].isel(time=forc_time_index+add_it).values) 
                    
                for bb,sfv in enumerate(sf_variables):
                    x_forcing_batch[:,len(df_variables)+bb,:,:,:] = torch.tensor(DSforc[sfv].values)

            x = torch.cat((x, x_forcing_batch), dim=1)


            y_pred = model(x)

            # ============================================= #
            # postblock opts outside of model

            # backup init state
            if flag_mass_conserve:
                if forecast_hour == 1:
                    x_init = x.clone()

            # mass conserve using initialization as reference
            if flag_mass_conserve:
                input_dict = {"y_pred": y_pred, "x": x_init}
                input_dict = opt_mass(input_dict)
                y_pred = input_dict["y_pred"]

            # water conserve use previous step output as reference
            if flag_water_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_water(input_dict)
                y_pred = input_dict["y_pred"]

            # energy conserve use previous step output as reference
            if flag_energy_conserve:
                input_dict = {"y_pred": y_pred, "x": x}
                input_dict = opt_energy(input_dict)
                y_pred = input_dict["y_pred"]
            y_pred = state_transformer.inverse_transform(y_pred.cpu())

            # if (
            #     "use_laplace_filter" in conf["predict"]
            #     and conf["predict"]["use_laplace_filter"]
            # ):
            #     y_pred = (
            #         dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
            #         .unsqueeze(0)
            #         .unsqueeze(2)
            #         .cpu()
            #     )

            # # Compute metrics
            # metrics_dict = metrics(
            #     y_pred.float(), y.float(), forecast_datetime=forecast_hour
            # )
            # for k, m in metrics_dict.items():
            #     metrics_results[k].append(m.item())
            # metrics_results["forecast_hour"].append(forecast_hour)

            # convert the current step result as x-array
            darray_upper_air, darray_single_level = make_xarray(
                y_pred.cpu(),
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf,
            )

            # Save the current forecast hour data in parallel
            result = p.apply_async(
                save_netcdf_increment,
                (
                    darray_upper_air,
                    darray_single_level,
                    init_datetime_str,
                    lead_time_periods * forecast_hour,
                    meta_data,
                    conf,
                ),
            )
            results.append(result)

            metrics_results["datetime"].append(utc_datetime)

            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "
            # print_str += f"ACC: {metrics_dict['acc']} "

            # Update the input
            # setup for next iteration, transform to z-space and send to device
            y_pred = state_transformer.transform_array(y_pred).to(device)

            # ============================================================ #
            # use previous step y_pred as the next step input
            if history_len == 1:
                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()

            # multi-step in
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = torch.cat(
                        [x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2
                    )
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)
            # ============================================================ #

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            add_it+=1

            if k == (conf["predict"]["climate_timesteps"] -1 ):
                # Wait for all processes to finish in order
                for result in results:
                    result.get()

                # save metrics file
                save_location = os.path.join(
                    os.path.expandvars(conf["save_loc"]), "forecasts", "metrics"
                )
                os.makedirs(
                    save_location, exist_ok=True
                )  # should already be made above
                df = pd.DataFrame(metrics_results)
                df.to_csv(
                    os.path.join(save_location, f"metrics{init_datetime_str}.csv")
                )

                # forecast count = a constant for each run
                forecast_count += 1

                # y_pred allocation
                y_pred = None

                gc.collect()

                if distributed:
                    torch.distributed.barrier()                        
                break
                    
            # -------------------------------------------------------------------------------------- #

    
    if distributed:
        torch.distributed.barrier()

    return 1


if __name__ == "__main__":
    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )

    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=int,
        default=False,
        help="Predict on subset X of forecasts",
    )
    parser.add_argument(
        "-ns",
        "--no_subset",
        type=int,
        default=False,
        help="Break the forecasts list into X subsets to be processed by X GPUs",
    )
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))
    subset = int(args_dict.pop("subset"))
    number_of_subsets = int(args_dict.pop("no_subset"))
    num_cpus = int(args_dict.pop("num_cpus"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # ======================================================== #
    # handling config args
    conf = credit_main_parser(
        conf, parse_training=False, parse_predict=True, print_summary=False
    )
    predict_data_check(conf, print_summary=False)
    # ======================================================== #

    # create a save location for rollout
    # ---------------------------------------------------- #
    assert (
        "save_forecast" in conf["predict"]
    ), "Please specify the output dir through conf['predict']['save_forecast']"

    forecast_save_loc = conf["predict"]["save_forecast"]
    os.makedirs(forecast_save_loc, exist_ok=True)

    print("Save roll-outs to {}".format(forecast_save_loc))

    # Create a project directory (to save launch.sh and model.yml) if they do not exist
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["predict"]["mode"] = mode

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="Derecho parallelism",
    #         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
    #         # track hyperparameters and run metadata
    #         config=conf
    #     )

    if number_of_subsets > 0:
        forecasts = load_forecasts(conf)
        if number_of_subsets > 0 and subset >= 0:
            subsets = np.array_split(forecasts, number_of_subsets)
            forecasts = subsets[subset - 1]  # Select the subset based on subset_size
            conf["predict"]["forecasts"] = forecasts

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(
                int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, p=p
            )
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
