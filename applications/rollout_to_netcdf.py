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

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd

# ---------- #
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.data import Predict_Dataset, concat_and_reshape, reshape_only, generate_datetime, nanoseconds_to_year, hour_to_nanoseconds
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper
from credit.models.checkpoint import load_model_state
from credit.parser import CREDIT_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class Predict_Dataset_Metrics(Predict_Dataset):

    def find_start_stop_indices(self, index):
        # ============================================================================ #
        # shift hours for history_len > 1, becuase more than one init times are needed
        # <--- !! it MAY NOT work when self.skip_period != 1
        shifted_hours = self.lead_time_periods * self.skip_periods * (self.history_len-1)
        # ============================================================================ #
        # subtrack shifted_hour form the 1st & last init times
        # convert to datetime object
        self.init_datetime[index][0] = datetime.strptime(
            self.init_datetime[index][0], '%Y-%m-%d %H:%M:%S') - timedelta(hours=shifted_hours)
        self.init_datetime[index][1] = datetime.strptime(
            self.init_datetime[index][1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=shifted_hours)

        # convert the 1st & last init times to a list of init times
        self.init_datetime[index] = generate_datetime(self.init_datetime[index][0], self.init_datetime[index][1], self.lead_time_periods)
        # convert datetime obj to nanosecondes
        init_time_list_dt = [np.datetime64(date.strftime('%Y-%m-%d %H:%M:%S')) for date in self.init_datetime[index]]
        
        # init_time_list_np: a list of python datetime objects, each is a forecast step
        # init_time_list_np[0]: the first initialization time
        # init_time_list_np[t]: the forcasted time of the (t-1)th step; the initialization time of the t-th step
        self.init_time_list_np = [np.datetime64(str(dt_obj) + '.000000000').astype(datetime) for dt_obj in init_time_list_dt]

        info = []
        for init_time in self.init_time_list_np:
            for i_file, ds in enumerate(self.all_files):
                # get the year of the current file
                ds_year = int(np.datetime_as_string(ds['time'][0].values, unit='Y'))
    
                # get the first and last years of init times
                init_year0 = nanoseconds_to_year(init_time)
    
                # found the right yearly file
                if init_year0 == ds_year:
                    
                    N_times = len(ds['time'])
                    # convert ds['time'] to a list of nanosecondes
                    ds_time_list = [np.datetime64(ds_time.values).astype(datetime) for ds_time in ds['time']]
                    ds_start_time = ds_time_list[0]
                    ds_end_time = ds_time_list[-1]
    
                    init_time_start = init_time
                    # if initalization time is within this (yearly) xr.Dataset
                    if ds_start_time <= init_time_start <= ds_end_time:
    
                        # try getting the index of the first initalization time
                        i_init_start = ds_time_list.index(init_time_start)
    
                        # for multiple init time inputs (history_len > 1), init_end is different for init_start
                        init_time_end = init_time_start + hour_to_nanoseconds(shifted_hours)
    
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

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self,
                                     num_replicas=num_workers*self.world_size,
                                     rank=self.rank*num_workers+worker_id,
                                     shuffle=False)
        for index in sampler:
            # get the init time info for the current sample
            data_lookup = self.find_start_stop_indices(index)

            for k, _ in enumerate(self.init_time_list_np):

                # the first initialization time: get initalization from data
                i_file, i_init_start, i_init_end, N_times = data_lookup[k]

                # allocate output dict
                output_dict = {}

                # get all inputs in one xr.Dataset
                sliced_x = self.load_zarr_as_input(i_file, i_init_start, i_init_end)
                #print(i_file, i_init_start, i_init_end, N_times)
                #print(sliced_x['time'])
                
                # Check if additional data from the next file is needed
                if (len(sliced_x['time']) < self.history_len) or (i_init_end+1 >= N_times):

                    # Load excess data from the next file
                    next_file_idx = self.filenames.index(self.filenames[i_file]) + 1

                    if next_file_idx >= len(self.filenames):
                        # not enough input data to support this forecast
                        raise OSError("You have reached the end of the available data. Exiting.")

                    else:
                        # i_init_start = 0 because we need the beginning of the next file only
                        sliced_x_next = self.load_zarr_as_input(next_file_idx, 0, self.history_len)

                        # Concatenate excess data from the next file with the current data
                        sliced_xy = xr.concat([sliced_x, sliced_x_next], dim='time')
                        sliced_x = sliced_xy.isel(time=slice(0, self.history_len))
                        
                        sliced_y = sliced_xy.isel(time=slice(self.history_len, self.history_len+1))
                        #self.load_zarr_as_input(next_file_idx, self.history_len, self.history_len+1)

                else:
                    sliced_y = self.load_zarr_as_input(i_file, i_init_end+1, i_init_end+1)

                # Prepare data for transform application
                # print(sliced_x['time'])
                # print(sliced_y['time'])
                
                sample_x = {'historical_ERA5_images': sliced_x, 'target_ERA5_images': sliced_y}

                if self.transform:
                    sample_x = self.transform(sample_x)

                for key in sample_x.keys():
                    output_dict[key] = sample_x[key]

                # <--- !! 'forecast_hour' is actually "forecast_step" but named by assuming hourly
                output_dict['forecast_hour'] = k + 1
                # Adjust stopping condition
                output_dict['stop_forecast'] = k == (len(self.init_time_list_np) - 1)
                output_dict['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]
                
                # return output_dict
                yield output_dict

                if output_dict['stop_forecast']:
                    break


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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
    lead_time_periods = conf['data']['lead_time_periods']

    # transform and ToTensor class
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == 'std_new':
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print('Scaler type {} not supported'.format(conf["data"]["scaler_type"]))
        raise
    # ----------------------------------------------------------------- #
    # parse varnames and save_locs from config

    # upper air variables
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))
    varname_upper_air = conf['data']['variables']

    # surface variables
    varname_surface = conf['data']['surface_variables']

    if conf["data"]['flag_surface']:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))
    else:
        surface_files = None

    # dynamic forcing variables
    varname_dyn_forcing = conf['data']['dynamic_forcing_variables']

    if conf["data"]['flag_dyn_forcing']:
        dyn_forcing_files = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
    else:
        dyn_forcing_files = None

    # forcing variables
    forcing_files = conf['data']['save_loc_forcing']
    varname_forcing = conf['data']['forcing_variables']

    # static variables
    static_files = conf['data']['save_loc_static']
    varname_static = conf['data']['static_variables']

    # ----------------------------------------------------------------- #\
    # get dataset
    dataset = Predict_Dataset_Metrics(
        conf,
        varname_upper_air,
        varname_surface,
        varname_dyn_forcing,
        varname_forcing,
        varname_static,
        filenames=all_ERA_files,
        filename_surface=surface_files,
        filename_dyn_forcing=dyn_forcing_files,
        filename_forcing=forcing_files,
        filename_static=static_files,
        fcst_datetime=load_forecasts(conf),
        history_len=history_len,
        rank=rank,
        world_size=world_size,
        transform=transform,
        rollout_p=0.0,
        which_forecast=None
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
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0

        # y_pred allocation
        y_pred = None
        static = None
        results = []

        # model inference loop
        for k, batch in enumerate(data_loader):

            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            # initialization on the first forecast hour
            if forecast_hour == 1:

                # Initialize x and x_surf with the first time step
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(device).float()

                init_datetime = datetime.utcfromtimestamp(date_time)
                init_datetime_str = init_datetime.strftime('%Y-%m-%dT%HZ')

            # -------------------------------------------------------------------------------------- #
            # add forcing and static variables (regardless of fcst hours)
            if 'x_forcing_static' in batch:

                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = batch['x_forcing_static'].to(device).permute(0, 2, 1, 3, 4).float()

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
            y_pred = model(x)
            y_pred = state_transformer.inverse_transform(y_pred.cpu())
            y = state_transformer.inverse_transform(y.cpu())

            if ("use_laplace_filter" in conf["predict"] and conf["predict"]["use_laplace_filter"]):
                y_pred = (
                    dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .cpu()
                )

            # Compute metrics
            metrics_dict = metrics(y_pred.float(), y.float(), forecast_datetime=forecast_hour)
            for k, m in metrics_dict.items():
                metrics_results[k].append(m.item())
            metrics_results["forecast_hour"].append(forecast_hour)

            # Save the current forecast hour data in parallel
            utc_datetime = init_datetime + timedelta(hours=lead_time_periods*forecast_hour)

            # convert the current step result as x-array
            darray_upper_air, darray_single_level = make_xarray(
                y_pred,
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
                     lead_time_periods*forecast_hour, 
                     meta_data, 
                     conf
                )
            )
            results.append(result)
            
            
            metrics_results["datetime"].append(utc_datetime)

            print_str = f"Forecast: {forecast_count} "
            print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
            print_str += f"Hour: {batch['forecast_hour'].item()} "
            print_str += f"ACC: {metrics_dict['acc']} "

            # Update the input
            # setup for next iteration, transform to z-space and send to device
            y_pred = state_transformer.transform_array(y_pred).to(device)

            if history_len == 1:
                x = y_pred.detach()
            else:
                # use multiple past forecast steps as inputs
                # static channels will get updated on next pass
                static_dim_size = abs(x.shape[1] - y_pred.shape[1])

                # if static_dim_size=0 then :0 gives empty range
                x_detach = x[:, :-static_dim_size, 1:].detach() if static_dim_size else x[:, :, 1:].detach()
                x = torch.cat([x_detach, y_pred.detach()], dim=2)

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            if batch["stop_forecast"][0]:
                # Wait for all processes to finish in order
                for result in results:
                    result.get()
                    
                # save metrics file
                save_location = os.path.join(os.path.expandvars(conf["save_loc"]), "forecasts", "metrics")
                os.makedirs(save_location, exist_ok=True)  # should already be made above
                df = pd.DataFrame(metrics_results)
                df.to_csv(os.path.join(save_location, f"metrics{init_datetime_str}.csv"))

                # forecast count = a constant for each run
                forecast_count += 1

                # y_pred allocation
                y_pred = None

                gc.collect()

                if distributed:
                    torch.distributed.barrier()

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
    if conf['data']['scaler_type'] == 'std_new':
        conf = CREDIT_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
        predict_data_check(conf, print_summary=False)
    # ======================================================== #

    # create a save location for rollout
    # ---------------------------------------------------- #
    assert 'save_forecast' in conf['predict'], "Please specify the output dir through conf['predict']['save_forecast']"

    forecast_save_loc = conf['predict']['save_forecast']
    os.makedirs(forecast_save_loc, exist_ok=True)

    print('Save roll-outs to {}'.format(forecast_save_loc))

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
            _ = predict(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
