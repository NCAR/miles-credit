# ---------- #
# System
import gc
import os
import sys
import yaml
import glob
import logging
import warnings
import traceback
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp

# ---------- #
# Numerics
import datetime
import pandas as pd
import xarray as xr
import numpy as np

# ---------- #
# AI libs
import torch
import torch.distributed as dist
from torchvision import transforms
# import wandb

# ---------- #
# credit
from credit.data import ERA5Dataset
from credit.models import load_model
from credit.transforms import ToTensor, NormalizeState, NormalizeState_Quantile
from credit.seed import seed_everything
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper
from credit.models.checkpoint import load_model_state
from credit.solar import TOADataLoader
from credit.output import split_and_reshape, load_metadata, make_xarray, save_netcdf_increment
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class PredictForecast(torch.utils.data.IterableDataset):
    def __init__(self,
                 filenames,
                 forecasts,
                 history_len,
                 skip_periods,
                 rank,
                 world_size,
                 shuffle=False,
                 transform=None,
                 rollout_p=0.0,
                 which_forecast=None):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=1,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_files = self.dataset.all_fils
        self.history_len = history_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p
        self.forecasts = forecasts
        self.skip_periods = skip_periods if skip_periods is not None else 1
        self.which_forecast = which_forecast

    def find_start_stop_indices(self, index):
        start_time = self.forecasts[index][0]
        date_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        shifted_hours = self.skip_periods * self.history_len
        date_object = date_object - datetime.timedelta(hours=shifted_hours)
        self.forecasts[index][0] = date_object.strftime('%Y-%m-%d %H:%M:%S')

        datetime_objs = [np.datetime64(date) for date in self.forecasts[index]]
        start_time, stop_time = [str(datetime_obj) + '.000000000' for datetime_obj in datetime_objs]
        self.start_time = np.datetime64(start_time).astype(datetime.datetime)
        self.stop_time = np.datetime64(stop_time).astype(datetime.datetime)

        info = {}

        for idx, dataset in enumerate(self.all_files):
            start_time = np.datetime64(dataset['time'].min().values).astype(datetime.datetime)
            stop_time = np.datetime64(dataset['time'].max().values).astype(datetime.datetime)
            track_start = False
            track_stop = False

            if start_time <= self.start_time <= stop_time:
                # Start time is in this file, use start time index
                dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                start_idx = np.searchsorted(dataset, self.start_time)
                start_idx = max(0, min(start_idx, len(dataset)-1))
                track_start = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                start_idx = 0
                track_start = True

            if start_time <= self.stop_time <= stop_time:
                # Stop time is in this file, use stop time index
                if isinstance(dataset, np.ndarray):
                    pass
                else:
                    dataset = np.array([np.datetime64(x.values).astype(datetime.datetime) for x in dataset['time']])
                stop_idx = np.searchsorted(dataset, self.stop_time)
                stop_idx = max(0, min(stop_idx, len(dataset)-1))
                track_stop = True

            elif start_time < self.stop_time and stop_time > self.start_time:
                # File overlaps time range, use full file
                stop_idx = len(dataset) - 1
                track_stop = True

            # Only include files that overlap the time range
            if track_start and track_stop:
                info[idx] = ((idx, start_idx), (idx, stop_idx))

        indices = []
        for dataset_idx, (start, stop) in info.items():
            for i in range(start[1], stop[1]+1):
                indices.append((start[0], i))
        return indices

    def __len__(self):
        return len(self.forecasts)

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers*self.world_size, rank=self.rank*num_workers+worker_id, shuffle=self.shuffle)

        for index in sampler:

            data_lookup = self.find_start_stop_indices(index)

            for k, (file_key, time_key) in enumerate(data_lookup):

                if k == 0:
                    concatenated_samples = {'x': [], 'x_surf': []}
                    sliced_x = xr.open_zarr(self.filenames[file_key], consolidated=True).isel(time=slice(time_key, time_key+self.history_len+1))

                    # Check if additional data from the next file is needed
                    if len(sliced_x['time']) < self.history_len + 1:
                        # Load excess data from the next file
                        next_file_idx = self.filenames.index(self.filenames[file_key]) + 1
                        if next_file_idx == len(self.filenames):
                            raise OSError("You have reached the end of the available data. Exiting.")
                        sliced_x_next = xr.open_zarr(
                            self.filenames[next_file_idx],
                            consolidated=True).isel(time=slice(0, self.history_len+1-len(sliced_x['time'])))

                        # Concatenate excess data from the next file with the current data
                        sliced_x = xr.concat([sliced_x, sliced_x_next], dim='time')

                    sample_x = {
                        'x': sliced_x.isel(time=slice(0, self.history_len))
                    }

                    if self.transform:
                        sample_x = self.transform(sample_x)
                        # Add static vars, if any, to the return dictionary
                        if "static" in sample_x:
                            concatenated_samples["static"] = []
                        if "TOA" in sample_x:
                            concatenated_samples["TOA"] = []

                    for key in concatenated_samples.keys():
                        concatenated_samples[key] = sample_x[key].squeeze(0) if self.history_len == 1 else sample_x[key]

                    concatenated_samples['forecast_hour'] = k + 1
                    concatenated_samples['stop_forecast'] = (k == (len(data_lookup)-self.history_len-1))  # Adjust stopping condition
                    concatenated_samples['datetime'] = sliced_x.time.values.astype('datetime64[s]').astype(int)[-1]

                else:
                    concatenated_samples['forecast_hour'] = k + 1
                    concatenated_samples['stop_forecast'] = (k == (len(data_lookup)-self.history_len-1))  # Adjust stopping condition

                yield concatenated_samples

                if concatenated_samples['stop_forecast']:
                    break


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def predict(rank, world_size, conf, p):

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # Config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    history_len = conf["data"]["history_len"]
    time_step = conf["data"]["time_step"] if "time_step" in conf["data"] else None

    # Load paths to all ERA5 data available
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))

    # Preprocessing transformations
    if conf["data"]["scaler_type"] == "std":
        state_transformer = NormalizeState(conf)
    else:
        state_transformer = NormalizeState_Quantile(conf)
    transform = transforms.Compose(
        [
            state_transformer,
            ToTensor(conf),
        ]
    )

    dataset = PredictForecast(
        filenames=all_ERA_files,
        forecasts=load_forecasts(conf),
        history_len=history_len,
        skip_periods=time_step,
        transform=transform,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )

    # setup the dataloder for this process
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
    distributed = conf["trainer"]["mode"] in ["ddp", "fsdp"]
    if distributed:  # A new field needs to be added to predict
        model = distributed_model_wrapper(conf, model, device)
        if conf["trainer"]["mode"] == "fsdp":
            # Load model weights (if any), an optimizer, scheduler, and gradient scaler
            model = load_model_state(conf, model, device)

    model.eval()

    # get lat/lons from x-array
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])

    meta_data = load_metadata("era5")

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
                x = model.concat_and_reshape(batch["x"], batch["x_surf"]).to(device)

                init_datetime_str = datetime.datetime.utcfromtimestamp(date_time)
                init_datetime_str = init_datetime_str.strftime('%Y-%m-%dT%HZ')

            # Add statics
            if "static" in batch:
                if static is None:
                    static = batch["static"].to(device).unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1).float()
                x = torch.cat((x, static.clone()), dim=1)

            # Add solar "statics"
            if "static_variables" in conf["data"] and "tsi" in conf["data"]["static_variables"]:
                if k == 0:
                    toaDL = TOADataLoader(conf)
                elapsed_time = pd.Timedelta(hours=k)
                tnow = pd.to_datetime(datetime.datetime.utcfromtimestamp(batch["datetime"]))
                tnow = tnow + elapsed_time
                if history_len == 1:
                    current_times = [pd.to_datetime(datetime.datetime.utcfromtimestamp(_t)) + elapsed_time for _t in tnow]
                else:
                    current_times = [tnow if hl == 0 else tnow - pd.Timedelta(hours=hl) for hl in range(history_len)]

                toa = torch.cat([toaDL(_t) for _t in current_times], dim=0).to(device)
                toa = toa.squeeze().unsqueeze(0)
                x = torch.cat([x, toa.unsqueeze(1).to(device).float()], dim=1)

            # Predict and convert to real space for laplace filter and metrics
            y_pred = model(x)
            y_pred = state_transformer.inverse_transform(y_pred.cpu())

            if ("use_laplace_filter" in conf["predict"] and conf["predict"]["use_laplace_filter"]):
                y_pred = (
                    dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .cpu()
                )

            # Save the current forecast hour data in parallel
            utc_datetime = datetime.datetime.utcfromtimestamp(date_time) + datetime.timedelta(hours=forecast_hour)

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
                (darray_upper_air, darray_single_level, init_datetime_str, forecast_hour, meta_data, conf)
            )
            results.append(result)

            # Update the input
            # setup for next iteration, transform to z-space and send to device
            y_pred = state_transformer.transform_array(y_pred).to(device)

            if history_len == 1:
                x = y_pred.detach()
            else:
                # use multiple past forecast steps as inputs
                static_dim_size = abs(x.shape[1] - y_pred.shape[1])  # static channels will get updated on next pass
                x_detach = x[:, :-static_dim_size, 1:].detach() if static_dim_size else x[:, :, 1:].detach()  # if static_dim_size=0 then :0 gives empty range
                x = torch.cat([x_detach, y_pred.detach()], dim=2)

            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            if batch["stop_forecast"][0]:
                # Wait for all processes to finish in order
                for result in results:
                    result.get()

                # Now merge all the files into one and delete leftovers
                # merge_netcdf_files(init_datetime_str, conf)

                # forecast count = a constant for each run
                forecast_count += 1

                # update lists
                results = []

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
    # create a save location for rollout
    forecast_save_loc = os.path.join(os.path.expandvars(conf['save_loc']), 'forecasts')
    os.makedirs(forecast_save_loc, exist_ok=True)

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

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
        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
