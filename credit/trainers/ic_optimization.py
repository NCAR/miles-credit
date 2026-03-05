import torch
import tqdm
import numpy as np
from collections import defaultdict
import gc
from itertools import cycle
from credit.trainers.utils import accum_log
import torch.distributed as dist
import logging
import pandas as pd
from torch.utils.data import IterableDataset
from credit.trainers.base_trainer import BaseTrainer
from credit.data import concat_and_reshape, reshape_only
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from torch.optim.lr_scheduler import CosineAnnealingLR  # , CosineAnnealingWarmRestarts
from credit.output import load_metadata, make_xarray
from credit.transforms import Normalize_ERA5_and_Forcing
from datetime import datetime as date_time, timedelta
import xarray as xr
import traceback
import optuna
import os

from credit.data import drop_var_from_dataset
from credit.interp import full_state_pressure_interpolation
from inspect import signature

logger = logging.getLogger(__name__)


def save_netcdf_increment(
    darray_upper_air: xr.DataArray,
    darray_single_level: xr.DataArray,
    nc_filename: str,
    forecast_hour: int,
    meta_data: dict,
    conf: dict,
    name_tag: str,
):
    """
    Save CREDIT model prediction output to netCDF file. Also performs pressure level
    interpolation on the output if you wish.

    Args:
        darray_upper_air (xr.DataArray): upper air variable predictions
        darray_single_level (xr.DataArray): surface variable predictions
        nc_filename (str): file description to go into output filenames
        forecast_hour (int):  how many hours since the initialization of the model.
        meta_data (dict): metadata dictionary for output variables
        conf (dict): configuration dictionary for training and/or rollout

    """
    try:
        """
        Save increment to a unique NetCDF file using Dask for parallel processing.
        """
        # Convert DataArrays to Datasets
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")

        # Merge datasets
        ds_merged = xr.merge([ds_upper, ds_single])

        # Add forecast_hour coordinate
        ds_merged["forecast_hour"] = forecast_hour

        # Add CF convention version
        ds_merged.attrs["Conventions"] = "CF-1.11"

        sig = signature(full_state_pressure_interpolation)
        pres_end = sig.parameters["pres_ending"].default
        height_end = sig.parameters["height_ending"].default
        if "interp_pressure" in conf["predict"].keys():
            if "surface_geopotential_var" in conf["predict"]["interp_pressure"].keys():
                surface_geopotential_var = conf["predict"]["interp_pressure"]["surface_geopotential_var"]
            else:
                surface_geopotential_var = "Z_GDS4_SFC"
            if "pres_ending" in conf["predict"]["interp_pressure"]:
                pres_end = conf["predict"]["interp_pressure"]["pres_ending"]
            if "height_ending" in conf["predict"]["interp_pressure"]:
                height_end = conf["predict"]["interp_pressure"]["height_ending"]

            with xr.open_dataset(conf["predict"]["static_fields"]) as static_ds:
                surface_geopotential = static_ds[surface_geopotential_var].values
            pressure_interp = full_state_pressure_interpolation(
                ds_merged, surface_geopotential, **conf["predict"]["interp_pressure"]
            )
            ds_merged = xr.merge([ds_merged, pressure_interp])

        # logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)

        unique_filename = os.path.join(save_location, f"{name_tag}.nc")
        # ---------------------------------------------------- #
        # If conf['predict']['save_vars'] provided --> drop useless vars
        if "save_vars" in conf["predict"]:
            if len(conf["predict"]["save_vars"]) > 0:
                ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

        # when there's no metafile --> meta_data = False
        if meta_data is not False:
            # Add metadata attributes to every model variable if available
            for var in ds_merged.variables.keys():
                if var in meta_data.keys():
                    if var != "time":
                        # use attrs.update for non-datetime variables
                        ds_merged[var].attrs.update(meta_data[var])
                    else:
                        # use time.encoding for datetime variables/coords
                        for metadata_time in meta_data["time"]:
                            ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
                if "interp_pressure" in conf["predict"].keys():
                    if pres_end in var:
                        var_short = var.strip(pres_end)
                        if var_short in meta_data.keys():
                            ds_merged[var].attrs.update(meta_data[var_short])
                            ds_merged[var].attrs["long_name"] += " (interpolated to isobaric levels)"
                    elif height_end in var:
                        var_short = var.strip(height_end)
                        if var_short in meta_data.keys():
                            ds_merged[var].attrs.update(meta_data[var_short])
                            ds_merged[var].attrs["long_name"] += " (interpolated to constant height AGL levels)"
        encoding_dict = {}
        if "ua_var_encoding" in conf["predict"].keys():
            for ua_var in conf["data"]["variables"]:
                encoding_dict[ua_var] = conf["predict"]["ua_var_encoding"]
        if "surface_var_encoding" in conf["predict"].keys():
            for surface_var in conf["data"]["surface_variables"]:
                encoding_dict[surface_var] = conf["predict"]["surface_var_encoding"]
        if "pressure_var_encoding" in conf["predict"].keys():
            for pres_var in conf["data"]["variables"]:
                encoding_dict[pres_var + pres_end] = conf["predict"]["pressure_var_encoding"]
        if "height_var_encoding" in conf["predict"].keys():
            for height_var in conf["data"]["variables"]:
                encoding_dict[height_var + height_end] = conf["predict"]["height_var_encoding"]
        # Use Dask to write the dataset in parallel
        ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)

        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")
    except Exception:
        print(traceback.format_exc())


class ForecastProcessor:
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device

        self.batch_size = conf["trainer"].get("batch_size", 1)
        self.ensemble_size = conf["trainer"].get("ensemble_size", 1)
        self.lead_time_periods = conf["data"]["lead_time_periods"]

        # transform and ToTensor class
        if conf["data"]["scaler_type"] == "std_new":
            self.state_transformer = Normalize_ERA5_and_Forcing(conf)
        else:
            print("Scaler type {} not supported".format(conf["data"]["scaler_type"]))
            raise

        # get lat/lons from x-array
        self.latlons = xr.open_dataset(conf["loss"]["latitude_weights"]).load()
        # grab ERA5 (etc) metadata
        self.meta_data = load_metadata(conf)

    def process(self, y_pred, datetimes, save_datetimes, nametag):
        try:
            # Transform predictions
            conf = self.conf
            y_pred = self.state_transformer.inverse_transform(y_pred)

            # Calculate correct datetime for current forecast
            utc_datetimes = date_time.utcfromtimestamp(datetimes) + timedelta(hours=self.lead_time_periods)

            # Convert to xarray and handle results
            upper_air_list, single_level_list = [], []
            darray_upper_air, darray_single_level = make_xarray(
                y_pred,
                utc_datetimes,
                self.latlons.latitude.values,
                self.latlons.longitude.values,
                conf,
            )
            upper_air_list.append(darray_upper_air)
            single_level_list.append(darray_single_level)

            all_upper_air = darray_upper_air
            all_single_level = darray_single_level

            # Save the current forecast hour data in parallel
            save_netcdf_increment(
                all_upper_air,
                all_single_level,
                save_datetimes,  # Use correct index for current batch item
                self.lead_time_periods,
                self.meta_data,
                conf,
                nametag,
            )
        except Exception as e:
            print(traceback.format_exc())
            raise e


class TimeStepper:
    def __init__(self, dataset):
        self.dataset = dataset
        self._active = False

    def __iter__(self):
        return self

    def reset(self, idx=0):
        """Initialize new sample starting from forecast step 0."""
        self.dataset.current_batch_indices = [idx]
        self.dataset.time_steps = [0]
        self.dataset.forecast_step_counts = [0]
        self._active = True

    def __next__(self):
        """Advance forecast steps until forecast_len + 1."""
        if not self._active or self.dataset.forecast_step_counts[0] > self.dataset.forecast_len:
            raise StopIteration

        return self.dataset[0]  # __getitem__ uses forecast_step_counts[0] internally


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        """
        Trainer class for handling the training, validation, and checkpointing of models.

        This class is responsible for executing the training loop, validating the model
        on a separate dataset, and managing checkpoints during training. It supports
        both single-GPU and distributed (FSDP, DDP) training.

        Attributes:
            model (torch.nn.Module): The model to be trained.
            rank (int): The rank of the process in distributed training.


        Methods:
            train_one_epoch(epoch, conf, trainloader, optimizer, criterion, scaler,
                            scheduler, metrics):
                Perform training for one epoch and return training metrics.

            validate(epoch, conf, valid_loader, criterion, metrics):
                Validate the model on the validation dataset and return validation metrics.

            fit_deprecated(conf, train_loader, valid_loader, optimizer, train_criterion,
                           valid_criterion, scaler, scheduler, metrics, trial=False):
                Perform the full training loop across multiple epochs, including validation
                and checkpointing.
        """
        super().__init__(model, rank)
        # Add any additional initialization if needed
        logger.info("Loading a multi-step trainer class")

    # Training function.
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            conf (dict): Configuration dictionary containing training settings.
            trainloader (DataLoader): DataLoader for the training dataset.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            criterion (callable): Loss function used for training.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            metrics (callable): Function to compute metrics for evaluation.

        Returns:
            dict: Dictionary containing training metrics and loss for the epoch.
        """

        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_max_norm = conf["trainer"].get("grad_max_norm", 0.0)
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        forecast_length = conf["data"]["forecast_len"]
        ensemble_size = conf["trainer"].get("ensemble_size", 1)
        precision = conf["trainer"].get("precision", "float32")
        conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])
        if ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {ensemble_size}")
        logger.info(f"Using grad-max-norm value: {grad_max_norm}")

        # number of diagnostic variables
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        # number of dynamic forcing + forcing + static
        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )

        ensemble_size = conf["trainer"].get("ensemble_size", 1)

        # Class for saving in parallel
        result_processor = ForecastProcessor(conf, self.device)

        # [Optional] Use the config option to set when to backprop
        if "backprop_on_timestep" in conf["data"]:
            backprop_on_timestep = conf["data"]["backprop_on_timestep"]
        else:
            # If not specified in config, use the range 1 to forecast_len
            backprop_on_timestep = list(range(0, conf["data"]["forecast_len"] + 1 + 1))

        assert forecast_length <= backprop_on_timestep[-1], (
            f"forecast_length ({forecast_length + 1}) must not exceed the max value in backprop_on_timestep {backprop_on_timestep}"
        )

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        # ------------------------------------------------------- #
        # clamp to remove outliers
        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

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

        # set up a custom tqdm
        if not isinstance(trainloader.dataset, IterableDataset):
            # Check if the dataset has its own batches_per_epoch method
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(trainloader)
            # Use the user-given number if not larger than the dataset
            batches_per_epoch = (
                batches_per_epoch if 0 < batches_per_epoch < dataset_batches_per_epoch else dataset_batches_per_epoch
            )

        # batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        if precision == "float64":
            self.model = self.model.double()

        # Set model to evaluation mode and freeze its parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        dl = TimeStepper(trainloader.dataset)

        for idx in range(batches_per_epoch):
            # idx = random.randint(0, len(trainloader.dataset) - conf["data"]["forecast_len"] + 1)

            # Initialize x0 with the first time step (this is what we optimize)
            dl.reset(idx=idx)
            batch = next(dl)

            if "x_surf" in batch:
                x0 = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
            else:
                x0 = reshape_only(batch["x"]).to(self.device)

            if precision == "float64":
                x0 = x0.double()

            x0 = x0.requires_grad_(True)

            # Create optimizer for x0 (this is what we're optimizing)
            optimizer = torch.optim.AdamW(
                [x0], lr=conf["trainer"]["learning_rate"], weight_decay=conf["trainer"]["weight_decay"]
            )

            init_datetimes = date_time.utcfromtimestamp(batch["datetime"][0].item()).strftime("%Y-%m-%dT%HZ")
            save_datetimes = init_datetimes

            # Progressive window optimization
            current_window_size = conf["trainer"]["starting_window_size"]  # Start with 2 days as per paper
            max_window_size = conf["data"]["forecast_len"] + 1
            batch_iterations = conf["trainer"]["batch_iterations"]
            window_size = conf["trainer"]["window_size"]
            num_windows = (max_window_size - current_window_size) // window_size + 1

            # Decay the scheduler
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_windows * batch_iterations - 1,
                eta_min=1e-4 * conf["trainer"]["learning_rate"],
            )
            # scheduler = CosineAnnealingWarmRestarts(
            #     optimizer,
            #     T_0=batch_iterations,  # restart every stage (window)
            #     T_mult=1,
            #     eta_min=1e-3 * conf["trainer"]["learning_rate"],
            # )

            batch_group_generator = tqdm.tqdm(
                range(num_windows * batch_iterations), total=num_windows * batch_iterations, leave=True
            )
            datetime = batch["datetime"]

            # Optimize for this window size
            results_dict = defaultdict(list)

            while current_window_size <= max_window_size:
                for _ in range(batch_iterations):  # Number of epochs for this window
                    optimizer.zero_grad()

                    logs = {}
                    total_loss = 0
                    stop_forecast = False
                    y_pred = None

                    # Reset to start of sequence
                    dl.reset(idx=idx)

                    # Start forecast from optimized x0
                    x = x0.clone()

                    # Ensemble x if needed
                    if ensemble_size > 1:
                        x = torch.repeat_interleave(x, ensemble_size, 0)

                    forecast_step = 0
                    while not stop_forecast and forecast_step < current_window_size:
                        batch = next(dl)
                        forecast_step = batch["forecast_step"].item()

                        # Add forcing and static variables
                        if "x_forcing_static" in batch:
                            x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                            if ensemble_size > 1:
                                x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
                            if precision == "float64":
                                x_forcing_batch = x_forcing_batch.double()
                            x = torch.cat((x, x_forcing_batch), dim=1)

                        # Clamp if needed
                        if flag_clamp:
                            x = torch.clamp(x, min=clamp_min, max=clamp_max)

                        # Predict with the model
                        with torch.autocast(device_type="cuda", enabled=amp):
                            y_pred = (
                                self.model(x, forecast_step=forecast_step - 1)
                                if conf["model"]["type"] == "crossformer-noisy"
                                else self.model(x)
                            )

                        # Apply conservation constraints
                        if flag_mass_conserve:
                            if forecast_step == 1:
                                x_init = x.clone()
                            input_dict = {"y_pred": y_pred, "x": x_init}
                            input_dict = opt_mass(input_dict)
                            y_pred = input_dict["y_pred"]

                        if flag_water_conserve:
                            input_dict = {"y_pred": y_pred, "x": x}
                            input_dict = opt_water(input_dict)
                            y_pred = input_dict["y_pred"]

                        if flag_energy_conserve:
                            input_dict = {"y_pred": y_pred, "x": x}
                            input_dict = opt_energy(input_dict)
                            y_pred = input_dict["y_pred"]

                        # Calculate loss against target (only if within current window)
                        if forecast_step <= current_window_size:
                            # Load target data
                            if "y_surf" in batch:
                                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                            else:
                                y = reshape_only(batch["y"]).to(self.device)

                            if "y_diag" in batch:
                                y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                                y = torch.cat((y, y_diag_batch), dim=1)

                            if flag_clamp:
                                y = torch.clamp(y, min=clamp_min, max=clamp_max)

                            with torch.autocast(enabled=amp, device_type="cuda"):
                                if precision == "float64":
                                    y = y.double()
                                loss = criterion(y, y_pred).mean()
                                total_loss += loss

                            # print(
                            #     forecast_step,
                            #     current_window_size,
                            #     max_window_size,
                            #     loss.item(),
                            #     current_window_size <= (max_window_size + 1),
                            # )

                            accum_log(logs, {"loss": loss.item()})

                        # Check if we should stop
                        stop_forecast = batch["stop_forecast"].item() or forecast_step >= current_window_size
                        if stop_forecast:
                            break

                        # Prepare input for next time step
                        if x.shape[2] == 1:
                            # Single timestep input
                            if "y_diag" in batch:
                                x = y_pred[:, :-varnum_diag, ...]  # .detach()
                            else:
                                x = y_pred  # .detach()
                        else:
                            # Multi-timestep input
                            if static_dim_size == 0:
                                x_detach = x[:, :, 1:, ...]  # .detach()
                            else:
                                x_detach = x[:, :-static_dim_size, 1:, ...]  # .detach()

                            if "y_diag" in batch:
                                x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...]], dim=2)
                            else:
                                x = torch.cat([x_detach, y_pred], dim=2)

                    # Backpropagation
                    if total_loss > 0:
                        # Scale loss by window size for consistency
                        scaled_loss = total_loss / current_window_size

                        with torch.autocast(enabled=amp, device_type="cuda"):
                            scaler.scale(scaled_loss).backward()

                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        if grad_max_norm == "dynamic":
                            local_norm = x0.grad.detach().norm(2)
                            if distributed:
                                dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                            global_norm = local_norm.sqrt()
                            torch.nn.utils.clip_grad_norm_([x0], max_norm=global_norm)
                        elif grad_max_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_([x0], max_norm=grad_max_norm)

                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()

                        # print(f"Window {current_window_size}, Iteration {iteration}, Loss: {scaled_loss.item()}")

                    if distributed:
                        torch.distributed.barrier()

                    # Metrics
                    metrics_dict = metrics(y_pred, y)
                    for name, value in metrics_dict.items():
                        value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                        if distributed:
                            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                        results_dict[f"train_{name}"].append(value[0].item())

                    batch_loss = torch.Tensor([scaled_loss]).cuda(self.device)
                    if distributed:
                        dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                    results_dict["train_loss"].append(batch_loss[0].item())
                    results_dict["train_forecast_len"].append(current_window_size)

                    if not np.isfinite(np.mean(results_dict["train_loss"])):
                        print(
                            results_dict["train_loss"],
                            batch["x"].shape,
                            batch["y"].shape,
                            batch["index"],
                        )
                        try:
                            raise optuna.TrialPruned()
                        except Exception as E:
                            raise E

                    # agg the results
                    df = pd.DataFrame.from_dict(results_dict).reset_index()
                    cond = df["train_forecast_len"] == current_window_size
                    to_print = "Epoch: {} IC {}: train_loss: {:.12f} train_acc: {:.12f} train_mae: {:.12f} forecast_len: {:.1f}".format(
                        epoch,
                        idx,
                        np.mean(df["train_loss"][cond]),
                        np.mean(df["train_acc"][cond]),
                        np.mean(df["train_mae"][cond]),
                        float(current_window_size),
                    )
                    if ensemble_size > 1:
                        to_print += f" std: {np.mean(df['train_std']):.6f}"
                    to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
                    if self.rank == 0:
                        batch_group_generator.update(1)
                        batch_group_generator.set_description(to_print)

                    # anneal the learning rate
                    scheduler.step()

                    # Save the metrics df
                    df.to_csv(os.path.join(f"{save_loc}", f"training_log_{datetime[0].item()}.csv"), index=False)

                # Expand window size (following paper's X-day (3-day) increments)
                current_window_size += window_size
                # current_window_size = min(current_window_size, max_window_size)

                # Optionally reduce learning rate for longer windows (14 days at 6 hour resolution)
                if current_window_size > 14 * 4:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.5  # Reduce learning rate as in paper

            # Save x optimized first
            _ = result_processor.process(x0.detach().cpu(), datetime[0], save_datetimes, "optimized")

            # Now save the original state for convenience
            dl.reset(idx=idx)
            batch = next(dl)

            if "x_surf" in batch:
                x0 = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
            else:
                x0 = reshape_only(batch["x"]).to(self.device)

            _ = result_processor.process(x0.cpu(), datetime[0], save_datetimes, "initial")

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        """
        Validates the model on the validation dataset.

        Args:
            epoch (int): Current epoch number.
            conf (dict): Configuration dictionary containing validation settings.
            valid_loader (DataLoader): DataLoader for the validation dataset.
            criterion (callable): Loss function used for validation.
            metrics (callable): Function to compute metrics for evaluation.

        Returns:
            dict: Dictionary containing validation metrics and loss for the epoch.
        """

        self.model.eval()

        # number of diagnostic variables
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        # number of dynamic forcing + forcing + static
        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        history_len = conf["data"]["valid_history_len"] if "valid_history_len" in conf["data"] else conf["history_len"]
        forecast_len = (
            conf["data"]["valid_forecast_len"] if "valid_forecast_len" in conf["data"] else conf["forecast_len"]
        )
        ensemble_size = conf["trainer"].get("ensemble_size", 1)

        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        if not isinstance(valid_loader.dataset, IterableDataset):
            # Check if the dataset has its own batches_per_epoch method
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(valid_loader)
            # Use the user-given number if not larger than the dataset
            valid_batches_per_epoch = (
                valid_batches_per_epoch
                if 0 < valid_batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        # ------------------------------------------------------- #
        # clamp to remove outliers
        if conf["data"]["data_clamp"] is None:
            flag_clamp = False
        else:
            flag_clamp = True
            clamp_min = float(conf["data"]["data_clamp"][0])
            clamp_max = float(conf["data"]["data_clamp"][1])

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

        batch_group_generator = tqdm.tqdm(range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True)

        stop_forecast = False
        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                loss = 0
                stop_forecast = False
                y_pred = None  # Place holder that gets updated after first roll-out
                while not stop_forecast:
                    batch = next(dl)
                    forecast_step = batch["forecast_step"].item()
                    stop_forecast = batch["stop_forecast"].item()
                    if forecast_step == 1:
                        # Initialize x and x_surf with the first time step
                        if "x_surf" in batch:
                            # combine x and x_surf
                            # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                            # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)  # .float()
                        else:
                            # no x_surf
                            x = reshape_only(batch["x"]).to(self.device)  # .float()
                        # --------------------------------------------- #
                        # ensemble x and x_surf on initialization
                        # copies each sample in the batch ensemble_size number of times.
                        # if samples in the batch are ordered (x,y,z) then the result tensor is (x, x, ..., y, y, ..., z,z ...)
                        # WARNING: needs to be used with a loss that can handle x with b * ensemble_size samples and y with b samples
                        if ensemble_size > 1:
                            x = torch.repeat_interleave(x, ensemble_size, 0)

                    # add forcing and static variables (regardless of fcst hours)
                    if "x_forcing_static" in batch:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)  # .float()
                        # ---------------- ensemble ----------------- #
                        # ensemble x_forcing_batch for concat. see above for explanation of code
                        if ensemble_size > 1:
                            x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
                        # --------------------------------------------- #

                        # concat on var dimension
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    # --------------------------------------------- #
                    # clamp
                    if flag_clamp:
                        x = torch.clamp(x, min=clamp_min, max=clamp_max)

                    y_pred = (
                        self.model(x, forecast_step=forecast_step - 1)
                        if conf["model"]["type"] == "crossformer-noisy"
                        else self.model(x)
                    )

                    # ============================================= #
                    # postblock opts outside of model

                    # backup init state
                    if flag_mass_conserve:
                        if forecast_step == 1:
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

                    # ================================================================================== #
                    # scope of reaching the final forecast_len
                    if forecast_step == (forecast_len + 1):
                        # ----------------------------------------------------------------------- #
                        # creating `y` tensor for loss compute
                        if "y_surf" in batch:
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                        else:
                            y = reshape_only(batch["y"]).to(self.device)

                        if "y_diag" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)  # .float()

                            # concat on var dimension
                            y = torch.cat((y, y_diag_batch), dim=1)

                        # --------------------------------------------- #
                        # clamp
                        if flag_clamp:
                            y = torch.clamp(y, min=clamp_min, max=clamp_max)

                        # ----------------------------------------------------------------------- #
                        # calculate rolling loss
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # Metrics
                        # metrics_dict = metrics(y_pred, y.float)
                        metrics_dict = metrics(y_pred.float(), y.float())

                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).cuda(self.device, non_blocking=True)

                            if distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)

                            results_dict[f"valid_{name}"].append(value[0].item())

                        assert stop_forecast
                        break  # stop after X steps

                    # ================================================================================== #
                    # scope of keep rolling out

                    # step-in-step-out
                    elif history_len == 1:
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
                                [x_detach, y_pred[:, :-varnum_diag, ...].detach()],
                                dim=2,
                            )
                        else:
                            x = torch.cat([x_detach, y_pred.detach()], dim=2)

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)

                if distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                stop_forecast = False

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                ensemble_size = conf["trainer"].get("ensemble_size", 0)
                if ensemble_size > 1:
                    to_print += f" std: {np.mean(results_dict['valid_std']):.6f}"
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
