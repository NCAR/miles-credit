import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
from credit.scheduler import update_on_batch
from credit.trainers.utils import cycle, accum_log
from credit.trainers.base_trainer import BaseTrainer
from credit.data import concat_and_reshape, reshape_only
import optuna

import os
import pandas as pd
import torch
from credit.models.checkpoint import TorchFSDPCheckpointIO, copy_checkpoint
from credit.scheduler import update_on_epoch
from credit.trainers.utils import cleanup
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer, GlobalDryMassFixer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class for handling the training, validation, and checkpointing of models.

    This class is responsible for executing the training loop, validating the model
    on a separate dataset, and managing checkpoints during training. It supports
    both single-GPU and distributed (FSDP, DDP) training.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        rank (int): The rank of the process in distributed training.
        module (bool): If True, use model with module parallelism (default: False).

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

    def __init__(self, model: torch.nn.Module, rank: int, module: bool = False):
        super().__init__(model, rank, module)
        # Add any additional initialization if needed
        logger.info("Loading a multi-step trainer class")

    # Training function.
    def train_one_epoch(
        self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics
    ):
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
        grad_max_norm = conf["trainer"]["grad_max_norm"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        forecast_length = conf["data"]["forecast_len"]

        # number of diagnostic variables
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        # number of dynamic forcing + forcing + static
        static_dim_size = (
            len(conf["data"]["dynamic_forcing_variables"])
            + len(conf["data"]["forcing_variables"])
            + len(conf["data"]["static_variables"])
        )

        # [Optional] Use the config option to set when to backprop
        if "backprop_on_timestep" in conf["data"]:
            backprop_on_timestep = conf["data"]["backprop_on_timestep"]
        else:
            # If not specified in config, use the range 1 to forecast_len
            backprop_on_timestep = list(range(0, conf["data"]["forecast_len"] + 1 + 1))

        assert (
            forecast_length <= backprop_on_timestep[-1]
        ), f"forecast_length ({forecast_length + 1}) must not exceed the max value in backprop_on_timestep {backprop_on_timestep}"

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
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

            if post_conf["global_drymass_fixer"]["activate"]:
                if post_conf["global_mass_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalDryMassFixer outside of model")
                    flag_mass_conserve = True
                    opt_mass = GlobalDryMassFixer(post_conf)

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
            else:
                dataset_batches_per_epoch = len(trainloader)
            # Use the user-given number if not larger than the dataset
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        self.model.train()

        dl = cycle(trainloader)

        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            stop_forecast = False
            y_pred = None  # Place holder that gets updated after first roll-out

            with autocast(enabled=amp):
                while not stop_forecast:
                    batch = next(dl)

                    for i, forecast_step in enumerate(batch["forecast_step"]):
                        # if self.rank == 0:
                        #     logger.info(f"i: {i}, forecast_step: {forecast_step}")
                        if forecast_step == 1:
                            # Initialize x and x_surf with the first time step
                            if "x_surf" in batch:
                                # combine x and x_surf
                                # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                                # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                                x = concat_and_reshape(batch["x"], batch["x_surf"]).to(
                                    self.device
                                )  # .float()
                            else:
                                # no x_surf
                                x = reshape_only(batch["x"]).to(self.device)  # .float()

                        # add forcing and static variables (regardless of fcst hours)
                        if "x_forcing_static" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            x_forcing_batch = (
                                batch["x_forcing_static"]
                                .to(self.device)
                                .permute(0, 2, 1, 3, 4)
                            )  # .float()
                            
                            # concat on var dimension
                            x = torch.cat((x, x_forcing_batch), dim=1)

                        # --------------------------------------------- #
                        # clamp
                        if flag_clamp:
                            x = torch.clamp(x, min=clamp_min, max=clamp_max)
                        
                        # predict with the model
                        y_pred = self.model(x)

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

                        # only load y-truth data if we intend to backprop (default is every step gets grads computed
                        if forecast_step in backprop_on_timestep:
                            # calculate rolling loss
                            if "y_surf" in batch:
                                y = concat_and_reshape(batch["y"], batch["y_surf"]).to(
                                    self.device
                                )
                            else:
                                y = reshape_only(batch["y"]).to(self.device)

                            if "y_diag" in batch:
                                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                                y_diag_batch = (
                                    batch["y_diag"]
                                    .to(self.device)
                                    .permute(0, 2, 1, 3, 4)
                                )  # .float()

                                # concat on var dimension
                                y = torch.cat((y, y_diag_batch), dim=1)

                            # --------------------------------------------- #
                            # clamp
                            if flag_clamp:
                                y = torch.clamp(y, min=clamp_min, max=clamp_max)
                                
                            loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                            # track the loss
                            accum_log(logs, {"loss": loss.item()})

                            # compute gradients
                            scaler.scale(loss).backward()

                        if distributed:
                            torch.distributed.barrier()

                        # stop after X steps
                        stop_forecast = batch["stop_forecast"][i]

                        # step-in-step-out
                        if x.shape[2] == 1:
                            # cut diagnostic vars from y_pred, they are not inputs
                            if "y_diag" in batch:
                                x = y_pred[:, :-varnum_diag, ...].detach()
                            else:
                                x = y_pred.detach()

                        # multi-step in
                        else:
                            # static channels will get updated on next pass

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

                    if stop_forecast:
                        break

                # scale, accumulate, backward

                if distributed:
                    torch.distributed.barrier()
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Metrics
            # metrics_dict = metrics(y_pred.float(), y.float())
            metrics_dict = metrics(y_pred, y)
            for name, value in metrics_dict.items():
                value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                if distributed:
                    dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(forecast_length + 1)

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
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                forecast_length + 1,
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch
            ):
                scheduler.step()

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
        history_len = (
            conf["data"]["valid_history_len"]
            if "valid_history_len" in conf["data"]
            else conf["history_len"]
        )
        forecast_len = (
            conf["data"]["valid_forecast_len"]
            if "valid_forecast_len" in conf["data"]
            else conf["forecast_len"]
        )
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        results_dict = defaultdict(list)

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = (
                valid_batches_per_epoch
                if 0 < valid_batches_per_epoch < len(valid_loader)
                else len(valid_loader)
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

            if post_conf["global_drymass_fixer"]["activate"]:
                if post_conf["global_drymass_fixer"]["activate_outside_model"]:
                    logger.info("Activate GlobalDryMassFixer outside of model")
                    flag_mass_conserve = True
                    opt_mass = GlobalDryMassFixer(post_conf)

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

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        stop_forecast = False
        with torch.no_grad():
            for k, batch in enumerate(valid_loader):
                y_pred = None  # Place holder that gets updated after first roll-out
                for _, forecast_step in enumerate(batch["forecast_step"]):
                    if forecast_step == 1:
                        # Initialize x and x_surf with the first time step
                        if "x_surf" in batch:
                            # combine x and x_surf
                            # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                            # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(
                                self.device
                            )  # .float()
                        else:
                            # no x_surf
                            x = reshape_only(batch["x"]).to(self.device)  # .float()

                    # add forcing and static variables (regardless of fcst hours)
                    if "x_forcing_static" in batch:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        x_forcing_batch = (
                            batch["x_forcing_static"]
                            .to(self.device)
                            .permute(0, 2, 1, 3, 4)
                        )  # .float()

                        # concat on var dimension
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    # --------------------------------------------- #
                    # clamp
                    if flag_clamp:
                        x = torch.clamp(x, min=clamp_min, max=clamp_max)
                        
                    y_pred = self.model(x)

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
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(
                                self.device
                            )
                        else:
                            y = reshape_only(batch["y"]).to(self.device)

                        if "y_diag" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            y_diag_batch = (
                                batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                            )  # .float()

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
                            value = torch.Tensor([value]).cuda(
                                self.device, non_blocking=True
                            )

                            if distributed:
                                dist.all_reduce(
                                    value, dist.ReduceOp.AVG, async_op=False
                                )

                            results_dict[f"valid_{name}"].append(value[0].item())

                        stop_forecast = True

                        break

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

                if not stop_forecast:
                    continue

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)

                if distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())

                stop_forecast = False

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

                if k // history_len >= valid_batches_per_epoch and k > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def fit_deprecated(
        self,
        conf,
        train_loader,
        valid_loader,
        optimizer,
        train_criterion,
        valid_criterion,
        scaler,
        scheduler,
        metrics,
        trial=False,
    ):
        save_loc = conf["save_loc"]
        start_epoch = conf["trainer"]["start_epoch"]
        epochs = conf["trainer"]["epochs"]
        skip_validation = (
            conf["trainer"]["skip_validation"]
            if "skip_validation" in conf["trainer"]
            else False
        )

        # Reload the results saved in the training csv if continuing to train
        if start_epoch == 0:
            results_dict = defaultdict(list)
            # Set start_epoch to the length of the training log and train for one epoch
            # This is a manual override, you must use train_one_epoch = True
            if (
                "train_one_epoch" in conf["trainer"]
                and conf["trainer"]["train_one_epoch"]
            ):
                epochs = 1
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(f"{save_loc}", "training_log.csv"))
            # Set start_epoch to the length of the training log and train for one epoch
            # This is a manual override, you must use train_one_epoch = True
            if (
                "train_one_epoch" in conf["trainer"]
                and conf["trainer"]["train_one_epoch"]
            ):
                start_epoch = len(saved_results)
                epochs = start_epoch + 1

            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        for epoch in range(start_epoch, epochs):
            logger.info(f"Starting epoch {epoch}")

            # set the epoch in the dataset and sampler to ensure distribured randomness is handled correctly
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)  # Start a new forecast

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(
                    epoch
                )  # Ensure we don't start in the middle of a forecast epoch-over-epoch

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch,
                conf,
                train_loader,
                optimizer,
                train_criterion,
                scaler,
                scheduler,
                metrics,
            )

            ############
            #
            # Validation
            #
            ############

            if skip_validation:
                valid_results = train_results

            else:
                valid_results = self.validate(
                    epoch, conf, valid_loader, valid_criterion, metrics
                )

            #################
            #
            # Save results
            #
            #################

            # update the learning rate if epoch-by-epoch updates

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_epoch
            ):
                if conf["trainer"]["scheduler"]["scheduler_type"] == "plateau":
                    scheduler.step(results_dict["valid_acc"][-1])
                else:
                    scheduler.step()

            # Put things into a results dictionary -> dataframe

            results_dict["epoch"].append(epoch)
            for name in ["loss", "acc", "mae"]:
                results_dict[f"train_{name}"].append(
                    np.mean(train_results[f"train_{name}"])
                )
                results_dict[f"valid_{name}"].append(
                    np.mean(valid_results[f"valid_{name}"])
                )
            results_dict["train_forecast_len"].append(
                np.mean(train_results["train_forecast_len"])
            )
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            df = pd.DataFrame.from_dict(results_dict).reset_index()

            # Save the dataframe to disk

            if trial:
                df.to_csv(
                    os.path.join(
                        f"{save_loc}",
                        "trial_results",
                        f"training_log_{trial.number}.csv",
                    ),
                    index=False,
                )
            else:
                df.to_csv(os.path.join(f"{save_loc}", "training_log.csv"), index=False)

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:
                if conf["trainer"]["mode"] != "fsdp":
                    if self.rank == 0:
                        # Save the current model

                        logger.info(
                            f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}"
                        )

                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                            if conf["trainer"]["use_scheduler"]
                            else None,
                            "scaler_state_dict": scaler.state_dict(),
                        }
                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                        if conf.get("trainer", {}).get("save_every_epoch", False):
                            copy_checkpoint(f"{save_loc}/checkpoint.pt", epoch)

                else:
                    logger.info(
                        f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}"
                    )

                    # Initialize the checkpoint I/O handler

                    checkpoint_io = TorchFSDPCheckpointIO()

                    # Save model and optimizer checkpoints

                    checkpoint_io.save_unsharded_model(
                        self.model,
                        os.path.join(save_loc, "model_checkpoint.pt"),
                        gather_dtensor=True,
                        use_safetensors=False,
                        rank=self.rank,
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pt"),
                        gather_dtensor=True,
                        rank=self.rank,
                    )

                    # Still need to save the scheduler and scaler states, just in another file for FSDP

                    state_dict = {
                        "epoch": epoch,
                        "scheduler_state_dict": scheduler.state_dict()
                        if conf["trainer"]["use_scheduler"]
                        else None,
                        "scaler_state_dict": scaler.state_dict(),
                    }

                    torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

                    if conf.get("trainer", {}).get("save_every_epoch", False):
                        copy_checkpoint(f"{save_loc}/checkpoint.pt", epoch)

                # This needs updated!
                # valid_loss = np.mean(valid_results["valid_loss"])
                # # save if this is the best model seen so far
                # if (self.rank == 0) and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
                #     if conf["trainer"]["mode"] == "ddp":
                #         shutil.copy(f"{save_loc}/checkpoint_{self.device}.pt", f"{save_loc}/best_{self.device}.pt")
                #     elif conf["trainer"]["mode"] == "fsdp":
                #         if os.path.exists(f"{save_loc}/best"):
                #             shutil.rmtree(f"{save_loc}/best")
                #         shutil.copytree(f"{save_loc}/checkpoint", f"{save_loc}/best")
                #     else:
                #         shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            gc.collect()

            training_metric = "train_loss" if skip_validation else "valid_loss"

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict[training_metric])
                if j == min(results_dict[training_metric])
            ][0]
            offset = epoch - best_epoch
            if offset >= conf["trainer"]["stopping_patience"]:
                logger.info(f"Trial {trial.number} is stopping early")
                break

            # Stop training if we get too close to the wall time
            if "stop_after_epoch" in conf["trainer"]:
                if conf["trainer"]["stop_after_epoch"]:
                    break

        training_metric = "train_loss" if skip_validation else "valid_loss"

        best_epoch = [
            i
            for i, j in enumerate(results_dict[training_metric])
            if j == min(results_dict[training_metric])
        ][0]

        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
