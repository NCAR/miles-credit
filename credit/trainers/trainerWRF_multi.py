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
import torch

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

    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)

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
            + len(conf['data']['boundary']['variables'])
            + len(conf['data']['boundary']['surface_variables'])
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
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

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
                batches_per_epoch
                if 0 < batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)
        for steps in range(batches_per_epoch):
            logs = {}; loss = 0
            stop_forecast = False
            y_pred = None  # Place holder that gets updated after first roll-out
            
            while not stop_forecast:
                
                batch = next(dl)
                forecast_step = batch["forecast_step"].item()
                
                if forecast_step == 1:
                    # --------------------------------------------------------------------------------- #
                    if "x_surf" in batch:
                        
                        # combine x and x_surf
                        # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                        # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                        x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                    else:
                        # no x_surf
                        x = reshape_only(batch["x"]).to(self.device)

                # --------------------------------------------------------------------------------- #
                # add forcing and static variables
                if "x_forcing_static" in batch:
                    
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    x_forcing_batch = (batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4))
                    
                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # boundary conditions
                if 'x_surf_boundary' in batch:
                    x_boundary = concat_and_reshape(batch['x_boundary'], batch['x_surf_boundary']).to(self.device)
                else:
                    x_boundary = reshape_only(batch['x_boundary']).to(self.device)

                # --------------------------------------------------------------------------------- #
                # time encoding
                x_time_encode = batch['x_time_encode'].to(self.device)
                
                # predict with the model
                with autocast(enabled=amp):
                    y_pred = self.model(x, x_boundary, x_time_encode)
                    
                # only load y-truth data if we intend to backprop (default is every step gets grads computed
                if forecast_step in backprop_on_timestep:
                    
                    # calculate rolling loss
                    # --------------------------------------------------------------------------------- #
                    # combine y and y_surf
                    if "y_surf" in batch:
                        y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                    else:
                        y = reshape_only(batch["y"]).to(self.device)
    
                    if "y_diag" in batch:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        y_diag_batch = (batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4))
    
                        # concat on var dimension
                        y = torch.cat((y, y_diag_batch), dim=1)
                        

                    with autocast(enabled=amp):
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                    # track the loss
                    accum_log(logs, {"loss": loss.item()})

                    # compute gradients
                    scaler.scale(loss).backward()

                if distributed:
                    torch.distributed.barrier()

                # stop after X steps
                stop_forecast = batch["stop_forecast"].item()
                if stop_forecast:
                    break

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
                        x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
                    else:
                        x = torch.cat([x_detach, y_pred.detach()], dim=2)

            if distributed:
                torch.distributed.barrier()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Metrics
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
            + len(conf['data']['boundary']['variables'])
            + len(conf['data']['boundary']['surface_variables'])
        )

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        history_len = conf["data"]["valid_history_len"]
        forecast_len = conf["data"]["valid_forecast_len"]


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
                        # --------------------------------------------------------------------------------- #
                        if "x_surf" in batch:
                            
                            # combine x and x_surf
                            # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                            # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                        else:
                            # no x_surf
                            x = reshape_only(batch["x"]).to(self.device)
    
                    # --------------------------------------------------------------------------------- #
                    # add forcing and static variables
                    if "x_forcing_static" in batch:
                        
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        x_forcing_batch = (batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4))
                        
                        # concat on var dimension
                        x = torch.cat((x, x_forcing_batch), dim=1)
    
                    # --------------------------------------------------------------------------------- #
                    # boundary conditions
                    if 'x_surf_boundary' in batch:
                        x_boundary = concat_and_reshape(batch['x_boundary'], batch['x_surf_boundary']).to(self.device)
                    else:
                        x_boundary = reshape_only(batch['x_boundary']).to(self.device)
    
                    # --------------------------------------------------------------------------------- #
                    # time encoding
                    x_time_encode = batch['x_time_encode'].to(self.device)

                    y_pred = self.model(x, x_boundary, x_time_encode)

                    # ================================================================================== #
                    # scope of reaching the final forecast_len
                    if forecast_step == (forecast_len + 1):
                        # calculate rolling loss
                        # --------------------------------------------------------------------------------- #
                        # combine y and y_surf
                        if "y_surf" in batch:
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                        else:
                            y = reshape_only(batch["y"]).to(self.device)
        
                        if "y_diag" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            y_diag_batch = (batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4))
        
                            # concat on var dimension
                            y = torch.cat((y, y_diag_batch), dim=1)

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
                            x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
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
