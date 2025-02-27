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
    """

    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
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
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        forecast_length = conf["data"]["forecast_len"]

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        # set up a custom tqdm
        if not isinstance(trainloader.dataset, IterableDataset):
            batches_per_epoch = (
                batches_per_epoch
                if 0 < batches_per_epoch < len(trainloader)
                else len(trainloader)
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
            y_pred = None  # Place holder that gets updated after first roll-out
            stop_forecast = False

            with autocast(enabled=amp):
                while not stop_forecast:
                    batch = next(dl)

                    for i, forecast_step in enumerate(batch["forecast_step"]):
                        if forecast_step == 1:
                            # Initialize x and x_surf with the first time step
                            if "x_surf" in batch:
                                # combine x and x_surf
                                # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                                # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                                x = (
                                    self.model.concat_and_reshape(
                                        batch["x"], batch["x_surf"]
                                    )
                                    .to(self.device)
                                    .float()
                                )
                            else:
                                # no x_surf
                                x = reshape_only(batch["x"]).to(self.device).float()

                        # add forcing and static variables (regardless of fcst hours)
                        if "x_forcing_static" in batch:
                            # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                            x_forcing_batch = (
                                batch["x_forcing_static"]
                                .to(self.device)
                                .permute(0, 2, 1, 3, 4)
                                .float()
                            )

                            # concat on var dimension
                            x = torch.cat((x, x_forcing_batch), dim=1)

                        # predict with the model
                        y_pred = self.model(x)

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
                                .float()
                            )

                            # concat on var dimension
                            y = torch.cat((y, y_diag_batch), dim=1)

                        # calculate rolling loss
                        batch_loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                        loss += batch_loss

                        # track the loss
                        accum_log(logs, {"loss": batch_loss.item()})

                        # stop after X steps
                        stop_forecast = batch["stop_forecast"][i]

                        # check if a single-step input
                        if x.shape[2] == 1:
                            x = y_pred.detach()
                        else:
                            # use multiple past forecast steps as inputs
                            # static channels will get updated on next pass
                            static_dim_size = abs(x.shape[1] - y_pred.shape[1])

                            # if static_dim_size=0 then :0 gives empty range
                            x_detach = (
                                x[:, :-static_dim_size, 1:].detach()
                                if static_dim_size
                                else x[:, :, 1:].detach()
                            )
                            x = torch.cat([x_detach, y_pred.detach()], dim=2)

                    if stop_forecast:
                        break

                # scale, accumulate, backward
                scaler.scale(loss).backward()

                if distributed:
                    torch.distributed.barrier()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Metrics

            metrics_dict = metrics(y_pred.float(), y.float())
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
                            x = (
                                self.model.concat_and_reshape(
                                    batch["x"], batch["x_surf"]
                                )
                                .to(self.device)
                                .float()
                            )
                        else:
                            # no x_surf
                            x = reshape_only(batch["x"]).to(self.device).float()

                    # add forcing and static variables (regardless of fcst hours)
                    if "x_forcing_static" in batch:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        x_forcing_batch = (
                            batch["x_forcing_static"]
                            .to(self.device)
                            .permute(0, 2, 1, 3, 4)
                            .float()
                        )

                        # concat on var dimension
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    y_pred = self.model(x)

                    # stop after user-defined number of steps
                    if forecast_step == (forecast_len + 1):
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
                                .float()
                            )

                            # concat on var dimension
                            y = torch.cat((y, y_diag_batch), dim=1)

                        # calculate rolling loss
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()

                        # Metrics
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
                    elif history_len == 1:
                        x = y_pred.detach()
                    else:
                        # use multiple past forecast steps as inputs
                        # static channels will get updated on next pass
                        static_dim_size = abs(x.shape[1] - y_pred.shape[1])

                        # if static_dim_size=0 then :0 gives empty range
                        x_detach = (
                            x[:, :-static_dim_size, 1:].detach()
                            if static_dim_size
                            else x[:, :, 1:].detach()
                        )
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
