""" """

import os
import gc
import tqdm
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

import optuna
from credit.data import concat_and_reshape, reshape_only
from credit.models.checkpoint import TorchFSDPCheckpointIO
from credit.scheduler import update_on_batch, update_on_epoch
from credit.trainers.utils import cleanup, accum_log, cycle
from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
        logger.info("LES single-step training")

    # Training function.
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        # training hyperparameters
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        grad_max_norm = conf["trainer"]["grad_max_norm"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        # forecast step
        if "total_time_steps" in conf["data"]:
            total_time_steps = conf["data"]["total_time_steps"]
        else:
            total_time_steps = forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        # ====================================================== #

        # # set up a custom tqdm
        # if not isinstance(trainloader.dataset, IterableDataset):
        #     # if batches_per_epoch = 0, use all training samples (i.e., full epoch)
        #     #batches_per_epoch = (batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader))
        #     pass

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        results_dict = defaultdict(list)

        # dataloader
        dl = cycle(trainloader)

        for i in batch_group_generator:
            # Get the next batch from the iterator
            batch = next(dl)

            # training log
            logs = {}

            with autocast(enabled=amp):
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
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # combine y and y_surf
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                else:
                    y = reshape_only(batch["y"]).to(self.device)

                # if "y_diag" in batch:
                #     # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                #     y_diag_batch = (batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4))

                #     # concat on var dimension
                #     y = torch.cat((y, y_diag_batch), dim=1)

                if "y_diag" in batch:
                    if len(batch["y_diag"].shape) == 6:
                        y_diag_batch = reshape_only(batch["y_diag"]).to(self.device)

                    else:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4).float()

                    # concat on var dimension
                    y = torch.cat((y, y_diag_batch), dim=1)

                # single step predict
                y_pred = self.model(x)
                y = y.to(device=self.device, dtype=y_pred.dtype)

                # loss compute
                loss = criterion(y, y_pred)

                # Metrics
                # metrics_dict = metrics(y_pred.float(), y.float())
                metrics_dict = metrics(y_pred, y)

                # save training metrics
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                # backpropagation
                loss = loss.mean()

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {"loss": loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            if grad_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)

            scaler.step(optimizer)
            scaler.update()

            # clear grad
            optimizer.zero_grad()

            # Handle batch_loss
            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)

            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)

            results_dict["train_loss"].append(batch_loss[0].item())

            if "forecast_hour" in batch:
                forecast_hour_tensor = batch["forecast_hour"].to(self.device)
                if distributed:
                    dist.all_reduce(forecast_hour_tensor, dist.ReduceOp.AVG, async_op=False)
                    forecast_hour_avg = forecast_hour_tensor[-1].item()
                else:
                    forecast_hour_avg = batch["forecast_hour"][-1].item()

                results_dict["train_forecast_len"].append(forecast_hour_avg + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    print("Invalid loss value: {}".format(np.mean(results_dict["train_loss"])))
                    raise optuna.TrialPruned()

                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                np.mean(results_dict["train_forecast_len"]),
            )

            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch:
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]

        forecast_len = conf["data"]["valid_forecast_len"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        total_time_steps = conf["data"]["total_time_steps"] if "total_time_steps" in conf["data"] else forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        results_dict = defaultdict(list)

        # ====================================================== #

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        dl = cycle(valid_loader)

        for i in batch_group_generator:
            batch = next(dl)

            with torch.no_grad():
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
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # combine y and y_surf
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                else:
                    y = reshape_only(batch["y"]).to(self.device)

                if "y_diag" in batch:
                    if len(batch["y_diag"].shape) == 6:
                        y_diag_batch = reshape_only(batch["y_diag"]).to(self.device)

                    else:
                        # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                        y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4).float()

                    # concat on var dimension
                    y = torch.cat((y, y_diag_batch), dim=1)

                y_pred = self.model(x)

                loss = criterion(y.to(y_pred.dtype), y_pred)

                # Metrics
                # metrics_dict = metrics(y_pred, y)
                metrics_dict = metrics(y_pred.float(), y.float())

                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)

                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)

                    results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)

                if distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )

                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

                if i >= valid_batches_per_epoch and i > 0:
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
