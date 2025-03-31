import gc
import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import xarray as xr
import torch
import torch.distributed as dist
import torch.fft
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

from credit.scheduler import update_on_batch

# from credit.solar import TOADataLoader
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log

logger = logging.getLogger(__name__)


class TOADataLoader:
    def __init__(self, conf):
        self.TOA = xr.open_dataset(conf["data"]["TOA_forcing_path"]).load()
        self.times_b = pd.to_datetime(self.TOA.time.values)

        # Precompute day of year and hour arrays
        self.days_of_year = self.times_b.dayofyear
        self.hours_of_day = self.times_b.hour

    def __call__(self, datetime_input):
        doy = datetime_input.dayofyear
        hod = datetime_input.hour

        # Use vectorized comparison for masking
        mask_toa = (self.days_of_year == doy) & (self.hours_of_day == hod)
        selected_tsi = self.TOA["tsi"].sel(time=mask_toa) / 2540585.74

        # Convert to tensor and add dimension
        return torch.tensor(selected_tsi.to_numpy()).unsqueeze(0).float()


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
        # Add any additional initialization if needed
        logger.info("Loading a batch trainer class")

    # Training function.
    def train_one_epoch(
        self,
        epoch: int,
        conf: Dict[str, Any],
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        # training hyperparameters
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        history_len = conf["data"]["history_len"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        rollout_p = (
            1.0
            if "stop_rollout" not in conf["trainer"]
            else conf["trainer"]["stop_rollout"]
        )

        total_time_steps = (
            conf["data"]["total_time_steps"]
            if "total_time_steps" in conf["data"]
            else forecast_len
        )

        if (
            "static_variables" in conf["data"]
            and "tsi" in conf["data"]["static_variables"]
        ):
            self.toa = TOADataLoader(conf)

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if (
            conf["trainer"]["use_scheduler"]
            and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"
        ):
            scheduler.step()

        # set up a custom tqdm
        batches_per_epoch = (
            batches_per_epoch
            if 0 < batches_per_epoch < len(trainloader)
            else len(trainloader)
        )

        batch_group_generator = tqdm.tqdm(
            enumerate(trainloader),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        static = None
        results_dict = defaultdict(list)

        for i, batch in batch_group_generator:
            logs = {}

            commit_loss = 0.0

            with autocast(enabled=amp):
                x = self.model.concat_and_reshape(batch["x"], batch["x_surf"]).to(
                    self.device
                )

                if "static" in batch:
                    if static is None:
                        static = (
                            batch["static"]
                            .to(self.device)
                            .unsqueeze(2)
                            .expand(-1, -1, x.shape[2], -1, -1)
                            .float()
                        )  # [batch, num_stat_vars, hist_len, lat, lon]
                    x = torch.cat((x, static.clone()), dim=1)

                if "TOA" in batch:
                    toa = batch["TOA"].to(self.device)
                    x = torch.cat([x, toa.unsqueeze(1)], dim=1)

                y = self.model.concat_and_reshape(
                    batch["y"], batch["y_surf"]
                )  # !! <------- .to(self.device)

                k = 0
                while True:
                    with torch.no_grad() if k != total_time_steps else torch.enable_grad():
                        self.model.eval() if k != total_time_steps else self.model.train()

                        if getattr(self.model, "use_codebook", False):
                            y_pred, cm_loss = self.model(x)
                            commit_loss += cm_loss
                        else:
                            y_pred = self.model(x)

                        if k == total_time_steps:
                            break

                        k += 1

                        if history_len > 1:
                            x_detach = x.detach()[:, :, 1:]
                            if "static" in batch:
                                y_pred = torch.cat(
                                    (y_pred, static[:, :, 0:1].clone()), dim=1
                                )

                            if "TOA" in batch:  # update the TOA based on doy and hod
                                elapsed_time = pd.Timedelta(hours=k)
                                current_times = [
                                    pd.to_datetime(_t, unit="ns") + elapsed_time
                                    for _t in batch["datetime"]
                                ]
                                toa = torch.cat(
                                    [self.toa(_t).unsqueeze(0) for _t in current_times],
                                    dim=0,
                                ).to(self.device)
                                y_pred = torch.cat([y_pred, toa], dim=1)

                            x = torch.cat([x_detach, y_pred], dim=2).detach()
                        else:
                            if "static" in batch or "TOA" in batch:
                                x = y_pred.detach()

                                if "static" in batch:
                                    x = torch.cat((x, static[:, :, 0:1].clone()), dim=1)

                                if (
                                    "TOA" in batch
                                ):  # update the TOA based on doy and hod
                                    elapsed_time = pd.Timedelta(hours=k)
                                    current_times = [
                                        pd.to_datetime(_t, unit="ns") + elapsed_time
                                        for _t in batch["datetime"]
                                    ]
                                    toa = torch.cat(
                                        [
                                            self.toa(_t).unsqueeze(0)
                                            for _t in current_times
                                        ],
                                        dim=0,
                                    ).to(self.device)
                                    x = torch.cat([x, toa], dim=1)
                            else:
                                x = y_pred.detach()

                y = y.to(device=self.device, dtype=y_pred.dtype)
                loss = criterion(y, y_pred)

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                loss = loss.mean() + commit_loss

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {"loss": loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Handle batch_loss
            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())

            if "forecast_hour" in batch:
                forecast_hour_tensor = batch["forecast_hour"].to(self.device)
                if distributed:
                    dist.all_reduce(
                        forecast_hour_tensor, dist.ReduceOp.AVG, async_op=False
                    )
                    forecast_hour_avg = forecast_hour_tensor[-1].item()
                else:
                    forecast_hour_avg = batch["forecast_hour"][-1].item()

                results_dict["train_forecast_len"].append(forecast_hour_avg + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
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

            if (
                conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch
            ):
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(
        self,
        epoch: int,
        conf: Dict[str, Any],
        valid_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
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
        total_time_steps = (
            conf["data"]["total_time_steps"]
            if "total_time_steps" in conf["data"]
            else forecast_len
        )

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
            enumerate(valid_loader),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        static = None

        for i, batch in batch_group_generator:
            with torch.no_grad():
                commit_loss = 0.0

                x = self.model.concat_and_reshape(batch["x"], batch["x_surf"]).to(
                    self.device
                )

                if "static" in batch:
                    if static is None:
                        static = (
                            batch["static"]
                            .to(self.device)
                            .unsqueeze(2)
                            .expand(-1, -1, x.shape[2], -1, -1)
                            .float()
                        )
                    x = torch.cat((x, static.clone()), dim=1)

                if "TOA" in batch:
                    toa = batch["TOA"].to(self.device)
                    x = torch.cat([x, toa.unsqueeze(1)], dim=1)

                y = self.model.concat_and_reshape(batch["y"], batch["y_surf"]).to(
                    self.device
                )

                k = 0
                while True:
                    if getattr(self.model, "use_codebook", False):
                        y_pred, cm_loss = self.model(x)
                        commit_loss += cm_loss
                    else:
                        y_pred = self.model(x)

                    if k == total_time_steps:
                        break

                    k += 1

                    if history_len > 1:
                        x_detach = x.detach()[:, :, 1:]
                        if "static" in batch:
                            y_pred = torch.cat(
                                (y_pred, static[:, :, 0:1].clone()), dim=1
                            )

                        if "TOA" in batch:  # update the TOA based on doy and hod
                            elapsed_time = pd.Timedelta(hours=k)
                            current_times = [
                                pd.to_datetime(_t, unit="ns") + elapsed_time
                                for _t in batch["datetime"]
                            ]
                            toa = torch.cat(
                                [self.toa(_t).unsqueeze(0) for _t in current_times],
                                dim=0,
                            ).to(self.device)
                            y_pred = torch.cat([y_pred, toa], dim=1)

                        x = torch.cat([x_detach, y_pred], dim=2).detach()

                    else:
                        if "static" in batch or "TOA" in batch:
                            x = y_pred.detach()

                            if "static" in batch:
                                x = torch.cat((x, static[:, :, 0:1].clone()), dim=1)

                            if "TOA" in batch:  # update the TOA based on doy and hod
                                elapsed_time = pd.Timedelta(hours=k)
                                current_times = [
                                    pd.to_datetime(_t, unit="ns") + elapsed_time
                                    for _t in batch["datetime"]
                                ]
                                toa = torch.cat(
                                    [self.toa(_t).unsqueeze(0) for _t in current_times],
                                    dim=0,
                                ).to(self.device)
                                x = torch.cat([x, toa], dim=1)
                        else:
                            x = y_pred.detach()

                loss = criterion(y.to(y_pred.dtype), y_pred)

                # Metrics
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
