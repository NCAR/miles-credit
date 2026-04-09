import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import IterableDataset

import optuna

from credit.data import concat_and_reshape, reshape_only
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, cycle

logger = logging.getLogger(__name__)


class TrainerERA5(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict):
        """
        Trainer for multi-step ERA5 rollout training.

        Extracts all trainer, model, and static data/rollout settings from conf
        at construction time. self.conf["data"] keys governed by the evolving data
        schema remain here but are stored as self attributes so methods stay clean.

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)
        logger.info("Loading a multi-step trainer class")

        # ---- Postblock conservation fixers ----
        post_conf = self.conf.get("model", {}).get("post_conf", {})
        self.flag_mass_conserve = False
        self.flag_water_conserve = False
        self.flag_energy_conserve = False
        self.opt_mass = None
        self.opt_water = None
        self.opt_energy = None

        if post_conf.get("activate", False):
            if post_conf.get("global_mass_fixer", {}).get("activate", False) and post_conf["global_mass_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalMassFixer outside of model")
                self.flag_mass_conserve = True
                self.opt_mass = GlobalMassFixer(post_conf)

            if post_conf.get("global_water_fixer", {}).get("activate", False) and post_conf["global_water_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalWaterFixer outside of model")
                self.flag_water_conserve = True
                self.opt_water = GlobalWaterFixer(post_conf)

            if post_conf.get("global_energy_fixer", {}).get("activate", False) and post_conf["global_energy_fixer"].get(
                "activate_outside_model", False
            ):
                logger.info("Activate GlobalEnergyFixer outside of model")
                self.flag_energy_conserve = True
                self.opt_energy = GlobalEnergyFixer(post_conf)

        # ---- Static data/rollout settings ----
        data_conf = self.conf["data"]
        self.varnum_diag = len(data_conf.get("diagnostic_variables", []))
        self.static_dim_size = (
            len(data_conf.get("dynamic_forcing_variables", []))
            + len(data_conf.get("forcing_variables", []))
            + len(data_conf.get("static_variables", []))
        )
        self.retain_graph = data_conf.get("retain_graph", False)
        self.forecast_len = data_conf["forecast_len"]
        if "backprop_on_timestep" in data_conf:
            self.backprop_on_timestep = data_conf["backprop_on_timestep"]
        else:
            self.backprop_on_timestep = list(range(0, self.forecast_len + 2))
        assert self.forecast_len <= self.backprop_on_timestep[-1], (
            f"forecast_len ({self.forecast_len + 1}) must not exceed the max value "
            f"in backprop_on_timestep {self.backprop_on_timestep}"
        )
        data_clamp = data_conf.get("data_clamp")
        if data_clamp is None:
            self.flag_clamp = False
            self.clamp_min = None
            self.clamp_max = None
        else:
            self.flag_clamp = True
            self.clamp_min = float(data_clamp[0])
            self.clamp_max = float(data_clamp[1])
        self.valid_history_len = (
            data_conf["valid_history_len"] if "valid_history_len" in data_conf else data_conf["history_len"]
        )
        self.valid_forecast_len = (
            data_conf["valid_forecast_len"] if "valid_forecast_len" in data_conf else data_conf["forecast_len"]
        )

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.
            conf: Full configuration dict (data keys accessed here for schema stability).
            trainloader: DataLoader for training.
            optimizer, criterion, scaler, scheduler, metrics: Standard training objects.

        Returns:
            dict: Training metrics for the epoch.
        """
        if self.ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {self.ensemble_size}")
        logger.info(f"Using grad-max-norm value: {self.grad_max_norm}")

        # lambda scheduler steps once per epoch before batches
        if self.use_scheduler and self.scheduler_type == "lambda":
            scheduler.step()

        # resolve effective batches_per_epoch from dataset if not fixed
        batches_per_epoch = self.batches_per_epoch
        if not isinstance(trainloader.dataset, IterableDataset):
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(trainloader)
            batches_per_epoch = (
                self.batches_per_epoch
                if 0 < self.batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            stop_forecast = False
            y_pred = None
            while not stop_forecast:
                batch = next(dl)
                forecast_step = batch["forecast_step"].item()

                if forecast_step == 1:
                    # input: (batch, time, var, level, lat, lon) + (batch, time, var, lat, lon)
                    # output: (batch, var, time, lat, lon)
                    if "x_surf" in batch:
                        x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                    else:
                        x = reshape_only(batch["x"]).to(self.device)
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)

                # add forcing and static variables (every step)
                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                    if self.ensemble_size > 1:
                        x_forcing_batch = torch.repeat_interleave(x_forcing_batch, self.ensemble_size, 0)
                    x = torch.cat((x, x_forcing_batch), dim=1)

                if self.flag_clamp:
                    x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                x = x.float()
                with torch.autocast(device_type="cuda", enabled=self.amp):
                    y_pred = self.model(x)

                # postblock opts outside of model
                if self.flag_mass_conserve:
                    if forecast_step == 1:
                        x_init = x.clone()
                if self.flag_mass_conserve:
                    input_dict = {"y_pred": y_pred, "x": x_init}
                    input_dict = self.opt_mass(input_dict)
                    y_pred = input_dict["y_pred"]

                if self.flag_water_conserve:
                    input_dict = {"y_pred": y_pred, "x": x}
                    input_dict = self.opt_water(input_dict)
                    y_pred = input_dict["y_pred"]

                if self.flag_energy_conserve:
                    input_dict = {"y_pred": y_pred, "x": x}
                    input_dict = self.opt_energy(input_dict)
                    y_pred = input_dict["y_pred"]

                # backprop only on specified timesteps
                if forecast_step in self.backprop_on_timestep:
                    if "y_surf" in batch:
                        y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                    else:
                        y = reshape_only(batch["y"]).to(self.device)
                    if "y_diag" in batch:
                        y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                        y = torch.cat((y, y_diag_batch), dim=1)
                    if self.flag_clamp:
                        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

                    with torch.autocast(enabled=self.amp, device_type="cuda"):
                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss).backward(retain_graph=self.retain_graph)

                if self.distributed:
                    torch.distributed.barrier()

                stop_forecast = batch["stop_forecast"].item()
                if stop_forecast:
                    break

                if not self.retain_graph:
                    y_pred = y_pred.detach()

                # roll x forward for next step
                if x.shape[2] == 1:
                    # single-timestep: step-in-step-out
                    x = y_pred[:, : -self.varnum_diag, ...] if "y_diag" in batch else y_pred
                else:
                    # multi-timestep: slide window, static channels re-added next pass
                    if self.static_dim_size == 0:
                        x_detach = x[:, :, 1:, ...].detach()
                    else:
                        x_detach = x[:, : -self.static_dim_size, 1:, ...].detach()
                    if "y_diag" in batch:
                        x = torch.cat([x_detach, y_pred[:, : -self.varnum_diag, ...]], dim=2)
                    else:
                        x = torch.cat([x_detach, y_pred], dim=2)

            if self.distributed:
                torch.distributed.barrier()

            # grad norm clipping
            scaler.unscale_(optimizer)
            if self.grad_max_norm == "dynamic":
                local_norm = torch.norm(
                    torch.stack([p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None])
                )
                if self.distributed:
                    dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
                global_norm = local_norm.sqrt()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
            elif self.grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if self.ema is not None:
                self.ema.update(self.model)

            # collect metrics
            metrics_dict = metrics(y_pred, y)
            for name, value in metrics_dict.items():
                value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                if self.distributed:
                    dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if self.distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(self.forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(results_dict["train_loss"], batch["x"].shape, batch["y"].shape, batch["index"])
                raise optuna.TrialPruned()

            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len: {:.6f}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                self.forecast_len + 1,
            )
            if self.ensemble_size > 1:
                to_print += f" std: {np.mean(results_dict['train_std']):.6f}"
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.update(1)
                batch_group_generator.set_description(to_print)

            if self.use_scheduler and self.scheduler_type in update_on_batch:
                scheduler.step()

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, valid_loader, criterion, metrics):
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number.
            conf: Full configuration dict.
            valid_loader: DataLoader for validation.
            criterion, metrics: Loss and metric callables.

        Returns:
            dict: Validation metrics for the epoch.
        """
        self.model.eval()

        # resolve effective valid_batches_per_epoch from dataset if not fixed
        valid_batches_per_epoch = self.valid_batches_per_epoch
        if not isinstance(valid_loader.dataset, IterableDataset):
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches_per_epoch = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches_per_epoch = len(valid_loader)
            valid_batches_per_epoch = (
                self.valid_batches_per_epoch
                if 0 < self.valid_batches_per_epoch < dataset_batches_per_epoch
                else dataset_batches_per_epoch
            )

        results_dict = defaultdict(list)
        batch_group_generator = tqdm.tqdm(range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True)

        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                loss = 0
                stop_forecast = False
                y_pred = None
                while not stop_forecast:
                    batch = next(dl)
                    forecast_step = batch["forecast_step"].item()
                    stop_forecast = batch["stop_forecast"].item()

                    if forecast_step == 1:
                        if "x_surf" in batch:
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                        else:
                            x = reshape_only(batch["x"]).to(self.device)
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)

                    # add forcing and static variables (every step)
                    if "x_forcing_static" in batch:
                        x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                        if self.ensemble_size > 1:
                            x_forcing_batch = torch.repeat_interleave(x_forcing_batch, self.ensemble_size, 0)
                        x = torch.cat((x, x_forcing_batch), dim=1)

                    if self.flag_clamp:
                        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    y_pred = self.model(x.float())

                    # postblock opts outside of model
                    if self.flag_mass_conserve:
                        if forecast_step == 1:
                            x_init = x.clone()
                    if self.flag_mass_conserve:
                        input_dict = {"y_pred": y_pred, "x": x_init}
                        input_dict = self.opt_mass(input_dict)
                        y_pred = input_dict["y_pred"]

                    if self.flag_water_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_water(input_dict)
                        y_pred = input_dict["y_pred"]

                    if self.flag_energy_conserve:
                        input_dict = {"y_pred": y_pred, "x": x}
                        input_dict = self.opt_energy(input_dict)
                        y_pred = input_dict["y_pred"]

                    # compute loss and metrics only at the final rollout step
                    if forecast_step == (self.valid_forecast_len + 1):
                        if "y_surf" in batch:
                            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                        else:
                            y = reshape_only(batch["y"]).to(self.device)
                        if "y_diag" in batch:
                            y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                            y = torch.cat((y, y_diag_batch), dim=1)
                        if self.flag_clamp:
                            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                        metrics_dict = metrics(y_pred.float(), y.float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                            if self.distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                        assert stop_forecast
                        break

                    # roll x forward: step-in-step-out
                    elif self.valid_history_len == 1:
                        x = y_pred[:, : -self.varnum_diag, ...].detach() if "y_diag" in batch else y_pred.detach()

                    # roll x forward: multi-timestep sliding window
                    else:
                        if self.static_dim_size == 0:
                            x_detach = x[:, :, 1:, ...].detach()
                        else:
                            x_detach = x[:, : -self.static_dim_size, 1:, ...].detach()
                        if "y_diag" in batch:
                            x = torch.cat([x_detach, y_pred[:, : -self.varnum_diag, ...].detach()], dim=2)
                        else:
                            x = torch.cat([x_detach, y_pred.detach()], dim=2)

                batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                if self.distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(self.valid_forecast_len + 1)

                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )
                if self.ensemble_size > 1:
                    to_print += f" std: {np.mean(results_dict['valid_std']):.6f}"
                if self.rank == 0:
                    batch_group_generator.update(1)
                    batch_group_generator.set_description(to_print)

        batch_group_generator.close()

        if self.distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict


Trainer = TrainerERA5  # backward-compatible alias
