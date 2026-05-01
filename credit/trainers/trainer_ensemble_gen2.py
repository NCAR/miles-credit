import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm

import optuna

from credit.losses.crps import ring_crps_loss
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer
from credit.preblock import build_preblocks, apply_preblocks
from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, apply_gradient_checkpointing, cycle

logger = logging.getLogger(__name__)


class TrainerEnsembleGen2(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict):
        """
        Gen 2 ensemble trainer for the ERA5 nested data schema.

        Uses preblock-assembled batches (same data schema as TrainerERA5Gen2) but
        replaces the standard criterion with ring_crps_loss for training, enabling
        fair CRPS computation across ensemble members distributed over GPUs under DDP.

        One ensemble member per GPU is the intended use case. The ring-reduce CRPS
        accumulates |x_r - x_j| for every pair (r, j) without materialising the full
        ensemble on a single device (O(1) extra memory).

        Key differences from TrainerERA5Gen2:
          - Training loss: ring_crps_loss instead of criterion (CRPS across DDP ranks).
          - Validation loss: standard criterion (single-member MSE/MAE, unchanged).
          - Adds "std" to the training log (ensemble spread proxy).

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)
        logger.info("Loading ERA5 Gen 2 ensemble trainer (ring-reduce CRPS, preblock-assembled batches)")

        self.preblocks = build_preblocks(conf.get("preblocks", {}))

        post_conf = conf.get("model", {}).get("post_conf", {})
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

        data_conf = conf["data"]
        source = next(iter(data_conf["source"].values()))
        vars_conf = source["variables"]
        prog = vars_conf.get("prognostic") or {}
        diag = vars_conf.get("diagnostic") or {}
        dyn = vars_conf.get("dynamic_forcing") or {}
        static_v = vars_conf.get("static") or {}
        num_levels = len(source.get("levels", []))
        self.varnum_diag = (len(diag.get("vars_3D", [])) * num_levels + len(diag.get("vars_2D", []))) if diag else 0

        self.static_dim_size = len(dyn.get("vars_2D", [])) + len(static_v.get("vars_2D", []))

        self.retain_graph = data_conf.get("retain_graph", False)

        self.forecast_len = data_conf["forecast_len"]
        trainer_conf = conf.get("trainer", {})
        bpt = trainer_conf.get("backprop_on_timestep") or data_conf.get("backprop_on_timestep")
        self.backprop_on_timestep = bpt if bpt is not None else list(range(1, self.forecast_len + 1))

        data_clamp = data_conf.get("data_clamp")
        if data_clamp is None:
            self.flag_clamp = False
            self.clamp_min = None
            self.clamp_max = None
        else:
            self.flag_clamp = True
            self.clamp_min = float(data_clamp[0])
            self.clamp_max = float(data_clamp[1])

        data_valid = conf.get("validation_data", data_conf)
        self.valid_history_len = data_valid.get("history_len", data_conf.get("history_len", 1))
        self.valid_forecast_len = data_valid.get("forecast_len", self.forecast_len)

        self.skip_nan_prune = conf.get("trainer", {}).get("skip_nan_prune", False)

        gc_conf = conf.get("trainer", {}).get("gradient_checkpointing", False)
        if gc_conf is not False and gc_conf is not None:
            block_names = gc_conf if isinstance(gc_conf, list) else None
            n_wrapped = apply_gradient_checkpointing(model, block_names)
            logger.info(
                f"Gradient checkpointing enabled: {n_wrapped} blocks wrapped"
                f" (patterns={block_names or ['Block', 'Layer', 'Encoder', 'Decoder', 'Attention']})"
            )

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch using ring-reduce CRPS as the training loss.

        Each GPU holds one ensemble member (DDP). The ring_crps_loss accumulates
        |y_pred_r - y_pred_j| for all j != r via K-1 P2P ring shifts without
        materialising the full ensemble on any single device.

        Args:
            epoch: Current epoch number.
            trainloader: DataLoader for training.
            optimizer, criterion, scaler, scheduler, metrics: Standard training objects.
                Note: criterion is NOT used here; ring_crps_loss replaces it.

        Returns:
            dict: Training metrics for the epoch.
        """
        if self.ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {self.ensemble_size}")
        logger.info(f"Using grad-max-norm value: {self.grad_max_norm}")

        if self.use_scheduler and self.scheduler_type == "lambda":
            scheduler.step()

        from torch.utils.data import IterableDataset

        batches_per_epoch = self.batches_per_epoch
        if not isinstance(trainloader.dataset, IterableDataset):
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(trainloader)
            batches_per_epoch = (
                self.batches_per_epoch if 0 < self.batches_per_epoch < dataset_batches else dataset_batches
            )

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            x_init = None
            y_pred = None
            y = None

            for t in range(1, self.forecast_len + 1):
                batch = next(dl)
                x_raw, y_raw, _ = apply_preblocks(self.preblocks, batch)
                if x_raw.dim() == 5:
                    x_raw = x_raw.flatten(1, 2)
                    y_raw = y_raw.flatten(1, 2)

                if t == 1:
                    x = x_raw.to(self.device).float()
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)
                else:
                    x_dynfrc = x_raw.to(self.device).float()
                    if self.ensemble_size > 1:
                        x_dynfrc = torch.repeat_interleave(x_dynfrc, self.ensemble_size, 0)
                    n_dynfrc = x_dynfrc.shape[1]
                    n_prog = x.shape[1] - self.static_dim_size
                    y_pred_prog = y_pred[:, :n_prog, ...]
                    if not self.retain_graph:
                        y_pred_prog = y_pred_prog.detach()
                    x_new = x.clone()
                    x_new[:, :n_dynfrc, ...] = x_dynfrc
                    x_new[:, self.static_dim_size :, ...] = y_pred_prog
                    x = x_new

                if self.flag_clamp:
                    x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                with torch.autocast(device_type="cuda", enabled=self.amp):
                    y_pred = self.model(x)
                if y_pred.dim() == 5:
                    y_pred = y_pred.flatten(1, 2)

                if self.flag_mass_conserve:
                    if t == 1:
                        x_init = x.clone()
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

                if t in self.backprop_on_timestep:
                    y = y_raw.to(self.device).float()
                    if self.flag_clamp:
                        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

                    loss = ring_crps_loss(y_pred.to(y.dtype), y, self.rank, dist.get_world_size())
                    total_std = (y_pred - y).detach().std()

                    accum_log(logs, {"loss": loss.item()})
                    accum_log(logs, {"std": total_std.item()})

                    scaler.scale(loss).backward(retain_graph=self.retain_graph)

                if self.distributed:
                    torch.distributed.barrier()

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

            if y_pred is not None and y is not None:
                metrics_dict = metrics(y_pred, y)
                for name, value in metrics_dict.items():
                    value = torch.Tensor([value]).to(self.device, non_blocking=True)
                    if self.distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

            batch_loss = torch.Tensor([logs.get("loss", 0.0)]).to(self.device)
            if self.distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(self.forecast_len)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(results_dict["train_loss"])
                if self.skip_nan_prune:
                    logger.warning("NaN/Inf loss detected but skip_nan_prune=True; continuing.")
                else:
                    raise optuna.TrialPruned()

            self._log_batch_progress(epoch, results_dict, optimizer, batch_group_generator, phase="train")

            if self.use_scheduler and self.scheduler_type in update_on_batch:
                scheduler.step()

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, valid_loader, criterion, metrics):
        """
        Validate for one epoch using the standard criterion.

        Runs self.valid_forecast_len autoregressive steps per sample.
        Loss and metrics are computed only at the final step.

        Args:
            epoch: Current epoch number.
            valid_loader: DataLoader for validation.
            criterion, metrics: Loss and metric callables.

        Returns:
            dict: Validation metrics for the epoch.
        """
        self.model.eval()

        from torch.utils.data import IterableDataset

        valid_batches_per_epoch = self.valid_batches_per_epoch
        if not isinstance(valid_loader.dataset, IterableDataset):
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(valid_loader)
            valid_batches_per_epoch = (
                self.valid_batches_per_epoch if 0 < self.valid_batches_per_epoch < dataset_batches else dataset_batches
            )

        results_dict = defaultdict(list)
        batch_group_generator = tqdm.tqdm(range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True)

        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                y_pred = None
                y = None
                loss = 0
                x_init = None

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)
                    x_raw, y_raw, _ = apply_preblocks(self.preblocks, batch)
                    if x_raw.dim() == 5:
                        x_raw = x_raw.flatten(1, 2)
                        y_raw = y_raw.flatten(1, 2)

                    if t == 1:
                        x = x_raw.to(self.device).float()
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)
                    else:
                        x_dynfrc = x_raw.to(self.device).float()
                        if self.ensemble_size > 1:
                            x_dynfrc = torch.repeat_interleave(x_dynfrc, self.ensemble_size, 0)
                        n_dynfrc = x_dynfrc.shape[1]
                        n_prog = x.shape[1] - self.static_dim_size
                        x_new = x.clone()
                        x_new[:, :n_dynfrc, ...] = x_dynfrc
                        x_new[:, self.static_dim_size :, ...] = y_pred[:, :n_prog, ...].detach()
                        x = x_new

                    if self.flag_clamp:
                        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    y_pred = self.model(x.float())
                    if y_pred.dim() == 5:
                        y_pred = y_pred.flatten(1, 2)

                    if self.flag_mass_conserve:
                        if t == 1:
                            x_init = x.clone()
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

                    if t == self.valid_forecast_len:
                        y = y_raw.to(self.device).float()
                        if self.flag_clamp:
                            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)

                        loss = criterion(y.to(y_pred.dtype), y_pred).mean()
                        metrics_dict = metrics(y_pred.float(), y.float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).to(self.device, non_blocking=True)
                            if self.distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.Tensor([loss.item() if torch.is_tensor(loss) else loss]).to(self.device)
                if self.distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(self.valid_forecast_len)

                self._log_batch_progress(epoch, results_dict, optimizer=None, pbar=batch_group_generator, phase="valid")

        batch_group_generator.close()

        if self.distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict


Trainer = TrainerEnsembleGen2
