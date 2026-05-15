import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm

import optuna

from credit.postblock import build_postblocks, apply_postblocks
from credit.preblock import build_preblocks, apply_preblocks
from credit.trainers.rollout_utils import build_rollout_input
from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, cycle

logger = logging.getLogger(__name__)


class TrainerERA5Gen2(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict):
        """
        Gen 2 trainer for the ERA5 nested data schema.

        Key differences from TrainerERA5Gen1:
          - Uses new nested data schema: conf["data"]["source"]["ERA5"]["variables"]
          - Applies preblocks to assemble batch tensors before the model forward pass
          - forecast_len semantics: 1 = 1 step (Gen 1 used 0 = 1 step)
          - backprop_on_timestep: range(1, forecast_len+1) instead of range(0, forecast_len+2)
          - Validation config read from conf["validation_data"] if present, else conf["data"]
          - Postblocks applied after model forward pass via apply_postblocks()
          - Multi-step rollout uses build_rollout_input() driven by the t=1 channel map

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)
        logger.info("Loading ERA5 Gen 2 trainer (new nested data schema, preblock-assembled batches)")

        self.preblocks = build_preblocks(conf.get("preblocks", {}))
        self.postblocks = build_postblocks(conf.get("postblocks", {}))

        # Cached from the first t=1 batch; covers ALL input variable groups.
        # Used by build_rollout_input to assemble x at t > 1.
        self.ic_channel_map = None

        # ---- Data schema extraction (new nested schema) ----
        data_conf = conf["data"]
        source = next(iter(data_conf["source"].values()))
        vars_conf = source["variables"]
        diag = vars_conf.get("diagnostic") or {}
        num_levels = len(source.get("levels", []))
        self.varnum_diag = (len(diag.get("vars_3D", [])) * num_levels + len(diag.get("vars_2D", []))) if diag else 0

        self.retain_graph = data_conf.get("retain_graph", False)

        # forecast_len: 1 = 1 step (new semantics, unlike v1 where 0 = 1 step)
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

        # Validation config: use validation_data block if present, else fall back to data
        data_valid = conf.get("validation_data", data_conf)
        self.valid_history_len = data_valid.get("history_len", data_conf.get("history_len", 1))
        self.valid_forecast_len = data_valid.get("forecast_len", self.forecast_len)

        # If True, log a warning on NaN loss instead of raising TrialPruned.
        self.skip_nan_prune = conf.get("trainer", {}).get("skip_nan_prune", False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_full_data_dict(self, batch: dict, _batch: dict, x: torch.Tensor) -> dict:
        """Build the full_data_dict at t=1 from the raw batch and preblock output."""
        return {
            "raw": batch,
            "preprocessed": {
                "input": x,
                "target": _batch.get("target"),
                "metadata": _batch["metadata"],
            },
            "predicted": None,
        }

    def _update_full_data_dict(self, full_data_dict: dict, _batch: dict, x: torch.Tensor) -> None:
        """Update preprocessed input/target in full_data_dict at t > 1 (in-place)."""
        full_data_dict["preprocessed"]["input"] = x
        full_data_dict["preprocessed"]["target"] = _batch.get("target")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch.

        The inner loop iterates over forecast_len autoregressive steps. For each step:
          1. Pull the next batch from the dataloader (contains per-field tensors).
          2. Apply preblocks to assemble batch["preprocessed"]["input"] and ["target"].
          3. At t=1: cache self.ic_channel_map and initialize full_data_dict.
             At t>1: assemble x via build_rollout_input from the previous prediction.
          4. Forward pass; compute loss on the flat (scaled) y_pred.
          5. Apply postblocks (Reconstruct converts y_pred to nested variable dict).

        Args:
            epoch: Current epoch number.
            trainloader: DataLoader for training.
            optimizer, criterion, scaler, scheduler, metrics: Standard training objects.

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
            y_pred_flat = None  # flat model output; used for loss, metrics, and rollout
            y = None
            full_data_dict = None

            for t in range(1, self.forecast_len + 1):
                batch = next(dl)
                _batch = apply_preblocks(self.preblocks, batch, device=self.device)

                if t == 1:
                    self.ic_channel_map = _batch["metadata"]["input"]["_channel_map"]
                    x = _batch["input"].float()
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)
                    full_data_dict = self._init_full_data_dict(batch, _batch, x)
                else:
                    curr_dyn_input = _batch["input"].float()
                    if self.ensemble_size > 1:
                        curr_dyn_input = torch.repeat_interleave(curr_dyn_input, self.ensemble_size, 0)
                    x = build_rollout_input(full_data_dict, curr_dyn_input, self.ic_channel_map)
                    self._update_full_data_dict(full_data_dict, _batch, x)

                if self.flag_clamp:
                    x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
                    full_data_dict["preprocessed"]["input"] = x

                with torch.autocast(device_type="cuda", enabled=self.amp):
                    y_pred_flat = self.model(x)

                # Loss on flat scaled prediction — computed before postblocks so
                # the target and prediction are in the same (scaled) space.
                if t in self.backprop_on_timestep:
                    y = _batch["target"].float()
                    if self.flag_clamp:
                        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        loss = criterion(y.to(y_pred_flat.dtype), y_pred_flat).mean()
                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss).backward(retain_graph=self.retain_graph)

                # Postblocks — Reconstruct (if configured) splits y_pred_flat into
                # a nested variable dict for the next rollout step.
                full_data_dict["predicted"] = y_pred_flat if self.retain_graph else y_pred_flat.detach()
                if self.postblocks:
                    full_data_dict = apply_postblocks(self.postblocks, full_data_dict)

                if self.distributed:
                    torch.distributed.barrier()

            # optimizer step
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

            if y_pred_flat is not None and y is not None:
                metrics_dict = metrics(y_pred_flat, y)
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

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    def validate(self, epoch, valid_loader, criterion, metrics):
        """
        Validate for one epoch.

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
                y_pred_flat = None
                y = None
                loss = 0
                full_data_dict = None

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)
                    _batch = apply_preblocks(self.preblocks, batch, device=self.device)

                    if t == 1:
                        self.ic_channel_map = _batch["metadata"]["input"]["_channel_map"]
                        x = _batch["input"].float()
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)
                        full_data_dict = self._init_full_data_dict(batch, _batch, x)
                    else:
                        curr_dyn_input = _batch["input"].float()
                        if self.ensemble_size > 1:
                            curr_dyn_input = torch.repeat_interleave(curr_dyn_input, self.ensemble_size, 0)
                        x = build_rollout_input(full_data_dict, curr_dyn_input, self.ic_channel_map)
                        self._update_full_data_dict(full_data_dict, _batch, x)

                    if self.flag_clamp:
                        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
                        full_data_dict["preprocessed"]["input"] = x

                    y_pred_flat = self.model(x.float())

                    full_data_dict["predicted"] = y_pred_flat
                    if self.postblocks:
                        full_data_dict = apply_postblocks(self.postblocks, full_data_dict)

                    if t == self.valid_forecast_len:
                        y = _batch["target"].float()
                        if self.flag_clamp:
                            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
                        loss = criterion(y.to(y_pred_flat.dtype), y_pred_flat).mean()
                        metrics_dict = metrics(y_pred_flat.float(), y.float())
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


Trainer = TrainerERA5Gen2  # canonical alias, matches other trainer modules
