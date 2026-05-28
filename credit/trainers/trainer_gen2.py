import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm

import optuna

from credit.parallel.domain import (
    get_domain_manager,
    get_raw_model,
    shard_spatial,
    unpad_shard_interp,
    sync_domain_gradients,
)
from credit.postblock import build_postblocks, apply_postblocks
from credit.preblock import build_preblocks, apply_preblocks
from credit.trainers.rollout_utils import assemble_rollout_batch
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
          - Postblocks applied after model forward pass via apply_postblocks(phase="per_step")
          - Multi-step rollout uses assemble_rollout_batch() + apply_preblocks(phase="per_step") each step

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)
        logger.info("Loading ERA5 Gen 2 trainer (new nested data schema, preblock-assembled batches)")

        # ---- Domain parallel manager (None when not using domain parallel) ----
        self.domain_manager = get_domain_manager(model)
        self._raw_model = get_raw_model(model)
        raw_m = self._raw_model
        if (
            self.domain_manager is not None
            and self.domain_manager.domain_parallel_size > 1
            and getattr(raw_m, "use_padding", False)
        ):
            self._domain_pre_pad = raw_m.padding_opt
            self._domain_image_h = raw_m.image_height
            self._domain_image_w = raw_m.image_width
            raw_m.use_padding = False
            raw_m.use_interp = False
        else:
            self._domain_pre_pad = None
            self._domain_image_h = None
            self._domain_image_w = None

        preblock_cfg = conf.get("preblocks", {})
        self.ic_preblocks = build_preblocks(preblock_cfg, phase="ic_only")
        self.step_preblocks = build_preblocks(preblock_cfg, phase="per_step")

        postblock_cfg = conf.get("postblocks", {})
        self.step_postblocks = build_postblocks(postblock_cfg, phase="per_step")
        self.rollout_postblocks = build_postblocks(postblock_cfg, phase="post_rollout")

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
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch.

        The inner loop iterates over forecast_len autoregressive steps. For each step:
          1. Pull the next batch from the dataloader (raw, unnormalized).
          2. At t=1: IC-only preblocks produce ic_preprocessed (regridded statics);
             rollout preblocks produce the final normalized input x.
             At t>1: assemble rollout batch from corrected_pred (prognostic),
             ic_preprocessed (statics), and curr_batch (dynamic forcing);
             rollout preblocks normalize and concat.
          3. Forward pass → y_pred_flat (flat, normalized).
          4. Apply postblocks: Reconstruct → inverse scaler → physics fixers.
             After this, full_data_dict["y_processed"] is a nested dict split by Reconstruct.
          5. Compute loss on y_pred_flat vs the normalized target from preblocks.

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

        grad_accum_every = self.conf.get("trainer", {}).get("grad_accum_every", 1)

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            full_data_dict = {}

            for t in range(1, self.forecast_len + 1):
                batch = next(dl)

                if t == 1:
                    full_data_dict["ic_raw"] = batch["input"]
                    full_data_dict["x_raw"] = batch["input"]
                    full_data_dict["y_raw"] = batch["target"]
                    full_data_dict["ic_preprocessed"] = apply_preblocks(self.ic_preblocks, batch, device=self.device)
                    full_data_dict.update(
                        apply_preblocks(self.step_preblocks, full_data_dict["ic_preprocessed"], device=self.device)
                    )
                else:
                    full_data_dict["x_raw"] = batch["input"]
                    full_data_dict["y_raw"] = batch["target"]
                    full_data_dict.update(
                        apply_preblocks(
                            self.step_preblocks, assemble_rollout_batch(full_data_dict, batch), device=self.device
                        )
                    )

                if self.ensemble_size > 1:
                    full_data_dict["x"] = torch.repeat_interleave(full_data_dict["x"], self.ensemble_size, 0)

                if self.flag_clamp:
                    full_data_dict["x"] = torch.clamp(full_data_dict["x"], min=self.clamp_min, max=self.clamp_max)

                # domain parallel: pad + shard input before forward
                if self._domain_pre_pad is not None:
                    full_data_dict["x"] = self._domain_pre_pad.pad(full_data_dict["x"])
                full_data_dict["x"] = shard_spatial(full_data_dict["x"], self.domain_manager)

                # FSDP2 uses MixedPrecisionPolicy; skip manual autocast to avoid
                # conflicts with SpectralNorm power-iteration buffers.
                _amp = self.amp and self.mode != "fsdp2"
                if self._domain_pre_pad is not None:
                    self._raw_model._skip_internal_padding = True
                try:
                    with torch.autocast(device_type="cuda", enabled=_amp):
                        full_data_dict["y_pred"] = self.model(full_data_dict["x"])
                finally:
                    if self._domain_pre_pad is not None:
                        self._raw_model._skip_internal_padding = False
                if self._domain_pre_pad is not None:
                    full_data_dict["y_pred"] = unpad_shard_interp(
                        full_data_dict["y_pred"],
                        self._domain_pre_pad,
                        self.domain_manager,
                        self._domain_image_h,
                        self._domain_image_w,
                    )
                if full_data_dict["y_pred"].dim() == 5:
                    full_data_dict["y_pred"] = full_data_dict["y_pred"].flatten(1, 2)

                # domain parallel: shard y to match y_pred's spatial shard
                if "y" in full_data_dict and full_data_dict["y"] is not None:
                    _y = full_data_dict["y"]
                    if _y.dim() == 5:
                        _y = _y.flatten(1, 2)
                    full_data_dict["y"] = shard_spatial(_y, self.domain_manager)

                full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)

                if t in self.backprop_on_timestep:
                    if self.flag_clamp:
                        full_data_dict["y"] = torch.clamp(
                            full_data_dict["y"].float(), min=self.clamp_min, max=self.clamp_max
                        )
                    with torch.autocast(device_type="cuda", enabled=_amp):
                        loss = criterion(
                            full_data_dict["y"].float().to(full_data_dict["y_pred"].dtype),
                            full_data_dict["y_pred"],
                        ).mean()
                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss / grad_accum_every).backward(retain_graph=self.retain_graph)

                if self.distributed:
                    torch.distributed.barrier()

            full_data_dict = apply_postblocks(self.rollout_postblocks, full_data_dict)

            # optimizer step at accumulation boundary
            if (steps + 1) % grad_accum_every == 0 or steps == batches_per_epoch - 1:
                sync_domain_gradients(self.model, self.domain_manager)
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

            if full_data_dict.get("y_pred") is not None and full_data_dict.get("y") is not None:
                metrics_dict = metrics(full_data_dict["y_pred"], full_data_dict["y"])
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

                full_data_dict = {}

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)

                    if t == 1:
                        full_data_dict["ic_raw"] = batch["input"]
                        full_data_dict["x_raw"] = batch["input"]
                        full_data_dict["y_raw"] = batch["target"]
                        full_data_dict["ic_preprocessed"] = apply_preblocks(
                            self.ic_preblocks, batch, device=self.device
                        )
                        full_data_dict.update(
                            apply_preblocks(self.step_preblocks, full_data_dict["ic_preprocessed"], device=self.device)
                        )
                    else:
                        full_data_dict["x_raw"] = batch["input"]
                        full_data_dict["y_raw"] = batch["target"]
                        full_data_dict.update(
                            apply_preblocks(
                                self.step_preblocks, assemble_rollout_batch(full_data_dict, batch), device=self.device
                            )
                        )

                    if self.ensemble_size > 1:
                        full_data_dict["x"] = torch.repeat_interleave(full_data_dict["x"], self.ensemble_size, 0)

                    if self.flag_clamp:
                        full_data_dict["x"] = torch.clamp(full_data_dict["x"], min=self.clamp_min, max=self.clamp_max)

                    # domain parallel: pad + shard input before forward
                    if self._domain_pre_pad is not None:
                        full_data_dict["x"] = self._domain_pre_pad.pad(full_data_dict["x"])
                    full_data_dict["x"] = shard_spatial(full_data_dict["x"], self.domain_manager)

                    if self._domain_pre_pad is not None:
                        self._raw_model._skip_internal_padding = True
                    try:
                        full_data_dict["y_pred"] = self.model(full_data_dict["x"])
                    finally:
                        if self._domain_pre_pad is not None:
                            self._raw_model._skip_internal_padding = False
                    if self._domain_pre_pad is not None:
                        full_data_dict["y_pred"] = unpad_shard_interp(
                            full_data_dict["y_pred"],
                            self._domain_pre_pad,
                            self.domain_manager,
                            self._domain_image_h,
                            self._domain_image_w,
                        )
                    if full_data_dict["y_pred"].dim() == 5:
                        full_data_dict["y_pred"] = full_data_dict["y_pred"].flatten(1, 2)

                    # domain parallel: shard y to match y_pred's spatial shard
                    if "y" in full_data_dict and full_data_dict["y"] is not None:
                        _y = full_data_dict["y"]
                        if _y.dim() == 5:
                            _y = _y.flatten(1, 2)
                        full_data_dict["y"] = shard_spatial(_y, self.domain_manager)

                    full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)

                    if t == self.valid_forecast_len:
                        if self.flag_clamp:
                            full_data_dict["y"] = torch.clamp(
                                full_data_dict["y"].float(), min=self.clamp_min, max=self.clamp_max
                            )
                        loss = criterion(
                            full_data_dict["y"].float().to(full_data_dict["y_pred"].dtype),
                            full_data_dict["y_pred"],
                        ).mean()
                        metrics_dict = metrics(full_data_dict["y_pred"].float(), full_data_dict["y"].float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).to(self.device, non_blocking=True)
                            if self.distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                full_data_dict = apply_postblocks(self.rollout_postblocks, full_data_dict)

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
