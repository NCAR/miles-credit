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
             After this, full_data_dict["predicted"] is a nested physical-space dict.
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

        batch_group_generator = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True)
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            y_pred_flat = None  # flat model output; used for loss, metrics, and rollout
            y = None

            for t in range(1, self.forecast_len + 1):
                batch = next(dl)

                if t == 1:
                    ic_preprocessed = apply_preblocks(self.ic_preblocks, batch, device=self.device)
                    preprocessed_batch = apply_preblocks(self.step_preblocks, ic_preprocessed, device=self.device)
                    x = preprocessed_batch["input"].float()
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)
                    full_data_dict = {
                        "ic_preprocessed": ic_preprocessed,
                        "metadata": preprocessed_batch["metadata"],
                        "predicted": None,
                    }
                else:
                    rollout_batch = assemble_rollout_batch(
                        corrected_pred=full_data_dict["predicted"],
                        ic_preprocessed=full_data_dict["ic_preprocessed"],
                        curr_batch=batch,
                    )
                    preprocessed_batch = apply_preblocks(self.step_preblocks, rollout_batch, device=self.device)
                    x = preprocessed_batch["input"].float()
                    if self.ensemble_size > 1:
                        x = torch.repeat_interleave(x, self.ensemble_size, 0)

                if self.flag_clamp:
                    x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                with torch.autocast(device_type="cuda", enabled=self.amp):
                    y_pred_flat = self.model(x)

                # Postblocks first: Reconstruct splits y_pred_flat into a nested
                # physical-space dict; subsequent blocks apply corrections in-place.
                full_data_dict["predicted"] = y_pred_flat if self.retain_graph else y_pred_flat.detach()
                full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)

                # Loss on flat normalized prediction — y_pred_flat ref is still valid
                # after postblocks replaced full_data_dict["predicted"].
                if t in self.backprop_on_timestep:
                    y = preprocessed_batch["target"].float()
                    if self.flag_clamp:
                        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        loss = criterion(y.to(y_pred_flat.dtype), y_pred_flat).mean()
                    accum_log(logs, {"loss": loss.item()})
                    scaler.scale(loss).backward(retain_graph=self.retain_graph)

                if self.distributed:
                    torch.distributed.barrier()

            apply_postblocks(self.rollout_postblocks, full_data_dict)

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

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)

                    if t == 1:
                        ic_preprocessed = apply_preblocks(self.ic_preblocks, batch, device=self.device)
                        preprocessed_batch = apply_preblocks(self.step_preblocks, ic_preprocessed, device=self.device)
                        x = preprocessed_batch["input"].float()
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)
                        full_data_dict = {
                            "ic_preprocessed": ic_preprocessed,
                            "metadata": preprocessed_batch["metadata"],
                            "predicted": None,
                        }
                    else:
                        rollout_batch = assemble_rollout_batch(
                            corrected_pred=full_data_dict["predicted"],
                            ic_preprocessed=full_data_dict["ic_preprocessed"],
                            curr_batch=batch,
                        )
                        preprocessed_batch = apply_preblocks(self.step_preblocks, rollout_batch, device=self.device)
                        x = preprocessed_batch["input"].float()
                        if self.ensemble_size > 1:
                            x = torch.repeat_interleave(x, self.ensemble_size, 0)

                    if self.flag_clamp:
                        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

                    y_pred_flat = self.model(x.float())

                    full_data_dict["predicted"] = y_pred_flat
                    full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)

                    if t == self.valid_forecast_len:
                        y = preprocessed_batch["target"].float()
                        if self.flag_clamp:
                            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
                        loss = criterion(y.to(y_pred_flat.dtype), y_pred_flat).mean()
                        metrics_dict = metrics(y_pred_flat.float(), y.float())
                        for name, value in metrics_dict.items():
                            value = torch.Tensor([value]).to(self.device, non_blocking=True)
                            if self.distributed:
                                dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                            results_dict[f"valid_{name}"].append(value[0].item())

                apply_postblocks(self.rollout_postblocks, full_data_dict)

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
