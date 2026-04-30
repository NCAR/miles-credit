"""
base_trainer.py
-------------------------------------------------------
Content:
    - EMATracker
    - BaseTrainer (abstract)
        - train_one_epoch  (abstract)
        - validate         (abstract)
        - fit
        - _save_checkpoint (internal)
"""

import gc
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from credit.models.checkpoint import TorchFSDPCheckpointIO, copy_checkpoint
from credit.scheduler import update_on_epoch
from credit.trainers.preflight import check_dataloader_startup, check_model_gpu_memory
from credit.trainers.utils import cleanup

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:
    _SummaryWriter = None

logger = logging.getLogger(__name__)


class EMATracker:
    """Exponential moving average of model weights.

    Maintains a shadow copy of the model parameters:
        shadow = decay * shadow + (1 - decay) * param

    Uses adaptive decay so short runs are not dominated by initial random weights:
        effective_decay = min(max_decay, (1 + step) / (10 + step))

    This ramps from ~0.09 at step 0 to max_decay asymptotically, so validation
    always reflects recent training weights regardless of run length.

    Usage:
        ema = EMATracker(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # before validation:
        ema.swap(model)   # model now holds EMA weights
        ...validate...
        ema.swap(model)   # restore training weights

    Typical max_decay: 0.9999 for long runs, 0.999 for short runs.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step = 0
        # shadow lives on CPU to save GPU memory
        self.shadow: OrderedDict = OrderedDict()
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        for k, v in state.items():
            self.shadow[k] = v.detach().float().cpu().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        effective_decay = min(self.decay, (1 + self.step) / (10 + self.step))
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        for k, v in state.items():
            self.shadow[k].mul_(effective_decay).add_(v.detach().float().cpu(), alpha=1.0 - effective_decay)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module):
        """Swap model weights with EMA shadow weights (and vice-versa)."""
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        device = next(iter(state.values())).device
        dtype = next(iter(state.values())).dtype
        for k in self.shadow:
            tmp = state[k].clone()
            state[k].copy_(self.shadow[k].to(device=device, dtype=dtype))
            self.shadow[k] = tmp.float().cpu()
        if hasattr(model, "module"):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay, "step": self.step}

    def load_state_dict(self, d):
        self.decay = d["decay"]
        self.shadow = d["shadow"]
        self.step = d.get("step", 0)


class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, rank: int, conf: Dict[str, Any]):
        """
        Abstract base class for training and validating machine learning models.

        Extracts all trainer and model settings from conf at construction time so
        that fit(), train_one_epoch(), and validate() have clean, minimal signatures.

        Args:
            model: The (possibly DDP/FSDP-wrapped) model to train.
            rank: Global rank of this process (0 for non-distributed).
            conf: Full configuration dict loaded from YAML.
        """
        super().__init__()
        self.model = model
        self.rank = rank
        self.device = (
            torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Store full conf for subclass method bodies that still need it
        self.conf = conf

        # ---- Extract all trainer settings ----
        trainer_conf = conf["trainer"]
        self.save_loc = os.path.expandvars(conf["save_loc"])
        self.mode = trainer_conf.get("mode", "none")
        # V2 parallelism block: promote mode so checkpoint/AMP use the right path
        _p = trainer_conf.get("parallelism", {})
        _data_mode = _p.get("data", "none")
        if _data_mode == "fsdp2":
            self.mode = "fsdp2"
        elif _data_mode == "ddp" and self.mode not in ("ddp", "fsdp", "fsdp2"):
            self.mode = "ddp"
        self.distributed = self.mode in ("fsdp", "ddp", "fsdp2", "domain_parallel", "fsdp+domain_parallel") or (
            int(_p.get("domain", 1)) > 1 or int(_p.get("tensor", 1)) > 1
        )
        self.start_epoch = trainer_conf.get("start_epoch", 0)
        self.epochs = trainer_conf.get("epochs", 70)
        self.skip_validation = trainer_conf.get("skip_validation", False)
        self.load_weights = trainer_conf.get("load_weights", False)
        self.use_scheduler = trainer_conf.get("use_scheduler", True)
        self.scheduler_type = trainer_conf.get("scheduler", {}).get("scheduler_type", "")
        self.amp = trainer_conf.get("amp", True)
        self.grad_max_norm = trainer_conf.get("grad_max_norm", 0.0)
        self.batches_per_epoch = trainer_conf.get("batches_per_epoch", 0)
        self.valid_batches_per_epoch = trainer_conf.get("valid_batches_per_epoch", 0)
        self.ensemble_size = trainer_conf.get("ensemble_size", 1)
        self.save_best_weights = trainer_conf.get("save_best_weights", True)
        self.save_backup_weights = trainer_conf.get("save_backup_weights", True)
        self.stopping_patience = trainer_conf.get("stopping_patience", 100)
        self.save_every_epoch = trainer_conf.get("save_every_epoch", False)
        self.stop_after_epoch = trainer_conf.get("stop_after_epoch", False)
        # num_epoch caps how many additional epochs run from start_epoch.
        # Defaults to a large sentinel so `epochs` is the effective limit when
        # num_epoch is not set in the config.
        self.num_epoch = trainer_conf.get("num_epoch", int(1e8))
        self.save_metric_vars = trainer_conf.get("save_metric_vars", [])
        self.train_one_epoch_mode = trainer_conf.get("train_one_epoch", False)

        training_metric = trainer_conf.get("training_metric", "train_loss" if self.skip_validation else "valid_loss")
        self.training_metric = training_metric
        direction = trainer_conf.get("training_metric_direction", "min")
        self.direction = min if direction == "min" else max

        logger.info(f"Training metric: {self.training_metric} (direction: {'min' if self.direction is min else 'max'})")

        # ---- EMA setup ----
        use_ema = trainer_conf.get("use_ema", False)
        if use_ema:
            ema_decay = trainer_conf.get("ema_decay", 0.9999)
            self.ema: Optional[EMATracker] = EMATracker(self.model, decay=ema_decay)
            ema_path = os.path.join(self.save_loc, "checkpoint_ema.pt")
            if os.path.exists(ema_path):
                ema_ckpt = torch.load(ema_path, map_location="cpu", weights_only=False)
                self.ema.load_state_dict(ema_ckpt["model_state_dict"])
                logger.info(f"Resumed EMA from {ema_path} (step {self.ema.step})")
            else:
                logger.info(f"EMA enabled (decay={ema_decay})")
        else:
            self.ema = None

        # ---- TensorBoard setup ----
        use_tb = trainer_conf.get("use_tensorboard", False)
        if use_tb and self.rank == 0:
            if _SummaryWriter is None:
                logger.warning(
                    "use_tensorboard=True but torch.utils.tensorboard is not available. "
                    "Install tensorboard: pip install tensorboard"
                )
                self.tb_writer = None
            else:
                tb_dir = os.path.join(self.save_loc, "tensorboard")
                self.tb_writer = _SummaryWriter(log_dir=tb_dir)
                logger.info(f"TensorBoard log dir: {tb_dir}")
                logger.info(f"  View with: tensorboard --logdir {tb_dir}")
        else:
            self.tb_writer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_state_dict(model: torch.nn.Module) -> dict:
        """Return state dict, unwrapping DDP .module wrapper if present."""
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    @staticmethod
    def _load_model_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
        """Load state dict, unwrapping DDP .module wrapper if present."""
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train_one_epoch(
        self,
        epoch: int,
        trainloader: DataLoader,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        scaler: GradScaler,
        scheduler: LRScheduler,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        epoch: int,
        valid_loader: DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Progress logging
    # ------------------------------------------------------------------

    def _log_batch_progress(
        self,
        epoch: int,
        results_dict: dict,
        optimizer: Optional[Optimizer],
        pbar,
        phase: str = "train",
    ) -> None:
        """Update a tqdm progress bar with rolling-mean batch metrics."""
        parts = [f"Epoch: {epoch}"]
        for key in (f"{phase}_loss", f"{phase}_acc", f"{phase}_mae"):
            if results_dict.get(key):
                parts.append(f"{key}: {np.mean(results_dict[key]):.6f}")
        if self.ensemble_size > 1 and results_dict.get(f"{phase}_std"):
            parts.append(f"{phase}_std: {np.mean(results_dict[f'{phase}_std']):.6f}")
        if phase == "train":
            parts.append(f"lr: {optimizer.param_groups[0]['lr']:.12f}")
        if self.rank == 0:
            pbar.update(1)
            pbar.set_description(" ".join(parts))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler,
    ) -> None:
        """Save model, optimizer, scheduler, and scaler state."""
        sched_state = scheduler.state_dict() if self.use_scheduler and scheduler is not None else None

        if self.mode == "fsdp2":
            from credit.parallel.fsdp2 import fsdp2_state_dict

            # FSDP2: all ranks gather full state dict, rank 0 saves
            logger.info(f"Saving FSDP2 checkpoint to {self.save_loc}")
            model_sd = fsdp2_state_dict(self.model)
            if self.rank == 0:
                state_dict = {
                    "epoch": epoch,
                    "model_state_dict": model_sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": sched_state,
                    "scaler_state_dict": scaler.state_dict(),
                }
                torch.save(state_dict, os.path.join(self.save_loc, "checkpoint.pt"))
                if self.save_every_epoch:
                    copy_checkpoint(os.path.join(self.save_loc, "checkpoint.pt"), epoch)
        elif self.mode != "fsdp":
            if self.rank == 0:
                logger.info(f"Saving checkpoint to {self.save_loc}")
                state_dict = {
                    "epoch": epoch,
                    "model_state_dict": self._model_state_dict(self.model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": sched_state,
                    "scaler_state_dict": scaler.state_dict(),
                }
                torch.save(state_dict, os.path.join(self.save_loc, "checkpoint.pt"))

                if self.ema is not None:
                    torch.save(
                        {"epoch": epoch, "model_state_dict": self.ema.state_dict()},
                        os.path.join(self.save_loc, "checkpoint_ema.pt"),
                    )

                if self.save_every_epoch:
                    copy_checkpoint(os.path.join(self.save_loc, "checkpoint.pt"), epoch)
        else:
            logger.info(f"Saving FSDP checkpoint to {self.save_loc}")
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.save_unsharded_model(
                self.model,
                os.path.join(self.save_loc, "model_checkpoint.pt"),
                gather_dtensor=True,
                use_safetensors=False,
                rank=self.rank,
            )
            checkpoint_io.save_unsharded_optimizer(
                optimizer,
                os.path.join(self.save_loc, "optimizer_checkpoint.pt"),
                gather_dtensor=True,
                rank=self.rank,
            )
            state_dict = {
                "epoch": epoch,
                "scheduler_state_dict": sched_state,
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(state_dict, os.path.join(self.save_loc, "checkpoint.pt"))
            if self.save_every_epoch:
                copy_checkpoint(os.path.join(self.save_loc, "model_checkpoint.pt"), epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        conf: Dict[str, Any],
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: Optimizer,
        train_criterion: torch.nn.Module,
        valid_criterion: torch.nn.Module,
        scaler: GradScaler,
        scheduler: LRScheduler,
        metrics: Dict[str, Any],
        rollout_scheduler: Optional[callable] = None,
        trial: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Args:
            conf: Full configuration dict (passed through to train_one_epoch/validate
                  for data-related settings; trainer settings are accessed via self).
            train_loader, valid_loader: DataLoaders.
            optimizer, train_criterion, valid_criterion, scaler, scheduler, metrics:
                Training objects.
            rollout_scheduler: Optional callable to schedule rollout probability.
            trial: Optuna trial object, or False.

        Returns:
            Dict with the best epoch's results.
        """
        os.makedirs(self.save_loc, exist_ok=True)

        start_epoch = self.start_epoch
        epochs = self.epochs

        # Reload results log if resuming from a checkpoint
        results_dict: Dict[str, list] = defaultdict(list)
        log_path = os.path.join(self.save_loc, "training_log.csv")
        if start_epoch > 0 and self.load_weights and os.path.exists(log_path):
            saved = pd.read_csv(log_path)
            for key in saved.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved[key])

        # train_one_epoch mode: run exactly one more epoch
        if self.train_one_epoch_mode:
            if results_dict:
                start_epoch = len(results_dict.get("epoch", []))
            epochs = start_epoch + 1

        # Cap total epochs by num_epoch
        epoch_limit = min(epochs, start_epoch + self.num_epoch)

        # Some datasets (e.g. ERA5_MultiStep_Batcher) need set_epoch called before
        # __getitem__ is accessible; initialize here so preflight doesn't crash.
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(start_epoch)

        # Preflight: check DataLoader memory and first-batch latency (rank-0 only)
        timeout_s = conf.get("trainer", {}).get("dataloader_timeout_s", 300)
        check_dataloader_startup(conf, train_loader, rank=self.rank, timeout_s=timeout_s)

        # Preflight: synthetic forward/backward/optimizer step to measure peak VRAM
        check_model_gpu_memory(conf, self.model, optimizer, rank=self.rank)

        for epoch in range(start_epoch, epoch_limit):
            # Backup previous epoch's checkpoint
            if epoch > start_epoch and self.save_backup_weights and self.rank == 0:
                for fname in ("checkpoint.pt",):
                    src = os.path.join(self.save_loc, fname)
                    if os.path.exists(src):
                        shutil.copyfile(src, os.path.join(self.save_loc, f"backup_{fname}"))
                if self.mode == "fsdp":
                    for fname in ("model_checkpoint.pt", "optimizer_checkpoint.pt"):
                        src = os.path.join(self.save_loc, fname)
                        if os.path.exists(src):
                            shutil.copyfile(src, os.path.join(self.save_loc, f"backup_{fname}"))

            logger.info(f"Beginning epoch {epoch}")

            # Set epoch on sampler/dataset for reproducible distributed shuffling
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            if not self.skip_validation:
                with torch.no_grad():
                    if hasattr(valid_loader, "sampler") and hasattr(valid_loader.sampler, "set_epoch"):
                        valid_loader.sampler.set_epoch(epoch)
                    if hasattr(valid_loader.dataset, "set_epoch"):
                        valid_loader.dataset.set_epoch(epoch)

            # ---- Train ----
            train_results = self.train_one_epoch(
                epoch, train_loader, optimizer, train_criterion, scaler, scheduler, metrics
            )

            # ---- Validate ----
            if self.skip_validation:
                valid_results = train_results
            else:
                if self.ema is not None:
                    self.ema.swap(self.model)
                valid_results = self.validate(epoch, valid_loader, valid_criterion, metrics)
                if self.ema is not None:
                    self.ema.swap(self.model)

            # ---- Collect results ----
            results_dict["epoch"].append(epoch)

            required_metrics = ["loss", "acc", "mae", "forecast_len"]
            if isinstance(self.save_metric_vars, list) and len(self.save_metric_vars) > 0:
                names = [
                    key.replace("train_", "")
                    for key in train_results.keys()
                    if any(var in key for var in self.save_metric_vars)
                ]
            elif isinstance(self.save_metric_vars, bool) and self.save_metric_vars:
                names = [key.replace("train_", "") for key in train_results.keys()]
            else:
                names = []
            names = list(set(names + required_metrics))

            for name in names:
                if f"train_{name}" in train_results:
                    results_dict[f"train_{name}"].append(np.mean(train_results[f"train_{name}"]))
                if not self.skip_validation and f"valid_{name}" in valid_results:
                    results_dict[f"valid_{name}"].append(np.mean(valid_results[f"valid_{name}"]))
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            # Update scheduler (epoch-level)
            if self.use_scheduler and self.scheduler_type in update_on_epoch:
                if self.scheduler_type == "plateau":
                    scheduler.step(results_dict[self.training_metric][-1])
                else:
                    scheduler.step()

            # ---- Save training log ----
            max_len = max(len(lst) for lst in results_dict.values())
            padded = OrderedDict((k, [np.nan] * (max_len - len(v)) + v) for k, v in results_dict.items())
            df = pd.DataFrame.from_dict(padded).reset_index()
            if trial:
                trial_dir = os.path.join(self.save_loc, "trial_results")
                os.makedirs(trial_dir, exist_ok=True)
                df.to_csv(os.path.join(trial_dir, f"training_log_{trial.number}.csv"), index=False)
            else:
                df.to_csv(log_path, index=False)

            # ---- TensorBoard logging ----
            if self.tb_writer is not None:
                for key, vals in results_dict.items():
                    if key == "epoch" or not vals:
                        continue
                    val = vals[-1]
                    if np.isfinite(val):
                        # Group train_*/valid_* under "Loss/", "Acc/", etc.; others under "train/"
                        if key.startswith("train_") or key.startswith("valid_"):
                            prefix, metric = key.split("_", 1)
                            tag = f"{metric}/{prefix}"
                        else:
                            tag = f"train/{key}"
                        self.tb_writer.add_scalar(tag, val, epoch)
                self.tb_writer.flush()

            # ---- Save checkpoint ----
            if not trial:
                self._save_checkpoint(epoch, optimizer, scheduler, scaler)

            torch.cuda.empty_cache()
            gc.collect()

            # ---- Best checkpoint / early stopping ----
            if not self.skip_validation and self.training_metric in results_dict:
                metric_history = results_dict[self.training_metric]
                best_val = self.direction(metric_history)
                best_idx = metric_history.index(best_val)
                best_epoch_actual = results_dict["epoch"][best_idx]
                offset = epoch - best_epoch_actual

                if offset == 0 and self.save_best_weights and self.rank == 0:
                    for fname in ("checkpoint.pt", "checkpoint_ema.pt"):
                        src = os.path.join(self.save_loc, fname)
                        if os.path.exists(src):
                            shutil.copyfile(src, os.path.join(self.save_loc, f"best_{fname}"))
                    if self.mode == "fsdp":
                        for fname in ("model_checkpoint.pt", "optimizer_checkpoint.pt"):
                            src = os.path.join(self.save_loc, fname)
                            if os.path.exists(src):
                                shutil.copyfile(src, os.path.join(self.save_loc, f"best_{fname}"))

                if offset >= self.stopping_patience:
                    logger.info(
                        f"Early stopping: best {self.training_metric} at epoch {best_epoch_actual}, "
                        f"current epoch {epoch}"
                    )
                    break

            if self.stop_after_epoch:
                break

        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()

        # Return best epoch results
        if self.training_metric in results_dict and results_dict[self.training_metric]:
            metric_history = results_dict[self.training_metric]
            best_idx = metric_history.index(self.direction(metric_history))
            result = {k: v[best_idx] for k, v in results_dict.items() if len(v) > best_idx}
        else:
            result = dict(results_dict)

        if self.distributed:
            cleanup()

        return result
