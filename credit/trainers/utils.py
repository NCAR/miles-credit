import os

import torch
from torch.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist

from credit.scheduler import load_scheduler
from credit.samplers import DistributedMultiStepBatchSampler
from credit.datasets.multi_source import MultiSourceDataset
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def inject_flat_var_keys(conf: dict) -> None:
    """Inject Gen1-compatible variable keys into conf["data"] for metrics/loss.

    ``LatWeightedMetrics`` and ``VariableTotalLoss2D`` expect flat lists at:
    - ``conf["data"]["variables"]``
    - ``conf["data"]["surface_variables"]``
    - ``conf["data"]["diagnostic_variables"]``

    These are derived from the nested Gen2 source config.
    """
    if "variables" in conf["data"]:
        return

    source_conf = next(iter(conf["data"]["source"].values()))
    vars_conf = source_conf.get("variables", {})

    prog = vars_conf.get("prognostic") or {}
    diag = vars_conf.get("diagnostic") or {}

    conf["data"]["variables"] = prog.get("vars_3D", [])
    conf["data"]["surface_variables"] = prog.get("vars_2D", [])
    conf["data"]["diagnostic_variables"] = (diag.get("vars_3D", []) + diag.get("vars_2D", [])) if diag else []


def load_dataset(conf: dict, is_train: bool) -> MultiSourceDataset:
    """Build a MultiSourceDataset for train or validation."""
    if is_train:
        data_conf = conf["data"]
    else:
        data_conf = {**conf["data"], **conf.get("validation_data", {})}
        data_conf["source"] = conf["data"]["source"]

    return MultiSourceDataset(data_conf, return_target=True)


def load_dataloader(
    conf: dict,
    dataset: MultiSourceDataset,
    rank: int,
    world_size: int,
    is_train: bool,
) -> DataLoader:
    """Build a DataLoader with DistributedMultiStepBatchSampler."""
    if is_train:
        batch_size = conf["trainer"]["train_batch_size"]
        shuffle = True
        seed = conf.get("seed", 42) + rank
    else:
        batch_size = conf["trainer"]["valid_batch_size"]
        shuffle = False
        seed = conf.get("seed", 42)

    forecast_len = conf["data"]["forecast_len"]
    num_workers = conf["trainer"].get("thread_workers" if is_train else "valid_thread_workers", 4)
    prefetch_factor = conf["trainer"].get("prefetch_factor", 2) if num_workers > 0 else None

    sampler = DistributedMultiStepBatchSampler(
        dataset,
        batch_size=batch_size,
        num_forecast_steps=forecast_len,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )


def load_model_states_and_optimizer(conf, model, device):
    """Load model weights, optimizer, scheduler, and gradient scaler."""
    conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

    learning_rate = float(conf["trainer"]["learning_rate"])
    weight_decay = float(conf["trainer"]["weight_decay"])
    amp = conf["trainer"]["amp"]

    load_weights = conf["trainer"].get("load_weights", False)
    load_optimizer_conf = conf["trainer"].get("load_optimizer", False)
    load_scaler_conf = conf["trainer"].get("load_scaler", False)
    load_scheduler_conf = conf["trainer"].get("load_scheduler", False)

    def _make_optimizer(model):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            opt = FSDPOptimizerWrapper(opt, model)
        return opt

    def _make_scaler():
        return ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    if not load_weights:
        optimizer = _make_optimizer(model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

    elif not (load_optimizer_conf or load_scaler_conf or load_scheduler_conf):
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = _make_optimizer(model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
        else:
            ckpt = os.path.join(save_loc, "checkpoint.pt")
            checkpoint = torch.load(ckpt, map_location=device)
            if conf["trainer"]["mode"] == "ddp":
                load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            load_state_dict_error_handler(load_msg)
            optimizer = _make_optimizer(model)

        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

        if conf["trainer"].get("reload_epoch", False) and os.path.exists(os.path.join(save_loc, "training_log.csv")):
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        if conf["trainer"]["mode"] == "fsdp":
            optimizer = _make_optimizer(model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if load_optimizer_conf:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))
        else:
            if conf["trainer"]["mode"] == "ddp":
                load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            load_state_dict_error_handler(load_msg)
            optimizer = _make_optimizer(model)
            if load_optimizer_conf:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = _make_scaler()

        if conf["trainer"].get("reload_epoch", False):
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

        if conf["trainer"]["start_epoch"] > 0:
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if conf["trainer"].get("update_learning_rate", False):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler
