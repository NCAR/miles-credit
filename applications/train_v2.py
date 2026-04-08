"""
train_v2.py
-----------
Gen2 training entry point for the nested ERA5 data schema.

This script bypasses ``credit_main_parser`` entirely. It reads the config
directly and assumes the new ``conf["data"]["source"]`` structure produced by
``MultiSourceDataset`` / ``ERA5Dataset``.

For the legacy flat schema (v1), use ``applications/train.py``.
"""

import os
import sys
import yaml
import shutil
import logging
import warnings

from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from credit.distributed import distributed_model_wrapper, setup, get_rank_info
from credit.seed import seed_everything
from credit.losses import load_loss
from credit.scheduler import load_scheduler
from credit.trainers import load_trainer
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)
from credit.metrics import LatWeightedMetrics
from credit.datasets.multi_source import MultiSourceDataset
from credit.samplers import DistributedMultiStepBatchSampler

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _inject_flat_var_keys(conf: dict) -> None:
    """Inject Gen1-compatible variable keys into conf["data"] for metrics/loss.

    ``LatWeightedMetrics`` and ``VariableTotalLoss2D`` expect flat lists at:
    - ``conf["data"]["variables"]``       (upper-air / 3D vars)
    - ``conf["data"]["surface_variables"]`` (2D prognostic vars)
    - ``conf["data"]["diagnostic_variables"]``

    These are derived from the nested Gen2 source config so that metrics/loss
    classes can be used without modification.
    """
    if "variables" in conf["data"]:
        return  # already injected or user provided them manually

    # Extract from the first (and usually only) source — ERA5
    source_conf = next(iter(conf["data"]["source"].values()))
    vars_conf = source_conf.get("variables", {})

    prog = vars_conf.get("prognostic") or {}
    diag = vars_conf.get("diagnostic") or {}

    conf["data"]["variables"] = prog.get("vars_3D", [])
    conf["data"]["surface_variables"] = prog.get("vars_2D", [])
    conf["data"]["diagnostic_variables"] = (diag.get("vars_3D", []) + diag.get("vars_2D", [])) if diag else []


def _load_dataset(conf: dict, is_train: bool) -> MultiSourceDataset:
    """Build a MultiSourceDataset for train or validation."""
    if is_train:
        data_conf = conf["data"]
    else:
        # Merge data_valid overrides on top of data
        data_conf = {**conf["data"], **conf.get("data_valid", {})}
        # source config always comes from conf["data"]
        data_conf["source"] = conf["data"]["source"]

    return MultiSourceDataset(data_conf, return_target=True)


def _load_dataloader(
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


def main(rank, world_size, conf, backend=None, trial=False):
    """Set up and run training."""
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp", "domain_parallel", "fsdp+domain_parallel"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

    train_dataset = _load_dataset(conf, is_train=True)
    valid_dataset = _load_dataset(conf, is_train=False)

    train_loader = _load_dataloader(conf, train_dataset, rank=rank, world_size=world_size, is_train=True)
    valid_loader = _load_dataloader(conf, valid_dataset, rank=rank, world_size=world_size, is_train=False)

    seed = conf.get("seed", 42) + rank
    seed_everything(seed)

    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    if conf["trainer"]["mode"] in ["ddp", "fsdp", "domain_parallel", "fsdp+domain_parallel"]:
        model = distributed_model_wrapper(conf, m, device)
    else:
        model = m

    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    train_criterion = load_loss(conf)
    valid_criterion = load_loss(conf, validation=True)

    # Inject flat variable keys so LatWeightedMetrics / load_loss can build
    # their variable lists without touching credit_main_parser.
    _inject_flat_var_keys(conf)
    metrics = LatWeightedMetrics(conf)

    trainer_cls = load_trainer(conf)
    trainer = trainer_cls(model, rank, conf)

    return trainer.fit(
        conf,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        train_criterion=train_criterion,
        valid_criterion=valid_criterion,
        scaler=scaler,
        scheduler=scheduler,
        metrics=metrics,
        trial=trial,
    )


def main_cli():
    description = (
        "Train a Gen2 AI weather model using the nested ERA5 data schema "
        "(conf['data']['source']). For the legacy flat schema, use train.py."
    )
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        dest="model_config",
        type=str,
        required=True,
        help="Path to the model config YAML (Gen2 nested schema).",
    )
    parser.add_argument("-l", dest="launch", type=int, default=0, help="Submit workers to PBS.")
    parser.add_argument(
        "--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"], help="Backend for distributed training."
    )
    args = parser.parse_args()
    config = args.model_config
    launch = int(args.launch)
    backend = args.backend

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    gettrace = getattr(sys, "gettrace", None)
    ch.setLevel(logging.DEBUG if gettrace and gettrace() else logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "train_v2.py requires the Gen2 nested data schema (conf['data']['source']). "
        "For the legacy flat schema, use applications/train.py."
    )

    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    if launch:
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    main(world_rank, world_size, conf, backend)


if __name__ == "__main__":
    main_cli()
