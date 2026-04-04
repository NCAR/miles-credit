"""
train_v2_regional.py
-------------------------------------------------------
V2 training entry point for regional (WRF / downscaling) models.
Does NOT use credit_main_parser — the config is used as-is.
Supports dataset_type: wrf_singlestep | wrf_multistep | dscale_singlestep
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

from credit.distributed import distributed_model_wrapper, setup, get_rank_info
from credit.seed import seed_everything
from credit.losses import load_loss
from credit.scheduler import load_scheduler
from credit.trainers import load_trainer
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
from credit.metrics_downscaling import UnWeightedMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model_states_and_optimizer(conf, model, device):
    conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

    learning_rate = float(conf["trainer"]["learning_rate"])
    weight_decay = float(conf["trainer"]["weight_decay"])
    amp = conf["trainer"]["amp"]

    load_weights = conf["trainer"].get("load_weights", False)
    load_optimizer_conf = conf["trainer"].get("load_optimizer", False)
    load_scaler_conf = conf["trainer"].get("load_scaler", False)
    load_scheduler_conf = conf["trainer"].get("load_scheduler", False)

    if not load_weights:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    elif load_weights and not (load_optimizer_conf or load_scaler_conf or load_scheduler_conf):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        checkpoint_io = TorchFSDPCheckpointIO()
        if conf["trainer"]["mode"] == "fsdp":
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
        elif os.path.isfile(os.path.join(save_loc, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(save_loc, "checkpoint.pt"), map_location=device)
            load_state_dict_error_handler(model, checkpoint)
            logging.info(f"Loaded model weights from {save_loc}/checkpoint.pt")

    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        checkpoint_io = TorchFSDPCheckpointIO()
        if conf["trainer"]["mode"] == "fsdp":
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if load_optimizer_conf:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))
        elif os.path.isfile(os.path.join(save_loc, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(save_loc, "checkpoint.pt"), map_location=device)
            load_state_dict_error_handler(model, checkpoint)
            if load_optimizer_conf and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if load_scaler_conf and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if load_scheduler_conf and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if conf["trainer"].get("update_learning_rate", False):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler


def _regional_varnames(conf):
    """Build a flat variable name list for UnWeightedMetrics."""
    levels = conf["data"].get("levels", conf["model"].get("levels", 1))
    upper = conf["data"]["variables"]
    surface = conf["data"].get("surface_variables", [])
    varnames = [f"{v}_{k}" for v in upper for k in range(levels)]
    varnames += surface
    return varnames


def _populate_post_conf(conf):
    """Replicate the tracer_inds computation that credit_main_parser normally performs."""
    post_conf = conf.get("model", {}).get("post_conf", {})
    if not post_conf.get("activate", False):
        return
    tf_conf = post_conf.get("tracer_fixer", {})
    if not tf_conf.get("activate", False):
        return

    varname_output = _regional_varnames(conf)
    varname_tracers = tf_conf["tracer_name"]
    tracers_thres_input = tf_conf.get("tracer_thres", [0.0] * len(varname_tracers))
    tracers_thres_maximum = tf_conf.get("tracer_thres_max", None)

    tracer_threshold_dict = dict(zip(varname_tracers, tracers_thres_input))
    tracer_threshold_dict_max = dict(zip(varname_tracers, tracers_thres_maximum)) if tracers_thres_maximum else {}

    tracer_inds, tracer_thres, tracer_thres_max = [], [], []
    for i_var, var in enumerate(varname_output):
        if var in tracer_threshold_dict:
            tracer_inds.append(i_var)
            tracer_thres.append(float(tracer_threshold_dict[var]))
            if tracers_thres_maximum is not None:
                tracer_thres_max.append(float(tracer_threshold_dict_max[var]))

    tf_conf["tracer_inds"] = tracer_inds
    tf_conf["tracer_thres"] = tracer_thres
    if tracers_thres_maximum is not None:
        tf_conf["tracer_thres_max"] = tracer_thres_max


def main(rank, world_size, conf, backend=None):
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Populate keys that credit_main_parser would normally set, so transforms work.
    if "all_varnames" not in conf["data"]:
        conf["data"]["all_varnames"] = (
            conf["data"].get("variables", [])
            + conf["data"].get("surface_variables", [])
            + conf["data"].get("dynamic_forcing_variables", [])
            + conf["data"].get("diagnostic_variables", [])
        )
    if "all_varnames" not in conf["data"].get("boundary", {}):
        bnd = conf["data"].get("boundary", {})
        bnd["all_varnames"] = bnd.get("variables", []) + bnd.get("surface_variables", [])

    train_dataset = load_dataset(conf, rank=rank, world_size=world_size, is_train=True)
    valid_dataset = load_dataset(conf, rank=rank, world_size=world_size, is_train=False)

    train_loader = load_dataloader(conf, train_dataset, rank=rank, world_size=world_size, is_train=True)
    valid_loader = load_dataloader(conf, valid_dataset, rank=rank, world_size=world_size, is_train=False)

    seed_everything(conf["seed"] + rank)

    # Populate tracer_inds in post_conf (normally done by credit_main_parser).
    _populate_post_conf(conf)

    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    if conf["trainer"]["mode"] in ["ddp", "fsdp"]:
        model = distributed_model_wrapper(conf, m, device)
    else:
        model = m

    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    train_criterion = load_loss(conf)
    valid_criterion = load_loss(conf, validation=True)

    varnames = _regional_varnames(conf)
    metrics = UnWeightedMetrics(conf, varnames)

    trainer_cls = load_trainer(conf)
    trainer = trainer_cls(model, rank)

    result = trainer.fit(
        conf,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        train_criterion=train_criterion,
        valid_criterion=valid_criterion,
        scaler=scaler,
        scheduler=scheduler,
        metrics=metrics,
    )

    return result


def main_cli():
    description = "Train a regional (WRF/downscaling) AI model — V2, no parser."
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, required=True, help="Path to the model config (yml)."
    )
    parser.add_argument("-l", dest="launch", type=int, default=0, help="Submit workers to PBS.")
    parser.add_argument(
        "--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"], help="Backend for distributed training."
    )
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(args.model_config, os.path.join(save_loc, "model.yml"))

    if args.launch:
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            launch_script(args.model_config, script_path)
        else:
            launch_script_mpi(args.model_config, script_path)
        sys.exit()

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    main(world_rank, world_size, conf, args.backend)


if __name__ == "__main__":
    main_cli()
