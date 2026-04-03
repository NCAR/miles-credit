"""
train_v2.py
-----------
Training entry point for the new nested data schema (conf["data"]["source"]).
Uses MultiSourceDataset + DistributedMultiStepBatchSampler directly.
No credit_main_parser / training_data_check — those are for the old schema.
"""

import logging
import os
import shutil
import warnings
from argparse import ArgumentParser

import optuna
import torch
import yaml
from torch.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from credit.datasets.multi_source import MultiSourceDataset
from credit.distributed import distributed_model_wrapper, distributed_model_wrapper_v2, get_rank_info, setup
from credit.parallel.mesh import parse_parallelism_conf
from credit.losses import load_loss
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)
from credit.samplers import DistributedMultiStepBatchSampler
from credit.scheduler import load_scheduler
from credit.seed import seed_everything
from credit.trainers import load_trainer

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger(__name__)


def _load_dataset(data_conf, is_train: bool) -> MultiSourceDataset:
    return MultiSourceDataset(data_conf, return_target=True)


def _load_dataloader(conf, dataset: MultiSourceDataset, rank: int, world_size: int, is_train: bool) -> DataLoader:
    training_type = "train" if is_train else "valid"
    batch_size = conf["trainer"][f"{training_type}_batch_size"]
    num_workers = conf["trainer"]["thread_workers"] if is_train else conf["trainer"]["valid_thread_workers"]
    prefetch_factor = conf["trainer"].get("prefetch_factor", 4) if num_workers > 0 else None
    shuffle = is_train
    seed = conf["seed"]

    # Pick forecast_len from the appropriate data block
    data_conf = conf["data"] if is_train else conf.get("data_valid", conf["data"])
    forecast_len = data_conf["forecast_len"]

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
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
    )


def load_model_states_and_optimizer(conf, model, device):
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
        if conf["trainer"].get("mode", "none") == "fsdp":
            opt = FSDPOptimizerWrapper(opt, model)
        return opt

    if not load_weights:
        optimizer = _make_optimizer(model)
        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp) if conf["trainer"].get("mode", "none") == "fsdp" else GradScaler(enabled=amp)
        )

    elif load_weights and not (load_optimizer_conf or load_scaler_conf or load_scheduler_conf):
        optimizer = _make_optimizer(model)
        if conf["trainer"].get("mode", "none") == "fsdp":
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
        else:
            ckpt = torch.load(os.path.join(save_loc, "checkpoint.pt"), map_location=device)
            load_msg = (model.module if conf["trainer"].get("mode", "none") == "ddp" else model).load_state_dict(
                ckpt["model_state_dict"], strict=False
            )
            load_state_dict_error_handler(load_msg)
            if conf["trainer"].get("reload_epoch") and os.path.exists(os.path.join(save_loc, "training_log.csv")):
                conf["trainer"]["start_epoch"] = ckpt["epoch"] + 1
        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp) if conf["trainer"].get("mode", "none") == "fsdp" else GradScaler(enabled=amp)
        )

    else:
        ckpt = torch.load(os.path.join(save_loc, "checkpoint.pt"), map_location=device)
        optimizer = _make_optimizer(model)
        if conf["trainer"].get("mode", "none") == "fsdp":
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if load_optimizer_conf:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))
        else:
            load_msg = (model.module if conf["trainer"].get("mode", "none") == "ddp" else model).load_state_dict(
                ckpt["model_state_dict"], strict=False
            )
            load_state_dict_error_handler(load_msg)
            if load_optimizer_conf:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler = load_scheduler(optimizer, conf)
        scaler = (
            ShardedGradScaler(enabled=amp) if conf["trainer"].get("mode", "none") == "fsdp" else GradScaler(enabled=amp)
        )
        if conf["trainer"].get("reload_epoch"):
            conf["trainer"]["start_epoch"] = ckpt["epoch"] + 1
        if conf["trainer"]["start_epoch"] > 0:
            if scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    if conf["trainer"].get("update_learning_rate", False):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler


def main(rank, world_size, conf, backend=None, trial=False):
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # Setup distributed process group.
    # V2 parallelism block takes precedence; falls back to legacy mode string.
    p_conf = parse_parallelism_conf(conf)
    _mode = p_conf["data"] if p_conf["data"] != "none" else conf["trainer"].get("mode", "none")
    _any_distributed = _mode in ("fsdp2", "fsdp", "ddp") or p_conf.get("tensor", 1) > 1 or p_conf.get("domain", 1) > 1
    if _any_distributed:
        setup(rank, world_size, _mode, backend)

    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    seed_everything(conf["seed"] + rank)

    # Datasets and dataloaders
    train_data_conf = conf["data"]
    valid_data_conf = conf.get("data_valid", conf["data"])

    train_dataset = _load_dataset(train_data_conf, is_train=True)
    valid_dataset = _load_dataset(valid_data_conf, is_train=False)

    train_loader = _load_dataloader(conf, train_dataset, rank=rank, world_size=world_size, is_train=True)
    valid_loader = _load_dataloader(conf, valid_dataset, rank=rank, world_size=world_size, is_train=False)

    # Inject tracer_inds into post_conf so TracerFixer can initialize.
    # train_v2.py bypasses the v1 parser, so we compute channel indices here
    # from the v2 variable layout: prognostic/3d (each var × n_levels), prognostic/2d, diagnostic/2d.
    tracer_conf = conf.get("model", {}).get("post_conf", {}).get("tracer_fixer", {})
    if tracer_conf.get("activate", False) and "tracer_inds" not in tracer_conf:
        era5_src = conf["data"]["source"]["ERA5"]
        n_levels = len(era5_src.get("levels", []))
        v = era5_src["variables"]
        vars_3d = (v.get("prognostic") or {}).get("vars_3D", [])
        vars_2d = (v.get("prognostic") or {}).get("vars_2D", [])
        diag_2d = (v.get("diagnostic") or {}).get("vars_2D", [])
        output_vars = [vn for vn in vars_3d for _ in range(n_levels)] + vars_2d + diag_2d
        tracer_names = tracer_conf.get("tracer_name", [])
        tracer_thres_cfg = tracer_conf.get("tracer_thres", [])
        thres_map = dict(zip(tracer_names, tracer_thres_cfg))
        tracer_inds, tracer_thres = [], []
        for i, vn in enumerate(output_vars):
            if vn in thres_map:
                tracer_inds.append(i)
                tracer_thres.append(thres_map[vn])
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = tracer_inds
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = tracer_thres
        # denorm requires v1 inverse-transform which doesn't apply to v2 normalized tensors
        conf["model"]["post_conf"]["tracer_fixer"]["denorm"] = False

    # If tracer_fixer is active, inject channel indices for the v2 output layout.
    # The v1 parser normally computes this; train_v2.py skips the parser.
    # v2 output order: prognostic/3d (each var × n_levels), prognostic/2d, diagnostic/2d
    post_conf = conf.get("model", {}).get("post_conf", {})
    if post_conf.get("activate") and post_conf.get("tracer_fixer", {}).get("activate"):
        src = conf["data"]["source"]["ERA5"]
        n_levels = len(src.get("levels", []))
        prog = src["variables"].get("prognostic") or {}
        diag = src["variables"].get("diagnostic") or {}
        vars_3d = prog.get("vars_3D", [])
        vars_2d = prog.get("vars_2D", [])
        diag_2d = diag.get("vars_2D", [])
        output_vars = []
        for v in vars_3d:
            output_vars.extend([v] * n_levels)
        output_vars.extend(vars_2d)
        output_vars.extend(diag_2d)
        tracer_names = post_conf["tracer_fixer"].get("tracer_name", [])
        tracer_thres = post_conf["tracer_fixer"].get("tracer_thres", [])
        thres_dict = dict(zip(tracer_names, tracer_thres))
        inds, matched_thres = [], []
        for i, v in enumerate(output_vars):
            if v in thres_dict:
                inds.append(i)
                matched_thres.append(thres_dict[v])
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = inds
        conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = matched_thres
        conf["model"]["post_conf"]["tracer_fixer"]["denorm"] = False  # inverse transform uses v1 schema

    # Model
    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    # V2 parallelism: use new wrapper if parallelism: block present or FSDP2 mode requested
    _p = parse_parallelism_conf(conf)
    _use_v2_wrapper = (
        "parallelism" in conf.get("trainer", {})
        or _p["data"] == "fsdp2"
        or _p.get("tensor", 1) > 1
        or _p.get("domain", 1) > 1
    )
    if _use_v2_wrapper:
        model = distributed_model_wrapper_v2(conf, m, device)
    elif conf["trainer"].get("mode", "none") in ("ddp", "fsdp"):
        model = distributed_model_wrapper(conf, m, device)
    else:
        model = m

    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    # Inject v1-schema keys expected by load_loss and LatWeightedMetrics.
    # Must happen before load_loss — VariableTotalLoss2D reads conf["data"]["variables"].
    if "variables" not in conf["data"]:
        era5_vars = conf["data"]["source"]["ERA5"]["variables"]
        prog = era5_vars.get("prognostic") or {}
        diag = era5_vars.get("diagnostic") or {}
        conf["data"]["variables"] = prog.get("vars_3D", [])
        conf["data"]["surface_variables"] = prog.get("vars_2D", [])
        conf["data"]["diagnostic_variables"] = diag.get("vars_3D", []) + diag.get("vars_2D", []) if diag else []

    train_criterion = load_loss(conf)
    valid_criterion = load_loss(conf, validation=True)

    metrics = LatWeightedMetrics(conf)

    trainer_cls = load_trainer(conf)
    trainer = trainer_cls(model, rank, conf)

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
        trial=trial,
    )
    return result


class Objective:
    def __init__(self, config, metric="val_loss", device="cpu"):
        self.config = config
        self.metric = metric
        self.device = device

    def train(self, trial, conf):
        try:
            return main(0, 1, conf, trial=trial)
        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                raise optuna.TrialPruned()
            raise E


def main_cli():
    parser = ArgumentParser(description="Train a CREDIT v2 model using the new nested data schema.")
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"])
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "train_v2.py requires the new nested data schema (conf['data']['source']). "
        "For the old schema, use applications/train.py."
    )

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(args.model_config, os.path.join(conf["save_loc"], "model.yml"))

    _p = parse_parallelism_conf(conf)
    _rank_mode = _p["data"] if _p["data"] != "none" else conf["trainer"].get("mode", "none")
    # get_rank_info only knows fsdp/ddp/domain_parallel; map fsdp2 → fsdp for rank detection
    _rank_mode_compat = "fsdp" if _rank_mode == "fsdp2" else _rank_mode
    if _p.get("tensor", 1) > 1 or _p.get("domain", 1) > 1:
        _rank_mode_compat = "fsdp"  # use torchrun env vars path
    local_rank, world_rank, world_size = get_rank_info(_rank_mode_compat)
    main(world_rank, world_size, conf, args.backend)


if __name__ == "__main__":
    main_cli()
