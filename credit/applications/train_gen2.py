"""
train_gen2.py
-------------
Gen2 training entry point for the nested ERA5 data schema.

This script bypasses ``credit_main_parser`` entirely. It reads the config
directly and assumes the new ``conf["data"]["source"]`` structure produced by
``MultiSourceDataset`` / ``BaseDataset``.

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

from credit.distributed import distributed_model_wrapper_gen2, setup, get_rank_info
from credit.seed import seed_everything
from credit.losses.base_losses import is_crps_loss
from credit.trainers import load_trainer
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.trainers.utils import (
    inject_flat_var_keys,
    load_dataset,
    load_dataloader,
    load_model_states_and_optimizer,
)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
    if not root.handlers:
        root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "train.py requires the Gen2 nested data schema (conf['data']['source']). "
        "For the legacy flat schema, use applications/train.py."
    )
    loss_name = conf["loss"]["training_loss"]
    ensemble_size = int(conf["trainer"].get("ensemble_size", 1))
    if is_crps_loss(loss_name) and ensemble_size <= 1:
        raise ValueError(
            f"{loss_name} is an ensemble CRPS loss and requires trainer.ensemble_size > 1; "
            f"got trainer.ensemble_size={ensemble_size}."
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

    _trainer_conf = conf["trainer"]
    assert "parallelism" in _trainer_conf, (
        "Gen2 training configs must define trainer.parallelism with data, tensor, "
        "and domain fields. Configs from before the parallelism block (legacy "
        "trainer.mode) can be migrated with `credit convert`."
    )

    # V2 parallelism configs read rank info from the launcher (torchrun or MPI).
    # Without a launcher (plain `python`/`credit train` on one GPU), run
    # single-process instead of letting get_rank_info sys.exit hunting for env vars.
    # RANK: torchrun; OMPI: Open MPI; PMI: cray-mpich/MPICH; PALS: Cray PALS
    # without PMI passthrough; SLURM_PROCID: srun.
    _launcher_env = ("LOCAL_RANK", "RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PALS_RANKID", "SLURM_PROCID")
    if any(v in os.environ for v in _launcher_env):
        local_rank, world_rank, world_size = get_rank_info("ddp")
    else:
        local_rank, world_rank, world_size = 0, 0, 1
    rank = world_rank

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    if world_size > 1:
        setup(rank, world_size, "ddp", backend, device_id=device if torch.cuda.is_available() else None)

    # Dataset sharding uses the DATA-PARALLEL coordinate, not the global rank.
    # Ranks that differ only in tensor/domain coordinate must see the same batch
    # (TP all_reduce sums partial outputs; domain halo exchange passes boundary
    # rows) — see credit.parallel.mesh.data_parallel_coords for the full contract.
    from credit.parallel.mesh import data_parallel_coords

    data_rank, data_world_size = data_parallel_coords(conf)
    if loss_name == "ring-crps" and ensemble_size != data_world_size:
        raise ValueError(
            "ring-crps uses one ensemble member per data-parallel rank, so "
            f"trainer.ensemble_size must match the data-parallel world size; "
            f"got trainer.ensemble_size={ensemble_size}, data-parallel world size={data_world_size}."
        )

    # CRPS ensemble training requires every dp rank to see the SAME training
    # batches. Member diversity comes from ensemble sampling in the trainer and
    # per-dp-rank RNG seeds. Validation keeps dp sharding.
    ensemble_mode = is_crps_loss(loss_name) and ensemble_size > 1
    train_rank, train_world_size = (0, 1) if ensemble_mode else (data_rank, data_world_size)
    if ensemble_mode and data_world_size > 1:
        logging.info(
            f"CRPS ensemble training: {data_world_size} dp ranks receive shared batches; dp data sharding disabled"
        )

    train_dataset = load_dataset(conf, is_train=True)
    train_loader = load_dataloader(conf, train_dataset, rank=train_rank, world_size=train_world_size, is_train=True)

    skip_validation = conf["trainer"].get("skip_validation", False)
    if skip_validation:
        valid_loader = None
    else:
        valid_dataset = load_dataset(conf, is_train=False)
        valid_loader = load_dataloader(conf, valid_dataset, rank=data_rank, world_size=data_world_size, is_train=False)

    # Two-stage seeding. Stage 1: seed identically on ALL ranks before model
    # construction so every rank initializes the SAME weights. FSDP2's
    # fully_shard does not broadcast params from rank 0 (unlike DDP), so each
    # rank keeps its own dim-0 shard of whatever it built locally — per-rank
    # init seeds would make the global model a mixture of different inits,
    # varying with the mesh layout (e.g. tp=1 vs tp=2 train different models).
    # Ring-CRPS also requires identical weights on all dp replicas.
    seed = conf.get("seed", 42)
    seed_everything(seed)
    inject_flat_var_keys(conf)
    if "post_conf" in conf["model"]:
        warnings.warn(
            "Gen 2 training does not support Gen 1 postblocks (conf['model']['post_conf']). "
            "These will be ignored. Gen 2 postblocks (conf['postblocks']) are still applied normally."
        )
        conf["model"].pop("post_conf", None)
    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    model = distributed_model_wrapper_gen2(conf, m, device)

    # Stage 2: now that the model is built and wrapped, re-seed by the
    # data-parallel rank so training-time RNG (dropout masks, stochastic
    # preblocks, ensemble perturbations) differs across dp replicas for
    # ensemble diversity. TP/domain peers share a data_rank, so they still
    # draw identical masks as required by the parallelism contract.
    seed_everything(seed + data_rank)

    conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

    from credit.losses import load_loss

    train_criterion = load_loss(conf)
    valid_criterion = load_loss(conf, validation=True)
    from credit.metrics import LatWeightedMetrics

    metrics = LatWeightedMetrics(conf)

    trainer_cls = load_trainer(conf)
    trainer = trainer_cls(model, rank, conf)

    trainer.fit(
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


if __name__ == "__main__":
    main_cli()
