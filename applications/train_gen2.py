"""
train_gen2.py
-------------
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

from credit.distributed import distributed_model_wrapper, setup, get_rank_info
from credit.seed import seed_everything
from credit.losses import load_loss
from credit.trainers import load_trainer
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.metrics import LatWeightedMetrics
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
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "train_gen2.py requires the Gen2 nested data schema (conf['data']['source']). "
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
    rank = world_rank

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp", "domain_parallel", "fsdp+domain_parallel"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_dataset = load_dataset(conf, is_train=True)
    train_loader = load_dataloader(conf, train_dataset, rank=rank, world_size=world_size, is_train=True)

    skip_validation = conf["trainer"].get("skip_validation", False)
    if skip_validation:
        valid_loader = None
    else:
        valid_dataset = load_dataset(conf, is_train=False)
        valid_loader = load_dataloader(conf, valid_dataset, rank=rank, world_size=world_size, is_train=False)

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

    inject_flat_var_keys(conf)

    train_criterion = load_loss(conf)
    valid_criterion = load_loss(conf, validation=True)
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
