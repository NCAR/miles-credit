import argparse
import logging
from credit.distributed import get_rank_info, setup
from credit.seed import seed_everything
from os.path import expandvars
import os
import torch
import yaml
import sys
from pathlib import Path
from credit.pbs import launch_script, launch_script_mpi
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
import shutil
from credit.preblock import build_preblocks, apply_preblocks_before_scaler
from credit.trainers.utils import cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="model_config", required=True, type=str, help="Path to config file")
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

    with open(args.config) as config_file:
        conf = yaml.safe_load(config_file)
    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    rank = world_rank
    save_loc = expandvars(conf["save_loc"])
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

    if conf["trainer"]["mode"] in ["fsdp", "ddp", "domain_parallel", "fsdp+domain_parallel"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    trainer_conf = conf["trainer"]
    train_dataset = load_dataset(conf, is_train=True)
    train_loader = load_dataloader(conf, train_dataset, rank=rank, world_size=world_size, is_train=True)
    seed = conf.get("seed", 42) + rank
    seed_everything(seed)
    preblocks = build_preblocks(conf["preblocks"])
    scaler_block_key = None
    for k, v in preblocks.items():
        if v["type"] == "bridgescaler_transform":
            scaler_block_key = k
            break
    if scaler_block_key is None:
        raise ValueError("BridgeScalerTransformer not found in preblocks.")
    batches_per_epoch = trainer_conf.get("batches_per_epoch", 1)
    dl = cycle(train_loader)
    for i in range(batches_per_epoch):
        batch = next(dl)
        processed_batch = apply_preblocks_before_scaler(preblocks, batch, device)
        preblocks[scaler_block_key].fit_scaler_batch(processed_batch)

    return


if __name__ == "__main__":
    main()
