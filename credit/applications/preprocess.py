import argparse
import logging

from bridgescaler import save_scaler_dict

from credit.distributed import get_rank_info, setup
from credit.seed import seed_everything
from os.path import expandvars
import os
import torch
import yaml
import sys
import shutil
from credit.preblock import build_preblocks, apply_preblocks_before_scaler, BridgeScalerTransformer
from credit.preblock.scaler import combine_scaler_dicts
from credit.trainers.utils import cycle, load_dataset, load_dataloader
import torch.distributed as dist
from torch.distributed import gather_object, barrier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="model_config", required=True, type=str, help="Path to config file")
    parser.add_argument(
        "--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"], help="Backend for distributed training."
    )
    args = parser.parse_args()
    config = args.model_config
    backend = args.backend
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    gettrace = getattr(sys, "gettrace", None)
    ch.setLevel(logging.DEBUG if gettrace and gettrace() else logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    root.info("Loading Config file")
    with open(args.model_config) as config_file:
        conf = yaml.safe_load(config_file)
    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    rank = world_rank
    save_loc = expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

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
        if isinstance(v, BridgeScalerTransformer):
            scaler_block_key = k
            break
    if scaler_block_key is None:
        raise ValueError("BridgeScalerTransformer not found in preblocks.")
    batches_per_epoch = trainer_conf.get("batches_per_epoch", 1)
    dl = cycle(train_loader)
    for i in range(batches_per_epoch):
        root.info(f"Worker {rank}: Processing batch {i} of {batches_per_epoch}.")
        batch = next(dl)
        processed_batch = apply_preblocks_before_scaler(preblocks, batch, device)
        preblocks[scaler_block_key].fit_scaler_batch(processed_batch)

    scaler_block = preblocks[scaler_block_key]
    # Gather the per-rank fitted scaler dicts onto rank 0 (or run single-process).
    if dist.is_initialized() and world_size > 1:
        barrier()
        all_scalers = [None] * world_size if rank == 0 else None
        gather_object(scaler_block.scaler, all_scalers, dst=0)
    else:
        all_scalers = [scaler_block.scaler]

    if rank == 0:
        root.info("Combining scalers.")
        combined_scaler = combine_scaler_dicts(all_scalers)
        save_scaler_dict(combined_scaler, scaler_block.scaler_path)
    return


if __name__ == "__main__":
    main()
