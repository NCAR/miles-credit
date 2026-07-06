import argparse
import copy
import logging
import warnings

from bridgescaler import save_scaler_dict

from credit.distributed import get_rank_info, setup
from credit.seed import seed_everything
from os.path import expandvars
import os
import torch
import yaml
import sys
import shutil
from credit.datasets.schema import DEFAULT_SCHEMA_FILENAME, ChannelSchema
from credit.preblock import build_preblocks, apply_preblocks_before_scaler, BridgeScalerTransform
from credit.preblock.scaler import combine_scaler_dicts, move_scaler_dict_to_cpu
from credit.trainers.utils import cycle, load_dataset, load_dataloader, effective_mode
import torch.distributed as dist
from torch.distributed import gather_object, barrier


def _scaler_probe_range(scaler):
    """Return ``(lo, hi, label)`` normalized probe endpoints for a fitted scaler.

    The endpoints span the range of normalized values the scaler typically
    produces, so an inverse transform reveals what physical values each scaler
    maps that range to:

      - minmax    -> ``[0, 1]``
      - standard  -> ``[-4, 4]`` (±4 standard deviations)
      - quantile  -> ``[0, 1]`` for a uniform output distribution, otherwise
                     ``[-4, 4]`` (e.g. normal/logistic)
    """
    cls = type(scaler).__name__
    if "MinMax" in cls:
        return 0.0, 1.0, "minmax [0, 1]"
    if "Quantile" in cls:
        distribution = getattr(scaler, "distribution", "uniform")
        if distribution == "uniform":
            return 0.0, 1.0, f"quantile-{distribution} [0, 1]"
        return -4.0, 4.0, f"quantile-{distribution} [-4, 4]"
    return -4.0, 4.0, "standard [-4, 4]"


def _scaler_device(scaler):
    """Best-effort lookup of the torch device holding a scaler's fitted stats."""
    for attr in ("mean_x_", "var_x_", "min_x_", "max_x_", "min_tensor", "max_tensor"):
        t = getattr(scaler, attr, None)
        if torch.is_tensor(t):
            return t.device
        if isinstance(t, dict):
            for v in t.values():
                if torch.is_tensor(v):
                    return v.device
    return torch.device("cpu")


def _log_single_scaler(scaler, name, logger):
    """Log the fitted parameters of one leaf scaler, one line per channel/level."""
    columns = list(getattr(scaler, "x_columns_", []) or [])
    cls = type(scaler).__name__
    lo, hi, label = _scaler_probe_range(scaler)
    logger.info("Scaler '%s' [%s] — %d channel(s), probe range %s", name, cls, len(columns), label)

    mean = getattr(scaler, "mean_x_", None)
    var = getattr(scaler, "var_x_", None)
    vmin = getattr(scaler, "min_x_", None)
    vmax = getattr(scaler, "max_x_", None)

    # Inverse-transform the normalized probe endpoints to physical units. A
    # (2, n_channels) probe tensor works for both channels-first and
    # channels-last scalers (the channel dim is dim 1 either way).
    inv = None
    if columns:
        try:
            device = _scaler_device(scaler)
            probe = torch.tensor([[lo] * len(columns), [hi] * len(columns)], dtype=torch.float32, device=device)
            with torch.no_grad():
                inv = scaler.inverse_transform(probe).detach().cpu()
        except Exception as exc:  # noqa: BLE001 - logging-only, never fail preprocessing
            logger.warning("  could not inverse-transform probe for '%s': %s", name, exc)

    for i, col in enumerate(columns):
        parts = []
        if mean is not None and var is not None:
            parts.append(f"mean={float(mean[i]):.4g} var={float(var[i]):.4g}")
        if vmin is not None and vmax is not None:
            parts.append(f"min={float(vmin[i]):.4g} max={float(vmax[i]):.4g}")
        if inv is not None:
            parts.append(f"inv[{lo:g} -> {hi:g}]=[{float(inv[0, i]):.4g}, {float(inv[1, i]):.4g}]")
        logger.info("    %s: %s", col, "  ".join(parts) if parts else "(no stats available)")


def log_fitted_scalers(scaler_dict, logger, path=()):
    """Recursively log every fitted scaler in a nested ``scaler[data_type][source][var]`` dict."""
    for key, value in scaler_dict.items():
        if isinstance(value, dict):
            log_fitted_scalers(value, logger, path + (key,))
        else:
            _log_single_scaler(value, "/".join(path + (key,)), logger)


def main():
    # gcsfs/aiohttp leave their async SSL transports open until garbage collection,
    # which triggers a benign "unclosed transport" ResourceWarning from asyncio at
    # teardown (after the scalers have already been fitted and saved). Silence it.
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="bridgescaler")
    parser = argparse.ArgumentParser(
        epilog="""
Examples:
  # Recommended: submit via PBS using the credit CLI.
  # Note: torchrun is hardcoded internally; --device and --backend cannot be forwarded.
  # Device selection is automatic based on hardware detected at runtime (GPU if available, CPU otherwise).
  credit submit --cluster casper  -c config.yml --mode preprocess
  credit submit --cluster derecho -c config.yml --mode preprocess

  # For reference only (not the recommended path): run directly with torchrun.
  # GPU run (auto-selects nccl backend)
  torchrun --nproc_per_node=4 preprocess.py -c config.yml

  # CPU-only run (auto-selects gloo backend)
  torchrun --nproc_per_node=4 preprocess.py -c config.yml --device cpu

  # Force a specific device and backend
  torchrun --nproc_per_node=4 preprocess.py -c config.yml --device cuda:0 --backend nccl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", dest="model_config", required=True, type=str, help="Path to config file")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default=None,
        choices=["nccl", "gloo", "mpi"],
        help="Backend for distributed training. Defaults to 'nccl' for GPU, 'gloo' for CPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cpu', 'cuda', 'cuda:0'). Defaults to GPU if available.",
    )
    args = parser.parse_args()
    config = args.model_config
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
    local_rank, world_rank, world_size = get_rank_info(effective_mode(conf))
    rank = world_rank
    save_loc = expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    # Write the channel schema alongside the scaler. Always refreshed (unlike
    # model.yml) so a re-run after a config change never leaves a stale layout;
    # inference reads this file to reconstruct diagnostics, which never appear
    # in target-less batches. Training re-derives and re-validates it anyway.
    if rank == 0:
        try:
            ChannelSchema.from_config(conf).save(os.path.join(save_loc, DEFAULT_SCHEMA_FILENAME))
        except (KeyError, ValueError) as e:
            root.warning(
                "Could not derive channel schema from config (%s); %s not written.", e, DEFAULT_SCHEMA_FILENAME
            )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    backend = args.backend or ("nccl" if device.type == "cuda" else "gloo")
    if effective_mode(conf) in ["fsdp", "ddp", "domain_parallel", "fsdp+domain_parallel"]:
        setup(rank, world_size, effective_mode(conf), backend)

    trainer_conf = conf["trainer"]
    # Force forecast_len=1 so the dataloader only yields IC batches (step i=0).
    # Using forecast_len > 1 would feed the same timestamps at multiple step
    # offsets into the scaler fit, duplicating samples and skewing statistics.
    preprocess_conf = copy.deepcopy(conf)
    preprocess_conf["data"]["forecast_len"] = 1
    train_dataset = load_dataset(preprocess_conf, is_train=True)
    train_loader = load_dataloader(preprocess_conf, train_dataset, rank=rank, world_size=world_size, is_train=True)
    seed = conf.get("seed", 42) + rank
    seed_everything(seed)
    preblocks = build_preblocks(conf["preblocks"])
    print(preblocks)
    scaler_block_key = None
    for k, v in preblocks.items():
        if isinstance(v, BridgeScalerTransform):
            scaler_block_key = k
            break
    if scaler_block_key is None:
        raise ValueError("BridgeScalerTransform not found in preblocks.")
    _bpe = trainer_conf.get("batches_per_epoch", 0) or 0
    if hasattr(train_loader.sampler, "batches_per_epoch"):
        dataset_batches = train_loader.sampler.batches_per_epoch()
    elif hasattr(train_loader.dataset, "batches_per_epoch"):
        dataset_batches = train_loader.dataset.batches_per_epoch()
    else:
        dataset_batches = len(train_loader)
    batches_per_epoch = _bpe if 0 < _bpe < dataset_batches else dataset_batches
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
        gather_object(move_scaler_dict_to_cpu(scaler_block.scaler), all_scalers, dst=0)
    else:
        all_scalers = [scaler_block.scaler]

    if rank == 0:
        root.info("Combining scalers.")
        combined_scaler = combine_scaler_dicts(all_scalers)
        save_scaler_dict(combined_scaler, scaler_block.scaler_path)
        root.info("Saved fitted scaler to %s", scaler_block.scaler_path)
        root.info("Fitted scaler values by variable:")
        log_fitted_scalers(combined_scaler, root)
    return


if __name__ == "__main__":
    main()
