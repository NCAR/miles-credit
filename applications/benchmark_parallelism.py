"""Parallelism benchmark for CREDIT v2 WXFormer.

Measures step time (ms), peak GPU memory (GB), and throughput (samples/sec)
using synthetic data. No real ERA5 data required.

Usage (torchrun):
    torchrun --standalone --nproc-per-node=4 applications/benchmark_parallelism.py \
        -c config/fsdp2_parallel_test.yml \
        [--data fsdp2] [--tensor 2] [--domain 1] \
        [--warmup 5] [--steps 20]

Output: one TSV row per rank-0 with:
    config_name  dp  tp  domain  world_size  step_ms  peak_mem_gb  samples_per_sec
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import yaml

from credit.distributed import distributed_model_wrapper_v2, setup
from credit.models import load_model
from credit.parallel.mesh import parse_parallelism_conf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--data", default=None, help="parallelism.data override (fsdp2|ddp|none)")
    p.add_argument("--tensor", type=int, default=None, help="TP degree override")
    p.add_argument("--domain", type=int, default=None, help="domain parallel degree override")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--name", default=None, help="label for output row")
    return p.parse_args()


def make_synthetic_input(conf, device):
    """Build a random input tensor matching the model's expected channels/size."""
    model_conf = conf.get("model", {})
    H = model_conf["image_height"]
    W = model_conf["image_width"]
    levels = model_conf.get("levels", 18)
    channels = model_conf.get("channels", 4)  # 3D vars per level
    surface_ch = model_conf.get("surface_channels", 4)
    input_only = model_conf.get("input_only_channels", 0)
    frames = model_conf.get("frames", 1)

    C = (channels * levels + surface_ch + input_only) * frames
    return torch.randn(1, C, H, W, device=device)


def main():
    args = parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    # Apply parallelism overrides
    if "parallelism" not in conf.get("trainer", {}):
        conf["trainer"]["parallelism"] = {}
    p = conf["trainer"]["parallelism"]
    if args.data is not None:
        p["data"] = args.data
    if args.tensor is not None:
        p["tensor"] = args.tensor
    if args.domain is not None:
        p["domain"] = args.domain
    p.setdefault("data", "fsdp2")
    p.setdefault("tensor", 1)
    p.setdefault("domain", 1)

    p_conf = parse_parallelism_conf(conf)
    _mode = p_conf["data"] if p_conf["data"] != "none" else conf["trainer"].get("mode", "none")
    _any_dist = _mode in ("fsdp2", "fsdp", "ddp") or p_conf.get("tensor", 1) > 1 or p_conf.get("domain", 1) > 1
    if _any_dist:
        setup(rank, world_size, _mode)

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Build model
    model = load_model(conf)
    model.to(device)
    model = distributed_model_wrapper_v2(conf, model, device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = make_synthetic_input(conf, device)
    # Build target: same shape as output (channels * levels + surface_channels output channels)
    model_conf = conf["model"]
    levels = model_conf.get("levels", 18)
    out_ch = (
        model_conf.get("channels", 4) * levels
        + model_conf.get("surface_channels", 4)
        + model_conf.get("output_only_channels", 0)
    )
    H, W = model_conf["image_height"], model_conf["image_width"]
    y = torch.randn(1, out_ch, H, W, device=device)

    # Handle domain sharding: shard x/y along H if domain > 1
    domain = p_conf.get("domain", 1)
    if domain > 1:
        h_per = H // domain
        x = x[..., rank % domain * h_per : (rank % domain + 1) * h_per, :]
        y = y[..., rank % domain * h_per : (rank % domain + 1) * h_per, :]

    criterion = torch.nn.MSELoss()

    def step():
        optimizer.zero_grad()
        pred = model(x)
        # Model may return [B, C, T, H, W]; collapse T if present
        if pred.dim() == 5:
            pred = pred.squeeze(2)
        loss = criterion(pred, y.to(pred.dtype))
        loss.backward()
        optimizer.step()

    # Warmup
    for _ in range(args.warmup):
        step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Timed steps
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    step_ms = elapsed / args.steps * 1000
    samples_per_sec = args.steps / elapsed * world_size  # global throughput

    if _any_dist:
        dist.barrier()

    if rank == 0:
        name = args.name or f"dp={p['data']}_tp={p['tensor']}_domain={p['domain']}"
        print(
            f"\n{'=' * 70}\n"
            f"  Config : {name}\n"
            f"  World  : {world_size} GPU(s)  |  dp={p['data']}  tp={p['tensor']}  domain={p['domain']}\n"
            f"  Step   : {step_ms:.1f} ms/step\n"
            f"  Memory : {peak_mem_gb:.2f} GB peak (rank 0)\n"
            f"  Thru   : {samples_per_sec:.2f} samples/sec\n"
            f"{'=' * 70}"
        )
        # TSV for easy aggregation
        print(
            f"TSV\t{name}\t{p['data']}\t{p['tensor']}\t{p['domain']}\t{world_size}\t{step_ms:.1f}\t{peak_mem_gb:.2f}\t{samples_per_sec:.2f}"
        )

    if _any_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
