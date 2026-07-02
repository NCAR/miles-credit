"""
rollout_gen2.py
---------------
Combined batch + single-forecast rollout for CREDIT Gen2 models.

Mirrors the trainer_gen2.py inner loop exactly — the same full_data_dict super-dict
flows through preblocks → model → postblocks → assemble_rollout_batch at every step.
No manual denormalization, no flat-tensor surgery (update_x / build_channel_layout).

Config key:  inference.run_mode   (batch | single)
CLI override: --run-mode, --init-time, --save-dir

Usage
-----
# Batch hindcast (uses inference.batch_forecast from config):
    python rollout_gen2.py -c config/example-end-to-end.yml

# Single forecast (overrides inference.single_forecast.start_datetime):
    python rollout_gen2.py -c config/example-end-to-end.yml --init-time 2020-06-01T00

# Multi-GPU DDP:
    torchrun --standalone --nproc-per-node=4 rollout_gen2.py -c config.yml
"""

import logging
import multiprocessing as mp
import os
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from credit.datasets.multi_source import MultiSourceDataset
from credit.datasets.schema import ChannelSchema
from credit.distributed import get_rank_info, setup
from credit.output_gen2 import ForecastWriter
from credit.pbs import launch_script, launch_script_mpi
from credit.postblock import build_postblocks
from credit.preblock import attach_channel_schema, build_preblocks
from credit.samplers import DistributedMultiStepBatchSampler
from credit.seed import seed_everything
from credit.trainers.rollout_utils import (
    batch_init_times,
    load_model_for_inference,
    parse_length,
    run_forecast,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(
        description="CREDIT Gen2 combined batch + single-forecast rollout.",
        epilog="""
Examples:
  # Batch hindcast (run_mode from config):
      python rollout_gen2.py -c config/example-end-to-end.yml

  # Single forecast (overrides start_datetime):
      python rollout_gen2.py -c config.yml --run-mode single --init-time 2020-06-01T00

  # Multi-GPU DDP:
      torchrun --standalone --nproc-per-node=4 rollout_gen2.py -c config.yml
        """,
    )
    parser.add_argument("-c", "--config", dest="model_config", required=True, help="Path to Gen2 YAML config.")
    parser.add_argument("-l", dest="launch", type=int, default=0, help="Submit to PBS if 1.")
    parser.add_argument(
        "--run-mode",
        type=str,
        default=None,
        choices=["batch", "single"],
        help="Override inference.run_mode from config.",
    )
    parser.add_argument(
        "--init-time",
        type=str,
        default=None,
        help="Single-forecast init time (ISO 8601, e.g. 2020-06-01T00). "
        "Overrides inference.single_forecast.start_datetime.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Output directory. Overrides inference.save_forecast."
    )
    parser.add_argument(
        "-p", "--procs", dest="num_cpus", type=int, default=4, help="CPU workers for async output pool."
    )
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(ch)

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.model_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    assert "source" in conf["data"], (
        "rollout_gen2.py requires the Gen2 nested data schema (conf['data']['source']). "
        "For Gen1 configs use the legacy rollout scripts."
    )
    assert "inference" in conf, "Config is missing an 'inference:' section. Use example-end-to-end.yml as a template."

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    # ── CLI overrides ─────────────────────────────────────────────────────────
    inf_conf = conf["inference"]
    if args.run_mode is not None:
        inf_conf["run_mode"] = args.run_mode
    if args.save_dir is not None:
        inf_conf["save_forecast"] = args.save_dir
    if args.init_time is not None:
        inf_conf.setdefault("single_forecast", {})["start_datetime"] = args.init_time
        inf_conf["run_mode"] = "single"  # --init-time implies single mode

    run_mode = inf_conf.get("run_mode", "batch")
    assert run_mode in ("batch", "single"), f"inference.run_mode must be 'batch' or 'single', got {run_mode!r}"

    save_dir = os.path.expandvars(inf_conf["save_forecast"])
    os.makedirs(save_dir, exist_ok=True)

    # ── PBS launch ───────────────────────────────────────────────────────────
    if args.launch:
        script_path = Path(__file__).absolute()
        if conf.get("pbs", {}).get("queue") == "casper":
            launch_script(args.model_config, str(script_path))
        else:
            launch_script_mpi(args.model_config, str(script_path))
        sys.exit()

    # ── Init times ───────────────────────────────────────────────────────────
    timestep = conf["data"]["timestep"]
    if run_mode == "batch":
        assert "batch_forecast" in inf_conf, "inference.batch_forecast is required for run_mode=batch."
        all_init_times = batch_init_times(inf_conf["batch_forecast"])
        n_steps = parse_length(inf_conf["batch_forecast"]["forecast_length"], timestep)
    else:
        sf = inf_conf.get("single_forecast", {})
        assert "start_datetime" in sf, (
            "inference.single_forecast.start_datetime is required for run_mode=single (or pass --init-time on the CLI)."
        )
        all_init_times = [pd.Timestamp(sf["start_datetime"])]
        n_steps = parse_length(
            sf.get("forecast_length", inf_conf.get("batch_forecast", {}).get("forecast_length", "10d")), timestep
        )

    # ── Distributed setup ────────────────────────────────────────────────────
    seed_everything(conf["seed"])
    mode = inf_conf.get("mode", "none")
    local_rank, world_rank, world_size = get_rank_info(mode)

    if mode in ("ddp", "fsdp"):
        setup(world_rank, world_size, mode)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Preblocks / postblocks ───────────────────────────────────────────────
    preblock_cfg = conf.get("preblocks", {})
    ic_preblocks = build_preblocks(preblock_cfg, phase="ic_only")
    step_preblocks = build_preblocks(preblock_cfg, phase="per_step")

    # Channel schema: inference batches carry no target (and diagnostics exist
    # only in targets), so without a schema the reconstruction map would cover
    # prognostics only and every diagnostic would be silently dropped from the
    # output. Prefer the schema saved at training time in save_loc.
    channel_schema = ChannelSchema.load_or_from_config(conf)
    attach_channel_schema(ic_preblocks, channel_schema)
    attach_channel_schema(step_preblocks, channel_schema)

    postblock_cfg = conf.get("postblocks", {})
    step_postblocks = build_postblocks(postblock_cfg, phase="per_step")
    rollout_postblocks = build_postblocks(postblock_cfg, phase="post_rollout")

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model_for_inference(conf, device)
    model.eval()

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    # Inject desired init times into dataset_conf so _build_master_clock uses
    # exactly these timestamps (short-circuits the full date-range scan).
    # "datetimes" is an internal key — it never appears in the user's YAML.
    # The sampler distributes init times across ranks and sequences steps so
    # that for each init time the loader yields: IC batch (step=0), then
    # (n_steps-1) dynamic-forcing-only batches (step>0).
    dataset_conf = {
        **conf["data"],
        "forecast_len": n_steps,
        "datetimes": all_init_times,
    }
    dataset = MultiSourceDataset(dataset_conf, return_target=False)

    sampler = DistributedMultiStepBatchSampler(
        dataset=dataset,
        batch_size=1,
        num_forecast_steps=n_steps,  # IC + (n_steps-1) forcing batches = n_steps total
        num_replicas=world_size,
        rank=world_rank,
        shuffle=False,
        seed=conf.get("seed", 0),
    )

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)

    logger.info(
        "Rank %d/%d: %d init time(s), %d steps each",
        world_rank,
        world_size,
        sampler.num_samples,
        n_steps,
    )

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = ForecastWriter(
        output_conf=inf_conf.get("output", {}),
        conf=conf,
        n_steps=n_steps,
    )

    # ── Rollout ──────────────────────────────────────────────────────────────
    # batch_iter is shared across all forecasts. The sampler groups batches so
    # that each forecast consumes exactly n_steps consecutive batches (1 IC +
    # n_steps-1 forcing), in init-time order.
    with mp.Pool(args.num_cpus) as pool:
        batch_iter = iter(loader)

        for _ in range(sampler.num_samples):
            run_forecast(
                conf=conf,
                n_steps=n_steps,
                save_dir=save_dir,
                ic_preblocks=ic_preblocks,
                step_preblocks=step_preblocks,
                step_postblocks=step_postblocks,
                rollout_postblocks=rollout_postblocks,
                model=model,
                batch_iter=batch_iter,
                device=device,
                pool=pool,
                save_output_fn=writer,
            )

            if mode in ("ddp", "fsdp"):
                torch.distributed.barrier()

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
