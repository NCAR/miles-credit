"""
rollout_to_netcdf_v2.py
------------------------
Ensemble-aware rollout-to-netCDF entry point for the Gen2 nested data schema.

This mirrors ``rollout_gen2.py``'s pipeline (preblocks -> model -> postblocks ->
ForecastWriter, driven by ``credit.trainers.rollout_utils.run_forecast``) and adds
one ensemble-specific knob on top: an ``inference.noise_scale`` override that scales
the learned noise amplitude of every ``StochasticDecompositionLayer`` (SDL) in the
model before rollout starts. ``noise_scale: 0.0`` collapses an SDL ensemble model to
its deterministic mean member; the trained noise amplitude is used unmodified when
``noise_scale`` is absent or ``1.0``. Useful for ad hoc spread tuning (e.g. `credit
check`/`--dry-run`-style experiments) without retraining or editing the checkpoint.

Everything else -- data loading, preblocks/postblocks, distributed setup, async
netCDF writing -- is identical to ``rollout_gen2.py``; see that script's docstring
for the general Gen2 rollout contract.

Usage:
    python -m credit.applications.rollout_to_netcdf_v2 -c config.yml --noise-scale 0.5

    torchrun --standalone --nproc-per-node=4 -m credit.applications.rollout_to_netcdf_v2 \\
        -c config.yml
"""

import logging
import multiprocessing as mp
import os
import sys
import warnings
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from credit.datasets.gen_2.multi_source import MultiSourceDataset
from credit.datasets.gen_2.channel_utils import ChannelSchema
from credit.datasets.gen_2._utils import to_calendar  # pyright: ignore[reportPrivateUsage]
from credit.distributed import get_rank_info, select_device, setup
from credit.output_gen2 import ForecastWriter
from credit.postblock import build_postblocks
from credit.preblock import attach_channel_schema, build_preblocks
from credit.seed import seed_everything
from credit.trainers.rollout_utils import (
    apply_inference_overrides,
    batch_init_times,
    load_model_for_inference,
    parse_length,
    run_forecast,
    with_inference_datetime_bounds,
)
from credit.trainers.utils import cleanup
from credit.samplers import MultiStepBatchSamplerSubset

logger = logging.getLogger("rollout_to_netcdf_v2")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def _apply_sdl_noise_scale(model: torch.nn.Module, noise_scale) -> None:
    """Scale every SDL noise_factor in-place. No-op when noise_scale is None or 1.0."""
    if noise_scale is None or noise_scale == 1.0:
        return
    from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer

    n_scaled = 0
    for m in model.modules():
        if isinstance(m, StochasticDecompositionLayer):
            m.noise_factor.data.mul_(noise_scale)
            n_scaled += 1
    if n_scaled:
        logger.info(f"noise_scale={noise_scale}: scaled {n_scaled} SDL noise layers")
    else:
        logger.warning(f"noise_scale={noise_scale} set but no StochasticDecompositionLayer found in model")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(
        description="CREDIT Gen2 ensemble rollout-to-netCDF (SDL noise_scale override).",
        epilog="""
Examples:
  # Batch hindcast at the trained noise amplitude (run_mode from config):
      python rollout_to_netcdf_v2.py -c config/example-end-to-end.yml

  # Collapse to the deterministic mean member:
      python rollout_to_netcdf_v2.py -c config.yml --noise-scale 0.0

  # Multi-GPU DDP:
      torchrun --standalone --nproc-per-node=4 rollout_to_netcdf_v2.py -c config.yml
        """,
    )
    parser.add_argument("-c", "--config", dest="model_config", required=True, help="Path to Gen2 YAML config.")
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
        "--noise-scale",
        type=float,
        default=None,
        help="Scale factor applied to every SDL noise_factor before rollout. "
        "0.0 collapses to the deterministic mean member; omit (or 1.0) to keep the "
        "trained noise amplitude. Overrides inference.noise_scale from config.",
    )
    parser.add_argument(
        "-p", "--procs", dest="num_cpus", type=int, default=4, help="CPU workers for async output pool."
    )
    parser.add_argument(
        "--log-all-ranks",
        action="store_true",
        default=False,
        help="Emit INFO logs from all workers, not just rank 0. Useful for debugging per-worker issues.",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    try:
        with open(args.model_config) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as exc:
        print(f"ERROR: failed to load config file '{args.model_config}': {exc}", file=sys.stderr)
        sys.exit(1)

    assert "source" in conf["data"], (
        "rollout_to_netcdf_v2.py requires the Gen2 nested data schema (conf['data']['source']). "
        "For Gen1/ensemble configs use trainer_ensemble_gen1's companion rollout scripts."
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
    noise_scale = args.noise_scale if args.noise_scale is not None else inf_conf.get("noise_scale")

    run_mode = inf_conf.get("run_mode", "batch")
    assert run_mode in ("batch", "single"), f"inference.run_mode must be 'batch' or 'single', got {run_mode!r}"

    save_dir = os.path.expandvars(inf_conf["save_forecast"])
    os.makedirs(save_dir, exist_ok=True)

    # ── Inference-scoped data/preblocks/postblocks overrides ────────────────────
    schema_conf = apply_inference_overrides(conf)

    # ── Init times ───────────────────────────────────────────────────────────
    timestep = conf["data"]["timestep"]
    calendar = conf["data"].get("calendar", "standard")
    if run_mode == "batch":
        assert "batch_forecast" in inf_conf, "inference.batch_forecast is required for run_mode=batch."
        all_init_times = batch_init_times(inf_conf["batch_forecast"], calendar=calendar)
        n_steps = parse_length(inf_conf["batch_forecast"]["forecast_length"], timestep)
    else:
        sf = inf_conf.get("single_forecast", {})
        assert "start_datetime" in sf, (
            "inference.single_forecast.start_datetime is required for run_mode=single (or pass --init-time on the CLI)."
        )
        all_init_times = [to_calendar(pd.Timestamp(sf["start_datetime"]), calendar)]
        n_steps = parse_length(
            sf.get("forecast_length", inf_conf.get("batch_forecast", {}).get("forecast_length", "10d")), timestep
        )

    # ── Distributed setup ────────────────────────────────────────────────────
    seed_everything(conf["seed"])
    mode = inf_conf.get("mode", "none")
    local_rank, world_rank, world_size = get_rank_info(mode)
    rank = world_rank

    # ── Logging ──────────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(ch)
    gettrace = getattr(sys, "gettrace", None)
    level = (
        (logging.DEBUG if gettrace and gettrace() else logging.INFO)
        if (rank == 0 or args.log_all_ranks)
        else logging.WARNING
    )
    for h in root.handlers:
        h.setLevel(level)

    device = select_device(local_rank)

    if mode in ("ddp", "fsdp"):
        setup(world_rank, world_size, mode, device_id=device if torch.cuda.is_available() else None)

    # ── Preblocks / postblocks ───────────────────────────────────────────────
    ic_preblocks = build_preblocks(conf, phase="ic_only")
    step_preblocks = build_preblocks(conf, phase="per_step")

    channel_schema = ChannelSchema.load_or_from_config(schema_conf)
    attach_channel_schema(ic_preblocks, channel_schema)
    attach_channel_schema(step_preblocks, channel_schema)

    step_postblocks = build_postblocks(conf, phase="per_step")
    rollout_postblocks = build_postblocks(conf, phase="post_rollout")

    # ── Model ────────────────────────────────────────────────────────────────
    model = load_model_for_inference(conf, device)
    model.eval()
    _apply_sdl_noise_scale(model, noise_scale)

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    dataset_conf = {
        **with_inference_datetime_bounds(conf["data"], all_init_times, n_steps, timestep),
        "forecast_len": n_steps,
        "datetimes": all_init_times,
        "save_loc": conf.get("save_loc"),
    }
    from credit.registry import load_custom_objects  # imported here to avoid a module-level credit.registry import

    load_custom_objects(conf)
    dataset = MultiSourceDataset(dataset_conf, return_target=False)
    calendar = getattr(dataset, "calendar", calendar)

    rank_indices = list(range(world_rank, len(all_init_times), world_size))
    sampler = MultiStepBatchSamplerSubset(
        dataset=dataset,
        batch_size=1,
        index_subset=rank_indices,
        num_forecast_steps=n_steps,
    )

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)

    logger.info(
        "Rank %d/%d: %d init time(s), %d steps each",
        world_rank,
        world_size,
        len(rank_indices),
        n_steps,
    )

    verbose = rank == 0 or args.log_all_ranks

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = ForecastWriter(
        output_conf=inf_conf.get("output", {}),
        conf=conf,
        n_steps=n_steps,
        dataset=dataset,
        ic_preblocks=ic_preblocks,
        step_preblocks=step_preblocks,
        verbose=verbose,
    )

    # ── Rollout ──────────────────────────────────────────────────────────────
    with mp.get_context("spawn").Pool(args.num_cpus) as pool:
        batch_iter = iter(loader)

        for _ in range(len(rank_indices)):
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
                verbose=verbose,
                calendar=calendar,
            )

        pool.close()
        pool.join()

    if mode in ("ddp", "fsdp"):
        dist.barrier()
        cleanup()


if __name__ == "__main__":
    main()
