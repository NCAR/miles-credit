"""CLI parser definition and main entrypoint."""

import argparse
import sys
import textwrap

from ._ask import _ask
from ._common import _setup_logging
from ._convert import _convert, _init
from ._plot import _metrics, _plot
from ._submit import _realtime, _rollout, _rollout_ensemble, _submit, _train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit",
        description="CREDIT — AI-NWP model training and inference platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              credit init     --grid 0.25deg -o my_config.yml
              credit train    -c config.yml
              credit realtime -c config.yml --init-time 2024-01-15T00 --steps 40
              credit rollout  -c config.yml
              credit submit   --cluster casper  -c config.yml --gpus 1
              credit submit   --cluster casper  -c config.yml --mode rollout --jobs 10
              credit submit   --cluster casper  -c config.yml --mode realtime --init-time 2024-01-15T00
              credit submit   --cluster derecho -c config.yml --gpus 4 --nodes 2
        """),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- init ----
    p = sub.add_parser("init", help="Generate a starter config from a built-in template")
    p.add_argument(
        "--grid", choices=["0.25deg", "1deg"], default="0.25deg", help="Horizontal grid resolution (default: 0.25deg)"
    )
    p.add_argument(
        "--model",
        choices=["crossformer", "wxformer"],
        default="wxformer",
        help="Model architecture (default: wxformer)",
    )
    p.add_argument(
        "-o", "--output", default="config.yml", metavar="FILE", help="Output file path (default: config.yml)"
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")

    # ---- train ----
    p = sub.add_parser("train", help="Train a CREDIT v2 model")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to YAML training config")
    p.add_argument(
        "--backend", default="nccl", choices=["nccl", "gloo", "mpi"], help="Distributed backend (default: nccl)"
    )

    # ---- rollout ----
    p = sub.add_parser("rollout", help="Batch forecast rollout to NetCDF")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to YAML config")
    p.add_argument("-m", "--mode", default="none", help="Distributed mode: none | ddp | fsdp (default: none)")
    p.add_argument("-p", "--procs", type=int, default=4, help="CPU workers for async NetCDF save (default: 4)")
    p.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        metavar="N",
        dest="ensemble_size",
        help="Override predict.ensemble_size from config.",
    )

    # ---- rollout-ensemble ----
    p = sub.add_parser(
        "rollout-ensemble",
        help="Submit N parallel PBS rollout jobs covering all ensemble init times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Split an ensemble rollout across N parallel PBS jobs — one job per
            subset of init times.  All jobs start at once (no afterok chain).

            Examples:
              credit rollout-ensemble --cluster casper -c config.yml --jobs 10 --dry-run
              credit rollout-ensemble --cluster casper -c config.yml --jobs 10
              credit rollout-ensemble --cluster derecho -c config.yml --jobs 20 --gpus 1
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to v2 YAML config")
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"], help="Target NCAR HPC cluster")
    p.add_argument("--jobs", type=int, default=1, metavar="N", help="Number of parallel PBS jobs (default: 1)")
    p.add_argument("--gpus", type=int, default=None, metavar="N", help="GPUs per job")
    p.add_argument("--cpus", type=int, default=None, metavar="N", help="CPUs per job")
    p.add_argument("--mem", default=None, help="Memory per job")
    p.add_argument("--walltime", default=None, metavar="HH:MM:SS", help="Walltime per job")
    p.add_argument("--account", metavar="ACCOUNT", help="PBS account code")
    p.add_argument("--queue", metavar="QUEUE", help="PBS queue")
    p.add_argument("--gpu-type", dest="gpu_type", default=None, help="Casper GPU type (default: a100_80gb)")
    p.add_argument("--torchrun", default=None, metavar="PATH", help="Path to torchrun binary")
    p.add_argument("--conda-env", dest="conda_env", default=None, metavar="PATH", help="Conda env path")
    p.add_argument("--dry-run", action="store_true", help="Print PBS scripts without submitting")

    # ---- realtime ----
    p = sub.add_parser("realtime", help="Operational realtime forecast (single init time)")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to YAML config")
    p.add_argument("--init-time", required=True, metavar="YYYY-MM-DDTHH", help="Forecast initialisation time")
    p.add_argument("--steps", type=int, default=40, help="Number of autoregressive forecast steps (default: 40)")
    p.add_argument("--save-dir", metavar="DIR", help="Override output directory from config")
    p.add_argument("-m", "--mode", default="none", help="Distributed mode: none | ddp | fsdp (default: none)")
    p.add_argument("-p", "--procs", type=int, default=4, help="CPU workers for async NetCDF save (default: 4)")

    # ---- submit ----
    p = sub.add_parser(
        "submit",
        help="Generate and submit a PBS training, rollout, or realtime job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate a PBS batch script and optionally submit it via qsub.
            Use --dry-run to inspect the script before submitting.

            Modes:
              train     (default) Submit a training job.  Use --reload / --chain for
                        resuming and chaining multiple epochs across jobs.
              rollout   Submit N parallel PBS rollout jobs covering all init times.
                        Use --jobs N to set parallelism.  No afterok chain.
              realtime  Submit a single realtime forecast job.
                        Requires --init-time and --steps.

            Examples:
              credit submit --cluster casper  -c config.yml --gpus 1 --walltime 04:00:00
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 2 --dry-run
              credit submit --cluster casper  -c config.yml --mode train --reload
              credit submit --cluster derecho -c config.yml --mode train --chain 10
              credit submit --cluster casper  -c config.yml --mode rollout --jobs 10
              credit submit --cluster casper  -c config.yml --mode realtime --init-time 2024-01-15T00 --steps 40
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG")
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"], help="Target NCAR HPC cluster")
    p.add_argument(
        "--mode",
        dest="submit_mode",
        default="train",
        choices=["train", "rollout", "realtime"],
        help="Submission mode: train (default), rollout, or realtime",
    )
    p.add_argument("--gpus", type=int, default=None, metavar="N", help="GPUs per node")
    p.add_argument("--nodes", type=int, default=None, metavar="N", help="Number of nodes, derecho only")
    p.add_argument("--cpus", type=int, default=None, metavar="N", help="CPUs per node")
    p.add_argument("--mem", default=None, help="Memory per node")
    p.add_argument("--walltime", default=None, metavar="HH:MM:SS", help="Job walltime")
    p.add_argument("--account", metavar="ACCOUNT", help="PBS account code")
    p.add_argument("--queue", metavar="QUEUE", help="PBS queue")
    p.add_argument("--gpu-type", dest="gpu_type", default=None, help="Casper GPU type")
    p.add_argument("--torchrun", default=None, metavar="PATH", help="Path to torchrun binary")
    p.add_argument("--conda-env", dest="conda_env", default=None, metavar="PATH", help="Conda environment path")
    p.add_argument("--dry-run", action="store_true", help="Print the PBS script without submitting")
    p.add_argument(
        "--jobs", type=int, default=1, metavar="N", help="Parallel PBS rollout jobs for --mode rollout (default: 1)"
    )
    p.add_argument(
        "--init-time", dest="init_time", default=None, metavar="YYYY-MM-DDTHH", help="Init time for --mode realtime"
    )
    p.add_argument("--steps", type=int, default=40, metavar="N", help="Autoregressive steps for --mode realtime")
    p.add_argument(
        "--reload",
        action="store_true",
        help="(train mode) Resume from checkpoint: patch load_weights/optimizer/scaler/scheduler/reload_epoch",
    )
    p.add_argument(
        "--chain",
        type=int,
        default=None,
        metavar="N",
        help="(train mode) Submit N jobs in sequence using PBS afterok dependencies.",
    )

    # ---- plot ----
    p = sub.add_parser(
        "plot",
        help="Quick global map: truth vs prediction from a saved checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Load a checkpoint, run one forward pass on a validation sample, and
            produce a 3-panel global map (truth | prediction | difference).

            Examples:
              credit plot -c config.yml --field VAR_2T --denorm
              credit plot -c config.yml --field VAR_2T SP --level 5 --denorm
              credit plot -c config.yml --field SP --checkpoint /path/to/checkpoint.pt --denorm
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Training config YAML")
    p.add_argument("--field", nargs="+", required=True, metavar="VAR", help="Variable name(s) to plot")
    p.add_argument("--level", type=int, default=0, metavar="IDX", help="Level index for 3-D variables (default: 0)")
    p.add_argument(
        "--checkpoint", default=None, metavar="PATH", help="Checkpoint file (default: <save_loc>/checkpoint.pt)"
    )
    p.add_argument(
        "--sample-date", default=None, metavar="YYYY-MM-DDTHH", dest="sample_date", help="Validation sample init time"
    )
    p.add_argument("--output-dir", default=None, metavar="DIR", dest="output_dir", help="Where to save plots")
    p.add_argument("--denorm", action="store_true", help="Inverse-normalise output to physical units")

    # ---- ask ----
    p = sub.add_parser(
        "ask",
        help="Ask the CREDIT AI assistant a question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Ask the CREDIT AI assistant anything about training, config, or debugging.

            Provider priority (first key found wins):
              ANTHROPIC_API_KEY  → Claude (agent mode with file/bash tools)
              OPENAI_API_KEY     → GPT-4o
              GOOGLE_API_KEY     → Gemini 1.5 Pro  (free for NCAR)
              GROQ_API_KEY       → Llama 3 Instant (free tier)
              OPENROUTER_API_KEY → Qwen3-Next-80B  (free tier, thinking mode)

            Examples:
              credit ask "why is my training loss stuck at 2.5?"
              credit ask -c config.yml "why did my training run crash?"
        """),
    )
    p.add_argument("question", nargs="+", metavar="QUESTION", help="Your question")
    p.add_argument("-c", "--config", default=None, metavar="CONFIG", help="Optional config YAML for context")
    p.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai", "gemini", "groq", "openrouter"],
        help="Force a specific LLM provider",
    )
    p.add_argument("--max-turns", type=int, default=20, dest="max_turns", help="Max agentic turns (default: 20)")

    # ---- convert ----
    p = sub.add_parser(
        "convert",
        help="Interactively convert a v1 config to v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convert a v1 CREDIT config to v2 format.

            Automatic changes:
              - trainer.type: era5 → era5-gen2
              - data: flat V1 schema → nested V2 source schema
              - data.forecast_len: +1  (v2 semantics: 1 = single step, v1 used 0)
              - data.valid_forecast_len: +1

            Interactive prompts for new v2 features:
              - EMA (exponential moving average of weights)
              - TensorBoard logging
              - Ensemble settings (kept if detected)
              - PBS / job settings (account, conda env, nodes, walltime, ...)

            Example:
              credit convert -c old_model.yml          # saves to old_model_gen2.yml
              credit convert -c old_model.yml -o new.yml
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="v1 config YAML to convert")
    p.add_argument("-o", "--output", default=None, metavar="OUTPUT", help="Output path (default: <input>_gen2.yml)")
    p.add_argument("-y", "--defaults", action="store_true", help="Accept all defaults non-interactively")

    # ---- metrics ----
    p = sub.add_parser(
        "metrics",
        help="WeatherBench2-style evaluation: RMSE, ACC, and scorecard plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Run WeatherBench2-style verification on CREDIT forecast output.

            Input modes (one required):
              --csv     Directory of per-init CSV files (fast path, already scored)
              --netcdf  Directory of forecast netCDF files (full scoring pipeline)

            Examples:
              credit metrics --netcdf /path/to/forecasts --out scores.csv
              credit metrics --csv /path/to/csv_dir --plot figures/ --label WXFormer-v2
        """),
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", type=str, metavar="DIR", help="Directory of per-init metrics CSVs")
    input_group.add_argument("--netcdf", type=str, metavar="DIR", help="Directory of forecast netCDFs")
    p.add_argument("--era5", type=str, default=None, metavar="GLOB", help="Glob pattern for ERA5 zarr files")
    p.add_argument("--clim", type=str, default=None, metavar="FILE", help="ERA5 climatology netCDF for true ACC")
    p.add_argument("--out", type=str, default="wb2_scores.csv", metavar="FILE", help="Output scores CSV")
    p.add_argument("--lead-time-hours", type=int, default=6, dest="lead_time_hours", help="Hours per forecast step")
    p.add_argument(
        "--max-inits", type=int, default=None, dest="max_inits", metavar="N", help="Limit number of init dates"
    )
    p.add_argument(
        "--plot", type=str, default=None, metavar="DIR", dest="plot_dir", help="Generate WB2 scorecard figures here"
    )
    p.add_argument("--label", type=str, default="CREDIT", help="Model label for plot legends")
    p.add_argument("--no-refs", action="store_true", dest="no_refs", help="Omit reference model lines from plots")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers for --netcdf mode")
    p.add_argument("-v", "--verbose", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _setup_logging()

    dispatch = {
        "init": _init,
        "train": _train,
        "rollout": _rollout,
        "rollout-ensemble": _rollout_ensemble,
        "realtime": _realtime,
        "submit": _submit,
        "convert": _convert,
        "plot": _plot,
        "ask": _ask,
        "metrics": _metrics,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
