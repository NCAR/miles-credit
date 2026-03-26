"""CREDIT unified command-line interface.

Single entrypoint for training, rollout, job submission, and config generation.

Examples
--------
  credit train -c config.yml
  credit realtime -c config.yml --init-time 2024-01-15T00 --steps 40
  credit rollout -c config.yml
  credit submit --cluster derecho -c config.yml --gpus 4 --nodes 2
  credit submit --cluster casper  -c config.yml --gpus 1 --dry-run
  credit init --grid 0.25deg -o my_config.yml
"""

import argparse
import logging
import os
import sys
import textwrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)


def _repo_root() -> str:
    """Absolute path to the miles-credit repo root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _train(args: argparse.Namespace) -> None:
    from credit.applications.train_v2 import main_cli

    sys.argv = ["credit-train", "-c", args.config, "--backend", args.backend]
    main_cli()


def _rollout(args: argparse.Namespace) -> None:
    from credit.applications.rollout_to_netcdf_v2 import main

    sys.argv = [
        "credit-rollout",
        "-c",
        args.config,
        "-m",
        args.mode,
        "-cpus",
        str(args.procs),
    ]
    main()


def _realtime(args: argparse.Namespace) -> None:
    from credit.applications.rollout_realtime_v2 import main

    argv = [
        "credit-realtime",
        "-c",
        args.config,
        "--init-time",
        args.init_time,
        "--steps",
        str(args.steps),
        "-m",
        args.mode,
        "-p",
        str(args.procs),
    ]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    sys.argv = argv
    main()


def _write_reload_config(config_path: str) -> str:
    """Patch trainer reload fields and write a reload config next to the checkpoint.

    Reads the YAML at *config_path*, sets the five fields required for a clean
    resume, and writes the result to ``<save_loc>/config_reload.yml``.

    Returns the path to the written reload config.
    """
    import yaml

    with open(config_path) as f:
        conf = yaml.safe_load(f)

    trainer = conf.setdefault("trainer", {})
    trainer["load_weights"] = True
    trainer["load_optimizer"] = True
    trainer["load_scaler"] = True
    trainer["load_scheduler"] = True
    trainer["reload_epoch"] = True

    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    os.makedirs(save_loc, exist_ok=True)
    reload_path = os.path.join(save_loc, "config_reload.yml")

    with open(reload_path, "w") as f:
        yaml.dump(conf, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return reload_path


def _build_pbs_script(args: argparse.Namespace, config: str, repo: str, account: str, depend_on: str = None) -> str:
    """Return a PBS batch script string for the given args and config path.

    Args:
        depend_on: If set, adds ``#PBS -W depend=afterok:<depend_on>`` so this
                   job only starts after the given job ID completes successfully.
    """
    depend_line = f"#PBS -W depend=afterok:{depend_on}\n" if depend_on else ""

    if args.cluster == "casper":
        cpus = args.cpus or 8
        mem = args.mem or "128GB"
        gpu_type = args.gpu_type or "a100_80gb"
        queue = args.queue or "casper"
        torchrun = args.torchrun or _find_torchrun()

        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N credit_v2
            #PBS -l select=1:ncpus={cpus}:ngpus={args.gpus}:mem={mem}:gpu_type={gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {account}
            #PBS -q {queue}
            #PBS -j oe
            #PBS -k eod
            {depend_line}
            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}

            echo "Config : ${{CONFIG}}"
            echo "Node   : $(hostname)"
            echo "GPUs   : ${{NGPUS}}"

            export PYTHONPATH="${{REPO}}:${{PYTHONPATH}}"
            export PYTHONNOUSERSITE=1
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/applications/train_v2.py -c ${{CONFIG}}
        """)

    else:  # derecho
        nodes = args.nodes
        cpus = args.cpus or 64
        mem = args.mem or "480GB"
        queue = args.queue or "main"
        conda_env = args.conda_env or "/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221"

        header = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {account}
            #PBS -N credit_v2
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={cpus}:ngpus={args.gpus}:mem={mem}
            #PBS -q {queue}
            #PBS -j oe
            #PBS -k eod
            #PBS -r n
            {depend_line}
            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

            conda activate {conda_env}

            REPO={repo}
            CONFIG={config}

            export PYTHONPATH="${{REPO}}:${{PYTHONPATH}}"
            export LOGLEVEL=INFO
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            export NCCL_SOCKET_IFNAME=hsn
            export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
            export MPICH_OFI_NIC_POLICY=GPU
            export MPICH_GPU_SUPPORT_ENABLED=1
            export NCCL_IB_DISABLE=1
            export NCCL_CROSS_NIC=1
            export NCCL_NCHANNELS_PER_NET_PEER=4
            export MPICH_RDMA_ENABLED_CUDA=1
            export NCCL_NET="AWS Libfabric"
            export NCCL_NET_GDR_LEVEL=PBH
            export FI_CXI_DISABLE_HOST_REGISTER=1
            export FI_CXI_OPTIMIZED_MRS=false
            export FI_MR_CACHE_MONITOR=userfaultfd
            export FI_CXI_DEFAULT_CQ_SIZE=131072

            total_gpus=$(( {nodes} * {args.gpus} ))
            echo "Nodes     : {nodes}"
            echo "GPUs/node : {args.gpus}"
            echo "Total GPUs: ${{total_gpus}}"
            echo "Config    : ${{CONFIG}}"
            cd ${{REPO}}
        """)

        if nodes == 1:
            # Single-node: torchrun --standalone, no MPI or rendezvous endpoint needed
            launch = textwrap.dedent(f"""\
                torchrun \\
                    --standalone \\
                    --nnodes=1 \\
                    --nproc-per-node={args.gpus} \\
                    ${{REPO}}/applications/train_v2.py -c ${{CONFIG}}
            """)
        else:
            # Multi-node: MPI + c10d rendezvous via head-node IP
            launch = textwrap.dedent(f"""\
                nodes_arr=( $( cat $PBS_NODEFILE ) )
                head_node="${{nodes_arr[0]}}"
                head_node_ip=$(ssh "${{head_node}}" hostname -i | awk '{{print $1}}')
                echo "Head node : ${{head_node_ip}}"
                RDZV_PORT=$(( RANDOM % 10000 + 20000 ))

                mpiexec -n "${{total_gpus}}" --ppn {args.gpus} --cpu-bind none \\
                    torchrun \\
                        --nnodes={nodes} \\
                        --nproc-per-node={args.gpus} \\
                        --rdzv-backend=c10d \\
                        --rdzv-endpoint="${{head_node_ip}}:${{RDZV_PORT}}" \\
                    ${{REPO}}/applications/train_v2.py -c ${{CONFIG}}
            """)

        return header + launch


def _qsub(script: str) -> str:
    """Write *script* to a temp file, call qsub, and return the job ID string."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(["qsub", script_path], capture_output=True, text=True)
    finally:
        os.unlink(script_path)

    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout.strip()


def _compute_chain(args: argparse.Namespace) -> int:
    """Return the number of jobs to chain.

    If --chain was explicitly passed use it as-is.  Otherwise read
    ``trainer.epochs`` and ``trainer.num_epoch`` from the config and compute
    ``ceil(epochs / num_epoch)``, falling back to 1 if the keys are absent.
    """
    if args.chain is not None:
        return args.chain
    try:
        import math
        import yaml

        with open(args.config) as f:
            conf = yaml.safe_load(f)
        trainer = conf.get("trainer", {})
        epochs = int(trainer["epochs"])
        num_epoch = int(trainer["num_epoch"])
        return math.ceil(epochs / num_epoch)
    except Exception:
        return 1


def _print_job_plan(args: argparse.Namespace, n_jobs: int) -> None:
    """Print a human-readable summary of what is about to be submitted."""
    import yaml
    from credit.trainers.preflight import estimate_dataloader_memory_gb

    try:
        with open(args.config) as f:
            conf = yaml.safe_load(f)
    except Exception:
        conf = {}

    trainer = conf.get("trainer", {})
    epochs = trainer.get("epochs", "?")
    per_job = trainer.get("num_epoch", "?")
    walltime = args.walltime

    gpu_str = f"{args.gpus} GPU(s)"
    if args.cluster == "derecho" and getattr(args, "nodes", 1) > 1:
        gpu_str = f"{args.gpus} GPU(s) × {args.nodes} nodes ({args.gpus * args.nodes} total)"

    mem_est = estimate_dataloader_memory_gb(conf)
    mem_str = f"~{mem_est:.0f} GB" if mem_est > 0 else "unknown"
    mem_warn = "  ⚠  consider reducing thread_workers / prefetch_factor" if mem_est > 24 else ""

    chain_desc = f"{n_jobs} job(s)  ({epochs} epochs ÷ {per_job} per job)" if n_jobs > 1 else "1 job (no chaining)"

    print()
    print("=" * 52)
    print("  Job plan")
    print("=" * 52)
    print(f"  Cluster  : {args.cluster}")
    print(f"  Config   : {args.config}")
    print(f"  GPUs     : {gpu_str}")
    print(f"  Walltime : {walltime} per job")
    print(f"  Chain    : {chain_desc}")
    print(f"  DataLoader memory est. : {mem_str}{mem_warn}")
    print("=" * 52)
    print()


def _submit(args: argparse.Namespace) -> None:
    """Generate and optionally submit PBS batch scripts, with optional chaining."""
    repo = _repo_root()
    account = args.account or os.environ.get("PBS_ACCOUNT", "NAML0001")
    n_jobs = _compute_chain(args)

    _print_job_plan(args, n_jobs)

    # First job: fresh or reload depending on --reload flag
    first_config = os.path.abspath(args.config)
    if args.reload:
        first_config = _write_reload_config(first_config)
        logger.info(f"Reload config written: {first_config}")

    # Reload config reused for all subsequent chained jobs (written once)
    reload_config = _write_reload_config(os.path.abspath(args.config)) if n_jobs > 1 else None

    if args.dry_run:
        script = _build_pbs_script(args, first_config, repo, account, depend_on=None)
        print(f"# --- Job 1/{n_jobs} ---")
        print(script)
        if n_jobs > 1:
            script2 = _build_pbs_script(args, reload_config, repo, account, depend_on="<job_1_id>")
            print(f"# --- Jobs 2..{n_jobs}/{n_jobs} (afterok chained, reload config) ---")
            print(script2)
        return

    # Submit job 1
    script = _build_pbs_script(args, first_config, repo, account, depend_on=None)
    job_id = _qsub(script)
    print(f"[1/{n_jobs}] {job_id}  {first_config}")

    # Submit remaining chained reload jobs
    for i in range(2, n_jobs + 1):
        script = _build_pbs_script(args, reload_config, repo, account, depend_on=job_id)
        job_id = _qsub(script)
        print(f"[{i}/{n_jobs}] {job_id}  afterok  (reload)")


def _find_torchrun() -> str:
    """Return the path to torchrun, preferring the active conda env."""
    import shutil

    # Check if torchrun is on PATH (active conda env)
    tr = shutil.which("torchrun")
    if tr:
        return tr
    # Fallback: common NCAR casper path
    home = os.path.expanduser("~")
    fallback = os.path.join(home, ".conda", "envs", "credit-casper", "bin", "torchrun")
    if os.path.isfile(fallback):
        return fallback
    return "torchrun"  # hope it's on PATH at job time


def _init(args: argparse.Namespace) -> None:
    """Copy a config template to the user's desired location."""
    import shutil

    templates = {
        ("0.25deg", "wxformer_v2"): "config/wxformer_025deg_6hr_v2.yml",
        ("0.25deg", "crossformer"): "config/wxformer_025deg_6hr_v2.yml",
        ("1deg", "wxformer_v2"): "config/wxformer_1dg_6hr_v2.yml",
        ("1deg", "crossformer"): "config/wxformer_1dg_6hr_v2.yml",
    }

    repo = _repo_root()
    key = (args.grid, args.model)
    template_rel = templates.get(key)

    if template_rel is None:
        print(f"No template available for grid={args.grid}, model={args.model}", file=sys.stderr)
        sys.exit(1)

    template = os.path.join(repo, template_rel)
    if not os.path.exists(template):
        print(f"Template not found: {template}", file=sys.stderr)
        sys.exit(1)

    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.force:
        print(f"File already exists: {output}  (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    shutil.copy(template, output)
    print(f"Created  : {output}")
    print(f"Template : {template_rel}")
    print()
    print("Next steps:")
    print("  1. Check 'save_loc' — defaults to /glade/derecho/scratch/$USER/CREDIT_runs/...")
    print("     (NCAR users: no edits needed; others: update to a writable path)")
    print("  2. Verify data paths under 'data.source'")
    print("     (NCAR users: paths point to /glade/campaign/cisl/aiml/ksha/CREDIT_data/ — readable by all staff)")
    print(f"  3. credit submit --cluster casper -c {output} --gpus 4 --chain 14")


# ---------------------------------------------------------------------------
# credit plot
# ---------------------------------------------------------------------------


def _build_channel_map(conf):
    """Return a dict mapping variable name -> list of channel indices in the output tensor.

    Channel order mirrors ConcatPreblock TARGET_FIELD_ORDER:
        prognostic/3D (each var × n_levels), prognostic/2D, diagnostic/2D
    """
    src = conf["data"]["source"]["ERA5"]
    n_levels = len(src.get("levels", []))
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    channel_map = {}
    idx = 0
    for vn in prog.get("vars_3D", []):
        channel_map[vn] = list(range(idx, idx + n_levels))
        idx += n_levels
    for vn in prog.get("vars_2D", []):
        channel_map[vn] = [idx]
        idx += 1
    for vn in diag.get("vars_2D", []):
        channel_map[vn] = [idx]
        idx += 1
    return channel_map


def _build_denorm_stats(conf):
    """Return (mean_arr, std_arr) aligned with ConcatPreblock TARGET_FIELD_ORDER output channels.

    Channel order: prognostic/3D (each var × n_levels), prognostic/2D, diagnostic/2D.
    Variables missing from the stat files get mean=0, std=1 (pass-through).

    Returns:
        mean_arr: np.ndarray of shape (C_out,)
        std_arr:  np.ndarray of shape (C_out,)
    """
    import numpy as np
    import xarray as xr

    src = conf["data"]["source"]["ERA5"]
    levels = src["levels"]
    level_coord = src["level_coord"]
    n_levels = len(levels)
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    mean_ds = xr.open_dataset(conf["data"]["mean_path"]).load()
    std_ds = xr.open_dataset(conf["data"]["std_path"]).load()

    def _stats(varname, is_3d):
        if varname not in mean_ds or varname not in std_ds:
            n = n_levels if is_3d else 1
            return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
        if is_3d:
            m = mean_ds[varname].sel({level_coord: levels}).values.astype(np.float32)
            s = std_ds[varname].sel({level_coord: levels}).values.astype(np.float32)
        else:
            m = np.array([float(mean_ds[varname].values)], dtype=np.float32)
            s = np.array([float(std_ds[varname].values)], dtype=np.float32)
        return m, s

    means, stds = [], []
    for vn in prog.get("vars_3D", []):
        m, s = _stats(vn, True)
        means.append(m)
        stds.append(s)
    for vn in prog.get("vars_2D", []):
        m, s = _stats(vn, False)
        means.append(m)
        stds.append(s)
    for vn in diag.get("vars_2D", []):
        m, s = _stats(vn, False)
        means.append(m)
        stds.append(s)

    return np.concatenate(means), np.concatenate(stds)


_CREDIT_SYSTEM_PROMPT = """\
You are an expert assistant for CREDIT (Community Research Earth-system Digital Intelligence Tool),
an AI-based numerical weather prediction framework developed by the NCAR MILES group.

## What CREDIT is
CREDIT trains deep learning models (primarily WXFormer) to forecast global atmospheric state.
It runs on NCAR HPC clusters: Casper (single-node, A100/H100 GPUs) and Derecho (multi-node, A100 GPUs).
The main entry point is the `credit` CLI.

## Key CLI commands
- `credit train -c config.yml`                    — start/resume training
- `credit submit --cluster casper|derecho -c config.yml --gpus N [--nodes N] [--chain N] [--reload]`
- `credit plot -c config.yml --field VAR_2T --denorm`   — quick visualisation from checkpoint
- `credit rollout -c config.yml`                  — batch forecast to NetCDF
- `credit realtime -c config.yml --init-time YYYY-MM-DDTHH --steps N`
- `credit init --grid 1deg|0.25deg -o my_config.yml`    — generate starter config
- `credit ask "..."`                              — this command

## v2 data schema (YAML)
```yaml
data:
  source:
    ERA5:
      levels: [...]          # pressure/model levels
      variables:
        prognostic:
          vars_3D: [U, V, T, Q, Z]    # each × n_levels channels
          vars_2D: [SP, VAR_2T, ...]
        diagnostic:
          vars_2D: [precip, evap, ...]
  mean_path: /path/to/mean.nc
  std_path:  /path/to/std.nc
```

## Trainer config
```yaml
trainer:
  type: era5-v2
  mode: ddp           # none | ddp | fsdp
  train_batch_size: 8
  num_epoch: 5        # epochs per PBS job
  epochs: 70          # total training target
  thread_workers: 4   # DataLoader workers per GPU
  prefetch_factor: 4
  use_tensorboard: True
  use_ema: True
  ema_decay: 0.9999
  use_scheduler: True
  scheduler:
    scheduler_type: linear-warmup-cosine
    warmup_steps: 1000
    total_steps: 500000
    min_lr: 1.0e-5
  dataloader_timeout_s: 300   # preflight hang detection
```

## Cluster specifics
- **Casper**: 1 node, torchrun --standalone, GPUs: V100/A100/H100, queue: casper
  - Pre-built env: `/glade/u/home/schreck/.conda/envs/credit-casper`
- **Derecho single-node**: torchrun --standalone (NOT mpiexec)
- **Derecho multi-node**: mpiexec + torchrun --rdzv-backend=c10d
  - Pre-built env: `/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221`
- Data root: `/glade/campaign/cisl/aiml/ksha/CREDIT_data/`
- Default save_loc: `/glade/derecho/scratch/$USER/CREDIT_runs/`

## Common problems and fixes
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Training loop hangs on startup | DataLoader OOM (too many workers × prefetch × batch × channels) | Reduce `thread_workers` to 1 or 0, or `prefetch_factor` to 1 |
| `RendezvousConnectionError` on Derecho | Single-node job using c10d rendezvous | Use `--nodes 1` so `credit submit` generates `--standalone` |
| Loss > 100 or growing | Bad normalization or wrong data paths | Check `mean_path`/`std_path`; run `credit plot --denorm` |
| Loss stuck (not decreasing) | LR too low/high, wrong scheduler, EMA misconfigured | Check scheduler config; try reducing LR 10×; check warmup_steps |
| `KeyError: 'linear-warmup-cosine'` | Old CREDIT version | `pip install -e . --no-deps` to reinstall |
| Checkpoint not found | Wrong `save_loc` or first epoch | Set `load_weights: False` for first run |
| PBS job cancelled after failure | Normal: `afterok` chain auto-cancels remaining jobs | Use `credit submit --reload --chain N` to restart |
| FSDP + EMA slow | EMA does extra full-param sync on FSDP | Use `use_ema: False` with FSDP or accept overhead |

## How --chain works
`--chain N` submits N PBS jobs with afterok dependencies. Job 1 runs fresh (or --reload).
Jobs 2..N auto-generate `config_reload.yml` and resume from checkpoint.
Rule of thumb: chain = ceil(total_epochs / num_epoch). E.g., 70 epochs / 5 per job = 14.

## What healthy training looks like
- After epoch 1: train_loss ≈ 1–3 (order 1)
- Loss should decrease steadily each epoch
- Validation loss should track training loss (not diverge)
- `credit plot -c config.yml --field VAR_2T --denorm` should show recognisable weather patterns after ~10 epochs

Be concise, specific, and actionable. When referencing config keys use inline code. If you see a training log or config in the context, use it to give run-specific advice.
"""


def _collect_run_context(args: argparse.Namespace) -> str:
    """Gather config, training log, and recent PBS output for context injection."""
    import glob as _glob

    parts = []

    # ---- Config ----
    if getattr(args, "config", None):
        try:
            with open(args.config) as f:
                raw = f.read()
            parts.append(f"## Active config ({args.config})\n```yaml\n{raw}\n```")
        except OSError:
            pass

        # ---- Training log ----
        try:
            import yaml
            import pandas as pd

            with open(args.config) as f:
                conf = yaml.safe_load(f)
            save_loc = conf.get("save_loc", "")
            log_path = os.path.join(save_loc, "training_log.csv")
            if os.path.exists(log_path):
                df = pd.read_csv(log_path)
                tail = df.tail(20).to_string(index=False)
                parts.append(f"## training_log.csv (last {min(20, len(df))} rows)\n```\n{tail}\n```")
        except Exception:
            pass

        # ---- Most recent PBS output file ----
        try:
            pbs_files = _glob.glob("credit_v2.o*") + _glob.glob("credit.o*")
            if pbs_files:
                newest = max(pbs_files, key=os.path.getmtime)
                with open(newest) as f:
                    lines = f.readlines()
                tail_lines = "".join(lines[-60:])
                parts.append(f"## Most recent PBS output ({newest}, last 60 lines)\n```\n{tail_lines}\n```")
        except Exception:
            pass

    return "\n\n".join(parts)


class _ProviderError(Exception):
    """Raised when a provider call fails in a way that should trigger fallback."""


def _ask_anthropic(user_msg: str) -> None:
    """Stream a response via the Anthropic API (claude-haiku)."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_CREDIT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
    except anthropic.BadRequestError as e:
        if "credit balance is too low" in str(e):
            raise _ProviderError(
                "Anthropic API key is set but the account has no credits.\n"
                "Add credits at: https://console.anthropic.com/settings/billing\n"
                "Note: a Claude.ai Pro subscription does NOT include API access."
            ) from e
        raise


def _ask_groq(user_msg: str) -> None:
    """Stream a response via the Groq API (llama3 — free tier)."""
    import groq

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _CREDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def _ask_openai(user_msg: str) -> None:
    """Stream a response via the OpenAI API (gpt-4o)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stream = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _CREDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


def _ask_gemini(user_msg: str) -> None:
    """Stream a response via the Google Gemini API (gemini-1.5-pro)."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=_CREDIT_SYSTEM_PROMPT,
    )
    for chunk in model.generate_content(user_msg, stream=True):
        print(chunk.text, end="", flush=True)


_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic", "Claude Haiku"),
    "openai": ("OPENAI_API_KEY", "openai", "GPT-4o"),
    "gemini": ("GOOGLE_API_KEY", "google.generativeai", "Gemini 1.5 Pro"),
    "groq": ("GROQ_API_KEY", "groq", "Llama 3 Instant (free)"),
}
_PROVIDER_INSTALL = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google-generativeai",
    "groq": "groq",
}
_PROVIDER_RUNNERS = {
    "anthropic": _ask_anthropic,
    "openai": _ask_openai,
    "gemini": _ask_gemini,
    "groq": _ask_groq,
}


def _ask(args: argparse.Namespace) -> None:
    """Ask the CREDIT AI assistant a question about your run.

    Provider priority when --provider is not set (first key found wins):
      1. ANTHROPIC_API_KEY  → Claude Haiku        (pip install anthropic)
      2. OPENAI_API_KEY     → GPT-4o              (pip install openai)
      3. GOOGLE_API_KEY     → Gemini 1.5 Pro      (pip install google-generativeai)
      4. GROQ_API_KEY       → Llama 3 Instant     (pip install groq — free tier)
    """
    # ---- Resolve provider ----
    explicit = getattr(args, "provider", None)
    if explicit:
        if explicit not in _PROVIDERS:
            print(f"Unknown provider {explicit!r}. Choose: {', '.join(_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)
        env_key, pkg, label = _PROVIDERS[explicit]
        if not os.environ.get(env_key):
            print(f"{env_key} is not set.", file=sys.stderr)
            sys.exit(1)
        provider = explicit
    else:
        provider = None
        for name, (env_key, pkg, label) in _PROVIDERS.items():
            if os.environ.get(env_key):
                provider = name
                break
        if provider is None:
            print(
                "No API key found.  Set one of:\n\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com\n"
                "  export OPENAI_API_KEY=sk-...           # https://platform.openai.com\n"
                "  export GOOGLE_API_KEY=AIza...          # https://aistudio.google.com  (free for NCAR)\n"
                "  export GROQ_API_KEY=gsk_...            # https://console.groq.com     (free tier)\n\n"
                "Add to ~/.bashrc to persist.\n"
                "See: https://miles-credit.readthedocs.io/en/latest/quickstart.html"
                "#get-help-from-the-ai-assistant",
                file=sys.stderr,
            )
            sys.exit(1)

    # ---- Check package is installed ----
    _, pkg, label = _PROVIDERS[provider]
    try:
        __import__(pkg)
    except ImportError:
        print(f"pip install {_PROVIDER_INSTALL[provider]}", file=sys.stderr)
        sys.exit(1)

    question = " ".join(args.question)
    context = _collect_run_context(args)
    user_msg = f"{context}\n\n## Question\n{question}" if context else question

    # Build ordered list of providers to try: chosen one first, then the rest
    if explicit:
        ordered = [provider]
    else:
        ordered = list(_PROVIDERS.keys())
        # Only include providers whose keys are actually set
        ordered = [p for p in ordered if os.environ.get(_PROVIDERS[p][0])]

    print()
    for attempt, p in enumerate(ordered):
        _, pkg, label = _PROVIDERS[p]
        # Skip if package not installed
        try:
            __import__(pkg)
        except ImportError:
            continue
        if p != "anthropic" or attempt > 0:
            print(f"(using {label})\n")
        try:
            _PROVIDER_RUNNERS[p](user_msg)
            print("\n")
            return
        except _ProviderError as exc:
            remaining = [x for x in ordered[attempt + 1 :] if os.environ.get(_PROVIDERS[x][0])]
            print(f"\nWarning: {exc}", file=sys.stderr)
            if remaining:
                print("Falling back to next available provider…\n", file=sys.stderr)
            else:
                sys.exit(1)

    print("No working provider found.", file=sys.stderr)
    sys.exit(1)


def _plot(args: argparse.Namespace) -> None:
    """Load checkpoint, run one forward pass, produce global maps."""
    import yaml
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for credit plot. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        _HAS_CARTOPY = True
    except ImportError:
        _HAS_CARTOPY = False
        logger.warning("cartopy not found — using plain lat/lon axes. Install cartopy for globe projections.")

    import torch
    import torch.nn as nn

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    ckpt_path = args.checkpoint or os.path.join(save_loc, "checkpoint.pt")
    out_dir = args.output_dir or os.path.join(save_loc, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Load model ----
    from credit.models import load_model
    from credit.models.checkpoint import load_state_dict_error_handler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(conf, load_weights=False)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_msg = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    load_state_dict_error_handler(load_msg)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ---- Load one validation sample ----
    import pandas as pd
    from credit.datasets.multi_source import MultiSourceDataset
    from credit.preblock import ERA5Normalizer, ConcatPreblock, apply_preblocks

    data_conf = conf.get("data_valid", conf["data"])

    dataset = MultiSourceDataset(data_conf, return_target=True)

    # pick timestamp
    if args.sample_date is not None:
        target_dt = pd.Timestamp(args.sample_date)
        ts = None
        for t in dataset.datetimes:
            if pd.Timestamp(t) >= target_dt:
                ts = t
                break
        if ts is None:
            logger.warning("sample_date %s not found — using first sample", args.sample_date)
            ts = dataset.datetimes[0]
    else:
        ts = dataset.datetimes[0]

    sample = dataset[(ts, 0)]

    # ---- Pre-process (normalise + concat) ----
    from torch.utils.data import default_collate

    # default_collate adds the batch dimension exactly as the DataLoader would
    batch = default_collate([sample])

    preblocks = nn.ModuleDict(
        {
            "norm": ERA5Normalizer(conf),
            "concat": ConcatPreblock(),
        }
    )
    batch = apply_preblocks(preblocks, batch)

    x = batch["x"].to(device)  # (1, C_in, T, H, W)
    y = batch["y"]  # (1, C_out, T, H, W)

    # ---- Forward pass ----
    with torch.no_grad():
        y_pred = model(x)

    # squeeze batch dim and time dim → (C, H, W)
    def _squeeze(t):
        t = t.squeeze(0).cpu().float()  # remove batch dim → (C, T, H, W) or (C, H, W)
        if t.ndim == 4:
            t = t[:, 0]  # take first time step → (C, H, W)
        return t

    y_true_np = _squeeze(y).numpy()  # (C_out, H, W)
    y_pred_np = _squeeze(y_pred).numpy()

    # ---- Inverse-normalise to physical units (optional) ----
    unit_label = "normalised"
    if args.denorm:
        mean_arr, std_arr = _build_denorm_stats(conf)
        mean_arr = mean_arr[:, None, None]  # broadcast over H, W
        std_arr = std_arr[:, None, None]
        y_true_np = y_true_np * std_arr + mean_arr
        y_pred_np = y_pred_np * std_arr + mean_arr
        unit_label = "physical units"
        logger.info("Inverse-normalised outputs to physical units")

    # ---- Channel map ----
    channel_map = _build_channel_map(conf)

    # ---- Plot each requested field ----
    for field in args.field:
        if field not in channel_map:
            available = ", ".join(sorted(channel_map.keys()))
            print(f"Field '{field}' not found. Available: {available}", file=sys.stderr)
            continue

        chans = channel_map[field]
        level_idx = min(args.level, len(chans) - 1)
        c = chans[level_idx]

        truth = y_true_np[c]  # (H, W)
        pred = y_pred_np[c]
        diff = pred - truth

        # lat/lon grid
        H, W = truth.shape
        lats = np.linspace(90, -90, H)
        lons = np.linspace(0, 360, W, endpoint=False)

        vmin = float(np.percentile(truth, 2))
        vmax = float(np.percentile(truth, 98))
        dabs = float(np.percentile(np.abs(diff), 98))

        title_suffix = f"  level {level_idx}" if len(chans) > 1 else ""
        ckpt_epoch = ckpt.get("epoch", "?")
        fig_title = f"{field}{title_suffix}  |  epoch {ckpt_epoch}  |  {unit_label}"

        if _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                subplot_kw={"projection": proj},
            )

            def _add_features(ax):
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.set_global()

            im0 = axes[0].pcolormesh(lons, lats, truth, vmin=vmin, vmax=vmax, cmap="RdBu_r", transform=proj)
            _add_features(axes[0])
            axes[0].set_title("Truth")
            plt.colorbar(im0, ax=axes[0], shrink=0.6)

            im1 = axes[1].pcolormesh(lons, lats, pred, vmin=vmin, vmax=vmax, cmap="RdBu_r", transform=proj)
            _add_features(axes[1])
            axes[1].set_title("Prediction")
            plt.colorbar(im1, ax=axes[1], shrink=0.6)

            im2 = axes[2].pcolormesh(lons, lats, diff, vmin=-dabs, vmax=dabs, cmap="bwr", transform=proj)
            _add_features(axes[2])
            axes[2].set_title("Difference (pred − truth)")
            plt.colorbar(im2, ax=axes[2], shrink=0.6)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(truth, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto", origin="upper")
            axes[0].set_title("Truth")
            axes[0].axis("off")
            axes[1].imshow(pred, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto", origin="upper")
            axes[1].set_title("Prediction")
            axes[1].axis("off")
            im2 = axes[2].imshow(diff, vmin=-dabs, vmax=dabs, cmap="bwr", aspect="auto", origin="upper")
            axes[2].set_title("Difference (pred − truth)")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], shrink=0.7)

        fig.suptitle(fig_title, fontsize=13)
        plt.tight_layout()

        safe_field = field.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_field}_lev{level_idx:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit",
        description="CREDIT — AI-NWP model training and inference platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              credit train -c config.yml
              credit realtime -c config.yml --init-time 2024-01-15T00 --steps 40
              credit rollout  -c config.yml
              credit submit   --cluster casper  -c config.yml --gpus 1
              credit submit   --cluster derecho -c config.yml --gpus 4 --nodes 2
              credit init     --grid 0.25deg -o my_config.yml
        """),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

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

    # ---- realtime ----
    p = sub.add_parser("realtime", help="Operational realtime forecast (single init time)")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to YAML config")
    p.add_argument(
        "--init-time", required=True, metavar="YYYY-MM-DDTHH", help="Forecast initialisation time, e.g. 2024-01-15T00"
    )
    p.add_argument("--steps", type=int, default=40, help="Number of autoregressive forecast steps (default: 40)")
    p.add_argument("--save-dir", metavar="DIR", help="Override output directory from config")
    p.add_argument("-m", "--mode", default="none", help="Distributed mode: none | ddp | fsdp (default: none)")
    p.add_argument("-p", "--procs", type=int, default=4, help="CPU workers for async NetCDF save (default: 4)")

    # ---- submit ----
    p = sub.add_parser(
        "submit",
        help="Generate and submit a PBS training job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate a PBS batch script and optionally submit it via qsub.
            Use --dry-run to inspect the script before submitting.
            Use --reload to resume from the latest checkpoint automatically.
            Use --chain N to submit N back-to-back jobs via PBS afterok dependencies.

            Examples:
              credit submit --cluster casper  -c config.yml --gpus 1 --walltime 04:00:00
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 2 --dry-run
              credit submit --cluster casper  -c config.yml --gpus 4 --reload
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 1 --chain 10
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG")
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"], help="Target NCAR HPC cluster")
    p.add_argument("--gpus", type=int, default=4, metavar="N", help="GPUs per node (default: 4)")
    p.add_argument("--nodes", type=int, default=1, metavar="N", help="Number of nodes, derecho only (default: 1)")
    p.add_argument("--cpus", type=int, default=None, metavar="N", help="CPUs per node (default: 8 casper / 64 derecho)")
    p.add_argument("--mem", default=None, help="Memory per node (default: 128GB casper / 480GB derecho)")
    p.add_argument("--walltime", default="12:00:00", metavar="HH:MM:SS", help="Job walltime (default: 12:00:00)")
    p.add_argument("--account", metavar="ACCOUNT", help="PBS account code (default: $PBS_ACCOUNT or NAML0001)")
    p.add_argument("--queue", metavar="QUEUE", help="PBS queue (default: casper / main)")
    p.add_argument("--gpu-type", dest="gpu_type", default=None, help="Casper GPU type (default: a100_80gb)")
    p.add_argument(
        "--torchrun", default=None, metavar="PATH", help="Path to torchrun binary (default: auto-detect from PATH)"
    )
    p.add_argument(
        "--conda-env",
        dest="conda_env",
        default=None,
        metavar="PATH",
        help="Conda environment path for derecho (default: credit-derecho-torch28-nccl221)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the PBS script without submitting")
    p.add_argument(
        "--reload",
        action="store_true",
        help="Resume from checkpoint: patch load_weights/optimizer/scaler/"
        "scheduler/reload_epoch in the config and submit the reload job",
    )
    p.add_argument(
        "--chain",
        type=int,
        default=None,
        metavar="N",
        help="Submit N jobs in sequence using PBS afterok dependencies. "
        "Job 1 uses the base config (or --reload config); jobs 2..N "
        "are automatic reload jobs. If omitted, computed automatically "
        "from ceil(trainer.epochs / trainer.num_epoch) in the config. "
        "Example: --chain 10 submits 10 back-to-back jobs.",
    )

    # ---- plot ----
    p = sub.add_parser(
        "plot",
        help="Quick global map: truth vs prediction from a saved checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Load a checkpoint, run one forward pass on a validation sample, and
            produce a 3-panel global map (truth | prediction | difference) for each
            requested field. Saved to <save_loc>/plots/.

            Examples:
              credit plot -c config.yml --field VAR_2T --denorm
              credit plot -c config.yml --field VAR_2T SP --level 5 --denorm
              credit plot -c config.yml --field SP --checkpoint /path/to/checkpoint.pt --denorm
              credit plot -c config.yml --field VAR_2T --sample-date 2020-06-01T00 --denorm
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Training config YAML")
    p.add_argument(
        "--field", nargs="+", required=True, metavar="VAR", help="Variable name(s) to plot, e.g. temperature SP VAR_10U"
    )
    p.add_argument(
        "--level",
        type=int,
        default=0,
        metavar="IDX",
        help="Level index for 3-D variables (0 = first level, default: 0)",
    )
    p.add_argument(
        "--checkpoint", default=None, metavar="PATH", help="Checkpoint file (default: <save_loc>/checkpoint.pt)"
    )
    p.add_argument(
        "--sample-date",
        default=None,
        metavar="YYYY-MM-DDTHH",
        dest="sample_date",
        help="Validation sample init time (default: first sample in valid set)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        dest="output_dir",
        help="Where to save plots (default: <save_loc>/plots/)",
    )
    p.add_argument(
        "--denorm",
        action="store_true",
        help="Inverse-normalise output to physical units using mean/std files"
        " from the config (e.g. K for temperature, Pa for surface pressure)",
    )

    # ---- ask ----
    p = sub.add_parser(
        "ask",
        help="Ask the CREDIT AI assistant a question about your run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Ask the CREDIT AI assistant anything about training, config, or debugging.
            Supports Anthropic, OpenAI, Google Gemini, and Groq — whichever key is set.

            Provider priority (first key found wins, or use --provider to pick):
              ANTHROPIC_API_KEY  → Claude Haiku       https://console.anthropic.com
              OPENAI_API_KEY     → GPT-4o             https://platform.openai.com
              GOOGLE_API_KEY     → Gemini 1.5 Pro     https://aistudio.google.com (free for NCAR)
              GROQ_API_KEY       → Llama 3 Instant    https://console.groq.com    (free tier)

            Examples:
              credit ask "why is my training loss stuck at 2.5?"
              credit ask -c config.yml "is my batch size too large for 0.25 degree?"
              credit ask --provider gemini -c config.yml "what do I do if my Derecho job hangs?"
        """),
    )
    p.add_argument("question", nargs="+", metavar="QUESTION", help="Your question (quote it or pass as multiple words)")
    p.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="CONFIG",
        help="Optional config YAML — injects your run's config, training log, and most recent PBS output as context",
    )
    p.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai", "gemini", "groq"],
        help="Force a specific LLM provider (default: auto-detect from env keys)",
    )

    # ---- init ----
    p = sub.add_parser("init", help="Generate a starter config from a built-in template")
    p.add_argument(
        "--grid", choices=["0.25deg", "1deg"], default="0.25deg", help="Horizontal grid resolution (default: 0.25deg)"
    )
    p.add_argument(
        "--model",
        choices=["crossformer", "wxformer_v2"],
        default="wxformer_v2",
        help="Model architecture (default: wxformer_v2)",
    )
    p.add_argument(
        "-o", "--output", default="config.yml", metavar="FILE", help="Output file path (default: config.yml)"
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _setup_logging()

    dispatch = {
        "train": _train,
        "rollout": _rollout,
        "realtime": _realtime,
        "submit": _submit,
        "init": _init,
        "plot": _plot,
        "ask": _ask,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
