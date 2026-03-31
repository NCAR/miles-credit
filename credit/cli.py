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
import pathlib
import logging
import os
import sys
import textwrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt(prompt: str, default=None) -> str:
    """Print a prompt and return stripped input, or *default* if empty."""
    hint = f" [{default}]" if default is not None else ""
    val = input(f"  {prompt}{hint}: ").strip()
    return val if val else (str(default) if default is not None else "")


def _prompt_bool(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


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

    argv = [
        "credit-rollout",
        "-c",
        args.config,
        "-m",
        args.mode,
        "-cpus",
        str(args.procs),
    ]
    if getattr(args, "ensemble_size", None) is not None:
        argv += ["--ensemble-size", str(args.ensemble_size)]
    sys.argv = argv
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


def _convert(args: argparse.Namespace) -> None:
    """Interactive v1 → v2 config converter."""
    import yaml

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    trainer_type = conf.get("trainer", {}).get("type", "unknown")

    print()
    print("=" * 62)
    print("  CREDIT config converter  (v1 → v2)")
    print("=" * 62)
    print(f"  Input  : {args.config}")
    print(f"  Trainer: {trainer_type}")
    print()

    # ------------------------------------------------------------------
    # Auto-transformations — no questions needed
    # ------------------------------------------------------------------
    changes = []

    # trainer.type
    V1_TYPES = {"era5", "standard", "universal"}
    if trainer_type in V1_TYPES:
        conf["trainer"]["type"] = "era5-v2"
        changes.append(f"trainer.type: '{trainer_type}' → 'era5-v2'")

    # forecast_len: v1 uses 0 = single step, v2 uses 1 = single step
    fl = conf.get("data", {}).get("forecast_len", 0)
    new_fl = fl + 1
    conf["data"]["forecast_len"] = new_fl
    changes.append(f"data.forecast_len: {fl} → {new_fl}  (v2: 1 = single step)")

    vfl = conf.get("data", {}).get("valid_forecast_len", fl)
    new_vfl = vfl + 1
    conf["data"]["valid_forecast_len"] = new_vfl
    changes.append(f"data.valid_forecast_len: {vfl} → {new_vfl}")

    # backprop_on_timestep: v1 is 0-indexed, v2 is 1-indexed
    bpt = conf.get("data", {}).get("backprop_on_timestep")
    if bpt is not None:
        new_bpt = [t + 1 for t in bpt]
        conf["data"]["backprop_on_timestep"] = new_bpt
        changes.append(f"data.backprop_on_timestep: {bpt} → {new_bpt}  (1-indexed in v2)")

    print("  Auto-applied:")
    for c in changes:
        print(f"    + {c}")
    print()

    # ------------------------------------------------------------------
    # New v2 trainer features
    # ------------------------------------------------------------------
    print("  --- New v2 trainer features ---")

    use_ema = _prompt_bool("Enable EMA (exponential moving average of weights)? Recommended", default=True)
    conf["trainer"]["use_ema"] = use_ema
    if use_ema:
        ema_decay = _prompt("EMA decay", default="0.9999")
        conf["trainer"]["ema_decay"] = float(ema_decay)

    use_tb = _prompt_bool("Enable TensorBoard logging", default=True)
    conf["trainer"]["use_tensorboard"] = use_tb
    print()

    # ------------------------------------------------------------------
    # Ensemble detection
    # ------------------------------------------------------------------
    ensemble_size = conf.get("trainer", {}).get("ensemble_size", 1)
    loss_type = conf.get("loss", {}).get("training_loss", "")
    is_ensemble = ensemble_size > 1 or "crps" in loss_type.lower()

    if is_ensemble:
        print(f"  --- Ensemble settings (detected: ensemble_size={ensemble_size}, loss={loss_type}) ---")
        keep_ensemble = _prompt_bool("Keep ensemble training", default=True)
        if keep_ensemble:
            new_size = _prompt("Ensemble size", default=str(ensemble_size))
            conf["trainer"]["ensemble_size"] = int(new_size)
        else:
            conf["trainer"]["ensemble_size"] = 1
            print("  Note: consider changing loss.training_loss from crps to mse or mae")
        print()

    # ------------------------------------------------------------------
    # PBS / job settings
    # ------------------------------------------------------------------
    print("  --- PBS / job settings ---")
    pbs = conf.get("pbs", {})

    cluster = _prompt("Cluster (casper/derecho)", default="derecho")
    account = _prompt(
        "PBS account code",
        default=pbs.get("project") or pbs.get("account") or os.environ.get("PBS_ACCOUNT") or "NAML0001",
    )
    conda = _prompt(
        "Conda env (name or full path)", default=pbs.get("conda") or pbs.get("conda_env") or "credit-derecho"
    )
    walltime = _prompt("Walltime (HH:MM:SS)", default=pbs.get("walltime") or "12:00:00")
    job_name = _prompt("Job name", default=pbs.get("job_name") or "credit_v2")

    new_pbs = {
        "project": account,
        "job_name": job_name,
        "walltime": walltime,
        "conda": conda,
    }

    if cluster == "derecho":
        nodes = int(_prompt("Nodes", default=str(pbs.get("nodes") or 1)))
        gpus = int(_prompt("GPUs per node", default=str(pbs.get("ngpus") or pbs.get("gpus") or 4)))
        cpus = int(_prompt("CPUs per node", default=str(pbs.get("ncpus") or pbs.get("cpus") or 64)))
        mem = _prompt("Memory per node", default=pbs.get("mem") or "480GB")
        queue = _prompt("Queue", default=pbs.get("queue") or "main")
        new_pbs.update({"nodes": nodes, "ngpus": gpus, "ncpus": cpus, "mem": mem, "queue": queue})
    else:
        gpus = int(_prompt("GPUs", default=str(pbs.get("ngpus") or pbs.get("gpus") or 4)))
        cpus = int(_prompt("CPUs per node", default=str(pbs.get("ncpus") or pbs.get("cpus") or 8)))
        mem = _prompt("Memory", default=pbs.get("mem") or "128GB")
        gpu_type = _prompt("GPU type", default=pbs.get("gpu_type") or "a100_80gb")
        queue = _prompt("Queue", default=pbs.get("queue") or "casper")
        new_pbs.update({"ngpus": gpus, "ncpus": cpus, "mem": mem, "gpu_type": gpu_type, "queue": queue})

    conf["pbs"] = new_pbs

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print()
    base, ext = os.path.splitext(args.config)
    default_out = getattr(args, "output", None) or (f"{base}_v2{ext}" if ext else f"{args.config}_v2.yml")
    out_path = _prompt("Output config path", default=default_out)

    with open(out_path, "w") as f:
        yaml.dump(conf, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print()
    print(f"  Saved → {out_path}")
    print("=" * 62)
    print()


def _load_pbs_config(config_path: str) -> dict:
    """Return the ``pbs:`` section from a YAML config file, or an empty dict."""
    try:
        import yaml

        with open(config_path) as f:
            conf = yaml.safe_load(f)
        return conf.get("pbs", {}) or {}
    except Exception:
        return {}


def _resolve_pbs_opts(args: argparse.Namespace, pbs_cfg: dict) -> argparse.Namespace:
    """Return a copy of *args* with None fields filled from *pbs_cfg* then cluster defaults.

    Resolution order: CLI flag > config ``pbs:`` section > cluster default.

    Config key aliases (both spellings accepted):
      project / account  → PBS account code
      ngpus   / gpus     → GPUs per node
      ncpus   / cpus     → CPUs per node
      conda   / conda_env → conda environment path
    """
    import copy

    r = copy.copy(args)

    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    is_casper = args.cluster == "casper"

    r.account = _first(
        args.account,
        pbs_cfg.get("project") or pbs_cfg.get("account"),
        os.environ.get("PBS_ACCOUNT"),
        "NAML0001",
    )
    r.walltime = _first(args.walltime, pbs_cfg.get("walltime"), "12:00:00")
    r.gpus = int(_first(args.gpus, pbs_cfg.get("ngpus") or pbs_cfg.get("gpus"), 4))
    r.nodes = int(_first(args.nodes, pbs_cfg.get("nodes"), 1))
    r.cpus = int(_first(args.cpus, pbs_cfg.get("ncpus") or pbs_cfg.get("cpus"), 8 if is_casper else 64))
    r.mem = _first(args.mem, pbs_cfg.get("mem"), "128GB" if is_casper else "480GB")
    r.queue = _first(args.queue, pbs_cfg.get("queue"), "casper" if is_casper else "main")
    r.gpu_type = _first(args.gpu_type, pbs_cfg.get("gpu_type"), "a100_80gb")
    r.conda_env = _first(
        args.conda_env,
        pbs_cfg.get("conda") or pbs_cfg.get("conda_env"),
        "/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221",
    )
    r.job_name = pbs_cfg.get("job_name", "credit_v2")
    return r


def _build_pbs_script(
    args: argparse.Namespace, config: str, repo: str, account: str = None, depend_on: str = None
) -> str:
    """Return a PBS batch script string for the given args and config path.

    *args* must already be resolved via :func:`_resolve_pbs_opts` so that all
    fields (gpus, nodes, walltime, …) are concrete values.

    Args:
        account:   PBS account string; overrides ``args.account`` when provided.
        depend_on: If set, adds ``#PBS -W depend=afterok:<depend_on>`` so this
                   job only starts after the given job ID completes successfully.
    """
    # Apply defaults for fields not set by the caller (mirrors _resolve_pbs_opts defaults).
    is_casper = getattr(args, "cluster", "casper") == "casper"
    _d = argparse.Namespace(
        job_name="credit_v2",
        account="NAML0001",
        cpus=8 if is_casper else 64,
        mem="128GB" if is_casper else "480GB",
        queue="casper" if is_casper else "main",
        gpu_type="a100_80gb",
        torchrun=None,
        conda_env="/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221",
    )
    args = argparse.Namespace(**{**vars(_d), **{k: v for k, v in vars(args).items() if v is not None}})
    if account is not None:
        args.account = account
    depend_line = f"#PBS -W depend=afterok:{depend_on}\n" if depend_on else ""

    if args.cluster == "casper":
        torchrun = args.torchrun or _find_torchrun()

        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {args.job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}:gpu_type={args.gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {args.account}
            #PBS -q {args.queue}
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
        conda_env = args.conda_env

        header = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {args.account}
            #PBS -N {args.job_name}
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod
            #PBS -r n
            {depend_line}
            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
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

    gpu_str = f"{args.gpus} GPU(s)"
    if args.cluster == "derecho" and args.nodes > 1:
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
    print(f"  Account  : {getattr(args, 'account', 'unset')}")
    print(f"  Config   : {getattr(args, 'config', 'unset')}")
    print(f"  GPUs     : {gpu_str}")
    print(f"  Walltime : {args.walltime} per job")
    print(f"  Chain    : {chain_desc}")
    print(f"  DataLoader memory est. : {mem_str}{mem_warn}")
    print("=" * 52)
    print()


def _submit(args: argparse.Namespace) -> None:
    """Generate and optionally submit PBS batch scripts, with optional chaining."""
    if getattr(args, "rollout", False):
        _do_submit_rollout(args)
        return

    repo = _repo_root()
    pbs_cfg = _load_pbs_config(args.config)
    args = _resolve_pbs_opts(args, pbs_cfg)
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
        script = _build_pbs_script(args, first_config, repo, depend_on=None)
        print(f"# --- Job 1/{n_jobs} ---")
        print(script)
        if n_jobs > 1:
            script2 = _build_pbs_script(args, reload_config, repo, depend_on="<job_1_id>")
            print(f"# --- Jobs 2..{n_jobs}/{n_jobs} (afterok chained, reload config) ---")
            print(script2)
        return

    # Submit job 1
    script = _build_pbs_script(args, first_config, repo, depend_on=None)
    job_id = _qsub(script)
    print(f"[1/{n_jobs}] {job_id}  {first_config}")

    # Submit remaining chained reload jobs
    for i in range(2, n_jobs + 1):
        script = _build_pbs_script(args, reload_config, repo, depend_on=job_id)
        job_id = _qsub(script)
        print(f"[{i}/{n_jobs}] {job_id}  afterok  (reload)")


def _build_rollout_pbs_script(args: argparse.Namespace, config: str, repo: str, subset: int, n_subsets: int) -> str:
    """Return a PBS script for one subset of an ensemble rollout.

    Each job runs ``rollout_to_netcdf_v2.py --subset <subset> --no_subset <n_subsets>``.
    Jobs are independent (no afterok chain) — they all start at once.
    """
    job_name = f"{args.job_name[:10]}-{subset:02d}of{n_subsets:02d}"

    if args.cluster == "casper":
        torchrun = args.torchrun or _find_torchrun()
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}:gpu_type={args.gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {args.account}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod

            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}

            echo "Ensemble rollout — subset {subset} of {n_subsets}"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"
            echo "GPUs    : ${{NGPUS}}"

            export PYTHONPATH="${{REPO}}:${{PYTHONPATH}}"
            export PYTHONNOUSERSITE=1
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/applications/rollout_to_netcdf_v2.py \\
                -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}
        """)

    else:  # derecho — single node for rollout (inference doesn't need multi-node)
        conda_env = args.conda_env
        return textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {args.account}
            #PBS -N {job_name}
            #PBS -l walltime={args.walltime}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod
            #PBS -r n

            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest

            conda activate {conda_env}

            REPO={repo}
            CONFIG={config}

            echo "Ensemble rollout — subset {subset} of {n_subsets}"
            echo "Config  : ${{CONFIG}}"

            export PYTHONPATH="${{REPO}}:${{PYTHONPATH}}"
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            torchrun --standalone --nnodes=1 --nproc-per-node={args.gpus} \\
                ${{REPO}}/applications/rollout_to_netcdf_v2.py \\
                -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}
        """)


def _print_ensemble_rollout_plan(args: argparse.Namespace, n_jobs: int, n_forecasts: int, ensemble_size: int) -> None:
    """Print a human-readable summary of an ensemble rollout submission."""
    per_job = -(-n_forecasts // n_jobs)  # ceiling division
    total_runs = n_forecasts * ensemble_size

    print()
    print("=" * 56)
    print("  Ensemble rollout plan")
    print("=" * 56)
    print(f"  Cluster        : {args.cluster}")
    print(f"  Account        : {args.account}")
    print(f"  Config         : {args.config}")
    print(f"  Init times     : {n_forecasts}  ({per_job} per job)")
    print(f"  Ensemble size  : {ensemble_size}  →  {total_runs} total forecasts")
    print(f"  Parallel jobs  : {n_jobs}  (all start at once, no dependencies)")
    print(f"  GPUs per job   : {args.gpus}")
    print(f"  Walltime/job   : {args.walltime}")
    print("=" * 56)
    print()


def _rollout_ensemble(args: argparse.Namespace) -> None:
    """Deprecated: use ``credit submit --rollout`` instead."""
    print(
        "WARNING: credit rollout-ensemble is deprecated.\n"
        "Use instead:\n"
        "  credit submit --cluster <casper|derecho> --rollout --jobs N -c config.yml\n",
        file=sys.stderr,
    )
    args.rollout = True
    if not hasattr(args, "reload"):
        args.reload = False
    if not hasattr(args, "chain"):
        args.chain = None
    if not hasattr(args, "nodes"):
        args.nodes = None
    _submit(args)


def _do_submit_rollout(args: argparse.Namespace) -> None:
    """Submit N parallel PBS rollout jobs to cover all init times."""
    import yaml

    repo = _repo_root()
    pbs_cfg = _load_pbs_config(args.config)

    # Rollout-specific cluster defaults (inference is lighter than training)
    is_casper = args.cluster == "casper"
    _rollout_defaults = {
        "ngpus": 1,
        "ncpus": 8 if is_casper else 16,
        "mem": "128GB",
        "walltime": "06:00:00",
    }
    # Apply rollout defaults only where pbs_cfg doesn't already specify
    merged_pbs = {**_rollout_defaults, **{k: v for k, v in pbs_cfg.items() if v is not None}}

    # _resolve_pbs_opts expects these fields; rollout always runs single-node
    if not hasattr(args, "nodes"):
        args.nodes = None
    if not hasattr(args, "torchrun"):
        args.torchrun = None

    args = _resolve_pbs_opts(args, merged_pbs)

    # Rollout defaults to 1 GPU per job (inference, not training)
    # _resolve_pbs_opts set args.gpus from config/CLI; if still defaulting from
    # training defaults (4), honour the user choice — they may want more.

    n_jobs = args.jobs

    # Count init times from config so we can show a useful plan
    try:
        from credit.forecast import load_forecasts

        with open(args.config) as f:
            conf = yaml.safe_load(f)
        all_forecasts = load_forecasts(conf)
        n_forecasts = len(all_forecasts)
        ensemble_size = conf.get("predict", {}).get("ensemble_size", 1)
    except Exception:
        n_forecasts = "?"
        ensemble_size = "?"

    _print_ensemble_rollout_plan(args, n_jobs, n_forecasts, ensemble_size)

    config_abs = os.path.abspath(args.config)

    if args.dry_run:
        for i in range(1, n_jobs + 1):
            print(f"# {'=' * 50}")
            print(f"# Job {i}/{n_jobs}  (subset {i} of {n_jobs})")
            print(f"# {'=' * 50}")
            print(_build_rollout_pbs_script(args, config_abs, repo, i, n_jobs))
        return

    job_ids = []
    for i in range(1, n_jobs + 1):
        script = _build_rollout_pbs_script(args, config_abs, repo, i, n_jobs)
        job_id = _qsub(script)
        job_ids.append(job_id)
        print(f"[{i:2d}/{n_jobs}] {job_id}")

    print()
    print(f"Submitted {n_jobs} parallel rollout jobs.")
    print(f"Output will be written to: {conf.get('predict', {}).get('save_forecast', '<save_forecast in config>')}")
    print()
    print("Monitor with:")
    print("  qstat -u $USER")


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
You are CREDIT-Ask, an AI assistant for the CREDIT software package (Community Research Earth Digital Intelligence Twin),
an AI-based numerical weather prediction framework developed by the NCAR MILES group.
When introducing yourself, use the name "CREDIT-Ask". Do not call yourself "CREDIT" — that is the name of the software package you support.

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
    import logging

    import anthropic

    # httpx logs every request at INFO level — suppress it for clean CLI output
    logging.getLogger("httpx").setLevel(logging.WARNING)

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
                "Anthropic API key has no credits.\n"
                "  → Add credits: https://console.anthropic.com/settings/billing\n"
                "  → Note: Claude.ai Pro does NOT include API access."
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
    """Unified AI assistant: tries agentic mode (Anthropic tool-use) first, falls back to simple chat.

    Agent mode (when ANTHROPIC_API_KEY is set and anthropic is installed):
      Multi-turn loop — reads files, runs shell commands, iterates until it has a confident answer.

    Simple chat fallback (Groq, Gemini, OpenAI, or Anthropic Haiku):
      One-shot Q&A using whichever key is set.  Provider priority:
        1. ANTHROPIC_API_KEY  → Claude Haiku        (pip install anthropic)
        2. OPENAI_API_KEY     → GPT-4o              (pip install openai)
        3. GOOGLE_API_KEY     → Gemini 1.5 Pro      (pip install google-generativeai)
        4. GROQ_API_KEY       → Llama 3 Instant     (pip install groq — free tier)
    """
    import logging

    question = " ".join(args.question)
    context = _collect_run_context(args)
    explicit = getattr(args, "provider", None)

    # ------------------------------------------------------------------
    # Agent mode: Anthropic tool-use (skipped when --provider forces
    # a non-Anthropic provider).
    # ------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    _try_agent = (explicit is None or explicit == "anthropic") and bool(api_key)
    _skip_anthropic = False  # set True if agent billing fails

    if _try_agent:
        _ant = None
        try:
            import anthropic as _ant_mod

            _ant = _ant_mod
        except ImportError:
            pass

        if _ant is not None:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            user_msg = f"{context}\n\n## Task\n{question}" if context else question
            client = _ant.Anthropic(api_key=api_key)
            messages = [{"role": "user", "content": user_msg}]
            max_turns = getattr(args, "max_turns", 20)
            print()

            for _turn in range(max_turns):
                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=4096,
                        system=_AGENT_SYSTEM_PROMPT,
                        tools=_AGENT_TOOL_DEFS,
                        messages=messages,
                    )
                except _ant.BadRequestError as exc:
                    if "credit balance is too low" in str(exc):
                        print(
                            "Anthropic API key has no credits — falling back to simple chat.\n",
                            file=sys.stderr,
                        )
                        _skip_anthropic = True
                        break
                    print(f"API error: {exc}", file=sys.stderr)
                    sys.exit(1)

                for block in response.content:
                    if block.type == "text" and block.text:
                        print(block.text, end="", flush=True)

                tool_calls = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason == "end_turn" or not tool_calls:
                    print("\n")
                    return

                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tc in tool_calls:
                    if not any(b.type == "text" for b in response.content):
                        print(f"\n[{tc.name}: {tc.input}]", flush=True)
                    result = _dispatch_tool(tc.name, tc.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
            else:
                print(f"\n[reached max_turns={max_turns}]", file=sys.stderr)
                print("\n")
                return
            # loop exited via break (billing error) — fall through to simple chat

    # ------------------------------------------------------------------
    # Simple chat fallback
    # ------------------------------------------------------------------
    user_msg = f"{context}\n\n## Question\n{question}" if context else question

    if explicit:
        if explicit not in _PROVIDERS:
            print(f"Unknown provider {explicit!r}. Choose: {', '.join(_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)
        env_key, _pkg, _label = _PROVIDERS[explicit]
        if not os.environ.get(env_key):
            print(f"{env_key} is not set.", file=sys.stderr)
            sys.exit(1)
        ordered = [explicit]
    else:
        ordered = [p for p in _PROVIDERS if os.environ.get(_PROVIDERS[p][0])]
        if _skip_anthropic and "anthropic" in ordered:
            ordered.remove("anthropic")

    if not ordered:
        if _skip_anthropic:
            print(
                "\nNo working provider found.\n"
                "Anthropic credits are exhausted. Set a fallback key:\n"
                "  export GROQ_API_KEY=gsk_...    # https://console.groq.com  (free)\n"
                "  export GOOGLE_API_KEY=AIza...  # https://aistudio.google.com\n"
                "  export OPENAI_API_KEY=sk-...   # https://platform.openai.com",
                file=sys.stderr,
            )
        else:
            _ncar = _is_ncar_system()
            msg = "No API key found."
            if _ncar:
                msg += (
                    "\n\nOn NCAR systems (Casper/Derecho) you can use the shared Anthropic credits:\n\n"
                    "  module use /glade/work/bdobbins/llms/modules\n"
                    "  module load llms\n\n"
                    "Add those two lines to ~/.bashrc to persist across sessions."
                )
            msg += (
                "\n\nOr set your own key:\n\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com\n"
                "  export OPENAI_API_KEY=sk-...           # https://platform.openai.com\n"
                "  export GOOGLE_API_KEY=AIza...          # https://aistudio.google.com  (free for NCAR)\n"
                "  export GROQ_API_KEY=gsk_...            # https://console.groq.com     (free tier)\n\n"
                "See: https://miles-credit.readthedocs.io/en/latest/quickstart.html"
                "#get-help-from-the-ai-assistant"
            )
            print(msg, file=sys.stderr)
        sys.exit(1)

    print()
    for attempt, p in enumerate(ordered):
        _, pkg, label = _PROVIDERS[p]
        try:
            __import__(pkg)
        except ImportError:
            print(
                f"Skipping {label}: package not installed.  Fix: pip install {_PROVIDER_INSTALL[p]}",
                file=sys.stderr,
            )
            continue
        if p != "anthropic" or attempt > 0:
            print(f"(using {label})\n")
        try:
            _PROVIDER_RUNNERS[p](user_msg)
            print("\n")
            return
        except _ProviderError as exc:
            print(f"\n{exc}", file=sys.stderr)
            remaining = [x for x in ordered[attempt + 1 :] if os.environ.get(_PROVIDERS[x][0])]
            if remaining:
                print("Trying next provider…\n", file=sys.stderr)

    print(
        "\nNo working provider found. Set one of:\n"
        "  export GROQ_API_KEY=gsk_...      # https://console.groq.com  (free)\n"
        "  export GOOGLE_API_KEY=AIza...    # https://aistudio.google.com  (free for NCAR)\n"
        "  export OPENAI_API_KEY=sk-...     # https://platform.openai.com\n"
        "  export ANTHROPIC_API_KEY=sk-ant- # https://console.anthropic.com  (requires credits)",
        file=sys.stderr,
    )
    sys.exit(1)


def _is_ncar_system() -> bool:
    """Return True if running on a known NCAR HPC system (Casper or Derecho)."""
    import socket

    host = socket.gethostname()
    return any(name in host for name in ("casper", "crhtc", "derecho", "dec", "crlogin"))


# ---------------------------------------------------------------------------
# credit agent — agentic session with file/bash tools
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are CREDIT-Agent, an agentic AI assistant for the CREDIT software package (Community Research Earth Digital Intelligence Twin),
an AI-based numerical weather prediction framework developed by the NCAR MILES group.
When introducing yourself, use the name "CREDIT-Agent". Do not call yourself "CREDIT" — that is the name of the software package you support.

You have access to tools that let you read files, list files, and run safe read-only shell commands.
Use them to investigate the user's question thoroughly before answering.

Typical tasks:
- Diagnose why a training run crashed (read PBS logs, config, Python tracebacks)
- Explain what a config option does (read the relevant source file)
- Suggest config changes based on the user's hardware and dataset
- Check whether a job is still running (qstat) and interpret its output
- Diff configs between two experiments

Guidelines:
- Always read relevant files before speculating — the answer is usually in the logs or config.
- When reading PBS output files (*.o*), focus on the last 100 lines first.
- Suggest concrete, actionable fixes — not generic advice.
- Keep responses concise; use markdown headers and code blocks.
"""

_AGENT_TOOL_DEFS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Useful for configs (*.yml), PBS output logs (*.o*), "
            "Python tracebacks, and source code. Returns at most 400 lines from the end of the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (absolute or relative to cwd)"},
                "tail": {
                    "type": "integer",
                    "description": "Read only the last N lines (default 400). Pass 0 for the full file.",
                    "default": 400,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Useful for finding configs, logs, checkpoints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to cwd, e.g. '*.yml', 'logs/*.txt', '**/*.py'",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a read-only shell command and return stdout+stderr. "
            "Allowed commands: grep, find, tail, head, cat, ls, wc, diff, git log, git diff, git status, "
            "qstat, squeue, python -c (imports only). Destructive commands are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
]

# Commands that could cause harm — block them in the agent bash tool
_AGENT_BASH_BLOCKLIST = (
    "rm ",
    "rmdir",
    "mv ",
    "cp ",
    "> ",
    ">>",
    "tee ",
    "dd ",
    "mkfs",
    "chmod",
    "chown",
    "curl",
    "wget",
    "pip install",
    "conda install",
    "git commit",
    "git push",
    "git reset",
    "git checkout",
    "kill ",
    "pkill",
    "qdel",
    "scancel",
    "sudo",
)


def _agent_read_file(path: str, tail: int = 400) -> str:
    try:
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        if p.stat().st_size > 10 * 1024 * 1024:
            return f"File too large to read (>{10} MB): {path}"
        lines = p.read_text(errors="replace").splitlines()
        if tail and len(lines) > tail:
            skipped = len(lines) - tail
            text = "\n".join(lines[-tail:])
            return f"[… {skipped} lines omitted …]\n{text}"
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def _agent_list_files(pattern: str) -> str:
    import glob as _glob

    matches = sorted(_glob.glob(pattern, recursive=True))
    if not matches:
        return f"No files matched: {pattern}"
    return "\n".join(matches[:200])


def _agent_bash(command: str) -> str:
    import subprocess

    lower = command.lower()
    for blocked in _AGENT_BASH_BLOCKLIST:
        if blocked in lower:
            return f"Blocked: '{blocked}' is not allowed in agent bash. Use read_file or list_files instead."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (result.stdout + result.stderr).strip()
        if len(out) > 8000:
            out = out[-8000:]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 s."
    except Exception as exc:
        return f"Error: {exc}"


def _dispatch_tool(name: str, tool_input: dict) -> str:
    if name == "read_file":
        return _agent_read_file(tool_input["path"], tool_input.get("tail", 400))
    if name == "list_files":
        return _agent_list_files(tool_input["pattern"])
    if name == "bash":
        return _agent_bash(tool_input["command"])
    return f"Unknown tool: {name}"


def _agent(args: argparse.Namespace) -> None:
    """Run an agentic session: Claude reads files and runs commands to answer your question."""
    import logging

    try:
        import anthropic
    except ImportError:
        print("anthropic package required: pip install 'miles-credit[agent]'", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        msg = "ANTHROPIC_API_KEY is not set.\n"
        if _is_ncar_system():
            msg += (
                "\nOn NCAR systems (Casper/Derecho) you can use the shared Anthropic credits:\n\n"
                "  module use /glade/work/bdobbins/llms/modules\n"
                "  module load llms\n\n"
                "Add those two lines to ~/.bashrc to persist across sessions."
            )
        else:
            msg += (
                "\ncredit agent requires an Anthropic API key with active credits.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  → https://console.anthropic.com"
            )
        print(msg, file=sys.stderr)
        sys.exit(1)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    question = " ".join(args.question)
    context = _collect_run_context(args)
    user_msg = f"{context}\n\n## Task\n{question}" if context else question

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_msg}]
    max_turns = getattr(args, "max_turns", 20)

    print()
    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_AGENT_SYSTEM_PROMPT,
                tools=_AGENT_TOOL_DEFS,
                messages=messages,
            )
        except anthropic.BadRequestError as exc:
            if "credit balance is too low" in str(exc):
                print(
                    "Anthropic API key has no credits.\n"
                    "  → Add credits: https://console.anthropic.com/settings/billing\n"
                    "  → Note: Claude.ai Pro does NOT include API access.",
                    file=sys.stderr,
                )
            else:
                print(f"API error: {exc}", file=sys.stderr)
            sys.exit(1)

        # Print any text blocks
        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text, end="", flush=True)

        # Collect tool calls
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Execute tools, print what we're doing
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            if not any(b.type == "text" for b in response.content):
                # Print tool call summary when Claude didn't add narrative text
                print(f"\n[{tc.name}: {tc.input}]", flush=True)
            result = _dispatch_tool(tc.name, tc.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"\n[reached max_turns={max_turns}]", file=sys.stderr)

    print("\n")


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
def _metrics(args: argparse.Namespace) -> None:
    """Run WeatherBench2-style evaluation and optionally generate scorecard plots."""
    import subprocess

    script = os.path.join(_repo_root(), "applications", "eval_weatherbench.py")
    if not os.path.exists(script):
        print(
            "eval_weatherbench.py not found. This command requires the v2.1/weatherbench branch to be merged.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [sys.executable, script]

    if args.csv:
        cmd += ["--csv", args.csv]
    elif args.netcdf:
        cmd += ["--netcdf", args.netcdf]

    if args.era5:
        cmd += ["--era5", args.era5]
    if args.clim:
        cmd += ["--clim", args.clim]
    if args.out:
        cmd += ["--out", args.out]
    if args.lead_time_hours:
        cmd += ["--lead-time-hours", str(args.lead_time_hours)]
    if args.max_inits:
        cmd += ["--max-inits", str(args.max_inits)]
    if args.plot_dir:
        cmd += ["--plot", args.plot_dir]
    if args.label:
        cmd += ["--label", args.label]
    if args.no_refs:
        cmd += ["--no-refs"]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.verbose:
        cmd += ["-v"]

    sys.exit(subprocess.call(cmd))


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
              credit submit   --cluster casper  -c config.yml --rollout --jobs 10
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
    p.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        metavar="N",
        dest="ensemble_size",
        help="Override predict.ensemble_size from config. N=1 deterministic; N>1 saves ensemble members to NetCDF.",
    )

    # ---- rollout-ensemble ----
    p = sub.add_parser(
        "rollout-ensemble",
        help="Submit N parallel PBS rollout jobs covering all ensemble init times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Split an ensemble rollout across N parallel PBS jobs — one job per
            subset of init times.  All jobs start at once (no afterok chain).

            The number of init times and ensemble_size are read from the config's
            predict: section.  PBS settings follow the same resolution order as
            credit submit: CLI flag > config pbs: > cluster default.

            For rollout (inference), 1 GPU per job is usually sufficient on Casper.

            Examples:
              credit rollout-ensemble --cluster casper -c config.yml --jobs 10 --dry-run
              credit rollout-ensemble --cluster casper -c config.yml --jobs 10
              credit rollout-ensemble --cluster derecho -c config.yml --jobs 20 --gpus 1
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="Path to v2 YAML config")
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"], help="Target NCAR HPC cluster")
    p.add_argument("--jobs", type=int, default=10, metavar="N", help="Number of parallel PBS jobs (default: 10)")
    p.add_argument("--gpus", type=int, default=None, metavar="N", help="GPUs per job (config pbs.ngpus → 1)")
    p.add_argument(
        "--cpus", type=int, default=None, metavar="N", help="CPUs per job (config pbs.ncpus → 8 casper / 16 derecho)"
    )
    p.add_argument("--mem", default=None, help="Memory per job (config pbs.mem → 128GB casper / 128GB derecho)")
    p.add_argument(
        "--walltime", default=None, metavar="HH:MM:SS", help="Walltime per job (config pbs.walltime → 06:00:00)"
    )
    p.add_argument(
        "--account", metavar="ACCOUNT", help="PBS account code (config pbs.project → $PBS_ACCOUNT → NAML0001)"
    )
    p.add_argument("--queue", metavar="QUEUE", help="PBS queue (config pbs.queue → casper / main)")
    p.add_argument(
        "--gpu-type", dest="gpu_type", default=None, help="Casper GPU type (config pbs.gpu_type → a100_80gb)"
    )
    p.add_argument("--torchrun", default=None, metavar="PATH", help="Path to torchrun binary (default: auto-detect)")
    p.add_argument(
        "--conda-env",
        dest="conda_env",
        default=None,
        metavar="PATH",
        help="Conda env path for derecho (default: from config pbs.conda)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print PBS scripts without submitting")

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
        help="Generate and submit a PBS training or rollout job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate a PBS batch script and optionally submit it via qsub.
            Use --dry-run to inspect the script before submitting.
            Use --reload to resume from the latest checkpoint automatically.
            Use --chain N to submit N back-to-back jobs via PBS afterok dependencies.

            For rollout (ensemble or deterministic), use --rollout --jobs N to split
            init times across N parallel independent PBS jobs (no afterok chain).
            Ensemble behaviour is controlled by predict.ensemble_size in the config.

            Examples:
              credit submit --cluster casper  -c config.yml --gpus 1 --walltime 04:00:00
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 2 --dry-run
              credit submit --cluster casper  -c config.yml --gpus 4 --reload
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 1 --chain 10
              credit submit --cluster casper  -c config.yml --rollout --jobs 10
              credit submit --cluster derecho -c config.yml --rollout --jobs 20 --gpus 1
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG")
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"], help="Target NCAR HPC cluster")
    p.add_argument("--gpus", type=int, default=None, metavar="N", help="GPUs per node (config pbs.ngpus → 4)")
    p.add_argument(
        "--nodes", type=int, default=None, metavar="N", help="Number of nodes, derecho only (config pbs.nodes → 1)"
    )
    p.add_argument(
        "--cpus", type=int, default=None, metavar="N", help="CPUs per node (config pbs.ncpus → 8 casper / 64 derecho)"
    )
    p.add_argument("--mem", default=None, help="Memory per node (config pbs.mem → 128GB casper / 480GB derecho)")
    p.add_argument("--walltime", default=None, metavar="HH:MM:SS", help="Job walltime (config pbs.walltime → 12:00:00)")
    p.add_argument(
        "--account", metavar="ACCOUNT", help="PBS account code (config pbs.project → $PBS_ACCOUNT → NAML0001)"
    )
    p.add_argument("--queue", metavar="QUEUE", help="PBS queue (config pbs.queue → casper / main)")
    p.add_argument(
        "--gpu-type", dest="gpu_type", default=None, help="Casper GPU type (config pbs.gpu_type → a100_80gb)"
    )
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
        "--rollout",
        action="store_true",
        help="Submit parallel rollout jobs instead of a training job. "
        "Use with --jobs N to split init times across N independent PBS jobs. "
        "Ensemble behaviour is controlled by predict.ensemble_size in the config.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=10,
        metavar="N",
        help="Number of parallel PBS rollout jobs when using --rollout (default: 10)",
    )
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
        help="Ask the CREDIT AI assistant a question — agentic if Anthropic is available, simple chat otherwise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Ask the CREDIT AI assistant anything about training, config, or debugging.

            When ANTHROPIC_API_KEY is set, runs in AGENT mode: a multi-turn loop that
            reads your files, runs grep/tail/qstat, and iterates until it has a confident
            answer.  When Anthropic is unavailable, falls back to simple one-shot chat
            using whichever key is set (Groq, Gemini, or OpenAI).

            Provider priority for simple chat fallback (first key found wins):
              ANTHROPIC_API_KEY  → Claude Haiku       https://console.anthropic.com
              OPENAI_API_KEY     → GPT-4o             https://platform.openai.com
              GOOGLE_API_KEY     → Gemini 1.5 Pro     https://aistudio.google.com (free for NCAR)
              GROQ_API_KEY       → Llama 3 Instant    https://console.groq.com    (free tier)

            NCAR users (Casper/Derecho) — shared Anthropic credits available:
              module use /glade/work/bdobbins/llms/modules
              module load llms

            Examples:
              credit ask "why is my training loss stuck at 2.5?"
              credit ask -c config.yml "why did my training run crash?"
              credit ask -c config.yml "is my batch size too large for 0.25 degree?"
              credit ask --provider gemini "what do I do if my Derecho job hangs?"
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
        help="Force a specific LLM provider for simple chat (default: auto-detect; Anthropic agent always tried first)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=20,
        dest="max_turns",
        help="Maximum agentic turns before stopping (default: 20; only applies in agent mode)",
    )

    # ---- convert ----
    p = sub.add_parser(
        "convert",
        help="Interactively convert a v1 config to v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Convert a v1 CREDIT config to v2 format.

            Automatic changes:
              - trainer.type: era5 → era5-v2
              - data.forecast_len: +1  (v2 semantics: 1 = single step, v1 used 0)
              - data.valid_forecast_len: +1
              - data.backprop_on_timestep: shifted to 1-indexed

            Interactive prompts for new v2 features:
              - EMA (exponential moving average of weights)
              - TensorBoard logging
              - Ensemble settings (kept if detected)
              - PBS / job settings (account, conda env, nodes, walltime, ...)

            Example:
              credit convert -c old_model.yml          # saves to old_model_v2.yml
              credit convert -c old_model.yml -o new.yml
        """),
    )
    p.add_argument("-c", "--config", required=True, metavar="CONFIG", help="v1 config YAML to convert")
    p.add_argument("-o", "--output", default=None, metavar="OUTPUT", help="Output path (default: <input>_v2.yml)")

    # ---- init ----
    # ---- metrics ----
    p = sub.add_parser(
        "metrics",
        help="WeatherBench2-style evaluation: RMSE, ACC, and scorecard plots vs IFS/Pangu/GraphCast baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Run WeatherBench2-style verification on CREDIT forecast output.

            Computes RMSE and ACC vs ERA5 ground truth across lead times, optionally
            comparing against IFS, Pangu-Weather, and GraphCast reference baselines.
            Requires the v2.1/weatherbench branch (eval_weatherbench.py).

            Input modes (one required):
              --csv     Directory of per-init CSV files (fast path, already scored)
              --netcdf  Directory of forecast netCDF files (full scoring pipeline)

            Examples:
              credit metrics --netcdf /path/to/forecasts --out scores.csv
              credit metrics --csv /path/to/csv_dir --plot figures/ --label WXFormer-v2
              credit metrics --netcdf /path/to/forecasts --plot figures/ --no-refs
        """),
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", type=str, metavar="DIR", help="Directory of per-init metrics CSVs (fast path)")
    input_group.add_argument("--netcdf", type=str, metavar="DIR", help="Directory of forecast netCDFs (full path)")
    p.add_argument(
        "--era5",
        type=str,
        default=None,
        metavar="GLOB",
        help="Glob pattern for ERA5 zarr files (required for --netcdf mode)",
    )
    p.add_argument("--clim", type=str, default=None, metavar="FILE", help="ERA5 climatology netCDF for true ACC")
    p.add_argument(
        "--out", type=str, default="wb2_scores.csv", metavar="FILE", help="Output scores CSV (default: wb2_scores.csv)"
    )
    p.add_argument(
        "--lead-time-hours", type=int, default=6, dest="lead_time_hours", help="Hours per forecast step (default: 6)"
    )
    p.add_argument(
        "--max-inits",
        type=int,
        default=None,
        dest="max_inits",
        metavar="N",
        help="Limit number of init dates (for testing)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="DIR",
        dest="plot_dir",
        help="If set, generate WB2 scorecard figures in this directory",
    )
    p.add_argument("--label", type=str, default="CREDIT", help="Model label for plot legends (default: CREDIT)")
    p.add_argument(
        "--no-refs", action="store_true", dest="no_refs", help="Omit IFS/Pangu/GraphCast reference lines from plots"
    )
    p.add_argument("--workers", type=int, default=None, help="Parallel workers for --netcdf mode (default: cpu count)")
    p.add_argument("-v", "--verbose", action="store_true")

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
        "rollout-ensemble": _rollout_ensemble,
        "realtime": _realtime,
        "submit": _submit,
        "convert": _convert,
        "init": _init,
        "plot": _plot,
        "ask": _ask,
        "metrics": _metrics,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
