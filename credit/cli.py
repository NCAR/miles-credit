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
        "credit-rollout", "-c", args.config,
        "-m", args.mode,
        "-cpus", str(args.procs),
    ]
    main()


def _realtime(args: argparse.Namespace) -> None:
    from credit.applications.rollout_realtime_v2 import main
    argv = [
        "credit-realtime", "-c", args.config,
        "--init-time", args.init_time,
        "--steps", str(args.steps),
        "-m", args.mode,
        "-p", str(args.procs),
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


def _build_pbs_script(args: argparse.Namespace, config: str, repo: str,
                      account: str, depend_on: str = None) -> str:
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

        return textwrap.dedent(f"""\
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

            nodes=( $( cat $PBS_NODEFILE ) )
            head_node="${{nodes[0]}}"
            head_node_ip=$(ssh "${{head_node}}" hostname -i | awk '{{print $1}}')
            total_gpus=$(( {nodes} * {args.gpus} ))

            echo "Nodes     : {nodes}"
            echo "GPUs/node : {args.gpus}"
            echo "Total GPUs: ${{total_gpus}}"
            echo "Config    : ${{CONFIG}}"
            echo "Head node : ${{head_node_ip}}"

            cd ${{REPO}}
            RDZV_PORT=$(( RANDOM % 10000 + 20000 ))

            mpiexec -n "${{total_gpus}}" --ppn {args.gpus} --cpu-bind none \\
                torchrun \\
                    --nnodes={nodes} \\
                    --nproc-per-node={args.gpus} \\
                    --rdzv-backend=c10d \\
                    --rdzv-endpoint="${{head_node_ip}}:${{RDZV_PORT}}" \\
                ${{REPO}}/applications/train_v2.py -c ${{CONFIG}}
        """)


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


def _submit(args: argparse.Namespace) -> None:
    """Generate and optionally submit PBS batch scripts, with optional chaining."""
    repo = _repo_root()
    account = args.account or os.environ.get("PBS_ACCOUNT", "NAML0001")
    n_jobs = args.chain  # 1 = single job (default), N = chain of N jobs

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

    print(f"\n  Cluster : {args.cluster}")
    print(f"  Config  : {args.config}")
    print(f"  GPUs    : {args.gpus}" + (f" x {args.nodes} nodes" if args.cluster == "derecho" else ""))


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
        ("0.25deg", "wxformer_v2"):   "config/wxformer_025deg_6hr_v2.yml",
        ("0.25deg", "crossformer"):   "config/wxformer_025deg_6hr_v2.yml",
        ("1deg",    "wxformer_v2"):   "config/wxformer_1dg_6hr_v2.yml",
        ("1deg",    "crossformer"):   "config/wxformer_1dg_6hr_v2.yml",
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
    p.add_argument("-c", "--config", required=True, metavar="CONFIG",
                   help="Path to YAML training config")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo", "mpi"],
                   help="Distributed backend (default: nccl)")

    # ---- rollout ----
    p = sub.add_parser("rollout", help="Batch forecast rollout to NetCDF")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG",
                   help="Path to YAML config")
    p.add_argument("-m", "--mode", default="none",
                   help="Distributed mode: none | ddp | fsdp (default: none)")
    p.add_argument("-p", "--procs", type=int, default=4,
                   help="CPU workers for async NetCDF save (default: 4)")

    # ---- realtime ----
    p = sub.add_parser("realtime", help="Operational realtime forecast (single init time)")
    p.add_argument("-c", "--config", required=True, metavar="CONFIG",
                   help="Path to YAML config")
    p.add_argument("--init-time", required=True, metavar="YYYY-MM-DDTHH",
                   help="Forecast initialisation time, e.g. 2024-01-15T00")
    p.add_argument("--steps", type=int, default=40,
                   help="Number of autoregressive forecast steps (default: 40)")
    p.add_argument("--save-dir", metavar="DIR",
                   help="Override output directory from config")
    p.add_argument("-m", "--mode", default="none",
                   help="Distributed mode: none | ddp | fsdp (default: none)")
    p.add_argument("-p", "--procs", type=int, default=4,
                   help="CPU workers for async NetCDF save (default: 4)")

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
    p.add_argument("--cluster", required=True, choices=["casper", "derecho"],
                   help="Target NCAR HPC cluster")
    p.add_argument("--gpus", type=int, default=4, metavar="N",
                   help="GPUs per node (default: 4)")
    p.add_argument("--nodes", type=int, default=1, metavar="N",
                   help="Number of nodes, derecho only (default: 1)")
    p.add_argument("--cpus", type=int, default=None, metavar="N",
                   help="CPUs per node (default: 8 casper / 64 derecho)")
    p.add_argument("--mem", default=None,
                   help="Memory per node (default: 128GB casper / 480GB derecho)")
    p.add_argument("--walltime", default="12:00:00", metavar="HH:MM:SS",
                   help="Job walltime (default: 12:00:00)")
    p.add_argument("--account", metavar="ACCOUNT",
                   help="PBS account code (default: $PBS_ACCOUNT or NAML0001)")
    p.add_argument("--queue", metavar="QUEUE",
                   help="PBS queue (default: casper / main)")
    p.add_argument("--gpu-type", dest="gpu_type", default=None,
                   help="Casper GPU type (default: a100_80gb)")
    p.add_argument("--torchrun", default=None, metavar="PATH",
                   help="Path to torchrun binary (default: auto-detect from PATH)")
    p.add_argument("--conda-env", dest="conda_env", default=None, metavar="PATH",
                   help="Conda environment path for derecho (default: credit-derecho-torch28-nccl221)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the PBS script without submitting")
    p.add_argument("--reload", action="store_true",
                   help="Resume from checkpoint: patch load_weights/optimizer/scaler/"
                        "scheduler/reload_epoch in the config and submit the reload job")
    p.add_argument("--chain", type=int, default=1, metavar="N",
                   help="Submit N jobs in sequence using PBS afterok dependencies. "
                        "Job 1 uses the base config (or --reload config); jobs 2..N "
                        "are automatic reload jobs. Example: --chain 10 submits 10 "
                        "back-to-back jobs covering ~10x walltime of training.")

    # ---- init ----
    p = sub.add_parser("init", help="Generate a starter config from a built-in template")
    p.add_argument("--grid", choices=["0.25deg", "1deg"], default="0.25deg",
                   help="Horizontal grid resolution (default: 0.25deg)")
    p.add_argument("--model", choices=["crossformer", "wxformer_v2"], default="wxformer_v2",
                   help="Model architecture (default: wxformer_v2)")
    p.add_argument("-o", "--output", default="config.yml", metavar="FILE",
                   help="Output file path (default: config.yml)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _setup_logging()

    dispatch = {
        "train":    _train,
        "rollout":  _rollout,
        "realtime": _realtime,
        "submit":   _submit,
        "init":     _init,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
