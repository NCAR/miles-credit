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


def _submit(args: argparse.Namespace) -> None:
    """Generate and optionally submit a PBS batch script."""
    import subprocess
    import tempfile

    repo = _repo_root()
    config = os.path.abspath(args.config)
    account = args.account or os.environ.get("PBS_ACCOUNT", "NAML0001")

    if args.cluster == "casper":
        cpus = args.cpus or 8
        mem = args.mem or "128GB"
        gpu_type = args.gpu_type or "a100_80gb"
        queue = args.queue or "casper"
        # Try to find torchrun from the active conda environment first
        torchrun = args.torchrun or _find_torchrun()

        script = textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N credit_v2
            #PBS -l select=1:ncpus={cpus}:ngpus={args.gpus}:mem={mem}:gpu_type={gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {account}
            #PBS -q {queue}
            #PBS -j oe
            #PBS -k eod

            # Usage: credit submit --cluster casper -c {config} --gpus {args.gpus}

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

        script = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {account}
            #PBS -N credit_v2
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={cpus}:ngpus={args.gpus}:mem={mem}
            #PBS -q {queue}
            #PBS -j oe
            #PBS -k eod
            #PBS -r n

            # Usage: credit submit --cluster derecho -c {config} --gpus {args.gpus} --nodes {nodes}

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

    if args.dry_run:
        print(script)
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        result = subprocess.run(["qsub", script_path], capture_output=True, text=True)
    finally:
        os.unlink(script_path)

    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f"Submitted job: {job_id}")
        print(f"  Cluster : {args.cluster}")
        print(f"  Config  : {config}")
        print(f"  GPUs    : {args.gpus}" + (f"  x  {args.nodes} nodes" if args.cluster == "derecho" else ""))
    else:
        print(f"qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


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
    print("  1. Edit 'save_loc' to your scratch directory")
    print("  2. Verify data paths under 'data.source'")
    print(f"  3. credit train -c {output}")


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

            Examples:
              credit submit --cluster casper  -c config.yml --gpus 1 --walltime 04:00:00
              credit submit --cluster derecho -c config.yml --gpus 4 --nodes 2 --dry-run
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
