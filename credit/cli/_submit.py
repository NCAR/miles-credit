"""PBS job submission and training/rollout/realtime command handlers."""

import argparse
import copy
import math
import os
import re
import sys
import textwrap
import yaml

from ._common import _PBS_DEFAULTS, _SLURM_DEFAULTS, _find_torchrun, _repo_root
from ._convert import _write_reload_config

logger = __import__("logging").getLogger(__name__)


def _train(args: argparse.Namespace) -> None:
    from credit.applications.train_gen2 import main_cli

    sys.argv = ["credit-train", "-c", args.config, "--backend", args.backend]
    main_cli()


def _preprocess(args: argparse.Namespace) -> None:
    from credit.applications.preprocess import main

    sys.argv = ["credit-preprocess", "-c", args.config, "--backend", args.backend]
    main()


def _rollout(args: argparse.Namespace) -> None:
    from credit.applications.rollout_gen2 import main

    argv = ["credit-rollout", "-c", args.config, "-p", str(args.procs)]
    sys.argv = argv
    main()


def _realtime(args: argparse.Namespace) -> None:
    from credit.applications.rollout_realtime_gen2 import main

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


def _load_pbs_config(config_path: str) -> dict:
    """Return the ``pbs:`` section from a YAML config file."""
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    pbs = conf.get("pbs") or {}
    if not pbs:
        print(
            f"ERROR: config '{config_path}' is missing a required 'pbs:' section.\n"
            "Add a pbs: block — see config/example-v2026.1.0.yml for reference.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (pbs.get("conda") or pbs.get("conda_env")):
        print(
            "ERROR: pbs.conda is required but not set in the config.\n"
            "Specify the conda environment name or full path, e.g.:\n"
            "  pbs:\n"
            "    conda: /glade/u/home/$USER/.conda/envs/credit-casper",
            file=sys.stderr,
        )
        sys.exit(1)

    return pbs


def _resolve_pbs_opts(args: argparse.Namespace, pbs_cfg: dict) -> argparse.Namespace:
    """Return a copy of *args* with None fields filled from *pbs_cfg* then cluster defaults."""
    r = copy.copy(args)

    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    is_casper = args.cluster == "casper"
    d = _PBS_DEFAULTS["casper" if is_casper else "derecho"]

    r.account = _first(
        args.account, pbs_cfg.get("project") or pbs_cfg.get("account"), os.environ.get("PBS_ACCOUNT"), d["account"]
    )
    r.walltime = _first(args.walltime, pbs_cfg.get("walltime"), d["walltime"])
    r.gpus = int(_first(args.gpus, pbs_cfg.get("ngpus") or pbs_cfg.get("gpus"), d["gpus"]))
    r.nodes = int(_first(args.nodes, pbs_cfg.get("nodes"), d["nodes"]))
    r.cpus = int(_first(args.cpus, pbs_cfg.get("ncpus") or pbs_cfg.get("cpus"), d["cpus"]))
    r.mem = _first(args.mem, pbs_cfg.get("mem"), d["mem"])
    pbs_server = "casper-pbs" if is_casper else "desched1"
    raw_queue = _first(args.queue, pbs_cfg.get("queue"), d["queue"])
    r.queue = raw_queue if "@" in raw_queue else f"{raw_queue}@{pbs_server}"
    r.gpu_type = _first(args.gpu_type, pbs_cfg.get("gpu_type"), d["gpu_type"])
    r.conda_env = _first(args.conda_env, pbs_cfg.get("conda") or pbs_cfg.get("conda_env"))
    r.job_name = pbs_cfg.get("job_name", d["job_name"])
    return r


# ---------------------------------------------------------------------------
# SLURM support (mirrors the PBS helpers above)
# ---------------------------------------------------------------------------


def _load_slurm_config(config_path: str) -> dict:
    """Return the ``slurm:`` section from a YAML config, falling back to ``pbs:``."""
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    slurm = conf.get("slurm") or conf.get("pbs") or {}
    if not slurm:
        print(
            f"ERROR: config '{config_path}' is missing a required 'slurm:' (or 'pbs:') section.\n"
            "Add a slurm: block with at least a conda env and partition.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (slurm.get("conda") or slurm.get("conda_env")):
        print(
            "ERROR: slurm.conda is required but not set in the config.\n"
            "Specify the conda environment name or full path, e.g.:\n"
            "  slurm:\n"
            "    conda: /home/$USER/.conda/envs/credit",
            file=sys.stderr,
        )
        sys.exit(1)

    return slurm


def _resolve_slurm_opts(args: argparse.Namespace, slurm_cfg: dict) -> argparse.Namespace:
    """Return a copy of *args* with None fields filled from *slurm_cfg* then defaults."""
    r = copy.copy(args)

    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    d = _SLURM_DEFAULTS
    r.account = _first(
        args.account,
        slurm_cfg.get("project") or slurm_cfg.get("account"),
        os.environ.get("SLURM_ACCOUNT"),
        d["account"],
    )
    r.walltime = _first(args.walltime, slurm_cfg.get("walltime"), d["walltime"])
    r.gpus = int(_first(args.gpus, slurm_cfg.get("ngpus") or slurm_cfg.get("gpus"), d["gpus"]))
    r.nodes = int(_first(args.nodes, slurm_cfg.get("nodes"), d["nodes"]))
    r.cpus = int(_first(args.cpus, slurm_cfg.get("ncpus") or slurm_cfg.get("cpus"), d["cpus"]))
    r.mem = _first(args.mem, slurm_cfg.get("mem"), d["mem"])
    # --queue doubles as the SLURM partition selector.
    r.partition = _first(args.queue, slurm_cfg.get("partition") or slurm_cfg.get("queue"), d["partition"])
    r.queue = r.partition
    r.gpu_type = _first(args.gpu_type, slurm_cfg.get("gpu_type"), d["gpu_type"])
    r.conda_env = _first(args.conda_env, slurm_cfg.get("conda") or slurm_cfg.get("conda_env"))
    r.job_name = slurm_cfg.get("job_name", d["job_name"])
    r.modules = slurm_cfg.get("modules")
    r.env_setup = slurm_cfg.get("env_setup")
    return r


def _slurm_gres(args: argparse.Namespace) -> str:
    """Return a ``--gres=gpu:...`` value, pinned to a GPU type when requested."""
    if getattr(args, "gpu_type", None):
        return f"gpu:{args.gpu_type}:{args.gpus}"
    return f"gpu:{args.gpus}"


def _slurm_directives(args: argparse.Namespace, job_name: str, save_loc: str = None, depend_on: str = None) -> str:
    """Return the ``#SBATCH`` header block for a job."""
    lines = ["#!/bin/bash -l", f"#SBATCH --job-name={job_name}"]
    if getattr(args, "account", None):
        lines.append(f"#SBATCH --account={args.account}")
    lines += [
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --nodes={args.nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gres={_slurm_gres(args)}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.walltime}",
    ]
    if save_loc:
        logs_dir = os.path.join(os.path.expandvars(save_loc), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        lines.append(f"#SBATCH --output={logs_dir}/%x-%j.out")
    if depend_on:
        lines.append(f"#SBATCH --dependency=afterok:{depend_on}")
    return "\n".join(lines)


def _slurm_env(args: argparse.Namespace) -> str:
    """Return the ``module load`` / ``conda activate`` / extra env lines."""
    lines = ["source ~/.bashrc"]
    modules = getattr(args, "modules", None)
    if modules:
        if isinstance(modules, (list, tuple)):
            modules = " ".join(str(m) for m in modules)
        lines.append(f"module load {modules}")
    if getattr(args, "conda_env", None):
        lines.append(f"conda activate {args.conda_env}")
    env_setup = getattr(args, "env_setup", None)
    if env_setup:
        if isinstance(env_setup, str):
            env_setup = [env_setup]
        lines.extend(str(x) for x in env_setup)
    return "\n".join(lines)


def _slurm_torchrun(args: argparse.Namespace) -> str:
    """Return the torchrun path, preferring the configured conda env."""
    conda_env = getattr(args, "conda_env", None)
    if conda_env and os.path.isdir(conda_env):
        return f"{conda_env}/bin/torchrun"
    return _find_torchrun()


def _slurm_launch(args: argparse.Namespace, torchrun: str, app: str, app_args: str = "") -> str:
    """Return the torchrun launch command for single- or multi-node SLURM jobs."""
    extra = f" {app_args}" if app_args else ""
    if args.nodes == 1:
        return textwrap.dedent(f"""\
            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/{app} -c ${{CONFIG}}{extra}""")
    return textwrap.dedent(f"""\
        nodes_arr=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
        head_node="${{nodes_arr[0]}}"
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{{print $1}}')
        echo "Head node : ${{head_node_ip}}"

        srun {torchrun} \\
            --nnodes={args.nodes} \\
            --nproc-per-node=${{NGPUS}} \\
            --rdzv-id="$SLURM_JOB_ID" \\
            --rdzv-backend=c10d \\
            --rdzv-endpoint="${{head_node_ip}}:29500" \\
            ${{REPO}}/{app} -c ${{CONFIG}}{extra}""")


def _build_slurm_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    account: str = None,
    depend_on: str = None,
    save_loc: str = None,
) -> str:
    """Return a SLURM batch script string for a training job."""
    if account is not None:
        args = copy.copy(args)
        args.account = account
    directives = _slurm_directives(args, args.job_name, save_loc=save_loc, depend_on=depend_on)
    env = _slurm_env(args)
    torchrun = _slurm_torchrun(args)
    launch = _slurm_launch(args, torchrun, "credit/applications/train_gen2.py")
    return (
        f"{directives}\n\n"
        f"{env}\n\n"
        f"REPO={repo}\n"
        f"CONFIG={config}\n"
        f"NGPUS={args.gpus}\n\n"
        f'echo "Config    : ${{CONFIG}}"\n'
        f'echo "Nodes     : {args.nodes}  GPUs/node: {args.gpus}"\n'
        f'echo "Total GPUs: $(( {args.nodes} * {args.gpus} ))"\n'
        f"cd ${{REPO}}\n\n"
        f"{launch}\n"
    )


def _build_realtime_slurm_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    init_time: str,
    steps: int,
    save_loc: str = None,
) -> str:
    """Return a SLURM script that runs a single realtime forecast."""
    job_name = getattr(args, "job_name", "credit_realtime")
    directives = _slurm_directives(args, job_name, save_loc=save_loc)
    env = _slurm_env(args)
    torchrun = _slurm_torchrun(args)
    launch = _slurm_launch(
        args,
        torchrun,
        "credit/applications/rollout_realtime_gen2.py",
        app_args=f"--init-time {init_time} --steps {steps}",
    )
    return (
        f"{directives}\n\n"
        f"{env}\n\n"
        f"REPO={repo}\n"
        f"CONFIG={config}\n"
        f"NGPUS={args.gpus}\n\n"
        f'echo "Realtime forecast - init: {init_time}  steps: {steps}"\n'
        f'echo "Config  : ${{CONFIG}}"\n'
        f"cd ${{REPO}}\n\n"
        f"{launch}\n"
    )


def _build_preprocess_slurm_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    save_loc: str = None,
) -> str:
    """Return a SLURM script that runs the preprocessing / scaler-fitting job."""
    job_name = getattr(args, "job_name", "credit_preprocess")
    directives = _slurm_directives(args, job_name, save_loc=save_loc)
    env = _slurm_env(args)
    torchrun = _slurm_torchrun(args)
    launch = _slurm_launch(args, torchrun, "credit/applications/preprocess.py")
    return (
        f"{directives}\n\n"
        f"{env}\n\n"
        f"REPO={repo}\n"
        f"CONFIG={config}\n"
        f"NGPUS={args.gpus}\n\n"
        f'echo "Preprocessing - scaler fitting"\n'
        f'echo "Config  : ${{CONFIG}}"\n'
        f"cd ${{REPO}}\n\n"
        f"{launch}\n"
    )


def _build_rollout_slurm_script(
    args: argparse.Namespace, config: str, repo: str, subset: int, n_subsets: int, save_loc: str = None
) -> str:
    """Return a SLURM script for one subset of an ensemble rollout."""
    job_name = f"{args.job_name[:10]}-{subset:02d}of{n_subsets:02d}"
    directives = _slurm_directives(args, job_name, save_loc=save_loc)
    env = _slurm_env(args)
    torchrun = _slurm_torchrun(args)
    launch = _slurm_launch(args, torchrun, "credit/applications/rollout_gen2.py")
    return (
        f"{directives}\n\n"
        f"{env}\n\n"
        f"REPO={repo}\n"
        f"CONFIG={config}\n"
        f"NGPUS={args.gpus}\n\n"
        f'echo "Ensemble rollout - subset {subset} of {n_subsets}"\n'
        f'echo "Config  : ${{CONFIG}}"\n'
        f"cd ${{REPO}}\n\n"
        f"{launch}\n"
    )


def _sbatch(script: str, save_loc: str | None = None) -> str:
    """Write *script*, call ``sbatch``, and return the numeric job ID string."""
    import datetime
    import subprocess
    import tempfile

    if save_loc:
        scripts_dir = os.path.join(os.path.expandvars(save_loc), "slurm_scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        script_path = os.path.join(scripts_dir, f"submit_{ts}.sh")
        with open(script_path, "w") as f:
            f.write(script)
        delete_after = False
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        tmp.write(script)
        tmp.close()
        script_path = tmp.name
        delete_after = True

    try:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    finally:
        if delete_after:
            os.unlink(script_path)

    if result.returncode != 0:
        print(f"sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    out = result.stdout.strip()
    # sbatch prints "Submitted batch job 123456"
    match = re.search(r"(\d+)", out)
    return match.group(1) if match else out


def _scheduler_ctx(args: argparse.Namespace) -> dict:
    """Return the loader/resolver/builder/submitter bundle for the chosen scheduler."""
    if getattr(args, "scheduler", "pbs") == "slurm":
        return {
            "name": "SLURM",
            "load": _load_slurm_config,
            "resolve": _resolve_slurm_opts,
            "submit": _sbatch,
            "build_train": _build_slurm_script,
            "build_realtime": _build_realtime_slurm_script,
            "build_preprocess": _build_preprocess_slurm_script,
            "build_rollout": _build_rollout_slurm_script,
        }
    return {
        "name": "PBS",
        "load": _load_pbs_config,
        "resolve": _resolve_pbs_opts,
        "submit": _qsub,
        "build_train": _build_pbs_script,
        "build_realtime": _build_realtime_pbs_script,
        "build_preprocess": _build_preprocess_pbs_script,
        "build_rollout": _build_rollout_pbs_script,
    }


def _build_pbs_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    account: str = None,
    depend_on: str = None,
    save_loc: str = None,
) -> str:
    """Return a PBS batch script string for the given args and config path."""
    if save_loc:
        logs_dir = os.path.join(os.path.expandvars(save_loc), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    is_casper = getattr(args, "cluster", "casper") == "casper"
    d = _PBS_DEFAULTS["casper" if is_casper else "derecho"]
    _d = argparse.Namespace(torchrun=None, conda_env=None, **d)
    args = argparse.Namespace(**{**vars(_d), **{k: v for k, v in vars(args).items() if v is not None}})
    if account is not None:
        args.account = account
    depend_line = f"#PBS -W depend=afterok:{depend_on}\n" if depend_on else ""

    if args.cluster == "casper":
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {args.job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}:gpu_type={args.gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {args.account}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod
            {output_line}
            {depend_line}
            module load conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}
            TORCHRUN=$(which torchrun)

            echo "Config : ${{CONFIG}}"
            echo "Node   : $(hostname)"
            echo "GPUs   : ${{NGPUS}}"
            echo "torchrun: ${{TORCHRUN}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/credit/applications/train_gen2.py -c ${{CONFIG}}
        """)

    else:  # derecho
        nodes = args.nodes
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
            {output_line}
            {depend_line}
            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}

            total_gpus=$(( {nodes} * {args.gpus} ))
            echo "Nodes     : {nodes}"
            echo "GPUs/node : {args.gpus}"
            echo "Total GPUs: ${{total_gpus}}"
            echo "Config    : ${{CONFIG}}"
            cd ${{REPO}}
        """)

        conda_env = args.conda_env
        torchrun = f"{conda_env}/bin/torchrun" if (conda_env and os.path.isdir(conda_env)) else _find_torchrun()
        cuda_devices = ",".join(str(i) for i in range(args.gpus))

        if nodes == 1:
            launch = textwrap.dedent(f"""\
                {torchrun} \\
                    --standalone \\
                    --nnodes=1 \\
                    --nproc-per-node={args.gpus} \\
                    ${{REPO}}/credit/applications/train_gen2.py -c ${{CONFIG}}
            """)
        else:
            launch = textwrap.dedent(f"""\
                nodes_arr=( $( cat $PBS_NODEFILE ) )
                head_node="${{nodes_arr[0]}}"
                head_node_ip=$(ssh "${{head_node}}" hostname -i | awk '{{print $1}}')
                echo "Head node : ${{head_node_ip}}"
                MASTER_PORT=$(( RANDOM % 10000 + 20000 ))

                export CUDA_VISIBLE_DEVICES={cuda_devices}

                MASTER_ADDR=${{head_node_ip}} MASTER_PORT=${{MASTER_PORT}} \\
                mpiexec -n "${{total_gpus}}" --ppn {args.gpus} --cpu-bind none \\
                    python ${{REPO}}/credit/applications/train_gen2.py -c ${{CONFIG}}
            """)

        return header + launch


def _qsub(script: str, save_loc: str | None = None) -> str:
    """Write *script* to save_loc/pbs_scripts/, call qsub, and return the job ID string."""
    import datetime
    import subprocess
    import tempfile

    if save_loc:
        scripts_dir = os.path.join(os.path.expandvars(save_loc), "pbs_scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        script_path = os.path.join(scripts_dir, f"submit_{ts}.sh")
        with open(script_path, "w") as f:
            f.write(script)
        delete_after = False
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        tmp.write(script)
        tmp.close()
        script_path = tmp.name
        delete_after = True

    try:
        result = subprocess.run(["qsub", script_path], capture_output=True, text=True)
    finally:
        if delete_after:
            os.unlink(script_path)

    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout.strip()


def _compute_chain(args: argparse.Namespace) -> int:
    """Return the number of jobs to chain."""
    if args.chain is not None:
        return args.chain
    try:
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
    from credit.trainers.preflight import estimate_dataloader_memory_gib

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

    mem_est = estimate_dataloader_memory_gib(conf)
    mem_str = f"~{mem_est:.0f} GiB" if mem_est > 0 else "unknown"
    mem_warn = "  ⚠  consider reducing thread_workers / prefetch_factor" if mem_est > 24 else ""

    chain_desc = f"{n_jobs} job(s)  ({epochs} epochs ÷ {per_job} per job)" if n_jobs > 1 else "1 job (no chaining)"

    sep = "=" * 52
    logger.info(
        "\n%s\n  Job plan\n%s\n"
        "  Cluster  : %s\n"
        "  Account  : %s\n"
        "  Config   : %s\n"
        "  GPUs     : %s\n"
        "  Walltime : %s per job\n"
        "  Chain    : %s\n"
        "  DataLoader memory est. : %s%s\n"
        "%s",
        sep,
        sep,
        args.cluster,
        getattr(args, "account", "unset"),
        getattr(args, "config", "unset"),
        gpu_str,
        args.walltime,
        chain_desc,
        mem_str,
        mem_warn,
        sep,
    )


def _build_realtime_pbs_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    init_time: str,
    steps: int,
    save_loc: str = None,
) -> str:
    """Return a PBS script that runs a single realtime forecast."""
    if save_loc:
        logs_dir = os.path.join(os.path.expandvars(save_loc), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    job_name = getattr(args, "job_name", "credit_realtime")

    if args.cluster == "casper":
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}:gpu_type={args.gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {args.account}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod
            {output_line}

            module load conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}
            TORCHRUN=$(which torchrun)

            echo "Realtime forecast — init: {init_time}  steps: {steps}"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/credit/applications/rollout_realtime_gen2.py \\
                -c ${{CONFIG}} --init-time {init_time} --steps {steps}
        """)

    else:  # derecho
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
            {output_line}

            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            TORCHRUN={args.conda_env + "/bin/torchrun" if (args.conda_env and os.path.isdir(args.conda_env)) else _find_torchrun()}

            echo "Realtime forecast — init: {init_time}  steps: {steps}"
            echo "Config  : ${{CONFIG}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node={args.gpus} \\
                ${{REPO}}/credit/applications/rollout_realtime_gen2.py \\
                -c ${{CONFIG}} --init-time {init_time} --steps {steps}
        """)


def _build_preprocess_pbs_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    save_loc: str = None,
) -> str:
    """Return a PBS script that runs the preprocessing / scaler-fitting job."""
    if save_loc:
        logs_dir = os.path.join(os.path.expandvars(save_loc), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    job_name = getattr(args, "job_name", "credit_preprocess")

    if args.cluster == "casper":
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}:gpu_type={args.gpu_type}
            #PBS -l walltime={args.walltime}
            #PBS -A {args.account}
            #PBS -q {args.queue}
            #PBS -j oe
            #PBS -k eod
            {output_line}

            module load conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}
            TORCHRUN=$(which torchrun)

            echo "Preprocessing — scaler fitting"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"
            echo "GPUs    : ${{NGPUS}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/credit/applications/preprocess.py -c ${{CONFIG}}
        """)

    else:  # derecho
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
            {output_line}

            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            TORCHRUN={args.conda_env + "/bin/torchrun" if (args.conda_env and os.path.isdir(args.conda_env)) else _find_torchrun()}

            echo "Preprocessing — scaler fitting"
            echo "Config  : ${{CONFIG}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node={args.gpus} \\
                ${{REPO}}/credit/applications/preprocess.py -c ${{CONFIG}}
        """)


def _do_submit_preprocess(args: argparse.Namespace) -> None:
    """Submit a single job for preprocessing / scaler fitting."""
    repo = _repo_root()
    ctx = _scheduler_ctx(args)
    sched_cfg = ctx["load"](args.config)

    if not hasattr(args, "nodes"):
        args.nodes = None
    if not hasattr(args, "torchrun"):
        args.torchrun = None

    args = ctx["resolve"](args, sched_cfg)

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    config_abs = os.path.abspath(args.config)

    sep = "=" * 52
    logger.info(
        "\n%s\n  Preprocess job plan\n%s\n"
        "  Scheduler : %s\n"
        "  Cluster   : %s\n"
        "  Account   : %s\n"
        "  Config    : %s\n"
        "  GPUs      : %s\n"
        "  Walltime  : %s\n"
        "%s",
        sep,
        sep,
        ctx["name"],
        args.cluster,
        args.account,
        args.config,
        args.gpus,
        args.walltime,
        sep,
    )

    script = ctx["build_preprocess"](args, config_abs, repo, save_loc=save_loc)

    if args.dry_run:
        print(script)
        return

    job_id = ctx["submit"](script, save_loc=save_loc)
    logger.info("Submitted: %s", job_id)


def _do_submit_realtime(args: argparse.Namespace) -> None:
    """Submit a single job for a realtime forecast."""
    repo = _repo_root()
    ctx = _scheduler_ctx(args)
    sched_cfg = ctx["load"](args.config)

    if not hasattr(args, "nodes"):
        args.nodes = None
    if not hasattr(args, "torchrun"):
        args.torchrun = None

    args = ctx["resolve"](args, sched_cfg)

    with open(args.config) as f:
        conf = yaml.safe_load(f)
    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    config_abs = os.path.abspath(args.config)

    init_time = args.init_time
    steps = getattr(args, "steps", 40)

    sep = "=" * 52
    logger.info(
        "\n%s\n  Realtime job plan\n%s\n"
        "  Scheduler : %s\n"
        "  Cluster   : %s\n"
        "  Account   : %s\n"
        "  Config    : %s\n"
        "  Init time : %s\n"
        "  Steps     : %s\n"
        "  GPUs      : %s\n"
        "  Walltime  : %s\n"
        "%s",
        sep,
        sep,
        ctx["name"],
        args.cluster,
        args.account,
        args.config,
        init_time,
        steps,
        args.gpus,
        args.walltime,
        sep,
    )

    script = ctx["build_realtime"](args, config_abs, repo, init_time, steps, save_loc=save_loc)

    if args.dry_run:
        print(script)
        return

    job_id = ctx["submit"](script, save_loc=save_loc)
    logger.info("Submitted: %s", job_id)


def _submit(args: argparse.Namespace) -> None:
    """Generate and optionally submit PBS batch scripts, with optional chaining."""
    mode = getattr(args, "submit_mode", "train")
    if mode == "preprocess":
        _do_submit_preprocess(args)
        return
    if mode == "rollout":
        _do_submit_rollout(args)
        return
    if mode == "realtime":
        _do_submit_realtime(args)
        return

    repo = _repo_root()
    ctx = _scheduler_ctx(args)
    sched_cfg = ctx["load"](args.config)
    args = ctx["resolve"](args, sched_cfg)
    build = ctx["build_train"]
    submit = ctx["submit"]
    with open(args.config) as f:
        _full_conf = yaml.safe_load(f)
    save_loc = os.path.expandvars(_full_conf.get("save_loc", "."))
    n_jobs = _compute_chain(args)

    _print_job_plan(args, n_jobs)

    first_config = os.path.abspath(args.config)
    if args.reload:
        first_config = _write_reload_config(first_config)
        logger.info("Reload config written: %s", first_config)

    reload_config = _write_reload_config(os.path.abspath(args.config)) if n_jobs > 1 else None

    if args.dry_run:
        script = build(args, first_config, repo, depend_on=None, save_loc=save_loc)
        print(f"# --- Job 1/{n_jobs} ---")
        print(script)
        if n_jobs > 1:
            script2 = build(args, reload_config, repo, depend_on="<job_1_id>", save_loc=save_loc)
            print(f"# --- Jobs 2..{n_jobs}/{n_jobs} (afterok chained, reload config) ---")
            print(script2)
        return

    script = build(args, first_config, repo, depend_on=None, save_loc=save_loc)
    job_id = submit(script, save_loc=save_loc)
    logger.info("[1/%d] %s  %s", n_jobs, job_id, first_config)

    for i in range(2, n_jobs + 1):
        script = build(args, reload_config, repo, depend_on=job_id, save_loc=save_loc)
        job_id = submit(script, save_loc=save_loc)
        logger.info("[%d/%d] %s  afterok  (reload)", i, n_jobs, job_id)


def _build_rollout_pbs_script(
    args: argparse.Namespace, config: str, repo: str, subset: int, n_subsets: int, save_loc: str = None
) -> str:
    """Return a PBS script for one subset of an ensemble rollout."""
    job_name = f"{args.job_name[:10]}-{subset:02d}of{n_subsets:02d}"
    if save_loc:
        logs_dir = os.path.join(os.path.expandvars(save_loc), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

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
            {output_line}

            module load conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            NGPUS={args.gpus}

            echo "Ensemble rollout — subset {subset} of {n_subsets}"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"
            echo "GPUs    : ${{NGPUS}}"

            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}} \\
                ${{REPO}}/credit/applications/rollout_gen2.py \\
                -c ${{CONFIG}}
                # -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}  # rollout_gen2.py does not support these flags yet
        """)

    else:  # derecho
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
            {output_line}

            module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
                        cuda/12.3.2 conda/latest

            conda activate {args.conda_env}

            REPO={repo}
            CONFIG={config}
            TORCHRUN={args.conda_env + "/bin/torchrun" if (args.conda_env and os.path.isdir(args.conda_env)) else _find_torchrun()}

            echo "Ensemble rollout — subset {subset} of {n_subsets}"
            echo "Config  : ${{CONFIG}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node={args.gpus} \\
                ${{REPO}}/credit/applications/rollout_gen2.py \\
                -c ${{CONFIG}}
                # -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}  # rollout_gen2.py does not support these flags yet
        """)


def _print_ensemble_rollout_plan(args: argparse.Namespace, n_jobs: int, n_forecasts, ensemble_size) -> None:
    """Print a human-readable summary of an ensemble rollout submission."""
    if isinstance(n_forecasts, int):
        per_job = -(-n_forecasts // n_jobs)  # ceiling division
        total_runs = n_forecasts * ensemble_size if isinstance(ensemble_size, int) else "?"
    else:
        per_job = "?"
        total_runs = "?"

    sep = "=" * 56
    logger.info(
        "\n%s\n  Ensemble rollout plan\n%s\n"
        "  Cluster        : %s\n"
        "  Account        : %s\n"
        "  Config         : %s\n"
        "  Init times     : %s  (%s per job)\n"
        "  Ensemble size  : %s  →  %s total forecasts\n"
        "  Parallel jobs  : %s  (all start at once, no dependencies)\n"
        "  GPUs per job   : %s\n"
        "  Walltime/job   : %s\n"
        "%s",
        sep,
        sep,
        args.cluster,
        args.account,
        args.config,
        n_forecasts,
        per_job,
        ensemble_size,
        total_runs,
        n_jobs,
        args.gpus,
        args.walltime,
        sep,
    )


def _rollout_ensemble(args: argparse.Namespace) -> None:
    """Deprecated: use ``credit submit --mode rollout`` instead."""
    print(
        "WARNING: credit rollout-ensemble is deprecated.\n"
        "Use instead:\n"
        "  credit submit --cluster <casper|derecho> --mode rollout --jobs N -c config.yml\n",
        file=sys.stderr,
    )
    args.submit_mode = "rollout"
    if not hasattr(args, "reload"):
        args.reload = False
    if not hasattr(args, "chain"):
        args.chain = None
    if not hasattr(args, "nodes"):
        args.nodes = None
    _submit(args)


def _do_submit_rollout(args: argparse.Namespace) -> None:
    """Submit N parallel rollout jobs to cover all init times."""
    repo = _repo_root()
    ctx = _scheduler_ctx(args)
    sched_cfg = ctx["load"](args.config)

    is_casper = args.cluster == "casper"
    _rollout_defaults = {
        "ngpus": 1,
        "ncpus": 8 if is_casper else 16,
        "mem": "128GB",
        "walltime": "06:00:00",
    }
    merged_cfg = {**_rollout_defaults, **{k: v for k, v in sched_cfg.items() if v is not None}}

    if not hasattr(args, "nodes"):
        args.nodes = None
    if not hasattr(args, "torchrun"):
        args.torchrun = None

    args = ctx["resolve"](args, merged_cfg)

    n_jobs = args.jobs

    conf = {}
    try:
        with open(args.config) as f:
            conf = yaml.safe_load(f)

        if "inference" in conf:
            inf_conf = conf["inference"]
            if inf_conf.get("run_mode", "batch") == "single":
                n_forecasts = 1
            else:
                from credit.trainers.rollout_utils import batch_init_times

                n_forecasts = len(batch_init_times(inf_conf["batch_forecast"]))
            ensemble_size = inf_conf.get("ensemble_size", 1)
        else:
            from credit.forecast import load_forecasts

            n_forecasts = len(load_forecasts(conf))
            ensemble_size = conf.get("predict", {}).get("ensemble_size", 1)
    except Exception:
        n_forecasts = "?"
        ensemble_size = "?"

    _print_ensemble_rollout_plan(args, n_jobs, n_forecasts, ensemble_size)

    config_abs = os.path.abspath(args.config)
    rollout_save_loc = os.path.expandvars(conf.get("save_loc", "."))

    if args.dry_run:
        for i in range(1, n_jobs + 1):
            print(f"# {'=' * 50}")
            print(f"# Job {i}/{n_jobs}  (subset {i} of {n_jobs})")
            print(f"# {'=' * 50}")
            print(ctx["build_rollout"](args, config_abs, repo, i, n_jobs, save_loc=rollout_save_loc))
        return

    job_ids = []
    for i in range(1, n_jobs + 1):
        script = ctx["build_rollout"](args, config_abs, repo, i, n_jobs, save_loc=rollout_save_loc)
        job_id = ctx["submit"](script, save_loc=rollout_save_loc)
        job_ids.append(job_id)
        logger.info("[%2d/%d] %s", i, n_jobs, job_id)

    save_forecast = conf.get("inference", {}).get("save_forecast") or conf.get("predict", {}).get(
        "save_forecast", "<save_forecast in config>"
    )
    monitor = "squeue -u $USER" if ctx["name"] == "SLURM" else "qstat -u $USER"
    logger.info(
        "\nSubmitted %d parallel rollout jobs.\nOutput will be written to: %s\nMonitor with:\n  %s",
        n_jobs,
        save_forecast,
        monitor,
    )
