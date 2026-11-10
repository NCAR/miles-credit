"""PBS job submission and training/rollout/realtime command handlers."""

import argparse
import copy
import hashlib
import math
import os
import re
import shlex
import sys
import textwrap

import yaml

from ._common import (
    _PBS_DEFAULTS,
    _SLURM_CLUSTER_DEFAULTS,
    _SLURM_DEFAULTS,
    _repo_root,
    _resolve_torchrun,
    auto_mem,
    casper_mem,
    queue_cluster_error,
    torchrun_args_error,
)
from ._convert import _write_reload_config
from credit import runs
from credit.runs import SchedulerError, submit_job

logger = __import__("logging").getLogger(__name__)


def _train(args: argparse.Namespace) -> None:
    print("Starting credit train: loading imports ...", file=sys.stderr, flush=True)
    from credit.applications.train_gen2 import main_cli

    sys.argv = ["credit-train", "-c", args.config, "--backend", args.backend]
    main_cli()


def _preprocess(args: argparse.Namespace) -> None:
    print("Starting credit preprocess: loading imports ...", file=sys.stderr, flush=True)
    from credit.applications.preprocess import main

    sys.argv = ["credit-preprocess", "-c", args.config, "--backend", args.backend]
    main()


def _rollout(args: argparse.Namespace) -> None:
    print("Starting credit rollout: loading imports ...", file=sys.stderr, flush=True)
    from credit.applications.rollout_gen2 import main

    argv = ["credit-rollout", "-c", args.config, "-p", str(args.procs)]
    sys.argv = argv
    main()


def _realtime(args: argparse.Namespace) -> None:
    print("Starting credit realtime: loading imports ...", file=sys.stderr, flush=True)
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


class SubmitError(Exception):
    """A submission problem the user must fix (bad config section, queue/cluster mismatch, ...).

    The library paths used by ``credit.remote`` raise it; the CLI wrapper
    :func:`_resolve_pbs_opts` prints the message and exits 1, as it always has.
    *field* names the plan option at fault (``walltime``, or an unknown key), when there is one.
    """

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


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
    try:
        return _resolve_pbs(args, pbs_cfg)
    except SubmitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _resolve_pbs(args: argparse.Namespace, pbs_cfg: dict) -> argparse.Namespace:
    """Like :func:`_resolve_pbs_opts`, but raises :class:`SubmitError` instead of exiting."""
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
    pbs_server = "casper-pbs" if is_casper else "desched1"
    raw_queue = _first(args.queue, pbs_cfg.get("queue"), d["queue"])
    # A pbs block written for the other machine (Derecho's 'main' on Casper, say)
    # generates a script that qsub only rejects at submission time — catch it here.
    err = queue_cluster_error(raw_queue, args.cluster)
    if err:
        raise SubmitError(err)
    r.queue = raw_queue if "@" in raw_queue else f"{raw_queue}@{pbs_server}"
    r.gpu_type = _first(args.gpu_type, pbs_cfg.get("gpu_type"), d["gpu_type"])
    # On Casper the memory ask scales with the fraction of the node's GPUs in use
    # (64GB for one GPU up to the whole node's memory for all of them); an explicit
    # --mem / pbs.mem always wins.  ``auto`` (from either) sizes it from the config instead.
    fallback_mem = casper_mem(r.gpus, r.gpu_type) if is_casper else d["mem"]
    cli_mem = None if _is_auto(args.mem) else args.mem
    pbs_mem = None if _is_auto(pbs_cfg.get("mem")) else pbs_cfg.get("mem")
    if cli_mem is None and (_is_auto(args.mem) or _is_auto(pbs_cfg.get("mem"))):
        # Rollout / realtime (and an inestimable config) keep the configured or default request.
        pbs_mem = _auto_mem_request(args, r) or pbs_mem
    r.mem = _first(cli_mem, pbs_mem, fallback_mem)
    r.conda_env = _first(args.conda_env, pbs_cfg.get("conda") or pbs_cfg.get("conda_env"))
    r.torchrun_args = _resolve_torchrun_args(args, pbs_cfg, "pbs")
    r.job_name = pbs_cfg.get("job_name", d["job_name"])
    # Multi-node Derecho jobs under mpiexec run python directly, with no torchrun to take the flags.
    if not is_casper and r.nodes > 1 and r.torchrun_args and not _use_pbsdsh(r):
        logger.warning(
            "torchrun_args (%s) is ignored: multi-node Derecho jobs launch with mpiexec, not torchrun. "
            "Use --launcher pbsdsh to apply it.",
            r.torchrun_args,
        )
    return r


def _is_auto(mem) -> bool:
    return str(mem).strip().lower() == "auto"


def _auto_mem_request(args: argparse.Namespace, r: argparse.Namespace) -> str | None:
    """Resolve ``pbs.mem: auto`` from the config (train / preprocess); None means use the default."""
    mode = getattr(args, "submit_mode", "train")
    if mode not in ("train", "preprocess"):
        return None
    try:
        with open(args.config) as f:
            conf = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001 — an unreadable config is reported elsewhere
        return None
    # Preprocess only fits scalers on the training data, so it has no validation loader.
    estimate = auto_mem(conf, args.cluster, r.gpus, r.gpu_type, include_validation=mode == "train")
    if estimate is None:
        logger.warning("mem: auto — nothing to estimate memory from in the config; using the default request.")
        return None
    if estimate.exceeds_node:
        logger.warning(
            "Memory (mem: auto): %s = %s. The job likely needs more than one node offers; "
            "reduce trainer.train_batch_size, thread_workers or prefetch_factor, or use fewer levels.",
            estimate.mem,
            estimate.explanation,
        )
    else:
        logger.info("Memory (mem: auto): %s = %s.", estimate.mem, estimate.explanation)
    return estimate.mem


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


def _logs_dir(save_loc: str) -> str:
    """Return ``<save_loc>/logs``, where every batch script sends its output.

    Script builders only name it; it is created at submit time so that
    building a script (``--dry-run``, ``credit.remote plan``) touches no files.
    """
    return os.path.join(os.path.expandvars(save_loc), "logs")


def _as_list(val) -> list:
    """Normalize a str / list / None env-setup value to a list of shell lines."""
    if not val:
        return []
    if isinstance(val, str):
        return [val]
    return list(val)


def _resolve_torchrun_args(args: argparse.Namespace, sched_cfg: dict, block: str):
    """Return ``--torchrun-args`` if given, else the config block's ``torchrun_args``.

    Raises :class:`SubmitError` when the config value is neither a string nor a list of strings.
    """
    val = getattr(args, "torchrun_args", None)
    if val is None:
        val = sched_cfg.get("torchrun_args")
    err = torchrun_args_error(val)
    if err:
        raise SubmitError(f"{block}.{err}")
    return val


def _torchrun_extra(args: argparse.Namespace) -> str:
    """Return the extra ``torchrun`` flags (``torchrun_args``) as a leading-space string, or ``""``.

    ``torchrun_args`` comes from ``--torchrun-args`` or the ``slurm:`` / ``pbs:`` config block.
    A string is inserted verbatim, so it can hold several flags (``"--numa-binding=node
    --max-restarts=0"``). A list is shell-quoted, so each element stays one argument even
    with spaces or shell metacharacters. The Derecho multi-node ``mpiexec`` path runs
    python directly and ignores it.
    """
    val = getattr(args, "torchrun_args", None)
    if not val:
        return ""
    flags = val.strip() if isinstance(val, str) else shlex.join(val)
    return f" {flags}" if flags else ""


def _resolve_slurm_opts(args: argparse.Namespace, slurm_cfg: dict) -> argparse.Namespace:
    """Return a copy of *args* with None fields filled from *slurm_cfg* then defaults."""
    r = copy.copy(args)

    def _first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    # Layer any per-cluster overrides (e.g. Perlmutter) on top of the generic defaults.
    d = {**_SLURM_DEFAULTS, **_SLURM_CLUSTER_DEFAULTS.get(getattr(args, "cluster", ""), {})}
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
    # ``auto`` is PBS-only; on SLURM it means the site default.
    slurm_mem = slurm_cfg.get("mem")
    r.mem = _first(*(None if _is_auto(m) else m for m in (args.mem, slurm_mem)), d["mem"])
    # --queue doubles as the SLURM partition selector.
    r.partition = _first(args.queue, slurm_cfg.get("partition") or slurm_cfg.get("queue"), d["partition"])
    r.queue = r.partition
    r.constraint = _first(getattr(args, "constraint", None), slurm_cfg.get("constraint"), d["constraint"])
    r.qos = _first(getattr(args, "qos", None), slurm_cfg.get("qos"), d["qos"])
    r.gpu_type = _first(args.gpu_type, slurm_cfg.get("gpu_type"), d["gpu_type"])
    r.conda_env = _first(args.conda_env, slurm_cfg.get("conda") or slurm_cfg.get("conda_env"))
    r.torchrun_args = _resolve_torchrun_args(args, slurm_cfg, "slurm")
    r.job_name = slurm_cfg.get("job_name", d["job_name"])
    # ``modules`` falls back to the per-cluster default (e.g. Perlmutter's NCCL
    # module) when the config does not set it.  ``env_setup`` *extends* the
    # per-cluster default (e.g. Perlmutter's NCCL/libfabric exports) so a config
    # can add its own lines (NCCL_DEBUG, etc.) without dropping the defaults.
    r.modules = _first(slurm_cfg.get("modules"), d.get("modules"))
    r.env_setup = _as_list(d.get("env_setup")) + _as_list(slurm_cfg.get("env_setup")) or None

    # Perlmutter (NERSC) GPU allocations must charge a ``_g`` account.
    if getattr(args, "cluster", "") == "perlmutter" and r.account and not r.account.endswith("_g"):
        logger.info("Perlmutter GPU jobs require a *_g account; charging %s_g", r.account)
        r.account = f"{r.account}_g"
    return r


def _slurm_gres(args: argparse.Namespace) -> str:
    """Return a ``--gres=gpu:...`` value, pinned to a GPU type when requested."""
    if getattr(args, "gpu_type", None):
        return f"gpu:{args.gpu_type}:{args.gpus}"
    return f"gpu:{args.gpus}"


def _slurm_gpu_lines(args: argparse.Namespace) -> list:
    """Return the ``#SBATCH`` lines that request GPUs and any constraint/qos.

    Sites that set a ``constraint`` (e.g. Perlmutter's ``-C gpu``) reject
    ``--gres=gpu:N`` and must request GPUs with ``--gpus-per-node``; generic
    SLURM sites fall back to the portable ``--gres`` form.
    """
    lines = []
    constraint = getattr(args, "constraint", None)
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    qos = getattr(args, "qos", None)
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if constraint:
        lines.append(f"#SBATCH --gpus-per-node={args.gpus}")
    else:
        lines.append(f"#SBATCH --gres={_slurm_gres(args)}")
    return lines


def _slurm_directives(args: argparse.Namespace, job_name: str, save_loc: str = None, depend_on: str = None) -> str:
    """Return the ``#SBATCH`` header block for a job."""
    lines = ["#!/bin/bash -l", f"#SBATCH --job-name={job_name}"]
    if getattr(args, "account", None):
        lines.append(f"#SBATCH --account={args.account}")
    if getattr(args, "partition", None):
        lines.append(f"#SBATCH --partition={args.partition}")
    lines += [
        f"#SBATCH --nodes={args.nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]
    lines += _slurm_gpu_lines(args)
    lines.append(f"#SBATCH --cpus-per-task={args.cpus}")
    if getattr(args, "mem", None):
        lines.append(f"#SBATCH --mem={args.mem}")
    lines.append(f"#SBATCH --time={args.walltime}")
    if save_loc:
        logs_dir = _logs_dir(save_loc)
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
    """Return the torchrun path: ``--torchrun`` if given, else the configured conda env's."""
    return getattr(args, "torchrun", None) or _resolve_torchrun(getattr(args, "conda_env", None))


def _slurm_launch(args: argparse.Namespace, torchrun: str, app: str, app_args: str = "") -> str:
    """Return the torchrun launch command for single- or multi-node SLURM jobs."""
    extra = f" {app_args}" if app_args else ""
    tr_extra = _torchrun_extra(args)
    if args.nodes == 1:
        return textwrap.dedent(f"""\
            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}}{tr_extra} \\
                ${{REPO}}/{app} -c ${{CONFIG}}{extra}""")
    return textwrap.dedent(f"""\
        nodes_arr=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
        head_node="${{nodes_arr[0]}}"
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{{print $1}}')
        echo "Head node : ${{head_node_ip}}"

        srun {torchrun} \\
            --nnodes={args.nodes} \\
            --nproc-per-node=${{NGPUS}}{tr_extra} \\
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


def _rollout_job_name(job_name: str, subset: int, n_subsets: int) -> str:
    """Return the scheduler job name of one rollout subset job (``<job_name[:10]>-01of04``)."""
    return f"{job_name[:10]}-{subset:02d}of{n_subsets:02d}"


def _build_rollout_slurm_script(
    args: argparse.Namespace, config: str, repo: str, subset: int, n_subsets: int, save_loc: str = None
) -> str:
    """Return a SLURM script for one subset of an ensemble rollout."""
    job_name = _rollout_job_name(args.job_name, subset, n_subsets)
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


def _write_submit_script(script: str, save_loc: str | None, subdir: str) -> tuple[str, bool]:
    """Write *script* under ``<save_loc>/<subdir>/`` (or a temp file) and return ``(path, delete_after)``."""
    import datetime
    import tempfile

    if save_loc:
        os.makedirs(_logs_dir(save_loc), exist_ok=True)
        scripts_dir = os.path.join(os.path.expandvars(save_loc), subdir)
        os.makedirs(scripts_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        script_path = os.path.join(scripts_dir, f"submit_{ts}.sh")
        with open(script_path, "w") as f:
            f.write(script)
        return script_path, False
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    tmp.write(script)
    tmp.close()
    return tmp.name, True


def _sbatch(script: str, save_loc: str | None = None) -> str:
    """Write *script* to save_loc/slurm_scripts/, call ``sbatch``, and return the numeric job ID string.

    Raises :class:`credit.runs.SchedulerError`; ``credit submit`` prints it and exits 1.
    """
    script_path, delete_after = _write_submit_script(script, save_loc, "slurm_scripts")
    try:
        return submit_job("slurm", script_path)
    finally:
        if delete_after:
            os.unlink(script_path)


def _scheduler_ctx(args: argparse.Namespace) -> dict:
    """Return the loader/resolver/builder/submitter bundle for the chosen scheduler."""
    if getattr(args, "scheduler", "pbs") == "slurm":
        return {
            "name": "SLURM",
            "load": _load_slurm_config,
            "submit": _sbatch,
            "build_train": _build_slurm_script,
            "build_realtime": _build_realtime_slurm_script,
            "build_preprocess": _build_preprocess_slurm_script,
            "build_rollout": _build_rollout_slurm_script,
        }
    return {
        "name": "PBS",
        "load": _load_pbs_config,
        "submit": _qsub,
        "build_train": _build_pbs_script,
        "build_realtime": _build_realtime_pbs_script,
        "build_preprocess": _build_preprocess_pbs_script,
        "build_rollout": _build_rollout_pbs_script,
    }


def _derecho_nodes(args: argparse.Namespace) -> int:
    """Return the requested node count, defaulting to 1 when unset."""
    return int(getattr(args, "nodes", 1) or 1)


def _gpu_type_select(args: argparse.Namespace) -> str:
    """Return the ``:gpu_type=<x>`` fragment of a Casper ``select`` statement.

    Casper hosts a mix of GPUs (V100 / A100 / H100 / L40).  Omitting ``gpu_type``
    lets PBS schedule the job on *any* NVIDIA GPGPU, which usually queues much
    faster, so an unset (or ``any``/``none``) gpu_type drops the fragment
    entirely instead of pinning the job to one model.
    """
    gpu_type = getattr(args, "gpu_type", None)
    if not gpu_type or str(gpu_type).lower() in ("any", "none"):
        return ""
    return f":gpu_type={gpu_type}"


def _use_pbsdsh(args: argparse.Namespace) -> bool:
    """True when this job should launch via pbsdsh rather than mpiexec.

    pbsdsh only changes the *multi-node* path; single-node jobs always use
    ``torchrun --standalone`` regardless of ``--launcher``.
    """
    return _derecho_nodes(args) > 1 and (getattr(args, "launcher", "mpiexec") or "mpiexec") == "pbsdsh"


def _derecho_launch(args: argparse.Namespace, repo: str, app_relpath: str, app_args: str = "") -> str:
    """Return the derecho launch block for ``credit/applications/<app_relpath>``.

    One node → ``torchrun --standalone``; many nodes → ``mpiexec`` across
    ``nodes × gpus`` ranks with the head node's IP as the rendezvous master
    (``get_rank_info`` picks up rank/size from MPI in that case). Shared by every
    ``--mode``'s derecho builder so multi-node support stays uniform; the pbsdsh
    alternative is selected by the caller via :func:`_derecho_pbsdsh_script`.

    Expects the caller's header to define ``REPO`` and ``CONFIG``.
    """
    nodes = _derecho_nodes(args)
    app = f"${{REPO}}/credit/applications/{app_relpath}"
    extra = f" {app_args}" if app_args else ""
    torchrun = _resolve_torchrun(getattr(args, "conda_env", None))

    if nodes == 1:
        return textwrap.dedent(f"""\
            {torchrun} \\
                --standalone \\
                --nnodes=1 \\
                --nproc-per-node={args.gpus}{_torchrun_extra(args)} \\
                {app} -c ${{CONFIG}}{extra}
        """)

    cuda_devices = ",".join(str(i) for i in range(args.gpus))
    return textwrap.dedent(f"""\
        nodes_arr=( $( cat $PBS_NODEFILE ) )
        head_node="${{nodes_arr[0]}}"
        head_node_ip=$(ssh "${{head_node}}" hostname -i | awk '{{print $1}}')
        echo "Head node : ${{head_node_ip}}"
        MASTER_PORT=$(( RANDOM % 10000 + 20000 ))

        export CUDA_VISIBLE_DEVICES={cuda_devices}

        MASTER_ADDR=${{head_node_ip}} MASTER_PORT=${{MASTER_PORT}} \\
        mpiexec -n {nodes * args.gpus} --ppn {args.gpus} --cpu-bind none \\
            python {app} -c ${{CONFIG}}{extra}
    """)


def _derecho_pbsdsh_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    app_relpath: str,
    output_line: str,
    depend_line: str,
    save_loc: str = None,
    app_args: str = "",
) -> str:
    """Return a full derecho PBS script launching *app_relpath* via pbsdsh + torchrun.

    Multi-node alternative to the mpiexec launch, selected by ``credit submit --launcher
    pbsdsh``. One pbsdsh task per node, each running ``torchrun --nproc-per-node``; NCCL
    runs over the libfabric module. The launch logic and its rationale live in
    :func:`credit.pbs._pbsdsh_launch_block`, shared with the standalone launcher in
    ``credit/pbs.py`` so the two stay in lockstep.

    ``app_args`` carries entrypoint-specific flags (e.g. realtime's ``--init-time`` /
    ``--steps``) appended after ``-c <config>``. ``torchrun_args`` is appended to the
    torchrun invocation; the per-node script is written through an unquoted heredoc, so
    any ``$`` in it expands on the mother-superior node when that script is written.
    """
    from credit.pbs import PBSDSH_DERECHO_MODULES, _pbsdsh_launch_block

    logs_dir = os.path.join(os.path.expandvars(save_loc), "logs") if save_loc else "."
    conda_env = args.conda_env
    torchrun = _resolve_torchrun(conda_env)

    header = textwrap.dedent(f"""\
        #!/bin/bash -l
        #PBS -A {args.account}
        #PBS -N {args.job_name}
        #PBS -l walltime={args.walltime}
        #PBS -l select={args.nodes}:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
        #PBS -q {args.queue}
        #PBS -j oe
        #PBS -k eod
        #PBS -r n
        {output_line}
        {depend_line}
        module --force purge
        module load {PBSDSH_DERECHO_MODULES}

        conda activate {args.conda_env}

        echo "Nodes     : {args.nodes}"
        echo "GPUs/node : {args.gpus}"
        echo "Config    : {config}"
    """)
    block = _pbsdsh_launch_block(
        torchrun=torchrun + _torchrun_extra(args),
        script_path=f"{repo}/credit/applications/{app_relpath}",
        config_path=config,
        num_gpus=args.gpus,
        logdir_parent=logs_dir,
        app_args=app_args,
    )
    return header + block


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
        logs_dir = _logs_dir(save_loc)
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
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}{_gpu_type_select(args)}
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
            TORCHRUN={getattr(args, "torchrun", None) or "$(which torchrun)"}

            echo "Config : ${{CONFIG}}"
            echo "Node   : $(hostname)"
            echo "GPUs   : ${{NGPUS}}"
            echo "torchrun: ${{TORCHRUN}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}}{_torchrun_extra(args)} \\
                ${{REPO}}/credit/applications/train_gen2.py -c ${{CONFIG}}
        """)

    else:  # derecho
        nodes = _derecho_nodes(args)
        if _use_pbsdsh(args):
            return _derecho_pbsdsh_script(args, config, repo, "train_gen2.py", output_line, depend_line, save_loc)
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

        return header + _derecho_launch(args, repo, "train_gen2.py")


def _qsub(script: str, save_loc: str | None = None) -> str:
    """Write *script* to save_loc/pbs_scripts/, call qsub, and return the job ID string.

    Raises :class:`credit.runs.SchedulerError`; ``credit submit`` prints it and exits 1.
    """
    script_path, delete_after = _write_submit_script(script, save_loc, "pbs_scripts")
    try:
        return submit_job("pbs", script_path)
    finally:
        if delete_after:
            os.unlink(script_path)


def _print_submitted(
    job_ids: list, cluster: str, scheduler: str, config: str, save_loc: str, run_id: str | None = None
) -> None:
    """Print a confirmation of the submitted job number(s), the config, the log directory, and the run id.

    qsub returns ``<number>.<server>`` (e.g. ``5875448.casper-pbs``); only the
    number is shown.  The log directory matches the one the batch scripts write to.
    """
    numbers = [str(job_id).split(".", 1)[0] for job_id in job_ids]
    jobs = f"Job {numbers[0]}" if len(numbers) == 1 else f"Jobs {', '.join(numbers)}"
    print(f"[SUCCESS] {jobs} submitted to {cluster.capitalize()} ({scheduler}).")
    print(f"  Config: {config}")
    print(f"  Logs:   {os.path.join(save_loc, 'logs')}", flush=True)
    if run_id:
        print(f"  Run:    {run_id}  (credit status {run_id})", flush=True)


# ---------------------------------------------------------------------------
# Run records (credit/runs.py): self-reporting scripts and the run ledger
# ---------------------------------------------------------------------------

# Shell expression for the job key inside a running batch script.
_JOB_KEY_EXPR = {"pbs": "${PBS_JOBID%%.*}", "slurm": "${SLURM_JOB_ID}", "local": "local-$$"}

_RUN_RECORD_PREAMBLE = r"""# ---- CREDIT run record (credit/runs.py): job.started / job.exited / job.signaled ----
export PYTHONUNBUFFERED=1
@@EXPORTS@@
_credit_event() {
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3N)
    case "$ts" in *N) ts="${ts%.*}.000" ;; esac  # BSD date has no %N
    mkdir -p "$CREDIT_RUN_DIR/events"
    printf '{"v":1,"ts":"%sZ","type":"%s","src":{"writer":"script","job":"%s","host":"%s"},"data":%s}\n' \
        "$ts" "$1" "$CREDIT_JOB_KEY" "$(hostname)" "$2" >> "$CREDIT_RUN_DIR/events/$CREDIT_JOB_KEY.script.jsonl"
}
_credit_on_term() {
    _credit_event job.signaled '{"signal":"TERM"}'
    pkill -TERM -P "$_credit_child" 2>/dev/null
    wait "$_credit_child"
    _credit_rc=$?
    _credit_event job.exited "{\"code\":${_credit_rc}}"
    exit ${_credit_rc}
}
trap _credit_on_term TERM
_credit_event job.started "{\"host\":\"$(hostname)\"}"
# The job runs in the background so the TERM trap (walltime, qdel) fires at once. The
# subshell survives TERM itself and waits for the job, so job.exited has the job's own
# exit code and is written only after the job has finished cleaning up.
( trap : TERM
"""

_RUN_RECORD_EPILOGUE = r""") &
_credit_child=$!
wait "$_credit_child"
_credit_rc=$?
_credit_event job.exited "{\"code\":${_credit_rc}}"
exit ${_credit_rc}
"""


def _run_record_exports(run: dict, scheduler: str) -> str:
    """Return the ``export`` lines for the run record and any ``CREDIT_EVENTS_*`` settings in this environment."""
    import shlex

    env = {
        "CREDIT_RUN_ID": shlex.quote(run["run_id"]),
        "CREDIT_RUN_DIR": shlex.quote(run["run_dir"]),
        "CREDIT_JOB_KEY": _JOB_KEY_EXPR[scheduler],
    }
    env.update({k: shlex.quote(v) for k, v in sorted(os.environ.items()) if k.startswith("CREDIT_EVENTS_")})
    return "\n".join(f"export {k}={v}" for k, v in env.items())


def _with_run_record(script: str, run: dict | None, scheduler: str) -> str:
    """Return *script* made self-reporting for *run* (``{"run_id", "run_dir"}``); unchanged if *run* is None.

    The scheduler directives (the leading ``#`` / blank lines) stay first.  The
    rest of the script runs in a backgrounded subshell between a preamble that
    exports ``CREDIT_RUN_ID`` / ``CREDIT_RUN_DIR`` / ``CREDIT_JOB_KEY`` and
    appends ``job.started``, and an epilogue that appends ``job.exited`` with the
    job's exit status.  A ``TERM`` trap appends ``job.signaled``, forwards TERM
    to the job's processes (the subshell itself ignores it), waits for the job
    to exit, and then appends ``job.exited`` with the job's own exit code.  All markers go to
    ``events/<job_key>.script.jsonl`` using only ``printf``/``date``.
    """
    if not run:
        return script
    lines = script.rstrip("\n").split("\n")
    n = 0
    while n < len(lines) and (not lines[n].strip() or lines[n].lstrip().startswith("#")):
        n += 1
    header = "\n".join(lines[:n]).rstrip("\n")
    body = "\n".join(lines[n:])
    preamble = _RUN_RECORD_PREAMBLE.replace("@@EXPORTS@@", _run_record_exports(run, scheduler))
    return f"{header}\n\n{preamble}{body}\n{_RUN_RECORD_EPILOGUE}"


def _target(args: argparse.Namespace, scheduler: str, save_loc: str) -> dict:
    """Return the ``run.json`` ``target`` block for resolved *args*."""
    queue = getattr(args, "queue", None)
    server = queue.split("@", 1)[1] if scheduler == "pbs" and queue and "@" in queue else None
    return {
        "scheduler": scheduler,
        "cluster": getattr(args, "cluster", None),
        "server": server,
        "queue": queue,
        "account": getattr(args, "account", None),
        "nodes": getattr(args, "nodes", None),
        "gpus": getattr(args, "gpus", None),
        "cpus": getattr(args, "cpus", None),
        "mem": getattr(args, "mem", None),
        "walltime": getattr(args, "walltime", None),
        "log_dir": _logs_dir(save_loc),
    }


def _compute_chain(args: argparse.Namespace, conf: dict | None = None, done: int = 0) -> int:
    """Return the number of jobs to chain: ``args.chain``, else ``ceil((epochs - done) / num_epoch)``.

    ``epochs`` / ``num_epoch`` come from the ``trainer`` block of *conf* (read from
    ``args.config`` when None); *done* is the epochs a resumed run already trained.
    Anything missing or unreadable means one job.
    """
    if args.chain is not None:
        return max(1, int(args.chain))
    try:
        if conf is None:
            with open(args.config) as f:
                conf = yaml.safe_load(f)
        trainer = conf.get("trainer", {})
        return max(1, math.ceil((int(trainer["epochs"]) - done) / int(trainer["num_epoch"])))
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
        "  Memory   : %s per node\n"
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
        getattr(args, "mem", None) or "site default",
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
        logs_dir = _logs_dir(save_loc)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    job_name = getattr(args, "job_name", "credit_realtime")

    if args.cluster == "casper":
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}{_gpu_type_select(args)}
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
            TORCHRUN={getattr(args, "torchrun", None) or "$(which torchrun)"}

            echo "Realtime forecast — init: {init_time}  steps: {steps}"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}}{_torchrun_extra(args)} \\
                ${{REPO}}/credit/applications/rollout_realtime_gen2.py \\
                -c ${{CONFIG}} --init-time {init_time} --steps {steps}
        """)

    else:  # derecho
        nodes = _derecho_nodes(args)
        app_args = f"--init-time {init_time} --steps {steps}"
        if _use_pbsdsh(args):
            args = copy.copy(args)
            args.job_name = job_name
            return _derecho_pbsdsh_script(
                args, config, repo, "rollout_realtime_gen2.py", output_line, "", save_loc, app_args=app_args
            )
        header = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {args.account}
            #PBS -N {job_name}
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
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

            echo "Realtime forecast — init: {init_time}  steps: {steps}"
            echo "Config    : ${{CONFIG}}"
            echo "Nodes     : {nodes}  GPUs/node: {args.gpus}"
        """)
        return header + _derecho_launch(args, repo, "rollout_realtime_gen2.py", app_args=app_args)


def _build_preprocess_pbs_script(
    args: argparse.Namespace,
    config: str,
    repo: str,
    save_loc: str = None,
) -> str:
    """Return a PBS script that runs the preprocessing / scaler-fitting job."""
    if save_loc:
        logs_dir = _logs_dir(save_loc)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    job_name = getattr(args, "job_name", "credit_preprocess")

    if args.cluster == "casper":
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}{_gpu_type_select(args)}
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
            TORCHRUN={getattr(args, "torchrun", None) or "$(which torchrun)"}

            echo "Preprocessing — scaler fitting"
            echo "Config  : ${{CONFIG}}"
            echo "Node    : $(hostname)"
            echo "GPUs    : ${{NGPUS}}"

            ${{TORCHRUN}} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}}{_torchrun_extra(args)} \\
                ${{REPO}}/credit/applications/preprocess.py -c ${{CONFIG}}
        """)

    else:  # derecho
        nodes = _derecho_nodes(args)
        if _use_pbsdsh(args):
            # Preprocess has no chaining, so no depend line.
            args = copy.copy(args)
            args.job_name = job_name
            return _derecho_pbsdsh_script(args, config, repo, "preprocess.py", output_line, "", save_loc)
        header = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {args.account}
            #PBS -N {job_name}
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
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

            echo "Preprocessing — scaler fitting"
            echo "Config    : ${{CONFIG}}"
            echo "Nodes     : {nodes}  GPUs/node: {args.gpus}"
        """)
        return header + _derecho_launch(args, repo, "preprocess.py")


def _build_rollout_pbs_script(
    args: argparse.Namespace, config: str, repo: str, subset: int, n_subsets: int, save_loc: str = None
) -> str:
    """Return a PBS script for one subset of an ensemble rollout."""
    job_name = _rollout_job_name(args.job_name, subset, n_subsets)
    if save_loc:
        logs_dir = _logs_dir(save_loc)
        output_line = f"#PBS -o {logs_dir}"
    else:
        output_line = ""

    if args.cluster == "casper":
        torchrun = args.torchrun or _resolve_torchrun(args.conda_env)
        return textwrap.dedent(f"""\
            #!/bin/bash -l
            #PBS -N {job_name}
            #PBS -l select=1:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}{_gpu_type_select(args)}
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

            {torchrun} --standalone --nnodes=1 --nproc-per-node=${{NGPUS}}{_torchrun_extra(args)} \\
                ${{REPO}}/credit/applications/rollout_gen2.py \\
                -c ${{CONFIG}}
                # -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}  # rollout_gen2.py does not support these flags yet
        """)

    else:  # derecho
        nodes = _derecho_nodes(args)
        if _use_pbsdsh(args):
            # Rollout jobs are submitted in parallel, never chained, so no depend line.
            args = copy.copy(args)
            args.job_name = job_name
            return _derecho_pbsdsh_script(args, config, repo, "rollout_gen2.py", output_line, "", save_loc)
        header = textwrap.dedent(f"""\
            #!/bin/bash
            #PBS -A {args.account}
            #PBS -N {job_name}
            #PBS -l walltime={args.walltime}
            #PBS -l select={nodes}:ncpus={args.cpus}:ngpus={args.gpus}:mem={args.mem}
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

            echo "Ensemble rollout — subset {subset} of {n_subsets}"
            echo "Config    : ${{CONFIG}}"
            echo "Nodes     : {nodes}  GPUs/node: {args.gpus}"
        """)
        return (
            header
            + _derecho_launch(args, repo, "rollout_gen2.py")
            + f"    # -c ${{CONFIG}} --subset {subset} --no_subset {n_subsets}"
            "  # rollout_gen2.py does not support these flags yet\n"
        )


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
        "  Nodes per job  : %s\n"
        "  GPUs per job   : %s per node  (%s total)\n"
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
        args.nodes,
        args.gpus,
        args.nodes * args.gpus,
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


def _rollout_sched_cfg(cluster: str, sched_cfg: dict) -> dict:
    """Return *sched_cfg* over the rollout defaults (one GPU, 6 h), which differ from training's.

    ``pbs.mem: auto`` is sized for training, so a rollout keeps its own ``mem`` default.
    """
    defaults = {"ngpus": 1, "ncpus": 8 if cluster == "casper" else 16, "mem": "128GB", "walltime": "06:00:00"}
    return {
        **defaults,
        **{k: v for k, v in sched_cfg.items() if v is not None and not (k == "mem" and _is_auto(v))},
    }


# ---------------------------------------------------------------------------
# Library path for credit.remote: plan (pure, deterministic) and submit
# ---------------------------------------------------------------------------


class SubmitConflict(SubmitError):
    """The config or the re-planned scripts no longer match what was approved."""


_PLAN_OPTION_DEFAULTS = {
    "gpus": None,
    "nodes": None,
    "cpus": None,
    "mem": None,
    "walltime": None,
    "account": None,
    "queue": None,
    "gpu_type": None,
    "constraint": None,
    "qos": None,
    "chain": None,
    "reload": False,
    "jobs": 1,
    "init_time": None,
    "steps": 40,
    "conda_env": None,
    "launcher": "mpiexec",
    "resume_from": None,
}


# Request keys that never become builder options.
_NOT_OPTIONS = ("resume_from", "config", "cluster", "scheduler", "torchrun", "dry_run", "created_by")


_PBS_WALLTIME = re.compile(r"(\d+)(?::(\d+))?(?::(\d+))?")  # [[HH:]MM:]SS
_SLURM_DAYS = re.compile(r"(\d+)-(\d+)(?::(\d+))?(?::(\d+))?")  # D-HH, D-HH:MM, D-HH:MM:SS
_SLURM_TIME = re.compile(r"(\d+)(?::(\d+))?(?::(\d+))?")  # MM, MM:SS, HH:MM:SS


def _walltime_hours(walltime, scheduler: str = "pbs") -> float | None:
    """Hours in a walltime as *scheduler* reads it, or None if it doesn't parse.

    PBS reads ``[[HH:]MM:]SS``, so a bare ``600`` is seconds.  SLURM reads
    ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and ``D-HH:MM:SS``,
    so a bare ``600`` is minutes.
    """
    if walltime is None or isinstance(walltime, bool):
        return None
    text = str(walltime).strip()
    if scheduler == "slurm":
        m = _SLURM_DAYS.fullmatch(text)
        if m:
            d, h, mi, sec = (int(g or 0) for g in m.groups())
            return d * 24 + h + mi / 60 + sec / 3600
        m = _SLURM_TIME.fullmatch(text)
        if not m:
            return None
        parts = [int(g) for g in m.groups() if g is not None]
        if len(parts) == 1:
            return parts[0] / 60
        if len(parts) == 2:
            return parts[0] / 60 + parts[1] / 3600
        return parts[0] + parts[1] / 60 + parts[2] / 3600
    m = _PBS_WALLTIME.fullmatch(text)
    if not m:
        return None
    parts = [int(g) for g in m.groups() if g is not None]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, mi, sec = parts
    return h + mi / 60 + sec / 3600


# What a plan request's ``options`` may hold.  Every value ends up in a batch
# script (a ``#PBS``/``#SBATCH`` directive or a shell line), and the GPU-hour
# estimate only sees the resolved target, so values are checked strictly:
# nothing may add a directive, a chunk or a command.  ``env_setup``/``modules``
# and other script lines come only from the config file.
_INT_OPTIONS = ("gpus", "nodes", "cpus", "chain", "jobs", "steps")
_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
_STRING_OPTIONS = {
    "account": re.compile(_NAME),
    "queue": re.compile(rf"{_NAME}(?:@{_NAME})?"),
    "gpu_type": re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*"),
    "qos": re.compile(_NAME),
    "constraint": re.compile(r"[A-Za-z0-9_][A-Za-z0-9_&|-]*"),  # SLURM features, e.g. gpu&hbm80g
    "mem": re.compile(r"\d+(?:\.\d+)?(?:[KMGTP]i?)?B?", re.IGNORECASE),
    "conda_env": re.compile(r"[A-Za-z0-9_./+][A-Za-z0-9_./+-]*"),
    "init_time": re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}(?::\d{2}(?::\d{2})?)?)?"),
    "launcher": re.compile(r"mpiexec|pbsdsh"),
    "resume_from": re.compile(r"[a-z]+-\d{8}T\d{6}-[0-9a-f]{4}"),
}


def _check_plan_options(options, scheduler: str) -> None:
    """Refuse a plan request's ``options`` unless every key is known and every value is safe to template.

    Raises :class:`SubmitError`.  ``None`` means "the default" for any key.
    """
    if not isinstance(options, dict):
        raise SubmitError("options must be a mapping")
    unknown = sorted(str(k) for k in options if k not in _PLAN_OPTION_DEFAULTS)
    if unknown:
        raise SubmitError(
            f"unknown option {unknown[0]!r}; expected one of {', '.join(sorted(_PLAN_OPTION_DEFAULTS))} "
            "(anything else, such as env_setup or modules, goes in the config file)",
            field=unknown[0],
        )
    for key, value in options.items():
        if value is None:
            continue
        if key in _INT_OPTIONS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SubmitError(f"options.{key} must be a whole number of at least 1, not {value!r}", field=key)
        elif key == "reload":
            if not isinstance(value, bool):
                raise SubmitError(f"options.reload must be true or false, not {value!r}", field=key)
        elif key == "walltime":
            hours = None if isinstance(value, bool) else _walltime_hours(value, scheduler)
            if not isinstance(value, (str, int)) or hours is None:
                form = "MM, MM:SS, HH:MM:SS or D-HH[:MM[:SS]]" if scheduler == "slurm" else "[[HH:]MM:]SS"
                raise SubmitError(f"options.walltime must be a {scheduler} walltime ({form}), not {value!r}", field=key)
        elif not isinstance(value, str) or not _STRING_OPTIONS[key].fullmatch(value):
            raise SubmitError(
                f"options.{key} {str(value)[:80]!r} is not allowed: it must be a plain {key.replace('_', ' ')} "
                "(no spaces, newlines, shell characters or extra directives)",
                field=key,
            )


def _load_plan_config(request: dict) -> tuple:
    """Return ``(config_path, config_bytes, conf, parent_entry)`` for a plan request.

    ``request.config`` must lie in an allowed root (the same check as
    ``credit.remote read``) and be a ``.yml``/``.yaml`` mapping; a resume uses
    the parent run's snapshot instead.  Raises :class:`SubmitError` (or
    :class:`credit.runs.PathScopeError`).
    """
    options = request.get("options") or {}
    parent = None
    if options.get("resume_from"):
        parent = runs.find_run(options["resume_from"])
        if parent is None:
            raise SubmitError(f"resume_from: no run {options['resume_from']!r} in the ledger")
        config = os.path.join(parent["run_dir"], "config.yml")
    elif request.get("config"):
        if any(c.isspace() for c in str(request["config"])):
            raise SubmitError("the config path must not contain whitespace")
        config = runs.resolve_in_scope(request["config"])
    else:
        raise SubmitError("plan needs a config (or options.resume_from)")
    try:
        data, conf = runs.load_yaml_mapping(config)
    except runs.ConfigFileError as exc:
        raise SubmitError(str(exc)) from None
    if not conf.get("save_loc"):
        raise SubmitError(f"config '{config}' has no save_loc")
    if any(c.isspace() for c in str(conf["save_loc"])):
        raise SubmitError("save_loc must not contain whitespace")
    return config, data, conf, parent


def _build_local_script(kind: str, config: str, args: argparse.Namespace, run: dict) -> str:
    """Return the bash script the ``local`` scheduler runs: one ``python -m credit.cli <kind>`` call.

    ``$CREDIT_REMOTE_LOCAL_CMD`` (read when planning) replaces ``python -m
    credit.cli`` so tests can run a fake trainer through the real script.
    """
    import shlex

    cmd = os.environ.get("CREDIT_REMOTE_LOCAL_CMD") or f"{shlex.quote(sys.executable)} -m credit.cli"
    extra = f" --init-time {shlex.quote(str(args.init_time))} --steps {int(args.steps)}" if kind == "realtime" else ""
    body = f'echo "Config : {config}"\necho "Node   : $(hostname)"\n{cmd} {kind} -c {shlex.quote(config)}{extra}\n'
    return _with_run_record("#!/bin/bash\n\n" + body, run, "local")


def _plan_cli(kind: str, request: dict, config: str, options: dict, n_jobs: int) -> str:
    """Return the ``credit`` command line equivalent to a plan request."""
    import shlex

    scheduler = request.get("scheduler") or "pbs"
    if scheduler == "local":
        extra = f" --init-time {options['init_time']} --steps {options['steps']}" if kind == "realtime" else ""
        return f"credit {kind} -c {shlex.quote(config)}{extra}"
    parts = ["credit", "submit", "--cluster", str(request.get("cluster")), "-c", config]
    if scheduler != "pbs":
        parts += ["--scheduler", scheduler]
    if kind != "train":
        parts += ["--mode", kind]
    for flag, key in (
        ("--gpus", "gpus"),
        ("--nodes", "nodes"),
        ("--cpus", "cpus"),
        ("--mem", "mem"),
        ("--walltime", "walltime"),
        ("--account", "account"),
        ("--queue", "queue"),
        ("--gpu-type", "gpu_type"),
        ("--constraint", "constraint"),
        ("--qos", "qos"),
        ("--conda-env", "conda_env"),
    ):
        if options.get(key) is not None:
            parts += [flag, str(options[key])]
    if options.get("launcher") and options["launcher"] != "mpiexec":
        parts += ["--launcher", options["launcher"]]
    if kind == "train":
        if options.get("reload"):
            parts.append("--reload")
        if n_jobs > 1 or options.get("chain") is not None:
            parts += ["--chain", str(n_jobs)]
    if kind == "rollout" and int(options.get("jobs") or 1) != 1:
        parts += ["--jobs", str(options["jobs"])]
    if kind == "realtime":
        parts += ["--init-time", str(options["init_time"]), "--steps", str(options["steps"])]
    return " ".join(shlex.quote(p) for p in parts)


def _plan(request: dict, cli: argparse.Namespace | None = None) -> dict:
    """Return the full submission plan for a ``credit.remote plan`` request, writing nothing.

    It raises :class:`SubmitError` instead of exiting, and it is deterministic: the
    same request (with the same ``run_id``) and the same config bytes give the
    same scripts.  The scripts run the run directory's config snapshot (and
    ``config_reload.yml`` for chained or resumed jobs); chain dependencies are
    not in the scripts but go on the ``qsub``/``sbatch`` command line at submit.

    *cli* is ``credit submit``'s parsed arguments (see :func:`_submit`); it is never
    part of a remote request.  It keeps what the CLI has always done: the config
    may be any YAML file (no path scope; ``save_loc`` defaults to ``.`` with only
    ``$VARS`` expanded), option values are the user's own and are not restricted,
    ``--torchrun`` / ``--torchrun-args`` apply, ``mem: auto`` is sized for the
    job's mode, and each job carries ``job_name`` (the name its script sets,
    which the CLI keeps) and, after the first in a chain, ``depend_placeholder``:
    the ``<job_N_id>`` that stands for the job it depends on in the script's own
    ``afterok`` directive, replaced with the real id at submit.
    """
    kind = request.get("kind") or "train"
    if kind not in runs.KINDS:
        raise SubmitError(f"unknown kind {kind!r}; expected one of {', '.join(runs.KINDS)}")
    scheduler = request.get("scheduler") or "pbs"
    if scheduler not in ("pbs", "slurm", "local"):
        raise SubmitError(f"unknown scheduler {scheduler!r}; expected pbs, slurm or local")
    if cli is None:
        _check_plan_options(request.get("options") or {}, scheduler)
        if request.get("cluster") is not None and not re.fullmatch(r"[a-z][a-z0-9_-]*", str(request["cluster"])):
            raise SubmitError(f"unknown cluster {str(request['cluster'])[:40]!r}")
    options = {**_PLAN_OPTION_DEFAULTS, **(request.get("options") or {})}
    if cli is None:
        source_config, data, conf, parent = _load_plan_config(request)
        save_loc = runs.expand(conf["save_loc"])
    else:
        source_config, parent = os.path.abspath(request["config"]), None
        with open(source_config, "rb") as f:
            data = f.read()
        conf = yaml.safe_load(data) or {}
        save_loc = os.path.expandvars(conf.get("save_loc", "."))
    if parent is not None and kind != "train":
        raise SubmitError("resume_from applies to train runs only")
    run_id = request.get("run_id") or runs.new_run_id(kind)
    if not re.fullmatch(rf"{kind}-\d{{8}}T\d{{6}}-[0-9a-f]{{4}}", str(run_id)):
        raise SubmitError(f"run_id {run_id!r} is not a {kind} run id (<kind>-<YYYYmmddTHHMMSS>-<4 hex>)")
    run = {"run_id": run_id, "run_dir": runs.run_dir_for(save_loc, run_id)}
    run_config = os.path.join(run["run_dir"], "config.yml")
    reload_config = os.path.join(run["run_dir"], "config_reload.yml")
    log_dir = _logs_dir(save_loc)
    warnings = []

    args = argparse.Namespace(
        config=source_config,
        cluster=request.get("cluster") or ("local" if scheduler == "local" else None),
        scheduler=scheduler,
        torchrun=None,
        dry_run=False,
        submit_mode=kind,  # mem: auto sizes train / preprocess only
        **{k: v for k, v in options.items() if k not in _NOT_OPTIONS},
    )
    if cli is not None:
        args.torchrun = getattr(cli, "torchrun", None)
        args.torchrun_args = getattr(cli, "torchrun_args", None)

    trainer = conf.get("trainer") or {}
    epochs_total = epochs_per_job = None
    n_jobs = 1
    reload = bool(options["reload"]) or parent is not None
    if kind == "train":
        epochs_total, epochs_per_job = trainer.get("epochs"), trainer.get("num_epoch")
        last = runs.csv_last_epoch(save_loc) if parent is not None else None
        n_jobs = _compute_chain(args, conf, done=0 if last is None else last + 1)  # CSV epochs are 0-based
        if last is not None and epochs_total and last + 1 >= int(epochs_total):
            raise SubmitError(f"nothing left to train: {last + 1} of {epochs_total} epochs are done")
    if kind == "realtime" and not options["init_time"] and cli is None:
        raise SubmitError("realtime runs need options.init_time")
    if scheduler == "local":
        if n_jobs > 1:
            warnings.append(f"the local scheduler runs one job; the {n_jobs}-job chain is not submitted")
            n_jobs = 1
        if kind == "rollout" and int(args.jobs or 1) > 1:
            warnings.append("the local scheduler runs one rollout job")
        target = _target(args, scheduler, save_loc)
        target.update({"nodes": 1, "cpus": None, "mem": None, "walltime": None})
        first = reload_config if (kind == "train" and reload) else run_config
        jobs = [{"index": 1, "script": _build_local_script(kind, first, args, run), "config": first}]
        jobs[0]["depends_on_index"] = None
    else:
        if not args.cluster:
            raise SubmitError("plan needs a cluster for pbs/slurm runs")
        if scheduler == "slurm":
            sched_cfg = conf.get("slurm") or conf.get("pbs") or {}
        else:
            sched_cfg = conf.get("pbs") or {}
        if kind == "rollout":
            sched_cfg = _rollout_sched_cfg(args.cluster, sched_cfg)
        args = _resolve_slurm_opts(args, sched_cfg) if scheduler == "slurm" else _resolve_pbs(args, sched_cfg)
        if not args.conda_env:
            raise SubmitError(
                f"no conda environment: set options.conda_env, or {scheduler}.conda in the config "
                "(the batch script activates it before running CREDIT)"
            )
        ctx = _scheduler_ctx(args)
        target = _target(args, scheduler, save_loc)
        repo = _repo_root()
        jobs = []
        if kind == "train":
            first = reload_config if reload else run_config
            for i in range(1, n_jobs + 1):
                cfg = first if i == 1 else reload_config
                dep = f"<job_{i - 1}_id>" if cli is not None and i > 1 else None
                script = ctx["build_train"](args, cfg, repo, depend_on=dep, save_loc=save_loc)
                jobs.append({"index": i, "script": script, "config": cfg, "depends_on_index": i - 1 or None})
                if dep:
                    jobs[-1]["depend_placeholder"] = dep
        elif kind == "preprocess":
            script = ctx["build_preprocess"](args, run_config, repo, save_loc=save_loc)
            jobs.append({"index": 1, "script": script, "config": run_config, "depends_on_index": None})
        elif kind == "realtime":
            script = ctx["build_realtime"](args, run_config, repo, args.init_time, args.steps, save_loc=save_loc)
            jobs.append({"index": 1, "script": script, "config": run_config, "depends_on_index": None})
        else:  # rollout: parallel jobs, never chained
            n = max(1, int(args.jobs or 1))
            if n > 1:
                warnings.append(
                    "per-job init-time subsetting is not implemented for gen2 rollout, "
                    f"so {n} jobs each run the full set of init times"
                )
            for i in range(1, n + 1):
                script = ctx["build_rollout"](args, run_config, repo, i, n, save_loc=save_loc)
                jobs.append({"index": i, "script": script, "config": run_config, "depends_on_index": None})
        for job in jobs:
            job["script"] = _with_run_record(job["script"], run, scheduler)
            if cli is not None:
                name = args.job_name
                job["job_name"] = _rollout_job_name(name, job["index"], len(jobs)) if kind == "rollout" else name

    hours = _walltime_hours(target.get("walltime"), scheduler)
    gpu_hours = None
    if scheduler != "local" and hours is not None and target.get("gpus"):
        gpu_hours = round(hours * int(target.get("nodes") or 1) * int(target["gpus"]) * len(jobs), 2)
    echoed = {**request, "run_id": run_id}
    plan = {
        "run_id": run_id,
        "run_dir": run["run_dir"],
        "kind": kind,
        "config": source_config,
        "config_sha256": hashlib.sha256(data).hexdigest(),
        "save_loc": save_loc,
        "name": os.path.basename(runs.expand(save_loc).rstrip(os.sep)) or run_id,
        "target": target,
        "epochs_total": epochs_total,
        "epochs_per_job": epochs_per_job,
        "jobs": jobs,
        "warnings": warnings,
        "gpu_hours_estimate": gpu_hours,
        "cli": _plan_cli(kind, request, source_config, {**options, "reload": reload}, len(jobs)),
        "request": echoed,
        "scripts_sha256": [hashlib.sha256(j["script"].encode()).hexdigest() for j in jobs],
        "log_dir": log_dir,
        "parent_run_id": parent["run_id"] if parent else None,
    }
    return plan


def _submit_progress(run_dir: str) -> tuple:
    """Return ``(intents, submitted)`` from ``events/submit.jsonl`` (authoritative over run.json).

    *intents* is the set of job indexes with a ``job.submitting`` not followed
    by a ``job.submit_failed``; *submitted* maps index to
    ``{index, job_id, job_key}``.
    """
    lines, _, _ = runs.read_lines(os.path.join(run_dir, "events", runs.SUBMIT_EVENTS))
    intents, submitted = set(), {}
    for _, _, ev in lines:
        d = ev.get("data") or {}
        if ev.get("type") == "job.submitting":
            intents.add(d.get("index"))
        elif ev.get("type") == "job.submit_failed":
            intents.discard(d.get("index"))
        elif ev.get("type") == "job.submitted":
            submitted[d.get("index")] = {
                "index": d.get("index"),
                "job_id": d.get("job_id"),
                "job_key": d.get("job_key"),
            }
    return intents, submitted


def _submit_plan(request: dict, scripts_sha256: list, config_sha256: str, created_by: str = "mcp") -> dict:
    """Re-plan *request* and submit it; idempotent per ``run_id``.

    Raises :class:`SubmitConflict` if the config or any script changed since the
    plan was approved.  Jobs are submitted in order, each named
    ``<run_id>-<index>``: ``job.submitting`` is appended before each
    ``qsub``/``sbatch`` and ``job.submitted`` right after it succeeds (or
    ``job.submit_failed`` when the scheduler refused it).  A failed job is
    reported in ``errors``, earlier jobs stay submitted, and jobs that depend on
    it are skipped.

    Submitting again (the run directory exists) returns ``already_submitted``
    and submits only the jobs that never got as far as an intent.  A job with an
    intent but no outcome (a crash or a ``qsub`` timeout) may or may not be
    queued, so it is reported as ``partial_submit`` with its job name and never
    resubmitted blindly.
    """
    if not request.get("run_id"):
        raise SubmitError("submit needs request.run_id (from the plan)")
    plan = _plan(request)
    if config_sha256 != plan["config_sha256"]:
        raise SubmitConflict(f"config {plan['config']} changed since it was planned")
    if list(scripts_sha256 or []) != plan["scripts_sha256"]:
        raise SubmitConflict("the job scripts changed since they were planned; plan again")
    _, data, _, _ = _load_plan_config(request)
    if hashlib.sha256(data).hexdigest() != plan["config_sha256"]:
        raise SubmitConflict(f"config {plan['config']} changed while submitting")
    return _submit_planned(plan, data, created_by)


def _submit_planned(plan: dict, data: bytes | None, created_by: str, submit=None) -> dict:
    """Submit *plan* (from :func:`_plan`) while holding the run's submit lock; see :func:`_submit_plan`.

    *data* is the config content to snapshot (None copies ``plan["config"]``).
    ``credit submit`` passes *submit*, called as ``submit(job, script)`` in place
    of :func:`credit.runs.submit_job` (``qsub``/``sbatch`` with no ``-N`` or
    dependency flags: the script sets both); its first
    :class:`credit.runs.SchedulerError` is raised after ``job.submit_failed`` is
    logged, instead of being reported in ``errors``.
    """
    run_id, run_dir = plan["run_id"], plan["run_dir"]
    # One submit per run at a time: a retry racing a submit that is still
    # running (a client timeout, two clients) must not queue the jobs twice.
    import fcntl

    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    lock_fd = os.open(os.path.join(os.path.dirname(run_dir), f"{run_id}.submit.lock"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SubmitConflict("submit in progress; sync the run and retry") from None
        return _submit_locked(plan, data, created_by, submit)
    finally:
        os.close(lock_fd)  # releases the lock


def _submit_locked(plan: dict, data: bytes | None, created_by: str, submit=None) -> dict:
    """The body of :func:`_submit_planned`, run while holding the run's submit lock."""
    run_id, run_dir = plan["run_id"], plan["run_dir"]
    scheduler = plan["target"]["scheduler"]
    create = dict(
        created_by=created_by,
        name=plan["name"],
        epochs_total=plan["epochs_total"],
        epochs_per_job=plan["epochs_per_job"],
        run_dir=run_dir,
        parent_run_id=plan["parent_run_id"],
        config_bytes=data,
    )
    args = (run_id, plan["kind"], plan["config"], plan["save_loc"], plan["target"])
    try:
        record, already = runs.create_run(*args, **create), False
    except FileExistsError:
        record, already = runs.create_run(*args, exist_ok=True, **create), True
    os.makedirs(plan["log_dir"], exist_ok=True)
    run_config = os.path.join(run_dir, "config.yml")
    reload_config = os.path.join(run_dir, "config_reload.yml")
    if any(job["config"] == reload_config for job in plan["jobs"]) and not os.path.exists(reload_config):
        _write_reload_config(run_config, reload_config)

    intents, submitted = _submit_progress(run_dir)
    errors = []
    for job in plan["jobs"]:
        index = job["index"]
        name = job.get("job_name") or f"{run_id}-{index}"
        if index in submitted:
            continue
        if index in intents:
            errors.append(
                {
                    "code": "partial_submit",
                    "index": index,
                    "job_name": name,
                    "message": f"job {index} may have been submitted as {name}; check the queue before resubmitting",
                }
            )
            continue
        dep_index = job.get("depends_on_index")
        depend_on = (submitted.get(dep_index) or {}).get("job_id") if dep_index else None
        if dep_index and depend_on is None:
            errors.append(f"job {index} not submitted: job {dep_index} it depends on was not submitted")
            continue
        script = job["script"]
        if depend_on and job.get("depend_placeholder"):
            script = script.replace(job["depend_placeholder"], depend_on)
        rel = runs.write_script(run_dir, index, script)
        intent = runs.make_event("job.submitting", {"index": index, "job_name": name}, "submit")
        runs.append_event(run_dir, runs.SUBMIT_EVENTS, intent)
        try:
            if submit is not None:
                job_id = submit(job, script)
            else:
                job_id = submit_job(
                    scheduler,
                    os.path.join(run_dir, rel),
                    depend_on=depend_on,
                    log_dir=plan["log_dir"],
                    job_name=name,
                    env={"CREDIT_RUN_ID": run_id},
                )
        except SchedulerError as exc:
            if not isinstance(exc, runs.SchedulerTimeout):  # a timeout may still have queued the job
                failed = runs.make_event("job.submit_failed", {"index": index, "error": str(exc)[:500]}, "submit")
                runs.append_event(run_dir, runs.SUBMIT_EVENTS, failed)
            if submit is not None:
                raise
            errors.append(f"job {index}: {exc}")
            continue
        job_rec = {"index": index, "job_id": job_id, "depends_on": depend_on, "config": job["config"], "script": rel}
        rec = runs.add_job(run_dir, record, job_rec)
        submitted[index] = {"index": index, "job_id": job_id, "job_key": rec["job_key"]}
    result = {"run_id": run_id, "run_dir": run_dir, "jobs": [submitted[k] for k in sorted(submitted)], "errors": errors}
    if already:
        result["already_submitted"] = True
    return result


# ---------------------------------------------------------------------------
# credit submit: the plan path above, with the CLI's messages
# ---------------------------------------------------------------------------


def _cli_request(args: argparse.Namespace) -> dict:
    """Return the :func:`_plan` request for ``credit submit`` (or ``credit rollout-ensemble``) *args*."""
    keys = ("gpus", "nodes", "cpus", "mem", "walltime", "account", "queue", "gpu_type", "constraint", "qos")
    options = {k: getattr(args, k, None) for k in (*keys, "chain", "init_time", "conda_env")}
    options.update(
        reload=bool(getattr(args, "reload", False)),
        jobs=getattr(args, "jobs", 1),
        steps=getattr(args, "steps", 40),
        launcher=getattr(args, "launcher", None) or "mpiexec",
    )
    return {
        "kind": getattr(args, "submit_mode", None) or "train",
        "config": args.config,
        "cluster": args.cluster,
        "scheduler": getattr(args, "scheduler", None) or "pbs",
        "options": options,
    }


def _rollout_counts(conf: dict) -> tuple:
    """Return ``(n_forecasts, ensemble_size)`` for a rollout config, ``"?"`` where they can't be worked out."""
    try:
        if "inference" in conf:
            inf_conf = conf["inference"]
            if inf_conf.get("run_mode", "batch") == "single":
                n_forecasts = 1
            else:
                from credit.trainers.rollout_utils import batch_init_times

                n_forecasts = len(batch_init_times(inf_conf["batch_forecast"]))
            return n_forecasts, inf_conf.get("ensemble_size", 1)
        from credit.forecast import load_forecasts

        return len(load_forecasts(conf)), conf.get("predict", {}).get("ensemble_size", 1)
    except Exception:
        return "?", "?"


def _log_plan(kind: str, shown: argparse.Namespace, n_jobs: int, sched_name: str, args, conf: dict) -> None:
    """Log the human-readable job plan; *shown* holds the resolved target plus the config as given."""
    if kind == "train":
        _print_job_plan(shown, n_jobs)
        return
    if kind == "rollout":
        _print_ensemble_rollout_plan(shown, n_jobs, *_rollout_counts(conf))
        return
    sep = "=" * 52
    if kind == "preprocess":
        logger.info(
            "\n%s\n  Preprocess job plan\n%s\n"
            "  Scheduler : %s\n"
            "  Cluster   : %s\n"
            "  Account   : %s\n"
            "  Config    : %s\n"
            "  Nodes     : %s\n"
            "  GPUs      : %s per node  (%s total)\n"
            "  Memory    : %s per node\n"
            "  Walltime  : %s\n"
            "%s",
            sep,
            sep,
            sched_name,
            shown.cluster,
            shown.account,
            shown.config,
            shown.nodes,
            shown.gpus,
            shown.nodes * shown.gpus,
            shown.mem or "site default",
            shown.walltime,
            sep,
        )
        return
    logger.info(
        "\n%s\n  Realtime job plan\n%s\n"
        "  Scheduler : %s\n"
        "  Cluster   : %s\n"
        "  Account   : %s\n"
        "  Config    : %s\n"
        "  Init time : %s\n"
        "  Steps     : %s\n"
        "  Nodes     : %s\n"
        "  GPUs      : %s per node  (%s total)\n"
        "  Walltime  : %s\n"
        "%s",
        sep,
        sep,
        sched_name,
        shown.cluster,
        shown.account,
        shown.config,
        args.init_time,
        getattr(args, "steps", 40),
        shown.nodes,
        shown.gpus,
        shown.nodes * shown.gpus,
        shown.walltime,
        sep,
    )


def _print_dry_run(kind: str, jobs: list) -> None:
    """Print the planned scripts (``--dry-run``): every rollout job, else the first job and a chained one."""
    n = len(jobs)
    if kind == "rollout":
        for job in jobs:
            print(f"# {'=' * 50}")
            print(f"# Job {job['index']}/{n}  (subset {job['index']} of {n})")
            print(f"# {'=' * 50}")
            print(job["script"])
    elif kind == "train":
        print(f"# --- Job 1/{n} ---")
        print(jobs[0]["script"])
        if n > 1:
            print(f"# --- Jobs 2..{n}/{n} (afterok chained, reload config) ---")
            print(jobs[1]["script"])
    else:
        print(jobs[0]["script"])


def _log_submitted(kind: str, job: dict, job_id: str, n_jobs: int) -> None:
    if kind == "train" and job["index"] == 1:
        logger.info("[1/%d] %s  %s", n_jobs, job_id, job["config"])
    elif kind == "train":
        logger.info("[%d/%d] %s  afterok  (reload)", job["index"], n_jobs, job_id)
    elif kind == "rollout":
        logger.info("[%2d/%d] %s", job["index"], n_jobs, job_id)
    else:
        logger.info("Submitted: %s", job_id)


def _submit(args: argparse.Namespace) -> None:
    """``credit submit``: plan the jobs with :func:`_plan`, then print (``--dry-run``) or submit them.

    The plan and the submit are those of ``credit.remote plan`` / ``submit``
    (:func:`_plan`, :func:`_submit_planned`: the same scripts, run record, submit
    events and lock), with ``cli=args`` for what only the CLI does.  Each script
    is also copied to ``<save_loc>/pbs_scripts/`` (``slurm_scripts/``) and
    submitted from there, as ``credit submit`` always has.
    """
    request = _cli_request(args)
    kind = request["kind"]
    ctx = _scheduler_ctx(args)
    ctx["load"](args.config)  # exits with the CLI's message if the pbs:/slurm: block or its conda env is missing
    try:
        plan = _plan(request, cli=args)
    except SubmitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    with open(args.config) as f:
        conf = yaml.safe_load(f) or {}
    jobs = plan["jobs"]
    shown = argparse.Namespace(**plan["target"], config=args.config)
    _log_plan(kind, shown, len(jobs), ctx["name"], args, conf)
    if args.dry_run:
        _print_dry_run(kind, jobs)
        return

    reload_config = os.path.join(plan["run_dir"], "config_reload.yml")
    if kind == "train" and (getattr(args, "reload", False) or len(jobs) > 1):
        logger.info("Reload config written: %s", reload_config)

    def submit(job: dict, script: str) -> str:
        job_id = ctx["submit"](script, save_loc=plan["save_loc"])
        _log_submitted(kind, job, job_id, len(jobs))
        return job_id

    try:
        result = _submit_planned(plan, None, "cli", submit=submit)
    except SchedulerError as exc:
        print(f"{'sbatch' if ctx['name'] == 'SLURM' else 'qsub'} failed:\n{exc}", file=sys.stderr)
        sys.exit(1)
    except (OSError, SubmitError) as exc:
        print(f"ERROR: could not create the run record in {plan['run_dir']}: {exc}", file=sys.stderr)
        sys.exit(1)
    if result["errors"]:  # not expected: a fresh run, and the first scheduler error stops the CLI
        for err in result["errors"]:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    job_ids = [job["job_id"] for job in result["jobs"]]
    if kind == "rollout":
        save_forecast = conf.get("inference", {}).get("save_forecast") or conf.get("predict", {}).get(
            "save_forecast", "<save_forecast in config>"
        )
        monitor = "squeue -u $USER" if ctx["name"] == "SLURM" else "qstat -u $USER"
        logger.info(
            "\nSubmitted %d parallel rollout jobs.\nOutput will be written to: %s\nMonitor with:\n  %s",
            len(jobs),
            save_forecast,
            monitor,
        )
    shown_config = reload_config if kind == "train" and getattr(args, "reload", False) else plan["config"]
    _print_submitted(job_ids, args.cluster, ctx["name"], shown_config, plan["save_loc"], run_id=plan["run_id"])
