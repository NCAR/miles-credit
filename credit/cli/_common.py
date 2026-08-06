"""Shared helpers used across all CLI submodules."""

import logging
import os
import pathlib

from credit.conda_env import torchrun_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster defaults — single source of truth
# ---------------------------------------------------------------------------

# Casper GPU node hardware, as (max GPUs per node, node memory in GB), keyed by
# PBS ``gpu_type``.  Source: NCAR HPC docs "Casper hardware"
# (https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/).
#
#   v100      4x V100 32GB / 768 GB nodes and 8x V100 32GB / 1152 GB nodes
#   a100_80gb 4x A100 80GB / 1024 GB
#   h100      4x H100 80GB / 1024 GB
#   a100      1x A100 40GB / 384 GB   (data analysis & visualization nodes)
#   gp100     1x Quadro GP100 16GB / 384 GB
#   l40       L40 visualization nodes / 768 GB (GPUs-per-node is not published,
#             so treat it as 1 — the memory request then never exceeds the node)
#
# ``None`` is the "any NVIDIA GPGPU" default: use the 8x V100 node (the largest
# GPU count) so the per-GPU share stays small enough to also fit the 4-GPU nodes.
_CASPER_GPU_NODES = {
    None: (8, 1152),
    "v100": (8, 1152),
    "a100_80gb": (4, 1024),
    "h100": (4, 1024),
    "a100": (1, 384),
    "gp100": (1, 384),
    "l40": (1, 768),
}

# Memory requested for a single-GPU Casper job; the scale below grows from here
# to (nearly) the full node memory at the node's maximum GPU count.
_CASPER_MEM_1GPU_GB = 64

# PBS advertises slightly less memory than a node physically has (the OS and
# filesystem caches take a cut), so a request for the full figure above would
# never be schedulable.  Ask for at most this fraction of the node.
_CASPER_MEM_NODE_FRACTION = 0.95


def casper_mem(gpus: int, gpu_type: str = None) -> str:
    """Return the Casper memory request for *gpus* GPUs of *gpu_type*.

    Scales linearly from ``64GB`` for a single GPU up to (nearly) the whole
    node's memory when the job takes every GPU on the node, so a job asks for
    roughly the share of the node it actually occupies.  Unknown GPU types (and
    the "any GPGPU" default) fall back to the profile that is safe on every
    Casper GPU node.  Single-GPU node types keep the 64GB baseline rather than
    claiming their whole (shared, analysis-oriented) node.
    """
    max_gpus, node_mem = _CASPER_GPU_NODES.get(
        (gpu_type or "").lower() or None,
        _CASPER_GPU_NODES[None],
    )
    # Whole-node ask, rounded down to a multiple of 16 GB.
    full = int(node_mem * _CASPER_MEM_NODE_FRACTION) // 16 * 16
    gpus = max(1, int(gpus))
    if max_gpus <= 1:
        return f"{_CASPER_MEM_1GPU_GB}GB"
    if gpus >= max_gpus:
        return f"{full}GB"
    # Round up to the next 16 GB so the scale lands on tidy values.
    mem = _CASPER_MEM_1GPU_GB + (full - _CASPER_MEM_1GPU_GB) * (gpus - 1) / (max_gpus - 1)
    return f"{min(full, int(-(-mem // 16) * 16))}GB"


# PBS queues by cluster.  A ``pbs:`` block written for one NCAR machine is often
# reused against the other (``credit submit --cluster casper`` with a Derecho
# config, say), and the only symptom is a qsub rejection *after* the script has
# been generated.  These sets let the CLI catch the swap up front.  They are not
# exhaustive — an unrecognized queue is left alone, since sites add queues — so
# only a queue that is known to belong to the *other* cluster is an error.
_PBS_QUEUES = {
    "casper": {"casper", "cpu", "gpgpu", "gpudev", "largemem", "vis", "htc", "rda", "l40"},
    "derecho": {"main", "preempt", "develop", "cpudev", "gpudev"},
}


def queue_cluster_error(queue: str, cluster: str) -> str:
    """Return an error message if *queue* belongs to a cluster other than *cluster*.

    Returns ``""`` when the queue is valid for *cluster*, when it is not
    recognized on either machine (a site-specific or newly added queue), or when
    *cluster* is not an NCAR PBS machine.  ``queue`` may carry a ``@server``
    suffix, which is ignored.
    """
    if cluster not in _PBS_QUEUES:
        return ""
    name = str(queue).split("@", 1)[0].strip().lower()
    if name in _PBS_QUEUES[cluster]:
        return ""
    other = [c for c, queues in _PBS_QUEUES.items() if c != cluster and name in queues]
    if not other:
        return ""
    valid = ", ".join(sorted(_PBS_QUEUES[cluster]))
    return (
        f"queue '{name}' is a {other[0].capitalize()} queue, but this job targets {cluster}.\n"
        f"{cluster.capitalize()} queues: {valid}.\n"
        f"Fix the 'queue:' in the config's pbs block, or override it with --queue."
    )


_PBS_DEFAULTS = {
    "casper": {
        "cpus": 8,
        # Fallback only — `credit submit` sizes Casper memory with casper_mem().
        "mem": "128GB",
        "queue": "casper",
        # Unset => any NVIDIA GPGPU on Casper (shorter queue waits).  Set
        # pbs.gpu_type / --gpu-type to pin a specific model (a100_80gb, h100, ...).
        "gpu_type": None,
        "walltime": "12:00:00",
        "gpus": 4,
        "nodes": 1,
        "account": "NAML0001",
        "job_name": "credit_gen2",
    },
    "derecho": {
        "cpus": 64,
        "mem": "480GB",
        "queue": "main",
        "gpu_type": "a100_80gb",
        "walltime": "12:00:00",
        "gpus": 4,
        "nodes": 1,
        "account": "NAML0001",
        "job_name": "credit_gen2",
    },
}

# SLURM defaults are cluster-agnostic — SLURM sites vary too much to enumerate,
# so module loads / partitions come from the config's ``slurm:`` section.  A
# generic site requests GPUs with ``--gres=gpu:N`` and needs an explicit
# partition; ``constraint``/``qos`` stay unset.
_SLURM_DEFAULTS = {
    "cpus": 8,
    "mem": "128GB",
    "partition": "gpu",
    "qos": None,
    "constraint": None,
    "gpu_type": None,
    "walltime": "12:00:00",
    "gpus": 4,
    "nodes": 1,
    "account": None,
    "job_name": "credit_gen2",
}

# Perlmutter (NERSC) runtime environment for NCCL over the Slingshot 11 fabric.
# torch's bundled NCCL only reaches the high-speed network through the
# system-provided AWS-OFI plugin (``libnccl-net.so``), which must be on
# ``LD_LIBRARY_PATH``; the ``NCCL_*`` / ``FI_CXI_*`` vars select the ``hsn``
# interface and the ``cxi`` libfabric provider and apply NERSC's recommended
# tuning.  ``module load nccl`` alone is unreliable (it does not always export
# these or add the plugin dir), so we set them explicitly.
_PERLMUTTER_NCCL_PLUGIN_DIR = "/global/common/software/nersc9/nccl/2.24.3/plugin/lib"
_PERLMUTTER_ENV_SETUP = [
    f"export LD_LIBRARY_PATH={_PERLMUTTER_NCCL_PLUGIN_DIR}:$LD_LIBRARY_PATH",
    'export NCCL_NET="AWS Libfabric"',
    "export NCCL_NET_GDR_LEVEL=PHB",
    "export NCCL_SOCKET_IFNAME=hsn",
    "export NCCL_CROSS_NIC=1",
    "export FI_CXI_DISABLE_HOST_REGISTER=1",
    "export FI_MR_CACHE_MONITOR=userfaultfd",
    "export MPICH_GPU_SUPPORT_ENABLED=1",
]

# Per-cluster SLURM overrides layered on top of ``_SLURM_DEFAULTS``.  Perlmutter
# (NERSC) rejects ``--gres=gpu:N`` ("Job request does not match any supported
# policy") and selects GPU nodes via ``--constraint=gpu`` + ``--qos`` +
# ``--gpus-per-node``; it needs no ``--partition`` or ``--mem`` line, and GPU
# allocations require the ``_g`` account suffix.
_SLURM_CLUSTER_DEFAULTS = {
    "perlmutter": {
        "cpus": 64,
        "mem": None,
        "partition": None,
        "qos": "regular",
        "constraint": "gpu",
        "walltime": "12:00:00",
        "gpus": 4,
        "nodes": 1,
        "job_name": "credit_gen2",
        "modules": "nccl/2.24.3",
        "env_setup": _PERLMUTTER_ENV_SETUP,
    },
}


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
    if not root.handlers:
        root.addHandler(ch)


def _repo_root() -> str:
    """Absolute path to the miles-credit repo root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_torchrun() -> str:
    """Return the path to torchrun, preferring the active conda env."""
    import shutil

    tr = shutil.which("torchrun")
    if tr:
        return tr
    home = os.path.expanduser("~")
    fallback = os.path.join(home, ".conda", "envs", "credit-casper", "bin", "torchrun")
    if os.path.isfile(fallback):
        return fallback
    return "torchrun"


def _resolve_torchrun(conda_env) -> str:
    """Return a torchrun path for a conda env given as a name or a prefix path.

    See :func:`credit.conda_env.conda_prefix_expr` for how the env prefix is
    resolved.  Falls back to :func:`_find_torchrun` when no env is configured.
    """
    if not conda_env:
        return _find_torchrun()
    return torchrun_path(conda_env)


def _is_ncar_system() -> bool:
    """Return True if running on a known NCAR HPC system (Casper or Derecho)."""
    import socket

    host = socket.gethostname()
    return any(name in host for name in ("casper", "crhtc", "derecho", "dec", "crlogin"))


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
