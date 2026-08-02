"""Shared helpers used across all CLI submodules."""

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster defaults — single source of truth
# ---------------------------------------------------------------------------

_PBS_DEFAULTS = {
    "casper": {
        "cpus": 8,
        "mem": "128GB",
        "queue": "casper",
        "gpu_type": "a100_80gb",
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

    A value containing a path separator is treated as a full environment prefix
    (``<prefix>/bin/torchrun``).  A bare environment *name* resolves at run time
    via ``conda info --base`` so it is never mistaken for a same-named directory
    in the current working directory (e.g. the repo's ``credit/`` package dir,
    which made a ``conda: credit`` config yield a bogus ``credit/bin/torchrun``).
    Falls back to :func:`_find_torchrun` when no env is configured.
    """
    if not conda_env:
        return _find_torchrun()
    if "/" in conda_env:
        return f"{conda_env}/bin/torchrun"
    return f"$(conda info --base)/envs/{conda_env}/bin/torchrun"


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
