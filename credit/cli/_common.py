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
