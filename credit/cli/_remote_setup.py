"""``credit remote-setup``: choose the folders credit-mcp may read and write on this host.

Writes ``~/.credit/remote.yaml`` (see :mod:`credit.remote._guard`).  Until it
exists, ``credit.remote`` serves only the runs already in the ledger, so this
is the explicit, local step that lets credit-mcp see configs and outputs.  It
is deliberately not an MCP tool: widening what a remote client can reach is
something the user does in their own shell on the HPC system.
"""

import argparse
import os
import sys

import yaml

from credit import runs

from ._common import _prompt, _prompt_bool


def policy_path() -> str:
    from credit.remote import _guard

    return os.path.join(_guard.owner_home(), ".credit", "remote.yaml")


def suggest_roots() -> list:
    """Folders worth offering: the common parents of ledger runs, and the usual NCAR CREDIT_runs dirs."""
    suggestions = []
    parents = {}
    for entry in runs.ledger_index().values():
        save_loc = entry.get("save_loc")
        if save_loc:
            parent = os.path.dirname(os.path.realpath(runs.expand(save_loc)))
            parents[parent] = parents.get(parent, 0) + 1
    suggestions += sorted(parents, key=lambda p: -parents[p])
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user:
        for d in (f"/glade/work/{user}/CREDIT_runs", f"/glade/derecho/scratch/{user}/CREDIT_runs"):
            if os.path.isdir(d):
                suggestions.append(os.path.realpath(d))
    return [s for s in dict.fromkeys(suggestions) if not root_problem(s)]


def root_problem(path: str) -> str | None:
    """Why *path* can't be a root, or None: missing, ``/``, the home dir or above it, or hidden."""
    real = os.path.realpath(runs.expand(path))
    if not os.path.isdir(real):
        return "not an existing directory"
    if runs._too_broad(real):
        return "too broad: it is /, your home directory, or a folder above it"
    if runs._hidden_part(real):
        return "inside a hidden directory"
    return None


def write_policy(path: str, roots: list, redact: bool = True, audit: bool = True, budgets: dict | None = None) -> str:
    """Write the policy file (mode 0600, its directory 0700), keeping any previous one as ``.bak``.

    *budgets* (``max_gpu_hours_per_plan`` / ``max_gpu_hours_per_day``) is written
    as a ``budgets:`` block when given.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    os.chmod(directory, 0o700)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    data = {"roots": roots, "redact": redact, "audit": audit}
    if budgets:
        data["budgets"] = budgets
    text = (
        "# credit-mcp may read, list and write only inside these folders (written by `credit remote-setup`).\n"
        + yaml.safe_dump(data, sort_keys=False)
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def positive_hours(text: str):
    """argparse type for a GPU-hour limit: a positive number."""
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not (value > 0 and value != float("inf")):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of GPU-hours")
    return int(value) if value.is_integer() else value


def _budgets(path: str, args: argparse.Namespace) -> dict | None:
    """The ``budgets:`` block to write: the previous file's, with any limit given on the command line replaced.

    Re-running setup to change the folders must not silently drop a budget.
    """
    given = {
        key: getattr(args, key, None)
        for key in ("max_gpu_hours_per_plan", "max_gpu_hours_per_day")
        if getattr(args, key, None) is not None
    }
    old = None
    try:
        with open(path) as f:
            old = (yaml.safe_load(f) or {}).get("budgets")
    except (OSError, yaml.YAMLError, AttributeError):
        pass
    if not given:
        return old  # kept as-is, even if invalid (credit.remote then refuses to submit)
    return {**(old if isinstance(old, dict) else {}), **given}


def _remote_setup(args: argparse.Namespace) -> None:
    path = args.policy_file or policy_path()
    if args.show:
        if os.path.exists(path):
            print(open(path).read().rstrip())
        else:
            print(f"No policy at {path}: credit-mcp can see only the runs already in the ledger.")
        return

    roots = [os.path.realpath(runs.expand(r)) for r in (args.root or [])]
    if not args.root:
        if args.yes:
            roots = suggest_roots()
        else:
            print("Choose the folders credit-mcp may read and write on this host (configs, run outputs).")
            print("Hidden folders, your home directory itself, and anything above it are never allowed.\n")
            for s in suggest_roots():
                if _prompt_bool(f"Allow {s}?", True):
                    roots.append(s)
            while True:
                extra = _prompt("Another folder (Enter to finish)", "")
                if not extra:
                    break
                roots.append(os.path.realpath(runs.expand(extra)))

    problems = [(r, root_problem(r)) for r in roots if root_problem(r)]
    for r, why in problems:
        print(f"  refusing {r}: {why}", file=sys.stderr)
    roots = [r for r in dict.fromkeys(roots) if not root_problem(r)]
    if not roots:
        print("No usable folders chosen; nothing written.", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(path) and not args.yes and not _prompt_bool(f"Replace {path}?", True):
        print("Nothing written.")
        return
    budgets = _budgets(path, args)
    write_policy(path, roots, redact=not args.no_redact, audit=not args.no_audit, budgets=budgets)
    print(f"[SUCCESS] Wrote {path}")
    for r in roots:
        print(f"  allowed: {r}")
    if isinstance(budgets, dict):
        for key, value in budgets.items():
            print(f"  {key}: {value}")
