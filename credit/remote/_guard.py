"""Policy, secret redaction and the audit log for ``credit.remote`` (stdlib + yaml only).

The policy lives in ``remote.yaml`` files that can only *narrow* what the
remote serves::

    roots:            # replaces the default roots (home, work, scratch)
      - /glade/work/$USER/CREDIT_runs
      - /glade/work/$USER/configs
    redact: true      # mask secrets in text sent back (default true)
    audit: true       # append one line per call to $CREDIT_HOME/audit.jsonl (default true)
    budgets:          # GPU-hour limits on ``submit`` (pbs/slurm only; default none)
      max_gpu_hours_per_plan: 200   # above this a person must type the GPU-hours to confirm
      max_gpu_hours_per_day: 1000   # rolling 24 h, no override

Two files are read, ``~/.credit/remote.yaml`` in the account's real home
(from the password database, so an environment change can't skip it) and
``$CREDIT_HOME/remote.yaml``.  Each one that sets ``roots`` narrows the result
further; ``redact: false`` or ``audit: false`` takes effect only when no file
says otherwise.  Each budget is the smallest any file sets, and a budget that
is not a positive number (or an unreadable file) counts as 0, so nothing can be
submitted until it is fixed.  The GPU-hours actually submitted are appended to
``~/.credit/budget.jsonl`` in the real home (see :func:`record_gpu_hours`).
"""

import contextlib
import math
import os
import pwd
import re
import time

import yaml

from credit import runs

MARK = "«redacted»"

_TOKEN_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
]
# Credentials in URLs (scheme://user:secret@host) and Authorization headers keep their prefix.
_URL_PASSWORD = re.compile(r"(\b[a-zA-Z][\w+.-]*://[^/\s:@]+:)([^@\s/]+)(@)")
_BEARER = re.compile(r"(?i)(\bauthorization\s*[:=]\s*['\"]?bearer\s+)([^\s'\"]+)")
# key: value / key=value where the key names a secret.  The value must contain a letter and be at
# least 6 characters, so `token: anon`, `max_tokens: 100000` and `secret: null` are left alone.
_KEY_VALUE = re.compile(
    r"(?i)(\b[\w.-]*(?:password|passwd|secret|api[_-]?key|access[_-]?key|private[_-]?key|token|credential)[\w.-]*"
    r"\s*[:=]\s*['\"]?)(?!«)(?=[^\s'\"#,}]*[A-Za-z])([^\s'\"#,}]{6,})"
)


def owner_home() -> str:
    """The account's home directory from the password database, ignoring ``$HOME``."""
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return os.path.expanduser("~")


def _policy_files() -> list:
    paths = [os.path.join(owner_home(), ".credit", "remote.yaml"), os.path.join(runs.credit_home(), "remote.yaml")]
    out = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp not in out and os.path.isfile(rp):
            out.append(rp)
    return out


BUDGET_KEYS = ("max_gpu_hours_per_plan", "max_gpu_hours_per_day")


def _file_budgets(block) -> tuple:
    """Return ``(limits, problems)`` for one file's ``budgets:`` block; a bad block or value becomes 0."""
    if block is None:
        return {}, []
    if not isinstance(block, dict):
        return dict.fromkeys(BUDGET_KEYS, 0.0), [f"budgets must be a mapping with {' / '.join(BUDGET_KEYS)}"]
    unknown = [str(k) for k in block if k not in BUDGET_KEYS]
    if unknown:  # a misspelt key must not silently mean "no limit"
        return dict.fromkeys(BUDGET_KEYS, 0.0), [f"unknown budgets key {unknown[0]!r}"]
    limits, problems = {}, []
    for key in BUDGET_KEYS:
        value = block.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            limits[key] = 0.0
            problems.append(f"budgets.{key} must be a positive number of GPU-hours, not {value!r}")
        else:
            limits[key] = float(value)
    return limits, problems


def policy() -> dict:
    """Merge the policy files: ``roots`` narrow in turn; ``redact``/``audit`` are true unless every file says false.

    ``budgets`` holds the smallest limit any file sets (None where none does);
    ``budget_errors`` says why a limit was forced to 0.
    """
    roots = None
    flags = {"redact": [], "audit": []}
    budgets = dict.fromkeys(BUDGET_KEYS)
    budget_errors = []
    for path in _policy_files():
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            data = None
        if not isinstance(data, dict):  # an unreadable policy fails closed
            data = {"roots": []}
            budgets = dict.fromkeys(BUDGET_KEYS, 0.0)
            budget_errors.append(f"{path}: cannot be read as a YAML mapping")
        if "budgets" in data:
            limits, problems = _file_budgets(data["budgets"])
            for key, value in limits.items():
                budgets[key] = value if budgets[key] is None else min(budgets[key], value)
            budget_errors += [f"{path}: {p}" for p in problems]
        for key in flags:
            if key in data:
                flags[key].append(bool(data[key]))
        if "roots" in data:
            listed = [os.path.realpath(runs.expand(str(r))) for r in (data.get("roots") or [])]
            if roots is None:
                roots = listed
            else:
                roots = [r for r in listed if any(runs._under(r, base) for base in roots)] + [
                    base for base in roots if any(runs._under(base, r) for r in listed) and base not in listed
                ]
    return {
        "roots": roots,
        "redact": not flags["redact"] or any(flags["redact"]),
        "audit": not flags["audit"] or any(flags["audit"]),
        "budgets": budgets,
        "budget_errors": budget_errors,
        "files": _policy_files(),
    }


# ---------------------------------------------------------------------------
# GPU-hour accounting (in the real home, so a caller's environment can't move or hide it)
# ---------------------------------------------------------------------------

BUDGET_WINDOW_S = 24 * 3600
BUDGET_LOCK_TIMEOUT_S = 120.0


def _budget_dir() -> str:
    directory = os.path.join(owner_home(), ".credit")
    if not os.path.isdir(directory):
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    return directory


def budget_path() -> str:
    return os.path.join(owner_home(), ".credit", "budget.jsonl")


@contextlib.contextmanager
def budget_lock():
    """Hold the exclusive budget lock, so a check and the submit it allows are not interleaved with another's.

    Raises :class:`TimeoutError` after :data:`BUDGET_LOCK_TIMEOUT_S` seconds.
    """
    import fcntl

    fd = os.open(os.path.join(_budget_dir(), "budget.lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + BUDGET_LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("the GPU-hour budget is locked by another submit") from None
                time.sleep(0.1)
        yield
    finally:
        os.close(fd)  # releases the lock


def used_gpu_hours(now: float | None = None) -> float:
    """GPU-hours recorded in :func:`budget_path` over the last 24 h (a line with an unreadable time counts)."""
    now = time.time() if now is None else now
    total = 0.0
    for _, _, line in runs.read_lines(budget_path())[0]:
        hours = line.get("gpu_hours") if isinstance(line, dict) else None
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours):
            continue
        ts = runs.parse_ts(line.get("ts"))
        if ts is None or ts > now - BUDGET_WINDOW_S:
            total += hours
    return round(total, 4)


def record_gpu_hours(run_id: str, gpu_hours: float) -> None:
    """Append ``{ts, run_id, gpu_hours}`` to :func:`budget_path` (created 0600, its directory 0700)."""
    _budget_dir()
    line = runs.encode_line({"ts": runs.now_ts(), "run_id": run_id, "gpu_hours": round(float(gpu_hours), 4)})
    fd = os.open(budget_path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def redact(text):
    """Return ``(text, count)`` with secrets replaced by :data:`MARK`; non-strings pass through."""
    if not isinstance(text, str) or not text or not policy_redacts():
        return text, 0
    count = 0
    for pattern in _TOKEN_PATTERNS:
        text, n = pattern.subn(MARK, text)
        count += n
    for pattern in (_URL_PASSWORD, _BEARER):
        text, n = pattern.subn(lambda m: m.group(1) + MARK + (m.group(3) if m.lastindex >= 3 else ""), text)
        count += n
    text, n = _KEY_VALUE.subn(lambda m: m.group(1) + MARK, text)
    return text, count + n


_REDACTS = None


def policy_redacts() -> bool:
    global _REDACTS
    if _REDACTS is None:
        _REDACTS = policy()["redact"]
    return _REDACTS


def redact_event(event: dict) -> dict:
    """Mask secrets in the free-text fields of ``log`` and ``error`` events."""
    if event.get("type") in ("log", "error") and isinstance(event.get("data"), dict):
        data = dict(event["data"])
        for key in ("message", "traceback_tail"):
            data[key], _ = redact(data.get(key))
        event = {**event, "data": data}
    return event


_AUDIT_KEYS = (
    "path",
    "run_id",
    "job_key",
    "kind",
    "scope",
    "config",
    "save_loc",
    "scheduler",
    "cluster",
    "page",
    "name",
    "confirm_gpu_hours",
)


def audit(verb: str, req: dict, response: dict) -> None:
    """Append one line describing the call to ``$CREDIT_HOME/audit.jsonl`` (never file contents)."""
    try:
        if not policy()["audit"]:
            return
        summary = {k: str(req[k])[:300] for k in _AUDIT_KEYS if isinstance(req, dict) and req.get(k)}
        if isinstance(req, dict) and isinstance(req.get("request"), dict):  # submit
            summary.update(
                {k: str(req["request"][k])[:300] for k in ("run_id", "kind", "config") if req["request"].get(k)}
            )
        if isinstance(req, dict) and isinstance(req.get("runs"), dict):  # sync / follow
            summary["runs"] = len(req["runs"])
        line = {
            "ts": runs.now_ts(),
            "verb": verb,
            "ok": bool(response.get("ok")),
            "code": (response.get("error") or {}).get("code"),
            "client": (os.environ.get("SSH_CONNECTION") or "").split(" ")[0] or None,
            "req": summary,
        }
        runs.append_line(os.path.join(runs.credit_home(), "audit.jsonl"), line, lock=True)
    except Exception:  # noqa: BLE001 — auditing must never break a verb
        pass
