"""``credit.remote``: the HPC side of the CREDIT MCP / Studio protocol.

Run as ``python -P -m credit.remote <verb>`` with one JSON request on stdin
(empty means ``{}``).  The response is one line on the original stdout::

    \\x1eCREDIT\\x1e{"ok": true, "protocol": 1, "result": {...}}
    \\x1eCREDIT\\x1e{"ok": false, "protocol": 1, "error": {"code", "message", "fix"}}

Everything else a process might print (``.bashrc`` noise, library warnings)
goes to stderr: on entry, fd 1 is duplicated onto fd 2 and the response is
written to the saved descriptor.  The exit code is 0 whenever a response was
written.  ``follow`` instead streams one framed line per frame.

The protocol and every verb are specified in ``docs/source/remote-protocol.md``.
All but ``check``, ``components``, ``build_config``, ``docs`` and ``render``
import only the stdlib, ``yaml`` and ``credit.runs``; ``plan`` and ``submit``
also import the (torch-free) ``credit.cli`` script builders.
"""

import base64
import contextlib
import fnmatch
import glob
import hashlib
import io
import json
import os
import queue
import re
import socket
import sys
import threading
import time

from credit import runs
from credit.remote import _delta, _guard, _sched

PROTOCOL = 1
# How a user chooses the folders credit-mcp may serve (an explicit, local action; never an MCP tool).
SETUP_COMMAND = "credit remote-setup"
SENTINEL = b"\x1eCREDIT\x1e"
READ_DEFAULT = 1 << 20
READ_MAX = 8 << 20
GREP_MAX = 4 << 20
_JOB_ID = re.compile(r"^\d+(\[\d*\])?(\.[\w.-]+)?$")
SHA_LIMIT = 16 << 20
DIAGNOSE_LIMIT = 64 << 20
DEFAULT_FIELDS = ["loss.mean", "loss_epoch", "lr", "grad_norm"]
BANNERS = (
    "DATA LOADING HANG DETECTED",
    "CUDA out of memory",
    "OutOfMemoryError",
    "RendezvousConnectionError",
    "RendezvousTimeoutError",
    "NCCL error",
    "DUE TO TIME LIMIT",
    "walltime",
    "Killed",
    "Segmentation fault",
    "PBS: job killed",
)


class RemoteError(Exception):
    """An application error returned in the response envelope; *data* is optional machine-readable detail."""

    def __init__(self, code: str, message: str, fix: str | None = None, data: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.fix, self.data = code, message, fix, data


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


class _Out:
    """Writes sentinel-framed JSON lines to the real stdout; fd 1 itself is pointed at stderr."""

    def __init__(self):
        sys.stdout.flush()
        self.fd = os.dup(1)
        os.dup2(2, 1)
        self.lock = threading.Lock()

    def send(self, obj: dict) -> None:
        data = SENTINEL + json.dumps(runs.jsonable(obj), separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with self.lock:
            view = memoryview(data)
            while view:
                view = view[os.write(self.fd, view) :]


class _TeeStderr(io.TextIOBase):
    """Passes writes through to stderr and keeps the tail, so a ``SystemExit`` can report what was printed."""

    def __init__(self, stream):
        self.stream, self.buf = stream, ""

    def write(self, s):
        self.stream.write(s)
        self.buf = (self.buf + s)[-4000:]
        return len(s)

    def flush(self):
        self.stream.flush()


def _error(code: str, message: str, fix: str | None = None, data: dict | None = None) -> dict:
    return {"ok": False, "protocol": PROTOCOL, "error": {"code": code, "message": message, "fix": fix, "data": data}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    """The checkout or install root that holds ``docs/`` and ``config/`` (same as ``credit.cli._repo_root``)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))


def _require(req: dict, key: str, kind=str):
    value = req.get(key)
    if value is None or not isinstance(value, kind):
        raise RemoteError("bad_request", f"'{key}' is required")
    return value


def _run_entry(req: dict) -> tuple:
    """Return ``(ledger_entry, run.json record)`` for ``req['run_id']``."""
    run_id = _require(req, "run_id")
    entry = runs.find_run(run_id)
    if entry is None:
        raise RemoteError("not_found", f"no run {run_id!r} in the ledger")
    try:
        record = runs.load_run(entry["run_dir"])
    except (OSError, ValueError) as exc:
        raise RemoteError("not_found", f"run {run_id}: cannot read run.json ({exc})") from exc
    return entry, record


def _job_log(record: dict, job_key: str | None) -> tuple:
    """Return ``(job_key, log_path)``: the named job, or the latest job that has a log."""
    jobs = sorted(record.get("jobs") or [], key=lambda j: j.get("index") or 0, reverse=True)
    for job in jobs:
        key = job.get("job_key")
        if job_key not in (None, key):
            continue
        path = _delta.find_log(record, key) if key else None
        if path:
            return key, path
    raise RemoteError("not_found", f"no log file for job {job_key}" if job_key else "no job has a log file yet")


def _dotted(data, path: str):
    for part in path.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    if isinstance(data, bool) or not isinstance(data, (int, float)):
        return None
    return data


def _yaml_load(path: str) -> dict:
    import yaml

    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


# ---------------------------------------------------------------------------
# Light verbs
# ---------------------------------------------------------------------------


def hello(req: dict) -> dict:
    import getpass
    import platform
    import shutil

    pbs = shutil.which(runs.scheduler_bin("qstat", "pbs"))
    pol = _guard.policy()
    return {
        "protocol": PROTOCOL,
        "credit_version": runs.credit_version(),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "system": runs.system_name(socket.gethostname()),
        "schedulers": {"pbs": bool(pbs), "slurm": bool(shutil.which(runs.scheduler_bin("sbatch", "slurm")))},
        "credit_home": runs.credit_home(),
        "allowed_roots": runs.allowed_roots(),
        "repo_root": _repo_root(),
        "policy": {
            "configured": runs.policy_configured(),
            "files": pol["files"],
            "redact": pol["redact"],
            "audit": pol["audit"],
            "budgets": pol["budgets"],
            "budget_errors": pol["budget_errors"],
            "used_gpu_hours_24h": _used_gpu_hours(),
            "setup_command": SETUP_COMMAND,
        },
    }


def _used_gpu_hours() -> float | None:
    try:
        return _guard.used_gpu_hours()
    except OSError:
        return None


def sync(req: dict) -> dict:
    as_of = runs.now_ts()
    progress = req.get("progress", "all")
    if progress not in ("all", "latest"):
        raise RemoteError("bad_request", "progress must be all or latest")
    budget = [int(req.get("max_bytes") or 4 << 20)]
    log_tail = int(req.get("log_tail_bytes") or 65536)
    lines, new_offset, reset = runs.read_ledger(int(req.get("ledger_offset") or 0))
    index = runs.ledger_index()
    cursors = dict(req.get("runs") or {})
    for line in lines:
        if line.get("run_id") and not line.get("removed"):
            cursors.setdefault(line["run_id"], None)

    out, jobs = {}, []
    for run_id, cursor in cursors.items():
        entry = index.get(run_id)
        if entry is None:
            out[run_id] = {
                "run": None,
                "events": [],
                "csv": None,
                "csv_pending": False,
                "logs": {},
                "cursor": cursor,
                "missing": True,
                "more": False,
            }
            continue
        delta, record = _delta.run_delta(entry, cursor, progress, log_tail, budget)
        out[run_id] = delta
        scheduler = ((record or {}).get("target") or {}).get("scheduler") or "pbs"
        jobs += [(scheduler, j["job_id"], entry["run_dir"]) for j in _sched.unfinished_jobs(entry["run_dir"], record)]

    states, errors = _sched.query(jobs) if req.get("scheduler", True) else ({}, {})
    result = {
        "as_of": as_of,
        "ledger_offset": new_offset,
        "ledger": lines,
        "runs": out,
        "scheduler": states,
        "scheduler_errors": errors,
    }
    if reset:
        result["ledger_reset"] = True
    return result


def follow(out: _Out, stdin) -> None:
    """Stream frames for a changing set of runs until stdin reaches EOF (docs/source/remote-protocol.md, "follow")."""
    out.send({"type": "hello", "protocol": PROTOCOL, "ts": runs.now_ts()})
    messages = queue.Queue()

    def reader():
        for line in stdin:
            if line.strip():
                try:
                    messages.put(json.loads(line))
                except ValueError:
                    messages.put({"_bad": line.strip()[:200]})
        messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    cursors, entries = {}, {}
    state = {"interval": 2.0, "log_tail": 16384}

    def send_error(run_id: str | None, code: str, message: str) -> None:
        out.send({"type": "error", "run_id": run_id, "error": {"code": code, "message": message, "fix": None}})

    def add(run_cursors: dict) -> None:
        index = runs.ledger_index()
        for run_id, cursor in (run_cursors or {}).items():
            if run_id not in index:
                send_error(run_id, "not_found", f"no run {run_id!r} in the ledger")
                continue
            entries[run_id], cursors[run_id] = index[run_id], cursor

    def handle(msg) -> bool:
        if msg is None:
            return False
        if not isinstance(msg, dict) or "_bad" in msg:
            send_error(None, "bad_request", "bad control message")
            return True
        add(msg.get("runs") or {})
        add(msg.get("add") or {})
        for run_id in msg.get("remove") or []:
            cursors.pop(run_id, None)
            entries.pop(run_id, None)
        if msg.get("interval_s"):
            state["interval"] = max(0.2, float(msg["interval_s"]))
        if msg.get("log_tail_bytes"):
            state["log_tail"] = int(msg["log_tail_bytes"])
        return True

    if not handle(messages.get()):
        return
    last_sent = time.monotonic()
    while True:
        frame = {}
        for run_id in list(cursors):
            try:
                delta, _ = _delta.run_delta(entries[run_id], cursors[run_id], "all", state["log_tail"], [4 << 20])
            except Exception as exc:  # noqa: BLE001 — one bad run must not end the stream
                send_error(run_id, "internal", f"{type(exc).__name__}: {exc}")
                continue
            cursors[run_id] = delta["cursor"]
            if _delta.delta_changed(delta):
                frame[run_id] = delta
            if delta["missing"]:  # reported once, then dropped
                cursors.pop(run_id, None)
                entries.pop(run_id, None)
        if frame:
            out.send({"type": "frame", "ts": runs.now_ts(), "runs": frame})
            last_sent = time.monotonic()
        elif time.monotonic() - last_sent >= 15:
            out.send({"type": "ping", "ts": runs.now_ts()})
            last_sent = time.monotonic()
        deadline = time.monotonic() + state["interval"]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = messages.get(timeout=remaining)
            except queue.Empty:
                break
            if not handle(msg):
                return


def read(req: dict) -> dict:
    real = runs.resolve_in_scope(_require(req, "path"))
    if os.path.isdir(real):
        raise RemoteError("bad_request", f"{real} is a directory; use ls")
    if not os.path.isfile(real):
        raise RemoteError("not_found", f"{real} does not exist")
    st = os.stat(real)
    offset = max(0, int(req.get("offset") or 0))
    length = min(READ_MAX, max(0, int(req.get("length") or READ_DEFAULT)))
    with open(real, "rb") as f:
        sample = f.read(8192)
        f.seek(offset)
        data = f.read(length)
    encoding = req.get("encoding", "auto")
    if encoding == "auto":
        encoding = "base64" if b"\0" in sample else "utf-8"
    redacted = 0
    if encoding == "utf-8":
        text, redacted = _guard.redact(data.decode("utf-8", errors="replace"))
    elif encoding == "base64":
        text = base64.b64encode(data).decode("ascii")
    else:
        raise RemoteError("bad_request", "encoding must be auto, utf-8 or base64")
    return {
        "path": real,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "sha256": runs.sha256_file(real) if st.st_size <= SHA_LIMIT else None,
        "offset": offset,
        "data": text,
        "encoding": encoding,
        "eof": offset + len(data) >= st.st_size,
        "redacted": redacted,
    }


def ls(req: dict) -> dict:
    pattern = req.get("pattern") or "*"
    if "/" in pattern:
        raise RemoteError("bad_request", "pattern matches entry names and must not contain '/'")
    real = runs.resolve_in_scope(_require(req, "path"))
    if not os.path.isdir(real):
        raise RemoteError("not_found", f"{real} is not a directory")
    limit = max(1, int(req.get("limit") or 1000))
    entries, truncated = [], False
    for name in sorted(os.listdir(real)):
        if not fnmatch.fnmatch(name, pattern):
            continue
        child = os.path.join(real, name)
        if name.startswith("."):
            try:
                runs.resolve_in_scope(child)
            except runs.PathScopeError:
                continue
        if len(entries) >= limit:
            truncated = True
            break
        try:
            st = os.stat(child)
        except OSError:
            continue
        kind = "dir" if os.path.isdir(child) else "file"
        entries.append({"name": name, "type": kind, "size": st.st_size, "mtime": st.st_mtime})
    return {"path": real, "entries": entries, "truncated": truncated}


def logs(req: dict) -> dict:
    _, record = _run_entry(req)
    job_key, path = _job_log(record, req.get("job_key"))
    real = runs.resolve_in_scope(path)
    tail = int(req.get("tail_bytes") or 65536)
    r = _delta.tail_log(real, min(tail, GREP_MAX) if req.get("grep") else tail)
    text = r["text"]
    if req.get("grep"):
        try:
            pattern = re.compile(req["grep"])
        except re.error as exc:
            raise RemoteError("bad_request", f"bad grep pattern: {exc}") from exc
        text = "".join(line + "\n" for line in text.splitlines() if pattern.search(line))
    return {
        "job_key": job_key,
        "path": real,
        "text": _guard.redact(text)[0],
        "current_frame": _guard.redact(r["current_frame"])[0],
        # The cursor after the last complete line: what a sync of this log would continue from, so a reader can tell
        # which streamed appends it already has. (An unfinished last line in `text` lies past it.)
        "offset": r["off"],
    }


def write(req: dict) -> dict:
    text = _require(req, "text")
    real = runs.resolve_in_scope(_require(req, "path"))
    runs.check_writable(real)
    if _guard.MARK in text:
        raise RemoteError(
            "config_error",
            f"the new text still contains {_guard.MARK} placeholders, so writing it would erase secrets",
            fix="edit this file by hand on the HPC system",
        )
    expected = req.get("if_sha256")
    exists = os.path.exists(real)
    if expected is None and exists:
        raise RemoteError("conflict", f"{real} already exists", fix="pass if_sha256 with its current sha256")
    if expected is not None and (not exists or runs.sha256_file(real) != expected):
        raise RemoteError("conflict", f"{real} changed since it was read", fix="read it again and reapply the edit")
    data = text.encode("utf-8")
    runs.atomic_write(real, data)
    return {"path": real, "sha256": hashlib.sha256(data).hexdigest()}


def history(req: dict) -> dict:
    entry, _ = _run_entry(req)
    fields = req.get("fields") or DEFAULT_FIELDS
    buckets = max(1, int(req.get("buckets") or 1500))
    t0 = runs.parse_ts(req["t0"]) if req.get("t0") else None
    t1 = runs.parse_ts(req["t1"]) if req.get("t1") else None
    plain = [f for f in fields if not f.startswith("vars.")]
    var_fields = [f for f in fields if f.startswith("vars.")]
    points = []
    events_dir = os.path.join(entry["run_dir"], "events")
    names = sorted(n for n in os.listdir(events_dir) if n.endswith(".proc.jsonl")) if os.path.isdir(events_dir) else []
    for name in names:
        schema = []  # train.vars aligns to the latest metrics.schema in the same file
        for _, _, ev in runs.read_lines(os.path.join(events_dir, name))[0]:
            etype, data = ev.get("type"), ev.get("data") or {}
            if etype == "metrics.schema":
                schema = list(data.get("columns") or [])
                continue
            if etype not in ("train.progress", "train.vars"):
                continue
            t = runs.parse_ts(ev.get("ts"))
            if t is None or (t0 is not None and t < t0) or (t1 is not None and t > t1):
                continue
            if etype == "train.progress":
                values = {f: _dotted(data, f) for f in plain}
            else:
                row = data.get("values") or []
                values = {}
                for f in var_fields:
                    col = f[len("vars.") :]
                    i = schema.index(col) if col in schema else -1
                    v = row[i] if 0 <= i < len(row) else None
                    values[f] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            if any(v is not None for v in values.values()):
                points.append((t, values))
    empty = {f: {"min": [], "mean": [], "max": []} for f in fields}
    if not points:
        return {"t": [], "fields": empty, "points": 0}
    lo = t0 if t0 is not None else min(p[0] for p in points)
    hi = t1 if t1 is not None else max(p[0] for p in points)
    width = (hi - lo) / buckets or 1.0
    acc = {}
    for t, values in points:
        b = acc.setdefault(min(int((t - lo) / width), buckets - 1), {"t": [], "f": {}})
        b["t"].append(t)
        for f, v in values.items():
            if v is not None:
                b["f"].setdefault(f, []).append(v)
    result = {"t": [], "fields": empty, "points": len(points)}
    for key in sorted(acc):
        b = acc[key]
        result["t"].append(runs.now_ts(sum(b["t"]) / len(b["t"])))
        for f in fields:
            vals = b["f"].get(f)
            result["fields"][f]["min"].append(min(vals) if vals else None)
            result["fields"][f]["mean"].append(sum(vals) / len(vals) if vals else None)
            result["fields"][f]["max"].append(max(vals) if vals else None)
    return result


def cancel(req: dict) -> dict:
    entry, record = _run_entry(req)
    scope = req.get("scope", "all")
    if scope not in ("all", "pending"):
        raise RemoteError("bad_request", "scope must be all or pending")
    run_dir = entry["run_dir"]
    scheduler = (record.get("target") or {}).get("scheduler") or "pbs"
    active = _sched.unfinished_jobs(run_dir, record)
    states, sched_errors = _sched.query([(scheduler, j["job_id"], run_dir) for j in active])
    if scope == "pending" and sched_errors:
        raise RemoteError("scheduler_error", "; ".join(sched_errors.values()))
    wanted = ("queued", "held") if scope == "pending" else ("queued", "held", "running", "unknown")
    cancelled, errors = [], {}
    # Last job first, so deleting a parent does not make PBS drop its afterok
    # dependents before we reach them.
    for job in sorted(active, key=lambda j: j.get("index") or 0, reverse=True):
        jid = job["job_id"]
        if states.get(jid, {}).get("state", "unknown") not in wanted:
            continue
        try:
            _sched.cancel_job(scheduler, jid, run_dir)
            cancelled.append(jid)
        except runs.SchedulerError as exc:
            errors[jid] = str(exc)
    if cancelled:
        runs.append_event(
            run_dir,
            runs.SUBMIT_EVENTS,
            runs.make_event("run.cancelled", {"scope": scope, "job_ids": cancelled}, "submit"),
        )
        record = runs.load_run(run_dir)
        record["cancelled"] = {"at": runs.now_ts(), "scope": scope, "job_ids": cancelled}
        runs.write_run(run_dir, record)
    return {"cancelled": cancelled, "errors": errors}


def _submit_module():
    import importlib

    # By module path: ``credit.cli._submit`` the attribute is the CLI function.
    return importlib.import_module("credit.cli._submit")


def _config_error(exc) -> RemoteError:
    """A :class:`credit.cli._submit.SubmitError` as a ``config_error``; ``data.field`` names the bad plan option."""
    field = getattr(exc, "field", None)
    return RemoteError("config_error", str(exc), data={"field": field} if field else None)


def plan(req: dict) -> dict:
    _submit = _submit_module()
    try:
        return {"plan": _submit._plan(req)}
    except _submit.SubmitError as exc:
        raise _config_error(exc) from exc


def submit(req: dict) -> dict:
    _submit = _submit_module()
    request = req.get("request")
    if not isinstance(request, dict):
        raise RemoteError("bad_request", "'request' (the plan's echoed request) is required")
    confirm = req.get("confirm_gpu_hours")
    if confirm is not None and (isinstance(confirm, bool) or not isinstance(confirm, (int, float))):
        raise RemoteError("bad_request", "confirm_gpu_hours must be a number (the GPU-hours a person typed)")
    args = (request, req.get("scripts_sha256") or [], req.get("config_sha256"), req.get("created_by") or "mcp")
    pol = _guard.policy()
    budgeted = pol["budget_errors"] or any(v is not None for v in pol["budgets"].values())
    try:
        if not budgeted or (request.get("scheduler") or "pbs") == "local":
            return _submit._submit_plan(*args)
        return _submit_within_budget(_submit, args, pol, confirm)
    except _submit.SubmitConflict as exc:
        raise RemoteError("conflict", str(exc), fix="plan again and approve the new plan") from exc
    except _submit.SubmitError as exc:
        raise _config_error(exc) from exc


def _submit_within_budget(_submit, args: tuple, pol: dict, confirm) -> dict:
    """:func:`submit` for pbs/slurm when a GPU-hour budget is configured (see ``budgets`` in ``_guard``)."""
    request = args[0]
    if pol["budget_errors"]:
        raise RemoteError(
            "over_budget",
            "the GPU-hour budget in remote.yaml is invalid, so nothing may be submitted: "
            + "; ".join(pol["budget_errors"]),
            fix="fix the budgets block in the policy file, e.g. with `credit remote-setup --max-gpu-hours-per-day N`",
            data={"kind": "invalid", "estimate": None, "limit": 0},
        )
    if not request.get("run_id"):
        raise _submit.SubmitError("submit needs request.run_id (from the plan)")
    plan = _submit._plan(request)
    estimate, n_jobs = plan["gpu_hours_estimate"], len(plan["jobs"])
    if estimate is None or not n_jobs:
        raise RemoteError(
            "over_budget",
            "GPU-hours can't be estimated for this plan (no walltime), so it can't be checked against the budget",
            fix="set a walltime (options.walltime, or walltime in the config's pbs/slurm block) and plan again",
            data={"kind": "no_estimate", "estimate": None, "limit": None},
        )
    per_plan, per_day = pol["budgets"]["max_gpu_hours_per_plan"], pol["budgets"]["max_gpu_hours_per_day"]
    if per_plan is not None and estimate > per_plan and not (confirm is not None and confirm >= estimate):
        raise RemoteError(
            "over_budget",
            f"this plan is estimated at {estimate:g} GPU-hours, over the per-plan limit of {per_plan:g} GPU-hours",
            fix=f"a person has to confirm the GPU-hours: they type {estimate:g} to approve the plan "
            "(sent as confirm_gpu_hours); or plan a smaller run",
            data={"kind": "per_plan", "estimate": estimate, "limit": per_plan},
        )
    per_job = estimate / n_jobs
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(_guard.budget_lock())
        except TimeoutError as exc:  # before OSError: TimeoutError is one
            raise RemoteError("conflict", str(exc), fix="retry in a minute") from None
        except OSError as exc:
            raise RemoteError("over_budget", f"cannot open the GPU-hour ledger: {exc}") from None
        # Jobs already submitted (or with an outcome-less intent) are not submitted again, so they cost nothing.
        intents, before = _submit._submit_progress(plan["run_dir"])
        new_jobs = [j for j in plan["jobs"] if j["index"] not in before and j["index"] not in intents]
        new_hours = per_job * len(new_jobs)
        if per_day is not None and new_jobs:
            try:
                used = _guard.used_gpu_hours()
            except OSError as exc:
                raise RemoteError("over_budget", f"cannot read the GPU-hour ledger: {exc}") from None
            if used + new_hours > per_day + 1e-9:
                raise RemoteError(
                    "over_budget",
                    f"this submit adds {new_hours:g} GPU-hours to the {used:g} submitted in the last 24 hours, "
                    f"over the daily limit of {per_day:g} GPU-hours",
                    fix="wait for earlier submissions to leave the 24-hour window, or plan a smaller run; "
                    "the daily limit has no override",
                    data={
                        "kind": "per_day",
                        "estimate": estimate,
                        "limit": per_day,
                        "new_gpu_hours": round(new_hours, 4),
                        "used_gpu_hours_24h": used,
                    },
                )
        try:
            return _submit._submit_plan(*args)
        finally:
            _, after = _submit._submit_progress(plan["run_dir"])
            newly = len(set(after) - set(before))
            if newly:
                _guard.record_gpu_hours(plan["run_id"], per_job * newly)


def _in_static_scope(path: str) -> str:
    """:func:`credit.runs.resolve_in_scope` (with the dot rule) restricted to the static roots."""
    real = runs.resolve_in_scope(path)
    if not any(runs._under(real, r) for r in runs.static_roots()):
        raise runs.PathScopeError(f"{path} is outside the allowed roots")
    return real


def _request_config(path: str) -> dict:
    """Load a request's config file: a ``.yml``/``.yaml`` mapping, errors without file contents."""
    try:
        return runs.load_yaml_mapping(path)[1]
    except runs.ConfigFileError as exc:
        raise RemoteError("config_error", str(exc)) from None


def adopt(req: dict) -> dict:
    save_loc = _in_static_scope(_require(req, "save_loc"))
    kind = req.get("kind") or "train"
    if kind not in runs.KINDS:
        raise RemoteError("bad_request", f"kind must be one of {', '.join(runs.KINDS)}")
    config = req.get("config") or next(
        (os.path.join(save_loc, n) for n in ("model.yml", "config.yml") if os.path.isfile(os.path.join(save_loc, n))),
        None,
    )
    if not config:
        raise RemoteError("bad_request", "no config given and none found in save_loc (model.yml)")
    config = _in_static_scope(config)
    if not os.path.isfile(config):
        raise RemoteError("not_found", f"{config} does not exist")
    conf = _request_config(config)
    scheduler = req.get("scheduler") or "pbs"
    if scheduler not in ("pbs", "slurm", "local"):
        raise RemoteError("bad_request", "scheduler must be pbs, slurm or local")
    job_ids = [str(j) for j in req.get("job_ids") or []]
    bad = [j for j in job_ids if not _JOB_ID.match(j)]
    if bad:
        raise RemoteError("bad_request", f"not a scheduler job id: {bad[0][:40]!r}")
    servers = {j.split(".", 1)[1] for j in job_ids if scheduler == "pbs" and "." in j}
    target = {
        "scheduler": scheduler,
        "cluster": runs.system_name(socket.gethostname()),
        "server": servers.pop() if len(servers) == 1 else None,
        "queue": None,
        "account": None,
        "nodes": None,
        "gpus": None,
        "cpus": None,
        "mem": None,
        "walltime": None,
        "log_dir": os.path.join(save_loc, "logs"),
    }
    trainer = conf.get("trainer") or {}
    run_id = runs.new_run_id(kind)
    record = runs.create_run(
        run_id,
        kind,
        config,
        save_loc,
        target,
        created_by=req.get("created_by") or "mcp",
        name=req.get("name"),
        epochs_total=trainer.get("epochs") if kind == "train" else None,
        epochs_per_job=trainer.get("num_epoch") if kind == "train" else None,
        csv_baseline=False,  # an adopted run's training_log.csv rows are its own history
    )
    run_dir = runs.run_dir_for(save_loc, run_id)
    for i, jid in enumerate(job_ids, 1):
        runs.add_job(run_dir, record, {"index": i, "job_id": jid, "config": config, "script": None, "submitted": None})
    return {"run_id": run_id, "run_dir": run_dir}


def diagnose(req: dict) -> dict:
    entry, record = _run_entry(req)
    job_key, path = _job_log(record, req.get("job_key"))
    texts = []
    for p in [path] + sorted(
        glob.glob(os.path.join((record.get("target") or {}).get("log_dir") or "", "pbsdsh_logs", job_key, "node_*.out"))
    ):
        if p in texts or not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, size - DIAGNOSE_LIMIT))
            text = "\n".join(_delta._collapse_lines(f.read().replace(b"\r\n", b"\n") + b"\n"))
            texts.append(_guard.redact(text)[0])
    lines = "\n".join(texts).splitlines()
    traceback = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("Traceback (most recent call last)"):
            j = i + 1
            while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                j += 1
            traceback = "\n".join(lines[i : j + 1])
            break
    banners = [ln.strip() for ln in lines if any(b in ln for b in BANNERS)][-20:]
    done = _delta.job_finished(entry["run_dir"], job_key) or {}
    return {
        "job_key": job_key,
        "exit_status": done.get("code"),
        "signal": done.get("signal"),
        "traceback": traceback,
        "banners": banners,
        "excerpt": "\n".join(lines[-60:]),
    }


def outputs(req: dict) -> dict:
    entry, record = _run_entry(req)
    save_loc = runs.resolve_in_scope(record.get("save_loc") or entry.get("save_loc"))
    checkpoints = []
    for name in sorted(os.listdir(save_loc)) if os.path.isdir(save_loc) else []:
        if name.endswith(".pt"):
            st = os.stat(os.path.join(save_loc, name))
            checkpoints.append({"path": os.path.join(save_loc, name), "size": st.st_size, "mtime": st.st_mtime})
    conf = _yaml_load(os.path.join(entry["run_dir"], "config.yml"))
    forecast = (conf.get("inference") or {}).get("save_forecast") or (conf.get("predict") or {}).get("save_forecast")
    forecast_dir = runs.expand(forecast) if forecast else None
    files, truncated, listable = [], False, False
    if forecast_dir:
        try:
            forecast_dir = runs.resolve_in_scope(forecast_dir)
            listable = os.path.isdir(forecast_dir)
        except runs.PathScopeError:
            listable = False  # named, but not listed: it is outside the allowed roots
    if listable:
        for root, dirs, names in os.walk(forecast_dir):
            dirs.sort()
            for name in sorted(names):
                if len(files) >= 2000:
                    truncated = True
                    break
                p = os.path.join(root, name)
                st = os.stat(p)
                files.append({"path": p, "size": st.st_size, "mtime": st.st_mtime})
            if truncated:
                break
    return {"checkpoints": checkpoints, "forecast_dir": forecast_dir, "files": files, "truncated": truncated}


def _named_files(directory: str, suffix: str, name: str | None, key: str) -> dict:
    """List ``<directory>/*<suffix>`` names, or return one file's text (only names from the listing)."""
    names = (
        sorted(n[: -len(suffix)] for n in os.listdir(directory) if n.endswith(suffix))
        if os.path.isdir(directory)
        else []
    )
    if name is None:
        return {f"{key}s": names}
    name = name[: -len(suffix)] if name.endswith(suffix) else name
    if name not in names:
        raise RemoteError("not_found", f"no {key} {name!r}")
    with open(os.path.join(directory, name + suffix)) as f:
        return {key if key != "example" else "name": name, "text": f.read()}


def examples(req: dict) -> dict:
    return _named_files(os.path.join(_repo_root(), "config", "gen_2", "examples"), ".yml", req.get("name"), "example")


def docs(req: dict) -> dict:
    return _named_files(os.path.join(_repo_root(), "docs", "source"), ".md", req.get("page"), "page")


def convert(req: dict) -> dict:
    raise RemoteError("unsupported", "convert is not available over the remote protocol yet", fix="run credit convert")


def render(req: dict) -> dict:
    raise RemoteError("unsupported", "render is not implemented in protocol v1", fix="run credit plot on the host")


# ---------------------------------------------------------------------------
# Heavy verbs (import torch & co. lazily)
# ---------------------------------------------------------------------------


def _need_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RemoteError(
            "needs_dependency", f"this verb needs PyTorch: {exc}", fix="use an env with CREDIT installed"
        ) from exc


def check(req: dict) -> dict:
    import tempfile

    _need_torch()
    from credit.cli._check import _load_config as _check_load_config
    from credit.cli._check import _Report, _run_checks

    with tempfile.TemporaryDirectory() as tmp:
        if req.get("yaml") is not None:
            path = os.path.join(tmp, "config.yml")
            with open(path, "w") as f:
                f.write(req["yaml"])
        else:
            path = runs.resolve_in_scope(_require(req, "config"))
            _request_config(path)  # a .yml/.yaml mapping, before credit check quotes any of it
        rep = _Report(path if req.get("yaml") is None else "<yaml>")
        conf = _check_load_config(path, rep)
        if conf is not None:
            _run_checks(conf, rep, deep=bool(req.get("deep")))
    derived = {}
    if conf:
        trainer, model = conf.get("trainer") or {}, conf.get("model") or {}
        if trainer.get("epochs") and trainer.get("num_epoch"):
            derived["n_jobs"] = -(-int(trainer["epochs"]) // int(trainer["num_epoch"]))
        if model.get("image_height") and model.get("image_width"):
            derived["grid"] = [model["image_height"], model["image_width"]]
        for key, src in (("channels_in", "channels"), ("channels_out", "output_channels")):
            if isinstance(model.get(src), int):
                derived[key] = model[src]
        if trainer.get("batches_per_epoch"):
            derived["steps_per_epoch"] = trainer["batches_per_epoch"]
    return {"report": rep.to_dict(), "derived": derived}


_REGISTRIES = {
    "models": ("credit.models", "_MODEL_REGISTRY"),
    "trainers": ("credit.trainers", "_TRAINER_REGISTRY"),
    "datasets": ("credit.datasets.gen_2.multi_source", "_SOURCE_REGISTRY"),
    "preblocks": ("credit.preblock", "_PREBLOCK_REGISTRY"),
    "postblocks": ("credit.postblock", "_POSTBLOCK_REGISTRY"),
    "losses": ("credit.losses", "_LOSS_REGISTRY"),
    "metrics": ("credit.metrics", "_METRIC_REGISTRY"),
}


def _registry(kind: str) -> dict:
    import importlib

    module, attr = _REGISTRIES[kind]
    return getattr(importlib.import_module(module), attr)


def _entry_class(entry):
    """Resolve a registry entry: ``(module, name[, msg])`` lazy tuples, ``(cls, msg)`` tuples, or a class."""
    import importlib

    if isinstance(entry, tuple):
        if isinstance(entry[0], str):
            return getattr(importlib.import_module(entry[0]), entry[1])
        return entry[0]
    return entry


def components(req: dict) -> dict:
    import inspect

    _need_torch()
    kind, key = req.get("kind"), req.get("key")
    if kind is not None and kind not in _REGISTRIES:
        raise RemoteError("bad_request", f"kind must be one of {', '.join(_REGISTRIES)}")
    if key is None:
        return {"kinds": {k: sorted(_registry(k)) for k in ([kind] if kind else _REGISTRIES)}}
    if kind is None:
        raise RemoteError("bad_request", "'kind' is required with 'key'")
    registry = _registry(kind)
    if key not in registry:
        raise RemoteError("not_found", f"no {kind[:-1]} {key!r}")
    try:
        cls = _entry_class(registry[key])
    except ImportError as exc:
        raise RemoteError("needs_dependency", str(exc)) from exc
    params = []
    try:
        for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            required = p.default is p.empty
            default = None if required else p.default
            try:
                json.dumps(default)
            except (TypeError, ValueError):
                default = repr(default)
            annotation = None if p.annotation is p.empty else getattr(p.annotation, "__name__", str(p.annotation))
            params.append({"name": p.name, "default": default, "annotation": annotation, "required": required})
    except (TypeError, ValueError):
        pass
    return {
        "kind": kind,
        "key": key,
        "class": cls.__name__,
        "module": cls.__module__,
        "doc": inspect.getdoc(cls),
        "params": params,
    }


def build_config(req: dict) -> dict:
    state = req.get("state")
    if not isinstance(state, dict):
        raise RemoteError("bad_request", "'state' (a credit begin state dict) is required")
    if req.get("write"):
        raise RemoteError("bad_request", "build_config only returns the yaml; save it with the write verb")
    try:
        from credit.cli._begin import _build_config, _dump_config

        text = _dump_config(_build_config(state))
    except ImportError as exc:  # some model presets import the model zoo, and with it torch
        raise RemoteError("needs_dependency", str(exc), fix="use an env with CREDIT installed") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteError(
            "config_error", f"cannot build a config from this state: {type(exc).__name__}: {exc}"
        ) from exc
    return {"yaml": text}


VERBS = {
    "hello": hello,
    "sync": sync,
    "read": read,
    "ls": ls,
    "logs": logs,
    "write": write,
    "history": history,
    "cancel": cancel,
    "plan": plan,
    "submit": submit,
    "adopt": adopt,
    "diagnose": diagnose,
    "outputs": outputs,
    "examples": examples,
    "convert": convert,
    "check": check,
    "components": components,
    "build_config": build_config,
    "docs": docs,
    "render": render,
}


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = _Out()
    verb = argv[0] if argv else ""
    if verb == "follow":
        _guard.audit("follow", {}, {"ok": True})
        try:
            follow(out, sys.stdin)
        except KeyboardInterrupt:
            pass
        return 0
    tee = _TeeStderr(sys.stderr)
    req = {}
    try:
        if verb not in VERBS:
            raise RemoteError(
                "bad_request", f"unknown verb {verb!r}; expected one of {', '.join(sorted(VERBS))}, follow"
            )
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        try:
            req = json.loads(raw) if raw.strip() else {}
        except ValueError as exc:
            raise RemoteError("bad_request", f"request is not JSON: {exc}") from exc
        if not isinstance(req, dict):
            raise RemoteError("bad_request", "request must be a JSON object")
        with contextlib.redirect_stderr(tee):
            result = VERBS[verb](req)
        response = {"ok": True, "protocol": PROTOCOL, "result": result}
    except RemoteError as exc:
        response = _error(exc.code, exc.message, exc.fix, exc.data)
    except runs.PathScopeError as exc:
        fix = None
        if not runs.policy_configured():
            fix = f"No folders are set up for credit-mcp on this host yet. In a shell on it, run `{SETUP_COMMAND}`."
        response = _error("forbidden_path", str(exc), fix)
    except runs.SchedulerError as exc:
        response = _error("scheduler_error", str(exc))
    except SystemExit as exc:  # a reused CLI helper gave up
        response = _error("config_error", tee.buf.strip()[-1000:] or f"exited with status {exc.code}")
    except Exception as exc:  # noqa: BLE001 — the envelope must always be written
        import traceback

        # Bounded: the exception type and message, and at most the last 3 frames.
        frames = "".join(traceback.format_list(traceback.extract_tb(exc.__traceback__)[-3:]))
        response = _error("internal", f"{type(exc).__name__}: {str(exc)[:1000]}\n{frames}")
    _guard.audit(verb, req, response)
    out.send(response)
    return 0
