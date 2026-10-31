"""Run records: run ids, the run directory, ``run.json``, the ledger, and path scope.

A *run* is one ``credit submit`` (or MCP / GUI submission): one or more chained
scheduler jobs that share a config.  Every run has a directory on the shared
filesystem that holds its own record::

    <save_loc>/.credit/runs/<run_id>/
        run.json                    written only by submit, cancel and adopt
        config.yml                  snapshot of the config at submit time
        scripts/job-<index>.sh      the exact scripts that were submitted
        events/submit.jsonl         run.created, job.submitted, run.cancelled
        events/<job_key>.script.jsonl   job.started / job.exited / job.signaled
        events/<job_key>.proc.jsonl     everything CREDIT emits (credit.events)

and an append-only ledger at ``$CREDIT_HOME/runs/index.jsonl`` (default
``~/.credit``) points at every run directory, so any client can discover runs.

Every ``*.jsonl`` file has exactly one writer.  Readers ignore a trailing
partial line and never advance a byte offset past one.

This module is shared by the ``credit`` CLI and ``credit.remote``; it must stay
stdlib-only (plus ``yaml`` where noted) so neither pulls in torch.
"""

import hashlib
import json
import math
import os
import re
import secrets
import socket
import tempfile
import time

LEDGER_VERSION = 1
EVENT_VERSION = 1
KINDS = ("preprocess", "train", "rollout", "realtime")
SUBMIT_EVENTS = "submit.jsonl"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_ts(t: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision and a ``Z`` suffix."""
    t = time.time() if t is None else t
    ms = int(round((t - math.floor(t)) * 1000))
    secs = math.floor(t)
    if ms == 1000:
        secs, ms = secs + 1, 0
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs)) + f".{ms:03d}Z"


def parse_ts(ts: str) -> float | None:
    """Parse a ``now_ts``-style timestamp into epoch seconds (``None`` if unparseable)."""
    import calendar

    if not isinstance(ts, str) or not ts.endswith("Z"):
        return None
    body = ts[:-1]
    frac = 0.0
    if "." in body:
        body, f = body.split(".", 1)
        try:
            frac = float("0." + f)
        except ValueError:
            return None
    try:
        return calendar.timegm(time.strptime(body, "%Y-%m-%dT%H:%M:%S")) + frac
    except ValueError:
        return None


def expand(path: str) -> str:
    """Expand ``$VARS`` and ``~`` and make *path* absolute."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))


def credit_home() -> str:
    """Return the ledger root: ``$CREDIT_HOME`` or ``~/.credit``."""
    return expand(os.environ.get("CREDIT_HOME") or "~/.credit")


def ledger_path() -> str:
    return os.path.join(credit_home(), "runs", "index.jsonl")


def new_run_id(kind: str, t: float | None = None) -> str:
    """Return ``<kind>-<YYYYmmddTHHMMSS>-<4 hex>`` (UTC), e.g. ``train-20261030T153207-a1b2``."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {', '.join(KINDS)}")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(time.time() if t is None else t))
    return f"{kind}-{stamp}-{secrets.token_hex(2)}"


def run_dir_for(save_loc: str, run_id: str) -> str:
    return os.path.join(expand(save_loc), ".credit", "runs", run_id)


def job_key(job_id: str) -> str:
    """Return the job key for a scheduler job id.

    ``5875448.desched1`` -> ``5875448``; SLURM ids are already numeric; local
    jobs are recorded as ``local-<pid>`` and kept as is.
    """
    job_id = str(job_id).strip()
    if job_id.startswith("local-"):
        return job_id
    return job_id.split(".", 1)[0]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonfinite(value: float):
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def jsonable(value):
    """Return *value* ready for strict JSON.

    Non-finite floats become the strings ``"NaN"`` / ``"Infinity"`` /
    ``"-Infinity"``, numpy and torch scalars become Python numbers, tuples become
    lists, and anything else unknown becomes its ``str``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else _nonfinite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy / torch scalar
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def encode_line(obj: dict) -> bytes:
    """Return *obj* as one compact JSON line (``\\n``-terminated, UTF-8)."""
    return (json.dumps(jsonable(obj), separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def append_line(path: str, obj: dict, lock: bool = False) -> None:
    """Append one JSON line to *path* with a single ``write()`` on an ``O_APPEND`` descriptor.

    With *lock*, the write happens under ``fcntl.lockf`` on ``<path>.lock`` (used
    for the ledger, the one file several processes may append to).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = encode_line(obj)
    lock_fd = None
    if lock:
        import fcntl

        lock_fd = os.open(path + ".lock", os.O_WRONLY | os.O_CREAT, 0o644)
        fcntl.lockf(lock_fd, fcntl.LOCK_EX)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)  # closing releases the lock


def read_lines(path: str, offset: int = 0, max_bytes: int | None = None) -> tuple[list, int, bool]:
    """Read the complete JSON lines of *path* after byte *offset*.

    Returns ``(lines, new_offset, reset)`` where each line is ``(off, nbytes, obj)``
    and ``off`` is the byte offset at which that line starts.  A trailing
    partial line (no ``\\n`` yet) is left for the next call.  Complete lines
    that do not parse are skipped (but consumed) so one bad line cannot wedge a
    reader.  A missing file reads as empty.  A file shorter than *offset* is
    read from 0 with ``reset=True``.  With *max_bytes*, reading stops after the
    line that crosses that many bytes, and ``new_offset`` is just past it.
    """
    offset = int(offset or 0)
    reset = False
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size < offset:
                offset, reset = 0, True
            f.seek(offset)
            data = f.read(size - offset)
    except FileNotFoundError:
        return [], offset, False
    out = []
    pos = 0
    while True:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        raw = data[pos : nl + 1]
        line_off = offset + pos
        pos = nl + 1
        if raw.strip():
            try:
                obj = json.loads(raw)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                out.append((line_off, len(raw), obj))
        if max_bytes is not None and pos >= max_bytes:
            break
    return out, offset + pos, reset


def atomic_write(path: str, data, mode: int | None = None) -> None:
    """Write *data* (str or bytes) to *path* via a temp file in the same directory and ``os.replace``.

    The write goes to the realpath, so a symlink stays a symlink.  An existing
    file keeps its mode and group; a new one gets *mode* (default ``0o644``).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    path = os.path.realpath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    try:
        st = os.stat(path)
    except FileNotFoundError:
        st = None
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if st is not None:
            os.chmod(tmp, st.st_mode & 0o7777)
            try:
                os.chown(tmp, -1, st.st_gid)
            except OSError:
                pass
        else:
            os.chmod(tmp, 0o644 if mode is None else mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def append_ledger(entry: dict) -> None:
    append_line(ledger_path(), {"v": LEDGER_VERSION, **entry}, lock=True)


def read_ledger(offset: int = 0) -> tuple[list, int, bool]:
    """Return ``(ledger_lines, new_byte_offset, reset)`` for the lines after *offset*.

    ``reset`` is true when *offset* was past the end of the file (the ledger was
    replaced) and the lines were read from the start.
    """
    lines, new_offset, reset = read_lines(ledger_path(), offset)
    return [obj for _, _, obj in lines], new_offset, reset


def ledger_index() -> dict:
    """Return ``{run_id: ledger_entry}`` for every run not removed from view."""
    index = {}
    for line in read_ledger(0)[0]:
        run_id = line.get("run_id")
        if not run_id:
            continue
        if line.get("removed"):
            index.pop(run_id, None)
        elif "run_dir" in line:
            index[run_id] = line
    return index


def find_run(run_id: str) -> dict | None:
    return ledger_index().get(run_id)


# ---------------------------------------------------------------------------
# run.json and events
# ---------------------------------------------------------------------------


def read_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)


def write_run(run_dir: str, record: dict) -> None:
    atomic_write(os.path.join(run_dir, "run.json"), json.dumps(jsonable(record), indent=2) + "\n")


def make_event(type_: str, data: dict | None, writer: str, job: str | None = None, rank: int | None = None) -> dict:
    src = {"writer": writer, "host": socket.gethostname()}
    if job is not None:
        src["job"] = job
    if rank is not None:
        src["rank"] = rank
    return {"v": EVENT_VERSION, "ts": now_ts(), "type": type_, "src": src, "data": data or {}}


def append_event(run_dir: str, filename: str, event: dict) -> None:
    append_line(os.path.join(run_dir, "events", filename), event)


def _ledger_entry(record: dict, run_dir: str) -> dict:
    entry = {k: record.get(k) for k in ("run_id", "kind", "name", "config", "save_loc", "created", "created_by")}
    entry["run_dir"] = run_dir
    if record.get("parent_run_id"):
        entry["parent_run_id"] = record["parent_run_id"]
    return entry


def create_run(
    run_id: str,
    kind: str,
    config: str,
    save_loc: str,
    target: dict,
    created_by: str = "cli",
    name: str | None = None,
    epochs_total: int | None = None,
    epochs_per_job: int | None = None,
    run_dir: str | None = None,
    parent_run_id: str | None = None,
    config_bytes: bytes | None = None,
    exist_ok: bool = False,
) -> dict:
    """Create the run directory, snapshot the config, write ``run.json``, and append to the ledger.

    The run directory is created with ``exist_ok=False``: a second submission of
    the same ``run_id`` raises :class:`FileExistsError` and changes nothing.
    With *exist_ok* (recovering an interrupted submit), an existing ``run.json``
    is returned as is, after making sure the ledger has the run, and a run
    directory without one is completed.
    *config_bytes*, when given, is the exact config content to snapshot (the
    bytes that were hashed), otherwise *config* is copied.  Returns the new
    ``run.json`` record (with no jobs yet; add them with :func:`add_job`).
    """
    config = expand(config)
    save_loc = expand(save_loc)
    run_dir = run_dir or run_dir_for(save_loc, run_id)
    if exist_ok and os.path.exists(os.path.join(run_dir, "run.json")):
        record = read_run(run_dir)
        merged = jobs_of(run_dir, record)
        if merged != record.get("jobs"):  # run.json lagged behind submit.jsonl
            record["jobs"] = merged
            write_run(run_dir, record)
        if not any(line.get("run_id") == run_id for line in read_ledger(0)[0]):
            append_ledger(_ledger_entry(record, run_dir))
        return record
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    if exist_ok:
        os.makedirs(run_dir, exist_ok=True)
    else:
        os.mkdir(run_dir)
    os.makedirs(os.path.join(run_dir, "events"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "scripts"), exist_ok=True)
    if config_bytes is None:
        with open(config, "rb") as f:
            config_bytes = f.read()
    atomic_write(os.path.join(run_dir, "config.yml"), config_bytes)
    record = {
        "v": 1,
        "run_id": run_id,
        "name": name or os.path.basename(save_loc.rstrip(os.sep)) or run_id,
        "kind": kind,
        "created": now_ts(),
        "created_by": created_by,
        "config": config,
        "config_snapshot": "config.yml",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "save_loc": save_loc,
        "target": target,
        "epochs_total": epochs_total,
        "epochs_per_job": epochs_per_job,
        "jobs": [],
        "cancelled": None,
    }
    if parent_run_id:
        record["parent_run_id"] = parent_run_id
    write_run(run_dir, record)
    append_ledger(_ledger_entry(record, run_dir))
    append_event(run_dir, SUBMIT_EVENTS, make_event("run.created", {"kind": kind, "created_by": created_by}, "submit"))
    return record


def write_script(run_dir: str, index: int, text: str) -> str:
    """Write ``scripts/job-<index>.sh`` and return its path relative to *run_dir*."""
    rel = os.path.join("scripts", f"job-{index}.sh")
    atomic_write(os.path.join(run_dir, rel), text, mode=0o755)
    return rel


def jobs_of(run_dir: str, record: dict) -> list:
    """Return the run's jobs: ``record["jobs"]`` merged with ``events/submit.jsonl``.

    ``submit.jsonl`` is authoritative (CONTRACTS A1.2): a job whose
    ``job.submitted`` was written but whose ``run.json`` update was lost (a crash
    in between) still appears, so ``cancel``, scheduler queries and ``credit
    status`` reach it.
    """
    jobs = {j.get("index"): dict(j) for j in record.get("jobs") or []}
    lines, _, _ = read_lines(os.path.join(run_dir, "events", SUBMIT_EVENTS))
    for _, _, ev in lines:
        if ev.get("type") != "job.submitted":
            continue
        d = ev.get("data") or {}
        index, job_id = d.get("index"), d.get("job_id")
        if index is None or not job_id:
            continue
        job = jobs.setdefault(index, {"index": index, "config": None, "submitted": ev.get("ts")})
        if job.get("job_id") != job_id:
            job.update(job_id=job_id, job_key=d.get("job_key") or job_key(job_id), depends_on=d.get("depends_on"))
            script = os.path.join("scripts", f"job-{index}.sh")
            job.setdefault("script", script if os.path.exists(os.path.join(run_dir, script)) else None)
    return [jobs[k] for k in sorted(jobs, key=lambda i: (i is None, i or 0))]


def add_job(run_dir: str, record: dict, job: dict) -> dict:
    """Log ``job.submitted``, then append *job* to ``record['jobs']`` and rewrite ``run.json``.

    *job* needs ``index`` and ``job_id``; ``job_key`` and ``submitted`` are filled in.
    """
    job = dict(job)
    job.setdefault("job_key", job_key(job["job_id"]))
    job.setdefault("submitted", now_ts())
    job.setdefault("depends_on", None)
    # submit.jsonl first: it is authoritative whenever run.json lags behind it.
    data = {k: job.get(k) for k in ("index", "job_id", "job_key", "depends_on")}
    append_event(run_dir, SUBMIT_EVENTS, make_event("job.submitted", data, "submit"))
    record["jobs"].append(job)
    write_run(run_dir, record)
    return job


# ---------------------------------------------------------------------------
# Run state (CONTRACTS A1.8), shared by ``credit status`` and the MCP core
# ---------------------------------------------------------------------------


_TERMINAL_MARKERS = ("job.exited", "job.signaled")


def _event_file(event: dict) -> str | None:
    """Return the events file an event came from (``file``, or inferred from its ``src``)."""
    if event.get("file"):
        return event["file"]
    src = event.get("src") or {}
    if src.get("writer") == "submit":
        return SUBMIT_EVENTS
    if src.get("writer") in ("script", "proc") and src.get("job"):
        return f"{src['job']}.{src['writer']}.jsonl"
    return None


def derive_state(
    kind: str,
    run: dict,
    events: list,
    scheduler: dict,
    csv_last_epoch: int | None = None,
    has_child: bool = False,
) -> str:
    """Return ``cancelled | running | queued | completed | stopped | failed | unknown`` for a run.

    The rules of CONTRACTS A1.8 as refined by A4.1 and A5.2-A5.3, shared with
    the MCP core through ``tests/data/state_cases.json``.

    Args:
        kind: The run kind.
        run: ``run.json`` (``jobs`` with ``job_key``; ``epochs_total``; ``cancelled``).
        events: The run's events; each needs ``type``, ``data`` and ``file``
            (or a ``src`` it can be inferred from).
        scheduler: ``JobState`` per job, keyed by job key (or job id).
        csv_last_epoch: The last ``epoch`` in ``training_log.csv``, or None.
        has_child: True when another ledger run resumes this one.
    """
    jobs = run.get("jobs") or []
    by_file = {}
    for ev in events:
        by_file.setdefault(_event_file(ev), []).append(ev)

    def of(key, suffix, type_):
        return [e for e in by_file.get(f"{key}.{suffix}.jsonl", []) if e.get("type") == type_]

    def job_state(job) -> dict:
        key = job.get("job_key")
        st = dict(scheduler.get(key) or scheduler.get(job.get("job_id")) or {})
        # A terminal marker beats a stale scheduler answer.
        if any(of(key, "script", t) for t in _TERMINAL_MARKERS) and st.get("state") in ("running", "queued", "held"):
            st["state"] = "finished"
        return st

    states = [job_state(j) for j in jobs]
    cancels = [e for e in by_file.get(SUBMIT_EVENTS, []) if e.get("type") == "run.cancelled"]
    scopes = [(e.get("data") or {}).get("scope") for e in cancels]
    if run.get("cancelled"):
        scopes.append((run["cancelled"] or {}).get("scope"))
    if any(scope != "pending" for scope in scopes):
        return "cancelled"
    if any(st.get("state") == "running" for st in states):
        return "running"
    if any(st.get("state") in ("queued", "held") for st in states):
        return "queued"
    if scopes:  # pending-only cancel, and nothing is running or queued any more
        return "cancelled"

    started = [
        (j, st) for j, st in zip(jobs, states) if of(j.get("job_key"), "script", "job.started") or st.get("exec_host")
    ]
    if not started:
        return "unknown"
    job, st = max(started, key=lambda js: js[0].get("index") or 0)
    key = job.get("job_key")
    exited = of(key, "script", "job.exited")
    code = (exited[-1].get("data") or {}).get("code") if exited else st.get("exit_status")
    signaled = bool(of(key, "script", "job.signaled"))
    stops = of(key, "proc", "train.stopped")  # A5.2: only this job's own events
    reason = (stops[-1].get("data") or {}).get("reason") if stops else None
    errors = of(key, "proc", "error")

    if code == 0 and (kind != "train" or reason in ("completed", "early_stop")):
        return "completed"
    if signaled or reason == "epochs_per_job":
        return "stopped"
    if (code is not None and code != 0) or errors:
        return "failed"
    if code == 0 and kind == "train" and reason is None:
        # CREDIT without the event sink (A4.1); never for a run that was resumed (A5.3).
        if has_child:
            return "stopped"
        epochs_total = run.get("epochs_total")
        if csv_last_epoch is None or not epochs_total:
            return "unknown"
        return "completed" if csv_last_epoch + 1 >= int(epochs_total) else "stopped"
    return "unknown"


# ---------------------------------------------------------------------------
# Config files from requests
# ---------------------------------------------------------------------------


class ConfigFileError(ValueError):
    """A config file is not a readable YAML mapping; the message never quotes the file."""


def load_yaml_mapping(path: str) -> tuple:
    """Return ``(bytes, mapping)`` for a ``.yml``/``.yaml`` file that parses to a mapping.

    Parse errors report only the problem and its line/column, never a snippet
    of the file, so a request naming the wrong file cannot leak its contents.
    """
    import yaml

    if not str(path).lower().endswith((".yml", ".yaml")):
        raise ConfigFileError(f"{path}: a config must be a .yml or .yaml file")
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise ConfigFileError(f"{path}: cannot read ({exc.strerror or type(exc).__name__})") from None
    try:
        conf = yaml.safe_load(data)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigFileError(f"{path}: invalid YAML: {exc.problem or 'parse error'}{where}") from None
    except yaml.YAMLError:
        raise ConfigFileError(f"{path}: invalid YAML") from None
    if not isinstance(conf, dict):
        raise ConfigFileError(f"{path}: the config is not a YAML mapping")
    return data, conf


# ---------------------------------------------------------------------------
# Path scope: the security boundary for credit.remote read/ls/logs/write
# ---------------------------------------------------------------------------


class PathScopeError(PermissionError):
    """A path resolves outside every allowed root, or into a place that is never served."""


def _real(path: str) -> str:
    return os.path.realpath(expand(path))


def _under(path: str, root: str) -> bool:
    return root == os.sep or path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def static_roots() -> list:
    """Return the realpaths of the static roots.

    ``$HOME``; ``$SCRATCH``/``$WORK``/``$PSCRATCH`` when set; the NCAR GLADE work
    and scratch dirs when they exist; and each ``$CREDIT_REMOTE_ALLOW`` entry
    (colon-separated).  A ``remote.yaml`` policy with ``roots`` replaces all of
    these (see :mod:`credit.remote._guard`); the environment can't widen it.
    """
    from credit.remote import _guard

    policy_roots = _guard.policy()["roots"]
    if policy_roots is not None:
        return list(dict.fromkeys(policy_roots))
    roots = []
    for var in ("HOME", "SCRATCH", "WORK", "PSCRATCH"):
        if os.environ.get(var):
            roots.append(os.environ[var])
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if user:
        roots += [d for d in (f"/glade/work/{user}", f"/glade/derecho/scratch/{user}") if os.path.isdir(d)]
    roots += [r.strip() for r in (os.environ.get("CREDIT_REMOTE_ALLOW") or "").split(":") if r.strip()]
    out = []
    for r in roots:
        rp = _real(r)
        if rp not in out:
            out.append(rp)
    return out


def ledger_roots(index: dict | None = None, static: list | None = None) -> list:
    """Return each ledger run's ``run_dir``, ``save_loc`` and config dir that lies inside a static root."""
    static = static_roots() if static is None else static
    out = []
    for entry in (ledger_index() if index is None else index).values():
        candidates = [entry.get("run_dir"), entry.get("save_loc")]
        if entry.get("config"):
            candidates.append(os.path.dirname(entry["config"]))
        for c in candidates:
            if not c:
                continue
            rp = _real(c)
            if rp not in out and any(_under(rp, s) for s in static):
                out.append(rp)
    return out


def allowed_roots(index: dict | None = None) -> list:
    """Return every root ``credit.remote`` may serve: the static roots plus the in-scope ledger roots."""
    static = static_roots()
    return static + [r for r in ledger_roots(index, static) if r not in static]


def _hidden_ok(real: str, root: str, index: dict) -> bool:
    """True when *real* has no dot-component below *root*, except under ``<save_loc>/.credit/runs/``."""

    def no_dots(path, base):
        rel = os.path.relpath(path, base) if path != base else ""
        return not any(part.startswith(".") for part in rel.split(os.sep) if part and part != os.curdir)

    if no_dots(real, root):
        return True
    for entry in index.values():
        save_loc = entry.get("save_loc")
        if not save_loc:
            continue
        s = _real(save_loc)
        if _under(real, os.path.join(s, ".credit", "runs")) and _under(s, root) and no_dots(s, root):
            return True
    return False


def resolve_in_scope(path: str, allow_hidden: bool = False) -> str:
    """Return the realpath of *path* if it may be served, else raise :class:`PathScopeError`.

    The check is on the fully resolved path, so neither ``..`` nor a symlink can
    escape a root.  Unless *allow_hidden*, a path with a component starting
    with ``.`` (below its root) is refused, except inside a ledger run's
    ``<save_loc>/.credit/runs/``.
    """
    if not path or not isinstance(path, str):
        raise PathScopeError("path is required")
    real = _real(path)
    index = ledger_index()
    for root in allowed_roots(index):
        if _under(real, root):
            if allow_hidden or _hidden_ok(real, root, index):
                return real
            raise PathScopeError(f"{path} is inside a hidden directory")
    raise PathScopeError(f"{path} is outside the allowed roots")


def check_writable(real: str) -> None:
    """Refuse writes that ``credit.remote write`` must never make (on top of :func:`resolve_in_scope`).

    Only ``.yml``/``.yaml`` files, never under ``$CREDIT_HOME`` or any run
    directory, never in a dot-directory.
    """
    if not real.lower().endswith((".yml", ".yaml")):
        raise PathScopeError("only .yml / .yaml files can be written")
    if _under(real, _real(credit_home())):
        raise PathScopeError("files under $CREDIT_HOME cannot be written")
    for entry in ledger_index().values():
        if entry.get("run_dir") and _under(real, _real(entry["run_dir"])):
            raise PathScopeError("files inside a run directory cannot be written")
    if any(part.startswith(".") for part in os.path.dirname(real).split(os.sep) if part):
        raise PathScopeError("files in a hidden directory cannot be written")


# ---------------------------------------------------------------------------
# Scheduler commands (qsub / sbatch / local launch)
# ---------------------------------------------------------------------------


class SchedulerError(RuntimeError):
    """A scheduler command failed; the message holds its stderr."""


class SchedulerTimeout(SchedulerError):
    """A scheduler command did not answer in time; it may or may not have acted."""


def scheduler_bin(name: str, scheduler: str = "pbs") -> str:
    """Return the command to run for scheduler tool *name* (``qsub``, ``qstat``, ``sbatch``, ...).

    ``$CREDIT_PBS_BIN`` / ``$CREDIT_SLURM_BIN`` name the directory holding the
    tools.  PBS otherwise prefers ``/opt/pbs/bin`` when it exists: Casper's
    ``qstat`` on ``PATH`` is a caching wrapper that can lag by minutes.
    Anything else falls back to a bare name looked up on ``PATH``.
    """
    env = "CREDIT_SLURM_BIN" if scheduler == "slurm" else "CREDIT_PBS_BIN"
    d = os.environ.get(env)
    if not d and scheduler == "pbs" and os.path.isdir("/opt/pbs/bin"):
        d = "/opt/pbs/bin"
    return os.path.join(expand(d), name) if d else name


SUBMIT_TIMEOUT_S = 120
_PBS_JOB_ID = re.compile(r"^\d+(\[\d*\])?\.\S+$")
_SLURM_JOB_ID = re.compile(r"^\d+$")


def parse_job_id(scheduler: str, stdout: str) -> str:
    """Return the job id from ``qsub`` / ``sbatch`` output: the last line that looks like one.

    PBS ids look like ``5875448.desched1`` (or ``123[].server`` for arrays);
    SLURM ids are plain numbers (``sbatch --parsable`` may append ``;cluster``,
    and plain ``sbatch`` prints ``Submitted batch job N``).
    """
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if scheduler == "slurm":
            head = line.split(";", 1)[0]
            if _SLURM_JOB_ID.match(head):
                return head
            m = re.match(r"^Submitted batch job (\d+)", line)
            if m:
                return m.group(1)
        elif _PBS_JOB_ID.match(line):
            return line
    raise SchedulerError(f"no job id in the {'sbatch' if scheduler == 'slurm' else 'qsub'} output: {stdout!r}"[:300])


def submit_job(
    scheduler: str,
    script_path: str,
    depend_on: str | None = None,
    log_dir: str | None = None,
    job_name: str | None = None,
    env: dict | None = None,
) -> str:
    """Submit *script_path* and return the scheduler job id; raises :class:`SchedulerError`.

    A chain dependency goes on the command line (``qsub -W depend=afterok:<id>``,
    ``sbatch --dependency=afterok:<id>``), never into the script, and so does
    *job_name* (``-N`` / ``--job-name``).  The ``local`` scheduler starts the
    script detached in a new session (``setsid``) with its output in
    ``<log_dir>/local-<pid>.log`` and *env* added to its environment, and
    returns ``local-<pid>``.  ``qsub``/``sbatch`` time out after 120 s.
    """
    import subprocess

    if scheduler == "local":
        os.makedirs(log_dir, exist_ok=True)
        # bash's $$ is the pid Popen reports, and survives the final exec, so the
        # log name and the recorded job id agree.
        launcher = 'exec >>"$1/local-$$.log" 2>&1 </dev/null; exec /bin/bash "$2"'
        proc = subprocess.Popen(
            ["/bin/bash", "-c", launcher, "credit-local", log_dir, script_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, **(env or {})},
        )
        return f"local-{proc.pid}"

    if scheduler == "slurm":
        cmd = [scheduler_bin("sbatch", "slurm"), "--parsable"]
        if job_name:
            cmd.append(f"--job-name={job_name}")
        if depend_on:
            cmd.append(f"--dependency=afterok:{job_key(depend_on)}")
    else:
        cmd = [scheduler_bin("qsub", "pbs")]
        if job_name:
            cmd += ["-N", job_name]
        if depend_on:
            cmd += ["-W", f"depend=afterok:{depend_on}"]
    cmd.append(script_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SchedulerTimeout(f"{cmd[0]} did not answer within {SUBMIT_TIMEOUT_S} s") from exc
    except OSError as exc:
        raise SchedulerError(f"{cmd[0]}: {exc}") from exc
    if result.returncode != 0:
        raise SchedulerError((result.stderr or result.stdout or f"{cmd[0]} exited {result.returncode}").strip())
    return parse_job_id(scheduler, result.stdout)
