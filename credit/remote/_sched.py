"""Batched scheduler queries and cancellation for ``credit.remote``.

One ``qstat -x -f -F json <ids...>`` per PBS server, one ``sacct --json`` (plus
``squeue --json`` for pending jobs' estimated start) for SLURM, and a pid check
for local jobs.  Every job comes back as a ``JobState``::

    {"state": "queued|held|running|finished|gone|unknown", "raw_state", "exit_status",
     "exec_host", "estimated_start", "walltime_used", "walltime_limit", "comment", "hold_types"}
"""

import json
import os
import signal
import subprocess
import time

from credit import runs

# PBS job_state letters (docs/source/remote-protocol.md, "Scheduler queries").
_PBS_STATES = {
    "Q": "queued",
    "W": "queued",
    "T": "queued",
    "M": "queued",
    "H": "held",
    "S": "held",
    "U": "held",
    "R": "running",
    "E": "running",
    "B": "running",
    "F": "finished",
    "X": "finished",
}

_SLURM_STATES = {
    "PENDING": "queued",
    "REQUEUED": "queued",
    "CONFIGURING": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "SUSPENDED": "held",
    "STOPPED": "held",
}


def _state(state="unknown", raw=None, **kw) -> dict:
    out = {
        "state": state,
        "raw_state": raw,
        "exit_status": None,
        "exec_host": None,
        "estimated_start": None,
        "walltime_used": None,
        "walltime_limit": None,
        "comment": None,
        "hold_types": None,
    }
    out.update(kw)
    return out


def _run(cmd: list, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise runs.SchedulerError(f"{cmd[0]}: {exc}") from exc


def _ctime_to_ts(value) -> str | None:
    """Convert PBS's ``estimated.start_time`` (ctime in the server's local time, or epoch secs) to UTC."""
    if value in (None, ""):
        return None
    try:
        if str(value).isdigit():
            return runs.now_ts(float(value))
        return runs.now_ts(time.mktime(time.strptime(str(value).strip(), "%a %b %d %H:%M:%S %Y")))
    except (ValueError, OverflowError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PBS
# ---------------------------------------------------------------------------


def _pbs_job_state(attrs: dict) -> dict:
    raw = attrs.get("job_state")
    exec_host = attrs.get("exec_host")
    estimated = attrs.get("estimated") or {}
    return _state(
        _PBS_STATES.get(raw, "unknown"),
        raw,
        exit_status=_int(attrs.get("Exit_status")),
        exec_host=exec_host.split("/", 1)[0] if exec_host else None,
        estimated_start=_ctime_to_ts(estimated.get("start_time") if isinstance(estimated, dict) else None),
        walltime_used=(attrs.get("resources_used") or {}).get("walltime"),
        walltime_limit=(attrs.get("Resource_List") or {}).get("walltime"),
        comment=attrs.get("comment"),
        hold_types=attrs.get("Hold_Types"),
    )


def parse_qstat_text(text: str) -> dict:
    """Parse ``qstat -x -f`` text output into ``{job_id: attrs}`` with dotted keys nested."""
    jobs, attrs, key = {}, None, None
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            attrs = jobs.setdefault(line.split(":", 1)[1].strip(), {})
            key = None
        elif attrs is None or not line.strip():
            continue
        elif line.startswith("\t") and key:  # continuation of a wrapped value
            _set(attrs, key, _get(attrs, key) + line.strip())
        elif " = " in line:
            key, value = line.strip().split(" = ", 1)
            _set(attrs, key, value)
    return jobs


def _set(attrs: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        attrs = attrs.setdefault(p, {})
    attrs[leaf] = value


def _get(attrs: dict, dotted: str):
    for p in dotted.split("."):
        attrs = attrs.get(p, {})
    return attrs if isinstance(attrs, str) else ""


def query_pbs(job_ids: list) -> tuple:
    """Return ``({job_id: JobState}, {server: error})`` with one ``qstat`` call per server."""
    by_server = {}
    for jid in job_ids:
        by_server.setdefault(jid.split(".", 1)[1] if "." in jid else "", []).append(jid)
    states, errors = {}, {}
    qstat = runs.scheduler_bin("qstat", "pbs")
    for server, ids in by_server.items():
        try:
            res = _run([qstat, "-x", "-f", "-F", "json", *ids])
            try:
                found = json.loads(res.stdout).get("Jobs") or {}
            except ValueError:
                found = None
            if found is None:
                text = _run([qstat, "-x", "-f", *ids])
                found = parse_qstat_text(text.stdout)
                if not found and text.returncode != 0 and not _all_unknown(text.stderr):
                    raise runs.SchedulerError((text.stderr or res.stderr or "qstat failed").strip())
        except runs.SchedulerError as exc:
            errors[server or "pbs"] = str(exc)
            continue
        by_key = {runs.job_key(k): v for k, v in found.items()}
        for jid in ids:
            attrs = found.get(jid) or by_key.get(runs.job_key(jid))
            states[jid] = _pbs_job_state(attrs) if attrs else _state("gone")
    return states, errors


def _all_unknown(stderr: str) -> bool:
    lines = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    return bool(lines) and all("Unknown Job Id" in ln for ln in lines)


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


def _number(value):
    """SLURM's JSON wraps many numbers as ``{"set": true, "number": n}``."""
    if isinstance(value, dict):
        return value.get("number") if value.get("set", True) else None
    return value


def _hms(seconds) -> str | None:
    seconds = _int(seconds)
    if seconds is None:
        return None
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _slurm_job_state(job: dict) -> dict:
    st = job.get("state") or {}
    raw = st.get("current") if isinstance(st, dict) else st
    raw = (raw[0] if raw else None) if isinstance(raw, list) else raw
    reason = st.get("reason") if isinstance(st, dict) else None
    state = _SLURM_STATES.get(raw, "finished" if raw else "unknown")
    if state == "queued" and reason and "Held" in str(reason):
        state = "held"
    exit_code = job.get("exit_code") or {}
    rc = _number(exit_code.get("return_code")) if isinstance(exit_code, dict) else exit_code
    t = job.get("time") or {}
    limit = _number(t.get("limit"))
    nodes = job.get("nodes") or None
    return _state(
        state,
        raw,
        exit_status=_int(rc) if state == "finished" else None,
        exec_host=nodes.split(",", 1)[0] if isinstance(nodes, str) and nodes not in ("None assigned", "") else None,
        walltime_used=_hms(t.get("elapsed")) if state in ("running", "finished") else None,
        walltime_limit=_hms(limit * 60) if _int(limit) is not None else None,
        comment=reason if reason not in (None, "None") else None,
    )


def query_slurm(job_ids: list) -> tuple:
    """Return ``({job_id: JobState}, {"slurm": error})`` from ``sacct --json`` (+ ``squeue --json``)."""
    if not job_ids:
        return {}, {}
    try:
        res = _run([runs.scheduler_bin("sacct", "slurm"), "--json", "-j", ",".join(job_ids)])
        found = {}
        for job in json.loads(res.stdout or "{}").get("jobs") or []:
            found.setdefault(str(job.get("job_id")), job)
    except (runs.SchedulerError, ValueError) as exc:
        return {}, {"slurm": str(exc)}
    states = {jid: _slurm_job_state(found[jid]) if jid in found else _state("gone") for jid in job_ids}
    pending = [jid for jid, s in states.items() if s["state"] in ("queued", "held")]
    if pending:
        try:
            res = _run([runs.scheduler_bin("squeue", "slurm"), "--json", "--jobs", ",".join(pending)])
            for job in json.loads(res.stdout or "{}").get("jobs") or []:
                start = _int(_number(job.get("start_time")))
                jid = str(job.get("job_id"))
                if jid in states and start:
                    states[jid]["estimated_start"] = runs.now_ts(start)
        except (runs.SchedulerError, ValueError):
            pass  # the estimated start is a nicety
    return states, {}


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _process_start(pid: int) -> float | None:
    """Return a process's start time (epoch seconds) from ``ps -o lstart=``, or None."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        return time.mktime(time.strptime(" ".join(out.stdout.split()), "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def local_job_is_ours(pid: int, run_id: str | None, submitted: str | None) -> bool:
    """True if *pid* is still the job ``credit.remote submit`` started for *run_id*, not a reused pid.

    On Linux the process environment must hold ``CREDIT_RUN_ID=<run_id>`` (set
    at launch); elsewhere its start time must be within 2 minutes of the job's
    ``submitted`` time.
    """
    if not _pid_alive(pid):
        return False
    if os.path.isdir("/proc/self"):
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                return f"CREDIT_RUN_ID={run_id}".encode() in f.read().split(b"\0")
        except OSError:
            return False
    started, submitted_t = _process_start(pid), runs.parse_ts(submitted) if submitted else None
    return started is not None and submitted_t is not None and abs(started - submitted_t) < 120


def _local_job(job_id: str, run_dir: str) -> tuple:
    """Return ``(pid, run_id, submitted)`` for a local job from its run.json."""
    pid = _int(job_id.split("-", 1)[1]) if job_id.startswith("local-") else None
    try:
        record = runs.load_run(run_dir)
    except (OSError, ValueError):
        return pid, None, None
    job = next((j for j in record["jobs"] if j.get("job_id") == job_id), {})
    return pid, record.get("run_id"), job.get("submitted")


def unfinished_jobs(run_dir: str, record: dict | None) -> list:
    """The run's jobs with a job id and no exit marker yet: the ones to ask the scheduler about."""
    from credit.remote._delta import job_finished

    return [
        j for j in (record or {}).get("jobs") or [] if j.get("job_id") and not job_finished(run_dir, j.get("job_key"))
    ]


def query_local(jobs: list) -> dict:
    """Return ``{job_id: JobState}`` for local jobs ``[(job_id, run_dir)]`` from a pid check and the markers."""
    from credit.remote._delta import job_finished

    out = {}
    for jid, run_dir in jobs:
        pid, run_id, submitted = _local_job(jid, run_dir)
        done = job_finished(run_dir, jid)
        if pid and local_job_is_ours(pid, run_id, submitted):
            out[jid] = _state("running", "running", exec_host=os.uname().nodename)
        elif done:
            out[jid] = _state("finished", "exited", exit_status=done.get("code"))
        else:
            out[jid] = _state("gone")
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def query(jobs: list) -> tuple:
    """Return ``({job_id: JobState}, {server_or_scheduler: error})`` for ``[(scheduler, job_id, run_dir)]``."""
    states, errors = {}, {}
    pbs = [jid for sched, jid, _ in jobs if sched == "pbs"]
    slurm = [jid for sched, jid, _ in jobs if sched == "slurm"]
    local = [(jid, rd) for sched, jid, rd in jobs if sched == "local"]
    if pbs:
        s, e = query_pbs(pbs)
        states.update(s)
        errors.update(e)
    if slurm:
        s, e = query_slurm(slurm)
        states.update(s)
        errors.update(e)
    if local:
        states.update(query_local(local))
    return states, errors


def cancel_job(scheduler: str, job_id: str, run_dir: str | None = None) -> None:
    """Cancel one job (``qdel`` / ``scancel`` / TERM to the local process group); raises SchedulerError.

    A local job is signalled only after checking the pid still belongs to the run.
    """
    if scheduler == "local":
        pid, run_id, submitted = _local_job(job_id, run_dir) if run_dir else (None, None, None)
        if not pid:
            raise runs.SchedulerError(f"not a local job id: {job_id}")
        if not local_job_is_ours(pid, run_id, submitted):
            raise runs.SchedulerError(f"process {pid} is not (or no longer) job {job_id}; not signalling it")
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError as exc:
            raise runs.SchedulerError(f"process group {pid} is gone") from exc
        except PermissionError as exc:
            raise runs.SchedulerError(str(exc)) from exc
        return
    cmd = [runs.scheduler_bin("scancel", "slurm")] if scheduler == "slurm" else [runs.scheduler_bin("qdel", "pbs")]
    res = _run(cmd + [job_id])
    if res.returncode != 0:
        raise runs.SchedulerError((res.stderr or res.stdout or f"{cmd[0]} exited {res.returncode}").strip())
