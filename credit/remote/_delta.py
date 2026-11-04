"""Cursor-based deltas of a run directory: events, ``training_log.csv``, and job logs.

Shared by ``sync`` (one shot) and ``follow`` (streaming).  Stdlib only.

A cursor is a public, mergeable JSON object (CONTRACTS A1.3)::

    {"run": {"mtime", "size"},
     "events": {"<file>": byte_offset},
     "csv": {"rows", "size", "mtime", "header_sha", "first_row_sha"},
     "logs": {"<job_key>": {"path", "off", "size"}}}
"""

import csv
import glob
import hashlib
import io
import os
import time

from credit import runs
from credit.remote import _guard

PROGRESS_TYPES = ("train.progress", "train.vars")
FRAME_BYTES = 4096
CSV_SETTLE_S = 2.0


# ---------------------------------------------------------------------------
# Logs: tqdm-aware tails
# ---------------------------------------------------------------------------


def _collapse_lines(data: bytes) -> list:
    """Split *data* into ``\\n``-terminated lines, each collapsed to the text after its last ``\\r``.

    *data* must end at a line boundary; a trailing unterminated segment is dropped.
    """
    data = data.replace(b"\r\n", b"\n")
    out = []
    for raw in data.split(b"\n")[:-1]:
        out.append(raw.rsplit(b"\r", 1)[-1].decode("utf-8", errors="replace"))
    return out


def current_frame(path: str, size: int) -> str | None:
    """Return the live tqdm frame: the text after the last ``\\r`` that follows the last ``\\n``.

    Read from at most the last 4096 bytes.  ``None`` when the unterminated tail
    has no ``\\r`` (or the file ends at a line boundary).
    """
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - FRAME_BYTES))
            tail = f.read(min(size, FRAME_BYTES))
    except OSError:
        return None
    tail = tail.replace(b"\r\n", b"\n")
    seg = tail.rsplit(b"\n", 1)[-1]
    if b"\r" not in seg:
        return None
    parts = seg.split(b"\r")
    frame = parts[-1] or (parts[-2] if len(parts) > 1 else b"")
    return frame.decode("utf-8", errors="replace")


def read_log(path: str, off: int, cap: int) -> dict:
    """Return the new log text of *path* after byte *off*, capped at *cap* bytes.

    ``{"text", "current_frame", "off", "size", "truncated", "reset"}``.  ``\\r\\n``
    is normalized; each ``\\n``-terminated line is collapsed to its last ``\\r``
    segment (tqdm frames).  ``off`` advances past the last ``\\n`` or ``\\r``,
    whichever is later.  When the cap forces a skip, the first partial segment
    is dropped.  A file shorter than *off* restarts from 0 with ``reset``.
    """
    size = os.path.getsize(path)
    reset = size < off
    if reset:
        off = 0
    start, truncated = off, False
    if size - start > cap:
        start, truncated = size - cap, True
    with open(path, "rb") as f:
        f.seek(max(0, start - 1) if truncated else start)
        data = f.read(size - start + (1 if truncated and start > 0 else 0))
    if truncated and start > 0:
        # Drop the first partial segment unless the skip landed on a boundary.
        if data[:1] in (b"\n", b"\r"):
            data = data[1:]
        else:
            cut = [i for i in (data.find(b"\n"), data.find(b"\r")) if i >= 0]
            skip = min(cut) + 1 if cut else len(data)
            data, start = data[skip:], start - 1 + skip
    # A trailing lone \r may be half of a \r\n still being written.
    scan = data[:-1] if data.endswith(b"\r") else data
    last = max(scan.rfind(b"\n"), scan.rfind(b"\r"))
    consumed = data[: last + 1] if last >= 0 else b""
    nl = consumed.replace(b"\r\n", b"\n").rfind(b"\n")
    lines = _collapse_lines(consumed.replace(b"\r\n", b"\n")[: nl + 1]) if nl >= 0 else []
    return {
        "text": _guard.redact("".join(line + "\n" for line in lines))[0],
        "current_frame": _guard.redact(current_frame(path, size))[0],
        "off": start + last + 1 if last >= 0 else start,
        "size": size,
        "truncated": truncated,
        "reset": reset,
    }


def tail_log(path: str, tail_bytes: int) -> dict:
    """Return the last *tail_bytes* of a log as collapsed text (the ``logs`` verb).

    Unlike :func:`read_log`, an unterminated last line without ``\\r`` is
    included, since this is a snapshot rather than a cursor read.
    """
    size = os.path.getsize(path)
    r = read_log(path, max(0, size - tail_bytes), tail_bytes)
    with open(path, "rb") as f:
        f.seek(r["off"])
        rest = f.read().replace(b"\r\n", b"\n")
    if rest and b"\r" not in rest and b"\n" not in rest:
        r["text"] += _guard.redact(rest.decode("utf-8", errors="replace"))[0]
    return r


def find_log(record: dict, job_key: str) -> str | None:
    """Return the scheduler output file of job *job_key* in ``target.log_dir``, or None.

    For PBS: ``<job_key>.<server>.OU`` (what PBS writes with ``-k eod`` into an
    ``-o`` directory, as on Casper) or ``*.o<job_key>``, and rank 0's
    ``pbsdsh_logs/<job_key>/node_0.out`` when the job was launched with pbsdsh.
    ``*-<job_key>.out`` for SLURM, and ``<job_key>.log`` for the local scheduler.
    """
    target = record.get("target") or {}
    log_dir = target.get("log_dir") or os.path.join(record.get("save_loc", ""), "logs")
    scheduler = target.get("scheduler")
    if scheduler == "local":
        path = os.path.join(log_dir, f"{job_key}.log")
        return path if os.path.isfile(path) else None
    if scheduler == "slurm":
        matches = glob.glob(os.path.join(glob.escape(log_dir), f"*-{job_key}.out"))
    else:
        node0 = os.path.join(log_dir, "pbsdsh_logs", job_key, "node_0.out")
        if os.path.isfile(node0):
            return node0
        base = glob.escape(log_dir)
        matches = glob.glob(os.path.join(base, f"{job_key}.*.OU")) + glob.glob(os.path.join(base, f"*.o{job_key}"))
    return max(matches, key=os.path.getmtime) if matches else None


def _log_path(record: dict, job_key: str, known: str | None) -> str | None:
    """Return the job's log, reusing *known* (a stat) instead of globbing the log dir every tick.

    A PBS job switches to pbsdsh's ``node_0.out`` once that appears.
    """
    if not known or not os.path.isfile(known):
        return find_log(record, job_key)
    if (record.get("target") or {}).get("scheduler") in (None, "pbs") and not known.endswith("node_0.out"):
        log_dir = (record.get("target") or {}).get("log_dir") or ""
        node0 = os.path.join(log_dir, "pbsdsh_logs", job_key, "node_0.out")
        if os.path.isfile(node0):
            return node0
    return known


# ---------------------------------------------------------------------------
# training_log.csv
# ---------------------------------------------------------------------------


def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def read_csv_delta(path: str, cur: dict | None, now: float | None = None) -> tuple:
    """Return ``(csv_delta | None, new_csv_cursor, pending)`` for ``training_log.csv``.

    Older CREDIT rewrote the file in place once per epoch, so it is only read
    once it has settled: while its last line is unterminated or its mtime is
    under 2 s old, the delta is ``None``, the cursor is unchanged, and
    *pending* is true so the caller knows to look again.  A changed header or
    first row, or fewer rows than the cursor has seen, returns every row with
    ``reset``.
    """
    cur = cur or {}
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None, cur or None, False
    if cur and cur.get("size") == st.st_size and cur.get("mtime") == st.st_mtime:
        return None, cur, False
    if (time.time() if now is None else now) - st.st_mtime < CSV_SETTLE_S:
        return None, cur or None, True
    with open(path, "rb") as f:
        data = f.read()
    if not data or not data.endswith(b"\n"):
        return None, cur or None, True
    raw_lines = data.split(b"\n")
    header_sha = hashlib.sha256(raw_lines[0]).hexdigest()
    first_row_sha = hashlib.sha256(raw_lines[1]).hexdigest() if len(raw_lines) > 2 else None
    table = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace"))))
    columns, rows = (table[0], table[1:]) if table else ([], [])
    seen = int(cur.get("rows") or 0)
    reset = bool(cur) and (
        cur.get("header_sha") not in (None, header_sha)
        or cur.get("first_row_sha") not in (None, first_row_sha)
        or len(rows) < seen
    )
    first = 0 if reset else seen
    new_cur = {
        "rows": len(rows),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "header_sha": header_sha,
        "first_row_sha": first_row_sha,
    }
    if not reset and len(rows) <= first:
        return None, new_cur, False
    delta = {
        "columns": columns,
        "rows": [[_num(v) for v in row] for row in rows[first:]],
        "first_row": first,
        "reset": reset,
    }
    return delta, new_cur, False


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def read_events(run_dir: str, cur: dict, progress: str = "all", budget: list | None = None) -> tuple:
    """Return ``(events, new_events_cursor, more)`` for every ``events/*.jsonl`` past its offset.

    Each event gets ``file`` and ``off`` (where its line starts) and the list is
    sorted by ``ts``.  *progress* is ``all`` (every line, up to ``budget[0]``
    bytes, which is decremented), ``latest`` (only the last ``train.progress``
    and ``train.vars`` per file) or ``none`` (neither); ``latest`` and ``none``
    still advance the cursor to the end.
    """
    cur = dict(cur or {})
    events, more = [], False
    try:
        names = sorted(n for n in os.listdir(os.path.join(run_dir, "events")) if n.endswith(".jsonl"))
    except FileNotFoundError:
        return [], cur, False
    for name in names:
        path = os.path.join(run_dir, "events", name)
        off = int(cur.get(name) or 0)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size == off:
            continue
        limit = None
        if progress == "all" and budget is not None:
            if budget[0] <= 0:
                more = True
                continue
            limit = budget[0]
        lines, new_off, reset = runs.read_lines(path, off, max_bytes=limit)
        if limit is not None:
            budget[0] -= new_off - (0 if reset else off)
            if new_off < size and _has_complete_line(path, new_off, size):
                more = True
        if progress in ("latest", "none"):
            keep = set()
            if progress == "latest":
                for t in PROGRESS_TYPES:
                    idx = [i for i, (_, _, ev) in enumerate(lines) if ev.get("type") == t]
                    if idx:
                        keep.add(idx[-1])
            lines = [ln for i, ln in enumerate(lines) if ln[2].get("type") not in PROGRESS_TYPES or i in keep]
        for line_off, _, ev in lines:
            ev = _guard.redact_event(ev)
            ev["file"], ev["off"] = name, line_off
            events.append(ev)
        cur[name] = new_off
    events.sort(key=lambda e: str(e.get("ts", "")))
    return events, cur, more


def _has_complete_line(path: str, off: int, size: int) -> bool:
    with open(path, "rb") as f:
        f.seek(off)
        return b"\n" in f.read(min(size - off, 1 << 20))


def job_finished(run_dir: str, job_key: str) -> dict | None:
    """Return ``{"code", "signal"}`` from a job's script markers once it has ended, else None.

    ``job.signaled`` with no ``job.exited`` counts as ended.
    """
    lines, _, _ = runs.read_lines(os.path.join(run_dir, "events", f"{job_key}.script.jsonl"))
    out = None
    for _, _, ev in lines:
        if ev.get("type") == "job.exited":
            out = {"code": (ev.get("data") or {}).get("code"), "signal": (out or {}).get("signal")}
        elif ev.get("type") == "job.signaled":
            out = {"code": (out or {}).get("code"), "signal": (ev.get("data") or {}).get("signal")}
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

# run.json (jobs merged with submit.jsonl) by run dir, keyed on both files' stats, so a
# follow tick only stats them.
_RECORDS = {}


def run_delta(
    entry: dict, cursor: dict | None, progress: str = "all", log_tail_bytes: int = 65536, budget: list | None = None
) -> tuple:
    """Return ``(RunDelta, record)`` for a ledger *entry* since *cursor* (``None`` = from the start).

    *record* is the current ``run.json`` (or None), for callers that need the
    jobs, whether or not it is included in the delta.
    """
    cursor = dict(cursor or {})
    run_dir = entry["run_dir"]
    delta = {
        "run": None,
        "events": [],
        "csv": None,
        "csv_pending": False,
        "logs": {},
        "cursor": cursor,
        "missing": False,
        "more": False,
    }
    if not os.path.isdir(run_dir):
        delta["missing"] = True
        return delta, None

    record = None
    try:
        st = os.stat(os.path.join(run_dir, "run.json"))
        run_cur = {"mtime": st.st_mtime, "size": st.st_size}
        try:
            submit_size = os.path.getsize(os.path.join(run_dir, "events", runs.SUBMIT_EVENTS))
        except OSError:
            submit_size = 0
        cached = _RECORDS.get(run_dir)
        if cached and cached[0] == (run_cur, submit_size):
            record = cached[1]
        else:
            record = runs.read_run(run_dir)
            record["jobs"] = runs.jobs_of(run_dir, record)  # submit.jsonl is authoritative
            _RECORDS[run_dir] = ((run_cur, submit_size), record)
        if cursor.get("run") != run_cur:
            delta["run"] = record
        cursor["run"] = run_cur
    except (OSError, ValueError):
        pass

    delta["events"], cursor["events"], delta["more"] = read_events(run_dir, cursor.get("events"), progress, budget)

    save_loc = (record or {}).get("save_loc") or entry.get("save_loc")
    if save_loc:
        # A new run starts after the rows its save_loc already had (runs.csv_baseline); later cursors take over.
        delta["csv"], csv_cur, delta["csv_pending"] = read_csv_delta(
            os.path.join(save_loc, "training_log.csv"), cursor.get("csv") or (record or {}).get("csv_baseline")
        )
        if csv_cur:
            cursor["csv"] = csv_cur

    logs_cur = dict(cursor.get("logs") or {})
    for job in (record or {}).get("jobs") or []:
        key = job.get("job_key")
        if not key:
            continue
        prev = logs_cur.get(key) or {}
        path = _log_path(record, key, prev.get("path"))
        if not path:
            continue
        moved = bool(prev.get("path")) and prev["path"] != path  # e.g. .o -> pbsdsh node_0.out (A5.7)
        off = 0 if moved else int(prev.get("off") or 0)
        try:
            if not moved and prev.get("path") == path and os.path.getsize(path) == prev.get("size"):
                continue
            r = read_log(path, off, int(log_tail_bytes))
        except OSError:
            continue
        r["reset"] = r["reset"] or moved
        logs_cur[key] = {"path": path, "off": r["off"], "size": r["size"]}
        if r["text"] or r["current_frame"] is not None or r["reset"] or r["truncated"]:
            delta["logs"][key] = {
                "path": path,
                "text": r["text"],
                "current_frame": r["current_frame"],
                "offset": r["off"],
                "truncated": r["truncated"],
                "reset": r["reset"],
            }
    cursor["logs"] = logs_cur
    delta["cursor"] = cursor
    return delta, record


def delta_changed(delta: dict) -> bool:
    return bool(
        delta["run"] or delta["events"] or delta["csv"] or delta["logs"] or delta["missing"] or delta["csv_pending"]
    )
