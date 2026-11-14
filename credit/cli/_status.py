"""``credit status``: list submitted runs from the run ledger with their scheduler state."""

import argparse
import json
import sys

from credit import runs


def _summaries(entries: list, index: dict) -> list:
    """Return one status dict per ledger entry, with a single batched scheduler query for all of them."""
    from credit.remote import _delta, _sched

    parents = {e.get("parent_run_id") for e in index.values() if e.get("parent_run_id")}

    loaded, jobs = [], []
    for entry in entries:
        try:
            record = runs.load_run(entry["run_dir"])
        except (OSError, ValueError):
            loaded.append((entry, None))
            continue
        loaded.append((entry, record))
        scheduler = (record.get("target") or {}).get("scheduler") or "pbs"
        jobs += [(scheduler, j["job_id"], entry["run_dir"]) for j in _sched.unfinished_jobs(entry["run_dir"], record)]
    states, errors = _sched.query(jobs)
    for where, message in errors.items():
        print(f"WARNING: could not query {where}: {message}", file=sys.stderr)

    out = []
    for entry, record in loaded:
        summary = {
            "run_id": entry["run_id"],
            "kind": entry.get("kind"),
            "name": entry.get("name"),
            "created": entry.get("created"),
            "run_dir": entry["run_dir"],
        }
        if record is None:
            out.append({**summary, "state": "missing", "jobs": []})
            continue
        try:
            events, _, _ = _delta.read_events(entry["run_dir"], {}, "latest")
        except OSError:
            events = []
        progress = next((e for e in reversed(events) if e.get("type") == "train.progress"), None)
        job_rows = []
        for job in record.get("jobs") or []:
            st = states.get(job.get("job_id"))
            if st is None:
                done = _delta.job_finished(entry["run_dir"], job.get("job_key")) or {}
                st = {"state": "finished" if done else "unknown", "exit_status": done.get("code")}
            job_rows.append({"index": job.get("index"), "job_id": job.get("job_id"), **st})
        epochs = [e for e in events if e.get("type") == "train.epoch"]
        out.append(
            {
                **summary,
                "state": runs.derive_state(
                    record.get("kind"),
                    record,
                    events,
                    states,
                    csv_last_epoch=runs.csv_last_epoch(record.get("save_loc"))
                    if record.get("kind") == "train"
                    else None,
                    has_child=entry["run_id"] in parents,
                ),
                "config": record.get("config"),
                "save_loc": record.get("save_loc"),
                "target": record.get("target"),
                "epochs_total": record.get("epochs_total"),
                "last_epoch": (epochs[-1].get("data") or {}).get("epoch") if epochs else None,
                "progress": (progress or {}).get("data"),
                "jobs": job_rows,
            }
        )
    return out


def _jobs_brief(jobs: list) -> str:
    counts = {}
    for job in jobs:
        counts[job.get("state", "unknown")] = counts.get(job.get("state", "unknown"), 0) + 1
    return ", ".join(f"{n} {state}" for state, n in counts.items()) or "-"


def _print_table(summaries: list) -> None:
    if not summaries:
        print(f"No runs in {runs.ledger_path()}.")
        return
    print(f"{'RUN ID':<32} {'KIND':<10} {'STATE':<10} {'JOBS':<24} NAME")
    for s in summaries:
        print(
            f"{s['run_id']:<32} {s['kind'] or '-':<10} {s['state']:<10} {_jobs_brief(s['jobs']):<24} {s['name'] or ''}"
        )


def _print_detail(s: dict) -> None:
    print(f"Run       {s['run_id']}")
    print(f"Kind      {s['kind']}    Name: {s.get('name')}")
    print(f"State     {s['state']}")
    print(f"Created   {s.get('created')}")
    for label, key in (("Config", "config"), ("Save loc", "save_loc"), ("Run dir", "run_dir")):
        if s.get(key):
            print(f"{label:<9} {s[key]}")
    target = s.get("target") or {}
    if target.get("log_dir"):
        print(f"Logs      {target['log_dir']}")
    if s.get("epochs_total") is not None or s.get("last_epoch") is not None:
        print(f"Epochs    last finished {s.get('last_epoch')} of {s.get('epochs_total')}")
    progress = s.get("progress")
    if progress:
        loss = (progress.get("loss") or {}).get("mean")
        print(
            f"Progress  epoch {progress.get('epoch')}  batch {progress.get('batch')}/{progress.get('batches')}"
            f"  loss {loss}  lr {progress.get('lr')}"
        )
    print("Jobs")
    for job in s.get("jobs") or []:
        extra = []
        if job.get("exec_host"):
            extra.append(job["exec_host"])
        if job.get("walltime_used") or job.get("walltime_limit"):
            extra.append(f"walltime {job.get('walltime_used') or '-'}/{job.get('walltime_limit') or '-'}")
        if job.get("estimated_start"):
            extra.append(f"est. start {job['estimated_start']}")
        if job.get("exit_status") is not None:
            extra.append(f"exit {job['exit_status']}")
        print(f"  {job.get('index')}  {job.get('job_id'):<24} {job.get('state', 'unknown'):<9} {'  '.join(extra)}")


def _status(args: argparse.Namespace) -> None:
    """Print the runs in the ledger (or one run in detail) with live scheduler state."""
    index = runs.ledger_index()
    if args.run_id:
        if args.run_id not in index:
            print(f"ERROR: no run {args.run_id!r} in {runs.ledger_path()}", file=sys.stderr)
            sys.exit(1)
        (summary,) = _summaries([index[args.run_id]], index)
        if args.json:
            print(json.dumps(runs.jsonable(summary), indent=2))
        else:
            _print_detail(summary)
        return
    summaries = _summaries(list(index.values()), index)
    if args.json:
        print(json.dumps(runs.jsonable(summaries), indent=2))
    else:
        _print_table(summaries)
