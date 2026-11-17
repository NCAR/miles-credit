# Remote access (`credit.remote`)

`credit.remote` is the HPC-side half of the CREDIT MCP server and CREDIT Studio.
A laptop runs it over SSH, one short command per request, so nothing has to stay
connected while jobs queue and run:

```bash
echo '{}' | python -P -m credit.remote hello
```

Each verb reads one JSON request on stdin and writes one JSON response line,
prefixed with the bytes `\x1eCREDIT\x1e` so shell start-up noise can be ignored.
`follow` streams frames instead, until its stdin closes.  This page summarizes
what it reads and writes on the HPC side; the full specification (on-disk
layout, events, run state, path scope, cursors, every verb and error code) is
the [Remote protocol reference](remote-protocol.md).

## What it reads

- **The run ledger**, `$CREDIT_HOME/runs/index.jsonl` (default `~/.credit`), which
  `credit submit` appends to.
- **Run directories**, `<save_loc>/.credit/runs/<run_id>/`:
  - `run.json`: the target, resources and job ids
  - `config.yml`: the config snapshot the jobs run
  - `scripts/job-<n>.sh`: the exact scripts that were submitted
  - `events/*.jsonl`: one JSON object per line, one writer per file
- **`training_log.csv`** and the scheduler's job logs in `<save_loc>/logs/`.
  tqdm progress frames are collapsed to the last frame of each line.

`sync` returns everything new since the caller's cursors, plus one batched
scheduler query (`qstat -x -f -F json` per PBS server, `sacct --json` on SLURM).
Set `CREDIT_PBS_BIN` / `CREDIT_SLURM_BIN` to the directory holding the scheduler
tools; PBS defaults to `/opt/pbs/bin` when it exists.

## Events

Batch scripts from `credit submit` export `CREDIT_RUN_DIR`, `CREDIT_RUN_ID` and
`CREDIT_JOB_KEY`, and append `job.started`, `job.exited` and `job.signaled`
(on `TERM`, for walltime and `qdel`) to `events/<job_key>.script.jsonl`.  Inside
the job, `credit.events.EventSink` (rank 0 only) appends training progress,
per-variable running means, epoch rows, checkpoints, the stop reason, warnings,
errors and a heartbeat to `events/<job_key>.proc.jsonl`.  Outside a submitted job
(`CREDIT_RUN_DIR` unset) the sink does nothing.

The cadence is set by an optional config block (or the matching
`CREDIT_EVENTS_*_EVERY_S` environment variables):

```yaml
events:
  progress_every_s: 5   # train.progress
  vars_every_s: 60      # train.vars (per-variable means)
  alive_every_s: 60     # alive heartbeat
```

## Safety

`read`, `ls`, `logs` and `write` only touch paths whose real path lies under a
root.  The roots are the folders listed in `~/.credit/remote.yaml` (written by
`credit remote-setup`) or, without that file, the entries of
`CREDIT_REMOTE_ALLOW` (colon-separated); nothing in the home, work or scratch
areas is served until one of them names it.  A ledger run's directories are
roots too (with a policy, only those inside its roots; never `/`, the home
directory or above it).  Hidden directories are refused, except inside
`.credit/runs/`.  `write` only writes `.yml`/`.yaml` files, never inside a
run directory or `$CREDIT_HOME`, and refuses to overwrite a file whose hash
changed since it was read.

`plan` creates no files.  `submit` re-plans the request it is given and refuses
if the config or any script changed since the plan was approved; submitting the
same plan twice submits nothing the second time.  Optional GPU-hour budgets in
`remote.yaml` (`credit remote-setup --max-gpu-hours-per-plan N
--max-gpu-hours-per-day N`) cap what may be submitted: above the per-plan limit
a person has to type the GPU-hours to confirm, and the daily limit has no
override.  Secrets in returned text are masked, and every call is logged to
`$CREDIT_HOME/audit.jsonl`, unless the policy turns that off.  See
[Policy file](remote-protocol.md#policy-file) and
[Path scope](remote-protocol.md#path-scope).

## Trying it without a cluster

The `local` scheduler runs the job script detached on the current machine (output
in `<save_loc>/logs/local-<pid>.log`), which is enough to exercise the whole
plan → submit → sync → cancel cycle.
