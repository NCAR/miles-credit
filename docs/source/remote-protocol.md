# Remote protocol reference

This page specifies the HPC side of the CREDIT MCP server and CREDIT Studio:
what `credit submit` and `credit.remote` write on the shared filesystem, and
the JSON verbs a laptop runs over SSH.  It is implemented by `credit/runs.py`,
`credit/events.py`, `credit/remote/` and the batch-script builders in
`credit/cli/_submit.py`.  For an overview, see [Remote access](remote.md).
The laptop side (settings, transports, the studio's HTTP API, approvals) is
specified in `CONTRACTS.md` in the credit-mcp project.

## Conventions

- Timestamps are ISO-8601 UTC with a `Z` suffix and millisecond precision:
  `2026-10-30T15:32:07.412Z`.
- All JSON is UTF-8.
- Paths are absolute after `os.path.expandvars` and `expanduser`.
- Readers tolerate unknown keys everywhere, and unknown event `type`s.

### Non-finite numbers

NaN and ±Inf are written as the strings `"NaN"`, `"Infinity"` and
`"-Infinity"`: in every event (the event sink sanitizes), in CSV rows and in
every response.  Everything is written with `json.dumps(..., allow_nan=False)`
after sanitizing, so a diverging run never produces invalid JSON.

## On-disk layout

```
$CREDIT_HOME/                        default ~/.credit (tests always override it)
  runs/index.jsonl                   the ledger
  runs/index.jsonl.lock              lock file for ledger appends
  remote.yaml                        optional second policy file
  audit.jsonl                        one line per credit.remote call
~/.credit/ (in the account's real home)
  remote.yaml                        the policy file written by credit remote-setup
  budget.jsonl, budget.lock          GPU-hour accounting, when a budget is configured
<save_loc>/.credit/runs/<run_id>/    the run directory
  run.json                           written only by submit, cancel and adopt, never by jobs
  config.yml                         snapshot of the config at submit time; the jobs run this
  config_reload.yml                  written by submit for chained and resumed train jobs
  scripts/job-<index>.sh             the exact scripts that were submitted
  events/
    submit.jsonl                     writer "submit"
    <job_key>.script.jsonl           writer "script" (the batch script's markers)
    <job_key>.proc.jsonl             writer "proc": everything CREDIT itself emits (rank 0 only)
<save_loc>/.credit/runs/<run_id>.submit.lock
<save_loc>/logs/                     the scheduler output files (target.log_dir)
```

- **`run_id`**: `<kind>-<YYYYmmddTHHMMSS>-<4 lowercase hex>`, for example
  `train-20261030T153207-a1b2`.  The timestamp is UTC.
- **`job_key`**: the numeric part of the scheduler job id (`5875448` from
  `5875448.desched1`); SLURM ids as they are; `local-<pid>` for the local
  scheduler.
- **One writer per file.**  No file ever has two concurrent writers.  The
  ledger, the audit log and the budget ledger are the exceptions that several
  processes append to, and they append under a lock.
- **Line format.**  Every line is one complete JSON object followed by `\n`.
  A reader ignores a trailing partial line and never advances its cursor past
  one.  Complete lines that don't parse are skipped (but consumed), so one bad
  line can't wedge a reader.

## Batch scripts

`credit submit` and `credit.remote submit` make every batch script report on
itself.  After the scheduler directives, the script:

- exports `PYTHONUNBUFFERED=1`, `CREDIT_RUN_ID`, `CREDIT_RUN_DIR`, any
  `CREDIT_EVENTS_*` variables set when it was planned, and `CREDIT_JOB_KEY`,
  computed *in the script*: `${PBS_JOBID%%.*}`, `$SLURM_JOB_ID` or `local-$$`;
- appends `job.started` to `events/<job_key>.script.jsonl`;
- runs the CREDIT command in a backgrounded subshell and `wait`s on it, so the
  `TERM` trap (walltime, `qdel`) fires at once;
- appends `job.exited {code}` when the command ends.

The `TERM` trap appends `job.signaled {signal: "TERM"}`, forwards `TERM` to the
job's processes, waits for them, then appends `job.exited` with the job's own
exit code.  The subshell ignores `TERM` itself, so `job.exited` is written only
after the job has cleaned up.  The markers are written with `printf` and `date`
only.  **A `job.signaled` with no `job.exited` counts as terminal.**

Chain dependencies are never written into the scripts: `submit` passes them on
the command line (`qsub -W depend=afterok:<id>`,
`sbatch --dependency=afterok:<id>`), together with the job name
`<run_id>-<index>` (`-N` / `--job-name`).  Output goes to `target.log_dir`
(`<save_loc>/logs`) via `-o`; PBS scripts set `-k eod`.

**pbsdsh (Derecho multi-node).**  The launch block runs in a function or
subshell, so its `exit $status` doesn't skip the markers.  The per-node scripts
export the same `CREDIT_*` variables, and their output goes to
`<log_dir>/pbsdsh_logs/<job_key>/node_<i>.out`; rank 0's `node_0.out` is the
job's log while it runs.

**The local scheduler** runs one bash script, started detached in a new
session (`setsid`) with `CREDIT_RUN_ID` in its environment and its output in
`<log_dir>/local-<pid>.log` (that is, `<job_key>.log`).

## The ledger

`$CREDIT_HOME/runs/index.jsonl` is append-only, one object per line:

```json
{"v": 1, "run_id": "train-20261030T153207-a1b2", "run_dir": "/abs/.credit/runs/train-…", "kind": "train",
 "name": "wxformer_1deg_v3", "config": "/abs/config.yml", "save_loc": "/abs/save_loc",
 "created": "2026-10-30T15:32:07.412Z", "created_by": "cli", "parent_run_id": "train-…"}
```

- `parent_run_id` is present only for a resumed run.
- To remove a run from view, append `{"v": 1, "run_id": "…", "removed": true}`.
  Nothing is ever rewritten.
- Writers append the encoded line with a single `write()` on an `O_APPEND`
  descriptor, under `fcntl.lockf` on `index.jsonl.lock`.
- The ledger offset used by `sync` is a byte offset.  If the caller's offset is
  past the end of the file (the ledger was replaced), `sync` returns everything
  from 0 with `ledger_reset: true`.

## run.json

```json
{
  "v": 1,
  "run_id": "train-20261030T153207-a1b2",
  "name": "wxformer_1deg_v3",
  "kind": "train",
  "created": "2026-10-30T15:32:07.412Z",
  "created_by": "cli",
  "config": "/abs/original/config.yml",
  "config_snapshot": "config.yml",
  "config_sha256": "…",
  "save_loc": "/abs/save_loc",
  "target": {"scheduler": "pbs", "cluster": "derecho", "server": "desched1", "queue": "main",
             "account": "NAML0001", "nodes": 2, "gpus": 4, "cpus": 64, "mem": "480GB",
             "walltime": "12:00:00", "log_dir": "/abs/save_loc/logs"},
  "epochs_total": 100,
  "epochs_per_job": 10,
  "jobs": [
    {"index": 1, "job_id": "5875448.desched1", "job_key": "5875448", "depends_on": null,
     "config": "/abs/save_loc/.credit/runs/train-…/config.yml", "script": "scripts/job-1.sh",
     "submitted": "2026-10-30T15:32:08.001Z"}
  ],
  "cancelled": null,
  "csv_baseline": {"rows": 34, "header_sha": "…", "first_row_sha": "…"},
  "parent_run_id": "train-…"
}
```

- **`kind`**: `preprocess | train | rollout | realtime`.
- **`created_by`**: `cli | mcp | gui`.
- **`target.scheduler`**: `pbs | slurm | local`.
- **`epochs_total` / `epochs_per_job`**: train runs only, otherwise `null`.
  They come from `trainer.epochs` and `trainer.num_epoch`.
- **`cancelled`**: `null`, or `{"at": ts, "scope": "all" | "pending", "job_ids": [...]}`.
- **`csv_baseline`**: the state of `<save_loc>/training_log.csv` when the run
  was created (`null` if there was none).  Those rows belong to earlier runs in
  the same `save_loc`, so the run's CSV delta starts after them.  Adopted runs
  have no baseline: their rows are their own history.
- **`parent_run_id`**: present only for a resumed run.
- **Writes are atomic**: a temporary file in the same directory, then
  `os.replace`.

### Jobs: submit.jsonl is authoritative

`submit` appends `job.submitted` to `events/submit.jsonl` immediately after
each successful `qsub`/`sbatch`, before it rewrites `run.json`.  Whenever
`run.json` lags (a crash in between), `submit.jsonl` wins:
`runs.jobs_of(run_dir, record)` merges its `job.submitted` events into
`record["jobs"]`, and every reader of a run's jobs uses it (`cancel`,
`sync`/`follow`, scheduler queries, `logs`, `outputs`, `diagnose`,
`credit status` and the state derivation).  A resumed submit rewrites
`run.json` from the merged view.

## Events

### Envelope

Every line in `events/*.jsonl` has this shape:

```json
{"v": 1, "ts": "2026-10-30T15:32:07.412Z", "type": "train.progress",
 "src": {"writer": "proc", "job": "5875448", "host": "dec0123", "rank": 0},
 "data": {}}
```

### Event types

| type | writer | data |
|---|---|---|
| `run.created` | submit | `{kind, created_by}` |
| `job.submitting` | submit | `{index, job_name}`, appended before each `qsub`/`sbatch` |
| `job.submitted` | submit | `{index, job_id, job_key, depends_on}` |
| `job.submit_failed` | submit | `{index, error}`, when the scheduler refused the job (not after a timeout, which may still have queued it) |
| `run.cancelled` | submit | `{scope, job_ids}` |
| `job.started` | script | `{host}` |
| `job.exited` | script | `{code}`: the shell exit status of the CREDIT command |
| `job.signaled` | script | `{signal}`: `"TERM"`, written by the trap before the script exits |
| `process.started` | proc | `{pid, world_size, credit_version, torch_version, gpus}` |
| `metrics.schema` | proc | `{columns: [str]}`: per-variable train column names, e.g. `train_rmse/ERA5/prognostic/3d/temperature` |
| `train.progress` | proc | see [train.progress](#trainprogress) |
| `train.vars` | proc | `{epoch, batch, values: [number \| null]}`, aligned to the latest `metrics.schema` *in the same file* |
| `train.epoch` | proc | `{epoch, row: {column: value}}`: the `training_log.csv` row for that epoch |
| `checkpoint.saved` | proc | `{path, epoch, kind}`; `kind` is `latest`, `ema` or `epoch` |
| `train.stopped` | proc | `{reason, epoch}`; `reason` is `completed \| early_stop \| nan \| epochs_per_job` |
| `log` | proc | `{level, logger, message}`, WARNING and above, rate-limited |
| `error` | proc | `{exc_type, message, traceback_tail}`, written once before the process exits on an exception |
| `alive` | proc | `{}`, from a daemon thread every `alive_every_s` |

These types are reserved, and readers handle them, but CREDIT doesn't emit
them yet: `rollout.forecast {init_time, steps}`, `rollout.step {init_time,
step, fhr}`, `output.written {path}`, `output.failed {path, error}`,
`preprocess.progress {batch, batches}`, and the checkpoint kinds `best`,
`best_ema` and `backup`.

The `log` handler forwards the same (logger, message) at most once a minute,
and at most 60 records a minute in all, so a warning inside the batch loop
can't flood the file.

`credit.events` is a no-op unless `CREDIT_RUN_DIR` is set (an interactive
`credit train`, a notebook or the tests write nothing), and only rank 0 writes.
The sink never raises into training: an I/O error is logged once and later
events are dropped.

### train.progress

```json
{"epoch": 3, "batch": 1200, "batches": 5000,
 "loss": {"mean": 0.412, "min": 0.398, "max": 0.431, "n": 24},
 "loss_epoch": 0.455,
 "metrics": {"acc": {"mean": 0.91, "epoch": 0.90}},
 "lr": 0.00031, "grad_norm": 1.83, "samples_per_s": 182.4}
```

- **`loss`**: statistics over the batches since the previous `train.progress`.
- **`loss_epoch`**: the running mean for the epoch, the same number tqdm shows.
- **`metrics`**: the trainer's display metrics (from `train_<name>`), in the
  same window/epoch form.
- **`grad_norm`**: `null` when gradient clipping is off.
- **`samples_per_s`**: may be `null`.

### Cadence

`train.progress` is emitted at most every `progress_every_s` (default 5 s),
and always on the last batch of an epoch.  `train.vars` follows the same rule
with `vars_every_s` (default 60 s), and `alive` comes every `alive_every_s`
(default 60 s).  The environment variables `CREDIT_EVENTS_PROGRESS_EVERY_S`,
`CREDIT_EVENTS_VARS_EVERY_S` and `CREDIT_EVENTS_ALIVE_EVERY_S` set them; a
config block overrides those when present:

```yaml
events: {progress_every_s: 5, vars_every_s: 60, alive_every_s: 60}
```

## Run state

`credit.runs.derive_state` and the laptop's state derivation apply the same
rules, in order, and are tested against the same fixture
([Shared state fixture](#shared-state-fixture)).  The result is one of
`cancelled | running | queued | completed | stopped | failed | unknown`.

1. **Terminal markers override a stale scheduler.**  A job with a `job.exited`
   or `job.signaled` marker is finished, whatever a cached scheduler state says.
2. **`cancelled`** if there is a `run.cancelled` event (or `run.json`
   `cancelled`) whose scope is `all`.
3. **`running`** if any job is running (PBS `R`/`E`/`B`).
4. **`queued`** if any job is queued or held.  A PBS `H` whose `Hold_Types`
   contains `s` is waiting on its dependency, and counts as queued.
5. **`cancelled`** if there was a `pending`-scope cancel: it doesn't preempt
   `running` or `queued`, and means `cancelled` once nothing runs or waits.
6. Otherwise, judge the last job (by index) that has a `job.started` marker or
   an exec host; with none, the run is **`unknown`** (this includes a run whose
   jobs are all `gone`).  A job's `train.stopped` and `error` events are taken
   only from its own `<job_key>.proc.jsonl`, so an earlier job's events never
   decide a later one.
   - **`completed`** if it exited 0, and either the kind isn't `train` or its
     last `train.stopped.reason` is `completed` or `early_stop`.
   - **`stopped`** if it was signaled, or its `train.stopped.reason` is
     `epochs_per_job` (a later pending job would already have made the run
     `queued`).
   - **`failed`** if its exit code is non-zero, or it emitted an `error` event.
   - **Train jobs without the event sink.**  If a train job exited 0 with no
     `train.stopped` event: **`completed`** when the CSV's last epoch + 1 ≥
     `epochs_total`, otherwise **`stopped`** (the job's epochs-per-job budget is
     used up), and **`unknown`** with no CSV or no `epochs_total`.
   - **Resumed parents.**  A run that is the `parent_run_id` of another ledger
     run is never judged by the no-sink CSV rule: it is `completed` only through
     its own `train.stopped` with reason `completed` or `early_stop`, otherwise
     `stopped` or `failed` from its markers.
   - **`unknown`** otherwise.

**Epochs done = the last CSV `epoch` value + 1**: the CSV epoch is 0-based.
Resuming uses the same count.

### Shared state fixture

`tests/data/state_cases.json` is a list of cases shaped like this:

```json
{"name": "...", "kind": "train",
 "run": {"jobs": [{"index": 1, "job_key": "101"}], "epochs_total": 30, "epochs_per_job": 10, "cancelled": null},
 "events": [{"file": "101.proc.jsonl", "type": "train.stopped", "data": {"reason": "epochs_per_job", "epoch": 9}}],
 "scheduler": {"101": {"state": "finished", "exit_status": 0}},
 "csv_last_epoch": 9, "has_child": false,
 "expect": "stopped"}
```

`tests/test_runs.py` runs it through `credit.runs.derive_state`, and credit-mcp's
`tests/core/test_shared_states.py` runs it through the laptop's derivation
(reading the file from the sibling checkout, and skipping if it is missing).
It covers every rule above, plus a stale `R` with an exit marker, a
pending-cancel while running, job 2 crashing before its sink after job 1 hit
`epochs_per_job`, no sink with a partial CSV, and a resumed parent.

## Invoking credit.remote

```
<python> -P -m credit.remote <verb>
```

- **Requests.**  One JSON object on stdin; empty stdin means `{}`.  The
  client shell-quotes every argument it sends over SSH; `-P` keeps the working
  directory off `sys.path`.
- **Framing.**  On entry `credit.remote` duplicates stdout onto stderr and keeps
  the original descriptor for itself, so anything else printed (`.bashrc`
  noise, which sshd-launched bash does read, and library warnings) goes to
  stderr.  The response is written on the saved descriptor as one line,
  `\x1eCREDIT\x1e<json>\n`.  Clients take the last sentinel line (for
  `follow`, every sentinel line) and ignore everything else.
- **Responses.**  Every verb except `follow` writes exactly one object:
  - success: `{"ok": true, "protocol": 1, "result": {...}}`
  - failure: `{"ok": false, "protocol": 1, "error": {"code", "message", "fix", "data"}}`,
    where `fix` is a hint or `null` and `data` is machine-readable detail or `null`.
- **Exit codes.**  0 whenever a response was written, even for application
  errors.  Non-zero means the module itself crashed.
- **`SystemExit` from a reused CLI helper** becomes `config_error` carrying the
  tail of what it printed.  Any other exception becomes `internal`, with the
  exception type, its message and at most the last three frames.

### Error codes

| code | meaning |
|---|---|
| `bad_request` | a missing or malformed field, an unknown verb |
| `not_found` | no such run, file, job log, example or page |
| `forbidden_path` | the path is outside the allowed roots, hidden, or not writable (see [Path scope](#path-scope)); `fix` names `credit remote-setup` when no policy exists yet |
| `scheduler_error` | `qstat`/`qsub`/`qdel` (or the SLURM tools) failed |
| `config_error` | the config or a plan option is unusable; for a bad plan option `data.field` names it |
| `conflict` | a hash mismatch (`write`, `submit`), a submit already in progress, or the budget lock is busy |
| `over_budget` | `submit` refused by a GPU-hour budget; see [Budget checks](#budget-checks) |
| `unsupported` | the verb isn't available over the protocol (`convert`, `render`) |
| `needs_dependency` | a heavy verb couldn't import torch or another dependency |
| `internal` | anything else |

### Import budget

`hello`, `sync`, `follow`, `read`, `ls`, `logs`, `write`, `history`, `cancel`,
`plan`, `submit`, `adopt`, `diagnose`, `outputs`, `examples`, `docs`, `convert`
and `render` import only the stdlib, `yaml` and `credit.runs`/`credit.remote`,
never torch, numpy or xarray.  `plan` and `submit` also import the batch-script
builders in `credit.cli._submit`, which must stay torch-free; `plan` for a
rollout must not import `credit.trainers.rollout_utils` or anything else heavy,
and leaves out what it can't compute cheaply.  The heavy verbs, `check`,
`components` and `build_config`, import what they need inside the verb.

### Environment variables

- `CREDIT_HOME`: the ledger root (default `~/.credit`).
- `CREDIT_PBS_BIN`: the directory holding `qsub`/`qstat`/`qdel`.  Default
  `/opt/pbs/bin` when it exists, otherwise `PATH`: Casper's `qstat` on `PATH`
  is a caching wrapper that can lag by minutes.
- `CREDIT_SLURM_BIN`: likewise for `sbatch`/`sacct`/`squeue`/`scancel`.
- `CREDIT_REMOTE_ALLOW`: allowed roots, colon-separated, used only when no
  policy file sets `roots`.
- `CREDIT_EVENTS_*_EVERY_S`: event cadences, copied into the scripts at plan
  time (see [Cadence](#cadence)).
- `CREDIT_REMOTE_LOCAL_CMD`: replaces `python -m credit.cli` in local-scheduler
  scripts, so tests can run a fake trainer through the real script.

## Policy file

`credit remote-setup` writes `~/.credit/remote.yaml`, the policy that limits
what `credit.remote` may serve and submit on this host.  It is deliberately
not an MCP tool: widening what a remote client can reach is something the user
does in their own shell on the HPC system.

```yaml
roots:            # the folders that may be served
  - /glade/work/$USER/CREDIT_runs
  - /glade/work/$USER/configs
redact: true      # mask secrets in text sent back (default true)
audit: true       # append one line per call to $CREDIT_HOME/audit.jsonl (default true)
budgets:          # GPU-hour limits on submit (pbs/slurm only; default none)
  max_gpu_hours_per_plan: 200   # above this a person must type the GPU-hours to confirm
  max_gpu_hours_per_day: 1000   # rolling 24 h, no override
```

Two files are read: `~/.credit/remote.yaml` in the account's real home (from
the password database, so an environment change can't skip it) and
`$CREDIT_HOME/remote.yaml`.

- **`roots`**: each file that sets them narrows the result further.
- **`redact` / `audit`**: `false` takes effect only when no file says `true`.
- **`budgets`**: each limit is the smallest any file sets.  A value that isn't
  a positive number, an unknown key in the block, or an unreadable file counts
  as 0, so nothing can be submitted until it is fixed; `hello.policy.budget_errors`
  says why.
- An unreadable policy file also serves no roots: it fails closed.

`credit remote-setup` takes `--root DIR` (repeatable), `--yes`, `--show`,
`--no-redact`, `--no-audit`, `--max-gpu-hours-per-plan H` and
`--max-gpu-hours-per-day H`.

**Redaction.**  With `redact` on, text sent back by `read`, `logs`, `sync`,
`follow` and `diagnose` (and the `message`/`traceback_tail` of `log` and
`error` events) has secrets replaced by `«redacted»`: private keys, common API
token formats, URL passwords, `Authorization: Bearer` values, and the values of
`key: value` pairs whose key names a password, secret, token, API key or
credential.  `write` refuses text that still contains `«redacted»`, since
writing it back would erase the secrets.

**Audit log.**  With `audit` on, every call appends
`{ts, verb, ok, code, client, req}` to `$CREDIT_HOME/audit.jsonl`, where
`client` is the SSH client address and `req` holds only identifying fields
(paths, run ids, kinds, scopes, names, `confirm_gpu_hours`), never file
contents.  Auditing never makes a verb fail.

## Path scope

Path scope is the security boundary.  `read`, `ls`, `logs` and `write`, and
every path in a request (`plan`/`submit`'s `request.config` unless resuming,
`adopt`'s `save_loc` and `config`, `outputs`' `save_loc` and forecast
directory, `check`'s `config`), go through the same resolver:

- **Static roots.**  The policy's `roots`.  Without a policy, only the entries
  of `CREDIT_REMOTE_ALLOW`: nothing in the home, work or scratch areas is
  served until the user runs `credit remote-setup`.  The environment can't
  widen a policy.
- **Ledger roots.**  Each ledger run's `run_dir`, `save_loc` and config
  directory.  With a policy, only those inside a static root.  Without one,
  every ledger run (the user created it), except a folder that is `/`, the
  home directory or above it.  A ledger root with a hidden component (other
  than its own `.credit/runs/<run_id>`) never counts: a `save_loc` of `~/.ssh`
  would otherwise serve everything inside it.
- **The check is on the `realpath`**, so neither `..` nor a symlink can escape
  a root.
- **Hidden components.**  A path with a component that starts with `.` below
  its root is refused, except inside a ledger run's `<save_loc>/.credit/runs/`.
  `ls` leaves such entries out of its listing.
- **`adopt`** accepts a `save_loc` or `config` only inside the static roots.
- **`write`** additionally accepts only `.yml`/`.yaml` files, never under
  `$CREDIT_HOME`, never inside a run directory and never in a dot-directory.
- **Configs** named in a request must be `.yml`/`.yaml` files that parse to a
  mapping.  YAML errors report only the problem and its line and column, never
  a snippet, so a request naming the wrong file can't leak its contents.
- **Size caps:** `read` returns at most 8 MB per call, and `logs` with `grep`
  scans at most 4 MB of text.

`hello.allowed_roots` lists the static roots plus the ledger roots in scope.

## Cursors

A run cursor is a public, mergeable JSON object.  The laptop stores it and
sends it back; it may read it, and merges two cursors by taking the larger
value per key.

```json
{"run": {"mtime": 1730301127.5, "size": 1803},
 "events": {"submit.jsonl": 812, "5875448.proc.jsonl": 18234},
 "csv": {"rows": 34, "size": 20811, "mtime": 1730301127.5, "header_sha": "…", "first_row_sha": "…"},
 "logs": {"5875448": {"path": "/abs/save_loc/logs/…", "off": 120034, "size": 120500}}}
```

- `null` (or a missing part) means "from the beginning".  For the CSV, a new
  run starts from its `csv_baseline`.
- `logs` and `csv` merge by the entry with the larger `off` / `rows`.  When a
  delta carries `csv.reset` or `logs[k].reset`, the laptop stores that part of
  the new cursor as it is instead of taking the max.
- Every returned event carries `"file"` (the events file name) and `"off"`
  (the byte offset where its line starts), so a consumer can deduplicate
  events delivered by both `sync` and `follow` on `(run_id, file, off)`.

## Verbs

Each verb's section gives its request, then `→` and the `result` object of a
successful response.  The heavy verbs (see [Import budget](#import-budget))
are marked.

## hello

`{}` →

```json
{"protocol": 1, "credit_version": "…", "python": "3.11.9", "host": "casper-login1", "user": "…",
 "system": "casper", "schedulers": {"pbs": true, "slurm": false}, "credit_home": "/abs/.credit",
 "allowed_roots": ["…"], "repo_root": "/abs/miles-credit",
 "policy": {"configured": true, "files": ["…/remote.yaml"], "redact": true, "audit": true,
            "budgets": {"max_gpu_hours_per_plan": 200.0, "max_gpu_hours_per_day": null},
            "budget_errors": [], "used_gpu_hours_24h": 12.5, "setup_command": "credit remote-setup"}}
```

- `system` is `casper | derecho | perlmutter | ncar | local`, from hostname
  rules (Casper's hosts include `crlogin*` and `crhtc*`), without importing torch.
- `repo_root` is the checkout or install root that holds `docs/` and `config/`.
- `used_gpu_hours_24h` is `null` when the budget ledger can't be read.

## sync

Request:

```json
{"ledger_offset": 0,
 "runs": {"<run_id>": "<cursor or null>"},
 "progress": "all",
 "max_bytes": 4194304,
 "log_tail_bytes": 65536,
 "scheduler": true}
```

Response:

```json
{"as_of": "ts", "ledger_offset": 0, "ledger": ["ledger lines since the offset"],
 "runs": {"<run_id>": "RunDelta"},
 "scheduler": {"<job_id>": "JobState"},
 "scheduler_errors": {"<server or scheduler>": "message"},
 "ledger_reset": true}
```

- **Which runs:** every run named in `runs`, plus every run in the returned
  ledger lines (cursor `null`).  A named run that isn't in the ledger comes
  back with `missing: true`.
- **`progress`** is `"all"` (default) or `"latest"`; anything else is
  `bad_request`.
  - `latest` keeps only the last `train.progress` and the last `train.vars`
    per file, but advances the cursor to the end.  It is meant for the first
    sight of a run and for long gaps; charts then come from `history`.
  - `all` returns lines up to `max_bytes` (default 4 MB, one budget for the
    whole response).  When the budget runs out, `RunDelta.more` is `true` and
    the cursor stops after the last returned line.
- **`scheduler: true`** (default) queries every job of every included run that
  has no exit marker yet; see [Scheduler queries](#scheduler-queries).
- **`ledger_reset`** is present only when the offset was past the end.

### RunDelta

```json
{"run": "run.json, or null",
 "events": ["event", "…"],
 "csv": {"columns": ["str"], "rows": [["…"]], "first_row": 0, "reset": false},
 "csv_pending": false,
 "logs": {"<job_key>": {"path": "str", "text": "str", "current_frame": "str or null", "offset": 0,
                        "truncated": false, "reset": false}},
 "cursor": "the new cursor",
 "missing": false,
 "more": false}
```

- **`run`** is the current `run.json` (jobs merged from `submit.jsonl`) when
  `cursor.run` is null or the file's mtime or size changed, else `null`.
- **`events`**: every complete line from every `events/*.jsonl` past its
  cursor offset, each with `file` and `off`, sorted by `ts`.
- **`csv`**: new rows of `<save_loc>/training_log.csv`, or `null`; see
  [CSV rows](#csv-rows).
- **`csv_pending`**: the CSV exists but was withheld because it hasn't settled
  (an unterminated last line, or an mtime under 2 s old).  Look again shortly,
  even if the run is terminal.
- **`logs`**: per job, the new text of its log; see [Log text](#log-text).
- **`missing`**: the run directory (or its ledger entry) no longer exists.

### CSV rows

Older CREDIT rewrote `training_log.csv` in place once per epoch, so the file is
only read once it has settled:

- `csv: null` **without advancing the cursor**, and `csv_pending: true`, while
  the last line is unterminated or the mtime is less than 2 s old.
- `rows` are the rows at index ≥ `cursor.csv.rows`; `first_row` is that index.
- Every row with `reset: true` if `header_sha` or `first_row_sha` changed, or
  the row count shrank.
- Numbers are parsed as floats where possible; the rest stay strings.

### Log text

The log of a job is the scheduler output file in `target.log_dir`:

- PBS: rank 0's `pbsdsh_logs/<job_key>/node_0.out` when it exists, otherwise
  `<job_key>.<server>.OU` (what PBS writes with `-k eod` into an `-o`
  directory) or `*.o<job_key>`;
- SLURM: `*-<job_key>.out`;
- local: `<job_key>.log`.

Reading from the cursor offset to EOF, capped at `log_tail_bytes`:

- `\r\n` is normalized to `\n`, and **each `\n`-terminated line is collapsed to
  the text after its last `\r`**: those are tqdm frames.
- `off` (`offset` in the delta) advances to just after the last `\n` or `\r`,
  whichever is later.
- **Skipping ahead.**  When the cap forces a skip, the first partial segment is
  dropped and `truncated` is `true`.
- **Reset.**  If the file is now shorter than `off`, reading restarts from 0
  with `reset: true`.  If the job's log path changed (for example from the
  `.o` file to pbsdsh's `node_0.out`), the entry also carries `reset: true`.
- **`current_frame`** is the text after the last `\r` that follows the last
  `\n`: the live tqdm bar, read from at most the last 4096 bytes; `null` when
  the unterminated tail has no `\r`.

### JobState

```json
{"state": "queued|held|running|finished|gone|unknown", "raw_state": "Q", "exit_status": 0,
 "exec_host": "dec0123", "estimated_start": "ts", "walltime_used": "07:12:03",
 "walltime_limit": "12:00:00", "comment": "…", "hold_types": "s"}
```

Every field but `state` may be `null`.

### Scheduler queries

- **PBS**: one `qstat -x -f -F json <ids…>` per server.  Stdout is parsed
  whatever the exit code; ids missing from `Jobs` are `gone`.  If stdout isn't
  JSON, `qstat -x -f` text is parsed instead.
  - State: `Q`/`W`/`T`/`M` → queued; `H`/`S`/`U` → held; `R`/`E`/`B` →
    running; `F`/`X` → finished.
  - `exit_status` comes from `Exit_status` (256 and above mean a signal;
    negative values are PBS codes), `exec_host` is the part of `exec_host`
    before the first `/`, `walltime_limit` is `Resource_List.walltime`,
    `walltime_used` is `resources_used.walltime`, and `estimated_start` is
    `estimated.start_time` (ctime in the server's local time) converted to UTC.
  - The test fixtures are hand-written from the PBS Pro docs.
- **SLURM**: one `sacct --json -j <ids>`, plus `squeue --json` for pending
  jobs' estimated start.
- **Local**: a pid check (the process must still belong to the run, see
  [cancel](#cancel)) and the job's markers.

A failed query is reported in `scheduler_errors` and its jobs are left out of
`scheduler`.

## follow

A streaming verb.  stdin carries newline-delimited control messages:

- first: `{"runs": {"<run_id>": cursor}, "interval_s": 2, "log_tail_bytes": 16384}`
- later: `{"add": {"<run_id>": cursor}}`, `{"remove": ["<run_id>"]}`,
  `{"interval_s": n}` (at least 0.2), `{"log_tail_bytes": n}`

stdout carries sentinel-framed frames, flushed after each one:

- `{"type": "hello", "protocol": 1, "ts": …}` first;
- `{"type": "frame", "ts": …, "runs": {"<run_id>": RunDelta}}`, only when
  something changed.  Frames use `progress: "all"`, include `run` only when
  `run.json` changed, and never carry scheduler state;
- `{"type": "ping", "ts": …}` after 15 s without a frame;
- `{"type": "error", "run_id": …, "error": {code, message, fix}}` for a
  per-run problem (or a bad control message, with `run_id: null`); the stream
  continues.

A run that goes missing is reported once and dropped.  Each tick `stat`s only
the files it needs.  The verb exits cleanly when stdin reaches EOF.  Consumers
sync a run before opening (or reopening) its follow channel, so the cursors
are current and the scheduler state is fresh.

## read

`{"path": abs, "offset": 0, "length": 1048576, "encoding": "auto"}` →
`{path, size, mtime, sha256, offset, data, encoding: "utf-8"|"base64", eof, redacted}`

- With `encoding: "auto"`, a file with a NUL byte in its first 8 KB comes back
  as `base64`, anything else as `utf-8` (decoded with `errors=replace`).
- `length` is capped at 8 MB.
- `sha256` covers the whole file, and only for files of 16 MB or less.
- `redacted` is the number of secrets masked in `data`.

## ls

`{"path": abs, "pattern": "*", "limit": 1000}` →
`{path, entries: [{name, type: "file"|"dir", size, mtime}], truncated}`

`pattern` is an fnmatch pattern on entry names and must not contain `/`.

## logs

`{"run_id", "job_key": null, "tail_bytes": 65536, "grep": null}` →
`{job_key, path, text, current_frame, offset}`

- `job_key: null` means the latest job that has a log file.
- `text` gets the same `\r` collapsing as `sync`, and may end with an
  unfinished line.
- `offset` is the byte offset after the last complete line: the same cursor
  `sync` hands out as `logs.<job_key>.off`, so a reader can tell which streamed
  appends it already has.
- `grep` is a Python regex applied line by line after collapsing.

## write

`{"path": abs, "text": str, "if_sha256": str|null}` → `{path, sha256}`

- `if_sha256: null` (or absent) means the file must not exist yet.  A missing
  file or a hash mismatch returns `conflict`.
- The write is atomic and goes to the realpath, so a symlink isn't replaced,
  and an existing file keeps its mode and group.
- `.yml`/`.yaml` only, within [Path scope](#path-scope); text containing
  `«redacted»` is `config_error`.

## history

`{"run_id", "fields": ["loss.mean", "loss_epoch", "lr", "grad_norm"], "buckets": 1500, "t0": ts|null, "t1": ts|null}` →
`{"t": [ts], "fields": {"<field>": {"min": [], "mean": [], "max": []}}, "points": int}`

- Downsamples the `train.progress` events of every `*.proc.jsonl` into time
  buckets.
- Fields are dotted paths into `data`; `metrics.<name>.mean` works too.
- `train.vars` columns can be requested as `vars.<column>`, aligned to the
  latest `metrics.schema` in the same file.

## cancel

`{"run_id", "scope": "all"|"pending"}` → `{cancelled: [job_id], errors: {job_id: message}}`

- `pending` means jobs that are queued or held; `all` also cancels running
  jobs and jobs in an unknown state.  With `pending`, a failed scheduler query
  is `scheduler_error`, since it can't tell which jobs are pending.
- Jobs are cancelled last first, so deleting a parent doesn't make PBS drop
  its `afterok` dependents before they are reached.
- It uses `qdel`, `scancel`, or `TERM` to the local process group.  A local job
  is signalled only after checking the process still belongs to the run (on
  Linux `CREDIT_RUN_ID` in its environment, elsewhere a start time within 2
  minutes of the job's `submitted`), so a reused pid is never signalled.
- When anything was cancelled, it appends `run.cancelled` and updates
  `run.json`.

## plan

Request:

```json
{"kind": "train", "config": "/abs/config.yml", "scheduler": "pbs", "cluster": "derecho", "run_id": null,
 "options": {"gpus": 4, "nodes": 2, "cpus": null, "mem": null, "walltime": null, "account": null,
             "queue": null, "gpu_type": null, "constraint": null, "qos": null, "chain": null,
             "reload": false, "jobs": 1, "init_time": null, "steps": 40, "conda_env": null,
             "launcher": "mpiexec", "resume_from": null}}
```

Response `{"plan": Plan}`:

```json
{"run_id": "train-…", "run_dir": "/abs/save_loc/.credit/runs/train-…", "kind": "train",
 "config": "/abs/config.yml", "config_sha256": "…", "save_loc": "/abs/save_loc", "name": "…",
 "target": {"…": "as in run.json"}, "epochs_total": 100, "epochs_per_job": 10,
 "jobs": [{"index": 1, "script": "<full script text>", "config": "<run_dir>/config.yml",
           "depends_on_index": null}],
 "warnings": ["…"], "gpu_hours_estimate": 960.0,
 "cli": "credit submit --cluster derecho -c /abs/config.yml --gpus 4 --nodes 2 --chain 10",
 "request": {"…": "the request, with run_id"}, "scripts_sha256": ["…"], "log_dir": "/abs/save_loc/logs",
 "parent_run_id": null}
```

- **Purity.**  `plan` creates no files and no directories.  The builders only
  name the log directory; `submit` creates it.
- **Determinism.**  `run_id` is optional; without one, `plan` generates one
  (and must match `<kind>-<YYYYmmddTHHMMSS>-<4 hex>` when given).  The same
  request with the same `run_id` and the same config bytes produces the same
  scripts, which is what lets `submit` re-plan and compare.
- **Scripts** are final (markers, `CREDIT_*` exports, `-o <log_dir>`).  They
  run `<run_dir>/config.yml`, the snapshot; chained train jobs 2..N (and every
  job of a resumed or `reload` run) run `<run_dir>/config_reload.yml`, which
  `submit` writes.  Nothing writes `<save_loc>/config_reload.yml`.
- **The number of train jobs** is `options.chain` when given, otherwise
  `ceil((epochs_total − epochs done) / epochs_per_job)`, at least 1.
- **Rollout** jobs run in parallel, never chained (a warning says that per-job
  init-time subsetting isn't implemented for gen2 rollout).
- **`scheduler: "local"`** runs one job; asking for more produces a warning.
- **A config without a `pbs:`/`slurm:` block** works from `options` alone.  A
  conda environment is required for pbs/slurm (`options.conda_env`, or
  `<scheduler>.conda` in the config).  Anything missing is `config_error`,
  never a process exit.
- **`gpu_hours_estimate`** is walltime × nodes × GPUs × jobs, or `null` without
  a walltime or GPU count (and always for local).
- **Paths.**  `config` and the config's `save_loc` must be within
  [Path scope](#path-scope) and contain no whitespace.

### Plan options

Only the keys above are accepted, and every value is checked before it can
reach a script, because each one ends up in a `#PBS`/`#SBATCH` directive or a
shell line and the GPU-hour estimate only sees the resolved target: nothing may
add a directive, a chunk or a command.

- `gpus`, `nodes`, `cpus`, `chain`, `jobs`, `steps`: whole numbers ≥ 1.
- `reload`: a boolean.
- `walltime`: parsed the way the scheduler reads it.  PBS reads
  `[[HH:]MM:]SS`, so a bare `600` is seconds; SLURM reads `MM`, `MM:SS`,
  `HH:MM:SS` and `D-HH[:MM[:SS]]`, so a bare `600` is minutes.
- `account`, `queue` (`name` or `name@server`), `gpu_type`, `qos`,
  `constraint`, `mem`, `conda_env`, `init_time` (`YYYY-MM-DD[THH[:MM[:SS]]]`),
  `launcher` (`mpiexec` or `pbsdsh`), `resume_from` (a run id): plain values
  with no spaces, newlines or shell characters.
- `env_setup`, `modules` and other script lines come only from the config file.
- `null` means the default for any key.

A refused option is `config_error` with `data.field` naming the key.
`created_by` is not an option; it goes only to `submit`.

### Resuming a run

`options.resume_from: "<run_id>"` plans a new train run that continues the
named one.  Its config is the parent's `config.yml` snapshot with `reload`
forced, and its `run.json` and ledger entry carry `parent_run_id`.  The chain
defaults to the epochs left (see above, counting epochs done from the CSV).  A
parent with nothing left to train is `config_error` ("nothing left to train"),
and `resume_from` applies to train runs only.

## submit

`{"request": <plan.request>, "scripts_sha256": [...], "config_sha256": "…", "created_by": "mcp", "confirm_gpu_hours": null}` →
`{run_id, run_dir, jobs: [{index, job_id, job_key}], errors: [...], already_submitted?}`

1. Check the GPU-hour budgets ([Budget checks](#budget-checks)).
2. Re-plan from `request` (which must carry the plan's `run_id`).  A different
   `config_sha256` or any different script hash is `conflict` ("plan again"),
   so what runs is exactly what was approved.
3. Take `fcntl.lockf(LOCK_EX | LOCK_NB)` on
   `<save_loc>/.credit/runs/<run_id>.submit.lock` for the rest of the call.  If
   it is busy, return `conflict` ("submit in progress; sync the run and
   retry"): a retry racing a submit that is still running (a client timeout,
   two clients) must not queue the jobs twice.
4. Create the run directory with `exist_ok=False` (`config.yml` from the
   hashed bytes, `scripts/`, `run.json`, the ledger entry and `run.created`),
   and the log directory; write `config_reload.yml` if a job needs it.
5. Submit the jobs in order.  Before each `qsub`/`sbatch`, append
   `job.submitting {index, job_name}`; right after it succeeds, append
   `job.submitted` and update `run.json`; when the scheduler refuses it,
   append `job.submit_failed`.  `qsub`/`sbatch` time out after 120 s.  The job
   id is the last stdout line that looks like one (`^\d+(\[\d*\])?\.\S+$` for
   PBS; a number, or `Submitted batch job N`, for SLURM).

If job *k* fails, earlier jobs stay submitted, jobs that depend on it are
skipped, and `errors` says what happened.  An error is a string, or
`{"code": "partial_submit", "index", "job_name", "message"}` for a job that
may or may not be queued.

### Recovery and idempotency

Submitting the same plan again when the run directory exists returns
`already_submitted: true` with the recorded jobs, and submits only the jobs
that never got as far as an intent (or whose intent ended in
`job.submit_failed`).  A job with an intent but no outcome (a crash, or a
`qsub` timeout, which may still have queued it) is reported as
`partial_submit` with its job name and never resubmitted blindly.

### Budget checks

For pbs/slurm, when a budget is configured, `submit` refuses with
`over_budget`, `data: {kind, estimate, limit, …}`:

- `kind: "invalid"` when the policy's budgets are invalid (nothing may be
  submitted until they are fixed);
- `kind: "no_estimate"` when the GPU-hours can't be estimated (no walltime);
- `kind: "per_plan"` when the estimate is over `max_gpu_hours_per_plan`, unless
  the request carries `confirm_gpu_hours` ≥ the estimate.  Only a person types
  that number; CREDIT Studio sends it after they did;
- `kind: "per_day"` when the GPU-hours of the jobs this call would submit,
  added to those recorded in the last 24 h, go over `max_gpu_hours_per_day`
  (with `new_gpu_hours` and `used_gpu_hours_24h`).  There's no override.

The check and the submit hold an `flock` on `~/.credit/budget.lock` (in the
real home); after 120 s waiting for it, `submit` returns `conflict`.  The
GPU-hours of newly submitted jobs are appended to `~/.credit/budget.jsonl` as
`{ts, run_id, gpu_hours}`, only while a budget is configured.  Jobs already
submitted cost nothing on a retry.  The local scheduler has no budget (it
spends no allocation).

## adopt

`{"save_loc", "config": null, "kind": "train", "name": null, "job_ids": [], "scheduler": "pbs", "created_by": "mcp"}` →
`{run_id, run_dir}`

Registers a run that was started outside the ledger.

- `save_loc` and `config` must be inside the static roots.  Without `config`,
  `<save_loc>/model.yml` or `<save_loc>/config.yml` is used.
- `job_ids` must match `^\d+(\[\d*\])?(\.[\w.-]+)?$`; they are recorded as
  `job.submitted` in order.
- The adopted run has no `csv_baseline`: the CSV's rows are its own history.

## diagnose

`{"run_id", "job_key": null}` → `{job_key, exit_status, signal, traceback, banners: [str], excerpt}`

Scans the whole job log (capped at 64 MB, read from the end) plus the pbsdsh
`node_*.out` files, and returns the last Python traceback, up to 20 known
banner lines (`DATA LOADING HANG DETECTED`, `CUDA out of memory`,
`RendezvousConnectionError`, `NCCL error`, `DUE TO TIME LIMIT`, …), the last
60 lines, and the exit code and signal from the job's markers.

## outputs

`{"run_id"}` → `{checkpoints: [{path, size, mtime}], forecast_dir: str|null, files: [{path, size, mtime}], truncated}`

- `checkpoints` are the `*.pt` files directly in `save_loc`.
- The forecast directory comes from the snapshot config's
  `inference.save_forecast` (gen2) or `predict.save_forecast`.  Outside
  [Path scope](#path-scope) it is named but not listed.
- At most 2000 files are listed.

## examples

`{"name": null}` → `{"examples": [names]}`, or with a name `{"name", "text"}`,
from `<repo_root>/config/gen_2/examples/*.yml`.

## docs

`{"page": null}` → `{"pages": [names]}`, or with a page `{"page", "text"}`,
from `<repo_root>/docs/source/*.md`.  Only names from the listing are served.

## check

(heavy) `{"config": abs|null, "yaml": str|null, "deep": false}` →
`{"report": <credit check report>, "derived": {"channels_in", "channels_out", "grid", "steps_per_epoch", "n_jobs"}}`

Runs `credit check` on a file in scope (a `.yml`/`.yaml` mapping) or on YAML
text.  Derived values are best-effort: what can't be computed is left out.

## components

(heavy) `{"kind": null|str, "key": null|str}`

- Without `key`: `{"kinds": {"models": [keys], "trainers": [...], "datasets": [...], "preblocks": [...], "postblocks": [...], "losses": [...], "metrics": [...]}}`,
  or only the named kind.
- With `kind` and `key`: `{kind, key, class, module, doc, params: [{name, default, annotation, required}]}`.

## build_config

(heavy) `{"state": {…credit begin state…}}` → `{"yaml": str}`

The non-interactive `credit begin`: `_begin._build_config` plus the same
dumping `credit begin` does.  It only returns the YAML; save it with `write`.
A request with `write: true` is `bad_request`.

## convert and render

Both return `unsupported` (run `credit convert` or `credit plot` on the host).
The verbs exist so clients get a clear answer.
