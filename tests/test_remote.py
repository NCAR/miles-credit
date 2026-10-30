"""Tests for ``python -m credit.remote``: every verb through a real subprocess, plus the delta readers.

No real scheduler is used: PBS / SLURM tools are fake shell scripts on a temp
dir that ``CREDIT_PBS_BIN`` / ``CREDIT_SLURM_BIN`` point to, and the ``local``
scheduler runs a fake trainer through ``CREDIT_REMOTE_LOCAL_CMD``.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import yaml

from credit import runs
from credit.remote import _delta, _sched

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENTINEL = b"\x1eCREDIT\x1e"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _write_exe(path, text):
    path.write_text(textwrap.dedent(text))
    path.chmod(0o755)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An isolated HOME / CREDIT_HOME, fake PBS + SLURM tools, and the same env for subprocesses."""
    home = tmp_path / "home"
    home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.log"
    _write_exe(
        bindir / "qsub",
        f"""\
        #!/bin/bash
        echo "qsub $*" >> {calls}
        n=$(cat {tmp_path}/qsub.n 2>/dev/null || echo 1000)
        n=$((n + 1)); echo $n > {tmp_path}/qsub.n
        echo "$n.desched1"
        """,
    )
    _write_exe(bindir / "qstat", f'#!/bin/bash\necho "qstat $*" >> {calls}\ncat {tmp_path}/qstat.json\n')
    _write_exe(bindir / "qdel", f'#!/bin/bash\necho "qdel $*" >> {calls}\n')
    _write_exe(bindir / "sbatch", f'#!/bin/bash\necho "sbatch $*" >> {calls}\necho "Submitted batch job 77"\n')
    _write_exe(bindir / "sacct", f'#!/bin/bash\necho "sacct $*" >> {calls}\ncat {tmp_path}/sacct.json\n')
    _write_exe(bindir / "squeue", f'#!/bin/bash\necho "squeue $*" >> {calls}\necho "{{}}"\n')
    values = {
        "HOME": str(home),
        "CREDIT_HOME": str(home / ".credit"),
        "CREDIT_PBS_BIN": str(bindir),
        "CREDIT_SLURM_BIN": str(bindir),
        "PYTHONPATH": REPO,
    }
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    for var in ("SCRATCH", "WORK", "PSCRATCH", "CREDIT_REMOTE_ALLOW", "CREDIT_REMOTE_LOCAL_CMD", "CREDIT_RUN_DIR"):
        monkeypatch.delenv(var, raising=False)
    return {"home": home, "tmp": tmp_path, "calls": calls}


def call(verb, req=None, raw=None, check_ok=True):
    """Run one verb in a subprocess and return the decoded response."""
    proc = subprocess.run(
        [sys.executable, "-P", "-m", "credit.remote", verb],
        input=raw if raw is not None else json.dumps(req or {}).encode(),
        capture_output=True,
        env=os.environ.copy(),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    lines = [ln for ln in proc.stdout.split(b"\n") if ln.startswith(SENTINEL)]
    assert len(lines) == 1, proc.stdout
    resp = json.loads(lines[-1][len(SENTINEL) :])
    assert resp["protocol"] == 1
    if check_ok:
        assert resp["ok"], resp
        return resp["result"]
    return resp


def make_config(env, name="exp", **extra):
    exp = env["home"] / name
    exp.mkdir(exist_ok=True)
    conf = {
        "save_loc": str(exp / "save"),
        "trainer": {"epochs": 4, "num_epoch": 2, "train_batch_size": 2},
        "pbs": {"conda": "/opt/envs/credit", "account": "NAML0001", "queue": "main"},
    }
    conf.update(extra)
    path = exp / "config.yml"
    path.write_text(yaml.safe_dump(conf))
    return path


def submit_req(plan):
    return {
        "request": plan["request"],
        "scripts_sha256": plan["scripts_sha256"],
        "config_sha256": plan["config_sha256"],
        "created_by": "mcp",
    }


def make_run(env, save=None, jobs=(), scheduler="pbs"):
    """Create a run directly through credit.runs (no scheduler involved)."""
    cfg = make_config(env)
    save = save or env["home"] / "exp" / "save"
    run_id = runs.new_run_id("train")
    target = {"scheduler": scheduler, "log_dir": str(save / "logs")}
    record = runs.create_run(run_id, "train", str(cfg), str(save), target)
    run_dir = runs.run_dir_for(str(save), run_id)
    for i, jid in enumerate(jobs, 1):
        runs.add_job(run_dir, record, {"index": i, "job_id": jid})
    (save / "logs").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, save


def ev_line(type_, job="11", **data):
    return runs.encode_line(runs.make_event(type_, data, "proc", job=job))


def wait_for(predicate, timeout=30.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("timed out")


# ---------------------------------------------------------------------------
# Envelope and framing
# ---------------------------------------------------------------------------


def test_envelope_framing_and_errors(env):
    result = call("hello")
    assert result["protocol"] == 1 and result["system"] in ("local", "casper", "derecho", "ncar", "perlmutter")
    assert result["credit_home"] == str(env["home"] / ".credit")
    assert result["repo_root"] == REPO
    assert result["schedulers"]["pbs"] is True
    assert os.path.realpath(env["home"]) in result["allowed_roots"]
    assert call("nope", check_ok=False)["error"]["code"] == "bad_request"
    assert call("hello", raw=b"[1, 2]", check_ok=False)["error"]["code"] == "bad_request"
    assert call("hello", raw=b"{not json", check_ok=False)["error"]["code"] == "bad_request"
    assert call("hello", raw=b"")["protocol"] == 1  # empty stdin means {}
    assert call("render", {}, check_ok=False)["error"]["code"] == "unsupported"


def test_stdout_noise_goes_to_stderr(env):
    code = "import sys; from credit.remote import main; print('noise from bashrc'); sys.exit(main(['hello']))"
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code], input=b"{}", capture_output=True, env=os.environ.copy(), timeout=60
    )
    # noise printed before credit.remote starts (.bashrc) precedes the sentinel line clients look for
    assert proc.stdout.startswith(b"noise from bashrc\n" + SENTINEL) and proc.stdout.endswith(b"}\n")
    code2 = "import sys; from credit.remote import _Out; o = _Out(); print('late noise', flush=True); o.send({'x': 1})"
    proc = subprocess.run([sys.executable, "-P", "-c", code2], capture_output=True, env=os.environ.copy(), timeout=60)
    assert proc.stdout == SENTINEL + b'{"x":1}\n' and b"late noise" in proc.stderr


LIGHT_VERBS = {
    "hello": {},
    "sync": {"ledger_offset": 0},
    "ls": None,
    "read": None,
    "plan": None,
    "examples": {},
    "docs": {},
}


@pytest.mark.parametrize("verb", sorted(LIGHT_VERBS))
def test_light_verbs_do_not_import_heavy_modules(env, verb):
    cfg = make_config(env)
    req = LIGHT_VERBS[verb]
    if verb == "ls":
        req = {"path": str(cfg.parent)}
    elif verb == "read":
        req = {"path": str(cfg)}
    elif verb == "plan":
        req = {"kind": "rollout", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {}}
    code = (
        "import sys, json; from credit.remote import main; rc = main([sys.argv[1]]); "
        "bad = sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'numpy', 'xarray', 'pandas'}); "
        "sys.stderr.write('HEAVY=' + json.dumps(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code, verb],
        input=json.dumps(req).encode(),
        capture_output=True,
        env=os.environ.copy(),
        timeout=60,
    )
    assert b'"ok":true' in proc.stdout, proc.stdout
    assert proc.stderr.decode().rsplit("HEAVY=", 1)[1] == "[]"


# ---------------------------------------------------------------------------
# read / ls / write and path scope
# ---------------------------------------------------------------------------


def test_read_text_binary_and_ranges(env):
    f = env["home"] / "notes.txt"
    f.write_text("hello world\n")
    r = call("read", {"path": str(f)})
    assert r["data"] == "hello world\n" and r["encoding"] == "utf-8" and r["eof"]
    assert r["sha256"] == runs.sha256_file(str(f)) and r["size"] == 12
    r = call("read", {"path": str(f), "offset": 6, "length": 3})
    assert r["data"] == "wor" and not r["eof"]
    b = env["home"] / "blob.bin"
    b.write_bytes(b"\x00\x01\x02")
    r = call("read", {"path": str(b)})
    assert r["encoding"] == "base64" and r["data"] == "AAEC"
    assert call("read", {"path": str(env["home"] / "nope")}, check_ok=False)["error"]["code"] == "not_found"


def test_ls(env):
    d = env["home"] / "dir"
    (d / "sub").mkdir(parents=True)
    for name in ("a.yml", "b.yml", "c.txt", ".hidden.yml"):
        (d / name).write_text("x")
    r = call("ls", {"path": str(d)})
    assert [(e["name"], e["type"]) for e in r["entries"]] == [
        ("a.yml", "file"),
        ("b.yml", "file"),
        ("c.txt", "file"),
        ("sub", "dir"),
    ]
    r = call("ls", {"path": str(d), "pattern": "*.yml", "limit": 1})
    assert [e["name"] for e in r["entries"]] == ["a.yml"] and r["truncated"]
    assert call("ls", {"path": str(d), "pattern": "sub/*"}, check_ok=False)["error"]["code"] == "bad_request"


def test_path_scope_denials(env):
    outside = env["tmp"] / "outside"
    outside.mkdir()
    (outside / "secret.yml").write_text("s: 1\n")
    (env["home"] / "escape").symlink_to(outside)
    (env["home"] / ".ssh").mkdir()
    (env["home"] / ".ssh" / "config").write_text("Host x\n")
    denied = [
        ("read", {"path": str(outside / "secret.yml")}),
        ("read", {"path": str(env["home"] / ".." / "outside" / "secret.yml")}),
        ("read", {"path": str(env["home"] / "escape" / "secret.yml")}),
        ("read", {"path": str(env["home"] / ".ssh" / "config")}),
        ("ls", {"path": str(env["home"] / "escape")}),
        ("ls", {"path": "/"}),
        ("write", {"path": str(env["home"] / "escape" / "new.yml"), "text": "x", "if_sha256": None}),
        ("write", {"path": str(env["home"] / "x.json"), "text": "{}", "if_sha256": None}),
        ("write", {"path": str(env["home"] / ".credit" / "x.yml"), "text": "x", "if_sha256": None}),
    ]
    for verb, req in denied:
        resp = call(verb, req, check_ok=False)
        assert resp["error"]["code"] == "forbidden_path", (verb, req, resp)
    assert not (outside / "new.yml").exists()


def test_run_dir_readable_but_not_writable(env):
    run_id, run_dir, _ = make_run(env)
    r = call("read", {"path": os.path.join(run_dir, "run.json")})
    assert json.loads(r["data"])["run_id"] == run_id
    assert call("ls", {"path": os.path.join(run_dir, "events")})["entries"][0]["name"] == "submit.jsonl"
    resp = call("write", {"path": os.path.join(run_dir, "config.yml"), "text": "x", "if_sha256": None}, check_ok=False)
    assert resp["error"]["code"] == "forbidden_path"


def test_write_conflicts_and_atomic_update(env):
    path = env["home"] / "cfg" / "new.yml"
    r = call("write", {"path": str(path), "text": "a: 1\n", "if_sha256": None})
    assert path.read_text() == "a: 1\n" and r["sha256"] == runs.sha256_file(str(path))
    assert (
        call("write", {"path": str(path), "text": "a: 2\n", "if_sha256": None}, check_ok=False)["error"]["code"]
        == "conflict"
    )
    resp = call("write", {"path": str(path), "text": "a: 2\n", "if_sha256": "0" * 64}, check_ok=False)
    assert resp["error"]["code"] == "conflict" and path.read_text() == "a: 1\n"
    r2 = call("write", {"path": str(path), "text": "a: 2\n", "if_sha256": r["sha256"]})
    assert path.read_text() == "a: 2\n" and r2["sha256"] != r["sha256"]
    resp = call("write", {"path": str(env["home"] / "gone.yml"), "text": "x", "if_sha256": r["sha256"]}, check_ok=False)
    assert resp["error"]["code"] == "conflict"


# ---------------------------------------------------------------------------
# Delta readers: logs, csv, events
# ---------------------------------------------------------------------------


def test_read_log_collapses_tqdm_frames(tmp_path):
    log = tmp_path / "job.o11"
    log.write_bytes(
        b"start\r\nEpoch 0:  10%|#\rEpoch 0:  20%|##\rEpoch 0: 100%|####\nnext line\nEpoch 1:   5%|\rEpoch 1:  7%|"
    )
    r = _delta.read_log(str(log), 0, 65536)
    assert r["text"] == "start\nEpoch 0: 100%|####\nnext line\n"
    assert r["current_frame"] == "Epoch 1:  7%|"
    # the offset stops after the last \r, so the live frame is re-read next time
    assert r["off"] == log.read_bytes().rfind(b"\r") + 1
    with open(log, "ab") as f:
        f.write(b"\rEpoch 1: 100%|####\ndone\n")
    r2 = _delta.read_log(str(log), r["off"], 65536)
    assert r2["text"] == "Epoch 1: 100%|####\ndone\n" and r2["current_frame"] is None


def test_read_log_cap_skips_ahead_and_reset(tmp_path):
    log = tmp_path / "job.o11"
    log.write_bytes(b"".join(b"line %04d\n" % i for i in range(100)))
    r = _delta.read_log(str(log), 0, 25)
    assert r["truncated"] and r["text"] == "line 0098\nline 0099\n"
    assert r["off"] == log.stat().st_size
    log.write_bytes(b"new\n")
    r = _delta.read_log(str(log), 1000, 65536)
    assert r["reset"] and r["text"] == "new\n"


def test_read_log_holds_back_a_trailing_cr(tmp_path):
    log = tmp_path / "job.o11"
    log.write_bytes(b"a\r")
    r = _delta.read_log(str(log), 0, 65536)
    assert r["off"] == 0 and r["text"] == ""
    with open(log, "ab") as f:
        f.write(b"\n")
    assert _delta.read_log(str(log), 0, 65536)["text"] == "a\n"


def test_csv_delta_settle_rows_and_reset(tmp_path):
    p = tmp_path / "training_log.csv"
    p.write_text("index,epoch,train_loss\n0,0,1.5\n1,1,1.2\n")
    now = time.time()
    assert _delta.read_csv_delta(str(p), None, now=p.stat().st_mtime + 1) == (None, None, True)  # too fresh
    d, cur, pending = _delta.read_csv_delta(str(p), None, now=now + 10)
    assert not pending
    assert d == {
        "columns": ["index", "epoch", "train_loss"],
        "rows": [[0.0, 0.0, 1.5], [1.0, 1.0, 1.2]],
        "first_row": 0,
        "reset": False,
    }
    p.write_text("index,epoch,train_loss\n0,0,1.5\n1,1,1.2\n2,2,nan\n3,3,")  # unterminated
    assert _delta.read_csv_delta(str(p), cur, now=now + 10) == (None, cur, True)
    p.write_text("index,epoch,train_loss\n0,0,1.5\n1,1,1.2\n2,2,nan\n")
    d, cur, _ = _delta.read_csv_delta(str(p), cur, now=now + 20)
    assert d["first_row"] == 2 and not d["reset"]
    assert json.loads(runs.encode_line(d))["rows"] == [[2.0, 2.0, "NaN"]]  # NaN is encoded as a string
    p.write_text("index,epoch,train_loss\n0,0,9.9\n")  # a new run reused save_loc
    d, cur, _ = _delta.read_csv_delta(str(p), cur, now=now + 30)
    assert d["reset"] and d["first_row"] == 0 and d["rows"] == [[0.0, 0.0, 9.9]]
    p.write_text("index,epoch,valid_loss\n0,0,9.9\n")  # header changed
    d, _, _ = _delta.read_csv_delta(str(p), cur, now=now + 40)
    assert d["reset"]


def test_sync_cursor_resume_no_duplicates_no_gaps(env):
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    proc = os.path.join(run_dir, "events", "11.proc.jsonl")
    with open(proc, "wb") as f:
        for i in range(5):
            f.write(ev_line("train.progress", batch=i))
        f.write(b'{"v":1,"ts":"2026')  # partial last line
    open(env["tmp"] / "qstat.json", "w").write('{"Jobs": {}}')
    r1 = call("sync", {"ledger_offset": 0, "progress": "all"})
    assert [ln["run_id"] for ln in r1["ledger"]] == [run_id]
    d1 = r1["runs"][run_id]
    assert d1["run"]["run_id"] == run_id and not d1["missing"]
    batches = [e["data"]["batch"] for e in d1["events"] if e["type"] == "train.progress"]
    assert batches == [0, 1, 2, 3, 4]
    assert all("file" in e and "off" in e for e in d1["events"])
    assert d1["cursor"]["events"]["11.proc.jsonl"] == os.path.getsize(proc) - len(b'{"v":1,"ts":"2026')
    # finish the partial line (it was not consumed), then add more
    with open(proc, "r+b") as f:
        f.truncate(d1["cursor"]["events"]["11.proc.jsonl"])
        f.seek(0, 2)
        for i in range(5, 8):
            f.write(ev_line("train.progress", batch=i))
    r2 = call("sync", {"ledger_offset": r1["ledger_offset"], "runs": {run_id: d1["cursor"]}})
    assert r2["ledger"] == []
    d2 = r2["runs"][run_id]
    assert d2["run"] is None  # run.json unchanged
    assert [e["data"]["batch"] for e in d2["events"]] == [5, 6, 7]
    r3 = call("sync", {"ledger_offset": r2["ledger_offset"], "runs": {run_id: d2["cursor"]}})
    assert r3["runs"][run_id]["events"] == []


def test_sync_progress_modes_and_max_bytes(env):
    run_id, run_dir, _ = make_run(env, jobs=["11.desched1"])
    open(env["tmp"] / "qstat.json", "w").write('{"Jobs": {}}')
    proc = os.path.join(run_dir, "events", "11.proc.jsonl")
    with open(proc, "wb") as f:
        for i in range(50):
            f.write(ev_line("train.progress", batch=i))
            f.write(ev_line("train.vars", values=[i]))
        f.write(ev_line("checkpoint.saved", path="/x", epoch=0, kind="latest"))
    latest = call("sync", {"runs": {run_id: None}, "progress": "latest", "scheduler": False})["runs"][run_id]
    types = [e["type"] for e in latest["events"] if e["file"] == "11.proc.jsonl"]
    assert sorted(types) == ["checkpoint.saved", "train.progress", "train.vars"]
    assert [e["data"]["batch"] for e in latest["events"] if e["type"] == "train.progress"] == [49]
    assert latest["cursor"]["events"]["11.proc.jsonl"] == os.path.getsize(proc)
    none = call("sync", {"runs": {run_id: None}, "progress": "none", "scheduler": False})["runs"][run_id]
    assert [e["type"] for e in none["events"] if e["file"] == "11.proc.jsonl"] == ["checkpoint.saved"]
    # all + a small byte budget pages through without gaps or duplicates
    seen, cursor, pages = [], None, 0
    while True:
        d = call("sync", {"runs": {run_id: cursor}, "max_bytes": 3000, "scheduler": False})["runs"][run_id]
        seen += [(e["file"], e["off"]) for e in d["events"]]
        cursor, pages = d["cursor"], pages + 1
        if not d["more"]:
            break
    assert pages > 3
    proc_offs = [off for f, off in seen if f == "11.proc.jsonl"]
    assert len(proc_offs) == 101 and len(set(proc_offs)) == 101


def test_sync_logs_csv_and_missing(env):
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    (save / "logs" / "credit_gen2.o11").write_bytes(b"hello\nEpoch 0: 1%\rEpoch 0: 2%")
    csv_path = save / "training_log.csv"
    csv_path.write_text("index,epoch,train_loss\n0,0,1.5\n")
    os.utime(csv_path, (time.time() - 60, time.time() - 60))
    open(env["tmp"] / "qstat.json", "w").write('{"Jobs": {}}')
    r = call("sync", {"runs": {run_id: None, "train-20990101T000000-dead": None}, "log_tail_bytes": 1000})
    d = r["runs"][run_id]
    assert d["logs"]["11"]["text"] == "hello\n" and d["logs"]["11"]["current_frame"] == "Epoch 0: 2%"
    assert d["csv"]["rows"] == [[0.0, 0.0, 1.5]]
    assert r["runs"]["train-20990101T000000-dead"]["missing"]
    cursor = d["cursor"]
    assert cursor["logs"]["11"]["off"] == len(b"hello\nEpoch 0: 1%\r")
    # nothing new: no logs, no csv
    d2 = call("sync", {"runs": {run_id: cursor}, "scheduler": False})["runs"][run_id]
    assert d2["logs"] == {} and d2["csv"] is None and d2["events"] == []


# ---------------------------------------------------------------------------
# Scheduler queries (fixtures hand-written from the PBS Pro / SLURM docs)
# ---------------------------------------------------------------------------

# unverified against Derecho
QSTAT_JSON = {
    "timestamp": 1793400000,
    "pbs_version": "2022.1.3",
    "pbs_server": "desched1",
    "Jobs": {
        "1001.desched1": {
            "Job_Name": "credit_gen2",
            "job_state": "R",
            "exec_host": "dec0123/0*64+dec0124/0*64",
            "Resource_List": {"walltime": "12:00:00", "select": "2:ncpus=64:ngpus=4"},
            "resources_used": {"walltime": "01:02:03"},
            "comment": "Job run at Thu Oct 30 at 09:00 on (dec0123:ncpus=64)",
        },
        "1002.desched1": {
            "job_state": "H",
            "Hold_Types": "s",
            "Resource_List": {"walltime": "12:00:00"},
            "estimated": {"start_time": "Fri Oct 31 09:00:00 2026"},
        },
        "1000.desched1": {"job_state": "F", "Exit_status": 271, "Resource_List": {"walltime": "12:00:00"}},
    },
}

# unverified against Derecho
QSTAT_TEXT = """\
Job Id: 1001.desched1
    Job_Name = credit_gen2
    job_state = R
    exec_host = dec0123/0*64
    Resource_List.walltime = 12:00:00
    resources_used.walltime = 00:10:00
    comment = Job run at Thu Oct 30 at 09:00 on (dec0123:ncpus=64)
\t and more

"""


def test_query_pbs_batches_per_server_and_maps_states(env):
    (env["tmp"] / "qstat.json").write_text(json.dumps(QSTAT_JSON))
    ids = ["1000.desched1", "1001.desched1", "1002.desched1", "1003.desched1", "9.casper-pbs"]
    states, errors = _sched.query_pbs(ids)
    calls = env["calls"].read_text().splitlines()
    assert calls == [
        "qstat -x -f -F json 1000.desched1 1001.desched1 1002.desched1 1003.desched1",
        "qstat -x -f -F json 9.casper-pbs",
    ]
    assert states["1001.desched1"]["state"] == "running"
    assert states["1001.desched1"]["exec_host"] == "dec0123"
    assert states["1001.desched1"]["walltime_used"] == "01:02:03"
    assert states["1001.desched1"]["walltime_limit"] == "12:00:00"
    assert states["1002.desched1"]["state"] == "held" and states["1002.desched1"]["hold_types"] == "s"
    assert states["1002.desched1"]["estimated_start"].endswith("Z")
    assert states["1000.desched1"] == {**states["1000.desched1"], "state": "finished", "exit_status": 271}
    assert states["1003.desched1"]["state"] == "gone"
    assert states["9.casper-pbs"]["state"] == "gone"  # same fixture, no such job there
    assert errors == {}


def test_query_pbs_text_fallback():
    jobs = _sched.parse_qstat_text(QSTAT_TEXT)
    attrs = jobs["1001.desched1"]
    assert attrs["Resource_List"]["walltime"] == "12:00:00"
    assert attrs["comment"].endswith("and more")
    st = _sched._pbs_job_state(attrs)
    assert st["state"] == "running" and st["walltime_used"] == "00:10:00"


def test_query_pbs_reports_errors(env):
    _write_exe(env["tmp"] / "bin" / "qstat", "#!/bin/bash\necho 'qstat: cannot connect to server' >&2\nexit 1\n")
    states, errors = _sched.query_pbs(["1.desched1"])
    assert states == {} and "cannot connect" in errors["desched1"]


def test_query_slurm(env):
    # unverified against Perlmutter
    sacct = {
        "jobs": [
            {
                "job_id": 77,
                "state": {"current": ["RUNNING"], "reason": "None"},
                "nodes": "nid001",
                "time": {"elapsed": 3723, "limit": {"set": True, "number": 720}},
                "exit_code": {"return_code": {"set": True, "number": 0}},
            },
            {"job_id": 78, "state": {"current": ["COMPLETED"]}, "exit_code": {"return_code": {"number": 0}}},
        ]
    }
    (env["tmp"] / "sacct.json").write_text(json.dumps(sacct))
    states, errors = _sched.query_slurm(["77", "78", "79"])
    assert states["77"]["state"] == "running" and states["77"]["walltime_used"] == "01:02:03"
    assert states["77"]["walltime_limit"] == "12:00:00" and states["77"]["exec_host"] == "nid001"
    assert states["78"]["state"] == "finished" and states["78"]["exit_status"] == 0
    assert states["79"]["state"] == "gone" and errors == {}
    assert env["calls"].read_text().splitlines() == ["sacct --json -j 77,78,79"]


def test_sync_skips_finished_jobs_and_queries_once(env):
    (env["tmp"] / "qstat.json").write_text(json.dumps(QSTAT_JSON))
    run_id, run_dir, _ = make_run(env, jobs=["1000.desched1", "1001.desched1", "1002.desched1"])
    runs.append_line(
        os.path.join(run_dir, "events", "1000.script.jsonl"),
        runs.make_event("job.exited", {"code": 0}, "script", job="1000"),
    )
    r = call("sync", {"runs": {run_id: None}})
    assert set(r["scheduler"]) == {"1001.desched1", "1002.desched1"}
    assert env["calls"].read_text().splitlines() == ["qstat -x -f -F json 1001.desched1 1002.desched1"]


# ---------------------------------------------------------------------------
# plan / submit / cancel with fake PBS and SLURM
# ---------------------------------------------------------------------------


def test_pbs_plan_is_pure_and_submit_chains_on_command_line(env):
    cfg = make_config(env)
    save = env["home"] / "exp" / "save"
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"gpus": 4}}
    plan = call("plan", req)["plan"]
    assert not save.exists()  # plan is pure
    assert len(plan["jobs"]) == 2 and plan["jobs"][1]["depends_on_index"] == 1
    run_dir = plan["run_dir"]
    assert plan["jobs"][0]["config"] == os.path.join(run_dir, "config.yml")
    assert plan["jobs"][1]["config"] == os.path.join(run_dir, "config_reload.yml")
    for job in plan["jobs"]:
        script = job["script"]
        assert "depend=afterok" not in script
        assert f"export CREDIT_RUN_DIR={run_dir}" in script and "export CREDIT_JOB_KEY=${PBS_JOBID%%.*}" in script
        assert f"#PBS -o {save}/logs" in script and "job.started" in script and "trap _credit_on_term TERM" in script
        assert script.index("#PBS -q") < script.index("export CREDIT_RUN_ID")  # directives stay first
        assert subprocess.run(["bash", "-n"], input=script.encode()).returncode == 0
    assert plan["target"]["server"] == "desched1" and plan["target"]["queue"] == "main@desched1"
    assert plan["gpu_hours_estimate"] == 12 * 4 * 2
    assert plan["log_dir"] == str(save / "logs") and plan["request"]["run_id"] == plan["run_id"]
    assert plan["cli"].startswith("credit submit --cluster derecho -c ") and "--chain 2" in plan["cli"]
    # deterministic: same request + run_id gives the same scripts
    again = call("plan", plan["request"])["plan"]
    assert again["scripts_sha256"] == plan["scripts_sha256"]

    out = call("submit", submit_req(plan))
    assert out["errors"] == [] and [j["job_id"] for j in out["jobs"]] == ["1001.desched1", "1002.desched1"]
    assert env["calls"].read_text().splitlines() == [
        f"qsub -N {plan['run_id']}-1 {run_dir}/scripts/job-1.sh",
        f"qsub -N {plan['run_id']}-2 -W depend=afterok:1001.desched1 {run_dir}/scripts/job-2.sh",
    ]
    record = runs.read_run(run_dir)
    assert [j["depends_on"] for j in record["jobs"]] == [None, "1001.desched1"]
    assert record["epochs_total"] == 4 and record["epochs_per_job"] == 2 and record["created_by"] == "mcp"
    assert os.path.exists(os.path.join(run_dir, "config_reload.yml"))
    assert yaml.safe_load(open(os.path.join(run_dir, "config_reload.yml")))["trainer"]["load_weights"] is True
    assert not os.path.exists(save / "config_reload.yml")
    assert (save / "logs").is_dir()
    lines = runs.read_lines(os.path.join(run_dir, "events", "submit.jsonl"))[0]
    assert [ev["type"] for _, _, ev in lines] == [
        "run.created",
        "job.submitting",
        "job.submitted",
        "job.submitting",
        "job.submitted",
    ]
    assert runs.find_run(plan["run_id"])["created_by"] == "mcp"

    # idempotent
    again = call("submit", submit_req(plan))
    assert again["already_submitted"] and len(again["jobs"]) == 2
    assert len(env["calls"].read_text().splitlines()) == 2


def test_submit_conflicts(env):
    cfg = make_config(env)
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 1}}
    plan = call("plan", req)["plan"]
    bad = submit_req(plan)
    bad["scripts_sha256"] = ["0" * 64]
    assert call("submit", bad, check_ok=False)["error"]["code"] == "conflict"
    cfg.write_text(cfg.read_text() + "# edited\n")
    assert call("submit", submit_req(plan), check_ok=False)["error"]["code"] == "conflict"
    assert not os.path.exists(plan["run_dir"])


def test_plan_config_errors(env):
    cfg = make_config(env)
    no_pbs = yaml.safe_load(cfg.read_text())
    no_pbs.pop("pbs")
    cfg.write_text(yaml.safe_dump(no_pbs))
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {}}
    assert call("plan", req, check_ok=False)["error"]["code"] == "config_error"
    # options alone are enough
    req["options"] = {"conda_env": "/opt/envs/credit", "account": "X"}
    assert call("plan", req)["plan"]["target"]["account"] == "X"
    req["options"]["queue"] = "casper"  # a Casper queue on Derecho
    assert call("plan", req, check_ok=False)["error"]["code"] == "config_error"
    rt = {
        "kind": "realtime",
        "config": str(cfg),
        "scheduler": "pbs",
        "cluster": "casper",
        "options": {"conda_env": "e"},
    }
    assert call("plan", rt, check_ok=False)["error"]["code"] == "config_error"


def test_plan_every_kind_builds_valid_scripts(env):
    cfg = make_config(env)
    for scheduler, cluster in (("pbs", "casper"), ("pbs", "derecho"), ("slurm", "perlmutter")):
        for kind, opts in (
            ("preprocess", {}),
            ("rollout", {"jobs": 2}),
            ("realtime", {"init_time": "2024-01-15T00", "steps": 4}),
            ("train", {"nodes": 2, "launcher": "pbsdsh"} if cluster == "derecho" else {}),
        ):
            opts = {**opts, "queue": "casper"} if cluster == "casper" else opts
            req = {"kind": kind, "config": str(cfg), "scheduler": scheduler, "cluster": cluster, "options": opts}
            plan = call("plan", req)["plan"]
            for job in plan["jobs"]:
                assert "export CREDIT_RUN_ID=" in job["script"], (scheduler, cluster, kind)
                assert subprocess.run(["bash", "-n"], input=job["script"].encode()).returncode == 0
            if kind == "rollout":
                assert len(plan["jobs"]) == 2 and plan["warnings"]


def test_slurm_submit_chain_dependency(env):
    cfg = make_config(env)
    req = {"kind": "train", "config": str(cfg), "scheduler": "slurm", "cluster": "perlmutter", "options": {}}
    plan = call("plan", req)["plan"]
    assert "--dependency" not in plan["jobs"][1]["script"]
    assert "export CREDIT_JOB_KEY=${SLURM_JOB_ID}" in plan["jobs"][0]["script"]
    out = call("submit", submit_req(plan))
    calls = env["calls"].read_text().splitlines()
    run_id = plan["run_id"]
    assert (
        calls[1]
        == f"sbatch --parsable --job-name={run_id}-2 --dependency=afterok:77 {plan['run_dir']}/scripts/job-2.sh"
    )
    assert out["jobs"][0]["job_key"] == "77"


def test_submit_partial_failure_keeps_earlier_jobs(env):
    cfg = make_config(env)
    _write_exe(
        env["tmp"] / "bin" / "qsub",
        f"""\
        #!/bin/bash
        echo "qsub $*" >> {env["calls"]}
        if [[ "$*" == *depend* ]]; then echo "qsub: Job violates queue limits" >&2; exit 1; fi
        echo 1001.desched1
        """,
    )
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 3}}
    plan = call("plan", req)["plan"]
    out = call("submit", submit_req(plan))
    assert [j["job_id"] for j in out["jobs"]] == ["1001.desched1"]
    assert len(out["errors"]) == 2 and "queue limits" in out["errors"][0]
    assert len(runs.read_run(plan["run_dir"])["jobs"]) == 1


def test_cancel_pbs(env):
    (env["tmp"] / "qstat.json").write_text(json.dumps(QSTAT_JSON))
    run_id, run_dir, _ = make_run(env, jobs=["1001.desched1", "1002.desched1"])
    out = call("cancel", {"run_id": run_id, "scope": "pending"})
    assert out == {"cancelled": ["1002.desched1"], "errors": {}}
    assert "qdel 1002.desched1" in env["calls"].read_text()
    record = runs.read_run(run_dir)
    assert record["cancelled"]["scope"] == "pending" and record["cancelled"]["job_ids"] == ["1002.desched1"]
    out = call("cancel", {"run_id": run_id, "scope": "all"})
    assert set(out["cancelled"]) == {"1001.desched1", "1002.desched1"}
    assert call("cancel", {"run_id": "train-20990101T000000-dead"}, check_ok=False)["error"]["code"] == "not_found"


def test_resume_from_counts_remaining_epochs(env):
    run_id, run_dir, save = make_run(env)
    (save / "training_log.csv").write_text("index,epoch,train_loss\n0,0,1.0\n")
    req = {"kind": "train", "scheduler": "pbs", "cluster": "derecho", "options": {"resume_from": run_id}}
    plan = call("plan", req)["plan"]
    assert plan["config"] == os.path.join(run_dir, "config.yml") and plan["parent_run_id"] == run_id
    assert len(plan["jobs"]) == 2  # ceil((4 - 1) / 2)
    assert all(j["config"].endswith("config_reload.yml") for j in plan["jobs"])
    out = call("submit", submit_req(plan))
    assert runs.read_run(out["run_dir"])["parent_run_id"] == run_id
    assert runs.find_run(out["run_id"])["parent_run_id"] == run_id


# ---------------------------------------------------------------------------
# The local scheduler, end to end
# ---------------------------------------------------------------------------

FAKE_TRAINER = """\
import os, sys, time
from credit.events import EventSink

sink = EventSink(conf={"events": {"progress_every_s": 0}})
sink.process_started(1)
results = {"train_loss": []}
for i in range(5):
    results["train_loss"].append(1.0 / (i + 1))
    sys.stdout.write(f"\\rEpoch 0: {i + 1}/5 loss {results['train_loss'][-1]:.3f}")
    sys.stdout.flush()
    sink.train_progress(0, results, i + 1, 5, lr=0.1)
sys.stdout.write("\\n")
print("args", " ".join(sys.argv[1:3]))
sink.emit("train.stopped", {"reason": "completed", "epoch": 0})
deadline = time.time() + float(os.environ.get("FAKE_SLEEP", "0"))
while time.time() < deadline:
    time.sleep(0.05)
"""


def _local_plan(env, monkeypatch, sleep="0"):
    fake = env["tmp"] / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER)
    monkeypatch.setenv("CREDIT_REMOTE_LOCAL_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("FAKE_SLEEP", sleep)
    cfg = make_config(env)
    req = {"kind": "train", "config": str(cfg), "scheduler": "local", "options": {"chain": 1}}
    return call("plan", req)["plan"]


def test_local_lifecycle(env, monkeypatch):
    plan = _local_plan(env, monkeypatch)
    assert "export CREDIT_JOB_KEY=local-$$" in plan["jobs"][0]["script"]
    assert plan["target"]["scheduler"] == "local"
    out = call("submit", submit_req(plan))
    (job,) = out["jobs"]
    assert job["job_id"].startswith("local-")
    run_id, run_dir = out["run_id"], out["run_dir"]

    def finished():
        return _delta.job_finished(run_dir, job["job_key"])

    assert wait_for(finished)["code"] == 0
    r = call("sync", {"ledger_offset": 0, "progress": "all"})
    d = r["runs"][run_id]
    types = [e["type"] for e in d["events"]]
    for t in ("run.created", "job.submitted", "job.started", "process.started", "train.progress", "job.exited"):
        assert t in types, (t, types)
    assert len([t for t in types if t == "train.progress"]) == 5
    log = d["logs"][job["job_key"]]
    assert "Epoch 0: 5/5 loss 0.200\n" in log["text"] and "Epoch 0: 1/5" not in log["text"]
    assert "args train -c\n" in log["text"]
    assert job["job_id"] not in r["scheduler"]  # final per the markers
    # backfill without progress lines
    latest = call("sync", {"runs": {run_id: None}, "progress": "none", "scheduler": False})["runs"][run_id]
    assert "train.progress" not in [e["type"] for e in latest["events"]]
    hist = call("history", {"run_id": run_id, "fields": ["loss.mean", "lr"], "buckets": 3})
    assert hist["points"] == 5 and len(hist["t"]) <= 3 and hist["fields"]["lr"]["max"][0] == 0.1
    logs = call("logs", {"run_id": run_id, "grep": "^args"})
    assert logs["text"] == "args train -c\n" and logs["job_key"] == job["job_key"]
    status = subprocess.run(
        [sys.executable, "-m", "credit.cli", "status", "--json"], capture_output=True, env=os.environ.copy(), timeout=60
    )
    (summary,) = json.loads(status.stdout)
    assert summary["run_id"] == run_id and summary["state"] == "completed"


def test_local_cancel_writes_signal_marker(env, monkeypatch):
    plan = _local_plan(env, monkeypatch, sleep="60")
    out = call("submit", submit_req(plan))
    (job,) = out["jobs"]
    run_dir = out["run_dir"]
    wait_for(lambda: os.path.exists(os.path.join(run_dir, "events", f"{job['job_key']}.proc.jsonl")))
    states, _ = _sched.query([("local", job["job_id"], run_dir)])
    assert states[job["job_id"]]["state"] == "running"
    res = call("cancel", {"run_id": out["run_id"], "scope": "all"})
    assert res["cancelled"] == [job["job_id"]]
    done = wait_for(lambda: _delta.job_finished(run_dir, job["job_key"]))
    assert done["signal"] == "TERM"
    wait_for(lambda: (_delta.job_finished(run_dir, job["job_key"]) or {}).get("code") is not None)
    pid = int(job["job_id"].split("-")[1])
    wait_for(lambda: not _sched._pid_alive(pid))
    types = [ev["type"] for _, _, ev in runs.read_lines(os.path.join(run_dir, "events", "submit.jsonl"))[0]]
    assert types[-1] == "run.cancelled"


# ---------------------------------------------------------------------------
# follow
# ---------------------------------------------------------------------------


class _Follower:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "credit.remote", "follow"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        self.frames = []
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith(SENTINEL):
                with self.lock:
                    self.frames.append(json.loads(line[len(SENTINEL) :]))

    def send(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def wait(self, predicate, timeout=20):
        def find():
            with self.lock:
                return next((f for f in self.frames if predicate(f)), None)

        return wait_for(find, timeout=timeout, interval=0.05)


def test_follow_frames_add_remove_and_eof(env):
    run_a, dir_a, _ = make_run(env, jobs=["11.desched1"])
    run_b, dir_b, _ = make_run(env, save=env["home"] / "exp" / "save_b", jobs=["12.desched1"])
    proc_a = os.path.join(dir_a, "events", "11.proc.jsonl")
    f = _Follower()
    try:
        assert f.wait(lambda fr: fr["type"] == "hello")["protocol"] == 1
        f.send({"runs": {run_a: None}, "interval_s": 0.2, "log_tail_bytes": 4096})
        first = f.wait(lambda fr: fr["type"] == "frame" and run_a in fr["runs"])
        delta = first["runs"][run_a]
        assert delta["run"]["run_id"] == run_a and "scheduler" not in first
        with open(proc_a, "ab") as fh:
            fh.write(ev_line("train.progress", batch=1))
        fr = f.wait(
            lambda fr: (
                fr["type"] == "frame"
                and any(e["type"] == "train.progress" for e in fr["runs"].get(run_a, {}).get("events", []))
            )
        )
        assert fr["runs"][run_a]["run"] is None  # run.json did not change
        f.send({"add": {run_b: None, "train-20990101T000000-dead": None}})
        err = f.wait(lambda fr: fr["type"] == "error")
        assert err["run_id"] == "train-20990101T000000-dead" and err["error"]["code"] == "not_found"
        f.wait(lambda fr: fr["type"] == "frame" and run_b in fr["runs"])
        f.send({"remove": [run_a]})
        time.sleep(0.6)
        with f.lock:
            n = len(f.frames)
        with open(proc_a, "ab") as fh:
            fh.write(ev_line("train.progress", batch=2))
        time.sleep(0.8)
        with f.lock:
            assert not any(run_a in fr.get("runs", {}) for fr in f.frames[n:])
        f.proc.stdin.close()
        assert f.proc.wait(timeout=10) == 0
    finally:
        if f.proc.poll() is None:
            f.proc.kill()


# ---------------------------------------------------------------------------
# adopt / diagnose / outputs / examples / docs
# ---------------------------------------------------------------------------


def test_adopt(env):
    save = env["home"] / "old_run"
    save.mkdir()
    (save / "model.yml").write_text("save_loc: x\ntrainer: {epochs: 10, num_epoch: 5}\n")
    out = call("adopt", {"save_loc": str(save), "job_ids": ["555.desched1"], "scheduler": "pbs"})
    record = runs.read_run(out["run_dir"])
    assert record["jobs"][0]["job_key"] == "555" and record["epochs_total"] == 10
    assert record["target"]["server"] == "desched1"
    outside = env["tmp"] / "elsewhere"
    outside.mkdir()
    resp = call("adopt", {"save_loc": str(outside), "config": str(save / "model.yml")}, check_ok=False)
    assert resp["error"]["code"] == "forbidden_path"


def test_diagnose_and_outputs(env):
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    (save / "logs" / "credit_gen2.o11").write_text(
        "INFO starting\n"
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in <module>\n'
        "    main()\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n"
        "after\n"
    )
    runs.append_line(
        os.path.join(run_dir, "events", "11.script.jsonl"),
        runs.make_event("job.exited", {"code": 1}, "script", job="11"),
    )
    d = call("diagnose", {"run_id": run_id})
    assert d["exit_status"] == 1 and d["job_key"] == "11"
    assert d["traceback"].startswith("Traceback") and d["traceback"].endswith("Tried to allocate 2 GiB")
    assert any("CUDA out of memory" in b for b in d["banners"])
    (save / "checkpoint.pt").write_bytes(b"x")
    out = call("outputs", {"run_id": run_id})
    assert [os.path.basename(c["path"]) for c in out["checkpoints"]] == ["checkpoint.pt"]
    assert out["forecast_dir"] is None and out["files"] == []


def test_examples_and_docs(env):
    names = call("examples")["examples"]
    assert "example-end-to-end" in names
    assert "save_loc" in call("examples", {"name": "example-end-to-end"})["text"]
    pages = call("docs")["pages"]
    assert "cli" in pages and "remote" in pages
    assert "credit status" in call("docs", {"page": "cli"})["text"]
    assert call("docs", {"page": "../../README"}, check_ok=False)["error"]["code"] == "not_found"


def test_pbsdsh_jobs_read_rank0_node_log(env):
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    (save / "logs" / "credit_gen2.o11").write_text("mother superior\n")
    d1 = call("sync", {"runs": {run_id: None}, "scheduler": False})["runs"][run_id]
    assert d1["logs"]["11"]["text"] == "mother superior\n"
    node0 = save / "logs" / "pbsdsh_logs" / "11" / "node_0.out"
    node0.parent.mkdir(parents=True)
    node0.write_text("rank 0 says hi\n")
    d2 = call("sync", {"runs": {run_id: d1["cursor"]}, "scheduler": False})["runs"][run_id]
    assert d2["logs"]["11"]["path"] == str(node0) and d2["logs"]["11"]["text"] == "rank 0 says hi\n"
    assert d2["logs"]["11"]["reset"]  # A5.7: a new log identity


# ---------------------------------------------------------------------------
# Fix round 1 (CONTRACTS A5)
# ---------------------------------------------------------------------------

SECRET = "gho_SECRETTOKEN123"


def test_adopt_refuses_hidden_and_non_yaml_configs(env):
    exp = env["home"] / "exp"
    exp.mkdir()
    gh = env["home"] / ".config" / "gh"
    gh.mkdir(parents=True)
    (gh / "hosts.yml").write_text(f"github.com:\n  oauth_token: {SECRET}\n")
    resp = call("adopt", {"save_loc": str(exp), "config": str(gh / "hosts.yml")}, check_ok=False)
    assert resp["error"]["code"] == "forbidden_path" and SECRET not in json.dumps(resp)
    (exp / "notes.txt").write_text(f"token {SECRET}\n")
    resp = call("adopt", {"save_loc": str(exp), "config": str(exp / "notes.txt")}, check_ok=False)
    assert resp["error"]["code"] == "config_error" and SECRET not in json.dumps(resp)
    (exp / "bad.yml").write_text(f"a: [{SECRET}\n")
    resp = call("adopt", {"save_loc": str(exp), "config": str(exp / "bad.yml")}, check_ok=False)
    assert (
        resp["error"]["code"] == "config_error"
        and SECRET not in json.dumps(resp)
        and "line" in resp["error"]["message"]
    )
    (exp / "list.yml").write_text("- 1\n")
    resp = call("adopt", {"save_loc": str(exp), "config": str(exp / "list.yml")}, check_ok=False)
    assert resp["error"]["code"] == "config_error"
    assert not (exp / ".credit").exists()


@pytest.mark.parametrize("job_id", ["-x", "123; rm -rf ~", "123 456", "abc.desched1", "1.$(id)"])
def test_adopt_validates_job_ids(env, job_id):
    cfg = make_config(env)
    resp = call("adopt", {"save_loc": str(cfg.parent), "config": str(cfg), "job_ids": [job_id]}, check_ok=False)
    assert resp["error"]["code"] == "bad_request"


def test_adopt_accepts_array_and_plain_ids(env):
    cfg = make_config(env)
    out = call("adopt", {"save_loc": str(cfg.parent), "config": str(cfg), "job_ids": ["12[].desched1", "77"]})
    assert [j["job_key"] for j in runs.read_run(out["run_dir"])["jobs"]] == ["12[]", "77"]


def test_plan_scope_and_yaml_errors_do_not_leak(env):
    outside = env["tmp"] / "outside"
    outside.mkdir()
    (outside / "c.yml").write_text("save_loc: /x\n")
    ssh = env["home"] / ".ssh"
    ssh.mkdir()
    (ssh / "config.yml").write_text(f"Host x\n  IdentityFile {SECRET}\n")
    base = {"kind": "train", "scheduler": "pbs", "cluster": "derecho", "options": {}}
    for path in (outside / "c.yml", ssh / "config.yml", env["home"] / "exp" / ".." / ".." / "outside" / "c.yml"):
        resp = call("plan", {**base, "config": str(path)}, check_ok=False)
        assert resp["error"]["code"] == "forbidden_path", (path, resp)
    bad = env["home"] / "bad.yml"
    bad.write_text(f"Host x\n  IdentityFile: {SECRET}\n")
    resp = call("plan", {**base, "config": str(bad)}, check_ok=False)
    assert resp["error"]["code"] == "config_error" and SECRET not in json.dumps(resp)
    assert "line 2" in resp["error"]["message"]
    resp = call("plan", {**base, "config": str(env["home"] / "notes.txt")}, check_ok=False)
    assert resp["error"]["code"] == "config_error"
    # whitespace in the config path or save_loc
    spaced = env["home"] / "my exp"
    spaced.mkdir()
    (spaced / "c.yml").write_text(yaml.safe_dump({"save_loc": str(env["home"] / "s"), "pbs": {"conda": "e"}}))
    assert call("plan", {**base, "config": str(spaced / "c.yml")}, check_ok=False)["error"]["code"] == "config_error"
    cfg = make_config(env, save_loc=str(env["home"] / "has space"))
    assert call("plan", {**base, "config": str(cfg)}, check_ok=False)["error"]["code"] == "config_error"


def test_resume_with_nothing_left_is_a_config_error(env):
    run_id, _, save = make_run(env)
    (save / "training_log.csv").write_text("index,epoch,train_loss\n" + "".join(f"{i},{i},1.0\n" for i in range(4)))
    req = {"kind": "train", "scheduler": "pbs", "cluster": "derecho", "options": {"resume_from": run_id}}
    resp = call("plan", req, check_ok=False)
    assert resp["error"]["code"] == "config_error" and "nothing left to train" in resp["error"]["message"]


def test_outputs_scope(env):
    run_id, run_dir, save = make_run(env)
    outside = env["tmp"] / "forecasts"
    outside.mkdir()
    (outside / "f.nc").write_text("x")
    conf = yaml.safe_load(open(os.path.join(run_dir, "config.yml")))
    conf["inference"] = {"save_forecast": str(outside)}
    with open(os.path.join(run_dir, "config.yml"), "w") as f:
        yaml.safe_dump(conf, f)
    out = call("outputs", {"run_id": run_id})
    assert out["files"] == []  # named, not listed
    # a ledger run whose save_loc is outside every root
    rogue = env["tmp"] / "rogue"
    rid = runs.new_run_id("train")
    runs.append_ledger({"run_id": rid, "run_dir": str(save / ".credit" / "runs" / run_id), "save_loc": str(rogue)})
    record = runs.read_run(run_dir)
    record["save_loc"] = str(rogue)
    runs.write_run(run_dir, record)
    assert call("outputs", {"run_id": rid}, check_ok=False)["error"]["code"] == "forbidden_path"


def test_read_and_grep_caps(env):
    big = env["home"] / "big.txt"
    big.write_bytes(b"x" * (9 << 20))
    r = call("read", {"path": str(big), "length": 10 << 20})
    assert len(r["data"]) == 8 << 20 and not r["eof"]
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    with open(save / "logs" / "credit_gen2.o11", "wb") as f:
        f.write(b"needle early\n")
        f.write((b"y" * 99 + b"\n") * 50_000)  # 5 MB
        f.write(b"needle late\n")
    out = call("logs", {"run_id": run_id, "grep": "needle", "tail_bytes": 64 << 20})
    assert out["text"] == "needle late\n"


def test_sync_reports_csv_pending(env):
    run_id, run_dir, save = make_run(env)
    csv_path = save / "training_log.csv"
    csv_path.write_text("index,epoch,train_loss\n0,0,1.5\n")
    d = call("sync", {"runs": {run_id: None}, "scheduler": False})["runs"][run_id]
    assert d["csv"] is None and d["csv_pending"]
    os.utime(csv_path, (time.time() - 10, time.time() - 10))
    d2 = call("sync", {"runs": {run_id: d["cursor"]}, "scheduler": False})["runs"][run_id]
    assert d2["csv"]["rows"] == [[0.0, 0.0, 1.5]] and not d2["csv_pending"]


def test_submit_retry_resubmits_refused_jobs_and_reports_partial(env):
    cfg = make_config(env)
    qsub = env["tmp"] / "bin" / "qsub"
    _write_exe(
        qsub,
        f"""\
        #!/bin/bash
        echo "qsub $*" >> {env["calls"]}
        if [[ "$*" == *depend* && ! -e {env["tmp"]}/allow ]]; then echo "qsub: queue limits" >&2; exit 1; fi
        echo "qsub: warning: account is nearly out of hours"
        n=$(cat {env["tmp"]}/qsub.n 2>/dev/null || echo 1000); n=$((n + 1)); echo $n > {env["tmp"]}/qsub.n
        echo "$n.desched1"
        """,
    )
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 3}}
    plan = call("plan", req)["plan"]
    first = call("submit", submit_req(plan))
    assert [j["job_id"] for j in first["jobs"]] == ["1001.desched1"]  # id parsed past the warning line
    assert "queue limits" in first["errors"][0] and "depends on" in first["errors"][1]
    (env["tmp"] / "allow").touch()
    second = call("submit", submit_req(plan))
    assert second["already_submitted"] and second["errors"] == []
    assert [(j["index"], j["job_id"]) for j in second["jobs"]] == [
        (1, "1001.desched1"),
        (2, "1002.desched1"),
        (3, "1003.desched1"),
    ]
    calls = env["calls"].read_text().splitlines()
    assert calls[-1].endswith(f"-W depend=afterok:1002.desched1 {plan['run_dir']}/scripts/job-3.sh")
    assert sum("job-1.sh" in c for c in calls) == 1  # job 1 never resubmitted
    # an intent with no outcome (a crash, or a qsub timeout) is reported, never blindly resubmitted
    plan2 = call("plan", {**req, "run_id": None, "options": {"chain": 2}})["plan"]
    run_dir2 = plan2["run_dir"]
    os.makedirs(os.path.join(run_dir2, "events"))
    os.makedirs(os.path.join(run_dir2, "scripts"))
    runs.append_event(run_dir2, "submit.jsonl", runs.make_event("job.submitting", {"index": 1}, "submit"))
    n_calls = len(env["calls"].read_text().splitlines())
    third = call("submit", submit_req(plan2))
    assert third["already_submitted"] and third["jobs"] == []
    partial = [e for e in third["errors"] if isinstance(e, dict)]
    assert partial == [
        {
            "code": "partial_submit",
            "index": 1,
            "job_name": f"{plan2['run_id']}-1",
            "message": partial[0]["message"],
        }
    ]
    assert len(env["calls"].read_text().splitlines()) == n_calls  # nothing submitted
    assert runs.find_run(plan2["run_id"])  # the interrupted run is in the ledger now


def test_local_cancel_and_state_check_process_identity(env):
    stranger = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        cfg = make_config(env)
        save = cfg.parent / "save"
        run_id = runs.new_run_id("train")
        record = runs.create_run(
            run_id, "train", str(cfg), str(save), {"scheduler": "local", "log_dir": str(save / "logs")}
        )
        run_dir = runs.run_dir_for(str(save), run_id)
        an_hour_ago = runs.now_ts(time.time() - 3600)
        runs.add_job(run_dir, record, {"index": 1, "job_id": f"local-{stranger.pid}", "submitted": an_hour_ago})
        states, _ = _sched.query([("local", f"local-{stranger.pid}", run_dir)])
        assert states[f"local-{stranger.pid}"]["state"] == "gone"
        assert call("cancel", {"run_id": run_id, "scope": "all"}) == {"cancelled": [], "errors": {}}
        with pytest.raises(runs.SchedulerError, match="not signalling"):
            _sched.cancel_job("local", f"local-{stranger.pid}", run_dir)
        assert stranger.poll() is None  # still alive
    finally:
        stranger.kill()


TERM_TRAINER = """\
import os, signal, sys, time
def on_term(*_):
    print("cleaning up", flush=True)
    time.sleep(1.0)
    open(os.environ["CLEANED"], "w").write("done")
    sys.exit(7)
signal.signal(signal.SIGTERM, on_term)
open(os.environ["READY"], "w").write("ok")
while True:
    time.sleep(0.05)
"""


@pytest.mark.parametrize("target", ["parent", "group"])
def test_term_trap_waits_for_the_job_and_records_its_exit_code(env, monkeypatch, target):
    import signal

    fake = env["tmp"] / "term_trainer.py"
    fake.write_text(TERM_TRAINER)
    monkeypatch.setenv("CREDIT_REMOTE_LOCAL_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("READY", str(env["tmp"] / "ready"))
    monkeypatch.setenv("CLEANED", str(env["tmp"] / "cleaned"))
    cfg = make_config(env)
    plan = call("plan", {"kind": "train", "config": str(cfg), "scheduler": "local", "options": {"chain": 1}})["plan"]
    out = call("submit", submit_req(plan))
    (job,) = out["jobs"]
    pid = int(job["job_id"].split("-")[1])
    wait_for(lambda: (env["tmp"] / "ready").exists())
    if target == "parent":
        os.kill(pid, signal.SIGTERM)
    else:
        os.killpg(pid, signal.SIGTERM)
    marker_file = os.path.join(out["run_dir"], "events", f"{job['job_key']}.script.jsonl")

    def exited():
        return [ev for _, _, ev in runs.read_lines(marker_file)[0] if ev["type"] == "job.exited"]

    wait_for(exited)
    assert (env["tmp"] / "cleaned").exists()  # the job finished cleaning up before job.exited
    types = [ev["type"] for _, _, ev in runs.read_lines(marker_file)[0]]
    assert types == ["job.started", "job.signaled", "job.exited"]
    assert exited()[0]["data"]["code"] == 7
    wait_for(lambda: not _sched._pid_alive(pid))


# ---------------------------------------------------------------------------
# Heavy verbs (need torch; skipped without it)
# ---------------------------------------------------------------------------


@pytest.fixture
def heavy_env(env):
    pytest.importorskip("torch")
    return env


def test_check_verb_on_a_real_example(heavy_env):
    import shutil

    cfg = heavy_env["home"] / "example-v2026.2.yml"
    shutil.copy(os.path.join(REPO, "config", "gen_2", "examples", "example-v2026.2.yml"), cfg)
    r = call("check", {"config": str(cfg)})
    report = r["report"]
    assert report["config"] == str(cfg) and isinstance(report["findings"], list) and report["findings"]
    assert {"errors", "warnings"} <= set(report) and isinstance(r["derived"], dict)
    assert {"severity", "where", "message", "fix"} == set(report["findings"][0])
    r2 = call("check", {"yaml": cfg.read_text()})
    assert r2["report"]["config"] == "<yaml>" and r2["report"]["findings"]
    resp = call("check", {"config": str(heavy_env["home"] / "notes.txt")}, check_ok=False)
    assert resp["error"]["code"] in ("config_error", "not_found")
    assert (
        call("check", {"config": str(heavy_env["tmp"] / "x.yml")}, check_ok=False)["error"]["code"] == "forbidden_path"
    )


def test_components_verb(heavy_env):
    kinds = call("components")["kinds"]
    assert set(kinds) == {"models", "trainers", "datasets", "preblocks", "postblocks", "losses", "metrics"}
    assert "wxformer" in kinds["models"] and "concat" in kinds["preblocks"]
    only = call("components", {"kind": "preblocks"})["kinds"]
    assert list(only) == ["preblocks"]
    d = call("components", {"kind": "preblocks", "key": "concat"})
    assert d["kind"] == "preblocks" and d["key"] == "concat" and d["class"] == "ConcatToTensor"
    assert d["module"] == "credit.preblock.concat" and isinstance(d["params"], list)
    assert all({"name", "default", "annotation", "required"} == set(p) for p in d["params"])
    assert call("components", {"kind": "models", "key": "nope"}, check_ok=False)["error"]["code"] == "not_found"
    assert call("components", {"kind": "bogus"}, check_ok=False)["error"]["code"] == "bad_request"


def test_build_config_verb(heavy_env):
    preset = {
        "dataset_type": "weatherbench2_era5",
        "level_coord": "level",
        "levels": [500, 700, 850],
        "vars_3D": ["temperature"],
        "vars_2D": ["surface_pressure"],
        "static": ["land_sea_mask"],
        "timestep": "6h",
        "start": "1959-01-01",
        "end": "2022-12-31",
        "height": 32,
        "width": 64,
        "resolution": "5.625deg",
        "tisr": [[87.1875, -87.1875, 32], [0, 354.375, 64]],
    }
    save = heavy_env["home"] / "exp" / "save"
    state = {
        "system": "local",
        "experiment": "test_exp",
        "save_loc": str(save),
        "seed": 3141,
        "dataset": preset["dataset_type"],
        "preset": preset,
        "start": preset["start"],
        "end": preset["end"],
        "timestep": "6h",
        "valid_start": "2021-01-01",
        "valid_end": "2021-12-31",
        "vars_3D": preset["vars_3D"],
        "vars_2D": preset["vars_2D"],
        "batch_size": 1,
        "batches_per_epoch": 1,
        "parallelism_data": "none",
        "nodes": 1,
        "gpus": 1,
        "pbs": None,
    }
    out = heavy_env["home"] / "exp" / "built.yml"
    r = call("build_config", {"state": state, "path": str(out), "write": False})
    assert yaml.safe_load(r["yaml"])["save_loc"] == str(save) and r["written"] is False and not out.exists()
    r = call("build_config", {"state": state, "path": str(out), "write": True})
    assert r["written"] and out.read_text() == r["yaml"] and r["path"] == os.path.realpath(out)
    assert (
        call("build_config", {"state": state, "path": str(out), "write": True}, check_ok=False)["error"]["code"]
        == "conflict"
    )
    assert call("build_config", {"state": {}}, check_ok=False)["error"]["code"] in ("config_error", "bad_request")


def test_docs_examples_and_unsupported_verbs(heavy_env):
    pages = call("docs")["pages"]
    assert "cli" in pages and "remote" in pages
    page = call("docs", {"page": "remote"})
    assert page["page"] == "remote" and "credit.remote" in page["text"]
    names = call("examples")["examples"]
    assert "example-v2026.2" in names
    ex = call("examples", {"name": "example-v2026.2"})
    assert ex["name"] == "example-v2026.2" and "save_loc" in ex["text"]
    for verb in ("convert", "render"):
        assert call(verb, {}, check_ok=False)["error"]["code"] == "unsupported"


def test_internal_errors_are_bounded(env):
    code = (
        "import sys; import credit.remote as r\n"
        "def boom(req):\n"
        "    def deep(n):\n"
        "        if n == 0: raise ValueError('x' * 5000)\n"
        "        deep(n - 1)\n"
        "    deep(20)\n"
        "r.VERBS['hello'] = boom\n"
        "sys.exit(r.main(['hello']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-P", "-c", code], input=b"{}", capture_output=True, env=os.environ.copy(), timeout=60
    )
    resp = json.loads(proc.stdout.split(SENTINEL)[-1])
    msg = resp["error"]["message"]
    assert resp["error"]["code"] == "internal" and msg.startswith("ValueError: xxx")
    assert msg.count('File "') == 3 and len(msg) < 2000


# ---------------------------------------------------------------------------
# Final review: concurrent submits (N1) and run.json lagging submit.jsonl (N2)
# ---------------------------------------------------------------------------


def test_concurrent_submits_queue_each_job_once(env):
    cfg = make_config(env)
    _write_exe(
        env["tmp"] / "bin" / "qsub",
        f"""\
        #!/bin/bash
        sleep 0.5
        echo "qsub $*" >> {env["calls"]}
        echo "$RANDOM$RANDOM.desched1"
        """,
    )
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 3}}
    plan = call("plan", req)["plan"]
    body = json.dumps(submit_req(plan)).encode()
    procs = [
        subprocess.Popen(
            [sys.executable, "-P", "-m", "credit.remote", "submit"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        for _ in range(2)
    ]
    outs = [json.loads(p.communicate(body, timeout=60)[0].split(SENTINEL)[-1]) for p in procs]
    assert len(env["calls"].read_text().splitlines()) == 3  # one qsub per job, not six
    winners = [o for o in outs if o["ok"] and not o["result"].get("already_submitted")]
    assert len(winners) == 1 and len(winners[0]["result"]["jobs"]) == 3
    loser = [o for o in outs if o is not winners[0]][0]
    assert (
        not loser["ok"] and loser["error"]["code"] == "conflict" and "in progress" in loser["error"]["message"]
    ) or (loser["ok"] and loser["result"]["already_submitted"])


def test_jobs_survive_a_crash_between_submitted_and_run_json(env, monkeypatch):
    import importlib

    submit_mod = importlib.import_module("credit.cli._submit")
    cfg = make_config(env)
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 3}}
    plan = submit_mod._plan(req)
    args = (plan["request"], plan["scripts_sha256"], plan["config_sha256"])
    real = runs.write_run

    def crash_once(run_dir, record):
        if record.get("jobs"):
            monkeypatch.setattr(runs, "write_run", real)
            raise KeyboardInterrupt("simulated crash after job.submitted, before run.json")
        return real(run_dir, record)

    monkeypatch.setattr(runs, "write_run", crash_once)
    with pytest.raises(KeyboardInterrupt):
        submit_mod._submit_plan(*args)
    run_dir, run_id = plan["run_dir"], plan["run_id"]
    assert runs.read_run(run_dir)["jobs"] == []  # run.json lags submit.jsonl
    assert [j["job_id"] for j in runs.jobs_of(run_dir, runs.read_run(run_dir))] == ["1001.desched1"]

    # sync, status and cancel all see the job
    (env["tmp"] / "qstat.json").write_text(json.dumps({"Jobs": {"1001.desched1": {"job_state": "Q"}}}))
    r = call("sync", {"runs": {run_id: None}})
    assert [j["job_id"] for j in r["runs"][run_id]["run"]["jobs"]] == ["1001.desched1"]
    assert r["scheduler"]["1001.desched1"]["state"] == "queued"
    status = subprocess.run(
        [sys.executable, "-m", "credit.cli", "status", run_id, "--json"],
        capture_output=True,
        env=os.environ.copy(),
        timeout=60,
    )
    summary = json.loads(status.stdout)
    assert [j["job_id"] for j in summary["jobs"]] == ["1001.desched1"] and summary["state"] == "queued"
    assert call("cancel", {"run_id": run_id, "scope": "all"})["cancelled"] == ["1001.desched1"]
    assert "qdel 1001.desched1" in env["calls"].read_text()
    assert [j["job_id"] for j in runs.read_run(run_dir)["jobs"]] == ["1001.desched1"]  # healed by cancel

    # a retry heals run.json and submits only the remaining jobs
    retry = submit_mod._submit_plan(*args)
    assert retry["already_submitted"] and [j["index"] for j in retry["jobs"]] == [1, 2, 3]
    assert [j["job_id"] for j in runs.read_run(run_dir)["jobs"]] == ["1001.desched1", "1002.desched1", "1003.desched1"]
    assert sum(" qsub " in f" {c}" and "job-1.sh" in c for c in env["calls"].read_text().splitlines()) == 1
