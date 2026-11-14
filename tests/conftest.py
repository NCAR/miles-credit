"""Shared pytest fixtures, and the ``credit.remote`` helpers used by the test_remote* modules."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import yaml

from credit import runs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENTINEL = b"\x1eCREDIT\x1e"


@pytest.fixture(autouse=True)
def _isolated_credit_home(tmp_path_factory, monkeypatch):
    """Point the run ledger (``$CREDIT_HOME``) at a temp dir so no test touches ``~/.credit``."""
    monkeypatch.setenv("CREDIT_HOME", str(tmp_path_factory.mktemp("credit_home")))


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
    for var in ("CREDIT_REMOTE_LOCAL_CMD", "CREDIT_RUN_DIR"):
        monkeypatch.delenv(var, raising=False)
    # Without a remote.yaml nothing is served by default, so the tests name the test HOME explicitly.
    monkeypatch.setenv("CREDIT_REMOTE_ALLOW", str(home))
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
