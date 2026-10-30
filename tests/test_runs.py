"""Tests for credit.runs: run ids, the ledger, run.json, line readers, path scope, and state derivation."""

import json
import os
import re
import stat

import pytest

from credit import runs


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake $HOME as the only static root, with the ledger under it."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("CREDIT_HOME", str(h / ".credit"))
    for var in ("SCRATCH", "WORK", "PSCRATCH", "CREDIT_REMOTE_ALLOW", "USER", "LOGNAME"):
        monkeypatch.delenv(var, raising=False)
    return h


def test_run_id_and_timestamp_format():
    run_id = runs.new_run_id("train", t=0)
    assert re.fullmatch(r"train-19700101T000000-[0-9a-f]{4}", run_id)
    assert runs.now_ts(1.2345) == "1970-01-01T00:00:01.234Z"
    assert runs.now_ts(1.9996) == "1970-01-01T00:00:02.000Z"
    assert runs.parse_ts("1970-01-01T00:00:01.250Z") == pytest.approx(1.25)
    with pytest.raises(ValueError):
        runs.new_run_id("plot")


def test_job_key():
    assert runs.job_key("5875448.desched1") == "5875448"
    assert runs.job_key("123") == "123"
    assert runs.job_key("local-42") == "local-42"


def test_jsonable_encodes_nonfinite_as_strings():
    out = runs.jsonable({"a": float("nan"), "b": [float("inf"), -float("inf"), 1.5], "c": (1, 2)})
    assert out == {"a": "NaN", "b": ["Infinity", "-Infinity", 1.5], "c": [1, 2]}
    assert json.loads(runs.encode_line({"x": float("nan")})) == {"x": "NaN"}


def test_read_lines_offsets_and_partial_line(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_bytes(b'{"a":1}\n{"a":2}\n{"a":')
    lines, off, reset = runs.read_lines(str(p), 0)
    assert [(o, obj["a"]) for o, _, obj in lines] == [(0, 1), (8, 2)]
    assert off == 16 and not reset
    # the partial line is not consumed; completing it yields exactly it
    with open(p, "ab") as f:
        f.write(b'3}\nnot json\n{"a":4}\n')
    lines, off2, _ = runs.read_lines(str(p), off)
    assert [obj["a"] for _, _, obj in lines] == [3, 4]  # the bad line is skipped
    assert lines[0][0] == 16
    assert off2 == p.stat().st_size
    # a file shorter than the offset is re-read from 0
    lines, _, reset = runs.read_lines(str(p), 10_000)
    assert reset and lines[0][2] == {"a": 1}
    # max_bytes stops after the line that crosses it
    lines, off3, _ = runs.read_lines(str(p), 0, max_bytes=9)
    assert [obj["a"] for _, _, obj in lines] == [1, 2] and off3 == 16


def test_ledger_append_read_and_remove(home):
    runs.append_ledger({"run_id": "train-1", "run_dir": "/x/1"})
    runs.append_ledger({"run_id": "train-2", "run_dir": "/x/2"})
    lines, off, reset = runs.read_ledger(0)
    assert [ln["run_id"] for ln in lines] == ["train-1", "train-2"] and all(ln["v"] == 1 for ln in lines)
    assert off == os.path.getsize(runs.ledger_path()) and not reset
    runs.append_ledger({"run_id": "train-1", "removed": True})
    new, off2, _ = runs.read_ledger(off)
    assert new == [{"v": 1, "run_id": "train-1", "removed": True}]
    assert set(runs.ledger_index()) == {"train-2"}
    _, _, reset = runs.read_ledger(off2 + 100)
    assert reset
    assert os.path.exists(runs.ledger_path() + ".lock")


def test_atomic_write_keeps_mode_and_symlink(tmp_path):
    target = tmp_path / "real.yml"
    target.write_text("a: 1\n")
    os.chmod(target, 0o640)
    link = tmp_path / "link.yml"
    link.symlink_to(target)
    runs.atomic_write(str(link), "a: 2\n")
    assert link.is_symlink()
    assert target.read_text() == "a: 2\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]


def test_create_run_add_job_and_exclusive(home):
    cfg = home / "exp" / "config.yml"
    cfg.parent.mkdir()
    cfg.write_text("save_loc: x\n")
    save = home / "exp" / "save"
    run_id = runs.new_run_id("train")
    record = runs.create_run(run_id, "train", str(cfg), str(save), {"scheduler": "pbs"}, created_by="mcp")
    run_dir = runs.run_dir_for(str(save), run_id)
    assert (save / ".credit" / "runs" / run_id / "config.yml").read_text() == "save_loc: x\n"
    assert record["config_sha256"] == runs.sha256_file(str(cfg))
    runs.add_job(run_dir, record, {"index": 1, "job_id": "11.desched1"})
    on_disk = runs.read_run(run_dir)
    assert on_disk["jobs"][0]["job_key"] == "11"
    types = [ev["type"] for _, _, ev in runs.read_lines(os.path.join(run_dir, "events", "submit.jsonl"))[0]]
    assert types == ["run.created", "job.submitted"]
    assert runs.find_run(run_id)["run_dir"] == run_dir
    with pytest.raises(FileExistsError):
        runs.create_run(run_id, "train", str(cfg), str(save), {}, run_dir=run_dir)


# ---------------------------------------------------------------------------
# Path scope (CONTRACTS A1.1)
# ---------------------------------------------------------------------------


def test_scope_static_roots_and_escapes(home, tmp_path):
    inside = home / "data" / "f.txt"
    inside.parent.mkdir()
    inside.write_text("x")
    outside = tmp_path / "outside" / "secret.txt"
    outside.parent.mkdir()
    outside.write_text("s")
    assert runs.resolve_in_scope(str(inside)) == os.path.realpath(inside)
    with pytest.raises(runs.PathScopeError):
        runs.resolve_in_scope(str(outside))
    with pytest.raises(runs.PathScopeError):  # .. escape
        runs.resolve_in_scope(str(home / "data" / ".." / ".." / "outside" / "secret.txt"))
    (home / "link").symlink_to(outside.parent)
    with pytest.raises(runs.PathScopeError):  # symlink escape
        runs.resolve_in_scope(str(home / "link" / "secret.txt"))
    (home / ".ssh").mkdir()
    with pytest.raises(runs.PathScopeError, match="hidden"):
        runs.resolve_in_scope(str(home / ".ssh" / "id_rsa"))


def test_scope_extra_roots_and_ledger_roots(home, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("CREDIT_REMOTE_ALLOW", str(extra))
    assert runs.resolve_in_scope(str(extra / "a.yml"))
    # a ledger run outside every static root does not become a root
    rogue = tmp_path / "rogue"
    runs.append_ledger({"run_id": "train-x", "run_dir": str(rogue / ".credit/runs/train-x"), "save_loc": str(rogue)})
    assert str(os.path.realpath(rogue)) not in runs.allowed_roots()
    with pytest.raises(runs.PathScopeError):
        runs.resolve_in_scope(str(rogue / "f"))


def test_scope_allows_run_dirs_under_save_loc(home):
    save = home / "exp" / "save"
    run_dir = save / ".credit" / "runs" / "train-1"
    (run_dir / "events").mkdir(parents=True)
    runs.append_ledger({"run_id": "train-1", "run_dir": str(run_dir), "save_loc": str(save)})
    assert runs.resolve_in_scope(str(run_dir / "events" / "submit.jsonl"))
    with pytest.raises(runs.PathScopeError):
        runs.resolve_in_scope(str(save / ".credit" / "other"))


def test_check_writable(home):
    save = home / "exp" / "save"
    run_dir = save / ".credit" / "runs" / "train-1"
    run_dir.mkdir(parents=True)
    runs.append_ledger({"run_id": "train-1", "run_dir": str(run_dir), "save_loc": str(save)})
    runs.check_writable(os.path.realpath(home / "exp" / "new.yaml"))
    for bad in (home / "exp" / "x.json", run_dir / "config.yml", home / ".credit" / "a.yml", home / ".cfg" / "a.yml"):
        with pytest.raises(runs.PathScopeError):
            runs.check_writable(os.path.realpath(bad))


# ---------------------------------------------------------------------------
# Scheduler commands
# ---------------------------------------------------------------------------


def test_scheduler_bin_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_PBS_BIN", str(tmp_path))
    assert runs.scheduler_bin("qsub") == os.path.join(str(tmp_path), "qsub")
    monkeypatch.delenv("CREDIT_SLURM_BIN", raising=False)
    assert runs.scheduler_bin("sbatch", "slurm") == "sbatch"


def test_submit_job_puts_dependency_on_command_line(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls"
    qsub = bindir / "qsub"
    qsub.write_text(f'#!/bin/bash\necho "$@" >> {log}\necho 99.desched1\n')
    qsub.chmod(0o755)
    monkeypatch.setenv("CREDIT_PBS_BIN", str(bindir))
    assert runs.submit_job("pbs", "/s.sh", depend_on="98.desched1") == "99.desched1"
    assert log.read_text() == "-W depend=afterok:98.desched1 /s.sh\n"
    qsub.write_text("#!/bin/bash\necho 'qsub: illegal -A value' >&2\nexit 2\n")
    with pytest.raises(runs.SchedulerError, match="illegal"):
        runs.submit_job("pbs", "/s.sh")


# ---------------------------------------------------------------------------
# State derivation (CONTRACTS A1.8, A4.1, A5.2-A5.3), shared with the MCP core
# ---------------------------------------------------------------------------

STATE_CASES = os.path.join(os.path.dirname(__file__), "data", "state_cases.json")
with open(STATE_CASES) as _f:
    _CASES = json.load(_f)


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_derive_state_shared_cases(case):
    state = runs.derive_state(
        case["kind"],
        case["run"],
        case["events"],
        case["scheduler"],
        csv_last_epoch=case["csv_last_epoch"],
        has_child=case["has_child"],
    )
    assert state == case["expect"]


def test_derive_state_infers_files_from_src():
    events = [
        {"type": "job.started", "src": {"writer": "script", "job": "7"}, "data": {}},
        {"type": "train.stopped", "src": {"writer": "proc", "job": "7"}, "data": {"reason": "completed"}},
        {"type": "job.exited", "src": {"writer": "script", "job": "7"}, "data": {"code": 0}},
    ]
    run = {"jobs": [{"index": 1, "job_key": "7", "job_id": "7.s"}], "epochs_total": 3}
    assert runs.derive_state("train", run, events, {"7.s": {"state": "running"}}) == "completed"


def test_load_yaml_mapping_never_quotes_the_file(tmp_path):
    secret = tmp_path / "hosts.yml"
    secret.write_text("github.com:\n  oauth_token: gho_SECRET\n  user: [unclosed\n")
    with pytest.raises(runs.ConfigFileError) as exc:
        runs.load_yaml_mapping(str(secret))
    assert "gho_SECRET" not in str(exc.value) and "line" in str(exc.value)
    (tmp_path / "list.yml").write_text("- a\n")
    with pytest.raises(runs.ConfigFileError, match="mapping"):
        runs.load_yaml_mapping(str(tmp_path / "list.yml"))
    with pytest.raises(runs.ConfigFileError, match=".yml"):
        runs.load_yaml_mapping(str(tmp_path / "config"))


def test_parse_job_id():
    assert runs.parse_job_id("pbs", "warning: something\n5875448.desched1\n") == "5875448.desched1"
    assert runs.parse_job_id("pbs", "123[].casper-pbs\n") == "123[].casper-pbs"
    assert runs.parse_job_id("slurm", "77;perlmutter\n") == "77"
    assert runs.parse_job_id("slurm", "Submitted batch job 78\n") == "78"
    with pytest.raises(runs.SchedulerError):
        runs.parse_job_id("pbs", "qsub: would you like fries\n")
