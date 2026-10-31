"""`credit remote-setup` and the no-policy default of credit.remote."""

import importlib
import os
import stat
import subprocess
import sys

import pytest
import yaml

from credit import runs
from tests.test_remote import REPO, call, env  # noqa: F401 — `env` is a fixture

# credit.cli re-exports the `_remote_setup` handler function under the module's own name.
setup = importlib.import_module("credit.cli._remote_setup")


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "credit.cli", "remote-setup", *argv],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=REPO,
        timeout=120,
    )


def test_root_problem(env):  # noqa: F811
    home = env["home"]
    (home / "CREDIT_runs").mkdir()
    (home / ".hidden").mkdir()
    assert setup.root_problem(str(home / "CREDIT_runs")) is None
    assert "too broad" in setup.root_problem(str(home))
    assert "too broad" in setup.root_problem(str(home.parent))
    assert "too broad" in setup.root_problem("/")
    assert "hidden" in setup.root_problem(str(home / ".hidden"))
    assert "not an existing" in setup.root_problem(str(home / "nope"))


def test_cli_writes_a_private_policy_that_remote_honours(env):  # noqa: F811
    runs_dir = env["home"] / "CREDIT_runs"
    runs_dir.mkdir()
    (env["home"] / "notes.txt").write_text("private\n")
    policy = env["home"] / ".credit" / "remote.yaml"  # $CREDIT_HOME/remote.yaml, which credit.remote reads
    r = run_cli("--root", str(runs_dir), "--yes", "--policy-file", str(policy))
    assert r.returncode == 0, r.stderr
    assert yaml.safe_load(policy.read_text()) == {"roots": [str(runs_dir.resolve())], "redact": True, "audit": True}
    assert stat.S_IMODE(policy.stat().st_mode) == 0o600
    assert stat.S_IMODE(policy.parent.stat().st_mode) == 0o700
    hello = call("hello")
    assert hello["policy"]["configured"] and hello["allowed_roots"] == [str(runs_dir.resolve())]
    resp = call("read", {"path": str(env["home"] / "notes.txt")}, check_ok=False)
    assert resp["error"]["code"] == "forbidden_path"


def test_cli_refuses_broad_roots(env):  # noqa: F811
    policy = env["tmp"] / "p" / "remote.yaml"
    r = run_cli("--root", str(env["home"]), "--yes", "--policy-file", str(policy))
    assert r.returncode == 1 and "too broad" in r.stderr
    assert not policy.exists()


def test_suggestions_come_from_the_ledger(env, monkeypatch):  # noqa: F811
    exp = env["home"] / "CREDIT_runs" / "exp"
    exp.mkdir(parents=True)
    runs.append_ledger({"run_id": "train-a", "run_dir": str(exp / ".credit/runs/train-a"), "save_loc": str(exp)})
    runs.append_ledger(
        {"run_id": "train-b", "run_dir": str(env["home"] / ".credit/runs/b"), "save_loc": str(env["home"])}
    )
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    # the CREDIT_runs parent is offered; the home dir (parent of a home save_loc) is not
    assert setup.suggest_roots() == [str((env["home"] / "CREDIT_runs").resolve())]


def test_no_policy_error_explains_setup(env, monkeypatch):  # noqa: F811
    monkeypatch.delenv("CREDIT_REMOTE_ALLOW")
    (env["home"] / "notes.txt").write_text("private\n")
    resp = call("read", {"path": str(env["home"] / "notes.txt")}, check_ok=False)
    assert resp["error"]["code"] == "forbidden_path" and "credit remote-setup" in resp["error"]["fix"]
    hello = call("hello")
    assert hello["policy"]["configured"] is False and hello["allowed_roots"] == []


@pytest.mark.parametrize("flag", ["--no-redact", "--no-audit"])
def test_flags_turn_features_off(env, flag):  # noqa: F811
    d = env["home"] / "CREDIT_runs"
    d.mkdir()
    policy = env["tmp"] / "q" / "remote.yaml"
    assert run_cli("--root", str(d), "--yes", flag, "--policy-file", str(policy)).returncode == 0
    data = yaml.safe_load(policy.read_text())
    assert data["redact"] is (flag != "--no-redact") and data["audit"] is (flag != "--no-audit")
