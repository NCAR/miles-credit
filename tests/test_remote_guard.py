"""credit.remote hardening: the remote.yaml policy, secret redaction, the audit log, and GPU-hour budgets."""

import json
import os
import stat
import time

import pytest
import yaml

from credit import remote, runs
from credit.remote import _guard
from tests.conftest import _write_exe, call, make_config, make_run, submit_req

GH = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
ANTHROPIC = "sk-ant-" + "api03-abcdefghijklmnopqrstuvwxyz012345"


def test_redact_known_secrets():
    text = "\n".join(
        [
            f"github: {GH}",
            f"key {ANTHROPIC} in a log line",
            "aws_access_key_id = AKIAABCDEFGHIJKLMNOP",
            "url: https://alice:hunter2pass@data.example.org/x",
            "Authorization: Bearer abc.def.ghi",
            "wandb_api_key: 0123456789abcdef0123",
            "password: 'correct horse'",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----",
        ]
    )
    out, n = _guard.redact(text)
    for secret in (
        GH,
        ANTHROPIC,
        "AKIAABCDEFGHIJKLMNOP",
        "hunter2pass",
        "abc.def.ghi",
        "0123456789abcdef0123",
        "correct",
        "MIIBOgIBAAJBAK",
    ):
        assert secret not in out, secret
    assert n >= 8
    assert "https://alice:" + _guard.MARK + "@data.example.org" in out
    assert "wandb_api_key: " + _guard.MARK in out


def test_redact_leaves_ordinary_config_alone():
    text = (
        "save_loc: '/glade/work/$USER/CREDIT_runs/x'\n"
        "storage_options: {token: anon}\n"
        "max_tokens: 100000\n"
        "secret: null\n"
        "trainer:\n  train_batch_size: 8\n  lr: 1.0e-3\n"
        "Epoch: 1 train_loss: 0.000002 lr: 0.000100000000\n"
    )
    out, n = _guard.redact(text)
    assert out == text and n == 0


def test_read_logs_and_sync_are_redacted(env):
    run_id, run_dir, save = make_run(env, jobs=["11.desched1"])
    cfg = env["home"] / "exp" / "secrets.yml"
    cfg.write_text(f"save_loc: /x\nhf_token: {GH}\n")
    r = call("read", {"path": str(cfg)})
    assert GH not in r["data"] and r["redacted"] == 1
    (save / "logs" / "credit_gen2.o11").write_text(f"starting\nusing token {GH}\n")
    runs.append_event(
        run_dir,
        "11.proc.jsonl",
        runs.make_event("error", {"exc_type": "X", "message": f"bad key {ANTHROPIC}"}, "proc", job="11"),
    )
    open(env["tmp"] / "qstat.json", "w").write('{"Jobs": {}}')
    d = call("sync", {"runs": {run_id: None}})["runs"][run_id]
    assert GH not in d["logs"]["11"]["text"] and _guard.MARK in d["logs"]["11"]["text"]
    err = [e for e in d["events"] if e["type"] == "error"][0]
    assert ANTHROPIC not in err["data"]["message"]
    logs = call("logs", {"run_id": run_id})
    assert GH not in logs["text"]


def test_write_refuses_redacted_text(env):
    cfg = env["home"] / "exp.yml"
    cfg.write_text(f"hf_token: {GH}\n")
    r = call("read", {"path": str(cfg)})
    resp = call("write", {"path": str(cfg), "text": r["data"], "if_sha256": r["sha256"]}, check_ok=False)
    assert not resp["ok"] and resp["error"]["code"] == "config_error"
    assert GH in cfg.read_text()  # the secret survived


def test_policy_roots_narrow_scope(env):
    allowed = env["home"] / "CREDIT_runs"
    allowed.mkdir()
    (allowed / "a.yml").write_text("x: 1\n")
    (env["home"] / "notes.txt").write_text("private\n")
    assert call("read", {"path": str(env["home"] / "notes.txt")})["data"] == "private\n"
    credit_home = env["home"] / ".credit"
    credit_home.mkdir(exist_ok=True)
    (credit_home / "remote.yaml").write_text(yaml.safe_dump({"roots": [str(allowed)]}))
    os.environ["CREDIT_REMOTE_ALLOW"] = str(env["tmp"])  # the environment must not widen a policy
    try:
        resp = call("read", {"path": str(env["home"] / "notes.txt")}, check_ok=False)
        assert resp["error"]["code"] == "forbidden_path"
        assert call("read", {"path": str(allowed / "a.yml")})["data"] == "x: 1\n"
        assert call("hello")["allowed_roots"] == [os.path.realpath(allowed)]
    finally:
        del os.environ["CREDIT_REMOTE_ALLOW"]


def test_policy_can_turn_redaction_off(env):
    (env["home"] / ".credit").mkdir(exist_ok=True)
    (env["home"] / ".credit" / "remote.yaml").write_text("redact: false\n")
    cfg = env["home"] / "plain.yml"
    cfg.write_text(f"hf_token: {GH}\n")
    assert GH in call("read", {"path": str(cfg)})["data"]


def test_audit_log_records_calls_without_contents(env):
    cfg = env["home"] / "a.yml"
    cfg.write_text("secret_value_in_file: 1\n")
    call("read", {"path": str(cfg)})
    call("write", {"path": str(env["home"] / "b.yml"), "text": "y: 2\n"})
    call("read", {"path": "/etc/hosts"}, check_ok=False)
    path = env["home"] / ".credit" / "audit.jsonl"
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [ln["verb"] for ln in lines] == ["read", "write", "read"]
    assert lines[0]["ok"] and lines[0]["req"]["path"] == str(cfg)
    assert lines[2]["code"] == "forbidden_path"
    blob = path.read_text()
    assert "secret_value_in_file" not in blob and "y: 2" not in blob


# ---------------------------------------------------------------------------
# GPU-hour budgets (in-process, so the account's real home can be a temp dir)
# ---------------------------------------------------------------------------


@pytest.fixture
def owner(env, monkeypatch):
    """A temp dir standing in for the password-database home that the budget ledger lives in."""
    home = env["tmp"] / "owner"
    home.mkdir()
    monkeypatch.setattr(_guard, "owner_home", lambda: str(home))
    return home


def set_budgets(directory, **budgets):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "remote.yaml").write_text(yaml.safe_dump({"budgets": budgets}))


def budget_plan(env, chain=2, **options):
    """A Derecho train plan of *chain* jobs, each 12 h x 4 GPUs = 48 GPU-hours."""
    cfg = make_config(env)
    opts = {"gpus": 4, "chain": chain, **options}
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": opts}
    return remote.plan(req)["plan"]


def refused(req):
    with pytest.raises(remote.RemoteError) as exc:
        remote.submit(req)
    return exc.value


def qsub_calls(env):
    return env["calls"].read_text().splitlines() if env["calls"].exists() else []


def ledger(owner):
    path = owner / ".credit" / "budget.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def test_no_budgets_submit_unchanged(env, owner):
    plan = budget_plan(env)
    assert plan["gpu_hours_estimate"] == 96
    out = remote.submit(submit_req(plan))
    assert out["errors"] == [] and len(out["jobs"]) == 2
    assert not (owner / ".credit").exists()  # nothing is accounted without a budget
    pol = remote.hello({})["policy"]
    assert pol["budgets"] == {"max_gpu_hours_per_plan": None, "max_gpu_hours_per_day": None}
    assert pol["used_gpu_hours_24h"] == 0 and pol["budget_errors"] == []


def test_per_plan_limit_needs_a_typed_confirmation(env, owner):
    set_budgets(owner / ".credit", max_gpu_hours_per_plan=50)
    plan = budget_plan(env)
    err = refused(submit_req(plan))
    assert err.code == "over_budget" and "96" in err.message and "50" in err.message
    assert "a person has to confirm the GPU-hours" in err.fix
    assert err.data == {"kind": "per_plan", "estimate": 96, "limit": 50}
    assert refused({**submit_req(plan), "confirm_gpu_hours": 95}).code == "over_budget"
    assert refused({**submit_req(plan), "confirm_gpu_hours": "96"}).code == "bad_request"
    assert qsub_calls(env) == [] and not os.path.exists(plan["run_dir"])
    out = remote.submit({**submit_req(plan), "confirm_gpu_hours": 96})
    assert len(out["jobs"]) == 2
    assert [(ln["run_id"], ln["gpu_hours"]) for ln in ledger(owner)] == [(plan["run_id"], 96)]
    assert ledger(owner)[0]["ts"].endswith("Z")
    assert remote.hello({})["policy"]["used_gpu_hours_24h"] == 96


def test_daily_limit_is_rolling_and_has_no_override(env, owner):
    set_budgets(owner / ".credit", max_gpu_hours_per_day=100)
    _guard.record_gpu_hours("train-old", 30)
    with open(owner / ".credit" / "budget.jsonl", "a") as f:  # older than 24 h: no longer counts
        f.write(json.dumps({"ts": runs.now_ts(time.time() - 25 * 3600), "run_id": "x", "gpu_hours": 1000}) + "\n")
    assert _guard.used_gpu_hours() == 30
    plan = budget_plan(env)
    err = refused({**submit_req(plan), "confirm_gpu_hours": 1000})
    assert err.code == "over_budget" and "96" in err.message and "30" in err.message and "100" in err.message
    assert err.data == {"kind": "per_day", "estimate": 96, "limit": 100, "new_gpu_hours": 96, "used_gpu_hours_24h": 30}
    assert qsub_calls(env) == []
    small = budget_plan(env, chain=1)
    remote.submit(submit_req(small))
    assert _guard.used_gpu_hours() == 78
    assert refused(submit_req(budget_plan(env, chain=1))).code == "over_budget"  # 78 + 48 > 100


def test_resubmits_count_only_newly_submitted_jobs(env, owner):
    set_budgets(owner / ".credit", max_gpu_hours_per_day=200)
    plan = budget_plan(env)
    remote.submit(submit_req(plan))
    again = remote.submit(submit_req(plan))
    assert again["already_submitted"] and len(ledger(owner)) == 1 and _guard.used_gpu_hours() == 96
    # job 2 of a second run is refused by the scheduler: only job 1 is charged, and the retry charges job 2
    qsub = env["tmp"] / "bin" / "qsub"
    good = qsub.read_text()
    _write_exe(
        qsub,
        f"""\
        #!/bin/bash
        echo "qsub $*" >> {env["calls"]}
        if [[ "$*" == *depend* ]]; then echo "qsub: Job violates queue limits" >&2; exit 1; fi
        echo 2001.desched1
        """,
    )
    second = budget_plan(env)
    assert len(remote.submit(submit_req(second))["jobs"]) == 1
    assert _guard.used_gpu_hours() == 144
    qsub.write_text(good)
    retry = remote.submit(submit_req(second))  # 144 + 48 <= 200
    assert retry["already_submitted"] and len(retry["jobs"]) == 2
    assert [ln["gpu_hours"] for ln in ledger(owner)] == [96, 48, 48]


def test_unestimable_plan_is_refused_when_budgeted(env, owner):
    set_budgets(owner / ".credit", max_gpu_hours_per_day=1000)
    pbs = {"conda": "/opt/envs/credit", "account": "NAML0001", "queue": "main", "walltime": "soon"}
    cfg = make_config(env, pbs=pbs)  # the config file isn't checked; only an estimate can be refused
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"gpus": 4}}
    plan = remote.plan(req)["plan"]
    assert plan["gpu_hours_estimate"] is None
    err = refused(submit_req(plan))
    assert err.code == "over_budget" and "can't be estimated" in err.message and "walltime" in err.fix
    assert err.data == {"kind": "no_estimate", "estimate": None, "limit": None}


def test_budgets_narrow_across_policy_files(env, owner):
    set_budgets(owner / ".credit", max_gpu_hours_per_plan=100, max_gpu_hours_per_day=500)
    set_budgets(env["home"] / ".credit", max_gpu_hours_per_plan=300, max_gpu_hours_per_day=50)
    assert _guard.policy()["budgets"] == {"max_gpu_hours_per_plan": 100, "max_gpu_hours_per_day": 50}
    set_budgets(env["home"] / ".credit", max_gpu_hours_per_plan=20)
    assert _guard.policy()["budgets"] == {"max_gpu_hours_per_plan": 20, "max_gpu_hours_per_day": 500}
    assert remote.hello({})["policy"]["budgets"] == {"max_gpu_hours_per_plan": 20, "max_gpu_hours_per_day": 500}


@pytest.mark.parametrize(
    "text",
    [
        "budgets: {max_gpu_hours_per_day: lots}\n",
        "budgets: {max_gpu_hours_per_day: -5}\n",
        "budgets: {max_gpu_hours_per_day: 0}\n",
        "budgets: {max_gpu_hours_per_day: true}\n",
        "budgets: {max_gpu_hours_per_dya: 100}\n",  # a misspelt key
        "budgets: [100]\n",
        "budgets: {max_gpu_hours_per_day: 100\n",  # unreadable YAML
    ],
)
def test_malformed_budget_fails_closed(env, owner, text):
    plan = budget_plan(env, chain=1)  # before the policy: an unreadable one also empties the roots
    (owner / ".credit").mkdir()
    (owner / ".credit" / "remote.yaml").write_text(text)
    pol = _guard.policy()
    assert pol["budgets"]["max_gpu_hours_per_day"] == 0 and pol["budget_errors"]
    err = refused({**submit_req(plan), "confirm_gpu_hours": 1000})
    assert err.code == "over_budget" and "invalid" in err.message and str(owner) in err.message
    assert err.data == {"kind": "invalid", "estimate": None, "limit": 0}
    assert qsub_calls(env) == []


def test_budget_ledger_is_private_and_in_the_owner_home(env, owner):
    set_budgets(env["home"] / ".credit", max_gpu_hours_per_day=1000)  # $CREDIT_HOME/remote.yaml
    remote.submit(submit_req(budget_plan(env, chain=1)))
    path = owner / ".credit" / "budget.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not (env["home"] / ".credit" / "budget.jsonl").exists()  # not $CREDIT_HOME or $HOME
    assert ledger(owner)[0]["gpu_hours"] == 48


def test_a_held_budget_lock_blocks_submits(env, owner, monkeypatch):
    set_budgets(owner / ".credit", max_gpu_hours_per_day=1000)
    monkeypatch.setattr(_guard, "BUDGET_LOCK_TIMEOUT_S", 0.3)
    plan = budget_plan(env, chain=1)
    with _guard.budget_lock():  # another submit between its check and its record
        assert refused(submit_req(plan)).code == "conflict"
    assert qsub_calls(env) == [] and ledger(owner) == []
    assert len(remote.submit(submit_req(plan))["jobs"]) == 1


def test_budget_refusals_are_audited(env):
    # Through the real subprocess; refused before the (real-home) ledger is touched.
    set_budgets(env["home"] / ".credit", max_gpu_hours_per_plan=10)
    cfg = make_config(env)
    req = {"kind": "train", "config": str(cfg), "scheduler": "pbs", "cluster": "derecho", "options": {"chain": 1}}
    plan = call("plan", req)["plan"]
    resp = call("submit", {**submit_req(plan), "confirm_gpu_hours": 5}, check_ok=False)
    assert resp["error"]["code"] == "over_budget"
    assert resp["error"]["data"] == {"kind": "per_plan", "estimate": 48, "limit": 10}  # in the envelope
    path = env["home"] / ".credit" / "audit.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert lines[-1]["verb"] == "submit" and lines[-1]["code"] == "over_budget"
    assert lines[-1]["req"]["confirm_gpu_hours"] == "5" and lines[-1]["req"]["run_id"] == plan["run_id"]
    assert call("read", {"path": "/etc/hosts"}, check_ok=False)["error"]["data"] is None  # other errors: null
