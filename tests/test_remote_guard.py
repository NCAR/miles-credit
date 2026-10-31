"""credit.remote hardening: the remote.yaml policy, secret redaction, and the audit log."""

import json
import os
import time

import yaml

from credit import runs
from credit.remote import _guard
from tests.test_remote import call, env, make_run  # noqa: F401 — `env` is a fixture

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


def test_read_logs_and_sync_are_redacted(env):  # noqa: F811
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


def test_write_refuses_redacted_text(env):  # noqa: F811
    cfg = env["home"] / "exp.yml"
    cfg.write_text(f"hf_token: {GH}\n")
    r = call("read", {"path": str(cfg)})
    resp = call("write", {"path": str(cfg), "text": r["data"], "if_sha256": r["sha256"]}, check_ok=False)
    assert not resp["ok"] and resp["error"]["code"] == "config_error"
    assert GH in cfg.read_text()  # the secret survived


def test_policy_roots_narrow_scope(env):  # noqa: F811
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


def test_policy_can_turn_redaction_off(env):  # noqa: F811
    (env["home"] / ".credit").mkdir(exist_ok=True)
    (env["home"] / ".credit" / "remote.yaml").write_text("redact: false\n")
    cfg = env["home"] / "plain.yml"
    cfg.write_text(f"hf_token: {GH}\n")
    assert GH in call("read", {"path": str(cfg)})["data"]


def test_audit_log_records_calls_without_contents(env):  # noqa: F811
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
