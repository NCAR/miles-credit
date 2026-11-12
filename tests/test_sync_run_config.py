"""Tests for ``credit.run_config.sync_run_config`` (save_loc/model.yml tracking via git)."""

import logging
import shutil
import subprocess

import pytest

from credit.run_config import sync_run_config

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _log(save_loc, *args):
    return subprocess.run(["git", "-C", str(save_loc), *args], capture_output=True, text=True, check=True).stdout


def _cfg(tmp_path, text, name="cfg.yml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


@needs_git
def test_sync_run_config_first_run_creates_repo(tmp_path):
    save = tmp_path / "run"
    sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "1"
    assert _log(save, "status", "--porcelain").strip() == ""  # .gitignore must not show as untracked


@needs_git
def test_sync_run_config_identical_rerun_no_new_commit(tmp_path):
    save = tmp_path / "run"
    cfg = _cfg(tmp_path, "a: 1\n")
    sync_run_config(cfg, str(save))
    sync_run_config(cfg, str(save))
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "1"


@needs_git
def test_sync_run_config_edited_config_updates_and_keeps_history(tmp_path, caplog):
    save = tmp_path / "run"
    sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 2\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 2\n"
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "2"
    assert _log(save, "show", "HEAD~1:model.yml") == "a: 1\n"
    assert "replaced" in caplog.text


@needs_git
def test_sync_run_config_legacy_untracked_model_yml_gets_baseline(tmp_path):
    save = tmp_path / "run"
    save.mkdir()
    (save / "model.yml").write_text("a: old\n")
    sync_run_config(_cfg(tmp_path, "a: new\n"), str(save))
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "2"
    assert _log(save, "show", "HEAD~1:model.yml") == "a: old\n"


@needs_git
def test_sync_run_config_same_file_is_noop(tmp_path):
    save = tmp_path / "run"
    sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    sync_run_config(str(save / "model.yml"), str(save))
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "1"


@needs_git
def test_sync_run_config_nested_in_other_repo_uses_own_repo(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    save = outer / "run"
    sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / ".git").is_dir()
    assert _log(outer, "status", "--porcelain", "--untracked-files=no").strip() == ""


def test_sync_run_config_without_git_still_copies_and_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("credit.run_config.shutil.which", lambda name: None)
    save = tmp_path / "run"
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert "git is required" in caplog.text


@needs_git
def test_sync_run_config_git_init_failure_still_copies_and_warns(tmp_path, monkeypatch, caplog):
    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

    monkeypatch.setattr("credit.run_config._git", fail)
    save = tmp_path / "run"
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert "boom" in caplog.text and "git is required" in caplog.text


def test_sync_run_config_expands_env_vars_in_save_loc(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_TEST_RUNS", str(tmp_path))
    dest = sync_run_config(_cfg(tmp_path, "a: 1\n"), "$CREDIT_TEST_RUNS/run")
    assert dest == str(tmp_path / "run" / "model.yml")
    assert (tmp_path / "run" / "model.yml").read_text() == "a: 1\n"


@needs_git
def test_sync_run_config_git_commit_failure_still_copies_and_warns(tmp_path, monkeypatch, caplog):
    import credit.run_config as run_config

    real_git = run_config._git

    def fail_commit(save_loc, *args, **kwargs):
        if args and args[0] == "commit":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="commit boom")
        return real_git(save_loc, *args, **kwargs)

    monkeypatch.setattr(run_config, "_git", fail_commit)
    save = tmp_path / "run"
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert "commit boom" in caplog.text


@needs_git
def test_sync_run_config_skips_user_owned_repo_at_save_loc(tmp_path, caplog):
    save = tmp_path / "userrepo"
    save.mkdir()
    subprocess.run(["git", "init", "-q", str(save)], check=True)
    (save / "x.txt").write_text("x")
    subprocess.run(["git", "-C", str(save), "add", "x.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(save), "-c", "user.name=u", "-c", "user.email=u@e", "commit", "-q", "-m", "user commit"],
        check=True,
    )
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert _log(save, "rev-list", "--count", "HEAD").strip() == "1"
    assert _log(save, "log", "-1", "--format=%s").strip() == "user commit"
    assert not (save / ".gitignore").exists()
    assert "not created by CREDIT" in caplog.text
    assert "outside the miles-credit repo" in caplog.text


@needs_git
def test_sync_run_config_dubious_ownership_suggests_safe_directory(tmp_path, monkeypatch, caplog):
    from credit import run_config

    real_git = run_config._git

    def dubious(save_loc, *args, **kwargs):
        if args[:3] == ("config", "--local", "--get"):
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: detected dubious ownership")
        return real_git(save_loc, *args, **kwargs)

    monkeypatch.setattr(run_config, "_git", dubious)
    save = tmp_path / "shared"
    with caplog.at_level(logging.WARNING):
        sync_run_config(_cfg(tmp_path, "a: 1\n"), str(save))
    assert (save / "model.yml").read_text() == "a: 1\n"
    assert "safe.directory" in caplog.text
    assert "outside the miles-credit repo" not in caplog.text
    assert subprocess.run(["git", "-C", str(save), "rev-parse", "HEAD"], capture_output=True).returncode != 0


# ---------------------------------------------------------------------------
# Callers: the config must reach save_loc/model.yml before a job runs -c <save_loc>/model.yml
# ---------------------------------------------------------------------------


def _launcher_cfg(tmp_path, save, text_extra=""):
    return _cfg(tmp_path, f"save_loc: {save}\npbs:\n  queue: main\n  project: TEST0000\n{text_extra}")


@pytest.mark.parametrize(
    "module, func",
    [
        ("credit.pbs", "launch_script_mpi"),
        ("credit.slurm", "launch_script"),
        ("credit.slurm", "launch_script_mpi"),
        ("credit.slurm", "launch_script_torchrun"),
    ],
)
def test_launchers_write_model_yml(tmp_path, monkeypatch, module, func):
    import importlib

    mod = importlib.import_module(module)

    def no_submit(*args, **kwargs):
        raise AssertionError("launch=False must not submit a job")

    monkeypatch.setattr(mod.subprocess, "Popen", no_submit)
    monkeypatch.setattr(mod.subprocess, "run", no_submit)
    monkeypatch.chdir(tmp_path)  # some launchers write launch.sh to the working directory
    save = tmp_path / "run"
    cfg = _launcher_cfg(tmp_path, save)
    getattr(mod, func)(cfg, str(tmp_path / "train.py"), launch=False)
    assert (save / "model.yml").read_text() == (tmp_path / "cfg.yml").read_text()


def _run_gen1_main_cli(monkeypatch, cfg, launch, world_rank=0):
    import sys

    from credit.applications import train_gen1

    calls = {}
    monkeypatch.setattr(train_gen1, "credit_main_parser", lambda conf, **kw: conf)
    monkeypatch.setattr(train_gen1, "training_data_check", lambda conf, **kw: None)
    monkeypatch.setattr(train_gen1, "get_rank_info", lambda mode: (0, world_rank, 2))
    monkeypatch.setattr(train_gen1, "main", lambda *a, **kw: calls.setdefault("main", True))

    def fake_launch(config, script_path, *a, **kw):
        # The generated PBS script reads save_loc/model.yml, so it must exist by now.
        with open(config) as f:
            calls["launched_with"] = f.read()

    monkeypatch.setattr(train_gen1, "launch_script", fake_launch)
    monkeypatch.setattr(train_gen1, "launch_script_mpi", fake_launch)
    monkeypatch.setattr(sys, "argv", ["credit_train_gen1", "-c", cfg, "-l", str(launch)])
    if launch:
        with pytest.raises(SystemExit):
            train_gen1.main_cli()
    else:
        train_gen1.main_cli()
    return calls


@pytest.mark.parametrize("queue", ["casper", "main"])
def test_train_gen1_launch_writes_model_yml_before_submitting(tmp_path, monkeypatch, queue):
    save = tmp_path / "run"
    text = f"save_loc: {save}\ndata:\n  datasets: {{}}\ntrainer:\n  mode: none\npbs:\n  queue: {queue}\n"
    cfg = _cfg(tmp_path, text)
    calls = _run_gen1_main_cli(monkeypatch, cfg, launch=1)
    assert "launched_with" in calls
    assert (save / "model.yml").read_text() == text
    assert "main" not in calls


@pytest.mark.parametrize("world_rank, expect_written", [(0, True), (1, False)])
def test_train_gen1_only_rank_zero_writes_model_yml(tmp_path, monkeypatch, world_rank, expect_written):
    save = tmp_path / "run"
    text = f"save_loc: {save}\ndata:\n  datasets: {{}}\ntrainer:\n  mode: ddp\n"
    calls = _run_gen1_main_cli(monkeypatch, _cfg(tmp_path, text), launch=0, world_rank=world_rank)
    assert calls.get("main")
    assert (save / "model.yml").exists() == expect_written
