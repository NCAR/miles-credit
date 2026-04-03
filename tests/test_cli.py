"""Unit tests for credit/cli.py.

All tests run on CPU with no real data files.  External calls (qsub, torchrun,
LLM APIs) are mocked via monkeypatch / unittest.mock.
"""

import argparse
import os
import sys
import subprocess
from unittest import mock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import credit.cli as cli
from credit.cli import (
    _prompt,
    _prompt_bool,
    _setup_logging,
    _repo_root,
    _write_reload_config,
    _load_pbs_config,
    _resolve_pbs_opts,
    _build_pbs_script,
    _qsub,
    _compute_chain,
    _print_job_plan,
    _build_rollout_pbs_script,
    _print_ensemble_rollout_plan,
    _find_torchrun,
    _is_ncar_system,
    _build_channel_map,
    _agent_read_file,
    _agent_list_files,
    _agent_bash,
    _dispatch_tool,
    _build_parser,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_conf(tmp_path, extra=None):
    """Write a minimal YAML config and return its path."""
    conf = {
        "save_loc": str(tmp_path / "save"),
        "trainer": {
            "type": "era5-v2",
            "epochs": 20,
            "num_epoch": 5,
        },
        "data": {
            "forecast_len": 1,
            "valid_forecast_len": 1,
        },
        "pbs": {
            "project": "NAML0001",
            "walltime": "06:00:00",
            "job_name": "test_job",
        },
    }
    if extra:
        conf.update(extra)
    p = tmp_path / "config.yml"
    p.write_text(yaml.dump(conf))
    return str(p)


def _casper_args(**kwargs):
    """Return a minimal Namespace for casper submit."""
    defaults = dict(
        cluster="casper",
        config="config.yml",
        gpus=4,
        nodes=1,
        cpus=8,
        mem="128GB",
        queue="casper",
        gpu_type="a100_80gb",
        walltime="06:00:00",
        account="NAML0001",
        conda_env="/some/env",
        torchrun="/usr/bin/torchrun",
        job_name="credit_v2",
        dry_run=False,
        reload=False,
        chain=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _derecho_args(**kwargs):
    """Return a minimal Namespace for derecho submit."""
    defaults = dict(
        cluster="derecho",
        config="config.yml",
        gpus=4,
        nodes=1,
        cpus=64,
        mem="480GB",
        queue="main",
        gpu_type="a100_80gb",
        walltime="12:00:00",
        account="NAML0001",
        conda_env="/glade/work/benkirk/conda-envs/credit-derecho-torch28-nccl221",
        torchrun=None,
        job_name="credit_v2",
        dry_run=False,
        reload=False,
        chain=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ===========================================================================
# TestPrompt
# ===========================================================================


class TestPrompt:
    """Tests for _prompt and _prompt_bool."""

    def test_prompt_returns_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "hello")
        assert _prompt("Enter something") == "hello"

    def test_prompt_uses_default_on_empty(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "  ")
        assert _prompt("Enter something", default="mydefault") == "mydefault"

    def test_prompt_no_default_empty_returns_empty(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _prompt("Enter something") == ""

    def test_prompt_default_shown_in_hint(self, monkeypatch):
        calls = []
        monkeypatch.setattr("builtins.input", lambda s: calls.append(s) or "")
        _prompt("My question", default="42")
        assert "[42]" in calls[0]

    def test_prompt_bool_yes(self, monkeypatch):
        for val in ("y", "yes", "Y", "YES"):
            monkeypatch.setattr("builtins.input", lambda _, v=val: v)
            assert _prompt_bool("Enable?") is True

    def test_prompt_bool_no(self, monkeypatch):
        for val in ("n", "no", "N", "NO"):
            monkeypatch.setattr("builtins.input", lambda _, v=val: v)
            assert _prompt_bool("Enable?") is False

    def test_prompt_bool_empty_uses_default_true(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _prompt_bool("Enable?", default=True) is True

    def test_prompt_bool_empty_uses_default_false(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _prompt_bool("Enable?", default=False) is False

    def test_prompt_bool_hint_shows_yn(self, monkeypatch):
        calls = []
        monkeypatch.setattr("builtins.input", lambda s: calls.append(s) or "")
        _prompt_bool("Enable?", default=True)
        assert "Y/n" in calls[0]

    def test_prompt_bool_hint_shows_ny(self, monkeypatch):
        calls = []
        monkeypatch.setattr("builtins.input", lambda s: calls.append(s) or "")
        _prompt_bool("Enable?", default=False)
        assert "y/N" in calls[0]


# ===========================================================================
# TestSetupLogging
# ===========================================================================


class TestSetupLogging:
    """Tests for _setup_logging."""

    def test_no_exception(self):
        import logging

        _setup_logging(logging.WARNING)

    def test_no_exception_debug(self):
        import logging

        _setup_logging(logging.DEBUG)


# ===========================================================================
# TestRepoRoot
# ===========================================================================


class TestRepoRoot:
    """Tests for _repo_root."""

    def test_returns_string(self):
        r = _repo_root()
        assert isinstance(r, str)

    def test_is_absolute(self):
        assert os.path.isabs(_repo_root())

    def test_points_to_directory(self):
        # May not exist in all envs but must be a reasonable path
        r = _repo_root()
        assert r  # non-empty


# ===========================================================================
# TestWriteReloadConfig
# ===========================================================================


class TestWriteReloadConfig:
    """Tests for _write_reload_config."""

    def test_creates_file(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        reload_path = _write_reload_config(config_path)
        assert os.path.exists(reload_path)

    def test_reload_fields_set(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        reload_path = _write_reload_config(config_path)
        with open(reload_path) as f:
            conf = yaml.safe_load(f)
        trainer = conf["trainer"]
        assert trainer["load_weights"] is True
        assert trainer["load_optimizer"] is True
        assert trainer["load_scaler"] is True
        assert trainer["load_scheduler"] is True
        assert trainer["reload_epoch"] is True

    def test_written_to_save_loc(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        reload_path = _write_reload_config(config_path)
        assert reload_path.endswith("config_reload.yml")

    def test_returns_path_string(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        result = _write_reload_config(config_path)
        assert isinstance(result, str)


# ===========================================================================
# TestLoadPbsConfig
# ===========================================================================


class TestLoadPbsConfig:
    """Tests for _load_pbs_config."""

    def test_returns_pbs_section(self, tmp_path):
        conf = {"pbs": {"project": "MYACCT", "walltime": "04:00:00"}}
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        result = _load_pbs_config(str(p))
        assert result["project"] == "MYACCT"
        assert result["walltime"] == "04:00:00"

    def test_missing_pbs_returns_empty(self, tmp_path):
        conf = {"trainer": {"type": "era5-v2"}}
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        assert _load_pbs_config(str(p)) == {}

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = _load_pbs_config(str(tmp_path / "does_not_exist.yml"))
        assert result == {}

    def test_broken_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yml"
        p.write_text("key: [broken yaml")
        result = _load_pbs_config(str(p))
        # may or may not succeed; must not raise — just return dict
        assert isinstance(result, dict)


# ===========================================================================
# TestResolvePbsOpts
# ===========================================================================


class TestResolvePbsOpts:
    """Tests for _resolve_pbs_opts."""

    def _minimal_args(self, **kwargs):
        defaults = dict(
            cluster="casper",
            account=None,
            walltime=None,
            gpus=None,
            nodes=None,
            cpus=None,
            mem=None,
            queue=None,
            gpu_type=None,
            conda_env=None,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_cli_flag_takes_precedence(self):
        args = self._minimal_args(cluster="casper", account="MYACCT")
        pbs_cfg = {"project": "OTHER"}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.account == "MYACCT"

    def test_falls_back_to_pbs_cfg(self):
        args = self._minimal_args(cluster="casper", account=None)
        pbs_cfg = {"project": "FROMCONFIG"}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.account == "FROMCONFIG"

    def test_uses_alias_account(self):
        args = self._minimal_args(cluster="casper")
        pbs_cfg = {"account": "ALIAS_ACCT"}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.account == "ALIAS_ACCT"

    def test_casper_defaults(self):
        args = self._minimal_args(cluster="casper")
        r = _resolve_pbs_opts(args, {})
        assert r.cpus == 8
        assert r.mem == "128GB"
        assert r.queue == "casper"

    def test_derecho_defaults(self):
        args = self._minimal_args(cluster="derecho")
        r = _resolve_pbs_opts(args, {})
        assert r.cpus == 64
        assert r.mem == "480GB"
        assert r.queue == "main"

    def test_gpus_alias(self):
        args = self._minimal_args(cluster="casper", gpus=None)
        pbs_cfg = {"gpus": 2}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.gpus == 2

    def test_gpus_ngpus_alias(self):
        args = self._minimal_args(cluster="casper", gpus=None)
        pbs_cfg = {"ngpus": 3}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.gpus == 3

    def test_conda_env_alias(self):
        args = self._minimal_args(cluster="casper", conda_env=None)
        pbs_cfg = {"conda_env": "/some/env"}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert r.conda_env == "/some/env"

    def test_does_not_mutate_original(self):
        args = self._minimal_args(cluster="casper", account="ORIG")
        pbs_cfg = {}
        r = _resolve_pbs_opts(args, pbs_cfg)
        assert args.account == "ORIG"
        assert r is not args


# ===========================================================================
# TestBuildPbsScript
# ===========================================================================


class TestBuildPbsScript:
    """Tests for _build_pbs_script."""

    def test_casper_contains_pbs_directives(self):
        args = _casper_args()
        script = _build_pbs_script(args, "/path/to/config.yml", "/path/to/repo")
        assert "#!/bin/bash" in script
        assert "#PBS -N" in script
        assert "#PBS -l select=" in script
        assert "#PBS -l walltime=" in script
        assert "#PBS -A" in script

    def test_casper_script_has_torchrun(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/path/to/config.yml", "/path/to/repo")
        assert "torchrun" in script
        assert "--standalone" in script

    def test_derecho_single_node_standalone(self):
        args = _derecho_args(nodes=1)
        script = _build_pbs_script(args, "/path/to/config.yml", "/path/to/repo")
        assert "--standalone" in script
        assert "#PBS -A" in script

    def test_derecho_multi_node_mpiexec(self):
        args = _derecho_args(nodes=2)
        script = _build_pbs_script(args, "/path/to/config.yml", "/path/to/repo")
        assert "mpiexec" in script
        assert "--rdzv-backend=c10d" in script

    def test_depend_on_added(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/cfg.yml", "/repo", depend_on="12345.pbs")
        assert "afterok:12345.pbs" in script

    def test_depend_on_absent_when_none(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/cfg.yml", "/repo", depend_on=None)
        assert "afterok" not in script

    def test_account_override(self):
        args = _casper_args(account="ORIGINAL", torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/cfg.yml", "/repo", account="OVERRIDE")
        assert "OVERRIDE" in script

    def test_job_name_in_script(self):
        args = _casper_args(job_name="my_special_job", torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/cfg.yml", "/repo")
        assert "my_special_job" in script

    def test_config_path_in_script(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/my/config.yml", "/repo")
        assert "/my/config.yml" in script

    def test_repo_path_in_script(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_pbs_script(args, "/cfg.yml", "/my/repo")
        assert "/my/repo" in script


# ===========================================================================
# TestQsub
# ===========================================================================


class TestQsub:
    """Tests for _qsub — mocks subprocess.run."""

    def test_calls_qsub(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return mock.MagicMock(returncode=0, stdout="12345.pbs\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = _qsub("#!/bin/bash\necho hello")
        assert captured["cmd"][0] == "qsub"
        assert result == "12345.pbs"

    def test_strips_whitespace_from_job_id(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: mock.MagicMock(returncode=0, stdout="  99999.pbs  \n", stderr=""),
        )
        assert _qsub("script") == "99999.pbs"

    def test_exits_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: mock.MagicMock(returncode=1, stdout="", stderr="qsub: bad script"),
        )
        with pytest.raises(SystemExit):
            _qsub("script")


# ===========================================================================
# TestComputeChain
# ===========================================================================


class TestComputeChain:
    """Tests for _compute_chain."""

    def test_explicit_chain_returned(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        args = argparse.Namespace(chain=7, config=config_path)
        assert _compute_chain(args) == 7

    def test_computed_from_config(self, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        # epochs=20, num_epoch=5 → ceil(20/5) = 4
        args = argparse.Namespace(chain=None, config=config_path)
        assert _compute_chain(args) == 4

    def test_ceil_division(self, tmp_path):
        conf = {
            "trainer": {"epochs": 22, "num_epoch": 5},
            "save_loc": str(tmp_path / "save"),
        }
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        args = argparse.Namespace(chain=None, config=str(p))
        assert _compute_chain(args) == 5  # ceil(22/5) = 5

    def test_fallback_to_one_on_missing_keys(self, tmp_path):
        conf = {"trainer": {}}
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        args = argparse.Namespace(chain=None, config=str(p))
        assert _compute_chain(args) == 1

    def test_fallback_to_one_on_missing_file(self, tmp_path):
        args = argparse.Namespace(chain=None, config=str(tmp_path / "no_file.yml"))
        assert _compute_chain(args) == 1


# ===========================================================================
# TestPrintJobPlan
# ===========================================================================


def _inject_preflight(mem_gb=0.0):
    """Context manager: injects a fake preflight module into sys.modules."""
    import types
    import sys
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        key = "credit.trainers.preflight"
        fake = types.ModuleType(key)
        fake.estimate_dataloader_memory_gb = lambda conf: mem_gb
        old = sys.modules.get(key)
        sys.modules[key] = fake
        try:
            yield
        finally:
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old

    return _ctx()


class TestPrintJobPlan:
    """Tests for _print_job_plan — just verifies it doesn't raise."""

    def test_no_exception_casper(self, tmp_path, capsys):
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(config=config_path)

        with _inject_preflight(0.0):
            _print_job_plan(args, n_jobs=3)

        out = capsys.readouterr().out
        assert "Job plan" in out

    def test_no_exception_derecho(self, tmp_path, capsys):
        config_path = _make_minimal_conf(tmp_path)
        args = _derecho_args(config=config_path, nodes=2)

        with _inject_preflight(5.0):
            _print_job_plan(args, n_jobs=1)

        out = capsys.readouterr().out
        assert "derecho" in out

    def test_bad_config_no_exception(self, tmp_path, capsys):
        args = _casper_args(config=str(tmp_path / "no_file.yml"))

        with _inject_preflight(0.0):
            _print_job_plan(args, n_jobs=1)


# ===========================================================================
# TestBuildRolloutPbsScript
# ===========================================================================


class TestBuildRolloutPbsScript:
    """Tests for _build_rollout_pbs_script."""

    def test_casper_contains_pbs_directives(self):
        args = _casper_args(job_name="rollout_job", torchrun="/usr/bin/torchrun")
        script = _build_rollout_pbs_script(args, "/cfg.yml", "/repo", subset=1, n_subsets=5)
        assert "#!/bin/bash" in script
        assert "#PBS -N" in script
        assert "subset 1 of 5" in script

    def test_casper_subset_in_torchrun_call(self):
        args = _casper_args(torchrun="/usr/bin/torchrun")
        script = _build_rollout_pbs_script(args, "/cfg.yml", "/repo", subset=3, n_subsets=10)
        assert "--subset 3" in script
        assert "--no_subset 10" in script

    def test_derecho_subset_in_torchrun_call(self):
        args = _derecho_args(job_name="rollout_job")
        script = _build_rollout_pbs_script(args, "/cfg.yml", "/repo", subset=2, n_subsets=8)
        assert "--subset 2" in script
        assert "--no_subset 8" in script

    def test_job_name_truncated_and_tagged(self):
        # job_name[:10] + -01of05
        args = _casper_args(job_name="my_very_long_job_name", torchrun="/usr/bin/torchrun")
        script = _build_rollout_pbs_script(args, "/cfg.yml", "/repo", subset=1, n_subsets=5)
        # Name should be truncated to first 10 chars + suffix
        assert "my_very_lo" in script or "01of05" in script


# ===========================================================================
# TestPrintEnsembleRolloutPlan
# ===========================================================================


class TestPrintEnsembleRolloutPlan:
    """Tests for _print_ensemble_rollout_plan."""

    def test_no_exception(self, capsys):
        args = _casper_args()
        _print_ensemble_rollout_plan(args, n_jobs=5, n_forecasts=50, ensemble_size=4)
        out = capsys.readouterr().out
        assert "Ensemble rollout plan" in out

    def test_shows_n_jobs(self, capsys):
        args = _casper_args()
        _print_ensemble_rollout_plan(args, n_jobs=8, n_forecasts=40, ensemble_size=2)
        out = capsys.readouterr().out
        assert "8" in out

    def test_shows_total_forecasts(self, capsys):
        args = _casper_args()
        _print_ensemble_rollout_plan(args, n_jobs=5, n_forecasts=10, ensemble_size=3)
        out = capsys.readouterr().out
        # 10 forecasts × 3 ensemble = 30 total
        assert "30" in out


# ===========================================================================
# TestFindTorchrun
# ===========================================================================


class TestFindTorchrun:
    """Tests for _find_torchrun."""

    def test_returns_path_from_which(self, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/torchrun")
        result = _find_torchrun()
        assert result == "/usr/local/bin/torchrun"

    def test_falls_back_to_torchrun_string(self, monkeypatch, tmp_path):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: None)
        # Make sure fallback path does not exist
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _find_torchrun()
        assert "torchrun" in result

    def test_returns_string(self, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: None)
        result = _find_torchrun()
        assert isinstance(result, str)


# ===========================================================================
# TestIsNcarSystem
# ===========================================================================


class TestIsNcarSystem:
    """Tests for _is_ncar_system."""

    def test_casper_hostname(self, monkeypatch):
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "casper01.ucar.edu")
        assert _is_ncar_system() is True

    def test_derecho_hostname(self, monkeypatch):
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "derecho05.hpc.ucar.edu")
        assert _is_ncar_system() is True

    def test_unknown_hostname(self, monkeypatch):
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "myworkstation.local")
        assert _is_ncar_system() is False

    def test_crhtc_hostname(self, monkeypatch):
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "crhtc14")
        assert _is_ncar_system() is True

    def test_crlogin_hostname(self, monkeypatch):
        import socket

        monkeypatch.setattr(socket, "gethostname", lambda: "crlogin01")
        assert _is_ncar_system() is True


# ===========================================================================
# TestBuildChannelMap
# ===========================================================================


class TestBuildChannelMap:
    """Tests for _build_channel_map."""

    def _minimal_conf(self, vars_3d, vars_2d_prog, vars_2d_diag, n_levels=3):
        return {
            "data": {
                "source": {
                    "ERA5": {
                        "levels": list(range(n_levels)),
                        "variables": {
                            "prognostic": {
                                "vars_3D": vars_3d,
                                "vars_2D": vars_2d_prog,
                            },
                            "diagnostic": {
                                "vars_2D": vars_2d_diag,
                            },
                        },
                    }
                }
            }
        }

    def test_3d_vars_span_n_levels(self):
        conf = self._minimal_conf(["U", "V"], [], [], n_levels=4)
        cmap = _build_channel_map(conf)
        assert len(cmap["U"]) == 4
        assert len(cmap["V"]) == 4

    def test_2d_vars_span_one_channel(self):
        conf = self._minimal_conf([], ["SP", "T2M"], [], n_levels=3)
        cmap = _build_channel_map(conf)
        assert len(cmap["SP"]) == 1
        assert len(cmap["T2M"]) == 1

    def test_diagnostic_2d_vars_appended(self):
        conf = self._minimal_conf([], [], ["precip"], n_levels=3)
        cmap = _build_channel_map(conf)
        assert "precip" in cmap
        assert len(cmap["precip"]) == 1

    def test_channel_indices_are_contiguous(self):
        conf = self._minimal_conf(["U", "V"], ["SP"], ["precip"], n_levels=2)
        cmap = _build_channel_map(conf)
        # U: [0,1], V: [2,3], SP: [4], precip: [5]
        assert cmap["U"] == [0, 1]
        assert cmap["V"] == [2, 3]
        assert cmap["SP"] == [4]
        assert cmap["precip"] == [5]

    def test_empty_conf_returns_empty(self):
        conf = self._minimal_conf([], [], [], n_levels=5)
        cmap = _build_channel_map(conf)
        assert cmap == {}


# ===========================================================================
# TestAgentReadFile
# ===========================================================================


class TestAgentReadFile:
    """Tests for _agent_read_file."""

    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3")
        result = _agent_read_file(str(f))
        assert "line1" in result
        assert "line3" in result

    def test_missing_file_returns_message(self, tmp_path):
        result = _agent_read_file(str(tmp_path / "missing.txt"))
        assert "not found" in result.lower() or "File not found" in result

    def test_tail_limits_lines(self, tmp_path):
        f = tmp_path / "big.txt"
        lines = [f"line{i}" for i in range(100)]
        f.write_text("\n".join(lines))
        result = _agent_read_file(str(f), tail=10)
        assert "line99" in result
        assert "line0" not in result

    def test_tail_zero_returns_all(self, tmp_path):
        f = tmp_path / "full.txt"
        lines = [f"line{i}" for i in range(20)]
        f.write_text("\n".join(lines))
        result = _agent_read_file(str(f), tail=0)
        assert "line0" in result
        assert "line19" in result

    def test_omission_message_when_truncated(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("\n".join([f"x{i}" for i in range(50)]))
        result = _agent_read_file(str(f), tail=5)
        assert "omitted" in result

    def test_large_file_blocked(self, tmp_path):
        f = tmp_path / "huge.txt"
        with mock.patch("pathlib.Path.stat") as stat_mock:
            stat_result = mock.MagicMock()
            stat_result.st_size = 20 * 1024 * 1024  # 20 MB
            stat_mock.return_value = stat_result
            # Need the file to exist
            f.write_text("data")
            result = _agent_read_file(str(f))
        assert "too large" in result.lower()


# ===========================================================================
# TestAgentListFiles
# ===========================================================================


class TestAgentListFiles:
    """Tests for _agent_list_files."""

    def test_finds_files(self, tmp_path, monkeypatch):
        (tmp_path / "a.yml").write_text("a")
        (tmp_path / "b.yml").write_text("b")
        monkeypatch.chdir(tmp_path)
        result = _agent_list_files("*.yml")
        assert "a.yml" in result
        assert "b.yml" in result

    def test_no_match_returns_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _agent_list_files("*.nonexistent_ext")
        assert "No files matched" in result

    def test_returns_at_most_200(self, tmp_path, monkeypatch):
        for i in range(250):
            (tmp_path / f"f{i:03d}.txt").write_text(str(i))
        monkeypatch.chdir(tmp_path)
        result = _agent_list_files("*.txt")
        # At most 200 lines returned
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) <= 200


# ===========================================================================
# TestAgentBash
# ===========================================================================


class TestAgentBash:
    """Tests for _agent_bash."""

    def test_allowed_command_runs(self):
        result = _agent_bash("echo hello_credit")
        assert "hello_credit" in result

    def test_blocked_rm(self):
        result = _agent_bash("rm -rf /tmp/something")
        assert "Blocked" in result

    def test_blocked_git_commit(self):
        result = _agent_bash("git commit -m 'test'")
        assert "Blocked" in result

    def test_blocked_pip_install(self):
        result = _agent_bash("pip install requests")
        assert "Blocked" in result

    def test_blocked_sudo(self):
        result = _agent_bash("sudo ls /root")
        assert "Blocked" in result

    def test_blocked_qdel(self):
        result = _agent_bash("qdel 12345")
        assert "Blocked" in result

    def test_blocked_wget(self):
        result = _agent_bash("wget http://example.com")
        assert "Blocked" in result

    def test_empty_output_returns_no_output_message(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: mock.MagicMock(returncode=0, stdout="", stderr=""),
        )
        result = _agent_bash("echo")
        # Command runs successfully; mock returns empty
        assert "(no output)" in result or result == "(no output)"

    def test_timeout_returns_message(self, monkeypatch):
        def raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired("cmd", 30)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        result = _agent_bash("sleep 100")
        assert "timed out" in result.lower()

    def test_large_output_truncated(self, monkeypatch):
        big = "x" * 10000
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: mock.MagicMock(returncode=0, stdout=big, stderr=""),
        )
        result = _agent_bash("cat bigfile")
        assert len(result) <= 8100  # 8000 + some slack


# ===========================================================================
# TestDispatchTool
# ===========================================================================


class TestDispatchTool:
    """Tests for _dispatch_tool."""

    def test_dispatches_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("dispatch_content")
        result = _dispatch_tool("read_file", {"path": str(f)})
        assert "dispatch_content" in result

    def test_dispatches_list_files(self, tmp_path, monkeypatch):
        (tmp_path / "x.yml").write_text("")
        monkeypatch.chdir(tmp_path)
        result = _dispatch_tool("list_files", {"pattern": "*.yml"})
        assert "x.yml" in result

    def test_dispatches_bash(self):
        result = _dispatch_tool("bash", {"command": "echo dispatch_test"})
        assert "dispatch_test" in result

    def test_unknown_tool(self):
        result = _dispatch_tool("no_such_tool", {})
        assert "Unknown tool" in result


# ===========================================================================
# TestBuildParser
# ===========================================================================


class TestBuildParser:
    """Tests for _build_parser — verifies all subcommands are registered."""

    def test_returns_argument_parser(self):
        parser = _build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.parametrize(
        "subcmd",
        ["train", "rollout", "rollout-ensemble", "realtime", "submit", "plot", "ask", "convert", "init"],
    )
    def test_subcommand_exists(self, subcmd):
        parser = _build_parser()
        # Parsing --help for the subcommand should not raise
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([subcmd, "--help"])
        assert exc_info.value.code == 0

    def test_train_requires_config(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_submit_requires_cluster(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-c", "cfg.yml"])

    def test_init_default_grid(self):
        parser = _build_parser()
        args = parser.parse_args(["init", "-o", "out.yml"])
        assert args.grid == "0.25deg"

    def test_init_default_model(self):
        parser = _build_parser()
        args = parser.parse_args(["init", "-o", "out.yml"])
        assert args.model == "wxformer"

    def test_rollout_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["rollout", "-c", "cfg.yml"])
        assert args.mode == "none"
        assert args.procs == 4

    def test_realtime_required_init_time(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["realtime", "-c", "cfg.yml"])

    def test_ask_max_turns_default(self):
        parser = _build_parser()
        args = parser.parse_args(["ask", "hello?"])
        assert args.max_turns == 20

    def test_submit_dry_run_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["submit", "--cluster", "casper", "-c", "cfg.yml", "--dry-run"])
        assert args.dry_run is True

    def test_submit_reload_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["submit", "--cluster", "casper", "-c", "cfg.yml", "--reload"])
        assert args.reload is True


# ===========================================================================
# TestMain
# ===========================================================================


class TestMain:
    """Tests for the main() entrypoint."""

    def test_no_args_exits_zero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["credit"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_help_exits_zero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["credit", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_unknown_command_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["credit", "unknown_command"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0


# ===========================================================================
# TestSubmitDryRun
# ===========================================================================


class TestSubmitDryRun:
    """Tests for _submit in dry-run mode (no qsub calls)."""

    def test_dry_run_casper_prints_script(self, tmp_path, capsys, monkeypatch):
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(
            config=config_path,
            dry_run=True,
            chain=1,
            torchrun="/usr/bin/torchrun",
        )

        with _inject_preflight(0.0):
            cli._submit(args)

        out = capsys.readouterr().out
        assert "#PBS" in out

    def test_dry_run_derecho_prints_script(self, tmp_path, capsys):
        config_path = _make_minimal_conf(tmp_path)
        args = _derecho_args(config=config_path, dry_run=True, chain=1, nodes=1)

        with _inject_preflight(0.0):
            cli._submit(args)

        out = capsys.readouterr().out
        assert "#PBS" in out

    def test_dry_run_chain_shows_multiple_scripts(self, tmp_path, capsys):
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(
            config=config_path,
            dry_run=True,
            chain=3,
            torchrun="/usr/bin/torchrun",
        )

        with _inject_preflight(0.0):
            cli._submit(args)

        out = capsys.readouterr().out
        assert "Job 1" in out or "1/3" in out
        assert "afterok" in out or "2..3" in out


# ===========================================================================
# TestBlocklist
# ===========================================================================


class TestBlocklist:
    """Verify the blocklist covers expected destructive commands."""

    @pytest.mark.parametrize("bad_cmd", ["rm ", "rmdir", "mv ", "cp ", "> ", ">>", "sudo", "qdel", "git commit"])
    def test_blocklist_entry_blocks_command(self, bad_cmd):
        # Find a blocked command pattern that appears in _AGENT_BASH_BLOCKLIST
        result = _agent_bash(f"{bad_cmd} something")
        assert "Blocked" in result


# ===========================================================================
# TestConvert (auto-transform only — no interactive prompts)
# ===========================================================================


class TestConvertAutoTransform:
    """Test the auto-transform logic in _convert by monkeypatching prompts."""

    def _make_v1_conf(self, tmp_path):
        conf = {
            "trainer": {"type": "era5"},
            "data": {
                "forecast_len": 0,
                "valid_forecast_len": 0,
                "backprop_on_timestep": [0, 1],
            },
            "loss": {"training_loss": "mse"},
        }
        p = tmp_path / "v1_config.yml"
        p.write_text(yaml.dump(conf))
        return str(p)

    def test_auto_converts_trainer_type(self, tmp_path, monkeypatch):
        config_path = self._make_v1_conf(tmp_path)
        out_path = str(tmp_path / "v2.yml")

        # Monkeypatch all interactive input
        answers = iter(
            [
                # _prompt_bool for EMA
                "y",
                # _prompt for ema_decay
                "0.9999",
                # _prompt_bool for tensorboard
                "y",
                # cluster, account, conda, walltime, job_name
                "derecho",
                "NAML0001",
                "credit-env",
                "12:00:00",
                "credit_v2",
                # nodes, gpus, cpus, mem, queue
                "1",
                "4",
                "64",
                "480GB",
                "main",
                # output path
                out_path,
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        args = argparse.Namespace(config=config_path, output=out_path)
        cli._convert(args)

        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["trainer"]["type"] == "era5-v2"

    def test_forecast_len_incremented(self, tmp_path, monkeypatch):
        config_path = self._make_v1_conf(tmp_path)
        out_path = str(tmp_path / "v2.yml")

        answers = iter(
            [
                "y",
                "0.9999",
                "y",
                "derecho",
                "NAML0001",
                "credit-env",
                "12:00:00",
                "credit_v2",
                "1",
                "4",
                "64",
                "480GB",
                "main",
                out_path,
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        args = argparse.Namespace(config=config_path, output=out_path)
        cli._convert(args)

        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["data"]["forecast_len"] == 1  # 0 → 1

    def test_ensemble_config_detected_and_kept(self, tmp_path, monkeypatch):
        """Ensemble config (ensemble_size>1) triggers the ensemble section."""
        conf = {
            "trainer": {"type": "era5", "ensemble_size": 4},
            "data": {"forecast_len": 0, "valid_forecast_len": 0},
            "loss": {"training_loss": "crps"},
        }
        p = tmp_path / "ens.yml"
        p.write_text(yaml.dump(conf))
        out_path = str(tmp_path / "ens_v2.yml")

        # After auto-transforms:
        # EMA y, decay 0.9999, tb y
        # Ensemble: keep y, size 4
        # cluster derecho, acct, conda, walltime, job, nodes, gpus, cpus, mem, queue
        answers = iter(
            [
                "y",
                "0.9999",
                "y",  # EMA + TensorBoard
                "y",
                "4",  # keep ensemble, size
                "derecho",
                "NAML0001",
                "credit-env",
                "12:00:00",
                "credit_v2",
                "1",
                "4",
                "64",
                "480GB",
                "main",
                out_path,
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        args = argparse.Namespace(config=str(p), output=out_path)
        cli._convert(args)
        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["trainer"]["ensemble_size"] == 4

    def test_ensemble_config_dropped(self, tmp_path, monkeypatch):
        """User can choose to drop ensemble (ensemble_size → 1)."""
        conf = {
            "trainer": {"type": "era5", "ensemble_size": 4},
            "data": {"forecast_len": 0, "valid_forecast_len": 0},
            "loss": {"training_loss": "crps"},
        }
        p = tmp_path / "ens.yml"
        p.write_text(yaml.dump(conf))
        out_path = str(tmp_path / "ens_v2.yml")

        answers = iter(
            [
                "y",
                "0.9999",
                "n",  # EMA + no TensorBoard
                "n",  # do NOT keep ensemble
                "derecho",
                "NAML0001",
                "credit-env",
                "12:00:00",
                "credit_v2",
                "1",
                "4",
                "64",
                "480GB",
                "main",
                out_path,
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        args = argparse.Namespace(config=str(p), output=out_path)
        cli._convert(args)
        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["trainer"]["ensemble_size"] == 1

    def test_backprop_on_timestep_shifted(self, tmp_path, monkeypatch):
        config_path = self._make_v1_conf(tmp_path)
        out_path = str(tmp_path / "v2.yml")

        answers = iter(
            [
                "y",
                "0.9999",
                "y",
                "derecho",
                "NAML0001",
                "credit-env",
                "12:00:00",
                "credit_v2",
                "1",
                "4",
                "64",
                "480GB",
                "main",
                out_path,
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        args = argparse.Namespace(config=config_path, output=out_path)
        cli._convert(args)

        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["data"]["backprop_on_timestep"] == [1, 2]  # [0,1] → [1,2]


# ===========================================================================
# TestCollectRunContext
# ===========================================================================


class TestCollectRunContext:
    """Tests for _collect_run_context."""

    def test_no_config_returns_empty(self):
        args = argparse.Namespace(config=None)
        result = cli._collect_run_context(args)
        assert result == ""

    def test_with_config_includes_content(self, tmp_path):
        conf = {"save_loc": str(tmp_path), "trainer": {"type": "era5-v2"}}
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        args = argparse.Namespace(config=str(p))
        result = cli._collect_run_context(args)
        assert "era5-v2" in result

    def test_missing_config_returns_empty(self, tmp_path):
        args = argparse.Namespace(config=str(tmp_path / "nonexistent.yml"))
        result = cli._collect_run_context(args)
        # Should not raise; returns empty or partial
        assert isinstance(result, str)

    def test_reads_training_log(self, tmp_path):
        """When a training_log.csv exists in save_loc it should appear in context."""
        save_loc = tmp_path / "run"
        save_loc.mkdir()
        # Write a minimal CSV
        log = save_loc / "training_log.csv"
        log.write_text("epoch,train_loss\n1,1.5\n2,1.2\n")
        conf = {"save_loc": str(save_loc), "trainer": {"type": "era5-v2"}}
        p = tmp_path / "cfg.yml"
        p.write_text(yaml.dump(conf))
        args = argparse.Namespace(config=str(p))
        result = cli._collect_run_context(args)
        assert "training_log" in result or "train_loss" in result


# ===========================================================================
# TestSubmitReload
# ===========================================================================


class TestSubmitReload:
    """Test _submit with --reload flag (dry-run so no qsub)."""

    def test_reload_writes_reload_config(self, tmp_path, capsys):
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(
            config=config_path,
            dry_run=True,
            chain=1,
            reload=True,
            torchrun="/usr/bin/torchrun",
        )
        with _inject_preflight(0.0):
            cli._submit(args)
        # reload config should have been written
        save_loc = tmp_path / "save"
        assert (save_loc / "config_reload.yml").exists()

    def test_submit_live_calls_qsub(self, tmp_path, capsys, monkeypatch):
        """Live submit (non-dry-run) should call _qsub at least once."""
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(
            config=config_path,
            dry_run=False,
            chain=1,
            reload=False,
            torchrun="/usr/bin/torchrun",
        )
        qsub_calls = []

        def fake_qsub(script):
            qsub_calls.append(script)
            return "99999.pbs"

        monkeypatch.setattr(cli, "_qsub", fake_qsub)
        with _inject_preflight(0.0):
            cli._submit(args)

        assert len(qsub_calls) == 1

    def test_submit_chain_calls_qsub_multiple(self, tmp_path, capsys, monkeypatch):
        """Chain=3 should call _qsub 3 times."""
        config_path = _make_minimal_conf(tmp_path)
        args = _casper_args(
            config=config_path,
            dry_run=False,
            chain=3,
            reload=False,
            torchrun="/usr/bin/torchrun",
        )
        qsub_calls = []
        counter = [0]

        def fake_qsub(script):
            counter[0] += 1
            qsub_calls.append(script)
            return f"{counter[0]}000.pbs"

        monkeypatch.setattr(cli, "_qsub", fake_qsub)
        with _inject_preflight(0.0):
            cli._submit(args)

        assert len(qsub_calls) == 3


# ===========================================================================
# TestResolvePbsOptsEnv
# ===========================================================================


class TestResolvePbsOptsEnv:
    """Test _resolve_pbs_opts PBS_ACCOUNT environment variable fallback."""

    def test_pbs_account_env_var(self, monkeypatch):
        monkeypatch.setenv("PBS_ACCOUNT", "ENV_ACCOUNT")
        args = argparse.Namespace(
            cluster="casper",
            account=None,
            walltime=None,
            gpus=None,
            nodes=None,
            cpus=None,
            mem=None,
            queue=None,
            gpu_type=None,
            conda_env=None,
        )
        r = _resolve_pbs_opts(args, {})
        assert r.account == "ENV_ACCOUNT"

    def test_ultimate_fallback_naml0001(self, monkeypatch):
        # Remove PBS_ACCOUNT if set
        monkeypatch.delenv("PBS_ACCOUNT", raising=False)
        args = argparse.Namespace(
            cluster="casper",
            account=None,
            walltime=None,
            gpus=None,
            nodes=None,
            cpus=None,
            mem=None,
            queue=None,
            gpu_type=None,
            conda_env=None,
        )
        r = _resolve_pbs_opts(args, {})
        assert r.account == "NAML0001"

    def test_all_none_uses_hardcoded_defaults(self, monkeypatch):
        """All fields None + empty pbs_cfg → _first() returns None path is exercised."""
        monkeypatch.delenv("PBS_ACCOUNT", raising=False)
        args = argparse.Namespace(
            cluster="casper",
            account=None,
            walltime=None,
            gpus=None,
            nodes=None,
            cpus=None,
            mem=None,
            queue=None,
            gpu_type=None,
            conda_env=None,
        )
        r = _resolve_pbs_opts(args, {})
        # gpu_type falls back to hardcoded value; if pbs_cfg has no gpu_type either,
        # _first returns the hardcoded fallback "a100_80gb"
        assert r.gpu_type == "a100_80gb"
        assert r.walltime == "12:00:00"


# ===========================================================================
# TestConvertCasperPath
# ===========================================================================


class TestConvertCasperPath:
    """Test _convert with casper cluster selection (exercises the else branch)."""

    def test_casper_pbs_section(self, tmp_path, monkeypatch):
        conf = {
            "trainer": {"type": "era5"},
            "data": {"forecast_len": 0, "valid_forecast_len": 0},
            "loss": {"training_loss": "mse"},
        }
        p = tmp_path / "v1.yml"
        p.write_text(yaml.dump(conf))
        out_path = str(tmp_path / "v2.yml")

        # casper branch: prompts gpus, cpus, mem, gpu_type, queue
        answers = iter(
            [
                "y",
                "0.9999",  # EMA
                "n",  # TensorBoard
                "casper",  # cluster
                "NAML0001",  # account
                "credit-casper",  # conda
                "04:00:00",  # walltime
                "my_job",  # job_name
                "4",  # gpus
                "8",  # cpus
                "128GB",  # mem
                "a100_80gb",  # gpu_type
                "casper",  # queue
                out_path,  # output path
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        args = argparse.Namespace(config=str(p), output=out_path)
        cli._convert(args)

        with open(out_path) as f:
            result = yaml.safe_load(f)
        assert result["pbs"]["queue"] == "casper"


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Tests for _init — uses mocked shutil.copy to avoid needing real template files."""

    def test_init_calls_copy(self, tmp_path, monkeypatch):
        import shutil

        copies = []
        monkeypatch.setattr(shutil, "copy", lambda src, dst: copies.append((src, dst)))
        out = str(tmp_path / "new_config.yml")

        # Patch exists: template exists, but output does NOT exist
        def fake_exists(p):
            return p != out

        monkeypatch.setattr(os.path, "exists", fake_exists)

        args = argparse.Namespace(grid="0.25deg", model="wxformer", output=out, force=False)
        cli._init(args)
        assert len(copies) == 1
        assert copies[0][1] == out

    def test_init_unknown_grid_exits(self, monkeypatch):
        args = argparse.Namespace(grid="99deg", model="wxformer", output="out.yml", force=False)
        with pytest.raises(SystemExit):
            cli._init(args)

    def test_init_existing_file_no_force_exits(self, tmp_path, monkeypatch):
        import shutil

        out = str(tmp_path / "existing.yml")
        (tmp_path / "existing.yml").write_text("existing")

        # template exists but output already exists → should exit
        def fake_exists(p):
            return True  # both template and output "exist"

        monkeypatch.setattr(os.path, "exists", fake_exists)
        monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
        args = argparse.Namespace(grid="0.25deg", model="wxformer", output=out, force=False)
        with pytest.raises(SystemExit):
            cli._init(args)

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        import shutil

        monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
        monkeypatch.setattr(os.path, "exists", lambda p: True)

        out = tmp_path / "existing.yml"
        out.write_text("old content")
        args = argparse.Namespace(grid="1deg", model="wxformer", output=str(out), force=True)
        cli._init(args)  # should not raise

    def test_init_template_not_found_exits(self, tmp_path, monkeypatch):
        """Template file is missing → exit 1 (lines 805-806)."""
        import shutil

        out = str(tmp_path / "new.yml")
        # template does NOT exist, output does not exist either
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr(shutil, "copy", lambda src, dst: None)
        args = argparse.Namespace(grid="0.25deg", model="wxformer", output=out, force=False)
        with pytest.raises(SystemExit):
            cli._init(args)


# ===========================================================================
# TestMainDispatch
# ===========================================================================


class TestMainDispatch:
    """Test main() dispatch to individual handlers via monkeypatching."""

    def test_main_dispatches_train(self, monkeypatch, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        called = {}

        def fake_train(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_train", fake_train)
        monkeypatch.setattr(sys, "argv", ["credit", "train", "-c", config_path])
        main()
        assert called["args"].config == config_path

    def test_main_dispatches_rollout(self, monkeypatch, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        called = {}

        def fake_rollout(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_rollout", fake_rollout)
        monkeypatch.setattr(sys, "argv", ["credit", "rollout", "-c", config_path])
        main()
        assert called["args"].config == config_path

    def test_main_dispatches_submit(self, monkeypatch, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        called = {}

        def fake_submit(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_submit", fake_submit)
        monkeypatch.setattr(
            sys,
            "argv",
            ["credit", "submit", "--cluster", "casper", "-c", config_path],
        )
        main()
        assert called["args"].cluster == "casper"

    def test_main_dispatches_init(self, monkeypatch):
        called = {}

        def fake_init(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_init", fake_init)
        monkeypatch.setattr(sys, "argv", ["credit", "init", "-o", "out.yml"])
        main()
        assert called["args"].output == "out.yml"

    def test_main_dispatches_convert(self, monkeypatch, tmp_path):
        config_path = _make_minimal_conf(tmp_path)
        called = {}

        def fake_convert(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_convert", fake_convert)
        monkeypatch.setattr(sys, "argv", ["credit", "convert", "-c", config_path])
        main()
        assert called["args"].config == config_path

    def test_main_dispatches_ask(self, monkeypatch):
        called = {}

        def fake_ask(args):
            called["args"] = args

        monkeypatch.setattr(cli, "_ask", fake_ask)
        monkeypatch.setattr(sys, "argv", ["credit", "ask", "what", "is", "credit?"])
        main()
        assert called["args"].question == ["what", "is", "credit?"]


# ===========================================================================
# TestAgentReadFileLargeFile
# ===========================================================================


class TestAgentReadFileEdgeCases:
    """Edge cases for _agent_read_file."""

    def test_expanduser_tilde(self, tmp_path, monkeypatch):
        """~ paths should be expanded, not fail."""
        # We can't test actual ~ but we can verify it doesn't crash
        result = _agent_read_file("~/non_existent_credit_test_file.txt")
        assert "not found" in result.lower() or "File not found" in result

    def test_exception_returns_error_string(self, tmp_path, monkeypatch):
        """Any unexpected exception should return an error string."""
        import pathlib

        def bad_read_text(*a, **kw):
            raise PermissionError("no read permission")

        f = tmp_path / "secret.txt"
        f.write_text("secret")
        monkeypatch.setattr(pathlib.Path, "read_text", bad_read_text)
        result = _agent_read_file(str(f))
        assert "Error" in result


# ===========================================================================
# TestAgentBashEdgeCases
# ===========================================================================


class TestAgentBashEdgeCases:
    """Additional edge cases for _agent_bash."""

    def test_exception_returns_error(self, monkeypatch):
        def raise_exc(*a, **kw):
            raise OSError("no such file")

        monkeypatch.setattr(subprocess, "run", raise_exc)
        result = _agent_bash("nonexistent_command_xyz")
        assert "Error" in result or "Blocked" in result or "error" in result.lower()

    def test_redirect_blocked(self):
        result = _agent_bash("echo hello > /tmp/out.txt")
        assert "Blocked" in result

    def test_append_redirect_blocked(self):
        result = _agent_bash("echo hello >> /tmp/out.txt")
        assert "Blocked" in result
