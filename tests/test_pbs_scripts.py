"""Tests for credit submit PBS script generation.

Every test uses --dry-run-style generation via _build_pbs_script directly,
so no qsub is called and no cluster is needed.

Key invariants verified:
  - Casper always uses torchrun --standalone, never mpiexec
  - Derecho single-node (--nodes 1) uses torchrun --standalone, never mpiexec
  - Derecho multi-node uses mpiexec + c10d rendezvous, never --standalone
  - afterok dependency line is present iff depend_on is provided
  - GPU count, config path, and account appear correctly in all scripts
"""

import argparse
import pytest

from credit.cli import _build_pbs_script


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _casper_args(gpus=4, walltime="12:00:00", **kw):
    defaults = dict(
        cluster="casper", gpus=gpus, nodes=1, cpus=None, mem=None,
        walltime=walltime, queue=None, gpu_type=None, torchrun=None,
        conda_env=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _derecho_args(nodes=1, gpus=4, walltime="12:00:00", **kw):
    defaults = dict(
        cluster="derecho", nodes=nodes, gpus=gpus, cpus=None, mem=None,
        walltime=walltime, queue=None, conda_env=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


FAKE_CONFIG = "/glade/work/user/my_run/config.yml"
FAKE_REPO   = "/glade/work/user/miles-credit"
FAKE_ACCOUNT = "NAML0001"


def _casper_script(depend_on=None, **kw):
    return _build_pbs_script(
        _casper_args(**kw), FAKE_CONFIG, FAKE_REPO, FAKE_ACCOUNT,
        depend_on=depend_on,
    )


def _derecho_script(nodes=1, depend_on=None, **kw):
    return _build_pbs_script(
        _derecho_args(nodes=nodes, **kw), FAKE_CONFIG, FAKE_REPO, FAKE_ACCOUNT,
        depend_on=depend_on,
    )


# ---------------------------------------------------------------------------
# Casper
# ---------------------------------------------------------------------------

class TestCasperScript:
    def test_has_standalone(self):
        assert "--standalone" in _casper_script()

    def test_no_mpiexec(self):
        assert "mpiexec" not in _casper_script()

    def test_no_rdzv_backend(self):
        assert "rdzv-backend" not in _casper_script()

    def test_gpu_count_correct(self):
        script = _casper_script(gpus=2)
        assert "ngpus=2" in script
        # Casper passes GPU count via $NGPUS bash variable, not a literal
        assert "NGPUS=2" in script
        assert "nproc-per-node=${NGPUS}" in script

    def test_config_path_in_script(self):
        assert FAKE_CONFIG in _casper_script()

    def test_account_in_header(self):
        assert f"#PBS -A {FAKE_ACCOUNT}" in _casper_script()

    def test_walltime_in_header(self):
        assert "#PBS -l walltime=06:00:00" in _casper_script(walltime="06:00:00")

    def test_depends_line_present_when_provided(self):
        script = _casper_script(depend_on="12345.casper-pbs")
        assert "#PBS -W depend=afterok:12345.casper-pbs" in script

    def test_depends_line_absent_when_none(self):
        assert "depend=afterok" not in _casper_script(depend_on=None)

    def test_pythonnousersite_set(self):
        assert "PYTHONNOUSERSITE" in _casper_script()

    def test_pytorch_cuda_alloc_conf_set(self):
        assert "PYTORCH_CUDA_ALLOC_CONF" in _casper_script()


# ---------------------------------------------------------------------------
# Derecho — single node
# ---------------------------------------------------------------------------

class TestDerechoSingleNode:
    def test_has_standalone(self):
        assert "--standalone" in _derecho_script(nodes=1)

    def test_no_mpiexec(self):
        assert "mpiexec" not in _derecho_script(nodes=1)

    def test_no_rdzv_backend(self):
        assert "rdzv-backend" not in _derecho_script(nodes=1)

    def test_nnodes_is_1(self):
        assert "--nnodes=1" in _derecho_script(nodes=1)

    def test_gpu_count_correct(self):
        script = _derecho_script(nodes=1, gpus=4)
        assert "ngpus=4" in script
        assert "nproc-per-node=4" in script

    def test_config_path_in_script(self):
        assert FAKE_CONFIG in _derecho_script(nodes=1)

    def test_account_in_header(self):
        assert f"#PBS -A {FAKE_ACCOUNT}" in _derecho_script(nodes=1)

    def test_depends_line_present_when_provided(self):
        script = _derecho_script(nodes=1, depend_on="99999.casper-pbs")
        assert "#PBS -W depend=afterok:99999.casper-pbs" in script

    def test_depends_line_absent_when_none(self):
        assert "depend=afterok" not in _derecho_script(nodes=1, depend_on=None)

    def test_ncarenv_module_loaded(self):
        assert "ncarenv" in _derecho_script(nodes=1)

    def test_nccl_env_vars_set(self):
        script = _derecho_script(nodes=1)
        assert "NCCL_SOCKET_IFNAME" in script
        assert "NCCL_NET" in script


# ---------------------------------------------------------------------------
# Derecho — multi-node
# ---------------------------------------------------------------------------

class TestDerechoMultiNode:
    def test_has_mpiexec(self):
        assert "mpiexec" in _derecho_script(nodes=4)

    def test_no_standalone(self):
        assert "--standalone" not in _derecho_script(nodes=4)

    def test_has_rdzv_backend_c10d(self):
        assert "--rdzv-backend=c10d" in _derecho_script(nodes=4)

    def test_has_rdzv_endpoint(self):
        assert "--rdzv-endpoint=" in _derecho_script(nodes=4)

    def test_nnodes_correct(self):
        assert "--nnodes=4" in _derecho_script(nodes=4)

    def test_ppn_matches_gpus(self):
        assert "--ppn 2" in _derecho_script(nodes=4, gpus=2)

    def test_select_line_correct(self):
        script = _derecho_script(nodes=4, gpus=4)
        assert "select=4:ncpus=" in script

    def test_depends_line_chained(self):
        script = _derecho_script(nodes=4, depend_on="5555.casper-pbs")
        assert "#PBS -W depend=afterok:5555.casper-pbs" in script

    def test_head_node_ip_lookup(self):
        # Multi-node script should SSH to find head node IP
        assert "hostname -i" in _derecho_script(nodes=4)


# ---------------------------------------------------------------------------
# _write_reload_config
# ---------------------------------------------------------------------------

class TestWriteReloadConfig:
    def test_five_fields_set(self, tmp_path):
        import yaml
        from credit.cli import _write_reload_config

        config = {
            "save_loc": str(tmp_path),
            "trainer": {"load_weights": False, "epochs": 70},
        }
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump(config))

        reload_path = _write_reload_config(str(cfg_path))

        with open(reload_path) as f:
            reloaded = yaml.safe_load(f)

        assert reloaded["trainer"]["load_weights"]   is True
        assert reloaded["trainer"]["load_optimizer"] is True
        assert reloaded["trainer"]["load_scaler"]    is True
        assert reloaded["trainer"]["load_scheduler"] is True
        assert reloaded["trainer"]["reload_epoch"]   is True

    def test_other_fields_preserved(self, tmp_path):
        import yaml
        from credit.cli import _write_reload_config

        config = {"save_loc": str(tmp_path), "trainer": {"epochs": 42, "amp": False}}
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump(config))
        reload_path = _write_reload_config(str(cfg_path))

        with open(reload_path) as f:
            reloaded = yaml.safe_load(f)
        assert reloaded["trainer"]["epochs"] == 42
        assert reloaded["trainer"]["amp"] is False

    def test_written_to_save_loc(self, tmp_path):
        import yaml
        from credit.cli import _write_reload_config

        config = {"save_loc": str(tmp_path), "trainer": {}}
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump(config))
        reload_path = _write_reload_config(str(cfg_path))

        assert reload_path == str(tmp_path / "config_reload.yml")
        assert (tmp_path / "config_reload.yml").exists()


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_load_scheduler_creates_linear_warmup_cosine(self):
        import torch
        from credit.scheduler import load_scheduler, LinearWarmupCosineScheduler

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        conf = {
            "trainer": {
                "use_scheduler": True,
                "scheduler": {
                    "scheduler_type": "linear-warmup-cosine",
                    "warmup_steps": 100,
                    "total_steps": 1000,
                    "min_lr": 1e-5,
                },
            }
        }
        scheduler = load_scheduler(optimizer, conf)
        assert isinstance(scheduler, LinearWarmupCosineScheduler)

    def test_linear_warmup_cosine_in_update_on_batch(self):
        from credit.scheduler import update_on_batch
        assert "linear-warmup-cosine" in update_on_batch

    def test_linear_warmup_cosine_not_in_update_on_epoch(self):
        from credit.scheduler import update_on_epoch
        assert "linear-warmup-cosine" not in update_on_epoch


# ---------------------------------------------------------------------------
# Channel map / denorm alignment
# ---------------------------------------------------------------------------

class TestChannelAlignment:
    """_build_channel_map and _build_denorm_stats must agree on C_out."""

    def _conf(self, vars_3d, vars_2d, diag_2d, n_levels=5):
        return {
            "data": {
                "source": {"ERA5": {
                    "level_coord": "level",
                    "levels": list(range(n_levels)),
                    "variables": {
                        "prognostic": {"vars_3D": vars_3d, "vars_2D": vars_2d},
                        "diagnostic": {"vars_2D": diag_2d},
                    },
                }},
                "mean_path": "/fake/mean.nc",
                "std_path":  "/fake/std.nc",
            }
        }

    def _patch_xr(self, monkeypatch, conf):
        import numpy as np
        import xarray as xr
        src = conf["data"]["source"]["ERA5"]
        n = len(src["levels"])
        v = src["variables"]
        all_3d = (v.get("prognostic") or {}).get("vars_3D", [])
        all_2d = list((v.get("prognostic") or {}).get("vars_2D", [])) + \
                 list((v.get("diagnostic") or {}).get("vars_2D", []))
        ds_vars = {vn: xr.DataArray(np.ones(n), dims=["level"],
                                    coords={"level": src["levels"]}) for vn in all_3d}
        ds_vars.update({vn: xr.DataArray(np.float32(1.0)) for vn in all_2d})
        ds = xr.Dataset(ds_vars)
        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: ds)

    def test_lengths_match_simple(self, monkeypatch):
        from credit.cli import _build_channel_map, _build_denorm_stats
        conf = self._conf(["U", "V"], ["SP"], [], n_levels=3)
        self._patch_xr(monkeypatch, conf)
        cm = _build_channel_map(conf)
        m, s = _build_denorm_stats(conf)
        total_from_map = sum(len(v) for v in cm.values())
        assert len(m) == total_from_map
        assert len(s) == total_from_map

    def test_lengths_match_with_diagnostics(self, monkeypatch):
        from credit.cli import _build_channel_map, _build_denorm_stats
        conf = self._conf(["T", "Q"], ["SP", "VAR_2T"], ["precip", "evap"], n_levels=4)
        self._patch_xr(monkeypatch, conf)
        cm = _build_channel_map(conf)
        m, s = _build_denorm_stats(conf)
        total_from_map = sum(len(v) for v in cm.values())
        assert len(m) == total_from_map == 4*2 + 2 + 2  # 12

    def test_channel_indices_are_contiguous(self):
        from credit.cli import _build_channel_map
        conf = self._conf(["U", "V", "T"], ["SP", "VAR_2T"], ["precip"], n_levels=3)
        cm = _build_channel_map(conf)
        all_idx = sorted(c for chans in cm.values() for c in chans)
        assert all_idx == list(range(len(all_idx))), "Channel indices must be contiguous 0..N-1"


# ---------------------------------------------------------------------------
# credit init template existence
# ---------------------------------------------------------------------------

class TestInitTemplates:
    """Every template referenced by _init must exist on disk."""

    def test_1deg_template_exists(self):
        import os
        from credit.cli import _repo_root
        path = os.path.join(_repo_root(), "config", "wxformer_1dg_6hr_v2.yml")
        assert os.path.exists(path), f"Template missing: {path}"

    def test_025deg_template_exists(self):
        import os
        from credit.cli import _repo_root
        path = os.path.join(_repo_root(), "config", "wxformer_025deg_6hr_v2.yml")
        assert os.path.exists(path), f"Template missing: {path}"

    def test_starter_template_exists(self):
        import os
        from credit.cli import _repo_root
        path = os.path.join(_repo_root(), "config", "starter_v2.yml")
        assert os.path.exists(path), f"Template missing: {path}"

    def test_templates_are_valid_yaml(self):
        import os, yaml
        from credit.cli import _repo_root
        repo = _repo_root()
        for name in ["wxformer_1dg_6hr_v2.yml", "wxformer_025deg_6hr_v2.yml", "starter_v2.yml"]:
            path = os.path.join(repo, "config", name)
            if os.path.exists(path):
                with open(path) as f:
                    conf = yaml.safe_load(f)
                assert isinstance(conf, dict), f"{name} did not parse to a dict"
                assert "trainer" in conf, f"{name} missing 'trainer' key"
                assert "data" in conf, f"{name} missing 'data' key"
