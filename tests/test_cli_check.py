"""Unit tests for ``credit check`` (credit/cli/_check.py).

Every test builds a config dict in memory and runs the checks against it — no
data files, no network, no GPU.  The baseline config below is deliberately
clean, so each test can introduce exactly one defect and assert that it is
found (and, just as importantly, that a clean config stays silent).
"""

import argparse
import copy
import json

import pytest
import yaml

from credit.cli import _Report, _check, _run_checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_conf(save_loc):
    """A minimal but *valid* gen2 config: 2 levels, 1 source, BaseLoss + twin."""
    scaler = str(save_loc / "scaler.json")
    return {
        "save_loc": str(save_loc),
        "data": {
            "source": {
                "ERA5": {
                    "dataset_type": "local",
                    "level_coord": "level",
                    "levels": [1, 2],
                    "variables": {
                        "prognostic": {"vars_3D": ["T", "Q"], "vars_2D": ["SP"]},
                        "diagnostic": {"vars_2D": ["TP"]},
                        "dynamic_forcing": {"vars_2D": ["tisr"]},
                        "static": {"vars_2D": ["lsm"]},
                    },
                }
            },
            "start_datetime": "2000-01-01",
            "end_datetime": "2000-12-31",
            "timestep": "6h",
            "history_len": 1,
            "forecast_len": 1,
        },
        "trainer": {
            "type": "gen2",
            "parallelism": {"data": "fsdp2", "tensor": 1, "domain": 1},
            "use_scheduler": True,
            "scheduler": {
                "scheduler_type": "linear-warmup-cosine",
                "warmup_steps": 10,
                "total_steps": 100,
                "min_lr": 1.0e-5,
            },
        },
        "model": {
            "type": "wxformer",
            "frames": 1,
            "image_height": 64,
            "image_width": 64,
            "levels": 2,
            "channels": 2,  # T, Q
            "surface_channels": 1,  # SP
            "input_only_channels": 2,  # tisr + lsm
            "output_only_channels": 1,  # TP
            # Statically valid: 4 stages, and 64x64 divides by the lcm the
            # strides and window sizes require.  (Not tuned to be *buildable* —
            # these tests exercise the static checks, not model construction.)
            "dim": [32, 64, 128, 256],
            "depth": [1, 1, 1, 1],
            "cross_embed_strides": [2, 2, 2, 2],
            "cross_embed_kernel_sizes": [[4, 8], [4, 4], [4, 4], [4, 4]],
            "global_window_size": [4, 2, 1, 1],
            "local_window_size": 4,
        },
        "preblocks": {"per_step": {"concat": {"type": "concat"}}},
        "postblocks": {
            "per_step": {
                "reconstruct": {"type": "reconstruct", "args": {"detach": False}},
                "scaler": {
                    "type": "bridgescaler_transform",
                    "args": {"scaler_path": scaler, "variables": [], "method": "inverse_transform"},
                },
                "reconstruct_target": {
                    "type": "reconstruct",
                    "args": {"in_key": "y", "out_key": "y_target_processed"},
                },
                "scaler_target": {
                    "type": "bridgescaler_transform",
                    "args": {
                        "scaler_path": scaler,
                        "variables": [],
                        "method": "inverse_transform",
                        "key": "y_target_processed",
                    },
                },
            }
        },
        "loss": {"type": "base", "args": {"training_loss": "mse", "scaler_path": scaler}},
    }


def _run(conf, deep=False):
    rep = _Report("test.yml")
    _run_checks(conf, rep, deep=deep)
    return rep


def _wheres(rep, severity="error"):
    return {f.where for f in rep.findings if f.severity == severity}


def _text(rep, severity="error"):
    """All rendered text for a severity — message *and* fix, as the user sees it."""
    return "\n".join(f"{f.message}\n{f.fix or ''}" for f in rep.findings if f.severity == severity)


@pytest.fixture
def conf(tmp_path):
    return _base_conf(tmp_path)


# ===========================================================================
# Baseline
# ===========================================================================


def test_clean_config_reports_no_errors(conf):
    """The whole point: a valid config must stay silent."""
    rep = _run(conf)
    assert _wheres(rep) == set(), _text(rep)
    assert rep.exit_code() == 0


def test_missing_scaler_is_a_warning_not_an_error(conf):
    """A not-yet-fitted scaler is expected before `credit preprocess`."""
    rep = _run(conf)
    assert any("scaler_path" in w for w in _wheres(rep, "warning"))
    assert rep.exit_code() == 0
    assert rep.exit_code(strict=True) == 1


# ===========================================================================
# Structure and registries
# ===========================================================================


def test_missing_required_section(conf):
    del conf["model"]
    assert "model" in _wheres(_run(conf))


def test_unknown_top_level_key_warns(conf):
    conf["modle"] = {}
    rep = _run(conf)
    assert "modle" in _wheres(rep, "warning")


def test_unknown_model_type_suggests_nearest(conf):
    conf["model"]["type"] = "wxformr"
    rep = _run(conf)
    assert "model.type" in _wheres(rep)
    assert "wxformer" in _text(rep)


def test_unknown_dataset_type_suggests_nearest(conf):
    conf["data"]["source"]["ERA5"]["dataset_type"] = "arco_era"
    rep = _run(conf)
    assert "data.source.ERA5.dataset_type" in _wheres(rep)
    assert "arco_era5" in _text(rep)


def test_unknown_postblock_type_suggests_nearest(conf):
    conf["postblocks"]["per_step"]["x"] = {"type": "mslp_diagnstic"}
    rep = _run(conf)
    assert "postblocks.per_step.x" in _wheres(rep)
    assert "mslp_diagnostic" in _text(rep)


def test_unknown_block_section_rejected(conf):
    conf["postblocks"]["every_step"] = {}
    assert "postblocks.every_step" in _wheres(_run(conf))


def test_data_only_fragment_is_noted_not_error_spammed(conf):
    """config/gen_2/examples/multi_source_data.yml documents data only, by design."""
    for key in ("model", "trainer", "loss", "postblocks"):
        conf.pop(key, None)
    rep = _run(conf)
    assert _wheres(rep) == set(), _text(rep)
    assert "config" in {f.where for f in rep.findings if f.severity == "info"}


def test_incomplete_config_with_a_trainer_still_errors(conf):
    """Having a trainer but no model is an incomplete config, not a fragment."""
    del conf["model"]
    assert "model" in _wheres(_run(conf))


def test_gen1_config_short_circuits(conf):
    conf["trainer"]["type"] = "era5"
    rep = _run(conf)
    assert "trainer.type" in _wheres(rep, "warning")
    assert _wheres(rep) == set()  # gen2-only checks skipped, not misreported


# ===========================================================================
# Block argument binding — the check that catches misplaced args
# ===========================================================================


def test_postblock_scaler_rejects_preblock_only_args(conf):
    """scaler_type/scaler_params are preblock-only; the postblock has no such kwargs."""
    conf["postblocks"]["per_step"]["scaler"]["args"]["scaler_type"] = "standard"
    rep = _run(conf)
    assert "postblocks.per_step.scaler" in _wheres(rep)
    assert "scaler_type" in _text(rep)


def test_preblock_scaler_accepts_scaler_params(conf):
    """The same keys ARE valid on the preblock — no false positive."""
    conf["preblocks"]["per_step"]["norm"] = {
        "type": "bridgescaler_transform",
        "args": {
            "scaler_path": str(conf["save_loc"] + "/scaler.json"),
            "variables": [],
            "scaler_type": "standard",
            "scaler_params": {"channels_last": False},
            "method": "transform",
        },
    }
    assert "preblocks.per_step.norm" not in _wheres(_run(conf))


def test_block_without_type(conf):
    conf["postblocks"]["per_step"]["x"] = {"args": {}}
    assert "postblocks.per_step.x" in _wheres(_run(conf))


# ===========================================================================
# Data source checks
# ===========================================================================


def test_empty_levels_list_is_an_error(conf):
    """`levels: []` selects zero levels — the key-presence trap."""
    conf["data"]["source"]["ERA5"]["levels"] = []
    rep = _run(conf)
    assert "data.source.ERA5.levels" in _wheres(rep)
    assert "ZERO" in _text(rep)


def test_unknown_field_type_suggests_nearest(conf):
    conf["data"]["source"]["ERA5"]["variables"]["prognostc"] = None
    rep = _run(conf)
    assert "data.source.ERA5.variables.prognostc" in _wheres(rep)
    assert "prognostic" in _text(rep)


def test_gen1_forecast_len_zero(conf):
    conf["data"]["forecast_len"] = 0
    rep = _run(conf)
    assert "data.forecast_len" in _wheres(rep)
    assert "gen1" in _text(rep)


def test_frames_history_len_mismatch_warns(conf):
    conf["data"]["history_len"] = 2
    assert "model.frames" in _wheres(_run(conf), "warning")


def test_partial_validation_data_rejected(conf):
    conf["validation_data"] = {"start_datetime": "2001-01-01"}
    rep = _run(conf)
    assert "validation_data" in _wheres(rep)
    assert "forecast_len" in _text(rep)


def test_complete_validation_data_accepted(conf):
    conf["validation_data"] = dict(conf["data"], start_datetime="2001-01-01")
    assert "validation_data" not in _wheres(_run(conf))


def test_data_valid_alias_warns(conf):
    """The gen2 loader reads validation_data; data_valid alone is silently ignored."""
    conf["data_valid"] = dict(conf["data"])
    assert "data_valid" in _wheres(_run(conf), "warning")


# ===========================================================================
# Model geometry and channel counts
# ===========================================================================


@pytest.mark.parametrize(
    "key,bad,expected",
    [
        ("channels", 3, 2),
        ("surface_channels", 2, 1),
        ("input_only_channels", 5, 2),
        ("output_only_channels", 0, 1),
    ],
)
def test_channel_count_mismatch(conf, key, bad, expected):
    conf["model"][key] = bad
    rep = _run(conf)
    assert f"model.{key}" in _wheres(rep)
    assert f"{key}: {expected}" in _text(rep)


@pytest.mark.parametrize("key", ["dim", "depth", "global_window_size", "cross_embed_strides"])
def test_wrong_stage_count(conf, key):
    """CrossFormer asserts len(...) == 4 with no message; catch it statically."""
    conf["model"][key] = [2, 2]
    rep = _run(conf)
    assert f"model.{key}" in _wheres(rep)
    assert "exactly 4" in _text(rep)


def test_flat_dim_breaks_the_decoder_groupnorm(conf):
    """dim[0] is used as GroupNorm num_groups against dim[-1]//8 channels."""
    conf["model"]["dim"] = [256, 256, 256, 256]
    rep = _run(conf)
    assert "model.dim" in _wheres(rep)
    assert "[32, 64, 128, 256]" in _text(rep)  # suggested pyramid


@pytest.mark.parametrize("dim", [[32, 64, 128, 256], [128, 256, 512, 1024], [256, 512, 1024, 2048]])
def test_pyramid_dims_accepted(conf, dim):
    """The shape every other config in the repo uses must not be flagged."""
    conf["model"]["dim"] = dim
    assert "model.dim" not in _wheres(_run(conf))


def test_scalar_stage_value_is_accepted(conf):
    """cast_tuple broadcasts a scalar to all 4 stages, so a scalar is fine."""
    conf["model"]["depth"] = 2
    assert "model.depth" not in _wheres(_run(conf))


def test_model_levels_mismatch(conf):
    conf["model"]["levels"] = 7
    rep = _run(conf)
    assert "model.levels" in _wheres(rep)


def test_three_dimensional_input_only_vars_count_per_level(conf):
    """A 3D static variable contributes n_levels channels, not one."""
    conf["data"]["source"]["ERA5"]["variables"]["static"]["vars_3D"] = ["soil"]
    rep = _run(conf)
    assert "model.input_only_channels" in _wheres(rep)
    assert "input_only_channels: 4" in _text(rep)  # 2 + 2 levels


def _arco_geometry(conf, **overrides):
    conf["model"].update(
        image_height=721,
        image_width=1440,
        cross_embed_strides=[2, 2, 2, 2],
        cross_embed_kernel_sizes=[[4, 8, 16, 32], [4, 4], [4, 4], [4, 4]],
        global_window_size=[8, 4, 2, 1],
        local_window_size=4,
        **overrides,
    )
    return conf


def test_indivisible_grid_reports_the_failing_stage(conf):
    """Verified against a real forward pass: 721x1440 dies at stage 2 (90x180, window 4)."""
    rep = _run(_arco_geometry(conf))
    assert "model.padding_conf" in _wheres(rep)
    text = _text(rep)
    assert "Stage 2" in text and "90x180" in text and "local_window_size = 4" in text


def test_padding_makes_grid_valid(conf):
    """768x1472 is what the arco config actually uses, and it forwards cleanly."""
    _arco_geometry(conf, padding_conf={"activate": True, "mode": "earth", "pad_lat": [23, 24], "pad_lon": [16, 16]})
    assert "model.padding_conf" not in _wheres(_run(conf))


def test_odd_padded_grid_is_accepted_when_it_floors_cleanly(conf):
    """Regression: conv striding FLOORS, so an odd 801 lands on 400 and is fine.

    Confirmed by running a real forward pass at 801x1600 with these settings.
    """
    conf["model"].update(
        image_height=721,
        image_width=1440,
        cross_embed_strides=[2, 2, 2, 2],
        cross_embed_kernel_sizes=[[2, 4, 6, 8], [2, 4], [2, 4], [2, 4]],
        global_window_size=[10, 5, 2, 1],
        local_window_size=10,
        padding_conf={"activate": True, "mode": "earth", "pad_lat": [40, 40], "pad_lon": [80, 80]},
    )
    assert "model.padding_conf" not in _wheres(_run(conf)), _text(_run(conf))


def test_suggested_padding_actually_fixes_the_geometry(conf):
    """Whatever padding the fix proposes must itself pass the check."""
    import re

    rep = _run(_arco_geometry(conf))
    fix = next(f.fix for f in rep.findings if f.where == "model.padding_conf")
    pads = {k: [int(a), int(b)] for k, a, b in re.findall(r"(pad_lat|pad_lon): \[(\d+), (\d+)\]", fix)}
    assert set(pads) == {"pad_lat", "pad_lon"}, f"fix should pad both dimensions: {fix}"
    _arco_geometry(conf, padding_conf={"activate": True, "mode": "earth", **pads})
    assert "model.padding_conf" not in _wheres(_run(conf))


# ===========================================================================
# BaseLoss postblock pipeline
# ===========================================================================


def test_missing_target_twin(conf):
    del conf["postblocks"]["per_step"]["reconstruct_target"]
    del conf["postblocks"]["per_step"]["scaler_target"]
    rep = _run(conf)
    assert "postblocks.per_step" in _wheres(rep)
    assert "y_target_processed" in _text(rep)


def test_detach_true_kills_gradients(conf):
    conf["postblocks"]["per_step"]["reconstruct"]["args"]["detach"] = True
    rep = _run(conf)
    assert "postblocks.per_step.reconstruct" in _wheres(rep)
    assert "learns nothing" in _text(rep)


def test_detach_omitted_defaults_to_true(conf):
    del conf["postblocks"]["per_step"]["reconstruct"]["args"]
    assert "postblocks.per_step.reconstruct" in _wheres(_run(conf))


def test_unit_transform_missing_from_twin(conf):
    """An exp_transform on the prediction but not the target compares log vs linear."""
    conf["postblocks"]["per_step"]["exp"] = {"type": "exp_transform", "args": {"variables": ["ERA5/prognostic/2d/SP"]}}
    rep = _run(conf)
    assert "postblocks.per_step" in _wheres(rep)
    assert "exp_transform" in _text(rep)


def test_unit_transform_mirrored_is_accepted(conf):
    for key, name in (("y_processed", "exp"), ("y_target_processed", "exp_target")):
        conf["postblocks"]["per_step"][name] = {
            "type": "exp_transform",
            "args": {"variables": ["ERA5/prognostic/2d/SP"], "key": key},
        }
    assert "postblocks.per_step" not in _wheres(_run(conf))


def test_computed_diagnostic_without_twin(conf):
    conf["postblocks"]["per_step"]["mslp"] = {"type": "mslp_diagnostic"}
    rep = _run(conf)
    assert "postblocks.per_step.mslp" in _wheres(rep)
    assert "include_computed_diagnostics" in _text(rep)


def test_computed_diagnostic_opt_out(conf):
    conf["postblocks"]["per_step"]["mslp"] = {"type": "mslp_diagnostic"}
    conf["loss"]["args"]["include_computed_diagnostics"] = False
    assert "postblocks.per_step.mslp" not in _wheres(_run(conf))


def test_inverse_variance_without_scaler_path(conf):
    del conf["loss"]["args"]["scaler_path"]
    rep = _run(conf)
    assert "loss.args.scaler_path" in _wheres(rep)


def test_latitude_weights_required_when_enabled(conf):
    conf["loss"]["args"]["use_latitude_weights"] = True
    assert "loss.args.latitude_weights" in _wheres(_run(conf))


def test_non_base_loss_skips_twin_checks(conf):
    """Only BaseLoss needs the twin; other losses must not be nagged about it."""
    conf["loss"] = {"type": "mse"}
    del conf["postblocks"]["per_step"]["reconstruct_target"]
    del conf["postblocks"]["per_step"]["scaler_target"]
    assert "postblocks.per_step" not in _wheres(_run(conf))


# ===========================================================================
# Trainer
# ===========================================================================


def test_missing_parallelism_block(conf):
    del conf["trainer"]["parallelism"]
    assert "trainer.parallelism" in _wheres(_run(conf))


def test_invalid_parallelism_mode(conf):
    conf["trainer"]["parallelism"]["data"] = "fsdp3"
    assert "trainer.parallelism" in _wheres(_run(conf))


def test_unknown_scheduler_type(conf):
    conf["trainer"]["scheduler"]["scheduler_type"] = "cosine"
    rep = _run(conf)
    assert "trainer.scheduler.scheduler_type" in _wheres(rep)
    assert "cosine-annealing" in _text(rep)


def test_misspelled_scheduler_arg(conf):
    conf["trainer"]["scheduler"]["warmup_stpes"] = 10
    rep = _run(conf)
    assert "trainer.scheduler" in _wheres(rep)
    assert "warmup_stpes" in _text(rep)


def test_num_epoch_exceeding_epochs_warns(conf):
    conf["trainer"]["epochs"] = 2
    conf["trainer"]["num_epoch"] = 5
    assert "trainer.num_epoch" in _wheres(_run(conf), "warning")


# ===========================================================================
# Channel schema drift
# ===========================================================================


def test_stale_saved_schema_detected(conf, tmp_path):
    """Training loads the SAVED schema, so drift from the config must be loud."""
    from credit.datasets.gen_2.channel_utils import DEFAULT_SCHEMA_FILENAME, ChannelSchema

    schema = ChannelSchema.from_config(conf)
    schema.save(str(tmp_path / DEFAULT_SCHEMA_FILENAME))
    assert "save_loc" not in _wheres(_run(conf))  # in sync

    conf["data"]["source"]["ERA5"]["variables"]["diagnostic"]["vars_2D"].append("SNOW")
    conf["model"]["output_only_channels"] = 2
    rep = _run(conf)
    assert "save_loc" in _wheres(rep)
    assert "disagrees" in _text(rep)


# ===========================================================================
# Driver, exit codes, and output formats
# ===========================================================================


def test_missing_file_errors(tmp_path):
    args = argparse.Namespace(config=str(tmp_path / "nope.yml"), deep=False, strict=False, json=False)
    with pytest.raises(SystemExit) as exc:
        _check(args)
    assert exc.value.code == 1


def test_malformed_yaml_reports_line(tmp_path, capsys):
    path = tmp_path / "bad.yml"
    path.write_text("data:\n  source:\n   - [unclosed\n")
    args = argparse.Namespace(config=str(path), deep=False, strict=False, json=False)
    with pytest.raises(SystemExit) as exc:
        _check(args)
    assert exc.value.code == 1
    assert "YAML failed to parse" in capsys.readouterr().out


def test_clean_config_exits_zero(tmp_path, capsys):
    path = tmp_path / "ok.yml"
    path.write_text(yaml.dump(_base_conf(tmp_path)))
    args = argparse.Namespace(config=str(path), deep=False, strict=False, json=False)
    with pytest.raises(SystemExit) as exc:
        _check(args)
    assert exc.value.code == 0
    assert "OK:" in capsys.readouterr().out


def test_json_output_is_parseable(tmp_path, capsys):
    conf = _base_conf(tmp_path)
    conf["model"]["type"] = "nope"
    path = tmp_path / "bad.yml"
    path.write_text(yaml.dump(conf))
    args = argparse.Namespace(config=str(path), deep=False, strict=False, json=True)
    with pytest.raises(SystemExit):
        _check(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"] >= 1
    assert any(f["where"] == "model.type" for f in payload["findings"])


def test_a_crashing_check_does_not_hide_the_others(conf, monkeypatch):
    """One check raising must degrade to a warning, not abort the run."""
    import importlib

    # `from ._check import _check` in credit/cli/__init__.py rebinds the name to
    # the FUNCTION, so `credit.cli._check` as an attribute is not the module.
    # import_module returns the real module out of sys.modules.
    check_mod = importlib.import_module("credit.cli._check")

    # Named to match the attribute it replaces: findings are labelled with the
    # failing check's __name__, so a stub called "boom" would hide that.
    def _check_trainer(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(check_mod, "_check_trainer", _check_trainer)
    conf["model"]["type"] = "wxformr"
    rep = _run(conf)
    assert "model.type" in _wheres(rep)
    assert "_check_trainer" in _wheres(rep, "warning")


def test_deep_skips_blocks_awaiting_preprocess(conf):
    """An unfitted scaler must not produce a second, noisier FileNotFoundError."""
    rep = _run(conf, deep=True)
    scaler_failures = [f for f in rep.findings if "scaler" in f.where and "failed to construct" in f.message]
    assert scaler_failures == [], _text(rep)
    assert not any("FileNotFoundError" in f.message for f in rep.findings), _text(rep)
    assert "--deep" in {f.where for f in rep.findings if f.severity == "info"}


def test_deep_constructs_blocks_that_do_not_need_a_scaler(conf, monkeypatch):
    """Blocks with no scaler dependency are genuinely instantiated in deep mode."""
    built = []
    import credit.postblock.reconstruct as reconstruct_mod

    original = reconstruct_mod.Reconstruct.__init__

    def spy(self, *args, **kwargs):
        built.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(reconstruct_mod.Reconstruct, "__init__", spy)
    _run(conf, deep=True)
    assert {"detach": False} in built


def test_deep_reports_block_construction_failure(conf):
    conf["postblocks"]["per_step"]["exp"] = {
        "type": "exp_transform",
        "args": {"variables": [], "key": "y_target_processed", "base": "7"},  # base must be e/2/10
    }
    rep = _run(conf, deep=True)
    assert "postblocks.per_step.exp" in _wheres(rep)
    assert "failed to construct" in _text(rep)


def test_parser_wires_check_subcommand():
    from credit.cli import _build_parser

    args = _build_parser().parse_args(["check", "-c", "x.yml", "--deep", "--strict", "--json"])
    assert (args.command, args.config, args.deep, args.strict, args.json) == ("check", "x.yml", True, True, True)


def test_checks_do_not_mutate_the_config(conf):
    """A linter that edits its input would corrupt anything running after it."""
    before = copy.deepcopy(conf)
    _run(conf)
    assert conf == before
