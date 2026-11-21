"""The config generator: arms differ from HF only where intended, and local configs carry no glade paths.

A stand-in local root is built from the file names the HF config uses, so the tests need no data.
"""

import os
import posixpath

import pytest
import yaml

from camulator_smallscale import make_configs as mc


def _flatten(node, prefix=""):
    """{dotted.key: leaf} for a nested config (lists are leaves)."""
    if not isinstance(node, dict):
        return {prefix: node}
    out = {}
    for key, value in node.items():
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _hf():
    with open(mc.HF_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture
def local_root(tmp_path):
    """data/ and assets/ holding an empty stand-in for every glade file the HF config names.

    The output metadata YAML is left out: it exists only on glade, and the local target replaces it.
    """
    root = tmp_path / "climate"
    (root / "data").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "data" / mc.LOCAL_YEAR_ZARR.replace("%Y", str(mc.LOCAL_YEAR))).mkdir()
    for path in mc.iter_strings(_hf()):
        if not path.startswith("/glade/") or "$USER" in path or "%Y" in path:
            continue
        if path == _hf()["inference"]["output"]["metadata"]:
            continue
        (root / "assets" / posixpath.basename(path)).touch()
    return root


def test_local_config_has_only_local_paths(local_root, tmp_path):
    conf = mc.build_config("null", "local", local_root=str(local_root), runs_root=str(tmp_path / "runs"))
    strings = list(mc.iter_strings(conf))
    assert not [s for s in strings if "/glade/" in s]
    for section in ("data", "validation_data"):
        source = conf[section]["source"]["CESM"]
        assert source["level_coord"] == "level"
        for field in ("prognostic", "diagnostic", "dynamic_forcing"):
            assert source["variables"][field]["path"] == str(local_root / "data" / mc.LOCAL_YEAR_ZARR)
    assert conf["loss"]["args"]["latitude_weights"].startswith(str(local_root / "assets"))
    assert "pbs" not in conf
    assert conf["inference"]["output"]["metadata"] == "cam.yaml"
    assert conf["save_loc"] == str(tmp_path / "runs" / "null")
    assert conf["inference"]["save_forecast"] == str(tmp_path / "runs" / "null" / "rollout")
    assert conf["trainer"]["parallelism"]["data"] == "none"
    assert conf["trainer"]["train_batch_size"] == 1
    # The local model is the MPS-safe HFCamulator; Derecho keeps hf_camulator.
    assert conf["model"]["type"] == "mps_safe_hf_camulator"
    assert conf["custom_objects"]["mps_safe_hf_camulator"] == mc.MPS_SAFE_MODEL
    assert mc.build_config("null", "derecho")["model"]["type"] == "hf_camulator"


def test_missing_local_file_raises(local_root):
    (local_root / "assets" / "f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc").unlink()
    with pytest.raises(FileNotFoundError, match="f.e21"):
        mc.build_config("null", "local", local_root=str(local_root))


def test_null_differs_from_hf_only_in_the_finetune_protocol():
    hf = _flatten(_hf())
    null = _flatten(mc.build_config("null", "derecho"))
    assert set(hf) <= set(null)
    changed = {key for key in null if hf.get(key, object()) != null[key]}
    allowed = {f"trainer.{key}" for key in mc.FINETUNE["trainer"]} | {
        "trainer.scheduler.T_max",
        "save_loc",
        "inference.save_forecast",
    }
    assert changed <= allowed, sorted(changed - allowed)


def test_finetune_protocol():
    conf = mc.build_config("null", "derecho")
    trainer = conf["trainer"]
    assert (trainer["start_epoch"], trainer["num_epoch"], trainer["epochs"]) == (0, 4, 4)
    assert trainer["reload_epoch"] is False and trainer["load_optimizer"] is False and trainer["load_weights"] is True
    assert trainer["learning_rate"] == 2.5e-05 and trainer["use_scheduler"] is False
    assert trainer["batches_per_epoch"] == 300 and trainer["train_batch_size"] == 12
    assert conf["data"]["forecast_len"] == 3
    assert conf["model"]["trainable_layers"] == ["up_block1", "up_block2", "up_block3", "up_block4"]
    assert conf["loss"]["type"] == "hf_loss"
    assert conf["save_loc"] == posixpath.join(mc.DERECHO_RUNS, "null")


def test_written_config_round_trips(tmp_path):
    conf = mc.build_config("null", "derecho")
    path = mc.write_config(conf, str(tmp_path / "null.yml"), "null", "derecho")
    with open(path) as f:
        text = f.read()
    assert "&id" not in text and "*id" not in text
    assert yaml.safe_load(text) == conf


def test_unknown_arm_and_target():
    with pytest.raises(KeyError):
        mc.build_config("nope", "derecho")
    with pytest.raises(ValueError):
        mc.build_config("null", "casper")


def test_stage_link_and_existing(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"weights")
    save_loc = tmp_path / "run"
    dest = mc.stage_checkpoint(str(save_loc), str(checkpoint), "link")
    assert os.path.islink(dest) and os.path.realpath(dest) == os.path.realpath(checkpoint)
    # An existing checkpoint.pt is never overwritten.
    assert mc.stage_checkpoint(str(save_loc), str(checkpoint), "clone") == dest
    assert os.path.islink(dest)


def test_stage_clone_is_a_copy(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"weights")
    dest = mc.stage_checkpoint(str(tmp_path / "run"), str(checkpoint), "clone")
    assert not os.path.islink(dest)
    with open(dest, "ab") as f:
        f.write(b"-trained")
    assert checkpoint.read_bytes() == b"weights"
