"""Tests for wxformer_simple (credit/models/wxformer/wxformer_simple.py).

The model takes only architecture settings and builds its data-shaped layers from
the first batch. These tests cover that build, the padding it picks, the saved
hyperparameters, and reloading weights without data. CPU only, small grids.
"""

import os

import pandas as pd
import pytest
import torch
import yaml

from credit.models import build_trained_shapes, load_model
from credit.models.base_model import BaseModel
from credit.models.wxformer.wxformer_simple import (
    HPARAMS_FILENAME,
    WXFormerSimple,
    load_hparams,
    materialize_from_data,
    required_divisor,
    resolve_padding,
)

# Small but real: divisor 64 with the default strides/windows.
ARCH = {"dim": [16, 32, 64, 128], "depth": [1, 1, 1, 1], "dim_head": 8}


def _model(**overrides):
    return WXFormerSimple(**{**ARCH, **overrides})


def _built(c_in=7, frames=1, c_out=6, height=37, width=70, seed=0, **overrides):
    torch.manual_seed(seed)
    model = _model(**overrides)
    x = torch.randn(2, c_in, frames, height, width)
    y = torch.randn(2, c_out, 1, height, width)
    return model.materialize(x, y), x


# ---------------------------------------------------------------------------
# Building from data
# ---------------------------------------------------------------------------


def test_unbuilt_model_has_no_parameters():
    model = _model()
    assert model.needs_materialize
    assert sum(p.numel() for p in model.parameters()) == 0


def test_materialize_measures_shapes_and_forwards():
    model, x = _built(c_in=7, frames=2, c_out=6, height=37, width=70)
    assert not model.needs_materialize
    assert model.data_shape["input_channels"] == 7
    assert model.data_shape["frames"] == 2
    assert model.data_shape["output_channels"] == 6
    assert (model.data_shape["image_height"], model.data_shape["image_width"]) == (37, 70)
    assert model(x).shape == (2, 6, 1, 37, 70)


def test_first_forward_builds_when_output_channels_given():
    model = _model(output_channels=4)
    out = model(torch.randn(1, 5, 1, 37, 70))
    assert out.shape == (1, 4, 1, 37, 70)
    assert not model.needs_materialize


def test_first_forward_without_output_width_explains_itself():
    with pytest.raises(ValueError, match="materialize\\(x, y\\)"):
        _model()(torch.randn(1, 5, 1, 37, 70))


def test_four_dimensional_input_is_one_frame():
    model = _model(output_channels=3)
    assert model(torch.randn(1, 5, 37, 70)).shape == (1, 3, 1, 37, 70)
    assert model.data_shape["frames"] == 1


def test_wrong_input_width_after_build_raises():
    model, _ = _built(c_in=7)
    with pytest.raises(ValueError, match="built for 7 input channels"):
        model(torch.randn(1, 8, 1, 37, 70))


def test_same_seed_builds_identical_weights():
    """The FSDP2 contract: every rank must initialize the same weights."""
    a, _ = _built(seed=123)
    b, _ = _built(seed=123)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(pa, pb), name


def test_stage0_stride_four_resizes_back_to_the_grid():
    model, x = _built(cross_embed_strides=[4, 2, 2, 2], height=181, width=360)
    assert model(x).shape == (2, 6, 1, 181, 360)


def test_target_on_another_grid_is_rejected():
    model = _model()
    with pytest.raises(ValueError, match="differ"):
        model.materialize(torch.randn(1, 5, 1, 37, 70), torch.randn(1, 3, 1, 18, 35))


def test_base_models_do_not_need_materializing():
    assert BaseModel.needs_materialize is False


# ---------------------------------------------------------------------------
# Architecture and legacy keys
# ---------------------------------------------------------------------------


def test_non_pyramid_dim_rejected_with_suggestion():
    with pytest.raises(ValueError, match=r"\[16, 32, 64, 128\]"):
        _model(dim=[32, 32, 64, 128])


def test_wrong_stage_count_rejected():
    with pytest.raises(ValueError, match="exactly 4"):
        _model(depth=[1, 1])


def test_legacy_keys_matching_the_data_are_accepted():
    model = _model(channels=2, levels=2, surface_channels=1, input_only_channels=2, output_only_channels=1)
    model.materialize(torch.randn(1, 7, 1, 37, 70), torch.randn(1, 6, 1, 37, 70))


def test_legacy_keys_disagreeing_with_the_data_raise():
    model = _model(channels=3, levels=2, surface_channels=1, input_only_channels=2, image_height=40)
    with pytest.raises(ValueError, match="input channels: configured 9, data has 7"):
        model.materialize(torch.randn(1, 7, 1, 37, 70), torch.randn(1, 6, 1, 37, 70))


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_required_divisor_defaults():
    assert required_divisor([2, 2, 2, 2], 4, [8, 4, 2, 1]) == 64


@pytest.mark.parametrize(
    "height,width,pad_lat,pad_lon",
    [(64, 128, [0, 0], [0, 0]), (181, 360, [5, 6], [12, 12]), (721, 1440, [23, 24], [16, 16])],
)
def test_resolve_padding_is_minimal_and_balanced(height, width, pad_lat, pad_lon):
    assert resolve_padding(height, width, 64) == (pad_lat, pad_lon)


def test_min_pad_is_respected_per_side():
    pad_lat, pad_lon = resolve_padding(181, 360, 64, {"min_pad_lat": 10, "min_pad_lon": 0})
    assert min(pad_lat) >= 10 and (181 + sum(pad_lat)) % 64 == 0
    assert pad_lon == [12, 12]


def test_mode_none_requires_a_divisible_grid():
    assert resolve_padding(64, 128, 64, "none") == ([0, 0], [0, 0])
    with pytest.raises(ValueError, match="not a multiple of 64"):
        resolve_padding(181, 360, 64, {"mode": "none"})


def test_pad_larger_than_the_grid_is_rejected():
    with pytest.raises(ValueError, match="pole reflection"):
        resolve_padding(10, 64, 64)


def test_unknown_padding_key_is_rejected():
    with pytest.raises(ValueError, match="pad_lat"):
        _model(padding={"mode": "earth", "pad_lat": [1, 1]})


@pytest.mark.parametrize("mode", ["earth", "mirror"])
def test_odd_grid_round_trips_through_padding(mode):
    model, x = _built(height=45, width=90, padding={"mode": mode})
    assert model.use_padding
    assert model(x).shape == (2, 6, 1, 45, 90)


# ---------------------------------------------------------------------------
# Saving and reloading
# ---------------------------------------------------------------------------


def test_hparams_include_defaults_and_measured_values(tmp_path):
    model, _ = _built()
    path = model.save_hparams(str(tmp_path))
    with open(path) as f:
        saved = yaml.safe_load(f)
    assert saved["type"] == "wxformer_simple"
    assert saved["cross_embed_strides"] == [2, 2, 2, 2]  # a default, never in the config
    assert saved["local_window_size"] == [4, 4, 4, 4]
    assert saved["input_channels"] == 7 and saved["output_channels"] == 6
    assert saved["pad_lat"] == [13, 14]


def test_existing_hparams_file_is_kept(tmp_path):
    first, _ = _built()
    first.save_hparams(str(tmp_path))
    other, _ = _built(c_in=9)
    other.save_hparams(str(tmp_path))
    assert load_hparams(str(tmp_path))["input_channels"] == 7


def _save_trained(tmp_path, model):
    model.save_hparams(str(tmp_path))
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(tmp_path, "checkpoint.pt"))


def test_load_model_rebuilds_from_hparams_without_data(tmp_path):
    model, x = _built()
    model.eval()
    _save_trained(tmp_path, model)

    conf = {"save_loc": str(tmp_path), "model": {"type": "wxformer_simple", **ARCH}}
    loaded = load_model(conf, load_weights=True).eval()
    assert not loaded.needs_materialize
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_load_model_prefers_saved_hparams_over_edited_config(tmp_path, caplog):
    model, x = _built()
    model.eval()
    _save_trained(tmp_path, model)

    conf = {"save_loc": str(tmp_path), "model": {"type": "wxformer_simple", **ARCH, "depth": [2, 2, 2, 2]}}
    loaded = load_model(conf, load_weights=True).eval()
    assert loaded.arch["depth"] == [1, 1, 1, 1]
    assert "using the saved values" in caplog.text
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_build_trained_shapes_lets_ddp_paths_load_a_checkpoint(tmp_path):
    """The DDP rollout and `credit plot` paths call load_model(conf) and then load_state_dict."""
    model, x = _built()
    model.eval()
    _save_trained(tmp_path, model)

    conf = {"save_loc": str(tmp_path), "model": {"type": "wxformer_simple", **ARCH}}
    rebuilt = build_trained_shapes(load_model(conf), conf)
    ckpt = torch.load(os.path.join(tmp_path, "checkpoint.pt"))
    rebuilt.load_state_dict(ckpt["model_state_dict"])
    rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), rebuilt(x))


def test_build_trained_shapes_leaves_other_models_alone():
    model = torch.nn.Linear(2, 2)
    assert build_trained_shapes(model, {}) is model


def test_load_model_without_hparams_explains_itself(tmp_path):
    model, _ = _built()
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(tmp_path, "checkpoint.pt"))
    conf = {"save_loc": str(tmp_path), "model": {"type": "wxformer_simple", **ARCH}}
    with pytest.raises(FileNotFoundError, match=HPARAMS_FILENAME):
        load_model(conf, load_weights=True)


# ---------------------------------------------------------------------------
# materialize_from_data: the train_gen2 path
# ---------------------------------------------------------------------------


class _OneSampleDataset:
    """Minimal stand-in for MultiSourceDataset: one unbatched nested sample."""

    datetimes = pd.DatetimeIndex(["2000-01-01"])

    def __getitem__(self, args):
        t3 = lambda: torch.randn(2, 1, 37, 70)  # noqa: E731
        t2 = lambda: torch.randn(1, 1, 37, 70)  # noqa: E731
        return {
            "input": {
                "ERA5": {"ERA5/prognostic/3d/T": t3(), "ERA5/prognostic/2d/SP": t2(), "ERA5/static/2d/lsm": t2()}
            },
            "target": {
                "ERA5": {"ERA5/prognostic/3d/T": t3(), "ERA5/prognostic/2d/SP": t2(), "ERA5/diagnostic/2d/TP": t2()}
            },
        }


def test_materialize_from_data_runs_the_preblock_chain():
    conf = {"preblocks": {"per_step": {"concat": {"type": "concat"}}}}
    model = _model()
    materialize_from_data(model, conf, _OneSampleDataset())
    # input: T (2 levels) + SP + lsm = 4; target: T (2 levels) + SP + TP = 4
    assert model.data_shape["input_channels"] == 4
    assert model.data_shape["output_channels"] == 4
    assert (model.data_shape["image_height"], model.data_shape["image_width"]) == (37, 70)


def _nested(B, C, H, W):
    """One nested batch of C 2D prognostic variables, as the gen2 loader yields it."""
    return {
        group: {"era5": {f"era5/prognostic/2d/v{i}": torch.randn(B, 1, 1, H, W) for i in range(C)}}
        for group in ("input", "target")
    }


class _Loader:
    dataset = None
    sampler = None

    def __init__(self, B, C, H, W, n_batches):
        self.shape, self.n = (B, C, H, W), n_batches

    def __len__(self):
        return self.n

    def __iter__(self):
        return (_nested(*self.shape) for _ in range(self.n))


def test_trains_two_autoregressive_steps_with_ema(tmp_path):
    """The gen2 trainer loop, EMA and optimizer all work on a data-built model."""
    from credit.trainers.trainer_gen2 import TrainerERA5Gen2

    B, C, H, W = 1, 4, 37, 70
    conf = {
        "save_loc": str(tmp_path),
        "trainer": {
            "mode": "none",
            "start_epoch": 0,
            "epochs": 1,
            "num_epoch": 1,
            "amp": False,
            "use_scheduler": False,
            "use_ema": True,
            "ema_decay": 0.9,
            "use_tensorboard": False,
            "skip_validation": True,
            "train_batch_size": B,
            "batches_per_epoch": 1,
        },
        "data": {
            "forecast_len": 2,
            "source": {
                "era5": {
                    "levels": [],
                    "variables": {"prognostic": {"vars_3D": [], "vars_2D": [f"v{i}" for i in range(C)]}},
                }
            },
        },
        "preblocks": {"per_step": {"concat": {"type": "concat"}}},
        "postblocks": {"per_step": {"reconstruct": {"type": "reconstruct"}}},
    }

    class _Dataset:
        datetimes = pd.DatetimeIndex(["2000-01-01"])

        def __getitem__(self, args):
            batch = _nested(1, C, H, W)
            return {g: {s: {v: t[0] for v, t in d.items()} for s, d in grp.items()} for g, grp in batch.items()}

    model = _model()
    materialize_from_data(model, conf, _Dataset())
    trainer = TrainerERA5Gen2(model, rank=0, conf=conf)
    model.to(trainer.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    results = trainer.train_one_epoch(
        epoch=0,
        trainloader=_Loader(B, C, H, W, n_batches=2),
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        scheduler=None,
        metrics=lambda pred, y: {},
    )
    assert torch.isfinite(torch.tensor(results["train_loss"][0]))
    assert results["train_forecast_len"][-1] == 2
    assert any(not torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())
    assert trainer.ema is not None and trainer.ema.step > 0
