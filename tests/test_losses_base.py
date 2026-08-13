"""Tests for credit.losses.base.BaseLoss and the Reconstruct in_key/out_key extension."""

import numpy as np
import pytest
import torch
from credit.datasets.gen_2.channel_utils import ChannelSchema
from credit.losses import BaseLoss, is_crps_loss, load_loss
from credit.losses.base import _load_target_variances, _scaler_channel_variance
from credit.postblock.reconstruct import FlattenToTensor, Reconstruct

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VAR_T = "ERA5/prognostic/3d/T"
VAR_SP = "ERA5/prognostic/2d/SP"
VAR_PRECIP = "ERA5/diagnostic/2d/precip"
VAR_MSLP = "ERA5/diagnostic/2d/MSLP_computed"  # postblock-computed diagnostic (not in data layout)
N_LEVELS = 3

# Per-variable physical scales (sigma^2) written into the synthetic scaler file.
VARIANCES = {VAR_T: [100.0, 25.0, 4.0], VAR_SP: [1.0e6], VAR_PRECIP: [1.0e-8]}


def _make_scaler_file(tmp_path):
    from bridgescaler import save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = {}
    for var_key, variances in VARIANCES.items():
        n = len(variances)
        # channels_last=False: postblock tensors are (B, levels, time, H, W) — channels at dim 1
        s = DStandardScalerTensor(channels_last=False)
        s.mean_x_ = torch.zeros(n)
        s.var_x_ = torch.tensor(variances, dtype=torch.float32)
        s.x_columns_ = list(range(n))  # tensors carry no column names — channels are integer-indexed
        s.n_ = 100
        s._fit = True
        scalers[var_key] = s
    path = str(tmp_path / "scaler.json")
    save_scaler_dict({"target": {"ERA5": scalers}}, path)
    return path


def _make_schema():
    """ChannelSchema for T(3D) + SP(2D) prognostic and precip(2D) diagnostic."""
    return ChannelSchema.from_config(
        {
            "data": {
                "source": {
                    "ERA5": {
                        "levels": list(range(N_LEVELS)),
                        "variables": {
                            "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                            "diagnostic": {"vars_2D": ["precip"]},
                        },
                    }
                }
            },
            "model": {"levels": N_LEVELS},
        }
    )


def _make_conf(scaler_path, tmp_path, loss_args=None):
    """Full config with the new-style {type, args} loss section (for load_loss tests)."""
    args = {
        "training_loss": "mse",
        "var_weighting": "inverse_variance",
        "scaler_path": scaler_path,
    }
    if loss_args:
        args.update(loss_args)
    return {
        "save_loc": str(tmp_path),
        "data": {
            "source": {
                "ERA5": {
                    "levels": list(range(N_LEVELS)),
                    "variables": {
                        "prognostic": {"vars_3D": ["T"], "vars_2D": ["SP"]},
                        "diagnostic": {"vars_2D": ["precip"]},
                    },
                }
            }
        },
        "model": {"levels": N_LEVELS},
        "loss": {"type": "base", "args": args},
    }


def _make_loss(scaler_path, **kwargs):
    """BaseLoss with kwargs, wired to the synthetic channel schema."""
    kwargs.setdefault("training_loss", "mse")
    kwargs.setdefault("var_weighting", "inverse_variance")
    if kwargs["var_weighting"] in ("inverse_variance", "learnable"):
        kwargs.setdefault("scaler_path", scaler_path)
    kwargs.setdefault("channel_schema", _make_schema())
    return BaseLoss(**kwargs)


def _make_state_dict(batch=2, height=4, width=5, pred_requires_grad=True, seed=0, computed=False):
    g = torch.Generator().manual_seed(seed)
    pred = {
        VAR_T: torch.randn(batch, N_LEVELS, 1, height, width, generator=g),
        VAR_SP: torch.randn(batch, 1, 1, height, width, generator=g) * 1000.0,
        VAR_PRECIP: torch.rand(batch, 1, 1, height, width, generator=g) * 1e-4,
    }
    if computed:
        pred[VAR_MSLP] = torch.randn(batch, 1, 1, height, width, generator=g) * 100.0
    target = {k: v + 0.1 * torch.randn(v.shape, generator=g) for k, v in pred.items() if k != VAR_MSLP}
    if pred_requires_grad:
        pred = {k: v.requires_grad_(True) for k, v in pred.items()}
    return {
        "y_processed": {"ERA5": pred},
        "y_target_processed": {"ERA5": target},
    }


# ---------------------------------------------------------------------------
# Variance extraction
# ---------------------------------------------------------------------------


def test_last_var_losses_initialized_before_forward():
    """The documented attribute exists before any forward pass."""
    loss = BaseLoss(training_loss="mse", var_weighting="none")
    assert loss.last_var_losses == {}


def test_scaler_channel_variance_standard():
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    s = DStandardScalerTensor()
    s.mean_x_ = torch.zeros(2)
    s.var_x_ = torch.tensor([3.0, 5.0])
    out = _scaler_channel_variance(s)
    assert torch.allclose(out, torch.tensor([3.0, 5.0]))


def test_scaler_channel_variance_quantile():
    from bridgescaler.distributed_tensor import DQuantileScalerTensor

    torch.manual_seed(0)
    data = torch.randn(20000, 2) * torch.tensor([3.0, 0.5])
    s = DQuantileScalerTensor()
    s.fit(data)
    s.tensorize_attributes()
    out = _scaler_channel_variance(s)
    # t-digest centroid moment estimate slightly underestimates; 20% tolerance
    assert torch.allclose(out, torch.tensor([9.0, 0.25]), rtol=0.2)


def test_scaler_channel_variance_numpy_backend():
    from bridgescaler import DeepStandardScaler

    s = DeepStandardScaler()
    s.sd_ = np.array([2.0, 4.0])
    out = _scaler_channel_variance(s)
    assert torch.allclose(out, torch.tensor([4.0, 16.0]))


def test_load_target_variances(tmp_path):
    path = _make_scaler_file(tmp_path)
    variances = _load_target_variances(path)
    assert variances[VAR_SP] == pytest.approx(1.0e6)
    assert variances[VAR_T] == pytest.approx(np.mean(VARIANCES[VAR_T]))
    assert variances[VAR_PRECIP] == pytest.approx(1.0e-8)


# ---------------------------------------------------------------------------
# Construction / registry
# ---------------------------------------------------------------------------


def test_load_loss_returns_base_loss(tmp_path):
    conf = _make_conf(_make_scaler_file(tmp_path), tmp_path)
    train_crit = load_loss(conf)
    valid_crit = load_loss(conf, validation=True)
    assert isinstance(train_crit, BaseLoss)
    assert isinstance(valid_crit, BaseLoss)
    assert not is_crps_loss("base")


def test_variable_list_from_schema(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path))
    state = _make_state_dict()
    loss_fn(state)
    assert loss_fn.var_keys == [VAR_T, VAR_SP, VAR_PRECIP]


def test_crps_base_rejected(tmp_path):
    with pytest.raises(ValueError, match="CRPS"):
        _make_loss(_make_scaler_file(tmp_path), training_loss="KCRPS")


def test_inverse_variance_requires_scaler_path(tmp_path):
    with pytest.raises(ValueError, match="scaler_path"):
        BaseLoss(training_loss="mse", var_weighting="inverse_variance", scaler_path=None, channel_schema=_make_schema())


def test_bad_weighting_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="var_weighting"):
        _make_loss(_make_scaler_file(tmp_path), var_weighting="bogus")


# ---------------------------------------------------------------------------
# Forward / combination math
# ---------------------------------------------------------------------------


def test_forward_combination_none_mode(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none")
    state = _make_state_dict()
    loss = loss_fn(state)

    expected_per_var = {}
    for var_key in loss_fn.var_keys:
        p = state["y_processed"]["ERA5"][var_key].float()
        t = state["y_target_processed"]["ERA5"][var_key].float()
        expected_per_var[var_key] = torch.mean((p - t) ** 2).item()
    expected = np.mean(list(expected_per_var.values()))
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert loss_fn.last_var_losses == pytest.approx(expected_per_var, rel=1e-5)


def test_forward_inverse_variance_weights(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path))
    state = _make_state_dict()
    loss = loss_fn(state)

    raw = {
        VAR_T: 1.0 / np.mean(VARIANCES[VAR_T]),
        VAR_SP: 1.0 / 1.0e6,
        VAR_PRECIP: 1.0 / 1.0e-8,
    }
    mean_w = np.mean(list(raw.values()))
    expected = np.mean([raw[k] / mean_w * loss_fn.last_var_losses[k] for k in raw])
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_inverse_variance_matches_normalized_mse(tmp_path):
    """With 2D variables only, inverse_variance (no weight normalization) must
    exactly reproduce the MSE computed in normalized space."""
    variances = {VAR_SP: [4.0], VAR_PRECIP: [9.0]}
    from bridgescaler import save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = {}
    for var_key, var_list in variances.items():
        s = DStandardScalerTensor(channels_last=False)
        s.mean_x_ = torch.zeros(1)
        s.var_x_ = torch.tensor(var_list)
        s.x_columns_ = [0]
        s.n_ = 10
        s._fit = True
        scalers[var_key] = s
    scaler_path = str(tmp_path / "scaler.json")
    save_scaler_dict({"target": {"ERA5": scalers}}, scaler_path)

    schema = ChannelSchema.from_config(
        {
            "data": {
                "source": {
                    "ERA5": {
                        "levels": [1],
                        "variables": {"prognostic": {"vars_2D": ["SP"]}, "diagnostic": {"vars_2D": ["precip"]}},
                    }
                }
            },
            "model": {"levels": 1},
        }
    )
    loss_fn = BaseLoss(
        training_loss="mse",
        var_weighting="inverse_variance",
        scaler_path=scaler_path,
        normalize_weights=False,
        channel_schema=schema,
    )

    g = torch.Generator().manual_seed(1)
    shape = (2, 1, 1, 4, 5)
    # normalized-space values; physical = normalized * sigma (mean 0)
    pred_n = {k: torch.randn(shape, generator=g) for k in variances}
    tgt_n = {k: torch.randn(shape, generator=g) for k in variances}
    state = {
        "y_processed": {"ERA5": {k: v * np.sqrt(variances[k][0]) for k, v in pred_n.items()}},
        "y_target_processed": {"ERA5": {k: v * np.sqrt(variances[k][0]) for k, v in tgt_n.items()}},
    }
    with torch.no_grad():
        loss = loss_fn(state)

    flat_pred = torch.cat([pred_n[VAR_SP], pred_n[VAR_PRECIP]], dim=1)
    flat_tgt = torch.cat([tgt_n[VAR_SP], tgt_n[VAR_PRECIP]], dim=1)
    expected = torch.mean((flat_pred - flat_tgt) ** 2).item()
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_manual_weights(tmp_path):
    manual = {VAR_T: 2.0, VAR_SP: 1.0, VAR_PRECIP: 0.5}
    loss_fn = _make_loss(
        _make_scaler_file(tmp_path),
        var_weighting="manual",
        variable_weights=manual,
        normalize_weights=False,
    )
    state = _make_state_dict()
    loss = loss_fn(state)
    expected = np.mean([manual[k] * loss_fn.last_var_losses[k] for k in manual])
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_base_loss_overrides(tmp_path):
    loss_fn = _make_loss(
        _make_scaler_file(tmp_path),
        var_weighting="none",
        base_loss_overrides={VAR_PRECIP: {"loss": "mae"}},
    )
    state = _make_state_dict()
    loss_fn(state)
    p = state["y_processed"]["ERA5"][VAR_PRECIP].float()
    t = state["y_target_processed"]["ERA5"][VAR_PRECIP].float()
    expected_mae = torch.mean(torch.abs(p - t)).item()
    assert loss_fn.last_var_losses[VAR_PRECIP] == pytest.approx(expected_mae, rel=1e-5)


def test_learnable_mode(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="learnable")
    assert isinstance(loss_fn.log_variance, torch.nn.Parameter)
    # initialized at log(sigma_v^2) from the scaler
    expected_init = [np.log(np.mean(VARIANCES[VAR_T])), np.log(1.0e6), np.log(1.0e-8)]
    assert loss_fn.log_variance.detach().numpy() == pytest.approx(expected_init, rel=1e-5)

    state = _make_state_dict()
    loss = loss_fn(state)
    assert loss_fn.var_keys == [VAR_T, VAR_SP, VAR_PRECIP]
    loss.backward()
    assert loss_fn.log_variance.grad is not None
    assert torch.isfinite(loss_fn.log_variance.grad).all()


def test_validation_loss_variant(tmp_path):
    loss_fn = _make_loss(
        _make_scaler_file(tmp_path),
        var_weighting="none",
        validation_loss="mae",
        validation=True,
    )
    state = _make_state_dict()
    loss_fn(state)
    p = state["y_processed"]["ERA5"][VAR_T].float()
    t = state["y_target_processed"]["ERA5"][VAR_T].float()
    assert loss_fn.last_var_losses[VAR_T] == pytest.approx(torch.mean(torch.abs(p - t)).item(), rel=1e-5)


def test_latitude_weights_applied(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none")
    height = 4
    loss_fn.lat_weights = torch.arange(1, height + 1, dtype=torch.float32)

    pred = {k: torch.ones(1, n, 1, height, 5) for k, n in ((VAR_T, N_LEVELS), (VAR_SP, 1), (VAR_PRECIP, 1))}
    target = {k: torch.zeros_like(v) for k, v in pred.items()}
    state = {"y_processed": {"ERA5": pred}, "y_target_processed": {"ERA5": target}}
    with torch.no_grad():
        loss_fn(state)
    # every elementwise entry is 1; weighted mean over H with weights 1..4 = 2.5
    assert loss_fn.last_var_losses[VAR_SP] == pytest.approx(np.mean([1, 2, 3, 4]), rel=1e-5)


# ---------------------------------------------------------------------------
# Computed diagnostics
# ---------------------------------------------------------------------------


def test_computed_diagnostics_included(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=True)
    state = _make_state_dict(computed=True)
    state["y_target_processed"]["ERA5"][VAR_MSLP] = torch.zeros_like(state["y_processed"]["ERA5"][VAR_MSLP])
    loss_fn(state)
    assert VAR_MSLP in loss_fn.last_var_losses
    assert loss_fn.var_keys == [VAR_T, VAR_SP, VAR_PRECIP, VAR_MSLP]


def test_computed_diagnostics_skipped(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=False)
    state = _make_state_dict(computed=True)
    loss_fn(state)
    assert VAR_MSLP not in loss_fn.last_var_losses
    assert loss_fn.var_keys == [VAR_T, VAR_SP, VAR_PRECIP]


def test_computed_diagnostics_missing_target_raises(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none", include_computed_diagnostics=True)
    state = _make_state_dict(computed=True)  # VAR_MSLP only in pred
    with pytest.raises(KeyError, match="y_target_processed"):
        loss_fn(state)


def test_computed_diagnostics_learnable_raises(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="learnable")
    state = _make_state_dict(computed=True)
    state["y_target_processed"]["ERA5"][VAR_MSLP] = torch.zeros_like(state["y_processed"]["ERA5"][VAR_MSLP])
    with pytest.raises(ValueError, match="learnable"):
        loss_fn(state)


# ---------------------------------------------------------------------------
# Error handling and gradient flow
# ---------------------------------------------------------------------------


def test_missing_target_dict_raises(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path))
    state = _make_state_dict()
    del state["y_target_processed"]
    with pytest.raises(KeyError, match="y_target_processed"):
        loss_fn(state)


def test_detached_pred_raises(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path))
    state = _make_state_dict(pred_requires_grad=False)
    with pytest.raises(RuntimeError, match="detach"):
        loss_fn(state)
    # no error under no_grad (validation path)
    with torch.no_grad():
        loss_fn(state)


def test_gradient_flows_through_reconstruct(tmp_path):
    loss_fn = _make_loss(_make_scaler_file(tmp_path), var_weighting="none")

    n_ch = N_LEVELS + 1 + 1  # T(3) + SP(1) + precip(1)
    channel_map = {
        VAR_T: {"slice": slice(0, N_LEVELS), "orig_shape": (N_LEVELS, 1)},
        VAR_SP: {"slice": slice(N_LEVELS, N_LEVELS + 1), "orig_shape": (1, 1)},
        VAR_PRECIP: {"slice": slice(N_LEVELS + 1, n_ch), "orig_shape": (1, 1)},
    }
    y_pred = torch.randn(2, n_ch, 4, 5, requires_grad=True)
    state = {
        "y_pred": y_pred,
        "metadata": {"target": {"_channel_map": channel_map}},
        "y_target_processed": {
            "ERA5": {k: torch.zeros(2, n, 1, 4, 5) for k, n in ((VAR_T, N_LEVELS), (VAR_SP, 1), (VAR_PRECIP, 1))}
        },
    }
    state = Reconstruct(detach=False)(state)
    loss = loss_fn(state)
    loss.backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


# ---------------------------------------------------------------------------
# Reconstruct in_key / out_key
# ---------------------------------------------------------------------------


def test_reconstruct_custom_keys_roundtrip():
    n_ch = N_LEVELS + 1
    channel_map = {
        VAR_T: {"slice": slice(0, N_LEVELS), "orig_shape": (N_LEVELS, 1)},
        VAR_SP: {"slice": slice(N_LEVELS, n_ch), "orig_shape": (1, 1)},
    }
    y = torch.randn(2, n_ch, 4, 5)
    state = {"y": y, "metadata": {"target": {"_channel_map": channel_map}}}

    state = Reconstruct(in_key="y", out_key="y_target_processed")(state)
    assert "y_target_processed" in state
    assert state["y_target_processed"]["ERA5"][VAR_T].shape == (2, N_LEVELS, 1, 4, 5)

    state = FlattenToTensor(key="y_target_processed", out_key="y_flat")(state)
    assert torch.allclose(state["y_flat"], y)


def test_reconstruct_defaults_unchanged():
    channel_map = {VAR_SP: {"slice": slice(0, 1), "orig_shape": (1, 1)}}
    y_pred = torch.randn(2, 1, 4, 5, requires_grad=True)
    state = {"y_pred": y_pred, "metadata": {"target": {"_channel_map": channel_map}}}
    state = Reconstruct()(state)
    assert "y_processed" in state
    assert not state["y_processed"]["ERA5"][VAR_SP].requires_grad


# ---------------------------------------------------------------------------
# End-to-end: the example-config postblock chain feeding BaseLoss
# ---------------------------------------------------------------------------


def test_postblock_chain_with_base_loss(tmp_path):
    from credit.postblock.scaler import BridgeScalerTransform

    scaler_path = _make_scaler_file(tmp_path)
    loss_fn = _make_loss(scaler_path, var_weighting="inverse_variance")

    n_ch = N_LEVELS + 1 + 1  # T(3) + SP(1) + precip(1)
    channel_map = {
        VAR_T: {"slice": slice(0, N_LEVELS), "orig_shape": (N_LEVELS, 1)},
        VAR_SP: {"slice": slice(N_LEVELS, N_LEVELS + 1), "orig_shape": (1, 1)},
        VAR_PRECIP: {"slice": slice(N_LEVELS + 1, n_ch), "orig_shape": (1, 1)},
    }
    variables = list(channel_map)
    y_pred = torch.randn(2, n_ch, 4, 5, requires_grad=True)
    state = {
        "y_pred": y_pred,
        "y": torch.randn(2, n_ch, 4, 5),
        "metadata": {"target": {"_channel_map": channel_map}},
    }

    # Same chain as the gen_2 example configs' postblocks.per_step
    state = Reconstruct(detach=False)(state)
    state = BridgeScalerTransform(scaler_path, variables, method="inverse_transform")(state)
    state = Reconstruct(in_key="y", out_key="y_target_processed")(state)
    state = BridgeScalerTransform(scaler_path, variables, method="inverse_transform", key="y_target_processed")(state)

    # Both dicts are now in physical units: SP scaled by sqrt(var) = 1000
    assert state["y_processed"]["ERA5"][VAR_SP].abs().mean() > 100.0

    loss = loss_fn(state)
    loss.backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
    assert set(loss_fn.last_var_losses) == {VAR_T, VAR_SP, VAR_PRECIP}
