"""Tests for the semi-Lagrangian advection pre/postblocks.

Covers the preblock's data_type selection and non-mutation of the caller's
batch, and that the block is reachable from both factory registries. The
core advection numerics (uniform-flow displacement, continuity-derived omega)
are covered by the postblock tests in ``test_postblock_gen2.py``; both blocks
share the same ``_SemiLagrangianAdvectionEngine``.
"""

import pytest
import torch

from credit.postblock import _load_postblock_entry
from credit.postblock.advect import SemiLagrangianAdvectionPost
from credit.preblock import _load_preblock_entry
from credit.preblock.advect import SemiLagrangianAdvectionPre

SRC = "ERA5"
U_VAR = f"{SRC}/prognostic/3d/u_component_of_wind"
V_VAR = f"{SRC}/prognostic/3d/v_component_of_wind"
SP_VAR = f"{SRC}/prognostic/2d/surface_pressure"
Q_VAR = f"{SRC}/prognostic/3d/specific_humidity"

B, L, T, H, W = 1, 4, 1, 8, 16


def _engine_args(**overrides):
    args = dict(
        tracer_vars=[Q_VAR],
        u_var=U_VAR,
        v_var=V_VAR,
        surface_pressure_var=SP_VAR,
        timestep_seconds=21600.0,
        levels=list(range(1, L + 1)),
    )
    args.update(overrides)
    return args


def _nested_state():
    """A uniform eastward flow with a longitudinal Gaussian tracer blob."""
    u = torch.full((B, L, T, H, W), 60.0)
    v = torch.zeros((B, L, T, H, W))
    sp = torch.full((B, 1, T, H, W), 101_325.0)
    lon = torch.arange(W).float()
    blob = torch.exp(-((lon - 4.0) ** 2) / (2 * 1.5**2))
    q = blob.view(1, 1, 1, 1, W).expand(B, L, T, H, W).clone()
    return {SRC: {U_VAR: u, V_VAR: v, SP_VAR: sp, Q_VAR: q}}


def test_preblock_transforms_requested_data_types():
    """The preblock advects input and target without mutating the caller's batch."""
    batch = {"input": _nested_state(), "target": _nested_state()}
    original_input_q = batch["input"][SRC][Q_VAR]
    mod = SemiLagrangianAdvectionPre(**_engine_args())
    out = mod(batch)

    assert not torch.allclose(out["input"][SRC][Q_VAR], original_input_q)
    assert not torch.allclose(out["target"][SRC][Q_VAR], batch["target"][SRC][Q_VAR])
    # caller's dict still holds the pre-advection tensor
    assert batch["input"][SRC][Q_VAR] is original_input_q


def test_preblock_data_type_selection():
    """data_types=["input"] leaves target untouched; invalid types raise."""
    batch = {"input": _nested_state(), "target": _nested_state()}
    original_input_q = batch["input"][SRC][Q_VAR].clone()
    original_target_q = batch["target"][SRC][Q_VAR].clone()
    mod = SemiLagrangianAdvectionPre(data_types=["input"], **_engine_args())
    out = mod(batch)

    assert not torch.allclose(out["input"][SRC][Q_VAR], original_input_q)
    assert torch.allclose(out["target"][SRC][Q_VAR], original_target_q)

    with pytest.raises(ValueError, match="data_types"):
        SemiLagrangianAdvectionPre(data_types=["metadata"], **_engine_args())


def test_preblock_missing_data_type_skipped():
    """A data type absent from the batch (e.g. no target during inference) is a no-op."""
    batch = {"input": _nested_state()}
    mod = SemiLagrangianAdvectionPre(**_engine_args())
    out = mod(batch)
    assert set(out.keys()) == {"input"}


def test_registered_in_both_registries():
    """The block is reachable from both factory registries under the same type key."""
    assert _load_postblock_entry("semilagrangian_advection") is SemiLagrangianAdvectionPost
    assert _load_preblock_entry("semilagrangian_advection") is SemiLagrangianAdvectionPre
