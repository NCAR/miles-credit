"""Tendency noise (SPPT-style tendency_noise postblock) and start perturbations (start_perturbation preblock)."""

import math

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from credit.datasets.gen_2._utils import encode_time
from credit.ensemble.red_noise import ar1_phi, reset_noise_state
from credit.postblock import build_postblocks
from credit.postblock.tendency_noise import TendencyNoise, tendency_noise_errors
from credit.preblock import build_preblocks
from credit.preblock.start_perturbation import StartPerturbation, start_perturbation_errors

_H, _W, _LEV = 24, 48, 4
_TREFHT = "SRC/prognostic/2d/TREFHT"
_T = "SRC/prognostic/3d/T"
_Q = "SRC/prognostic/3d/Qtot"


def _lag_corr(series: torch.Tensor, lag: int) -> float:
    a, b = series[:-lag].flatten(), series[lag:].flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std() * b.std()))


# ---------------------------------------------------------------------------
# Tendency noise
# ---------------------------------------------------------------------------

_ONE_SCALE = [{"sigma": 0.3, "tau_days": 1.0, "length_scale_km": 2000.0}]


def _tendency_block(**kw):
    args = dict(fields=[{"var": _TREFHT}, {"var": _T, "levels": [1, 2, 3]}], scales=_ONE_SCALE)
    return TendencyNoise(**{**args, **kw})


def _state_and_prediction(batch=3, x_batch=None, history=1, requires_grad=False):
    torch.manual_seed(1)
    f64 = {"dtype": torch.float64}  # exact ratios: (y - x) can be tiny
    x = {
        _TREFHT: torch.randn(x_batch or batch, 1, history, _H, _W, **f64),
        _T: torch.randn(x_batch or batch, _LEV, history, _H, _W, **f64),
        _Q: torch.randn(x_batch or batch, _LEV, history, _H, _W, **f64),
    }
    y = {
        _TREFHT: torch.randn(batch, 1, 1, _H, _W, requires_grad=requires_grad, **f64),
        _T: torch.randn(batch, _LEV, 1, _H, _W, **f64),
        _Q: torch.randn(batch, _LEV, 1, _H, _W, **f64),
    }
    return {"y_processed": {"SRC": dict(y)}, "x_physical": {"SRC": x}}, x, y


def test_tendency_noise_scales_the_change_by_one_shared_factor():
    batch, x, y = _state_and_prediction()
    out = _tendency_block()(batch)["y_processed"]["SRC"]

    factor_2d = (out[_TREFHT] - x[_TREFHT]) / (y[_TREFHT] - x[_TREFHT])  # 1 + r
    factor_3d = (out[_T] - x[_T]) / (y[_T] - x[_T])
    for level in (1, 2, 3):
        torch.testing.assert_close(factor_3d[:, level], factor_2d[:, 0], rtol=1e-4, atol=1e-4)
    assert torch.equal(out[_T][:, 0], y[_T][:, 0])  # level 0 is not listed
    assert torch.equal(out[_Q], y[_Q])  # Qtot is not listed
    assert (factor_2d.std(dim=(-2, -1)) > 0).all()  # a random field, different per member
    assert not torch.allclose(factor_2d[0], factor_2d[1])


def test_tendency_noise_leaves_points_without_change_alone():
    batch, x, _ = _state_and_prediction()
    batch["y_processed"]["SRC"][_TREFHT] = x[_TREFHT].clone()
    out = _tendency_block()(batch)["y_processed"]["SRC"]
    torch.testing.assert_close(out[_TREFHT], x[_TREFHT])


def test_tendency_noise_amplitude_memory_and_clip():
    torch.manual_seed(2)
    scales = [
        {"sigma": 0.2, "tau_days": 0.5, "length_scale_km": 2000.0},
        {"sigma": 0.1, "tau_days": 5.0, "length_scale_km": 3000.0},
    ]
    block = TendencyNoise(fields=[{"var": _TREFHT}], scales=scales)
    zeros, ones = torch.zeros(8, 1, 1, _H, _W), torch.ones(8, 1, 1, _H, _W)
    series = []
    for _ in range(200):
        batch = {"y_processed": {"SRC": {_TREFHT: ones}}, "x_physical": {"SRC": {_TREFHT: zeros}}}
        series.append(block(batch)["y_processed"]["SRC"][_TREFHT][:, 0, 0] - 1.0)  # r
    series = torch.stack(series)
    var1, var2 = 0.2**2, 0.1**2
    assert series.std().item() == pytest.approx(math.sqrt(var1 + var2), rel=0.05)
    phi1, phi2 = ar1_phi(12, 6), ar1_phi(120, 6)
    assert _lag_corr(series, 1) == pytest.approx((var1 * phi1 + var2 * phi2) / (var1 + var2), abs=0.03)

    reset_noise_state(torch.nn.ModuleDict({"sppt": block}))
    assert all(pattern.noise.state is None for pattern in block.patterns)

    wide = TendencyNoise(fields=[{"var": _TREFHT}], scales=[{**_ONE_SCALE[0], "sigma": 3.0}], clip=0.8)
    batch = {"y_processed": {"SRC": {_TREFHT: ones}}, "x_physical": {"SRC": {_TREFHT: zeros}}}
    r = wide(batch)["y_processed"]["SRC"][_TREFHT] - 1.0
    assert r.abs().max().item() == pytest.approx(0.8)  # clipped, and the clip is reached


def test_tendency_noise_uses_the_newest_input_step_and_repeats_it_for_batch_ensembles():
    batch, x, y = _state_and_prediction(batch=4, x_batch=2, history=2, requires_grad=True)
    leaf = y[_TREFHT]
    out = _tendency_block()(batch)["y_processed"]["SRC"][_TREFHT]
    newest = x[_TREFHT][:, :, -1:].repeat_interleave(2, dim=0)
    factor = (out - newest) / (leaf - newest)
    out.sum().backward()
    torch.testing.assert_close(leaf.grad, factor.detach(), rtol=1e-4, atol=1e-4)  # d out / d y = 1 + r


def test_tendency_noise_needs_the_field_in_the_inputs():
    batch, _, _ = _state_and_prediction()
    del batch["x_physical"]["SRC"][_T]
    with pytest.raises(KeyError, match="physical inputs"):
        _tendency_block()(batch)


def test_tendency_noise_builds_from_config_and_keeps_nothing_in_the_checkpoint():
    conf = {"postblocks": {"per_step": {"sppt": {"type": "tendency_noise", "args": {"fields": [{"var": _T}]}}}}}
    block = build_postblocks(conf)["sppt"]
    assert isinstance(block, TendencyNoise)
    assert [s["sigma"] for s in block.scales] == [0.52, 0.18, 0.06]  # ECMWF's three scales by default
    assert block.state_dict() == {}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fields": []}, "non-empty list"),
        ({"fields": [{"var": "T"}]}, "variable key"),
        ({"fields": [{"var": _T}, {"var": _T}]}, "twice"),
        ({"fields": [{"var": _T, "levels": "31"}]}, "levels"),
        ({"fields": [{"var": _T, "sigma": 1}]}, "unknown keys"),
        ({"fields": [{"var": _T}], "scales": []}, "scales must be"),
        ({"fields": [{"var": _T}], "scales": [{"sigma": 0.5, "tau_days": 1}]}, "exactly the keys"),
        ({"fields": [{"var": _T}], "scales": [{**_ONE_SCALE[0], "sigma": -1}]}, "sigma must be"),
        ({"fields": [{"var": _T}], "clip": 1.5}, "clip"),
    ],
)
def test_tendency_noise_rejects_bad_args(kwargs, match):
    assert any(match in e for e in tendency_noise_errors(**kwargs))
    with pytest.raises(ValueError, match="tendency_noise"):
        TendencyNoise(**kwargs)


# ---------------------------------------------------------------------------
# Start perturbation
# ---------------------------------------------------------------------------

_YEARS = (2001, 2002)


def _times(year: int, calendar: str):
    if calendar == "standard":
        return pd.date_range(f"{year}-01-01", f"{year}-02-28 18:00", freq="6h")
    return xr.date_range(f"{year}-01-01", f"{year}-02-28 18:00", freq="6h", calendar=calendar, use_cftime=True)


@pytest.fixture(params=["noleap", "standard"])
def stores(request, tmp_path):
    """One store per year: fields that differ by time, level and place, so every donor pair differs."""
    calendar = request.param
    rng = np.random.default_rng(0)
    for year in _YEARS:
        times = _times(year, calendar)
        n = len(times)
        xr.Dataset(
            {
                "T": (("time", "lev", "lat", "lon"), rng.normal(250, 10, (n, _LEV, _H, _W)).astype(np.float32)),
                "Qtot": (("time", "lev", "lat", "lon"), rng.uniform(0, 1e-3, (n, _LEV, _H, _W)).astype(np.float32)),
                "TREFHT": (("time", "lat", "lon"), rng.normal(280, 5, (n, _H, _W)).astype(np.float32)),
            },
            coords={"time": times},
        ).to_zarr(tmp_path / f"state_{year}.zarr")
    return str(tmp_path / "state_%Y.zarr"), calendar


def _start(calendar: str, year=2003, month=1, day=20, hour=6):
    if calendar == "standard":
        return pd.Timestamp(year, month, day, hour)
    return cftime.datetime(year, month, day, hour, calendar=calendar)


def _ic_batch(start, rows=1):
    fields = {
        _T: torch.full((rows, _LEV, 1, _H, _W), 250.0),
        _Q: torch.full((rows, _LEV, 1, _H, _W), 5e-4),
        _TREFHT: torch.full((rows, 1, 1, _H, _W), 280.0),
        "SRC/static/2d/Z": torch.zeros(rows, 1, 1, _H, _W),
    }
    encoded = torch.full((rows,), encode_time(start), dtype=torch.int64)
    return {"input": {"SRC": fields}, "metadata": {"SRC": {"input_datetime": encoded}}}


def _perturber(path, **kw):
    args = dict(path=path, fields=[_T, _Q, _TREFHT], amplitude=0.2, years=list(_YEARS), window_days=10, seed=7)
    return StartPerturbation(**{**args, **kw})


def test_start_perturbation_adds_a_scaled_difference_of_two_donor_states(stores):
    path, calendar = stores
    start = _start(calendar)
    block = _perturber(path)
    batch = _ic_batch(start)
    out = block(batch)["input"]["SRC"]

    # replay the donor draw for row 0
    replay = _perturber(path)
    replay._calendar = calendar
    rng = np.random.default_rng([7, (encode_time(start) // 10**9) & 0xFFFFFFFFFFFF, 0])
    (y1, i1), (y2, i2) = replay._donors(rng, start)
    for var in (_T, _TREFHT):
        expected = 0.2 * (replay._read(y1, i1, var) - replay._read(y2, i2, var)) / math.sqrt(2)
        torch.testing.assert_close(out[var][0] - batch["input"]["SRC"][var][0], expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(out["SRC/static/2d/Z"], batch["input"]["SRC"]["SRC/static/2d/Z"])  # statics untouched


def test_start_perturbation_donors_share_the_hour_and_season(stores):
    path, calendar = stores
    start = _start(calendar, day=20, hour=6)
    block = _perturber(path)
    block._calendar = calendar
    for row in range(20):
        rng = np.random.default_rng([1, row])
        for year, index in block._donors(rng, start):
            when = block._open(year).indexes["time"][index]
            assert year in _YEARS and when.year == year and when.hour == 6
            target = _start(calendar, year=year, day=20, hour=6)
            assert abs((when - target).total_seconds()) <= 10 * 86400


def test_start_perturbation_is_reproducible_per_seed_start_and_row(stores):
    path, calendar = stores
    start = _start(calendar)
    a = _perturber(path)(_ic_batch(start, rows=2))["input"]["SRC"][_T]
    b = _perturber(path)(_ic_batch(start, rows=2))["input"]["SRC"][_T]
    c = _perturber(path, seed=8)(_ic_batch(start, rows=2))["input"]["SRC"][_T]
    later = _perturber(path)(_ic_batch(_start(calendar, day=21)))["input"]["SRC"][_T]
    assert torch.equal(a, b)  # same seed, start and row: same kick, whatever else used the RNG
    assert not torch.equal(a[0], a[1])  # rows (members) differ
    assert not torch.equal(a, c)
    assert not torch.equal(a[:1], later)


def test_start_perturbation_clamps_tracers_and_checks_shapes(stores):
    path, calendar = stores
    out = _perturber(path, amplitude=50.0, nonnegative=[_Q])(_ic_batch(_start(calendar)))["input"]["SRC"]
    assert (out[_Q] >= 0).all() and (out[_Q] == 0).any()

    batch = _ic_batch(_start(calendar))
    batch["input"]["SRC"][_T] = batch["input"]["SRC"][_T][:, :2]  # e.g. a level subset
    with pytest.raises(ValueError, match="does not match"):
        _perturber(path)(batch)


def test_start_perturbation_reads_one_store_for_all_years(tmp_path):
    times = np.concatenate([_times(y, "noleap") for y in _YEARS])
    xr.Dataset(
        {"TREFHT": (("time", "lat", "lon"), np.random.default_rng(0).normal(280, 5, (len(times), _H, _W)))},
        coords={"time": times},
    ).to_zarr(tmp_path / "all.zarr")
    block = _perturber(str(tmp_path / "all.zarr"), fields=[_TREFHT])
    out = block(_ic_batch(_start("noleap")))["input"]["SRC"][_TREFHT]
    assert out.std() > 0


def test_start_perturbation_builds_from_config(stores):
    path, _ = stores
    args = dict(path=path, fields=[_T], amplitude=0.1, years=list(_YEARS))
    conf = {"preblocks": {"ic_only": {"kick": {"type": "start_perturbation", "args": args}}}}
    assert isinstance(build_preblocks(conf, phase="ic_only")["kick"], StartPerturbation)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fields": []}, "fields must be"),
        ({"fields": ["T"]}, "fields must be"),
        ({"amplitude": 0}, "amplitude must be"),
        ({"amplitude": {_T: 0.1}}, "one entry per field"),
        ({"years": [2002, 2001]}, "years must be"),
        ({"years": [2001]}, "years must be"),
        ({"window_days": 0}, "window_days"),
        ({"nonnegative": ["SRC/prognostic/3d/U"]}, "nonnegative"),
    ],
)
def test_start_perturbation_rejects_bad_args(kwargs, match):
    args = {"path": "x_%Y.zarr", "fields": [_T, _Q], "amplitude": 0.2, "years": [2001, 2002], **kwargs}
    assert any(match in e for e in start_perturbation_errors(**args))
    with pytest.raises(ValueError, match="start_perturbation"):
        StartPerturbation(**args)
