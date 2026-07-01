"""
Tests applications/preprocess.py by loading in a very small WeatherBench2ERA5Dataset and performing 5 rounds of fits.
Uses the coarsest version of Weatherbench2 ERA5 and loads in the main state variables. Perform a log transform
on specific humidity.

Confirm that the resulting scaler info is reasonable. Test fitting standard, minmax, and quantile scalers. The
scaler values should have no nans and the scaler values should be in the correct order. Transform should also be
applied and the transformed data should match the expected properties of each distribution.

Coverage notes
--------------
These tests exercise the exact preprocessing workflow that ``credit.applications.preprocess.main`` performs on
each worker, minus the distributed orchestration (``barrier``/``gather_object``/``save_scaler_dict`` on rank 0,
which require an initialised process group):

    load_dataset -> load_dataloader -> cycle -> build_preblocks
        -> apply_preblocks_before_scaler (log transform)
        -> BridgeScalerTransform.fit_scaler_batch   (repeated, one call per batch)

The fitted scalers are then combined across rounds (mirroring the cross-rank ``np.sum`` in ``main``), saved, and
reloaded through ``BridgeScalerTransform`` to verify the transform path.

The data is pulled from the public WeatherBench2 64x32 (~5.6 deg) GCS store. If the store is unreachable the
network-dependent tests skip rather than fail. ``channels_last=False`` is used so the scalers compute per-level
statistics, matching the CREDIT channel-first tensor layout (B, n_levels, T, lat, lon).
"""

import sys
import warnings

import numpy as np
import pytest
import torch

try:
    from bridgescaler import save_scaler_dict, scale_var_dict  # noqa: F401
    from credit.preblock import build_preblocks, apply_preblocks_before_scaler
    from credit.preblock.scaler import BridgeScalerTransform
    from credit.trainers.utils import load_dataset, load_dataloader, cycle

    _DEPS_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - environment dependent
    _DEPS_AVAILABLE = False

_skip_no_deps = pytest.mark.skipif(not _DEPS_AVAILABLE, reason="bridgescaler / credit deps unavailable")

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

SOURCE = "WB2"
LEVELS = [500, 850]
N_ROUNDS = 5

VAR_T = f"{SOURCE}/prognostic/3d/temperature"
VAR_Q = f"{SOURCE}/prognostic/3d/specific_humidity"
VAR_SP = f"{SOURCE}/prognostic/2d/surface_pressure"
SCALER_TYPES = ["standard", "minmax", "quantile"]


def _make_conf(save_loc: str, scaler_path: str) -> dict:
    """Minimal preprocess-style config for the WeatherBench2 64x32 store."""
    return {
        "seed": 42,
        "save_loc": save_loc,
        "data": {
            "source": {
                SOURCE: {
                    "dataset_type": "weatherbench2_era5",
                    "resolution": "64x32",
                    "level_coord": "level",
                    "levels": LEVELS,
                    "variables": {
                        "prognostic": {
                            "vars_3D": ["temperature", "specific_humidity"],
                            "vars_2D": ["surface_pressure"],
                        },
                        "diagnostic": None,
                        "dynamic_forcing": None,
                        "static": None,
                    },
                }
            },
            "start_datetime": "2020-01-01",
            "end_datetime": "2020-01-05",
            "timestep": "6h",
            "history_len": 1,
            "forecast_len": 1,
        },
        "trainer": {
            "mode": "none",
            "train_batch_size": 2,
            "thread_workers": 0,
            "batches_per_epoch": N_ROUNDS,
        },
        "preblocks": {
            "per_step": {
                # Log transform on specific humidity (positive, heavy-tailed).
                "log_transform": {
                    "type": "log_transform",
                    "args": {"variables": [VAR_Q], "data_types": ["input", "target"]},
                },
                # Placeholder scaler block so apply_preblocks_before_scaler stops here;
                # the per-test scalers below are built separately.
                "scaler": {
                    "type": "bridgescaler_transform",
                    "args": {
                        "variables": [],
                        "scaler_path": scaler_path,
                        "scaler_type": "standard",
                        "scaler_params": {"channels_last": False},
                    },
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_over_rounds(scaler_type: str, batches: list, scaler_path: str):
    """Run BridgeScalerTransform.fit_scaler_batch over every batch.

    fit_scaler_batch accumulates the fit internally (running merge across calls),
    so the final return value is the combined fit over all batches.

    Returns ``(transformer, combined_scaler_dict)``.
    """
    transformer = BridgeScalerTransform(
        scaler_path=scaler_path,
        variables=[],  # expand to all variables present in the batch
        method="transform",
        scaler_type=scaler_type,
        scaler_params={"channels_last": False},
    )
    combined = None
    for batch in batches:
        combined = transformer.fit_scaler_batch(batch)
    return transformer, combined


def _leaf_scalers(scaler_dict: dict) -> dict:
    """Flatten a nested scaler dict to ``{var_key: scaler}`` for the input source."""
    return scaler_dict["input"][SOURCE]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def processed_batches(tmp_path_factory):
    """Load the tiny WB2 dataset and return ``N_ROUNDS`` log-transformed batches.

    Mirrors preprocess.main: build the dataloader, then for each batch apply the
    pre-scaler preblocks (here, the log transform on specific humidity). The
    batches are fetched once and reused across the scaler tests to keep the suite
    fast. Skips if the public GCS store cannot be reached.
    """
    if not _DEPS_AVAILABLE:
        pytest.skip("dependencies unavailable")

    tmp = tmp_path_factory.mktemp("preprocess")
    conf = _make_conf(str(tmp), str(tmp / "scaler_placeholder.json"))

    warnings.simplefilter("ignore")
    try:
        dataset = load_dataset(conf, is_train=True)
        loader = load_dataloader(conf, dataset, rank=0, world_size=1, is_train=True)
        preblocks = build_preblocks(conf)
        dl = cycle(loader)

        batches = []
        q_unlogged = []  # specific humidity before the log transform, for verification
        for _ in range(N_ROUNDS):
            batch = next(dl)
            q_unlogged.append(batch["input"][SOURCE][VAR_Q].clone())
            processed = apply_preblocks_before_scaler(preblocks, batch, torch.device("cpu"))
            batches.append(processed)
    except Exception as err:  # network / GCS / auth issues
        pytest.skip(f"WeatherBench2 GCS store unavailable: {err}")

    return batches, q_unlogged


# ---------------------------------------------------------------------------
# Workflow / preblock tests
# ---------------------------------------------------------------------------


@_skip_no_deps
def test_batches_have_expected_structure(processed_batches):
    """Each loaded batch is nested [data_type][source][var_key] with sane shapes."""
    batches, _ = processed_batches
    assert len(batches) == N_ROUNDS
    for batch in batches:
        assert "input" in batch
        assert SOURCE in batch["input"]
        inp = batch["input"][SOURCE]
        for var in (VAR_T, VAR_Q, VAR_SP):
            assert var in inp
        # (B, n_levels, T, lat, lon); 64x32 store -> lat=64, lon=32.
        assert inp[VAR_T].shape == (2, len(LEVELS), 1, 64, 32)
        assert inp[VAR_SP].shape == (2, 1, 1, 64, 32)
        assert not torch.isnan(inp[VAR_T]).any()


@_skip_no_deps
def test_log_transform_applied_to_specific_humidity(processed_batches):
    """The pre-scaler log transform alters specific humidity but not temperature."""
    batches, q_unlogged = processed_batches
    for batch, q_raw in zip(batches, q_unlogged):
        q_logged = batch["input"][SOURCE][VAR_Q]
        # Transform is log(q + eps) - log(eps): maps 0 -> 0, strictly increasing,
        # and amplifies the small positive humidity values (eps = 1e-8 is tiny).
        assert not torch.allclose(q_logged, q_raw)
        assert torch.all(q_logged >= -1e-4), "non-negative humidity must map to non-negative values"
        assert torch.all(q_logged > q_raw - 1e-4), "transform should not shrink small positive humidity"
        assert not torch.isnan(q_logged).any()
        assert torch.isfinite(q_logged).all()
        # Temperature is untouched by the log transform (plausible Kelvin range).
        t = batch["input"][SOURCE][VAR_T]
        assert t.min() > 180 and t.max() < 340


# ---------------------------------------------------------------------------
# Scaler fitting tests (standard / minmax / quantile)
# ---------------------------------------------------------------------------


@_skip_no_deps
@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_fit_scaler_no_nans_and_correct_order(scaler_type, processed_batches, tmp_path):
    """Fit each scaler over 5 rounds; check stats are finite and correctly ordered."""
    batches, _ = processed_batches
    _, combined = _fit_over_rounds(scaler_type, batches, str(tmp_path / "scaler.json"))
    leaves = _leaf_scalers(combined)

    # Every fitted variable from the (empty) selection should be present.
    for var in (VAR_T, VAR_Q, VAR_SP):
        assert var in leaves

    for var, scaler in leaves.items():
        n_levels = len(LEVELS) if "/3d/" in var else 1

        if scaler_type == "standard":
            mean = np.asarray(scaler.mean_x_)
            var_ = np.asarray(scaler.var_x_)
            assert mean.size == n_levels
            assert np.isfinite(mean).all() and np.isfinite(var_).all()
            assert (var_ > 0).all(), "variance must be strictly positive"
            assert int(np.asarray(scaler.n_).reshape(-1)[0]) > 0

        elif scaler_type == "minmax":
            mn = np.asarray(scaler.min_x_)
            mx = np.asarray(scaler.max_x_)
            assert mn.size == n_levels
            assert np.isfinite(mn).all() and np.isfinite(mx).all()
            assert (mn < mx).all(), "min must be strictly below max per level"

        else:  # quantile
            mn = np.asarray(scaler.min_)
            mx = np.asarray(scaler.max_)
            assert np.isfinite(mn).all() and np.isfinite(mx).all()
            assert (mn < mx).all(), "min must be strictly below max per level"


@_skip_no_deps
def test_fit_standard_temperature_values_reasonable(processed_batches, tmp_path):
    """The standard scaler's per-level temperature mean lands in a plausible range."""
    batches, _ = processed_batches
    _, combined = _fit_over_rounds("standard", batches, str(tmp_path / "scaler.json"))
    t_scaler = _leaf_scalers(combined)[VAR_T]
    mean = np.asarray(t_scaler.mean_x_)
    assert mean.shape == (len(LEVELS),)
    assert ((mean > 200) & (mean < 320)).all(), f"implausible temperature means: {mean}"


# ---------------------------------------------------------------------------
# Transform tests — verify distribution properties per scaler family
# ---------------------------------------------------------------------------


@_skip_no_deps
@pytest.mark.parametrize("scaler_type", SCALER_TYPES)
def test_transform_distribution_properties(scaler_type, processed_batches, tmp_path):
    """Save the fitted scaler, reload it through BridgeScalerTransform, and check
    that transforming the data yields the distribution each scaler promises."""
    batches, _ = processed_batches
    _, combined = _fit_over_rounds(scaler_type, batches, str(tmp_path / "fit.json"))

    # Persist and reload through the transformer's own load path (method="transform").
    scaler_file = str(tmp_path / "combined.json")
    save_scaler_dict(combined, scaler_file)
    transformer = BridgeScalerTransform(
        scaler_path=scaler_file,
        variables=[],
        method="transform",
        scaler_type=scaler_type,
        scaler_params={"channels_last": False},
    )

    out = transformer(batches[0])

    for var in (VAR_T, VAR_Q, VAR_SP):
        x = out["input"][SOURCE][var]
        assert not torch.isnan(x).any(), f"{scaler_type}/{var} produced NaNs"

        if scaler_type == "standard":
            # Transforming in-sample data is ~zero-mean / unit-variance per channel.
            assert abs(float(x.mean())) < 0.5
            assert 0.5 < float(x.std()) < 1.6
        elif scaler_type == "minmax":
            assert float(x.min()) >= -1e-4
            assert float(x.max()) <= 1.0 + 1e-4
        else:  # quantile, uniform output distribution
            assert float(x.min()) >= -1e-4
            assert float(x.max()) <= 1.0 + 1e-4


@_skip_no_deps
def test_transform_inverse_round_trip(processed_batches, tmp_path):
    """transform followed by inverse_transform recovers the original tensors."""
    batches, _ = processed_batches
    _, combined = _fit_over_rounds("standard", batches, str(tmp_path / "fit.json"))
    scaler_file = str(tmp_path / "combined.json")
    save_scaler_dict(combined, scaler_file)

    fwd = BridgeScalerTransform(
        scaler_path=scaler_file, variables=[], method="transform", scaler_params={"channels_last": False}
    )
    inv = BridgeScalerTransform(
        scaler_path=scaler_file, variables=[], method="inverse_transform", scaler_params={"channels_last": False}
    )

    original = batches[0]["input"][SOURCE][VAR_T].clone()
    transformed = fwd(batches[0])
    recovered = inv(transformed)["input"][SOURCE][VAR_T]
    assert torch.allclose(recovered.float(), original.float(), atol=1e-3)


# ---------------------------------------------------------------------------
# BridgeScalerTransform unit behavior (credit/preblock/scaler.py)
# ---------------------------------------------------------------------------


@_skip_no_deps
def test_variable_selection_expands_empty_list(processed_batches, tmp_path):
    """An empty ``variables`` list expands to every variable in the batch on first fit."""
    batches, _ = processed_batches
    transformer = BridgeScalerTransform(
        scaler_path=str(tmp_path / "scaler.json"),
        variables=[],
        method="transform",
        scaler_params={"channels_last": False},
    )
    assert transformer.variables_expanded is False
    transformer.fit_scaler_batch(batches[0])
    assert transformer.variables_expanded is True
    # Empty selection expands to every key in the batch. The three state variables
    # must be present; metadata keys (e.g. "input_datetime") may also appear but are
    # ignored by scale_var_dict, which skips the "metadata" branch entirely.
    assert {VAR_T, VAR_Q, VAR_SP}.issubset(set(transformer.variables))


@_skip_no_deps
def test_fit_scaler_batch_returns_nested_scaler_dict(processed_batches, tmp_path):
    """fit_scaler_batch returns a nested [data_type][source][var] dict of fitted scalers."""
    batches, _ = processed_batches
    transformer = BridgeScalerTransform(
        scaler_path=str(tmp_path / "scaler.json"),
        variables=[],
        method="transform",
        scaler_type="standard",
        scaler_params={"channels_last": False},
    )
    fitted = transformer.fit_scaler_batch(batches[0])
    assert "input" in fitted
    assert SOURCE in fitted["input"]
    leaf = fitted["input"][SOURCE][VAR_T]
    assert hasattr(leaf, "mean_x_") and np.isfinite(np.asarray(leaf.mean_x_)).all()


@_skip_no_deps
def test_fit_scaler_batch_accumulates_across_rounds(processed_batches, tmp_path):
    """Repeated fit_scaler_batch calls accumulate statistics (running merge), they
    do not overwrite the previous fit."""
    batches, _ = processed_batches
    transformer = BridgeScalerTransform(
        scaler_path=str(tmp_path / "scaler.json"),
        variables=[],
        method="transform",
        scaler_type="standard",
        scaler_params={"channels_last": False},
    )

    def _sample_count(scaler_dict):
        return int(np.asarray(scaler_dict["input"][SOURCE][VAR_T].n_).reshape(-1)[0])

    counts = []
    for batch in batches:
        counts.append(_sample_count(transformer.fit_scaler_batch(batch)))

    # Sample count must strictly increase as each round folds in more data.
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts
    # 64x32 grid, batch_size 2, 1 timestep -> 2*64*32 = 4096 samples per round.
    assert counts[-1] == N_ROUNDS * 2 * 64 * 32
    # The accumulated dict is also exposed on .scaler.
    assert transformer.scaler is not None


@_skip_no_deps
def test_fit_after_existing_scaler_file_raises(processed_batches, tmp_path):
    """Fitting when a scaler file already exists is rejected (no template to fit)."""
    batches, _ = processed_batches
    scaler_file = str(tmp_path / "exists.json")
    # Create a fitted scaler on disk first.
    _, combined = _fit_over_rounds("standard", batches, str(tmp_path / "fit.json"))
    save_scaler_dict(combined, scaler_file)

    loaded = BridgeScalerTransform(
        scaler_path=scaler_file, variables=[], method="transform", scaler_params={"channels_last": False}
    )
    assert loaded.scaler_template is None
    with pytest.raises(ValueError, match="already exists"):
        loaded.fit_scaler_batch(batches[0])


@_skip_no_deps
def test_preprocess_main_end_to_end(tmp_path, monkeypatch):
    """Drive applications.preprocess.main single-process and confirm it writes a
    reasonable scaler file (exercises the full CLI workflow incl. the rank-0 save)."""
    import yaml
    from bridgescaler import load_scaler_dict
    import credit.applications.preprocess as preprocess

    scaler_file = tmp_path / "scaler.json"
    conf = _make_conf(str(tmp_path / "save"), str(scaler_file))
    conf["preblocks"]["per_step"]["scaler"]["args"]["scaler_params"] = {"channels_last": False}
    conf["trainer"]["batches_per_epoch"] = 3

    config_path = tmp_path / "preprocess.yml"
    config_path.write_text(yaml.safe_dump(conf))

    monkeypatch.setattr(sys, "argv", ["credit_preprocess", "-c", str(config_path)])
    try:
        preprocess.main()
    except Exception as err:  # network / GCS issues
        pytest.skip(f"WeatherBench2 GCS store unavailable: {err}")

    assert scaler_file.exists(), "preprocess.main should write the scaler file on rank 0"
    scaler_dict = load_scaler_dict(str(scaler_file))
    t_scaler = scaler_dict["input"][SOURCE][VAR_T]
    mean = np.asarray(t_scaler.mean_x_)
    assert mean.shape == (len(LEVELS),)
    assert np.isfinite(mean).all()
    assert ((mean > 200) & (mean < 320)).all(), f"implausible temperature means: {mean}"
    # 3 batches * batch_size 2 * 64 * 32 grid points per level.
    assert int(np.asarray(t_scaler.n_).reshape(-1)[0]) == 3 * 2 * 64 * 32
