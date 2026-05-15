"""test_rollout_physical.py — GPU integration test: postblocks produce physical values.

Loads the smoke_gen2 checkpoint, runs a 4-step (24h) rollout using the
era5_normalizer preblock (same as training) and BridgeScaler postblocks
built from the same NC files, then asserts that t2m and T are in
physically plausible ranges.

Run explicitly on a GPU node:
    pytest tests/test_rollout_physical.py -v -m integration
"""

import os

import pandas as pd
import pytest
import torch
import yaml

CHECKPOINT = "/glade/derecho/scratch/schreck/credit_tests/smoke_gen2/best_checkpoint.pt"
CONFIG = "/glade/derecho/scratch/schreck/credit_tests/smoke_gen2/rollout_test.yml"
MEAN_NC = "/glade/campaign/cisl/aiml/credit/static_scalers/mean_6h_1979_2018_16lev_0.25deg.nc"
STD_NC = "/glade/campaign/cisl/aiml/credit/static_scalers/std_6h_1979_2018_16lev_0.25deg.nc"

_skip = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT) or not os.path.exists(MEAN_NC),
    reason="smoke_gen2 checkpoint or GLADE data not available",
)


def _sample_to_batch(sample, source_name):
    meta = {
        k: torch.tensor(v).unsqueeze(0) if isinstance(v, (int, float)) else v for k, v in sample["metadata"].items()
    }
    return {
        "input": {source_name: {k: v.unsqueeze(0) for k, v in sample["input"].items()}},
        "metadata": {source_name: meta},
    }


@pytest.mark.integration
@_skip
def test_rollout_predictions_are_physical(tmp_path):
    """24h rollout via postblocks: t2m and T must be in physical ranges."""
    from credit.cli._convert import _build_bridgescaler_jsons
    from credit.datasets.channel_layout import build_channel_layout, update_x
    from credit.datasets.local import LocalDataset
    from credit.models import load_model
    from credit.postblock import apply_postblocks, build_postblocks
    from credit.preblock import apply_preblocks, build_preblocks

    with open(CONFIG) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build BridgeScaler postblock JSON from the same NC files used in training
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    var_groups = {}
    if v.get("prognostic", {}).get("vars_3D"):
        var_groups[("prognostic", "3d")] = v["prognostic"]["vars_3D"]
    if v.get("prognostic", {}).get("vars_2D"):
        var_groups[("prognostic", "2d")] = v["prognostic"]["vars_2D"]

    pre_json = str(tmp_path / "pre.json")
    post_json = str(tmp_path / "post.json")
    _, post_vars = _build_bridgescaler_jsons(MEAN_NC, STD_NC, var_groups, pre_json, post_json)

    # Preblocks: era5_normalizer (same as training) + concat (always needed for apply_preblocks)
    preblock_cfg = dict(conf.get("preblocks", {}))
    preblock_cfg.setdefault("concat", {"type": "concat"})
    preblocks = build_preblocks(preblock_cfg)

    # Postblocks: reconstruct + bridgescaler inverse
    postblocks = build_postblocks(
        {
            "reconstruct": {"type": "reconstruct"},
            "scaler": {
                "type": "bridgescaler_transform",
                "args": {
                    "scaler_path": post_json,
                    "variables": post_vars,
                    "method": "inverse_transform",
                    "key": "prediction",
                },
            },
        }
    )

    # Load model
    model = load_model(conf).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Channel layout for update_x
    slices, _ = build_channel_layout(conf)

    # Dataset: one init time (2020-01-01 00Z), 4 steps = 24h
    dataset_conf = dict(conf["data"])
    dataset_conf["forecast_len"] = 1
    dataset_conf["start_datetime"] = "2020-01-01"
    dataset_conf["end_datetime"] = "2020-01-02"
    # rollout_test.yml predates dataset_type field; inject for LocalDataset
    src_name = next(iter(dataset_conf["source"]))
    dataset_conf["source"][src_name].setdefault("dataset_type", "local")
    dataset = LocalDataset(dataset_conf, return_target=False)

    dt = pd.Timedelta(conf["data"]["timestep"])
    t0 = pd.Timestamp("2020-01-01")

    src_name = dataset.curr_source_name
    sample_full = dataset[(t0, 0)]
    batch_full = _sample_to_batch(sample_full, src_name)
    out = apply_preblocks(preblocks, batch_full, device=device)
    x = out["input"].float()
    meta = out["metadata"]

    n_steps = 4
    step_preds = []

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            y_pred = model(x)

            batch_dict = apply_postblocks(postblocks, {"prediction": y_pred.clone(), "metadata": meta})
            step_preds.append(batch_dict["prediction"])

            if step < n_steps:
                t_next = t0 + step * dt
                sample_frc = dataset[(t_next, 1)]
                batch_frc = _sample_to_batch(sample_frc, src_name)
                x_frc = apply_preblocks(preblocks, batch_frc, device=device)["input"].float()
                x = update_x(x, x_frc, y_pred.detach(), slices)

    # Check physical ranges at the final (24h) step
    pred = step_preds[-1]
    assert "ERA5" in pred, "Reconstruct should produce ERA5 source key"

    t2m = pred["ERA5"]["ERA5/prognostic/2d/t2m"].squeeze().cpu().numpy()
    T3d = pred["ERA5"]["ERA5/prognostic/3d/T"].squeeze().cpu().numpy()

    print(f"\n24h step — t2m: [{t2m.min():.1f}, {t2m.max():.1f}] K")
    print(f"24h step — T (all levels): [{T3d.min():.1f}, {T3d.max():.1f}] K")

    # t2m globally cannot be below -80°C (~193 K) or above +60°C (~333 K)
    assert t2m.min() > 193, f"t2m min {t2m.min():.1f} K — looks normalized, not physical"
    assert t2m.max() < 340, f"t2m max {t2m.max():.1f} K — physically implausible"

    # 3D temperature: stratosphere (~150 K) to surface (~320 K)
    assert T3d.min() > 140, f"T min {T3d.min():.1f} K — looks normalized"
    assert T3d.max() < 340, f"T max {T3d.max():.1f} K — physically implausible"
