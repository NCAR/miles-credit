"""Spread of the SDL ensemble vs CAMulator's own error over one 6 h step, on local 1981 data.

Builds the dataset, pre/postblocks and model the way train_gen2 does from the SDL config, with the glade
paths swapped for the local copies (same md5s). For each start time it runs one zero-noise forward (the
deterministic CAMulator) and M noisy members, each through the per-step postblocks, and compares
y_processed with the target twin y_target_processed in physical units, latitude-weighted.

Seeding before load_model matches train_gen2, so with the base checkpoint the noise layers start exactly
where the Derecho run starts. Pass a fine-tuned checkpoint to see what training changed.
"""

import argparse
import copy
import os
import tempfile
import time

import numpy as np
import torch
import xarray as xr
import yaml

from credit.datasets.gen_2.channel_utils import ChannelSchema
from credit.models import load_model
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer
from credit.postblock import apply_postblocks, build_postblocks
from credit.preblock import apply_preblocks, attach_channel_schema, build_preblocks
from credit.seed import seed_everything
from credit.trainers.rollout_utils import apply_rollout_renames
from credit.trainers.utils import inject_flat_var_keys, load_dataloader, load_dataset, load_state_dict_error_handler

LOCAL = os.path.expanduser("~/PycharmProjects/miles-credit/climate")
CAMPAIGN = "/glade/campaign/cisl/aiml/credit/cesm_exercise"
STAGING = "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING"
SWAP = {
    f"{CAMPAIGN}/data/b.e21.CREDIT_climate_branch_1980_%Y_zmdata_ERA5scaled_zmdata_Qtot.zarr": f"{LOCAL}/tutorial_data/state_%Y.zarr",
    # tutorial_data/forcing_%Y.nc is a dangling symlink; this file matches the campaign zarr (checked 2026-11-05)
    f"{CAMPAIGN}/forcing/dynamic_forcing/%Y_forcing.zarr": f"{LOCAL}/data/forcing_real_%Y.nc",
    f"{CAMPAIGN}/static/statics_b_credit_runs_f32_02.nc": f"{LOCAL}/assets/statics_b_credit_runs_f32_02.nc",
    f"{CAMPAIGN}/scaler_spatial_PS.json": f"{LOCAL}/data/scaler_spatial_PS.json",
    f"{CAMPAIGN}/camulator_metadata.yaml": f"{LOCAL}/camulator_metadata.yaml",
    f"{STAGING}/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc": (
        f"{LOCAL}/assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
    ),
}
REF_LEVELS_HPA = (200, 500, 850)


def swap_paths(node):
    if isinstance(node, dict):
        return {k: swap_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [swap_paths(v) for v in node]
    return SWAP.get(node, node) if isinstance(node, str) else node


def glade_strings(node, where=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from glade_strings(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from glade_strings(v, f"{where}[{i}]")
    elif isinstance(node, str) and "/glade/" in node:
        yield where, node


def leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{prefix}{k}/")
        elif torch.is_tensor(v):
            yield f"{prefix}{k}", v


def as_levels(t):
    """(B, ..., H, W) -> (B, L, H, W) float64 on CPU."""
    t = t.detach().to("cpu", torch.float64)
    return t.reshape(t.shape[0], -1, *t.shape[-2:])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=f"{LOCAL}/assets/checkpoint.pt00069.pt")
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4, help="members per forward pass")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default=None, help="save the accumulated statistics here (.npz)")
    args = p.parse_args()

    conf = swap_paths(yaml.safe_load(open(args.config)))
    work = tempfile.mkdtemp(prefix="sdl_spread_check_")  # save_loc: the trainer-side blocks write a schema here
    conf["save_loc"] = work
    conf["data"]["start_datetime"], conf["data"]["end_datetime"] = "1981-01-01", "1981-12-31"
    conf["trainer"].update(train_batch_size=1, thread_workers=0)
    conf.pop("pbs", None)
    conf.pop("inference", None)
    conf.pop("validation_data", None)
    left = [(w, s) for w, s in glade_strings(conf) if not w.startswith(".loss.variance")]
    assert not left, f"unswapped glade paths: {left}"

    inject_flat_var_keys(conf)
    ic_pre = build_preblocks(conf, phase="ic_only")
    step_pre = build_preblocks(conf, phase="per_step")
    schema = ChannelSchema.load_or_from_config(conf)
    attach_channel_schema(ic_pre, schema)
    attach_channel_schema(step_pre, schema)
    post = build_postblocks(conf, phase="per_step")

    loader = load_dataloader(conf, load_dataset(conf, is_train=True), rank=0, world_size=1, is_train=True)

    # train_gen2 order: seed, build on CPU, move, load weights (strict=False: the noise layers are new).
    seed_everything(conf.get("seed", 42))
    model = load_model(conf)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_state_dict_error_handler(model.load_state_dict(ckpt["model_state_dict"], strict=False))
    model.to(args.device).eval()
    sdl = [m for m in model.modules() if isinstance(m, StochasticDecompositionLayer)]
    trained_factors = [m.noise_factor.data.clone() for m in sdl]
    print(
        f"checkpoint {args.checkpoint} (epoch {ckpt.get('epoch')}); {len(sdl)} SDL layers, "
        f"noise_factor {[round(float(f), 4) for f in trained_factors]}; device {args.device}"
    )
    del ckpt

    z = xr.open_zarr(f"{LOCAL}/tutorial_data/state_1981.zarr")
    first = {"time": 0} if "time" in z.hyam.dims else {}
    p_ref = (
        z.hyam.isel(first).values * float(z.P0.isel(first if "time" in z.P0.dims else {}).values)
        + z.hybm.isel(first).values * 1.0e5
    ) / 100.0
    lat_w = torch.from_numpy(np.cos(np.deg2rad(z.latitude.values))).double()
    lat_w = (lat_w / lat_w.mean())[:, None]

    def wmean(f):  # (..., H, W) -> (...)
        return (f * lat_w).mean(dim=(-2, -1))

    def forward(x):
        with torch.no_grad():
            y = model(x.to(args.device)).to("cpu")
        return y.flatten(1, 2) if y.dim() == 5 else y

    def postprocess(base, y_pred):
        d = copy.deepcopy(base)
        d["y_pred"] = y_pred
        with torch.no_grad():
            d = apply_postblocks(post, d)
        return dict(leaves(d["y_processed"])), dict(leaves(d["y_target_processed"]))

    sums, it = {}, iter(loader)
    torch.manual_seed(0)
    t0 = time.time()
    for i in range(args.n_init):
        batch = next(it)
        base = {"ic_raw": batch["input"], "x_raw": batch["input"], "y_raw": batch["target"]}
        step_input = apply_rollout_renames(apply_preblocks(ic_pre, batch, device="cpu"), step_pre)
        base["x_physical"] = copy.deepcopy(step_input["input"])
        base.update(apply_preblocks(step_pre, step_input, device="cpu"))
        x = base["x"]

        for m in sdl:
            m.noise_factor.data.zero_()
        det, truth = postprocess(base, forward(x))
        for m, f in zip(sdl, trained_factors):
            m.noise_factor.data.copy_(f)

        members = []
        for start in range(0, args.members, args.chunk):
            n = min(args.chunk, args.members - start)
            y = forward(x.repeat(n, *([1] * (x.dim() - 1))))
            members += [postprocess(base, y[j : j + 1])[0] for j in range(n)]

        if i == 0:
            missing = sorted(set(det) ^ set(truth))
            assert not missing, f"prediction/target keys differ: {missing}"
            print("variables:", ", ".join(f"{k} {tuple(v.shape)}" for k, v in det.items()))
        for k in det:
            tr = as_levels(truth[k])[0]
            dt = as_levels(det[k])[0]
            ens = torch.cat([as_levels(mb[k]) for mb in members])
            stats = {
                "det_se": wmean((dt - tr) ** 2),
                "ensmean_se": wmean((ens.mean(0) - tr) ** 2),
                "spread_var": wmean(ens.var(0, unbiased=True)),
                "member_det_diff_sq": wmean(((ens - dt) ** 2).mean(0)),
            }
            for s, v in stats.items():
                sums.setdefault(k, {}).setdefault(s, torch.zeros_like(v))
                sums[k][s] += v
        print(f"start {i + 1}/{args.n_init} done ({time.time() - t0:.0f} s)", flush=True)

    M, n = args.members, args.n_init
    lev_idx = {hpa: int(np.abs(p_ref - hpa).argmin()) for hpa in REF_LEVELS_HPA}
    print(f"\n{n} start times x {M} members, one 6 h step, 1981 (inside the base model's training years).")
    print("RMSE and spread are latitude-weighted, physical units. SSR = spread / ens-mean RMSE * sqrt((M+1)/M).")
    print("noise/err = RMS member-minus-deterministic / deterministic RMSE (how big the noise is vs the error).\n")
    hdr = f"{'variable':26s} {'det RMSE':>10s} {'ensm RMSE':>10s} {'spread':>10s} {'SSR':>6s} {'noise/err':>9s}"
    print(hdr)
    print("-" * len(hdr))

    def row(name, s, idx=None):
        g = {k: (v if idx is None else v[idx : idx + 1]).mean() / n for k, v in s.items()}
        det, ensm, spr = (float(g[k].sqrt()) for k in ("det_se", "ensmean_se", "spread_var"))
        ssr = spr / ensm * np.sqrt((M + 1) / M) if ensm > 0 else float("nan")
        ratio = float(g["member_det_diff_sq"].sqrt()) / det if det > 0 else float("nan")
        print(f"{name:26s} {det:10.4g} {ensm:10.4g} {spr:10.4g} {ssr:6.2f} {ratio:9.2f}")

    for k, s in sums.items():
        short = "/".join(k.split("/")[-2:])
        if s["det_se"].numel() > 1:
            row(f"{short} (level mean)", s)
            for li in lev_idx.values():
                row(f"{short} {p_ref[li]:.0f} hPa [{li}]", s, li)
        else:
            row(short, s)

    if args.out:
        np.savez(
            args.out,
            **{f"{k}|{s}": v.numpy() for k, d in sums.items() for s, v in d.items()},
            n_init=n,
            members=M,
            p_ref=p_ref,
        )
        print("\nsaved", args.out)


if __name__ == "__main__":
    main()
