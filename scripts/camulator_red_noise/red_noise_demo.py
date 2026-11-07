"""Local demo of red-in-time noise on CAMulator (epoch 69, 1981 data): R1 (model.red_noise) and R2 (targeted_red_noise).

Runs real rollouts through credit.trainers.rollout_utils.run_forecast, the loop `credit rollout` uses (including
its per-forecast noise reset), from config/gen_2/camulator/camulator_sdl.yml with the glade paths swapped for
local copies. The model runs on --device; pre/postblocks run on the CPU (the fixers use float64).

--mode r2  Deterministic CAMulator (SDL noise_scale 0) plus targeted_red_noise after inverse_scale, before the
           tracer clamp: TREFHT and level-31 T share one pattern (tau 11 d), level-31 Qtot has its own (tau 14 d),
           per-cell sigma maps, mask LANDFRAC + ICEFRAC. Hooks around the block record what it injects each step.
           Reports the injected noise's lag autocorrelation against exp(-lag / tau), and the members' TREFHT spread.
--mode r1  SDL at its untrained init (the noise layers start where the Derecho run starts) with
           model.red_noise (tau 11 d on latent and pixel noise). Checks that step 1 equals the white model's
           under the same seed, then records SDL's pixel-noise state each step.
--mode control  White SDL (--noise-scale 1) or deterministic CAMulator (--noise-scale 0): global-mean drift to
           compare R1 with.

    python scripts/camulator_red_noise/red_noise_demo.py --mode r2 --members 4 --days 30 --out r2.npz
"""

import argparse
import math
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

from credit.datasets.gen_2._utils import to_calendar
from credit.datasets.gen_2.channel_utils import ChannelSchema
from credit.datasets.gen_2.multi_source import MultiSourceDataset
from credit.models import load_model
from credit.models.wxformer.stochastic_decomposition_layer import scale_sdl_noise
from credit.postblock import build_postblocks
from credit.preblock import attach_channel_schema, build_preblocks
from credit.samplers import MultiStepBatchSamplerSubset
from credit.seed import seed_everything
from credit.trainers.rollout_utils import apply_inference_overrides, run_forecast, with_inference_datetime_bounds
from credit.trainers.utils import load_state_dict_error_handler

from common import (  # noqa: E402  (scripts/camulator_red_noise/common.py)
    BASE_CONFIG,
    LOCAL_CHECKPOINT,
    LOCAL_SIGMA,
    LOCAL_STATICS,
    LOWEST,
    QTOT,
    T,
    TAU_Q_DAYS,
    TAU_T_DAYS,
    TREFHT,
    inference_postblocks,
    land_noise_block,
    swap_paths,
)

STATICS, SIGMA = LOCAL_STATICS, LOCAL_SIGMA
WHITE_TAU_DAYS = 1e-3  # phi ~ 0: the same per-step sigma with no memory, the control for R2


def demo_config(mode: str, days: int, start: str = "1981-01-01T00:00Z", white: bool = False) -> dict:
    """The SDL config with local paths and an inference-only postblock chain (no target twin; R2 for mode r2)."""
    conf = swap_paths(yaml.safe_load(open(BASE_CONFIG)))
    conf["save_loc"] = tempfile.mkdtemp(prefix="red_noise_demo_")
    conf.pop("pbs", None)
    block = None
    if mode == "r2":
        taus = (WHITE_TAU_DAYS, WHITE_TAU_DAYS) if white else (TAU_T_DAYS, TAU_Q_DAYS)
        block = land_noise_block(SIGMA, STATICS, *taus)
    conf["inference"]["postblocks"] = inference_postblocks(conf, block)
    conf["inference"]["single_forecast"] = {"start_datetime": start, "forecast_length": f"{days}d"}
    if mode == "r1":
        conf["model"]["red_noise"] = {
            "tau_hours": TAU_T_DAYS * 24,
            "timestep_hours": 6,
            "apply_to": ["latent", "pixel"],
        }
    return conf


class OnDevice(torch.nn.Module):
    """Runs the model on ``device`` and hands its output back on the CPU, where the pre/postblocks run."""

    def __init__(self, model, device, timings):
        super().__init__()
        self.model, self.device, self.timings = model, device, timings

    def forward(self, x):
        t0 = time.perf_counter()
        out = self.model(x.to(self.device)).to("cpu")
        self.timings["model forward"] += time.perf_counter() - t0
        return out


class Recorder:
    """save_output_fn for run_forecast: keeps a few physical fields per step (CPU, float32)."""

    def __init__(self):
        self.steps = []
        self.times = []

    def __call__(self, y_processed, init_time, step, fhr_per_step, save_dir, pool):
        self.times.append(time.perf_counter())
        cesm = y_processed["CESM"]
        self.steps.append(
            {
                "TREFHT": cesm[TREFHT][0, 0, 0].float().numpy().copy(),
                "T31": cesm[T][0, LOWEST, 0].float().numpy().copy(),
                "Q31": cesm[QTOT][0, LOWEST, 0].float().numpy().copy(),
            }
        )


def setup(conf, device, checkpoint):
    schema_conf = apply_inference_overrides(conf)
    ic_pre, step_pre = build_preblocks(conf, phase="ic_only"), build_preblocks(conf, phase="per_step")
    schema = ChannelSchema.load_or_from_config(schema_conf)
    attach_channel_schema(ic_pre, schema)
    attach_channel_schema(step_pre, schema)
    step_post, rollout_post = build_postblocks(conf, phase="per_step"), build_postblocks(conf, phase="post_rollout")
    timings = defaultdict(float)  # seconds per stage, summed over a forecast
    for name, block in step_post.items():
        block.register_forward_pre_hook(lambda m, a: timings.__setitem__("_start", time.perf_counter()))
        block.register_forward_hook(
            lambda m, a, o, name=name: timings.__setitem__(
                name, timings[name] + time.perf_counter() - timings["_start"]
            )
        )

    seed_everything(conf["seed"])
    model = load_model(conf)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_state_dict_error_handler(model.load_state_dict(ckpt["model_state_dict"], strict=False))
    del ckpt
    model = model.to(device).eval()

    sf = conf["inference"]["single_forecast"]
    calendar = conf["data"].get("calendar", "standard")
    init_times = [to_calendar(pd.Timestamp(sf["start_datetime"]), calendar)]
    n_steps = int(pd.Timedelta(sf["forecast_length"]) / pd.Timedelta(conf["data"]["timestep"]))
    dataset_conf = {
        **with_inference_datetime_bounds(conf["data"], init_times, n_steps, conf["data"]["timestep"]),
        "forecast_len": n_steps,
        "datetimes": init_times,
        "save_loc": conf["save_loc"],
    }
    dataset = MultiSourceDataset(dataset_conf, return_target=False)
    sampler = MultiStepBatchSamplerSubset(dataset=dataset, batch_size=1, index_subset=[0], num_forecast_steps=n_steps)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)
    return dict(
        conf=conf, n_steps=n_steps, ic_pre=ic_pre, step_pre=step_pre, step_post=step_post,
        rollout_post=rollout_post, model=model, loader=loader, calendar=getattr(dataset, "calendar", calendar),
        timings=timings,
    )  # fmt: skip


def forecast(env, model, seed, recorder, device):
    seed_everything(seed)
    timings = env["timings"]
    timings.clear()
    t0 = time.perf_counter()
    run_forecast(
        conf=env["conf"], n_steps=env["n_steps"], save_dir=env["conf"]["save_loc"], ic_preblocks=env["ic_pre"],
        step_preblocks=env["step_pre"], step_postblocks=env["step_post"], rollout_postblocks=env["rollout_post"],
        model=OnDevice(model, device, timings), batch_iter=iter(env["loader"]), device=torch.device("cpu"), pool=None,
        save_output_fn=recorder, verbose=False, calendar=env["calendar"],
    )  # fmt: skip
    total, n = time.perf_counter() - t0, env["n_steps"]
    timings.pop("_start", None)
    stages = sorted(timings.items(), key=lambda kv: -kv[1])
    other = total - sum(timings.values())
    breakdown = ", ".join(f"{k} {v / n:.3f}" for k, v in stages if v / total > 0.01)
    print(f"  {n} steps, {total / n:.2f} s/step: {breakdown}, other (data, preblocks) {other / n:.3f}")
    if len(recorder.times) > 8:
        gaps = np.diff(recorder.times)
        q = max(len(gaps) // 4, 1)
        print(f"  step interval, first / last quarter: {gaps[:q].mean():.2f} / {gaps[-q:].mean():.2f} s")


def lag_corr(series: np.ndarray, lag: int) -> float:
    """Pooled lag correlation of a (time, ...) series."""
    a, b = series[:-lag].reshape(-1), series[lag:].reshape(-1)
    return float(np.mean(a * b) / (a.std() * b.std()))


def efold_days(series: np.ndarray, max_lag: int, dt_hours: float = 6.0) -> float:
    """e-folding time from a least-squares fit of log ACF over lags 1..max_lag (through the origin)."""
    lags = np.arange(1, max_lag + 1)
    acf = np.array([lag_corr(series, k) for k in lags])
    ok = acf > 0.05
    slope = np.sum(lags[ok] * np.log(acf[ok])) / np.sum(lags[ok] ** 2)
    return -dt_hours / slope / 24.0


def run_r2(args, device):
    conf = demo_config("r2", args.days, args.start, args.white)
    tau_days = WHITE_TAU_DAYS if args.white else TAU_T_DAYS
    env = setup(conf, device, args.checkpoint)
    scale_sdl_noise(env["model"], 0.0)  # deterministic CAMulator: all spread comes from R2
    block = env["step_post"]["land_red_noise"]
    land = xr.open_dataset(STATICS)["LANDFRAC"].values > 0.9
    sigma_t = xr.open_dataset(SIGMA)["TREFHT"].values

    injected, before = [], {}
    block.register_forward_pre_hook(lambda m, a: before.update(t=a[0]["y_processed"]["CESM"][TREFHT]))
    block.register_forward_hook(
        lambda m, a, out: injected[-1].append(
            ((out["y_processed"]["CESM"][TREFHT] - before["t"])[0, 0, 0].numpy() / sigma_t)[land]
        )
    )
    members = []
    for m in range(args.members):
        injected.append([])
        rec, t0 = Recorder(), time.time()
        forecast(env, env["model"], conf["seed"] + m, rec, device)
        members.append(rec.steps)
        print(f"member {m}: {env['n_steps']} steps in {time.time() - t0:.0f} s")

    noise = np.stack([np.stack(steps) for steps in injected])  # (member, step, land cell), in units of sigma
    trefht = np.stack([np.stack([s["TREFHT"] for s in steps]) for steps in members])  # (member, step, H, W)
    t31 = np.stack([np.stack([s["T31"] for s in steps]) for steps in members])
    phi = math.exp(-6.0 / (tau_days * 24))
    lags = [1, 4, 12, 22, 44, 88]
    print(f"\nInjected TREFHT noise / sigma over land (>90% land): std {noise.std():.3f} (target 1)")
    print("lag (steps)   ACF     exp(-lag/tau)")
    for k in lags:
        if k < noise.shape[1]:
            acf = np.mean([lag_corr(noise[m], k) for m in range(len(noise))])
            print(f"{k:5d} ({k / 4:4.1f} d) {acf:6.3f}  {phi**k:6.3f}")
    if not args.white:
        tau_fit = efold_days(noise.transpose(1, 0, 2), max_lag=min(44, noise.shape[1] - 1))
        print(f"fitted e-folding time: {tau_fit:.1f} d (configured {tau_days} d)")

    lat_w = np.cos(np.deg2rad(xr.open_dataset(STATICS).latitude.values))[:, None] * np.ones_like(land)
    spread = trefht.std(axis=0, ddof=1)  # (step, H, W)
    land_rms = lambda f: np.sqrt((f**2 * lat_w * land).sum((-2, -1)) / (lat_w * land).sum())  # noqa: E731
    gmean = (trefht * lat_w).sum((-2, -1)) / lat_w.sum()
    print("\nTREFHT member spread over land (K) and per-member global-mean TREFHT range (K):")
    for day in (1, 3, 5, 10, 15, 20, 30, 45, 60):
        k = day * 4 - 1
        if k < spread.shape[0]:
            print(f"day {day:3d}: spread {land_rms(spread[k]):.3f}   global-mean spread {np.ptp(gmean[:, k]):.3f}")
    if args.out:
        np.savez_compressed(args.out, noise=noise, trefht=trefht, t31=t31, land=land, tau_days=tau_days)
        print(f"saved {args.out}")


def run_control(args, device):
    """White SDL (or, with --noise-scale 0, deterministic CAMulator): the drift and spread R1 is compared with."""
    conf = demo_config("control", args.days, args.start)
    env = setup(conf, device, args.checkpoint)
    scale_sdl_noise(env["model"], args.noise_scale)
    members = []
    for m in range(args.members):
        rec, t0 = Recorder(), time.time()
        forecast(env, env["model"], conf["seed"] + m, rec, device)
        members.append(rec.steps)
        print(f"member {m}: {env['n_steps']} steps in {time.time() - t0:.0f} s")
    trefht = np.stack([np.stack([s["TREFHT"] for s in steps]) for steps in members])
    lat_w = np.cos(np.deg2rad(xr.open_dataset(STATICS).latitude.values))[:, None] * np.ones(trefht.shape[-2:])
    gmean = (trefht * lat_w).sum((-2, -1)) / lat_w.sum()
    print(
        f"noise_scale {args.noise_scale}: finite: {np.isfinite(trefht).all()}; "
        f"global-mean TREFHT day 1 / last: {gmean[:, 3].round(2)} / {gmean[:, -1].round(2)}"
    )
    if args.out:
        np.savez_compressed(args.out, gmean_trefht=gmean, noise_scale=args.noise_scale)
        print(f"saved {args.out}")


def run_r1(args, device):
    conf = demo_config("r1", args.days, args.start)
    env = setup(conf, device, args.checkpoint)
    model = env["model"]
    sdl = [model.noise_inject1, model.noise_inject2, model.noise_inject3]
    phi = math.exp(-6.0 / (TAU_T_DAYS * 24))
    print(
        f"red SDL: phi {phi:.4f} on {len(model._latent_ar1)} latent streams and {sum(m.pixel_ar1 is not None for m in sdl)} pixel streams"
    )

    # Step 1: red equals white under the same seed (the streams start stationary and consume the same draws).
    env1 = {**env, "n_steps": 1}
    streams = ([m.pixel_ar1 for m in sdl], dict(model._latent_ar1))
    for m in sdl:
        m.pixel_ar1 = None
    model._latent_ar1 = {}
    white = Recorder()
    forecast(env1, model, conf["seed"], white, device)
    for m, s in zip(sdl, streams[0]):
        m.pixel_ar1 = s
    model._latent_ar1 = streams[1]
    red = Recorder()
    forecast(env1, model, conf["seed"], red, device)
    same = all(np.array_equal(white.steps[0][k], red.steps[0][k]) for k in white.steps[0])
    print(f"step 1, white vs red, same seed: identical = {same}")

    # Multi-step red rollouts: SDL's pixel state on a fixed sample of elements, and the latent z.
    rng = np.random.default_rng(0)
    pixel, latent, members = [], [], []
    hook_state = {}
    inject = model.noise_inject3

    def record(module, a, out):
        state = inject.pixel_ar1.state.reshape(-1)
        if "idx" not in hook_state:
            hook_state["idx"] = torch.as_tensor(rng.choice(state.numel(), 50000, replace=False), device=state.device)
        pixel[-1].append(state[hook_state["idx"]].float().cpu().numpy())
        latent[-1].append(model._latent_ar1[("decoder", 2)].state.reshape(-1).float().cpu().numpy())

    inject.register_forward_hook(record)
    for m in range(args.members):
        pixel.append([])
        latent.append([])
        rec, t0 = Recorder(), time.time()
        forecast(env, model, conf["seed"] + m, rec, device)
        members.append(rec.steps)
        print(f"member {m}: {env['n_steps']} steps in {time.time() - t0:.0f} s")
    pixel = np.stack([np.stack(p) for p in pixel])  # (member, step, element)
    latent = np.stack([np.stack(z) for z in latent])
    print(f"\nSDL pixel-noise state (noise_inject3, 50k elements): std {pixel.std():.3f} (target 1)")
    print("lag (steps)   pixel ACF  latent ACF  exp(-lag/tau)")
    for k in (1, 4, 12, 22, 44):
        if k < pixel.shape[1]:
            pa = np.mean([lag_corr(pixel[m], k) for m in range(len(pixel))])
            za = np.mean([lag_corr(latent[m], k) for m in range(len(latent))])
            print(f"{k:5d} ({k / 4:4.1f} d) {pa:8.3f}  {za:9.3f}  {phi**k:9.3f}")
    trefht = np.stack([np.stack([s["TREFHT"] for s in steps]) for steps in members])
    lat_w = np.cos(np.deg2rad(xr.open_dataset(STATICS).latitude.values))[:, None] * np.ones(trefht.shape[-2:])
    gmean = (trefht * lat_w).sum((-2, -1)) / lat_w.sum()
    print(
        f"finite: {np.isfinite(trefht).all()}; global-mean TREFHT day 1 / last: {gmean[:, 3].round(2)} / {gmean[:, -1].round(2)}"
    )
    if args.out:
        np.savez_compressed(args.out, pixel=pixel, latent=latent, gmean_trefht=gmean, tau_days=TAU_T_DAYS)
        print(f"saved {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["r1", "r2", "control"], required=True)
    p.add_argument("--noise-scale", type=float, default=1.0, help="control: SDL noise scale (0 = deterministic)")
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", default="1981-01-01T00:00Z", help="init time (1981 data only)")
    p.add_argument("--white", action="store_true", help="r2: same per-step sigma, no memory (the control)")
    p.add_argument("--checkpoint", default=LOCAL_CHECKPOINT)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default=None, help="save the series here (.npz)")
    p.add_argument("--write-config", default=None, help="only write the demo config (for credit check) and exit")
    args = p.parse_args()
    if args.write_config:
        conf = demo_config(args.mode, args.days, args.start, args.white)
        with open(args.write_config, "w") as f:
            yaml.safe_dump(conf, f, sort_keys=False)
        print(f"wrote {args.write_config}")
        return
    device = torch.device(args.device)
    {"r1": run_r1, "r2": run_r2, "control": run_control}[args.mode](args, device)


if __name__ == "__main__":
    main()
