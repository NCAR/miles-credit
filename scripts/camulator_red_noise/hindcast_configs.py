"""Write one `credit rollout` config per (arm, member) for CAMulator SDL hindcasts, plus a Casper submit script.

Every member is an ordinary gen2 batch rollout over the same start dates with its own seed and output
directory, so the noise comes only from the model and the noise blocks. Members share seeds across arms
(common random numbers), so arm differences are paired. Output: 7 near-surface fields at 00Z only (PRECT is
the 6-h accumulation ending at 00Z), one compressed file per start date
(<out>/<arm>/member_NN/YYYYMMDD_HHMMZ/full.nc), which score_hindcasts.py reads. Arms are named after the
noise models of NOISE_MODEL_COMPARISON_PLAN.md:

    plain              plain CAMulator (the SDL checkpoint with noise_scale 0; frozen base); one member
    plain_kick         plain CAMulator from perturbed starts (start_perturbation, --kick-amplitude)
    sdl                white SDL as trained
    sdl_kick           white SDL from perturbed starts
    sdl_red            red SDL: model.red_noise on its latent and pixel noise (--red-sdl-tau-days)
    sdl_land           white SDL + land noise (targeted_red_noise on land and sea ice)
    plain_land         plain CAMulator + land noise
    sdl_land_white     white SDL + white land noise, amplitude x --white-land-scale (spread-matched)
    plain_land_white   plain CAMulator + white land noise, the same scaling
    plain_sppt         plain CAMulator + tendency noise (T, Qtot, U, V below ~100 hPa)
    sdl_sppt           white SDL + tendency noise
    plain_sppt_thermo  plain CAMulator + tendency noise on T and Qtot only
    plain_land_sppt    plain CAMulator + tendency noise + land noise

First wave (the default --arms): plain, plain_kick, sdl, sdl_red, sdl_land, plain_land, plain_sppt.

Casper (paths on glade, `credit submit --cluster casper --mode rollout` per config):

    python scripts/camulator_red_noise/hindcast_configs.py \\
        --save-loc /glade/derecho/scratch/$USER/CREDIT_runs/camulator_sdl \\
        --out /glade/derecho/scratch/$USER/CREDIT_runs/camulator_sdl_hindcast \\
        --sigma-file /glade/work/$USER/camulator_red_noise/red_noise_sigma_camulator.nc \\
        --conda /glade/work/$USER/conda-envs/credit-sdl --members 10

Local test (glade paths swapped for local copies; run each config with `credit rollout -c <config>`):

    python scripts/camulator_red_noise/hindcast_configs.py --local --save-loc <dir with checkpoint.pt> \\
        --out <dir> --arms plain,plain_kick,plain_sppt --members 3 --first-init 1981-01-01 \\
        --last-init 1981-01-08 --length 14d --kick-years 1981 1981
"""

import argparse
import copy
import os

import yaml

from common import (
    BASE_CONFIG,
    GLADE_STATICS,
    KICK_YEARS,
    LOCAL_SIGMA,
    LOCAL_STATICS,
    TAU_Q_DAYS,
    TAU_T_DAYS,
    WHITE_TAU_DAYS,
    inference_postblocks,
    land_noise_block,
    scaled_sigma_file,
    start_perturbation_block,
    swap_paths,
    tendency_block,
)

PLAIN = {"noise_scale": 0.0}
ARMS = {
    "plain": PLAIN,
    "plain_kick": {**PLAIN, "kick": True},
    "sdl": {},
    "sdl_kick": {"kick": True},
    "sdl_red": {"red_sdl": True},
    "sdl_land": {"land": "red"},
    "plain_land": {**PLAIN, "land": "red"},
    "sdl_land_white": {"land": "white"},
    "plain_land_white": {**PLAIN, "land": "white"},
    "plain_sppt": {**PLAIN, "sppt": "full"},
    "sdl_sppt": {"sppt": "full"},
    "plain_sppt_thermo": {**PLAIN, "sppt": "thermo"},
    "plain_land_sppt": {**PLAIN, "land": "red", "sppt": "full"},
}
FIRST_WAVE = "plain,plain_kick,sdl,sdl_red,sdl_land,plain_land,plain_sppt"
# Kick size: about one step of error. On 1981 states, one-step RMSE / RMS((s1 - s2) / sqrt 2) is about 0.21 for
# T, 0.12 for U, 0.26 for Qtot, 0.28 for TREFHT and 0.06 for PS; one amplitude keeps the kick balanced.
KICK_AMPLITUDE = 0.15
OUTPUT_VARS = ["TREFHT", "TS", "PRECT", "PS", "LHFLX", "SHFLX", "FLUT"]


def output_block(conf: dict) -> dict:
    """Seven near-surface 2-D fields at 00Z, one compressed file per start date."""
    sources = conf["data"]["source"]
    keys = []
    for name in OUTPUT_VARS:
        for src, src_conf in sources.items():
            for group, grp in src_conf["variables"].items():
                if name in ((grp or {}).get("vars_2D") or []) and group in ("prognostic", "diagnostic"):
                    keys.append(f"{src}/{group}/2d/{name}")
    return {
        "format": "netcdf",
        "output_interval": "24h",
        "group_by": "full",
        "metadata": conf["inference"]["output"]["metadata"],
        "variables": [{"name": k} for k in keys],
        "encoding": {"dtype": "float32", "zlib": True, "complevel": 1},
    }


def arm_config(base: dict, arm: str, member: int, args) -> dict:
    spec = ARMS[arm]
    conf = copy.deepcopy(base)
    conf["save_loc"] = args.save_loc
    conf["seed"] = args.seed + member  # the same seeds in every arm: paired members
    conf.pop("validation_data", None)

    noise_blocks = {}
    if spec.get("sppt"):
        noise_blocks["tendency_noise"] = tendency_block(thermo_only=spec["sppt"] == "thermo")
    if spec.get("land") == "red":
        noise_blocks["land_red_noise"] = land_noise_block(
            args.sigma_file, args.land_file, args.land_tau_t_days, args.land_tau_q_days
        )
    elif spec.get("land") == "white":
        sigma = scaled_sigma_file(args.sigma_file, args.white_land_scale, os.path.join(args.out, "sigma"))
        noise_blocks["land_white_noise"] = land_noise_block(sigma, args.land_file, WHITE_TAU_DAYS, WHITE_TAU_DAYS)
    if spec.get("red_sdl"):
        conf["model"]["red_noise"] = {"tau_hours": args.red_sdl_tau_days * 24, "timestep_hours": 6}
    if spec.get("kick"):
        path = conf["data"]["source"]["CESM"]["variables"]["prognostic"]["path"]
        kick = start_perturbation_block(path, args.kick_years, args.kick_amplitude, conf["seed"])
        conf["preblocks"] = {"ic_only": {"start_perturbation": kick}, **conf["preblocks"]}

    inference = {
        "run_mode": "batch",
        "mode": "none",
        "save_forecast": os.path.join(args.out, arm, f"member_{member:02d}"),
        "batch_forecast": {
            "forecast_length": args.length,
            "first_init_date": args.first_init,
            "last_init_date": args.last_init,
            "init_interval": args.init_interval,
        },
        "output": output_block(conf),
        "postblocks": inference_postblocks(conf, noise_blocks),
    }
    if "noise_scale" in spec:
        inference["noise_scale"] = spec["noise_scale"]
    conf["inference"] = inference

    if args.local:
        conf.pop("pbs", None)
    else:
        conf["pbs"] = {
            "conda": args.conda,
            "project": args.project,
            "job_name": f"{arm}_{member:02d}",  # credit submit keeps the first 10 characters
            "walltime": args.walltime,
            "ncpus": 8,
            "ngpus": 1,
            "queue": "casper",
        }
        if args.gpu_type:
            conf["pbs"]["gpu_type"] = args.gpu_type
    return conf


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default=BASE_CONFIG, help="training config to derive from")
    p.add_argument("--save-loc", required=True, help="directory holding the checkpoint.pt to roll out")
    p.add_argument("--out", required=True, help="forecast root; configs go to <out>/configs")
    p.add_argument("--arms", default=FIRST_WAVE, help=f"comma-separated, from {list(ARMS)}")
    p.add_argument("--members", type=int, default=10, help="members per noisy arm (plain gets one)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--first-init", default="2013-01-01")
    p.add_argument("--last-init", default="2014-11-18", help="last start date (the CESM data ends 2014-12-31)")
    p.add_argument("--init-interval", default="7d")
    p.add_argument("--length", default="42d")
    p.add_argument("--red-sdl-tau-days", type=float, default=TAU_T_DAYS)
    p.add_argument("--land-tau-t-days", type=float, default=TAU_T_DAYS)
    p.add_argument("--land-tau-q-days", type=float, default=TAU_Q_DAYS)
    p.add_argument(
        "--white-land-scale",
        type=float,
        default=2.4,
        help="white land noise amplitude / red's; 2.4 matched day 3-10 land TREFHT spread in the local runs",
    )
    p.add_argument("--kick-amplitude", type=float, default=KICK_AMPLITUDE)
    p.add_argument("--kick-years", type=int, nargs=2, default=list(KICK_YEARS), help="donor years, first last")
    p.add_argument("--sigma-file", default=None, help="per-cell sigma maps (build_sigma_maps.py)")
    p.add_argument("--land-file", default=None, help="statics file with LANDFRAC (default: the config's)")
    p.add_argument("--local", action="store_true", help="swap glade paths for local copies; no pbs block")
    p.add_argument("--conda", default=None, help="Casper conda env (required unless --local)")
    p.add_argument("--project", default="NAML0001")
    p.add_argument("--walltime", default="06:00:00")
    p.add_argument("--gpu-type", default=None, help="Casper gpu_type; unset = any GPU (shorter waits)")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        p.error(f"unknown arms {sorted(unknown)}; choose from {list(ARMS)}")
    if not args.local and not args.conda:
        p.error("--conda is required for Casper configs")
    if any(ARMS[a].get("land") for a in arms) and not args.sigma_file:
        if not args.local:
            p.error("--sigma-file is required for the land noise arms")
        args.sigma_file = LOCAL_SIGMA
    args.land_file = args.land_file or (LOCAL_STATICS if args.local else GLADE_STATICS)

    base = yaml.safe_load(open(args.base))
    if args.local:
        base = swap_paths(base)
    config_dir = os.path.join(args.out, "configs")
    os.makedirs(config_dir, exist_ok=True)
    written = []
    for arm in arms:
        n = 1 if ARMS[arm] == PLAIN else args.members
        for m in range(n):
            path = os.path.join(config_dir, f"{arm}_m{m:02d}.yml")
            with open(path, "w") as f:
                yaml.safe_dump(arm_config(base, arm, m, args), f, sort_keys=False)
            written.append(path)

    submit = os.path.join(args.out, "submit_casper.sh")
    with open(submit, "w") as f:
        f.write("#!/bin/bash\n# One single-GPU Casper job per member. Review one with --dry-run first.\nset -e\n")
        for path in written:
            f.write(f'credit submit --cluster casper --mode rollout -c {path} "$@"\n')
    os.chmod(submit, 0o755)
    print(f"wrote {len(written)} configs to {config_dir} and {submit}")


if __name__ == "__main__":
    main()
