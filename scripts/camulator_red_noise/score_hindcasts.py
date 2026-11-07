"""Score CAMulator SDL hindcast ensembles (from hindcast_configs.py) against CESM truth.

Reads <forecasts>/<arm>/member_NN/YYYYMMDD_HHMMZ/full.nc for every arm, member and start date, the CESM
state zarrs for truth, LANDFRAC (statics) and ICEFRAC at each start (forcing) for the regions. For each arm,
variable and region it accumulates over start dates, by daily lead and by weekly mean (weeks 1, 2, ...):

    rmse         ensemble-mean RMSE
    spread       sqrt(mean ensemble variance) x sqrt((M+1)/M), the small-ensemble correction, so that a
                 reliable ensemble has spread / rmse ~ 1
    ratio        spread / rmse
    crps         fair CRPS (unbiased for the M-member ensemble; MAE for one member)
    bias         area-mean ensemble-mean error (forecast - truth), the drift
    rank_hist    area-weighted rank histogram of the truth among the members (weekly; M >= 2)

and, for the noise-model comparisons (NOISE_MODEL_COMPARISON_PLAN.md):

    *_tercile    rmse, spread and ratio in thirds of each region's points, ranked by the ensemble-mean forecast
                 (a forecast quantity, so the split is fair): quiet / middle / active (question 2.3)
    brier        fair Brier score of PRECT above 2.5 and 6.25 mm per 6 h (about 10 and 25 mm/day), daily
                 leads, with the observed base rate (question 2.3)
    *_by_start   weekly area-mean squared error and ensemble variance per start date, for the spread-error
                 relation across starts (question 2.7)

--dress ARM adds the virtual arm ARM+dress: output dressing, multiplicative noise on outputs that do not feed
back. Each --dress-vars field of each member is multiplied by exp(s e - s^2 / 2), mean-preserving, with e a
unit-variance smooth random field (correlation length 500 km), drawn afresh for every member and day. A
one-member arm is copied to --dress-members members first. --dress-sigma s has no default: fit it from the
residuals before using it for science.

All area means are cos(latitude)-weighted. Regions: global, land (LANDFRAC > 0.9), ocean (LANDFRAC < 0.1 and
ice-free), seaice (ICEFRAC > 0.5 on the first lead day), tropics (|lat| <= 20), nh_extratropics, sh_extratropics.

    python scripts/camulator_red_noise/score_hindcasts.py --forecasts <out> --save scores.nc [--local]
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import xarray as xr
import yaml

from common import BASE_CONFIG, GLADE_STATICS, LOCAL_STATICS, swap_paths

VARS = ["TREFHT", "TS", "PRECT", "PS", "LHFLX", "SHFLX", "FLUT"]
REGIONS = ["global", "land", "ocean", "seaice", "tropics", "nh_extratropics", "sh_extratropics"]
SUMMARY = [("TREFHT", "land"), ("TS", "land"), ("PRECT", "land"), ("PRECT", "tropics"), ("PS", "global")]
TERCILES = ["quiet", "middle", "active"]
PRECT_THRESHOLDS = [0.0025, 0.00625]  # m per 6 h


def fair_crps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fair CRPS of the M-member ensemble x (member axis 0) for truth y; the absolute error for M = 1."""
    m = x.shape[0]
    mae = np.abs(x - y).mean(axis=0)
    if m == 1:
        return mae
    xs = np.sort(x, axis=0)
    i = np.arange(1, m + 1).reshape((m,) + (1,) * (x.ndim - 1))
    return mae - ((2 * i - m - 1) * xs).sum(axis=0) / (m * (m - 1))


def fair_brier(x: np.ndarray, y: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Fair Brier score of the event x > threshold for the M-member ensemble x (Ferro 2014), and the outcome."""
    m = x.shape[0]
    p = (x > threshold).mean(axis=0)
    o = (y > threshold).astype(np.float32)
    score = (p - o) ** 2
    if m > 1:
        score = score - p * (1 - p) / (m - 1)
    return score, o


def ranks(x: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Rank of y among the members (0..M), ties broken at random (zero precipitation ties often)."""
    below = (x < y).sum(axis=0)
    equal = (x == y).sum(axis=0)
    return below + np.floor(rng.random(y.shape) * (equal + 1)).astype(int)


def tercile_labels(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """0 / 1 / 2 for the lowest / middle / highest third of the masked points by value (-1 outside the mask).

    Thirds by rank, not by value thresholds, so the many zeros of precipitation still split evenly.
    """
    labels = np.full(values.shape, -1, dtype=np.int8)
    where = np.flatnonzero(mask)
    order = where[np.argsort(values.ravel()[where], kind="stable")]
    labels.ravel()[order] = (3 * np.arange(len(order)) // max(len(order), 1)).astype(np.int8)
    return labels


class Truth:
    """CESM fields and forcing by valid time, from the yearly (%Y) stores of the config."""

    def __init__(self, conf: dict):
        source = next(iter(conf["data"]["source"].values()))
        groups = source["variables"]
        self.var_path = {}
        for group in ("prognostic", "diagnostic"):
            for name in groups[group].get("vars_2D") or []:
                self.var_path[name] = groups[group]["path"]
        self.forcing_path = groups["dynamic_forcing"]["path"]
        self._open = {}

    def _store(self, template: str, year: int) -> xr.Dataset:
        path = template.replace("%Y", f"{year:04d}")
        if path not in self._open:
            opener = xr.open_zarr if path.endswith(".zarr") else xr.open_dataset
            self._open[path] = opener(path)
        return self._open[path]

    def _select(self, template, name, times) -> np.ndarray:
        out = []
        for year in sorted({t.year for t in times}):
            these = [t for t in times if t.year == year]
            out.append(self._store(template, year)[name].sel(time=these).values)
        return np.concatenate(out).astype(np.float32)

    def field(self, name, times) -> np.ndarray:
        return self._select(self.var_path[name], name, times)

    def icefrac(self, time) -> np.ndarray:
        return self._select(self.forcing_path, "ICEFRAC", [time])[0]


def region_weights(lat: np.ndarray, land: np.ndarray, ice: np.ndarray) -> dict:
    coslat = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(land)
    lat2d = lat[:, None] * np.ones_like(land)
    masks = {
        "global": np.ones_like(land, dtype=bool),
        "land": land > 0.9,
        "ocean": (land < 0.1) & (ice < 0.15),
        "seaice": ice > 0.5,
        "tropics": np.abs(lat2d) <= 20,
        "nh_extratropics": lat2d > 20,
        "sh_extratropics": lat2d < -20,
    }
    return {name: coslat * mask for name, mask in masks.items()}


def area_mean(field: np.ndarray, w: np.ndarray) -> np.ndarray:
    """cos-lat weighted mean over the last two axes."""
    return (field * w).sum(axis=(-2, -1)) / w.sum()


def discover(root: str) -> dict:
    """{arm: {member_dir: [init dirs]}}"""
    arms = {}
    for arm_dir in sorted(glob.glob(os.path.join(root, "*", "member_*"))):
        arm = os.path.basename(os.path.dirname(arm_dir))
        inits = sorted(
            os.path.basename(d)
            for d in glob.glob(os.path.join(arm_dir, "*Z"))
            if os.path.isfile(os.path.join(d, "full.nc"))
        )
        arms.setdefault(arm, {})[arm_dir] = inits
    return arms


class Dresser:
    """Output dressing: mean-preserving multiplicative noise exp(s e - s^2 / 2) with a smooth random field e."""

    def __init__(self, sigma: float, variables: list[str], members: int, seed: int = 0):
        import torch

        self.sigma, self.variables, self.members = sigma, variables, members
        torch.manual_seed(seed)  # the pattern draws from the global RNG; nothing else here uses it
        self.pattern = None

    def __call__(self, var: str, x: np.ndarray) -> np.ndarray:
        """x: (M, day, H, W); a one-member x is copied to ``members`` first."""
        from credit.ensemble.red_noise import SphericalAR1Pattern

        if x.shape[0] == 1:
            x = np.repeat(x, self.members, axis=0)
        if var not in self.variables:
            return x
        m, days, height, width = x.shape
        if self.pattern is None:
            self.pattern = SphericalAR1Pattern(height, width, 500.0, 0.0)  # white in time
        e = self.pattern.sample(m * days, "cpu").reshape(m, days, height, width).numpy()
        return x * np.exp(self.sigma * e - self.sigma**2 / 2)


def score_arm(members: dict, truth: Truth, land: np.ndarray, lat: np.ndarray, n_days: int, rng, dress=None) -> dict:
    inits = sorted(set.intersection(*(set(v) for v in members.values())))
    skipped = sorted(set.union(*(set(v) for v in members.values())) - set(inits))
    if skipped:
        print(f"  skipping {len(skipped)} start dates missing in some members: {skipped[:3]}...")
    m = len(members) if dress is None else max(len(members), dress.members)
    n_weeks = n_days // 7
    acc = defaultdict(float)  # (stat, var, region, lead kind, index[, tercile]) -> sum over starts
    hist = defaultdict(lambda: np.zeros(m + 1))
    by_start = defaultdict(list)  # (stat, var, region, week) -> per-start values, in init order
    for init in inits:
        files = [xr.open_dataset(os.path.join(d, init, "full.nc")) for d in members]
        times = list(files[0].time.values[:n_days])
        if len(times) < n_days:
            raise ValueError(f"{init}: only {len(times)} daily outputs, need {n_days} (--days)")
        weights = region_weights(lat, land, truth.icefrac(times[0]))  # prescribed; slow enough to fix per start
        for var in VARS:
            x = np.stack([f[var].values[:n_days] for f in files]).astype(np.float32)  # (M, day, H, W)
            if dress is not None:
                x = dress(var, x)
            y = truth.field(var, times)  # (day, H, W)
            windows = [("day", d, x[:, d], y[d]) for d in range(n_days)]
            windows += [
                ("week", k, x[:, 7 * k : 7 * k + 7].mean(axis=1), y[7 * k : 7 * k + 7].mean(axis=0))
                for k in range(n_weeks)
            ]
            for kind, k, xf, yf in windows:
                mean = xf.mean(axis=0)
                var_m = xf.var(axis=0, ddof=1) if m > 1 else np.zeros_like(yf)
                err2 = (mean - yf) ** 2
                crps = fair_crps(xf, yf)
                rank = ranks(xf, yf, rng) if (kind == "week" and m > 1) else None
                for region, w in weights.items():
                    if w.sum() == 0:
                        continue
                    key = (var, region, kind, k)
                    acc[("mse", *key)] += area_mean(err2, w)
                    acc[("var", *key)] += area_mean(var_m, w)
                    acc[("crps", *key)] += area_mean(crps, w)
                    acc[("bias", *key)] += area_mean(mean - yf, w)
                    acc[("n", *key)] += 1
                    if rank is not None:
                        hist[var, region, k] += np.bincount(rank.ravel(), weights=w.ravel(), minlength=m + 1)
                    labels = tercile_labels(mean, w > 0)
                    for t in range(3):
                        wt = w * (labels == t)
                        if wt.sum() > 0:
                            acc[("mse_t", *key, t)] += area_mean(err2, wt)
                            acc[("var_t", *key, t)] += area_mean(var_m, wt)
                            acc[("n_t", *key, t)] += 1
                    if kind == "week":
                        by_start["mse", var, region, k].append(float(area_mean(err2, w)))
                        by_start["var", var, region, k].append(float(area_mean(var_m, w)))
                    if kind == "day" and var == "PRECT":
                        for j, threshold in enumerate(PRECT_THRESHOLDS):
                            score, outcome = fair_brier(xf, yf, threshold)
                            acc["brier", j, region, k] += area_mean(score, w)
                            acc["base_rate", j, region, k] += area_mean(outcome, w)
                            acc["n_brier", j, region, k] += 1
        for f in files:
            f.close()
    return {"acc": acc, "hist": hist, "by_start": by_start, "members": m, "inits": inits, "n_weeks": n_weeks}


def to_dataset(results: dict, n_days: int) -> xr.Dataset:
    arms = list(results)
    n_weeks = n_days // 7
    max_m = max(r["members"] for r in results.values())
    starts = sorted(set().union(*(r["inits"] for r in results.values())))
    coords = {"arm": arms, "variable": VARS, "region": REGIONS}
    out = {}
    for kind, n_lead, dim in (("day", n_days, "lead_day"), ("week", n_weeks, "week")):
        shape = (len(arms), len(VARS), len(REGIONS), n_lead)
        stats = {s: np.full(shape, np.nan) for s in ("rmse", "spread", "ratio", "crps", "bias")}
        tstats = {s: np.full(shape + (3,), np.nan) for s in ("rmse_tercile", "spread_tercile", "ratio_tercile")}
        for a, arm in enumerate(arms):
            r = results[arm]
            acc = r["acc"]
            factor = np.sqrt((r["members"] + 1) / r["members"]) if r["members"] > 1 else 0.0
            for v, var in enumerate(VARS):
                for g, region in enumerate(REGIONS):
                    for k in range(n_lead):
                        key = (var, region, kind, k)
                        n = acc.get(("n", *key), 0)
                        if not n:
                            continue
                        rmse = np.sqrt(acc[("mse", *key)] / n)
                        spread = np.sqrt(acc[("var", *key)] / n) * factor
                        stats["rmse"][a, v, g, k] = rmse
                        stats["spread"][a, v, g, k] = spread
                        stats["ratio"][a, v, g, k] = spread / rmse if rmse > 0 else np.nan
                        stats["crps"][a, v, g, k] = acc[("crps", *key)] / n
                        stats["bias"][a, v, g, k] = acc[("bias", *key)] / n
                        for t in range(3):
                            nt = acc.get(("n_t", *key, t), 0)
                            if not nt:
                                continue
                            rmse_t = np.sqrt(acc[("mse_t", *key, t)] / nt)
                            spread_t = np.sqrt(acc[("var_t", *key, t)] / nt) * factor
                            tstats["rmse_tercile"][a, v, g, k, t] = rmse_t
                            tstats["spread_tercile"][a, v, g, k, t] = spread_t
                            tstats["ratio_tercile"][a, v, g, k, t] = spread_t / rmse_t if rmse_t > 0 else np.nan
        suffix = "" if kind == "day" else "_weekly"
        for s, values in stats.items():
            out[s + suffix] = (("arm", "variable", "region", dim), values)
        for s, values in tstats.items():
            out[s + suffix] = (("arm", "variable", "region", dim, "tercile"), values)

    hist = np.full((len(arms), len(VARS), len(REGIONS), n_weeks, max_m + 1), np.nan)
    brier = np.full((len(arms), len(PRECT_THRESHOLDS), len(REGIONS), n_days), np.nan)
    base_rate = np.full_like(brier, np.nan)
    mse_start = np.full((len(arms), len(VARS), len(REGIONS), n_weeks, len(starts)), np.nan)
    var_start = np.full_like(mse_start, np.nan)
    for a, arm in enumerate(arms):
        r = results[arm]
        for (var, region, k), counts in r["hist"].items():
            hist[a, VARS.index(var), REGIONS.index(region), k, : len(counts)] = counts / counts.sum()
        for j in range(len(PRECT_THRESHOLDS)):
            for g, region in enumerate(REGIONS):
                for k in range(n_days):
                    n = r["acc"].get(("n_brier", j, region, k), 0)
                    if n:
                        brier[a, j, g, k] = r["acc"]["brier", j, region, k] / n
                        base_rate[a, j, g, k] = r["acc"]["base_rate", j, region, k] / n
        columns = [starts.index(i) for i in r["inits"]]
        for (stat, var, region, k), values in r["by_start"].items():
            target = mse_start if stat == "mse" else var_start
            target[a, VARS.index(var), REGIONS.index(region), k, columns] = values
    out["rank_hist_weekly"] = (("arm", "variable", "region", "week", "rank"), hist)
    out["brier_prect"] = (("arm", "threshold", "region", "lead_day"), brier)
    out["base_rate_prect"] = (("arm", "threshold", "region", "lead_day"), base_rate)
    out["mse_by_start_weekly"] = (("arm", "variable", "region", "week", "start"), mse_start)
    out["var_by_start_weekly"] = (("arm", "variable", "region", "week", "start"), var_start)
    ds = xr.Dataset(
        out,
        coords={
            **coords,
            "lead_day": np.arange(1, n_days + 1),
            "week": np.arange(1, n_weeks + 1),
            "tercile": TERCILES,
            "threshold": PRECT_THRESHOLDS,
            "start": starts,
        },
    )
    ds["members"] = ("arm", [results[a]["members"] for a in arms])
    ds["n_starts"] = ("arm", [len(results[a]["inits"]) for a in arms])
    ds["threshold"].attrs["units"] = "m per 6 h (PRECT)"
    return ds


def spread_error_correlation(ds: xr.Dataset, arm: str, var: str, region: str, week: int) -> float:
    """Correlation across start dates of the ensemble spread and the ensemble-mean error (question 2.7)."""
    sel = dict(arm=arm, variable=var, region=region, week=week)
    spread = np.sqrt(ds.var_by_start_weekly.sel(**sel).values)
    error = np.sqrt(ds.mse_by_start_weekly.sel(**sel).values)
    ok = np.isfinite(spread) & np.isfinite(error)
    if ok.sum() < 3 or spread[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(spread[ok], error[ok])[0, 1])


def summary(ds: xr.Dataset) -> None:
    width = max(len(str(a)) for a in ds.arm.values)
    for var, region in SUMMARY:
        print(f"\n{var} over {region}, weekly means: rmse / spread-skill ratio / CRPS (bias)")
        for arm in ds.arm.values:
            sel = ds.sel(arm=arm, variable=var, region=region)
            cells = [
                f"w{int(w)} {float(sel.rmse_weekly.sel(week=w)):.3g}/{float(sel.ratio_weekly.sel(week=w)):.2f}/"
                f"{float(sel.crps_weekly.sel(week=w)):.3g} ({float(sel.bias_weekly.sel(week=w)):+.2g})"
                for w in ds.week.values
            ]
            print(f"  {arm:{width}s} M={int(ds.members.sel(arm=arm)):2d}  " + "  ".join(cells))

    days = [d for d in (1, 3, 7, 14) if d in ds.lead_day.values]
    print("\nPRECT over the tropics, spread-skill ratio by forecast tercile (quiet/middle/active), daily leads")
    for arm in ds.arm.values:
        sel = ds.ratio_tercile.sel(arm=arm, variable="PRECT", region="tropics")
        cells = ["d{} {}".format(d, "/".join(f"{v:.2f}" for v in sel.sel(lead_day=d).values)) for d in days]
        print(f"  {arm:{width}s}  " + "  ".join(cells))

    print(f"\nPRECT > {PRECT_THRESHOLDS[0]} m per 6 h over the tropics: fair Brier score (base rate)")
    for arm in ds.arm.values:
        sel = ds.sel(arm=arm, threshold=PRECT_THRESHOLDS[0], region="tropics")
        cells = [
            f"d{d} {float(sel.brier_prect.sel(lead_day=d)):.3f} ({float(sel.base_rate_prect.sel(lead_day=d)):.3f})"
            for d in days
        ]
        print(f"  {arm:{width}s}  " + "  ".join(cells))

    print("\nTREFHT over land: correlation of spread and error across start dates, by week")
    for arm in ds.arm.values:
        cells = [
            f"w{int(w)} {spread_error_correlation(ds, arm, 'TREFHT', 'land', int(w)):+.2f}" for w in ds.week.values
        ]
        print(f"  {arm:{width}s}  " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forecasts", required=True, help="the --out of hindcast_configs.py")
    p.add_argument("--arms", default=None, help="comma-separated subset (default: all found)")
    p.add_argument("--days", type=int, default=None, help="leads to score (default: the shortest forecast)")
    p.add_argument("--base", default=BASE_CONFIG, help="config whose data paths give the truth")
    p.add_argument("--statics", default=None, help="file with LANDFRAC (default: glade or local statics)")
    p.add_argument("--local", action="store_true", help="swap glade paths for local copies")
    p.add_argument("--dress", action="append", default=[], help="add ARM+dress, output dressing of ARM")
    p.add_argument("--dress-sigma", type=float, default=None, help="dressing amplitude s (required with --dress)")
    p.add_argument("--dress-vars", default="PRECT", help="comma-separated fields to dress")
    p.add_argument("--dress-members", type=int, default=10, help="members for a dressed one-member arm")
    p.add_argument("--save", default=None, help="write the scores here (.nc)")
    args = p.parse_args()
    if args.dress and args.dress_sigma is None:
        p.error("--dress needs --dress-sigma")

    conf = yaml.safe_load(open(args.base))
    if args.local:
        conf = swap_paths(conf)
    truth = Truth(conf)
    statics = xr.open_dataset(args.statics or (LOCAL_STATICS if args.local else GLADE_STATICS))
    land, lat = statics["LANDFRAC"].values.astype(np.float32), statics["latitude"].values

    found = discover(args.forecasts)
    arms = [a for a in (args.arms.split(",") if args.arms else found) if a in found]
    if not arms:
        raise SystemExit(f"no <arm>/member_*/<init>/full.nc under {args.forecasts}")
    missing = [a for a in args.dress if a not in found]
    if missing:
        raise SystemExit(f"--dress arms not found: {missing}")
    n_days = args.days
    if n_days is None:  # the shortest forecast among the first start date of every arm
        lengths = []
        for arm in arms:
            member, inits = next(iter(found[arm].items()))
            with xr.open_dataset(os.path.join(member, inits[0], "full.nc")) as f:
                lengths.append(f.sizes["time"])
        n_days = min(lengths)

    rng = np.random.default_rng(0)
    results = {}
    for arm in arms:
        print(f"{arm}: {len(found[arm])} members")
        results[arm] = score_arm(found[arm], truth, land, lat, n_days, rng)
    for i, arm in enumerate(args.dress):
        dresser = Dresser(args.dress_sigma, args.dress_vars.split(","), args.dress_members, seed=i)
        print(f"{arm}+dress: s = {args.dress_sigma} on {dresser.variables}")
        results[f"{arm}+dress"] = score_arm(found[arm], truth, land, lat, n_days, rng, dress=dresser)
    ds = to_dataset(results, n_days)
    summary(ds)
    if args.save:
        ds.to_netcdf(args.save)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
