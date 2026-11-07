"""Shared pieces for the CAMulator noise scripts: local path swaps, the inference postblock chain, the noise blocks."""

import copy
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE_CONFIG = os.path.join(REPO, "config/gen_2/camulator/camulator_sdl.yml")

LOCAL = os.path.expanduser("~/PycharmProjects/miles-credit/climate")
CAMPAIGN = "/glade/campaign/cisl/aiml/credit/cesm_exercise"
STAGING = "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING"
GLADE_STATICS = f"{CAMPAIGN}/static/statics_b_credit_runs_f32_02.nc"
LOCAL_STATICS = f"{LOCAL}/assets/statics_b_credit_runs_f32_02.nc"
LOCAL_SIGMA = os.path.join(HERE, "red_noise_sigma_camulator.nc")
LOCAL_CHECKPOINT = f"{LOCAL}/assets/checkpoint.pt00069.pt"
SWAP = {
    f"{CAMPAIGN}/data/b.e21.CREDIT_climate_branch_1980_%Y_zmdata_ERA5scaled_zmdata_Qtot.zarr": f"{LOCAL}/tutorial_data/state_%Y.zarr",
    # tutorial_data/forcing_%Y.nc is a dangling symlink; this file matches the campaign zarr (checked 2026-11-05)
    f"{CAMPAIGN}/forcing/dynamic_forcing/%Y_forcing.zarr": f"{LOCAL}/data/forcing_real_%Y.nc",
    GLADE_STATICS: LOCAL_STATICS,
    f"{CAMPAIGN}/scaler_spatial_PS.json": f"{LOCAL}/data/scaler_spatial_PS.json",
    f"{CAMPAIGN}/camulator_metadata.yaml": f"{LOCAL}/camulator_metadata.yaml",
    f"{STAGING}/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc": (
        f"{LOCAL}/assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
    ),
}

TREFHT, T, QTOT = "CESM/prognostic/2d/TREFHT", "CESM/prognostic/3d/T", "CESM/prognostic/3d/Qtot"
U, V, PS = "CESM/prognostic/3d/U", "CESM/prognostic/3d/V", "CESM/prognostic/2d/PS"
ICEFRAC = "CESM/dynamic_forcing/2d/ICEFRAC"
LOWEST = 31  # CAM's lowest model level (~992.6 hPa)
TROPOSPHERE = list(range(10, 32))  # CAM levels below ~100 hPa (level 10 is ~103 hPa for a 1000 hPa surface)
TAU_T_DAYS, TAU_Q_DAYS = 11.0, 14.0  # residual e-folding times: TREFHT (shared with level-31 T) and Qtot
WHITE_TAU_DAYS = 1e-6  # phi = exp(-6 h / tau) = 0: a fresh land-noise pattern every step

# Start perturbation: every prognostic, donors from the base model's training years (the hindcasts are 2013-14).
KICK_FIELDS = [U, V, T, QTOT, PS, TREFHT]
KICK_YEARS = (1980, 2012)


def swap_paths(node):
    """Replace the config's glade paths with local copies."""
    if isinstance(node, dict):
        return {k: swap_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [swap_paths(v) for v in node]
    return SWAP.get(node, node) if isinstance(node, str) else node


def land_noise_block(sigma_file: str, land_file: str, tau_t: float = TAU_T_DAYS, tau_q: float = TAU_Q_DAYS) -> dict:
    """The land noise (targeted_red_noise) block: TREFHT + level-31 T share one pattern, level-31 Qtot has its own."""

    def sigma(var):
        return {"path": sigma_file, "var": var}

    return {
        "type": "targeted_red_noise",
        "args": {
            "lead_time_periods": 6,
            "length_scale_km": 500,
            "land_file": land_file,
            "land_var": "LANDFRAC",
            "ice_var": ICEFRAC,
            "fields": [
                {"var": TREFHT, "sigma": sigma("TREFHT"), "tau_days": tau_t, "pattern": "sfc_T"},
                {"var": T, "levels": [LOWEST], "sigma": sigma("T_lowest"), "tau_days": tau_t, "pattern": "sfc_T"},
                {"var": QTOT, "levels": [LOWEST], "sigma": sigma("Qtot_lowest"), "tau_days": tau_q},
            ],
        },
    }


def tendency_block(thermo_only: bool = False) -> dict:
    """The tendency_noise block: T, Qtot, U and V (or T and Qtot only) below ~100 hPa, ECMWF's three scales."""
    fields = [T, QTOT] if thermo_only else [T, QTOT, U, V]
    return {
        "type": "tendency_noise",
        "args": {"lead_time_periods": 6, "fields": [{"var": var, "levels": TROPOSPHERE} for var in fields]},
    }


def start_perturbation_block(path: str, years, amplitude, seed: int) -> dict:
    """The start_perturbation preblock for one member (its seed), Qtot kept non-negative."""
    return {
        "type": "start_perturbation",
        "args": {
            "path": path,
            "fields": list(KICK_FIELDS),
            "amplitude": amplitude,
            "years": list(years),
            "window_days": 15,
            "nonnegative": [QTOT],
            "seed": seed,
        },
    }


def scaled_sigma_file(sigma_file: str, factor: float, out_dir: str) -> str:
    """A copy of the sigma maps times ``factor`` (e.g. white land noise matched to red's spread)."""
    import xarray as xr

    out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(sigma_file))[0]}_x{factor:g}.nc")
    if not os.path.exists(out):
        os.makedirs(out_dir, exist_ok=True)
        with xr.open_dataset(sigma_file) as ds:
            scaled = ds * factor
            scaled.attrs = {**ds.attrs, "scaled_by": factor, "scaled_from": os.path.abspath(sigma_file)}
            scaled.to_netcdf(out)
    return out


def inference_postblocks(conf: dict, noise_blocks: dict | None = None) -> dict:
    """The training per_step chain for rollout: no target twin (rollouts have no y), noise after inverse_scale.

    ``noise_blocks`` maps postblock names to blocks, inserted in order (a single block is the land noise).
    Tendency noise goes before land noise, so that it scales the model's own change only.
    """
    if noise_blocks is not None and "type" in noise_blocks:
        noise_blocks = {"land_red_noise": noise_blocks}
    per_step = {}
    for name, block in conf["postblocks"]["per_step"].items():
        if name in ("reconstruct_target", "scaler_target"):
            continue
        block = copy.deepcopy(block)
        if name == "reconstruct":
            block["args"]["detach"] = True
        per_step[name] = block
        if name == "inverse_scale":
            for noise_name, noise_block in (noise_blocks or {}).items():
                per_step[noise_name] = copy.deepcopy(noise_block)
    return {"per_step": per_step}
