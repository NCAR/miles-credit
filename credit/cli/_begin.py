"""Interactive starter-config wizard for ``credit begin``."""

from __future__ import annotations

import copy
import datetime as dt
import glob
import io
import os
import random
import re
import socket
import subprocess
import sys
from pathlib import Path

import yaml

from ._common import _CASPER_GPU_NODES, _PBS_DEFAULTS, _SLURM_CLUSTER_DEFAULTS, _is_ncar_system, _prompt
from ._convert import _FlowSeqDumper


_PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
_DATASET_CHOICES = (
    "arco_era5",
    "weatherbench2_era5",
    "local",
    "gfs",
    "gefs",
    "hrrr",
    "hrrr_nat",
    "hrrr_subh",
    "mrms",
    "goes",
)
_WB2_GRIDS = {
    "1440x721": (721, 1440, [90, -90, 721], [0, 359.75, 1440]),
    "240x121": (121, 240, [90, -90, 121], [0, 359, 240]),
    "64x32": (32, 64, [90, -90, 32], [0, 360, 64]),
    "full": (721, 1440, [90, -90, 721], [0, 359.75, 1440]),
}
_WB2_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
_MODEL_LEVELS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 115, 120, 125, 127]
_SECTION_COMMENTS = {
    "save_loc": "# ---- Experiment output and reproducibility ----",
    "data": "# ---- Training data ----",
    "validation_data": "# ---- Validation data ----",
    "preblocks": "# ---- Preprocessing blocks ----",
    "postblocks": "# ---- Postprocessing blocks ----",
    "model": "# ---- Model architecture ----",
    "loss": "# ---- Training loss ----",
    "metrics": "# ---- Evaluation metrics ----",
    "trainer": "# ---- Trainer and optimization ----",
    "inference": "# ---- Inference and rollout ----",
    "pbs": "# ---- Batch submission settings ----",
    "slurm": "# ---- Batch submission settings ----",
}


class _QuotedVariable(str):
    pass


class _BeginDumper(_FlowSeqDumper):
    pass


def _represent_quoted_variable(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style='"')


_BeginDumper.add_representer(_QuotedVariable, _represent_quoted_variable)


def _detect_system() -> dict:
    """Return a friendly summary of the current host, environment, and GPUs."""
    hostname = socket.gethostname().lower()
    if any(name in hostname for name in ("derecho", "casper", "crhtc", "dec", "crlogin")):
        system = "derecho" if "derecho" in hostname else "casper" if "casper" in hostname else "ncar"
    elif any(name in hostname for name in ("perlmutter", "nid")):
        system = "perlmutter"
    else:
        system = "local"

    gpu_count = 0
    gpu_names = []
    mps_available = False
    try:
        import torch

        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        mps_available = sys.platform == "darwin" and torch.backends.mps.is_available()
    except (ImportError, RuntimeError, AttributeError):
        pass

    return {
        "hostname": hostname,
        "system": system,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or "not detected",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "mps_available": mps_available,
        "ncar": _is_ncar_system(),
    }


def _seed_default() -> int:
    return int("".join(random.choice(_PI_DIGITS) for _ in range(4)))


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        value = _prompt(prompt, default)
        try:
            return int(value)
        except ValueError:
            print("  Please enter a whole number.")


def _prompt_date(prompt: str, default: str) -> str:
    print("  Date format: YYYY-MM-DD (or ISO time, YYYY-MM-DDTHH:MM:SS), for example 2019-01-01.")
    while True:
        value = _prompt(prompt, default)
        try:
            import pandas as pd

            pd.Timestamp(value)
            return value
        except (ImportError, ValueError, TypeError):
            try:
                dt.datetime.fromisoformat(value)
                return value
            except ValueError:
                print("  Could not parse that date. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.")


def _prompt_list(prompt: str, default: list[str]) -> list[str]:
    value = _prompt(prompt, ", ".join(default))
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_dataset() -> tuple[str, dict]:
    print("\n  Available datasets:")
    for index, name in enumerate(_DATASET_CHOICES, 1):
        print(f"    {index}. {name}")
    selection = _prompt("Dataset name or number", "1").lower()
    if selection.isdigit() and 1 <= int(selection) <= len(_DATASET_CHOICES):
        selection = _DATASET_CHOICES[int(selection) - 1]
    while selection not in _DATASET_CHOICES:
        selection = _prompt("Choose one of the listed dataset names", "arco_era5").lower()

    if selection == "arco_era5":
        preset = {
            "dataset_type": selection,
            "level_coord": "hybrid",
            "levels": list(range(1, 138)),
            "vars_3D": ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"],
            "vars_2D": ["surface_pressure", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
            "static": ["geopotential_at_surface", "land_sea_mask", "sea_ice_cover"],
            "timestep": "6h",
            "start": "1979-01-01",
            "end": "2018-12-31",
            "height": 721,
            "width": 1440,
            "tisr": ([90, -90, 721], [0, 359.75, 1440]),
        }
    elif selection == "weatherbench2_era5":
        print("  Available WeatherBench2 resolutions: " + ", ".join(_WB2_GRIDS))
        resolution = _prompt("WeatherBench2 resolution", "1440x721")
        while resolution not in _WB2_GRIDS:
            resolution = _prompt("Choose 1440x721, 240x121, 64x32, or full", "1440x721")
        height, width, lat_spec, lon_spec = _WB2_GRIDS[resolution]
        preset = {
            "dataset_type": selection,
            "level_coord": "level",
            "levels": list(_WB2_LEVELS),
            "vars_3D": ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"],
            "vars_2D": ["surface_pressure", "2m_temperature"],
            "static": ["geopotential_at_surface", "land_sea_mask", "sea_ice_cover"],
            "timestep": "6h",
            "start": "1959-01-01",
            "end": "2022-12-31",
            "height": height,
            "width": width,
            "resolution": resolution,
            "tisr": (lat_spec, lon_spec),
        }
    elif selection == "local":
        preset = {
            "dataset_type": selection,
            "level_coord": "level",
            "levels": list(range(1, 19)),
            "vars_3D": ["T", "U", "V", "Q"],
            "vars_2D": ["SP", "t2m"],
            "static": ["LSM"],
            "timestep": "6h",
            "start": "2020-01-01",
            "end": "2020-12-31",
            "height": 181,
            "width": 360,
        }
    elif selection == "gfs":
        preset = {
            "dataset_type": selection,
            "level_coord": "pfull",
            "levels": list(_MODEL_LEVELS),
            "vars_3D": ["tmp", "ugrd", "vgrd", "spfh"],
            "vars_2D": ["pressfc", "tmp2m"],
            "static": [],
            "timestep": "6h",
            "start": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 384,
            "width": 768,
            "extra": {"system": "gdas", "mode": "remote", "level_type": "model", "check_availability": True},
        }
    elif selection == "gefs":
        preset = {
            "dataset_type": selection,
            "level_coord": "level",
            "levels": list(range(1, 17)),
            "vars_3D": ["t", "u_a", "v_a", "sphum"],
            "vars_2D": ["ps", "t2m"],
            "static": [],
            "timestep": "6h",
            "start": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 384,
            "width": 768,
            "extra": {"mode": "remote", "members": ["c00"]},
        }
    elif selection in ("hrrr", "hrrr_nat"):
        preset = {
            "dataset_type": selection,
            "level_coord": "level",
            "levels": list(range(1, 17)),
            "vars_3D": ["T", "U", "V", "Q"],
            "vars_2D": ["t2m", "sp"],
            "static": ["orog", "landmask"],
            "timestep": "1h",
            "start": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 1059,
            "width": 1799,
            "extra": {"mode": "remote", "product": "wrfprs" if selection == "hrrr" else "wrfnat"},
        }
    elif selection == "hrrr_subh":
        preset = {
            "dataset_type": selection,
            "level_coord": "level",
            "levels": [1],
            "vars_3D": [],
            "vars_2D": ["t2m", "sp", "prate"],
            "static": ["orog", "landmask"],
            "timestep": "15min",
            "start": (dt.date.today() - dt.timedelta(days=7)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 1059,
            "width": 1799,
            "extra": {"mode": "remote", "product": "wrfsubh"},
        }
    elif selection == "mrms":
        preset = {
            "dataset_type": selection,
            "level_coord": None,
            "levels": [1],
            "vars_3D": [],
            "vars_2D": ["MultiSensor_QPE_01H_Pass2_00.00"],
            "static": [],
            "timestep": "1h",
            "start": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 350,
            "width": 700,
            "extra": {"mode": "remote"},
        }
    else:
        preset = {
            "dataset_type": selection,
            "level_coord": None,
            "levels": [1],
            "vars_3D": [],
            "vars_2D": ["CMI_C04", "CMI_C07"],
            "static": [],
            "timestep": "1h",
            "start": (dt.date.today() - dt.timedelta(days=14)).isoformat(),
            "end": dt.date.today().isoformat(),
            "height": 500,
            "width": 500,
            "extra": {"mode": "remote", "product": "ABI-L2-MCMIPF"},
        }
    return selection, preset


def _glob_suggestions(top_path: str) -> tuple[str, str, str]:
    matches = sorted(glob.glob(os.path.join(top_path, "**"), recursive=True))
    files = [path for path in matches if path.endswith((".nc", ".nc4", ".zarr"))]
    if not files:
        return os.path.join(top_path, "*"), os.path.join(top_path, "*"), os.path.join(top_path, "*")
    static = next((path for path in files if "static" in os.path.basename(path).lower()), files[-1])
    dynamic = next((path for path in files if "diag" in path.lower()), files[0])
    return files[0], dynamic, static


def _local_details(preset: dict) -> dict:
    while True:
        top = os.path.expanduser(_prompt("Top-level local dataset directory", ""))
        if glob.glob(top):
            break
        print("  That path does not exist or cannot be globbed.")
    prog_path, diag_path, static_path = _glob_suggestions(top)
    print("  Suggested file paths were inferred from the directory contents.")
    prog_path = _prompt("Prognostic path", prog_path)
    diag_path = _prompt("Diagnostic path (blank disables diagnostics)", diag_path)
    static_path = _prompt("Static path (blank disables static fields)", static_path)
    preset["level_coord"] = _prompt("Vertical coordinate name", preset["level_coord"])
    preset["height"] = _prompt_int("Grid height", preset["height"])
    preset["width"] = _prompt_int("Grid width", preset["width"])
    n_levels = _prompt_int("Number of vertical levels", len(preset["levels"]))
    preset["levels"] = list(range(1, n_levels + 1))
    preset["local_paths"] = {"prognostic": prog_path, "diagnostic": diag_path, "static": static_path}
    preset["local_top"] = top
    years = [
        int(year)
        for year in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", " ".join(glob.glob(top + "/**", recursive=True)))
    ]
    if years:
        preset["start"] = f"{min(years)}-01-01"
        preset["end"] = f"{max(years)}-12-31"
    print("  Local filename_time_format defaults to %Y for annual files; use %Y_%m or %Y%m%d for finer files.")
    return preset


def _gefs_details(preset: dict) -> dict:
    print("  GEFS is cube-sphere data; a Regrid preblock is needed before WxFormer can use it.")
    print("  ESMF_RegridWeightGen can create weights. Public C384 files are available at:")
    print("  https://ftp.emc.ncep.noaa.gov/static_files/public/UFS/GFS/fix/fix_fv3/C384/")
    resolution = _prompt("GEFS remap resolution (0p125deg/0p25deg/0p5deg/1deg)", "1deg")
    while resolution not in {"0p125deg", "0p25deg", "0p5deg", "1deg"}:
        resolution = _prompt("Choose 0p125deg, 0p25deg, 0p5deg, or 1deg", "1deg")
    weight = _prompt("GEFS remap weight file", f"remap_weights_C384_{resolution}.nc")
    dimensions = {
        "0p125deg": (1441, 2880),
        "0p25deg": (721, 1440),
        "0p5deg": (361, 720),
        "1deg": (181, 360),
    }
    preset["height"], preset["width"] = dimensions[resolution]
    preset["regrid_weights"] = weight
    return preset


def _date_range(preset: dict) -> tuple[str, str, str, str, str]:
    start = _prompt_date("Training start_datetime", preset["start"])
    end = _prompt_date("Training end_datetime", preset["end"])
    timestep = _prompt("Timestep", preset["timestep"])
    try:
        import pandas as pd

        valid_start = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
        valid_end = (pd.Timestamp(valid_start) + pd.Timedelta(days=365)).isoformat()
    except (ImportError, ValueError):
        valid_start = end
        valid_end = end
    return start, end, timestep, valid_start, valid_end


def _padding_totals(height: int, width: int) -> tuple[int, int]:
    if (height, width) == (181, 360):
        return 75, 24
    strides = [2, 2, 2, 2]
    kernels = [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]]
    windows = [8, 4, 2, 1]

    def valid(size: int) -> bool:
        for index, (stride, stage_kernels, window) in enumerate(zip(strides, kernels, windows)):
            kernel = min(stage_kernels)
            size = (size + 2 * ((kernel - stride) // 2) - kernel) // stride + 1
            if size < 1 or size % window or size % 4:
                return False
        return True

    h_total = next(total for total in range(1024) if valid(height + total))
    w_total = next(total for total in range(1024) if valid(width + total))
    return h_total, w_total


def _source_config(name: str, preset: dict, variables: dict) -> dict:
    source = {"dataset_type": preset["dataset_type"]}
    if preset.get("level_coord"):
        source["level_coord"] = preset["level_coord"]
    if preset.get("levels"):
        source["levels"] = preset["levels"]
    source.update(copy.deepcopy(preset.get("extra", {})))
    if preset.get("resolution"):
        source["resolution"] = preset["resolution"]
    source["variables"] = variables
    if name == "ERA5" and preset.get("dataset_type") in ("arco_era5", "weatherbench2_era5"):
        source.pop("mode", None)
    return source


def _make_data(state: dict) -> tuple[dict, dict, int, int, int, int]:
    preset = state["preset"]
    local_paths = preset.get("local_paths", {})
    group = {"vars_3D": state["vars_3D"], "vars_2D": state["vars_2D"]}
    if local_paths.get("prognostic"):
        group.update({"path": local_paths["prognostic"], "filename_time_format": "%Y"})
    variables = {"prognostic": group}

    static = preset.get("static", [])
    if local_paths.get("static"):
        variables["static"] = {
            "vars_2D": static,
            "path": local_paths["static"],
        }
    elif static:
        variables["static"] = {"vars_2D": static}
    else:
        variables["static"] = None

    if local_paths.get("diagnostic") and preset.get("diagnostic"):
        variables["diagnostic"] = {
            "vars_2D": [],
            "path": local_paths["diagnostic"],
        }
    else:
        variables["diagnostic"] = None

    source = _source_config("ERA5", preset, variables)
    sources = {"ERA5": source}
    if preset.get("tisr"):
        lat_spec, lon_spec = preset["tisr"]
        hours = re.match(r"^(\d+(?:\.\d+)?)h$", state["timestep"])
        integration_steps = int(float(hours.group(1)) * 360) if hours else 360
        sources["SOLAR"] = {
            "dataset_type": "tisr",
            "num_integration_steps": integration_steps,
            "lat_spec": lat_spec,
            "lon_spec": lon_spec,
            "variables": {
                "prognostic": None,
                "diagnostic": None,
                "dynamic_forcing": {"vars_2D": ["tisr"]},
                "static": None,
            },
        }
    elif preset["dataset_type"] in {"gfs", "gefs", "hrrr", "hrrr_nat", "hrrr_subh", "mrms", "goes"}:
        print("  TISR skipped: its global rectangular grid does not align with this dataset's native grid.")

    data = {
        "source": sources,
        "start_datetime": state["start"],
        "end_datetime": state["end"],
        "timestep": state["timestep"],
        "history_len": 1,
        "forecast_len": 1,
    }
    validation = {
        "source": sources,
        "start_datetime": state["valid_start"],
        "end_datetime": state["valid_end"],
        "timestep": state["timestep"],
        "history_len": 1,
        "forecast_len": 1,
    }
    n_levels = len(preset["levels"])
    n_input_only = len(static) + (1 if preset.get("tisr") else 0)
    return data, validation, n_levels, len(state["vars_3D"]), len(state["vars_2D"]), n_input_only


def _build_config(state: dict) -> dict:
    data, validation, n_levels, channels, surface_channels, input_only = _make_data(state)
    pad_lat, pad_lon = _padding_totals(state["preset"]["height"], state["preset"]["width"])
    scaler_path = os.path.join(state["save_loc"], "standard_scaler.json")
    static = state["preset"].get("static", [])
    fill_variables = [f"ERA5/static/2d/{name}" for name in static if name == "sea_ice_cover"]
    config = {
        "save_loc": state["save_loc"],
        "seed": state["seed"],
        "data": data,
        "validation_data": validation,
        "preblocks": {
            "per_step": {
                "fill_vals": {
                    "type": "fill_values",
                    "args": {"rules": [{"search": "nan", "fill": 0.0}], "variables": fill_variables},
                },
                "scaler": {
                    "type": "bridgescaler_transform",
                    "args": {
                        "scaler_path": scaler_path,
                        "data_types": ["input", "target"],
                        "variables": [],
                        "scaler_type": "standard",
                        "scaler_params": {"channels_last": False},
                        "method": "transform",
                    },
                },
                "concat": {"type": "concat"},
            }
        },
        "postblocks": {
            "per_step": {
                "reconstruct": {"type": "reconstruct", "args": {"detach": False}},
                "scaler": {
                    "type": "bridgescaler_transform",
                    "args": {"scaler_path": scaler_path, "variables": [], "method": "inverse_transform"},
                },
                "reconstruct_target": {
                    "type": "reconstruct",
                    "args": {"in_key": "y", "out_key": "y_target_processed"},
                },
                "scaler_target": {
                    "type": "bridgescaler_transform",
                    "args": {
                        "scaler_path": scaler_path,
                        "variables": [],
                        "method": "inverse_transform",
                        "key": "y_target_processed",
                    },
                },
            }
        },
        "model": {
            "type": "wxformer",
            "frames": 1,
            "image_height": state["preset"]["height"],
            "image_width": state["preset"]["width"],
            "levels": n_levels,
            "channels": channels,
            "surface_channels": surface_channels,
            "input_only_channels": input_only,
            "output_only_channels": 0,
            "patch_height": 1,
            "patch_width": 1,
            "dim": [32, 64, 128, 256],
            "depth": [2, 2, 8, 2],
            "global_window_size": [8, 4, 2, 1],
            "local_window_size": 4,
            "cross_embed_kernel_sizes": [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]],
            "cross_embed_strides": [2, 2, 2, 2],
            "interp": True,
            "use_spectral_norm": True,
            "upsample_with_ps": True,
            "padding_conf": {
                "activate": True,
                "mode": "earth",
                "pad_lat": [pad_lat // 2, pad_lat - pad_lat // 2],
                "pad_lon": [pad_lon // 2, pad_lon - pad_lon // 2],
            },
        },
        "loss": {
            "type": "base",
            "args": {
                "training_loss": "mse",
                "var_weighting": "inverse_variance",
                "scaler_path": scaler_path,
                "normalize_weights": True,
                "include_computed_diagnostics": False,
                "use_latitude_weights": False,
            },
        },
        "metrics": {
            "type": "combined",
            "args": {
                "metrics": {"rmse": {}, "r2score": {}, "bias": {}},
                "var_weighting": "inverse_variance",
                "scaler_path": scaler_path,
                "normalize_weights": True,
            },
        },
        "trainer": {
            "type": "gen2",
            "parallelism": {"data": state["parallelism_data"], "tensor": 1, "domain": 1},
            "activation_checkpoint": True,
            "load_weights": False,
            "load_optimizer": False,
            "load_scaler": False,
            "load_scheduler": False,
            "learning_rate": 1.0e-3,
            "weight_decay": 0,
            "train_batch_size": 4,
            "valid_batch_size": 4,
            "batches_per_epoch": 5,
            "valid_batches_per_epoch": 5,
            "start_epoch": 0,
            "num_epoch": 2,
            "epochs": 2,
            "use_tensorboard": True,
            "use_ema": True,
            "ema_decay": 0.9999,
            "use_scheduler": True,
            "scheduler": {
                "scheduler_type": "linear-warmup-cosine",
                "warmup_steps": 1000,
                "total_steps": 500000,
                "min_lr": 1.0e-5,
            },
            "thread_workers": 4,
            "amp": False,
        },
        "inference": {
            "run_mode": "batch",
            "mode": "none" if state["system"] == "local" else "ddp",
            "save_forecast": os.path.join(state["save_loc"], "rollout"),
            "postblocks": {
                "per_step": {
                    "reconstruct": {"type": "reconstruct"},
                    "scaler": {
                        "type": "bridgescaler_transform",
                        "args": {"scaler_path": scaler_path, "variables": [], "method": "inverse_transform"},
                    },
                }
            },
            "batch_forecast": {
                "forecast_length": "10d",
                "first_init_date": state["valid_start"],
                "last_init_date": state["valid_end"],
                "init_interval": state["timestep"],
            },
            "single_forecast": {
                "forecast_length": "10d",
                "start_datetime": state["valid_start"],
            },
            "output": {
                "format": "netcdf",
                "output_interval": None,
                "group_by": "day",
                "variables": None,
                "metadata": None,
                "encoding": {"dtype": "float32", "zlib": True, "complevel": 4},
            },
        },
    }
    if state["preset"].get("regrid_weights"):
        config["preblocks"]["per_step"] = {
            "regrid": {
                "type": "regrid",
                "args": {"weight_file": state["preset"]["regrid_weights"], "variables": [], "reshape_to_xy": True},
            },
            **config["preblocks"]["per_step"],
        }
    if state["system"] != "local":
        config["pbs"] = state["pbs"]
    return config


def _quote_data_variable_names(config: dict) -> None:
    for source in config["data"]["source"].values():
        for group in (source.get("variables") or {}).values():
            if not group:
                continue
            for key in ("vars_3D", "vars_2D"):
                if key in group:
                    group[key] = [_QuotedVariable(name) for name in group[key]]


def _add_section_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line and not line[0].isspace():
            key = line.split(":", 1)[0]
            if key in _SECTION_COMMENTS:
                lines.append(_SECTION_COMMENTS[key])
        lines.append(line)
    return "\n".join(lines) + "\n"


def _parallelism_data() -> str:
    if sys.platform == "darwin":
        return "none"
    value = _prompt("Data parallelism (ddp, fsdp2, or none)", "ddp").lower()
    while value not in {"ddp", "fsdp2", "none"}:
        value = _prompt("Choose ddp, fsdp2, or none", "ddp").lower()
    return value


def _pbs_config(system: str, experiment: str, nodes: int, gpus: int) -> dict:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if system == "ncar":
        system = "derecho"
    if system in ("derecho", "casper"):
        defaults = _PBS_DEFAULTS[system]
        result = {
            "conda": env or ("credit-derecho" if system == "derecho" else "credit-casper"),
            "project": defaults["account"],
            "job_name": experiment,
            "walltime": defaults["walltime"],
            "nodes": nodes,
            "ncpus": defaults["cpus"],
            "ngpus": gpus,
            "mem": defaults["mem"] if system == "derecho" else "128GB",
            "queue": defaults["queue"],
        }
        if system == "casper":
            gpu_type = _prompt("Casper GPU type (any, v100, a100_80gb, h100)", "any").lower()
            if gpu_type not in ("", "any", "none") and gpu_type in _CASPER_GPU_NODES:
                result["gpu_type"] = gpu_type
        return result

    defaults = _SLURM_CLUSTER_DEFAULTS["perlmutter"]
    account = _prompt("Perlmutter account (include _g for GPU allocations)", "")
    qos = _prompt("Perlmutter QOS", defaults["qos"])
    return {
        "conda": env or "credit",
        "account": account,
        "project": account,
        "job_name": experiment,
        "walltime": defaults["walltime"],
        "nodes": nodes,
        "ncpus": defaults["cpus"],
        "ngpus": gpus,
        "constraint": defaults["constraint"],
        "qos": qos,
    }


def _run_check(path: str) -> int:
    command = [sys.executable, "-m", "credit.cli", "check", "-c", path]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode and "No module named credit.cli.__main__" in output:
        result = subprocess.run(
            [sys.executable, "-c", "from credit.cli._parser import main; main()", "check", "-c", path],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
    print(output.rstrip())
    return result.returncode


def _collect_state(args: object, system_info: dict) -> tuple[str, dict]:
    print()
    print("=" * 68)
    print("  Welcome to CREDIT — let's create a starter modeling config")
    print("=" * 68)
    print(f"  Host: {system_info['hostname']} ({system_info['system']})")
    print(f"  Conda: {system_info['conda_env']}   Python: {system_info['python']}")
    if system_info["gpu_names"]:
        print(f"  GPUs: {system_info['gpu_count']} ({', '.join(system_info['gpu_names'])})")
    elif system_info["mps_available"]:
        print("  Accelerator: Apple MPS")
    else:
        print("  GPUs: none detected (CPU configuration is still possible)")

    default_path = os.path.join(os.getcwd(), f"credit_config_{dt.date.today().isoformat()}.yml")
    config_path = os.path.abspath(
        os.path.expanduser(_prompt("Config path", getattr(args, "config", None) or default_path))
    )
    default_experiment = dt.datetime.now(dt.timezone.utc).strftime("credit_exp_%Y-%m-%dT%H%M")
    experiment = _prompt("Experiment name", default_experiment)
    if system_info["system"] in ("derecho", "casper"):
        default_save = f"$SCRATCH/{experiment}"
    elif system_info["system"] == "perlmutter":
        default_save = f"$PSCRATCH/{experiment}"
    else:
        default_save = f"./{experiment}"
    save_loc = _prompt("save_loc (press Enter to accept the suggested location)", default_save)
    seed = _prompt_int("Random seed", _seed_default())
    dataset, preset = _select_dataset()
    if dataset == "local":
        preset = _local_details(preset)
    elif dataset == "gefs":
        preset = _gefs_details(preset)
    start, end, timestep, valid_start, valid_end = _date_range(preset)
    vars_3d = _prompt_list("Prognostic 3D variables", preset["vars_3D"])
    vars_2d = _prompt_list("Prognostic 2D variables", preset["vars_2D"])
    parallelism_data = _parallelism_data()
    nodes = gpus = 1
    pbs = None
    if system_info["system"] != "local":
        nodes = _prompt_int("PBS/SLURM nodes", 1)
        gpus = _prompt_int("GPUs per node", 4)
        pbs = _pbs_config(system_info["system"], experiment, nodes, gpus)
    state = {
        "system": system_info["system"],
        "experiment": experiment,
        "save_loc": save_loc,
        "seed": seed,
        "dataset": dataset,
        "preset": preset,
        "start": start,
        "end": end,
        "timestep": timestep,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "vars_3D": vars_3d,
        "vars_2D": vars_2d,
        "parallelism_data": parallelism_data,
        "nodes": nodes,
        "gpus": gpus,
        "pbs": pbs,
    }
    return config_path, state


def _repair_state(state: dict) -> None:
    print("\n  The checker found errors. Update the main configuration choices below.")
    state["save_loc"] = _prompt("save_loc", state["save_loc"])
    dataset, preset = _select_dataset()
    state["dataset"] = dataset
    if dataset == "local":
        preset = _local_details(preset)
    elif dataset == "gefs":
        preset = _gefs_details(preset)
    state["preset"] = preset
    state["start"], state["end"], state["timestep"], state["valid_start"], state["valid_end"] = _date_range(preset)
    state["vars_3D"] = _prompt_list("Prognostic 3D variables", preset["vars_3D"])
    state["vars_2D"] = _prompt_list("Prognostic 2D variables", preset["vars_2D"])
    state["parallelism_data"] = _parallelism_data()
    if state["system"] != "local":
        state["nodes"] = _prompt_int("PBS/SLURM nodes", state["nodes"])
        state["gpus"] = _prompt_int("GPUs per node", state["gpus"])
        state["pbs"] = _pbs_config(state["system"], state["experiment"], state["nodes"], state["gpus"])


def _begin(args) -> None:
    system_info = _detect_system()
    config_path, state = _collect_state(args, system_info)
    for attempt in range(3):
        config = _build_config(state)
        _quote_data_variable_names(config)
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        stream = io.StringIO()
        yaml.dump(config, stream, Dumper=_BeginDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
        text = stream.getvalue().replace("&id001", "&data_sources").replace("*id001", "*data_sources")
        text = _add_section_comments(text)
        with open(config_path, "w") as output:
            output.write(text)
        print(f"\n  Saved config: {config_path}")
        print("  Running credit check ...")
        if _run_check(config_path) == 0:
            break
        if attempt == 2:
            print("  The config still needs corrections; inspect the checker report above.")
            break
        _repair_state(state)

    print("\nNext steps:")
    if state["system"] != "local":
        cluster = "derecho" if state["system"] == "ncar" else state["system"]
        print(f"  credit submit --cluster {cluster} --mode preprocess -c {config_path}")
        print(f"  credit submit --cluster {cluster} --mode train     -c {config_path}")
    else:
        print(f"  credit preprocess -c {config_path}")
        print(f"  credit train -c {config_path}")
    print("  The scaler JSON is created by credit preprocess; its absence during credit check is expected.")
