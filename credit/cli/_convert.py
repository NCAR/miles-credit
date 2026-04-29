"""credit init and credit convert command handlers."""

import argparse
import os
import sys
import yaml

from ._common import _prompt, _prompt_bool, _repo_root


class _FlowSeqDumper(yaml.Dumper):
    """YAML dumper that writes lists of scalars as flow sequences [a, b, c]."""

    def represent_sequence(self, tag, sequence, flow_style=None):
        if all(isinstance(x, (int, float, str, bool, type(None))) for x in sequence):
            flow_style = True
        return super().represent_sequence(tag, sequence, flow_style=flow_style)


def _yaml_dump(conf, f):
    yaml.dump(conf, f, Dumper=_FlowSeqDumper, default_flow_style=False, sort_keys=False, allow_unicode=True)


_SPATIAL_TIME_DIMS = frozenset(("time", "lat", "lon", "latitude", "longitude", "x", "y"))


def _zarr_dims(glob_path: str) -> list[str] | None:
    """Return all dimension names from the first matching file, or None on failure."""
    import glob as _glob

    if not glob_path:
        return None
    candidates = sorted(_glob.glob(glob_path))
    if not candidates:
        return None
    first = candidates[0]
    try:
        import xarray as xr

        ds = (
            xr.open_zarr(first, consolidated=False)
            if (first.endswith(".zarr") or os.path.isdir(first))
            else xr.open_dataset(first, engine="netcdf4")
        )
        dims = list(ds.dims)
        ds.close()
        return dims
    except Exception:
        return None


def _write_reload_config(config_path: str) -> str:
    """Patch trainer reload fields and write a reload config next to the checkpoint.

    Reads the YAML at *config_path*, sets the five fields required for a clean
    resume, and writes the result to ``<save_loc>/config_reload.yml``.

    Returns the path to the written reload config.
    """
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    trainer = conf.setdefault("trainer", {})
    trainer["load_weights"] = True
    trainer["load_optimizer"] = True
    trainer["load_scaler"] = True
    trainer["load_scheduler"] = True
    trainer["reload_epoch"] = True

    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    os.makedirs(save_loc, exist_ok=True)
    reload_path = os.path.join(save_loc, "config_reload.yml")

    with open(reload_path, "w") as f:
        _yaml_dump(conf, f)

    return reload_path


def _convert(args: argparse.Namespace) -> None:
    """Interactive v1 → v2 config converter."""
    use_defaults = getattr(args, "defaults", False)
    if use_defaults:

        def prompt_bool(prompt, default=True):
            return default

        def prompt(prompt, default=None):
            return str(default) if default is not None else ""

    else:
        prompt_bool = _prompt_bool
        prompt = _prompt

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    trainer_type = conf.get("trainer", {}).get("type", "unknown")

    print()
    print("=" * 62)
    print("  CREDIT config converter  (v1 → v2)")
    print("=" * 62)
    print(f"  Input  : {args.config}")
    print(f"  Trainer: {trainer_type}")
    print()

    # ------------------------------------------------------------------
    # Auto-transformations — no questions needed
    # ------------------------------------------------------------------
    changes = []

    # trainer.type
    V1_TYPES = {"era5", "standard", "universal"}
    if trainer_type in V1_TYPES:
        conf["trainer"]["type"] = "era5-gen2"
        changes.append(f"trainer.type: '{trainer_type}' → 'era5-gen2'")

    # data schema: v1 flat → v2 nested source
    _V1_DATA_FLAT_KEYS = {
        "variables",
        "surface_variables",
        "dynamic_forcing_variables",
        "static_variables",
        "save_loc",
        "save_loc_surface",
        "save_loc_dynamic_forcing",
        "save_loc_static",
        "save_loc_diagnostic",
        "diagnostic_variables",
        "mean_path",
        "std_path",
        "train_years",
        "valid_years",
        "lead_time_periods",
        "scaler_type",
        "history_len",
        "valid_history_len",
        "dataset_type",
        "static_first",
        "skip_periods",
        "one_shot",
        "level_ids",
    }
    data = conf.get("data", {})
    if "source" not in data and _V1_DATA_FLAT_KEYS & set(data.keys()):
        vars_3d = data.get("variables") or []
        vars_2d = data.get("surface_variables") or []
        dyn_vars = data.get("dynamic_forcing_variables") or []
        static_vars = data.get("static_variables") or []
        diag_vars = data.get("diagnostic_variables") or []
        prog_path = data.get("save_loc") or data.get("save_loc_surface") or ""
        dyn_path = data.get("save_loc_dynamic_forcing") or ""
        static_path = data.get("save_loc_static") or ""
        diag_path = data.get("save_loc_diagnostic") or ""
        mean_path = data.get("mean_path") or ""
        std_path = data.get("std_path") or ""
        lead_time = int(data.get("lead_time_periods") or 6)
        train_years = data.get("train_years") or [1979, 2018]
        valid_years = data.get("valid_years") or [2018, 2019]
        n_levels = conf.get("model", {}).get("levels", 16)
        _DEFAULT_LEVELS_16 = [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
        levels = list(data["level_ids"]) if "level_ids" in data else _DEFAULT_LEVELS_16[:n_levels]

        era5_vars = {
            "prognostic": {
                "vars_3D": vars_3d,
                "vars_2D": vars_2d,
                "path": prog_path,
                "filename_time_format": "%Y",
            },
        }
        if dyn_vars:
            era5_vars["dynamic_forcing"] = {
                "vars_2D": dyn_vars,
                "path": dyn_path,
                "filename_time_format": "%Y",
            }
        if static_vars:
            era5_vars["static"] = {"vars_2D": static_vars, "path": static_path}
        if diag_vars:
            era5_vars["diagnostic"] = {
                "vars_2D": diag_vars,
                "path": diag_path,
                "filename_time_format": "%Y",
            }
        else:
            era5_vars["diagnostic"] = None

        # Keep non-flat keys (forecast_len, valid_forecast_len, backprop_on_timestep, …)
        keep_keys = {k: v for k, v in data.items() if k not in _V1_DATA_FLAT_KEYS}
        level_coord = _prompt("level_coord (vertical coordinate name in your data files)").strip()
        _all_dims = _zarr_dims(prog_path)
        if _all_dims is not None and level_coord not in _all_dims:
            print(f"  '{level_coord}' not found in file. Available dims: {_all_dims}", file=sys.stderr)
            level_coord = _prompt("Choose from the above").strip()

        conf["data"] = {
            "source": {"ERA5": {"level_coord": level_coord, "levels": levels, "variables": era5_vars}},
            "timestep": f"{lead_time}h",
            "forecast_len": data.get("forecast_len", 0),
            "start_datetime": f"{train_years[0]}-01-01",
            "end_datetime": f"{train_years[1]}-12-31",
            **keep_keys,
        }
        conf.setdefault("validation_data", {}).update(
            {
                "start_datetime": f"{valid_years[0]}-01-01",
                "end_datetime": f"{valid_years[1]}-12-31",
            }
        )
        if mean_path or std_path:
            conf.setdefault("preblocks", {})["norm"] = {
                "type": "era5_normalizer",
                "args": {"mean_path": mean_path, "std_path": std_path},
            }
        changes.append(
            f"data: flat V1 schema → nested V2 source schema  (ERA5, {n_levels} levels, timestep={lead_time}h)"
        )
        changes.append(f"data.start_datetime: {train_years[0]}-01-01 .. {train_years[1]}-12-31")
        changes.append(f"validation_data: {valid_years[0]}-01-01 .. {valid_years[1]}-12-31")
        if mean_path:
            changes.append("preblocks.norm: moved from data.mean_path / data.std_path")
        changes.append("  NOTE: review data paths — glob patterns may need updating for v2 file layout")

    # forecast_len: v1 uses 0 = single step, v2 uses 1 = single step
    fl = conf.get("data", {}).get("forecast_len", 0)
    new_fl = fl + 1
    conf["data"]["forecast_len"] = new_fl
    changes.append(f"data.forecast_len: {fl} → {new_fl}  (v2: 1 = single step)")

    vfl = conf.get("data", {}).get("valid_forecast_len", fl)
    new_vfl = vfl + 1
    conf["data"]["valid_forecast_len"] = new_vfl
    changes.append(f"data.valid_forecast_len: {vfl} → {new_vfl}")

    print("  Auto-applied:")
    for c in changes:
        print(f"    + {c}")
    print()

    # ------------------------------------------------------------------
    # New v2 trainer features
    # ------------------------------------------------------------------
    print("  --- New v2 trainer features ---")

    use_ema = prompt_bool("Enable EMA (exponential moving average of weights)? Recommended", default=True)
    conf["trainer"]["use_ema"] = use_ema
    if use_ema:
        ema_decay = prompt("EMA decay", default="0.9999")
        conf["trainer"]["ema_decay"] = float(ema_decay)

    use_tb = prompt_bool("Enable TensorBoard logging", default=True)
    conf["trainer"]["use_tensorboard"] = use_tb
    print()

    # ------------------------------------------------------------------
    # Ensemble detection
    # ------------------------------------------------------------------
    ensemble_size = conf.get("trainer", {}).get("ensemble_size", 1)
    loss_type = conf.get("loss", {}).get("training_loss", "")
    is_ensemble = ensemble_size > 1 or "crps" in loss_type.lower()

    if is_ensemble:
        print(f"  --- Ensemble settings (detected: ensemble_size={ensemble_size}, loss={loss_type}) ---")
        keep_ensemble = prompt_bool("Keep ensemble training", default=True)
        if keep_ensemble:
            new_size = prompt("Ensemble size", default=str(ensemble_size))
            conf["trainer"]["ensemble_size"] = int(new_size)
        else:
            conf["trainer"]["ensemble_size"] = 1
            print("  Note: consider changing loss.training_loss from crps to mse or mae")
        print()

    # ------------------------------------------------------------------
    # PBS / job settings — skip entirely if pbs: already in config
    # ------------------------------------------------------------------
    pbs = conf.get("pbs", {})
    if not pbs:
        print("  --- PBS / job settings ---")
        cluster = prompt("Cluster (casper/derecho)", default="derecho")
        account = prompt(
            "PBS account code",
            default=os.environ.get("PBS_ACCOUNT") or "NAML0001",
        )
        conda = prompt("Conda env (name or full path)", default="")
        walltime = prompt("Walltime (HH:MM:SS)", default="12:00:00")
        job_name = prompt("Job name", default="credit_gen2")

        new_pbs = {"project": account, "job_name": job_name, "walltime": walltime, "conda": conda}

        if cluster == "derecho":
            new_pbs["nodes"] = int(prompt("Nodes", default="1"))
            new_pbs["ngpus"] = int(prompt("GPUs per node", default="4"))
            new_pbs["ncpus"] = int(prompt("CPUs per node", default="64"))
            new_pbs["mem"] = prompt("Memory per node", default="480GB")
            new_pbs["queue"] = prompt("Queue", default="main")
        else:
            new_pbs["ngpus"] = int(prompt("GPUs", default="4"))
            new_pbs["ncpus"] = int(prompt("CPUs per node", default="8"))
            new_pbs["mem"] = prompt("Memory", default="128GB")
            new_pbs["gpu_type"] = prompt("GPU type", default="a100_80gb")
            new_pbs["queue"] = prompt("Queue", default="casper")

        conf["pbs"] = new_pbs

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print()
    base, ext = os.path.splitext(args.config)
    default_out = getattr(args, "output", None) or (f"{base}_gen2{ext}" if ext else f"{args.config}_gen2.yml")
    out_path = prompt("Output config path", default=default_out)

    with open(out_path, "w") as f:
        _yaml_dump(conf, f)

    print()
    print(f"  Saved → {out_path}")
    print("=" * 62)
    print()


def _init(args: argparse.Namespace) -> None:
    """Copy a config template to the user's desired location."""
    import shutil

    templates = {
        ("0.25deg", "wxformer"): "config/wxformer_1dg_6hr.yml",
        ("0.25deg", "crossformer"): "config/wxformer_1dg_6hr.yml",
        ("1deg", "wxformer"): "config/wxformer_1dg_6hr.yml",
        ("1deg", "crossformer"): "config/wxformer_1dg_6hr.yml",
    }

    repo = _repo_root()
    key = (args.grid, args.model)
    template_rel = templates.get(key)

    if template_rel is None:
        print(f"No template available for grid={args.grid}, model={args.model}", file=sys.stderr)
        sys.exit(1)

    template = os.path.join(repo, template_rel)
    if not os.path.exists(template):
        print(f"Template not found: {template}", file=sys.stderr)
        sys.exit(1)

    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.force:
        print(f"File already exists: {output}  (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    shutil.copy(template, output)
    print(f"Created  : {output}")
    print(f"Template : {template_rel}")
    print()
    print("Next steps:")
    print("  1. Check 'save_loc' — defaults to /glade/derecho/scratch/$USER/CREDIT_runs/...")
    print("     (NCAR users: no edits needed; others: update to a writable path)")
    print("  2. Verify data paths under 'data.source'")
    print("     (NCAR users: paths point to /glade/campaign/cisl/aiml/ksha/CREDIT_data/ — readable by all staff)")
    print(f"  3. credit submit --cluster casper -c {output} --gpus 4 --chain 14")
