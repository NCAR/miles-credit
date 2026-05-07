"""credit plot and credit metrics command handlers."""

import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)


def _build_channel_map(conf):
    """Return a dict mapping variable name -> list of channel indices in the output tensor."""
    src = conf["data"]["source"]["ERA5"]
    n_levels = len(src.get("levels", []))
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    channel_map = {}
    idx = 0
    for vn in prog.get("vars_3D", []):
        channel_map[vn] = list(range(idx, idx + n_levels))
        idx += n_levels
    for vn in prog.get("vars_2D", []):
        channel_map[vn] = [idx]
        idx += 1
    for vn in diag.get("vars_2D", []):
        channel_map[vn] = [idx]
        idx += 1
    return channel_map


def _build_denorm_stats(conf):
    """Return (mean_arr, std_arr) aligned with ERA5Dataset target channel order."""
    import numpy as np
    import xarray as xr

    src = conf["data"]["source"]["ERA5"]
    levels = src["levels"]
    level_coord = src["level_coord"]
    n_levels = len(levels)
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}

    norm_args = conf.get("preblocks", {}).get("norm", {}).get("args", {})
    mean_ds = xr.open_dataset(norm_args["mean_path"]).load()
    std_ds = xr.open_dataset(norm_args["std_path"]).load()

    def _stats(varname, is_3d):
        if varname not in mean_ds or varname not in std_ds:
            n = n_levels if is_3d else 1
            return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
        if is_3d:
            m = mean_ds[varname].sel({level_coord: levels}).values.astype(np.float32)
            s = std_ds[varname].sel({level_coord: levels}).values.astype(np.float32)
        else:
            m = np.array([float(mean_ds[varname].values)], dtype=np.float32)
            s = np.array([float(std_ds[varname].values)], dtype=np.float32)
        return m, s

    means, stds = [], []
    for vn in prog.get("vars_3D", []):
        m, s = _stats(vn, True)
        means.append(m)
        stds.append(s)
    for vn in prog.get("vars_2D", []):
        m, s = _stats(vn, False)
        means.append(m)
        stds.append(s)
    for vn in diag.get("vars_2D", []):
        m, s = _stats(vn, False)
        means.append(m)
        stds.append(s)

    return __import__("numpy").concatenate(means), __import__("numpy").concatenate(stds)


def _plot(args) -> None:
    """Load checkpoint, run one forward pass, produce global maps."""
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for credit plot. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        _HAS_CARTOPY = True
    except ImportError:
        _HAS_CARTOPY = False
        logger.warning("cartopy not found — using plain lat/lon axes. Install cartopy for globe projections.")

    import torch

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    save_loc = os.path.expandvars(conf.get("save_loc", "."))
    ckpt_path = args.checkpoint or os.path.join(save_loc, "checkpoint.pt")
    out_dir = args.output_dir or os.path.join(save_loc, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    from credit.models import load_model
    from credit.models.checkpoint import load_state_dict_error_handler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(conf, load_weights=False)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_msg = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    load_state_dict_error_handler(load_msg)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    import pandas as pd
    from torch.utils.data import default_collate

    from credit.datasets.multi_source import MultiSourceDataset
    from credit.preblock import apply_preblocks, build_preblocks

    data_conf = conf.get("data_valid", conf["data"])
    dataset = MultiSourceDataset(data_conf, return_target=True)

    if args.sample_date is not None:
        target_dt = pd.Timestamp(args.sample_date)
        ts = None
        for t in dataset.datetimes:
            if pd.Timestamp(t) >= target_dt:
                ts = t
                break
        if ts is None:
            logger.warning("sample_date %s not found — using first sample", args.sample_date)
            ts = dataset.datetimes[0]
    else:
        ts = dataset.datetimes[0]

    sample = dataset[(ts, 0)]
    batch = default_collate([sample])
    preblocks = build_preblocks(conf.get("preblocks", {}))
    x, y, _ = apply_preblocks(preblocks, batch)
    x = x.to(device)

    with torch.no_grad():
        y_pred = model(x)

    def _squeeze(t):
        t = t.squeeze(0).cpu().float()
        if t.ndim == 4:
            t = t[:, 0]
        return t

    y_true_np = _squeeze(y).numpy()
    y_pred_np = _squeeze(y_pred).numpy()

    unit_label = "normalised"
    if args.denorm:
        mean_arr, std_arr = _build_denorm_stats(conf)
        mean_arr = mean_arr[:, None, None]
        std_arr = std_arr[:, None, None]
        y_true_np = y_true_np * std_arr + mean_arr
        y_pred_np = y_pred_np * std_arr + mean_arr
        unit_label = "physical units"
        logger.info("Inverse-normalised outputs to physical units")

    channel_map = _build_channel_map(conf)

    for field in args.field:
        if field not in channel_map:
            available = ", ".join(sorted(channel_map.keys()))
            print(f"Field '{field}' not found. Available: {available}", file=sys.stderr)
            continue

        chans = channel_map[field]
        level_idx = min(args.level, len(chans) - 1)
        c = chans[level_idx]

        truth = y_true_np[c]
        pred = y_pred_np[c]
        diff = pred - truth

        H, W = truth.shape
        lats = np.linspace(90, -90, H)
        lons = np.linspace(0, 360, W, endpoint=False)

        vmin = float(np.percentile(truth, 2))
        vmax = float(np.percentile(truth, 98))
        dabs = float(np.percentile(np.abs(diff), 98))

        title_suffix = f"  level {level_idx}" if len(chans) > 1 else ""
        ckpt_epoch = ckpt.get("epoch", "?")
        fig_title = f"{field}{title_suffix}  |  epoch {ckpt_epoch}  |  {unit_label}"

        if _HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": proj})

            def _add_features(ax):
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.set_global()

            im0 = axes[0].pcolormesh(lons, lats, truth, vmin=vmin, vmax=vmax, cmap="RdBu_r", transform=proj)
            _add_features(axes[0])
            axes[0].set_title("Truth")
            plt.colorbar(im0, ax=axes[0], shrink=0.6)

            im1 = axes[1].pcolormesh(lons, lats, pred, vmin=vmin, vmax=vmax, cmap="RdBu_r", transform=proj)
            _add_features(axes[1])
            axes[1].set_title("Prediction")
            plt.colorbar(im1, ax=axes[1], shrink=0.6)

            im2 = axes[2].pcolormesh(lons, lats, diff, vmin=-dabs, vmax=dabs, cmap="bwr", transform=proj)
            _add_features(axes[2])
            axes[2].set_title("Difference (pred − truth)")
            plt.colorbar(im2, ax=axes[2], shrink=0.6)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(truth, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto", origin="upper")
            axes[0].set_title("Truth")
            axes[0].axis("off")
            axes[1].imshow(pred, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto", origin="upper")
            axes[1].set_title("Prediction")
            axes[1].axis("off")
            im2 = axes[2].imshow(diff, vmin=-dabs, vmax=dabs, cmap="bwr", aspect="auto", origin="upper")
            axes[2].set_title("Difference (pred − truth)")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], shrink=0.7)

        fig.suptitle(fig_title, fontsize=13)
        plt.tight_layout()

        safe_field = field.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_field}_lev{level_idx:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def _metrics(args) -> None:
    """Run WeatherBench2-style evaluation and optionally generate scorecard plots."""
    import subprocess

    from ._common import _repo_root

    script = os.path.join(_repo_root(), "applications", "eval_weatherbench.py")
    if not os.path.exists(script):
        print(
            "eval_weatherbench.py not found. This command requires the v2.1/weatherbench branch to be merged.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [sys.executable, script]

    if args.csv:
        cmd += ["--csv", args.csv]
    elif args.netcdf:
        cmd += ["--netcdf", args.netcdf]

    if args.era5:
        cmd += ["--era5", args.era5]
    if args.clim:
        cmd += ["--clim", args.clim]
    if args.out:
        cmd += ["--out", args.out]
    if args.lead_time_hours:
        cmd += ["--lead-time-hours", str(args.lead_time_hours)]
    if args.max_inits:
        cmd += ["--max-inits", str(args.max_inits)]
    if args.plot_dir:
        cmd += ["--plot", args.plot_dir]
    if args.label:
        cmd += ["--label", args.label]
    if args.no_refs:
        cmd += ["--no-refs"]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.verbose:
        cmd += ["-v"]

    sys.exit(subprocess.call(cmd))
