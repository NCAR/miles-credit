"""credit plot and credit metrics command handlers."""

import logging
import os
import sys

import yaml

from credit.datasets.gen_2.channel_utils import resolve_num_levels

logger = logging.getLogger(__name__)


def _build_channel_map(conf):
    """Return a dict mapping variable name -> list of channel indices in the output tensor."""
    src = conf["data"]["source"]["ERA5"]
    n_levels = resolve_num_levels(src, conf)
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


# Postblocks that turn flat model output into physical-unit variables.  Physics fixers and
# diagnostics are deliberately excluded: they need inputs a quick-look plot does not have.
_DENORM_BLOCK_TYPES = frozenset({"reconstruct", "bridgescaler_transform", "exp_transform", "square_transform"})


def _denorm_block_confs(conf):
    """Return the ``postblocks.per_step`` entries that map model output to physical units.

    Keeps only the prediction-side ``reconstruct`` / scaler / transform blocks. The target-twin
    chain (anything reading ``y`` or writing ``y_target_processed``) is dropped, because
    ``credit plot`` pushes the truth through the prediction chain itself.

    Raises:
        ValueError: if the config has no inverse ``bridgescaler_transform`` to denormalize with.
    """
    per_step = (conf.get("postblocks") or {}).get("per_step") or {}
    blocks = {}
    for name, block in per_step.items():
        if not isinstance(block, dict) or block.get("type") not in _DENORM_BLOCK_TYPES:
            continue
        args = block.get("args") or {}
        if (args.get("out_key") or args.get("key") or "y_processed") != "y_processed":
            continue  # target twin
        if block["type"] == "reconstruct" and args.get("in_key", "y_pred") != "y_pred":
            continue  # target twin
        blocks[name] = block

    has_inverse = any(
        b["type"] == "bridgescaler_transform"
        and (b.get("args") or {}).get("method", "inverse_transform") == "inverse_transform"
        for b in blocks.values()
    )
    if not has_inverse:
        raise ValueError(
            "--denorm needs an inverse scaler in postblocks.per_step, e.g.\n"
            "  scaler:\n"
            "    type: bridgescaler_transform\n"
            "    args: {scaler_path: /path/to/scaler.json, variables: [], method: inverse_transform}\n"
            "Add it (after `reconstruct`), or omit --denorm to plot normalised values."
        )
    return blocks


def _build_denorm_postblocks(conf):
    """Instantiate the denormalisation postblock chain (see ``_denorm_block_confs``)."""
    from credit.postblock import _build_postblock_section

    return _build_postblock_section(_denorm_block_confs(conf))


def _denormalize(chain, metadata, tensor):
    """Run flat model-space ``tensor`` through ``chain``; return ``{var_key: (B, n_lev, n_time, H, W)}``."""
    import torch

    from credit.postblock import apply_postblocks

    with torch.no_grad():
        out = apply_postblocks(chain, {"y_pred": tensor, "metadata": metadata})
    return {var_key: t for source in out["y_processed"].values() for var_key, t in source.items()}


def _resolve_field(field, var_keys):
    """Match ``field`` to a variable key: exact key first, else the unique short name (last path segment)."""
    if field in var_keys:
        return field
    for key in var_keys:
        if key.split("/")[-1] == field:
            return key
    return None


def _plot(args) -> None:
    """Load checkpoint, run one forward pass, produce global maps."""
    print("Starting credit plot: loading imports ...", file=sys.stderr, flush=True)
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

    denorm_chain = None
    if args.denorm:  # fail fast, before the slow model/data loading
        try:
            denorm_chain = _build_denorm_postblocks(conf)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

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

    from credit.datasets.gen_2.multi_source import MultiSourceDataset
    from credit.preblock import apply_preblocks, build_preblocks
    from credit.registry import load_custom_objects

    # Build the effective validation data config: start from training data, then overlay
    # validation_data overrides (date range, etc.) if present.  This mirrors how the
    # trainer constructs its validation dataloader.
    import copy

    data_conf = copy.deepcopy(conf["data"])
    val_override = conf.get("validation_data") or conf.get("data_valid") or {}
    data_conf.update(val_override)
    load_custom_objects(conf)  # register any custom classes listed under custom_objects in the config
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
    preblocks = build_preblocks(conf)
    _batch = apply_preblocks(preblocks, batch, device=device)
    # apply_preblocks renames 'input' → 'x' and 'target' → 'y'
    x = _batch.get("x", _batch.get("input"))
    if x is None:
        print("ERROR: preblocks did not produce an 'x' or 'input' key in batch.", file=sys.stderr)
        sys.exit(1)
    y = _batch.get("y", _batch.get("target"))
    if "meta" in _batch:
        meta = _batch["meta"]
    with torch.no_grad():
        y_pred = model(x)

    def _squeeze(t):
        t = t.squeeze(0).cpu().float()
        if t.ndim == 4:
            t = t[:, 0]
        return t

    if args.denorm:
        metadata = _batch["metadata"]
        pred_vars = _denormalize(denorm_chain, metadata, y_pred)
        true_vars = _denormalize(denorm_chain, metadata, y)
        unit_label = "physical units"
        logger.info("Inverse-normalised outputs to physical units via postblocks.per_step")

        def _get_field(field):
            key = _resolve_field(field, pred_vars)
            if key is None:
                return None
            # (B, n_levels, n_time, H, W) -> (n_levels, H, W) for the first sample and time step
            return true_vars[key][0, :, 0].cpu().float().numpy(), pred_vars[key][0, :, 0].cpu().float().numpy()

        available_fields = sorted({k.split("/")[-1] for k in pred_vars})
    else:
        y_true_np = _squeeze(y).numpy()
        y_pred_np = _squeeze(y_pred).numpy()
        unit_label = "normalised"
        channel_map = _build_channel_map(conf)

        def _get_field(field):
            if field not in channel_map:
                return None
            chans = channel_map[field]
            return y_true_np[chans], y_pred_np[chans]

        available_fields = sorted(channel_map.keys())

    for field in args.field:
        got = _get_field(field)
        if got is None:
            print(f"Field '{field}' not found. Available: {', '.join(available_fields)}", file=sys.stderr)
            continue

        truth_levels, pred_levels = got
        n_chans = truth_levels.shape[0]
        level_idx = min(args.level, n_chans - 1)

        truth = truth_levels[level_idx]
        pred = pred_levels[level_idx]
        diff = pred - truth

        H, W = truth.shape
        lats = np.linspace(90, -90, H)
        lons = np.linspace(0, 360, W, endpoint=False)

        vmin = float(np.percentile(truth, 2))
        vmax = float(np.percentile(truth, 98))
        dabs = float(np.percentile(np.abs(diff), 98))

        title_suffix = f"  level {level_idx}" if n_chans > 1 else ""
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
