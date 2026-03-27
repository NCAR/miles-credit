"""
WeatherBench2-style scorecard plots for CREDIT forecasts.

Reads a WB2 scores CSV produced by eval_weatherbench.py and generates:
  - RMSE vs lead time (per variable, with IFS/Pangu/GraphCast reference lines)
  - ACC  vs lead time (per variable, with reference lines where available)
  - Bias vs lead time (per variable)
  - Scorecard heatmap (skill score vs IFS HRES)
  - Regional RMSE breakdown (global / tropics / extratropics)

Usage
-----
python plot_weatherbench.py --scores wb2_scores.csv --out figures/
python plot_weatherbench.py --scores wb2_scores.csv --label "WXFormer v2" --out figures/ --no-refs
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit.verification.wb2_references import WB2_SCORES, WB2_STYLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variables to plot and their display metadata
# ---------------------------------------------------------------------------
PLOT_VARS = {
    "Z500": {"title": "Z500 (500 hPa Geopotential)", "rmse_unit": "m²/s²", "acc": True},
    "T850": {"title": "T850 (850 hPa Temperature)", "rmse_unit": "K", "acc": True},
    "t2m": {"title": "2 m Temperature", "rmse_unit": "K", "acc": False},
    "U850": {"title": "U850 (850 hPa Zonal Wind)", "rmse_unit": "m/s", "acc": False},
    "V850": {"title": "V850 (850 hPa Meridional Wind)", "rmse_unit": "m/s", "acc": False},
    "T500": {"title": "T500 (500 hPa Temperature)", "rmse_unit": "K", "acc": False},
    "SP": {"title": "Surface Pressure", "rmse_unit": "Pa", "acc": False},
}

# Regions for regional breakdown plots
REGIONS = ["global", "tropics", "n_extratropics", "s_extratropics"]
REGION_LABELS = {
    "global": "Global",
    "tropics": "Tropics (20°S–20°N)",
    "n_extratropics": "N. Extratropics (20–90°N)",
    "s_extratropics": "S. Extratropics (90–20°S)",
}

# Standard WB2 lead times for x-axis ticks
DAY_TICKS = [1, 2, 3, 4, 5, 6, 7, 10]


def _day_ticks(ax, max_hours):
    """Set x-axis to show days."""
    ticks_h = [d * 24 for d in DAY_TICKS if d * 24 <= max_hours]
    ax.set_xticks(ticks_h)
    ax.set_xticklabels([f"Day {d}" for d in DAY_TICKS if d * 24 <= max_hours], rotation=30, ha="right")


def _add_ref_lines(ax, varname, metric, label_refs=True):
    """Overlay WB2 reference model lines on ax."""
    for model, style in WB2_STYLE.items():
        data = WB2_SCORES.get(model, {}).get(varname, {}).get(metric)
        if data is None:
            continue
        lts, vals = zip(*data)
        ax.plot(
            lts,
            vals,
            label=model if label_refs else "_nolegend_",
            **style,
        )


def plot_rmse(df, out_dir, label="CREDIT", show_refs=True):
    """RMSE vs lead time for each variable — one panel per variable."""
    os.makedirs(out_dir, exist_ok=True)
    for var, meta in PLOT_VARS.items():
        col = f"rmse_{var}"
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        lt = df["lead_time_hours"].values
        ax.plot(lt, df[col].values, "k-o", linewidth=2, markersize=4, label=label, zorder=5)

        if show_refs:
            _add_ref_lines(ax, var, "rmse")

        max_lt = lt.max()
        _day_ticks(ax, max_lt)
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"RMSE [{meta['rmse_unit']}]")
        ax.set_title(f"{meta['title']} — RMSE")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_lt + 12)
        ax.set_ylim(bottom=0)

        path = os.path.join(out_dir, f"rmse_{var}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_acc(df, out_dir, label="CREDIT", show_refs=True):
    """ACC vs lead time for variables where ACC is in the scores CSV."""
    os.makedirs(out_dir, exist_ok=True)
    for var, meta in PLOT_VARS.items():
        if not meta["acc"]:
            continue
        col = f"acc_{var}"
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        lt = df["lead_time_hours"].values
        ax.plot(lt, df[col].values, "k-o", linewidth=2, markersize=4, label=label, zorder=5)

        if show_refs:
            _add_ref_lines(ax, var, "acc")

        # ACC = 0.6 is a common "skill limit" line
        max_lt = lt.max()
        ax.axhline(0.6, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="ACC = 0.6")

        _day_ticks(ax, max_lt)
        ax.set_xlabel("Lead time")
        ax.set_ylabel("ACC")
        ax.set_title(f"{meta['title']} — Anomaly Correlation Coefficient")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_lt + 12)
        ax.set_ylim(0.4, 1.02)

        path = os.path.join(out_dir, f"acc_{var}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_bias(df, out_dir, label="CREDIT"):
    """Bias vs lead time — global mean bias per variable."""
    os.makedirs(out_dir, exist_ok=True)
    bias_cols = [c for c in df.columns if c.startswith("bias_") and not any(r in c for r in REGIONS)]
    vars_found = [c.replace("bias_", "") for c in bias_cols]

    if not vars_found:
        return

    n = len(vars_found)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    lt = df["lead_time_hours"].values
    for i, var in enumerate(vars_found):
        ax = axes[i]
        ax.plot(lt, df[f"bias_{var}"].values, "k-o", linewidth=1.5, markersize=3, label=label)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        meta = PLOT_VARS.get(var, {})
        _day_ticks(ax, lt.max())
        ax.set_title(var)
        ax.set_ylabel(f"Bias [{meta.get('rmse_unit', '')}]")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{label} — Mean Bias (forecast − ERA5)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "bias_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_scorecard(df, out_dir, label="CREDIT"):
    """
    WB2-style scorecard heatmap: skill score vs IFS HRES at key lead times.
    SS = 1 - RMSE_model / RMSE_IFS.  Green = better than IFS, red = worse.
    """
    os.makedirs(out_dir, exist_ok=True)

    key_vars = [v for v in PLOT_VARS if f"rmse_{v}" in df.columns]
    key_days = [1, 2, 3, 5, 7, 10]
    key_lts = [d * 24 for d in key_days]

    # Build skill score matrix
    rows = []
    for var in key_vars:
        col = f"rmse_{var}"
        ref_data = WB2_SCORES.get("IFS HRES", {}).get(var, {}).get("rmse")
        if ref_data is None:
            continue
        ref_dict = dict(ref_data)
        row_vals = []
        for lt in key_lts:
            credit_row = df[df["lead_time_hours"] == lt]
            if credit_row.empty or lt not in ref_dict:
                row_vals.append(np.nan)
                continue
            credit_rmse = credit_row[col].values[0]
            ifs_rmse = ref_dict[lt]
            ss = 1.0 - credit_rmse / ifs_rmse
            row_vals.append(ss)
        rows.append(row_vals)

    if not rows:
        logger.warning("No IFS reference data to build scorecard")
        return

    skill = np.array(rows)
    var_labels = [v for v in key_vars if WB2_SCORES.get("IFS HRES", {}).get(v, {}).get("rmse")]

    fig, ax = plt.subplots(figsize=(len(key_days) * 1.2 + 2, len(var_labels) * 0.6 + 1.5))
    im = ax.imshow(skill, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")

    # Annotate cells
    for r in range(skill.shape[0]):
        for c in range(skill.shape[1]):
            val = skill[r, c]
            if np.isfinite(val):
                txt = f"{val:+.2f}"
                color = "black" if abs(val) < 0.2 else "white"
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(len(key_days)))
    ax.set_xticklabels([f"Day {d}" for d in key_days])
    ax.set_yticks(range(len(var_labels)))
    ax.set_yticklabels(var_labels)
    ax.set_title(f"{label} — Skill Score vs IFS HRES\n(SS = 1 − RMSE_model/RMSE_IFS; green = better)")

    plt.colorbar(im, ax=ax, label="Skill Score", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(out_dir, "scorecard_vs_ifs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_regional_rmse(df, out_dir, label="CREDIT"):
    """Regional RMSE breakdown: one figure per variable, four lines for regions."""
    os.makedirs(out_dir, exist_ok=True)
    colors = {"global": "black", "tropics": "#d62728", "n_extratropics": "#1f77b4", "s_extratropics": "#9467bd"}

    for var in PLOT_VARS:
        # Check if regional columns exist
        region_cols = [f"rmse_{var}_{r}" for r in REGIONS]
        if not any(c in df.columns for c in region_cols):
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        lt = df["lead_time_hours"].values
        for region in REGIONS:
            col = f"rmse_{var}_{region}"
            if col not in df.columns:
                continue
            ax.plot(lt, df[col].values, label=REGION_LABELS[region], color=colors[region], linewidth=1.8)

        meta = PLOT_VARS.get(var, {})
        _day_ticks(ax, lt.max())
        ax.set_xlabel("Lead time")
        ax.set_ylabel(f"RMSE [{meta.get('rmse_unit', '')}]")
        ax.set_title(f"{label} — {meta.get('title', var)} RMSE by Region")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        path = os.path.join(out_dir, f"regional_rmse_{var}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def plot_all(scores_csv, out_dir, label="CREDIT", show_refs=True):
    df = pd.read_csv(scores_csv)
    logger.info(f"Loaded {len(df)} lead-time rows from {scores_csv}")

    plot_rmse(df, out_dir, label=label, show_refs=show_refs)
    plot_acc(df, out_dir, label=label, show_refs=show_refs)
    plot_bias(df, out_dir, label=label)
    plot_scorecard(df, out_dir, label=label)
    plot_regional_rmse(df, out_dir, label=label)

    logger.info(f"All figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="WeatherBench2-style scorecard plots for CREDIT")
    parser.add_argument("--scores", required=True, help="WB2 scores CSV from eval_weatherbench.py")
    parser.add_argument("--out", default="wb2_figures", help="Output directory for PNG figures")
    parser.add_argument("--label", default="CREDIT", help="Model label for legend")
    parser.add_argument("--no-refs", action="store_true", help="Omit IFS/Pangu/GraphCast reference lines")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    plot_all(args.scores, args.out, label=args.label, show_refs=not args.no_refs)


if __name__ == "__main__":
    main()
