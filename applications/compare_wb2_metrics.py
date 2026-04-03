"""
Compare eval_weatherbench.py output (wb2_scores.csv) against ensemble_metrics.py
aggregated CSVs. RMSE should match; ACC should differ (theirs is Pearson, ours is true anomaly).
"""

import argparse
import pandas as pd
import numpy as np

# Their variable names → my column suffix in wb2_scores.csv
VAR_MAP = {
    "T_PRES_500mb": "T500",
    "T_PRES_700mb": "T700",
    "T_PRES_850mb": "T850",
    "U_PRES_500mb": "U500",
    "U_PRES_850mb": "U850",
    "V_PRES_500mb": "V500",
    "V_PRES_850mb": "V850",
    "Q_PRES_500mb": "Q500",
    "Q_PRES_850mb": "Q850",
    "Z500": "Z500",
    "t2m": "t2m",
    "SP": "SP",
}

LEAD_TIMES_H = [6, 24, 48, 72, 120, 168, 240]  # key lead times to spot-check


def load_theirs(agg_csv):
    df = pd.read_csv(agg_csv)
    # keep only vars we care about, rename lead_time column
    df = df[df["variable"].isin(VAR_MAP.keys())].copy()
    df = df.rename(columns={"lead_time": "lead_time_hours"})
    return df


def load_mine(scores_csv):
    return pd.read_csv(scores_csv)


def compare(theirs_df, mine_df):
    rows = []
    for their_var, my_var in VAR_MAP.items():
        their = theirs_df[theirs_df["variable"] == their_var].set_index("lead_time_hours")
        rmse_col = f"rmse_{my_var}"
        acc_col = f"acc_{my_var}"

        # global rmse column (no region suffix) or _global suffix
        if rmse_col not in mine_df.columns:
            rmse_col = f"rmse_{my_var}_global"
        if acc_col not in mine_df.columns:
            acc_col = f"acc_{my_var}_global"

        mine = mine_df.set_index("lead_time_hours")

        for lt in LEAD_TIMES_H:
            if lt not in their.index or lt not in mine.index:
                continue

            their_rmse = their.loc[lt, "rmse"]
            their_acc = their.loc[lt, "acc"]

            my_rmse = mine.loc[lt, rmse_col] if rmse_col in mine.columns else np.nan
            my_acc = mine.loc[lt, acc_col] if acc_col in mine.columns else np.nan

            rmse_pct = (my_rmse - their_rmse) / their_rmse * 100 if not np.isnan(my_rmse) else np.nan

            rows.append(
                {
                    "variable": my_var,
                    "lead_h": lt,
                    "their_rmse": their_rmse,
                    "my_rmse": my_rmse,
                    "rmse_diff%": rmse_pct,
                    "their_acc": their_acc,
                    "my_acc": my_acc,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theirs",
        default=(
            "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/production"
            "/sixteen_tune/netcdf/metrics/aggregated_metrics.csv"
        ),
    )
    parser.add_argument("--mine", default="/glade/work/schreck/wb2_scores.csv")
    args = parser.parse_args()

    print(f"Loading theirs: {args.theirs}")
    print(f"Loading mine:   {args.mine}")

    theirs = load_theirs(args.theirs)
    mine = load_mine(args.mine)

    print(f"\nMy columns: {[c for c in mine.columns if 'rmse' in c or 'acc' in c][:20]}")
    print(f"My lead times: {sorted(mine['lead_time_hours'].unique())}\n")

    cmp = compare(theirs, mine)

    pd.set_option("display.float_format", "{:.5g}".format)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)

    print("=" * 80)
    print("RMSE COMPARISON (their=ensemble_metrics, mine=eval_weatherbench)")
    print("rmse_diff% = (mine - theirs) / theirs * 100  [should be ~0]")
    print("=" * 80)
    rmse_tbl = cmp.pivot_table(index="variable", columns="lead_h", values="rmse_diff%")
    print(rmse_tbl.to_string())

    print()
    print("=" * 80)
    print("ACC COMPARISON (their=Pearson/wrong, mine=true anomaly/correct)")
    print("their_acc should be ~1.0 at all leads; mine should decrease from ~1.0 at 6h to ~0.6-0.7 at day 7")
    print("=" * 80)
    for var in VAR_MAP.values():
        sub = cmp[cmp["variable"] == var][["lead_h", "their_acc", "my_acc"]]
        if sub.empty:
            continue
        print(f"\n  {var}:")
        print(sub.to_string(index=False))

    print()
    print("=" * 80)
    print("FULL TABLE")
    print("=" * 80)
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
