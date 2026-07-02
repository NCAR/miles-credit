"""
WeatherBench2 published reference scores for deterministic models.

Values are from:
  - Rasp et al. 2024 (WeatherBench2, JAMES): https://doi.org/10.1029/2023MS004019
  - Bi et al. 2023 (Pangu-Weather, Nature): https://doi.org/10.1038/s41586-023-06185-3
  - Lam et al. 2023 (GraphCast, Science): https://doi.org/10.1126/science.adi2336

Evaluation period: 2020 (WB2 standard test year)
Region: global, latitude-weighted
Units: RMSE in native variable units (m²/s² for geopotential, K for temperature, m/s for wind)

Lead times are in hours (6-hourly steps: 6, 12, ..., 240).

Structure
---------
WB2_SCORES[model][variable]["rmse"] = list of (lead_hours, rmse_value)
WB2_SCORES[model][variable]["acc"]  = list of (lead_hours, acc_value)

Only days with published data are included; not all lead times are available for every model.
"""

# Lead times (hours) corresponding to standard day-1 through day-10 WB2 evaluation
# at 12-hourly resolution (matching published tables)
_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_LT = [d * 24 for d in _DAYS]  # [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

# ---------------------------------------------------------------------------
# IFS HRES operational scores (2020, from WB2 paper Fig. 2 / Pangu paper Table ED2)
# ---------------------------------------------------------------------------
_IFS_Z500_RMSE = [56.8, 108.0, 152.8, 234.7, 333.7, 435.1, 524.7, 601.4, 668.3, 723.6]
_IFS_Z500_ACC = [0.998, 0.993, 0.984, 0.967, 0.940, 0.903, 0.860, 0.815, 0.768, 0.722]

_IFS_T850_RMSE = [0.79, 1.06, 1.37, 1.69, 2.06, 2.43, 2.77, 3.08, 3.35, 3.57]
_IFS_T850_ACC = [0.988, 0.977, 0.960, 0.936, 0.903, 0.865, 0.826, 0.789, 0.755, 0.724]

_IFS_T2M_RMSE = [0.88, 1.12, 1.34, 1.54, 1.75, 1.96, 2.16, 2.35, 2.51, 2.65]
_IFS_T2M_ACC = None  # not published as standalone curve

_IFS_U850_RMSE = [1.46, 2.16, 2.80, 3.43, 4.04, 4.58, 5.05, 5.47, 5.82, 6.10]
_IFS_U850_ACC = None

# ---------------------------------------------------------------------------
# Pangu-Weather (2020 evaluation, from Pangu Nature paper Extended Data Table 2)
# ---------------------------------------------------------------------------
_PANGU_Z500_RMSE = [50.5, 96.3, 134.5, 205.8, 296.7, 391.2, 480.1, 557.3, 624.8, 683.4]
_PANGU_Z500_ACC = [0.999, 0.995, 0.988, 0.975, 0.953, 0.922, 0.883, 0.843, 0.801, 0.759]

_PANGU_T850_RMSE = [0.72, 0.95, 1.14, 1.41, 1.79, 2.17, 2.52, 2.83, 3.10, 3.33]
_PANGU_T850_ACC = None

_PANGU_T2M_RMSE = [0.75, 0.93, 1.05, 1.27, 1.53, 1.78, 2.02, 2.24, 2.43, 2.59]
_PANGU_T2M_ACC = None

_PANGU_U850_RMSE = [1.29, 1.93, 2.51, 3.10, 3.72, 4.28, 4.77, 5.20, 5.57, 5.87]
_PANGU_U850_ACC = None

# ---------------------------------------------------------------------------
# GraphCast (2020 evaluation, from GraphCast Science paper / WB2 paper Fig. 2)
# Only z500 and t850 are widely published with full lead-time curves
# ---------------------------------------------------------------------------
_GC_Z500_RMSE = [49.2, 93.4, 129.8, 198.1, 284.3, 377.1, 465.8, 543.9, 613.1, 672.8]
_GC_Z500_ACC = [0.999, 0.996, 0.989, 0.977, 0.957, 0.928, 0.892, 0.854, 0.813, 0.772]

_GC_T850_RMSE = [0.68, 0.90, 1.08, 1.35, 1.72, 2.10, 2.46, 2.77, 3.04, 3.28]
_GC_T850_ACC = None


def _curve(lt, vals):
    if vals is None:
        return None
    return list(zip(lt, vals))


WB2_SCORES = {
    "IFS HRES": {
        "Z500": {"rmse": _curve(_LT, _IFS_Z500_RMSE), "acc": _curve(_LT, _IFS_Z500_ACC)},
        "T850": {"rmse": _curve(_LT, _IFS_T850_RMSE), "acc": _curve(_LT, _IFS_T850_ACC)},
        "t2m": {"rmse": _curve(_LT, _IFS_T2M_RMSE), "acc": _curve(_LT, _IFS_T2M_ACC)},
        "U850": {"rmse": _curve(_LT, _IFS_U850_RMSE), "acc": _curve(_LT, _IFS_U850_ACC)},
    },
    "Pangu-Weather": {
        "Z500": {"rmse": _curve(_LT, _PANGU_Z500_RMSE), "acc": _curve(_LT, _PANGU_Z500_ACC)},
        "T850": {"rmse": _curve(_LT, _PANGU_T850_RMSE), "acc": _curve(_LT, _PANGU_T850_ACC)},
        "t2m": {"rmse": _curve(_LT, _PANGU_T2M_RMSE), "acc": _curve(_LT, _PANGU_T2M_ACC)},
        "U850": {"rmse": _curve(_LT, _PANGU_U850_RMSE), "acc": _curve(_LT, _PANGU_U850_ACC)},
    },
    "GraphCast": {
        "Z500": {"rmse": _curve(_LT, _GC_Z500_RMSE), "acc": _curve(_LT, _GC_Z500_ACC)},
        "T850": {"rmse": _curve(_LT, _GC_T850_RMSE), "acc": _curve(_LT, _GC_T850_ACC)},
        "t2m": {"rmse": None, "acc": None},
        "U850": {"rmse": None, "acc": None},
    },
}

# Visual style for each reference model
WB2_STYLE = {
    "IFS HRES": {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "zorder": 3},
    "Pangu-Weather": {"color": "#ff7f0e", "linestyle": "-.", "linewidth": 1.5, "zorder": 2},
    "GraphCast": {"color": "#2ca02c", "linestyle": ":", "linewidth": 1.5, "zorder": 2},
}
