# Evaluation

CREDIT supports two evaluation workflows:

## Quick per-init metrics

`applications/rollout_metrics.py` computes per-initialisation metrics (RMSE,
ACC, bias) and writes individual CSVs. These can be aggregated for ensemble
verification.

## WeatherBench2-style evaluation

`applications/eval_weatherbench.py` computes latitude-weighted RMSE, true
anomaly ACC, and bias against ERA5, then generates scorecard plots comparable
to IFS HRES, WXFormer v1, and FuXi.

See [WeatherBench Evaluation](WeatherBench.md) for full details on the
methodology, truth source, climatology, and reference model lines.
