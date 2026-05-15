"""
rollout_utils_gen2.py
---------------------
Shared utilities for gen2 rollout scripts (rollout_to_netcdf_gen2 and
rollout_realtime_gen2).  Centralises the three helper functions that would
otherwise be duplicated between the two entry points.
"""

import torch


def _inject_flat_schema(conf):
    """Inject v1-style flat keys into conf['data'] so output.py utilities work."""
    if "variables" in conf["data"]:
        return
    src = conf["data"]["source"]["ERA5"]
    v = src["variables"]
    prog = v.get("prognostic") or {}
    diag = v.get("diagnostic") or {}
    conf["data"]["variables"] = prog.get("vars_3D", [])
    conf["data"]["surface_variables"] = prog.get("vars_2D", [])
    conf["data"]["diagnostic_variables"] = diag.get("vars_2D", []) if diag else []
    conf["data"]["level_ids"] = src.get("levels", list(range(conf["model"]["levels"])))
    if "scaler_type" not in conf["data"]:
        conf["data"]["scaler_type"] = "std_new"


def _inject_tracer_inds(conf):
    """Compute tracer_inds for TracerFixer from v2 variable layout."""
    tracer_conf = conf.get("model", {}).get("post_conf", {}).get("tracer_fixer", {})
    if not tracer_conf.get("activate", False) or "tracer_inds" in tracer_conf:
        return
    src = conf["data"]["source"]["ERA5"]
    n_levels = len(src.get("levels", []))
    v = src["variables"]
    vars_3d = (v.get("prognostic") or {}).get("vars_3D", [])
    vars_2d = (v.get("prognostic") or {}).get("vars_2D", [])
    diag_2d = (v.get("diagnostic") or {}).get("vars_2D", [])
    output_vars = [vn for vn in vars_3d for _ in range(n_levels)] + vars_2d + diag_2d
    thres_map = dict(zip(tracer_conf.get("tracer_name", []), tracer_conf.get("tracer_thres", [])))
    inds, thres = [], []
    for i, vn in enumerate(output_vars):
        if vn in thres_map:
            inds.append(i)
            thres.append(thres_map[vn])
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_inds"] = inds
    conf["model"]["post_conf"]["tracer_fixer"]["tracer_thres"] = thres
    conf["model"]["post_conf"]["tracer_fixer"]["denorm"] = False


def _sample_to_batch(sample, source_name):
    """Add batch dim and wrap a LocalDataset sample into preblock-compatible format.

    Produces the MultiSourceDataset structure expected by ERA5Normalizer and
    ConcatToTensor::

        {"input": {source_name: {var_key: tensor}}, "metadata": {source_name: {...}}}
    """
    meta = {
        k: torch.tensor(v).unsqueeze(0) if isinstance(v, (int, float)) else v for k, v in sample["metadata"].items()
    }
    return {
        "input": {source_name: {k: v.unsqueeze(0) for k, v in sample["input"].items()}},
        "metadata": {source_name: meta},
    }
