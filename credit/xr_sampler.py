from glob import glob
from credit.data import drop_var_from_dataset, get_forward_data
import numpy as np
import yaml
import pandas as pd
import xarray as xr


class XRSamplerByYear:
    """
    Given a conf and datetime, samples an xarray with only variables specified in conf["data"].
    Supports both single-file (CAM/CESM) and multi-file (ERA5) configurations where
    different variable groups may live in different zarr/netCDF stores.

    - Excludes static variables
    - Only compatible with files that have the year in their filename
    - Variable groups that share the same file pattern are loaded once and merged
    """

    def __init__(self, config_path, conf=None):
        if not conf:
            with open(config_path) as cf:
                conf = yaml.load(cf, Loader=yaml.FullLoader)

        data_conf = conf["data"]

        # Map each variable group to its source file pattern.
        # Keys are optional — groups absent from the config are silently skipped.
        group_keys = [
            ("variables", "save_loc"),
            ("surface_variables", "save_loc_surface"),
            ("dynamic_forcing_variables", "save_loc_dynamic_forcing"),
            ("diagnostic_variables", "save_loc_diagnostic"),
            ("forcing_variables", "save_loc_forcing"),
        ]

        # Accumulate variables per unique file pattern so each file is loaded once.
        path_to_vars: dict[str, list] = {}
        for var_key, loc_key in group_keys:
            var_list = data_conf.get(var_key, [])
            if not var_list:
                continue
            loc = data_conf.get(loc_key)
            if not loc:
                continue
            if loc not in path_to_vars:
                path_to_vars[loc] = []
            for v in var_list:
                if v not in path_to_vars[loc]:
                    path_to_vars[loc].append(v)

        # Resolve globs once at construction time.
        self._path_groups = [
            {"filenames": sorted(glob(pattern)), "variables": vars_list} for pattern, vars_list in path_to_vars.items()
        ]

        # Flat list of all tracked variables (for external reference).
        self.variables = [v for g in self._path_groups for v in g["variables"]]

        self.current_file_year = -1

    def __call__(self, timestamp: np.datetime64):
        year = timestamp.astype("datetime64[Y]").astype(int) + 1970

        if year != self.current_file_year:
            self.current_file_year = year
            datasets = []
            for group in self._path_groups:
                matching = [fn for fn in group["filenames"] if str(year) in fn]
                if len(matching) != 1:
                    raise RuntimeError(f"expected exactly 1 file for year {year}, found {len(matching)}: {matching}")
                ds = get_forward_data(matching[0])
                ds = drop_var_from_dataset(ds, group["variables"])
                if not isinstance(ds.time[0].values, np.datetime64):
                    time_index = pd.DatetimeIndex(ds["time"].astype("datetime64[ns]").values)
                    ds["time"] = time_index
                datasets.append(ds)

            self.ds = datasets[0] if len(datasets) == 1 else xr.merge(datasets)

        return self.ds.loc[{"time": [timestamp]}]


if __name__ == "__main__":
    config = "/glade/work/dkimpara/CREDIT_runs/cesm_rollout/model.yml"
    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    sampler = XRSamplerByYear(conf)

    timestamp = np.datetime64("2005-02-25", "h")
    ds = sampler(timestamp)
    print(ds.time.values)
