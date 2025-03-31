from glob import glob
from credit.data import drop_var_from_dataset, get_forward_data
import numpy as np
import yaml
import pandas as pd


class XRSamplerByYear:
    """
    given a conf and datetime, samples an xarray with only variables specified in conf["data"]
    WARNING: only compatible with single zarr store for all variables
    - excludes static variables
    - only compatible with files with the year in their filename
    
    future features: 
    - keep variables based on predict.save_vars
    - handle multiple zarr stores 
    """
    def __init__(self, 
                 config_path,
                 conf=None):
        if not conf:
            # Load the configuration and get the relevant variables
            with open(config_path) as cf:
                conf = yaml.load(cf, Loader=yaml.FullLoader)

        data_conf = conf["data"]

        # check if all source files same (cesm compatible, not compatible for era5)
        save_keywords = ['save_loc', 'save_loc_surface', 'save_loc_dynamic_forcing', 
                         'save_loc_diagnostic', 'save_loc_forcing']
        filestr = data_conf[f"{save_keywords[0]}"]

        for kw in save_keywords:
            if filestr != data_conf[kw]:
                raise RuntimeError("this sampler can only handle one dataset file for all vars")
            filestr = data_conf[kw]

        # get list of files
        self.filenames = glob(filestr)
        
        # get variables to sample
        self.variables = []
        for data_kw in ['variables', 'surface_variables', 'dynamic_forcing_variables',
                             'diagnostic_variables', 'forcing_variables']:
            self.variables += data_conf[data_kw]

        # retain the currently loaded file's year
        self.current_file_year = -1


    def __call__(self, timestamp: np.datetime64): # np.datetime64
        year = timestamp.astype('datetime64[Y]').astype(int) + 1970
        
        if year != self.current_file_year:
            self.current_file_year = year
            # load a new dataset
            filename = [fn for fn in self.filenames if str(year) in fn]
            if len(filename) != 1: 
                raise RuntimeError("filenames do not have unique years")
            
            self.ds = get_forward_data(filename[0])
            
            # drop variables
            self.ds = drop_var_from_dataset(self.ds, self.variables)

            # convert to datetime index if needed (in case of cftime)
            if not isinstance(self.ds.time[0].values , np.datetime64):
                time_index = pd.DatetimeIndex(self.ds['time'].astype("datetime64[ns]").values)
                self.ds['time'] = time_index

        return self.ds.loc[{"time": [timestamp]}]




if __name__ == "__main__":

    config = "/glade/work/dkimpara/CREDIT_runs/cesm_rollout/model.yml"
    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    sampler = XRSamplerByYear(conf)

    timestamp = np.datetime64('2005-02-25', "h")
    ds = sampler(timestamp)
    print(ds.time.values)
    