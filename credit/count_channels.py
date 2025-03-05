import yaml
import netCDF4 as nc
from glob import glob
import os


def count_channels(conf):
    dconf = conf['data']['datasets']
    total = {'boundary': 0, 'prognostic': 0, 'diagnostic': 0}
    for dset in dconf:
        dim = dconf[dset]['dim']
        dim = dim.upper() if len(dim) < 3 else dim.lower()

        if dim == '3D':
            # need to count levels, but only for 1st var/file; everything is uniform
            ncpath = glob(os.path.join(dconf[dset]['rootpath'],dconf[dset]['glob']))[0]
            with nc.Dataset(ncpath, 'r') as ncfile:
                var = list(dconf[dset]['variables'].values())[0][0]
                # data should be dimensioned [T, Z, Y, X]
                nlev = ncfile[var].shape[1]
                
            if 'zstride' in dconf[dset]:
                nlev = len(range(0, nlev, dconf[dset]['zstride']))
        else:
            nlev = 1
            
        for usage in total.keys():
            vars = dconf[dset]['variables']
            if usage in vars:
                total[usage] += len(vars[usage]) * nlev
                
    return total
