from glob import glob
import os
import netCDF4 as nc


def find_key(nested, key):
    """Recursive breadth-first search to find the value associated
    with the first instance of a key in a nested dict"""
    if isinstance(nested, dict):
        if key in nested.keys():
            return nested[key]

        for k in nested.keys():
            found = find_key(nested[k], key)
            if found is not None:
                return found
        return None
    return None


def count_channels(conf):
    """Tallies up the total number of tensor channels resulting from a
    'datasets' configurtion, taking into account z-levels in 3D
    variables and a possible z-stride across them.

    """
    dconf = find_key(conf, "datasets") or conf
    total = {"boundary": 0, "prognostic": 0, "diagnostic": 0}
    for dset in dconf:
        dim = dconf[dset]["dim"]
        dim = dim.upper() if len(dim) < 3 else dim.lower()

        if dim == "3D":
            # need to count levels, but only for 1st var/file;
            # everything is (assumed to be) uniform
            ncpath = glob(os.path.join(dconf[dset]["rootpath"], dconf[dset]["glob"]))[0]
            with nc.Dataset(ncpath, "r") as ncfile:
                var = list(dconf[dset]["variables"].values())[0][0]
                # data should be dimensioned [T, Z, Y, X]
                nlev = ncfile[var].shape[1]

            if "zstride" in dconf[dset]:
                nlev = len(range(0, nlev, dconf[dset]["zstride"]))
        else:
            nlev = 1

        for usage in total:
            uvars = dconf[dset]["variables"]
            if usage in uvars:
                total[usage] += len(uvars[usage]) * nlev

    return total
