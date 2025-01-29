#!/usr/bin/env python
"""Converts a directory of netcdf files into a single zarr store.

On Casper, zarrifying 1 year of hourly conus404 data (1 variable,
1015x1367 = 1.39e6 gridpoints, 8760 timesteps, total data volume
~25GB) takes ~10 minutes wallclock and requires about 300MB of memory.

Author: Seth McGinnis
Contact: mcginnis@ucar.edu

# Dependencies:
# - xarray
# - os.path
# - glob
# - argparse

"""

import xarray
import os.path
import glob
from argparse import ArgumentParser


parser = ArgumentParser(
    description="converts a directory of netcdf files into a zarr store"
)
parser.add_argument("indir", help="directory containing one or more .nc files")
parser.add_argument("zarrfile", help="zarr store to create (must not already exist)")

## locals().update creates these, but declare them to pacify flake
indir, zarrfile = None, None
locals().update(vars(parser.parse_args()))  ## convert args into vars

inglob = indir + "/*.nc"

## check output first since glob could be slow if large
if os.path.exists(os.path.expanduser(zarrfile)):
    print("error: zarrfile must not already exist")
    quit()

if len(glob.glob(os.path.expanduser(inglob))) < 1:
    print("error: input directory contains no .nc files")
    quit()


ds = xarray.open_mfdataset(inglob)

## delete all global attributes (WRF has *many* superfluous & obfuscatory atts)
ds.attrs = {}

## With lossy compression applied, xarray autochunking creates too
## many files; specify it manually.  We want each chunk to be 1 day
## (24 hours) over all space; i.e., the same as the netcdf input
## files.

chunks = dict(ds.sizes)
timedim = list(chunks.keys())[0]
chunks[timedim] = 24

ds = ds.chunk(chunks)

## >> If chunking is the same as the netcdf files, would virtual
## >> zarrfiles from kerchunk work well here?

ds.to_zarr(zarrfile)
