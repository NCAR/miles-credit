"""datamap.py
--------------------------------------------------

The xarray library can be slow on very large datasets, so we need to
manage data manually when training ML models.  A datamap object
performs many of the same functions as an xarray.Dataset (not to be
confused with a torch.utils.data.Dataset), but drops a lot of
functionality and checking so it's faster.

A datamap provides a list-like interface to a set of netcdf files,
virtually concatenating them along the time dimension.  It tracks
which index values live in which file, and lazily opens files and
reads data only for the requested indexes.  It also can interconvert
datetimes and indexes.

NOTE: the datamap object assumes that:

1) The time coordinates are uniformly spaced and have no gaps (i.e.,
after setup, it only tracks how many timesteps are in each file, not
the actual values of the time coordinate)

2) Lexicographic sort of the filenames is equivalent to chronological
ordering of the data.  This is true if the filesnames are uniformly
named except for a time element, and that the time element is ISO-8601
(YYYY-MM-DD etc.)

3) Contents of the files are uniform.  All of the variables in the
datamap have the same dimensionality, and all variables exist in each
file.

4) The time dimension has a coordinate variable with the attributes
'axis', 'calendar', and 'units' set to the appropriate CF-compliant
values.

Content:

"""

import os
from typing import List, TypedDict
from dataclasses import dataclass, field
from glob import glob
import netCDF4 as nc
import cftime as cf
import numpy as np


class VarDict(TypedDict, total=False):
    """a dictionary of the variables that could be in a dataset"""

    boundary: List
    prognostic: List
    diagnostic: List
    unused: List


def rescale_minmax(x):
    """rescale data to [0,1].  Don't use
    `sklearn.preprocessing.minmax_scale` because it requires reshaping
    the data, which is silly for a use case this simple.

    """
    x = x - np.min(x)
    xmax = np.max(x)
    if xmax > 0:
        x = x / np.max(x)
    return x


# The following table illustrates the indexing problem that the
# DataMap class solves.  Suppose we have a dataset of daily data
# organized into monthly files.  For reasons (e.g., a t/t/v split),
# we're using a subset of the data specified by a first and last date
# of 2000-12-22 through 2001-12-21 (winter solstice).  We're training
# using 3 timesteps of history and 2 timesteps of forecast, for a
# total sample length of 5.  The time coordinates of the data are
# stored netcdf-style as minutes since 2000-01-01.  The effective
# length of this dataset is 361 samples.

# If the dataset is large, you don't want to read everything into
# memory; you want to load data lazily when it's needed.  This can be
# done using xarray -- unless the dataset consists of a very large
# number of netcdf files.  The xarray.open_mfdataset() method uses the
# netcdf4.open_mfdata() method, which becomes unusably slow when the
# number of files is large.  You can avoid this problem if you convert
# the data to zarr, but if the way you normally convert netcdf to zarr
# is via xarray, you're stuck in the same bind.  You can concatenate
# the netcdf files together into a smaller number of files, but that,
# too, can be impractically slow if the chunking is bad and you don't
# use all the correct magical incantations to speed up concatenation.

# Some tests show that in an HPC environment, netcdf can outperform
# zarr, especially if the data is organized inside the file and
# chunked in a way that matches the way the data will be read.  So we
# need code that will combine a large number of netcdf files into a
# single virtual dataset with an index-based interface that lazily
# loads the data on demand.  That's what the DataMap class does,
# without using open_mfdataset() and without all the overhead of
# building xarray indexes.

# Going back to the example, a call to __getitem__(8) (or dmap[8])
# will open up files 0 and 1 (2000-12 and 2001-01) and read timesteps
# 29-30 of the former and 0-2 of the latter, and combine and return
# them as a 5-timestep sample.

# The leftmost column of the table is the time index, which is the
# number of the timestep within the concatenated collection of netcdf
# files.  If you have a list of the last time indexes in each file,
# you can np.searchsorted to find which file it it's in; you then
# subtract the first time index in that file (= last time index of
# previous file + 1) to get the index of the timestep within the file.
# The time index also has a linear relationsip with the time
# coordinate (multiply by dt and add the offset from epoch), and you
# can convert the time coordinate to/from a date using the cftime
# library.  The rightmost column is the sample index, which is the
# number of the sample with that timestep as the first forecast
# sample, taking into consideration the starting point of the split
# and the number of historical timesteps in the split (since a full
# sample has to fit entirely within the split).  The sample index is
# the timestep index minus an offset, which is the timestep index of
# the first timestep in the split and minus the number of historical
# timesteps.

# Time index is the numbering that gets used internally to find data
# within files.  Sample index is the numbering that's exposed to the
# outside world via __getitem__.

#  i  | file | fidx | time | mon | day | split | sample | index|
# --------------------------------------------------------------
#   0 |  0   |  0   |      | Dec |  1  |       |        |  -   |
#   1 |  0   |  1   |      | '00 |  2  |       |        |  -   |
#   2 |  0   |  2   |      |  "  |  3  |       |        |  -   |
# ... |  "   | ...  |      |  "  | ... |       |        |      |
#  20 |  0   |      |      |  "  |  21 |       | (1st)  |  -   |
#  21 |  0   |      |      |  "  |  22 | first | hist0  |  -   |
#  22 |  0   |      |      |  "  |  23 |       | hist1  |  -   |
#  23 |  0   |      |      |  "  |  24 |       | hist2  |  -   |
#  24 |  0   |      |      |  "  |  25 |       | fore0  |  0   |
#  25 |  0   |      |      |  "  |  26 |       | fore1  |  1   |
# ... |  "   | ...  | ...  |  "  | ... |       |        | ...  |
#  28 |  0   |  28  | -72  |  "  |  29 |       | ([8])  |  4   |
#  29 |  0   |  29  | -48  |  "  |  30 |       | hist0  |  5   |
#  30 |  0   |  30  | -24  |  "  |  31 |       | hist1  |  6   |
# -------------------------------------------------------------|
#  31 |  1   |  0   |  0   | Jan |  1  |       | hist2  |  7   |
#  32 |  1   |  1   |  24  | '01 |  2  |       | fore0  |  8   |
#  33 |  1   |  2   |  48  |  "  |  3  |       | fore1  |  9   |
# ... |  1   | ...  | ...  |  "  | ... |       |        | ...  |
#  60 |  1   |  29  | 719  | Jan |  30 |       |        |      |
#  61 |  1   |  30  | 720  | Jan |  31 |       |        |      |
# -------------------------------------------------------------|
#  62 |  2   |  0   | 744  | Feb |  1  |       |        |      |
#  63 |  2   |  1   | 768  |  "  |  2  |       |        |      |
# ... |  2   | ...  | ...  |  "  | ... |       |        | ...  |
# -------------------------------------------------------------|
# ...   ...    ...    ...    ...   ...                    ...  |
# -------------------------------------------------------------|
# ... |  11  | ...  | ...  | Nov | ... |       |        | ...  |
# 363 |  11  |  28  |      |  "  |  29 |       |        |      |
# 364 |  11  |  29  |      | Nov |  30 |       |        |      |
# -------------------------------------------------------------|
# 365 |  12  |  0   |      | Dec |  1  |       |        |      |
# ... |  12  | ...  |      | '01 | ... |       | (last) | ...  |
# 381 |  12  |  16  |      |  "  |  17 |       | hist0  | 357  |
# 382 |  12  |  17  |      |  "  |  18 |       | hist1  | 358  |
# 383 |  12  |  18  |      |  "  |  19 |       | hist2  | 359  |
# 384 |  12  |  19  |      |  "  |  20 |       | fore0  | 360  |
# 385 |  12  |  20  |      |  "  |  21 | last  | fore1  |  -   |
# 386 |  12  |  21  |      |  "  |  22 |       |        |  -   |
# ... |  "   | ...  | ...  |  "  | ... |       |        |      |
# 394 |  12  |  29  |      |  "  |  30 |       |        |  -   |
# 395 |  12  |  30  |      | Dec |  31 |       |        |  -   |
# --------------------------------------------------------------


# Written as a dataclass to avoid acres of boilerplate and for free repr(), etc.

# Normal use pattern is that all the args come from a yaml file that
# gets read into a dictionary that is passed via **kwargs


@dataclass
class DataMap:
    """Class for reading in netCDF data from multiple files.

    rootpath: pathway to the files
    glob: filename glob of netcdf files (relative to rootpath)
    dim: dimensions of the data:
        static: no time dimension; data is loaded on initialization
        3D: data has z-dimension; can subset levels using zstride
        2D: default: time-varying 2D data
    normalize: if dim=='static' & normalize == True, scale data to range [0,1]
    zstride: if dim=='3D', subset in Z dimension by ::zstride when reading
    boundary: list of variable names to use as (input-only) boundary conditions
    diagnostic: list of diagnostic (output-only) variable names
    prognostic: list of prognostic (state / input-output) variable names
    unused: list of unused variables (optional)
    history_len: number of input timesteps
    forecast_len: number of output timesteps
    first_date: restrict dataset to timesteps >= this point in time
    last_date: restrict dataset to timesteps <= this point in time

    first_date and last_date default to None, which means use the
    first/last timestep in the dataset.  Note that they must be
    YYYY-MM-DD strings (with optional HH:MM:SS), not datetime objects.

    Note also that the time coordinate must be contiguous across the
    files, with no gaps or overlaps.
    """

    rootpath: str
    glob: str
    dim: str = "2D"
    normalize: bool = False
    zstride: int = 1
    variables: VarDict[str, List] = field(default_factory=list)
    history_len: int = 2
    forecast_len: int = 1
    first_date: str = None
    last_date: str = None

    def __post_init__(self):
        super().__init__()

        self.sample_len = self.history_len + self.forecast_len

        self._mode = "train"

        # todo: auto-detect dataset dimensionality

        # canonicalize capitalization
        self.dim = self.dim.upper() if len(self.dim) < 3 else self.dim.lower()
        if self.dim not in ["static", "2D", "3D"]:
            raise ValueError(f"credit.datamap: unknown dimensionality: {self.dim}")

        if self.normalize and self.dim != "static":
            raise ValueError(
                "credit.datamap: 'normalize' only applies to dim=='static'"
            )

        if self.zstride != 1 and self.dim != "3D":
            raise ValueError("credit.datamap: zstride not applicable if dim != '3D'")

        # set any missing keys in VarDict
        for use in ("boundary", "prognostic", "diagnostic"):
            if use not in self.variables:
                self.variables[use] = ()

        if self.dim == "static":
            if len(glob(self.glob, root_dir=self.rootpath)) != 1:
                raise ValueError("credit.datamap: dim='static' requires a single file")

            if (
                len(self.variables["prognostic"]) > 0
                or len(self.variables["diagnostic"]) > 0
            ):
                raise ValueError("credit.datamap: static vars must be boundary vars")

            # if static, load data from netcdf
            staticfile = nc.Dataset(
                self.rootpath + "/" + self.glob, mask_and_scale=False
            )

            # [:] forces data to load
            staticdata = [
                np.array(staticfile[v][:]) for v in self.variables["boundary"]
            ]
            self.shape = staticdata[0].shape
            self.data = dict(zip(self.variables["boundary"], staticdata))

            # cleanup
            staticfile.close()
            del staticfile, staticdata

            if self.normalize:
                for k in self.data.keys():
                    self.data[k] = rescale_minmax(self.data[k])

        else:
            fileglob = sorted(glob(self.glob, root_dir=self.rootpath))
            self.filepaths = [os.path.join(self.rootpath, f) for f in fileglob]

            # get time coordinate characteristics from first file
            # calendar & units used to convert date <=> time coordinate
            # t0 & dt used to convert time coordinates <=> timestep index
            nc0 = nc.Dataset(self.filepaths[0], mask_and_scale=False)
            time0 = nc0.variables["time"]

            self.calendar = time0.calendar
            self.units = time0.units
            self.t0 = float(time0[0])
            self.dt = float(time0[1]) - self.t0

            # take shape from first variable we're using, minus time dimension.
            v0 = list(self.variables.values())[0][0]
            self.shape = nc0[v0].shape[1:]

            nc0.close()

            if self.first_date is None:
                self.first = 0
            else:
                self.first = self.date2tindex(self.first_date)

            if self.last_date is None:
                nclast = nc.Dataset(self.filepaths[0], mask_and_scale=False)
                t_last = nc0.variables["time"][-1]
                self.last = int((t_last - self.t0) / self.dt)
                nclast.close()
            else:
                self.last = self.date2tindex(self.last_date)

            self.length = self.last - self.first + 1 - (self.sample_len - 1)

            # get last timestep index in each file
            # do this in a loop to avoid many-many open filehandles at once

            self.ends = []
            cumlen = -1
            for f in self.filepaths:
                ncf = nc.Dataset(f, mask_and_scale=False)
                cumlen = cumlen + len(ncf.variables["time"])
                self.ends.append(cumlen)
                ncf.close()
                # file opens are slow; stop early if possible
                if self.last_date is not None and cumlen > self.last:
                    break

            if self.last_date is None:
                self.last = self.ends[-1]

    # end of __post_init__

    def date2tindex(self, datestring):
        """Convert datestring (in ISO8601 YYYY-MM-DD format) to
        internal time index.  Datestring can optionally also have an
        HH:MM:SS component; if absent, it defaults to 00:00:00.
        Returns 0 if dataset is static."""
        if self.dim == "static":
            return 0
        # todo: check that string matches expected format
        bits = datestring.split()
        if len(bits) == 1:
            bits.append("00:00:00")
        year, mon, day = [int(x) for x in bits[0].split("-")]
        hour, min, sec = [int(x) for x in bits[1].split(":")]
        cfdt = cf.datetime(year, mon, day, hour, min, sec, calendar=self.calendar)
        time = cf.date2num(cfdt, self.units, self.calendar)
        tindex = int((time - self.t0) / self.dt)
        return tindex

    def sindex2dates(self, sindex):
        """Returns dates associated with sample index as a dict
        containing time coordinates, units, calendar, and ISO8601
        dates from the cftime library.  Returns None if dataset is
        static.

        """
        if self.dim == "static":
            return None
        dates = {"calendar": self.calendar, "units": self.units}
        tindexes = [sindex + self.first + i for i in range(self.sample_len)]
        timecoords = [self.t0 + t * self.dt for t in tindexes]
        dates["time"] = timecoords
        cfdates = [str(cf.num2date(t, self.units, self.calendar)) for t in timecoords]
        dates["cf_datetimes"] = cfdates
        return dates

    def __len__(self):
        if self.dim == "static":
            return 1
        return self.length

    def __getitem__(self, index):
        if self.dim == "static":
            return {"boundary": self.data}

        if index < 0 or index > self.length - 1:
            raise IndexError()

        start = index + self.first + 1
        if self.mode == "train":
            finish = start + self.sample_len - 1
        else:
            finish = start + self.history_len - 1

        # get segment (which file) and subindex (within file) for start & finish.
        # subindexes are all negative, but that works fine & makes math simpler

        startseg = np.searchsorted(self.ends, start)
        finishseg = np.searchsorted(self.ends, finish)
        startsub = start - (self.ends[startseg] + 1)
        finishsub = finish - (self.ends[finishseg])
        if finishsub == 0:
            finishsub = None  # needed to get the last element in the array
            # x[-1:0] gives you an empty list

        if startseg == finishseg:
            result = self.read(startseg, startsub, finishsub)
        else:
            data1 = self.read(startseg, startsub, None)
            data2 = self.read(finishseg, None, finishsub)
            result = {}
            for use in data1:
                result[use] = {}
                for var in data1[use]:
                    a1 = data1[use][var]
                    a2 = data2[use][var]
                    result[use][var] = np.concatenate((a1, a2))

        return result

    # the mode property determines which variables to return by use type:
    # "train" = all; "init" = all but diagnostic; "infer" = static + boundary

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if mode not in ("train", "init", "infer"):
            raise ValueError("invalid DataMap mode")
        self._mode = mode

    def read(self, segment, start, finish):
        """open file & read data from start to finish for needed variables"""

        # Note: static DataMaps never call read; they short-circuit in getitem
        uses = ()
        match self.mode:
            case "train":
                uses = ("boundary", "prognostic", "diagnostic")
            case "init":
                uses = ("boundary", "prognostic")
            case "infer":
                uses = ("boundary",)
            case _:
                raise ValueError("invalid DataMap mode")

        ds = nc.Dataset(self.filepaths[segment], mask_and_scale=False)
        data = {}
        for use in uses:
            data[use] = {}
            for var in self.variables[use]:
                if self.dim == "3D" and self.zstride != 1:
                    data[use][var] = np.array(
                        ds[var][start:finish, :: self.zstride, ...]
                    )
                else:
                    data[use][var] = np.array(ds[var][start:finish, ...])
        ds.close()
        return data
