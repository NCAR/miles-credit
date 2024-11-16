'''datamap.py
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

'''

import os
from typing import List
from dataclasses import dataclass, field
from glob import glob
from warnings import warn
import netCDF4 as nc
import cftime as cf
import numpy as np

def rescale_minmax(x):
    '''rescale data to [0,1].  Don't use
    sklearn.preprocessing.minmax_scale() because it requires reshaping
    the data, which is silly for a use case this simple.

    '''
    x = x - np.min(x)
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
# length of this dataset is 362 samples.

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

#  i  | file | fidx | time | mon | day | split | sample | index
#--------------------------------------------------------------
#   1 |  0   |  0   |      | Dec |  1  |       |        |  -     
#   2 |  0   |  1   |      | '00 |  2  |       |        |  -     
#   3 |  0   |  2   |      |  "  |  3  |       |        |  -     
# ... |  "   | ...  |      |  "  | ... |       |        |       
#  21 |  0   |      |      |  "  |  21 |       | (1st)  |  -     
#  22 |  0   |      |      |  "  |  22 | first | hist0  |  -     
#  23 |  0   |      |      |  "  |  23 |       | hist1  |  -     
#  24 |  0   |      |      |  "  |  24 |       | hist2  |  -     
#  25 |  0   |      |      |  "  |  25 |       | fore0  |  0    
#  26 |  0   |      |      |  "  |  26 |       | fore1  |  1    
# ... |  "   | ...  | ...  |  "  | ... |       |        | ...      
#  29 |  0   |  28  | -72  |  "  |  29 |       | ( #8 ) |  4     
#  30 |  0   |  29  | -48  |  "  |  30 |       | hist0  |  5     
#  31 |  0   |  30  | -24  |  "  |  31 |       | hist1  |  6     
#--------------------------------------------------------------
#  32 |  1   |  0   |  0   | Jan |  1  |       | hist2  |  7     
#  33 |  1   |  1   |  24  | '01 |  2  |       | fore0  |  8     
#  34 |  1   |  2   |  48  |  "  |  3  |       | fore1  |  9     
# ... |  1   | ...  | ...  |  "  | ... |       |        | ...      
#  61 |  1   |  29  | 719  | Jan |  30 |       |        |       
#  62 |  1   |  30  | 720  | Jan |  31 |       |        |       
#--------------------------------------------------------------
#  63 |  2   |  0   | 744  | Feb |  1  |       |        |       
#  64 |  2   |  1   | 768  |  "  |  2  |       |        |       
# ... |  2   | ...  | ...  |  "  | ... |       |        | ...      
#--------------------------------------------------------------
# ...   ...    ...    ...    ...   ...                    ...      
#--------------------------------------------------------------
# ... |  11  | ...  | ...  | Nov | ... |       |        | ...   
# 364 |  11  |  28  |      |  "  |  29 |       |        |       
# 365 |  11  |  29  |      | Nov |  30 |       |        |       
#--------------------------------------------------------------
# 366 |  12  |  0   |      | Dec |  1  |       |        |       
# ... |  12  | ...  |      | '01 | ... |       | (last) | ...   
# 382 |  12  |  16  |      |  "  |  17 |       | hist0  | 358   
# 383 |  12  |  17  |      |  "  |  18 |       | hist1  | 359   
# 384 |  12  |  18  |      |  "  |  19 |       | hist2  | 360   
# 385 |  12  |  19  |      |  "  |  20 |       | fore0  | 361   
# 386 |  12  |  20  |      |  "  |  21 | last  | fore1  |  -    
# 387 |  12  |  21  |      |  "  |  22 |       |        |  -    
# ... |  "   | ...  | ...  |  "  | ... |       |        |       
# 395 |  12  |  29  |      |  "  |  30 |       |        |  -    
# 396 |  12  |  30  |      | Dec |  31 |       |        |  -    
#--------------------------------------------------------------


# Written as a dataclass to avoid acres of boilerplate and for free repr(), etc.

@dataclass
class DataMap:
    '''Class for reading in data from multiple files.

    rootpath: pathway to the files
    glob: filename glob of netcdf files
    label: used by higher-level classes to decide how to use the datamap
    dim: dimensions of the data:
        static: no time dimension; data is loaded on initialization
        3D: data has z-dimension; unstack Z to pseudo-variables when reading
        2D: default: time-varying 2D data
    normalize: if dim=='static' & normalize == True, scale data to range [0,1]
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
    '''
    rootpath:     str
    glob:         str
    label:        str = None
    dim:          str = "2D"
    normalize:    bool = False
    boundary:     List[str] = field(default_factory=list)
    prognostic:   List[str] = field(default_factory=list)
    diagnostic:   List[str] = field(default_factory=list)
    unused:       List[str] = field(default_factory=list)
    history_len:  int = 2
    forecast_len: int = 1
    first_date:   str = None
    last_date:    str = None
    
    def __post_init__(self):
        super().__init__()

        self.sample_len = self.history_len + self.forecast_len
        
        ## todo: accept & canonicalize different capitalization
        if self.dim not in ['static', '2D', '3D']:
            warn(f"credit.datamap: unknown dimensionality: {self.dim}; assuming 2D")
            self.dim = "2D"
            
        if self.normalize and self.dim != "static":
            warn(f"credit.datamap: normalize does nothing if dim != 'static'; setting to False")
            self.normalize = False
            
        if self.dim == "static":
            if len(glob(self.glob, root_dir=self.rootpath)) != 1:
                warn("credit.datamap: dim='static' requires a single file")
                ## TODO (someday): support multiple static files
                raise
            if len(self.prognostic) > 0 or len(self.diagnostic) > 0:
                warn("credit.datamap: static vars must be boundary")
                raise
            
            ## load data from netcdf
            staticfile = nc.Dataset(self.rootpath + '/' + self.glob)
            ## [:] forces data to load
            staticdata = [staticfile[v][:] for v in self.boundary]
            self.data = dict(zip(self.boundary, staticdata))
            ## cleanup
            staticfile.close()
            del staticfile, staticdata
            
            if self.normalize:
                for k in self.data.keys():
                    self.data[k] = rescale_minmax(self.data[k])
                    
        else:
            fileglob = sorted(glob(self.glob, root_dir=self.rootpath))
            self.filepaths = [os.path.join(self.rootpath, f) for f in fileglob]

            ## get time coordinate characteristics from first file
            ## calendar & units used to convert date <=> time coordinate
            ## t0 & dt used to convert time coordinates <=> timestep index
            nc0 = nc.Dataset(self.filepaths[0])
            time0 = nc0.variables["time"]
            
            self.calendar = time0.calendar
            self.units = time0.units
            self.t0 = float(time0[0])
            self.dt = float(time0[1]) - self.t0

            nc0.close()

            ## get last timestep index in each file
            ## do this in a loop to avoid many-many open filehandles at once
            file_lens = list()
            for f in self.filepaths:
                ncf = nc.Dataset(f)
                file_lens.append(len(ncf.variables["time"]))
                ncf.close()

            ## Do I need to subtract 1 from all of these?
            self.ends = list(np.cumsum(file_lens))

            if(self.first_date is None):
                self.first = 0
            else:
                self.first = self.timecoord2tindex(
                    self.date2timecoord(
                        self.first_date))

            if(self.last_date is None):
                self.last = self.ends[-1]
            else:
                self.last = self.timecoord2tindex(
                    self.date2timecoord(
                        self.last_date))
                
    # end of __post_init__   
                
    def date2timecoord(self, datestring):
        """Convert YYYY-MM-DD string to dataset time coordinate"""
        ## assert: datestring = YYYY-MM-DD [HH:MM:SS]
        bits = datestring.split()
        if(len(bits) == 1): bits.append("00:00:00")
        year, mon, day = [int(x) for x in bits[0].split("-")]
        hour, min, sec = [int(x) for x in bits[1].split(":")]
        cfdt = cf.datetime(year, mon, day, hour, min, sec, calendar=self.calendar)
        timecoord = cf.date2num(cfdt, self.units, self.calendar)
        return timecoord

    def timecoord2tindex(self, time):
        """Convert time coordinate value to timestep index"""
        return int((time - self.t0)/self.dt)

    def tindex2sindex(self, tindex):
        """Convert timestep index to sample index"""
        return tindex - self.first

    def sindex2tindex(self, sindex):
        """Convert sample index to timestep index"""
        return sindex + self.first

    
    def __len__(self):
       if self.dim == "static":
           return 1
       else:
           return self.last - self.first - self.sample_len + 1

    def __getitem__(self, index):
        if self.static:
            return self.data

        else:

            t = sindex2tindex(index)
            start = t - self.hist_len
            end = t + self.fore_len - 1
            segment = np.searchsorted(self.ends, start)
            subindex = index - starts[segment]  ## ... HERE

        ## get segment & subindex for start & end

        ## if startseg != endseg, sample spans file boundary
        ## sample startseg from startsubindex to end of file
        ## and endseg from beginning of file to endsubindex

        ## add method read so we can loop it easily?
        
        ## open filepaths[segment]
        ## read data from that segment

        ## mode: training -- get all variables, organize into subdicts by use
        ##       inference: only get boundary vars
        ##       initialize: get boundary & prognostic vars

        ## result: dictionary of key = varname, value = array
        
        ## return result
        pass

    ## normalization (except for static) & structural transformation
    ## (unstacking z, concatenate to tensor) happen in parent class.
    ## Datamap just gets you data from a file.

    ## need to deal with input & target slicing
    
