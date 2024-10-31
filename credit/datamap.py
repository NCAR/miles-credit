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

Content:

'''

import os
from dataclasses import dataclass, field
from glob import glob
from warnings import warn
import netCDF4 as nc


def rescale_minmax(x):
    '''rescale data to [0,1].  Don't use
    sklearn.preprocessing.minmax_scale() because it requires reshaping
    the data, which is silly for a use case this simple.

    '''
    x = x - np.min(x)
    x = x / np.max(x)
    return x


@dataclass
class DataMap:
    ''' Class for reading in data from multiple files.

    rootpath: pathway to the files
    glob: filename glob of netcdf files
    label: used by higher-level classes to decide how to use the datamap
    dim: dimensions of the data:
        static: no time dimension; data is loaded on initialization
        3D: data has z-dimension; unstack Z to pseudo-variables when reading
        2D: default: time-varying 2D data
    normalize: if dim=='static' & normalize == True, scale data to range [0,1]
    boundary, prognostic, diagnostic, unused: lists of variable names in files
    '''
    rootpath:    str
    glob:        str
    history_len  int = 2
    forecast_len int = 1
    label:       str = None
    dim:         str = "2d"
    normalize:   bool = False
    boundary:    List[str] = field(default_factory=list)
    prognostic:  List[str] = field(default_factory=list)
    diagnostic:  List[str] = field(default_factory=list)
    unused:      List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__init__()

        self.sample_len = self.history_len + self.forecast_len
        
        ## todo: accept & canonicalize different capitalization
        if self.dim not in ['static', '2D', '3D']:
            warn(f"credit.datamap: unknown dimensionality: {self.dim}")

        if self.normalize and self.dim != "static":
            warn(f"credit.datamap: normalize does nothing if dim != 'static'")
            
        if self.static():
            if len(glob(self.glob, self.rootpath)) != 1:
                warn("credit.datamap: dim='static' requires a single file")
                ## TODO (someday): support multiple static files
                raise
            if len(prognostic) > 0 or len(diagnostic) > 0:
                warn("credit.datamap: static vars must be boundary, not prognostic or diagnostic")
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
                    self.data[k] = minmax_scale(self.data[k])
                    
        else:
            self.filepaths = sorted(glob(self.glob, root_dir=self.rootpath))
            ## open first file
            ## get datetime of index 0
            ## get deltaT between t0 and t1
            ## get length of all files; construct starts array
            ## get total length of dataset, minus sample padding

    ## total effective length (length - sample_length + 1
    def __len__(self):
        if self.static:
            return 1
        else:
            pass

    def __getitem__(self, index):
        if self.static:
            return self.data

        else:
        
        segment = np.searchsorted(self.ends, index)
        subindex = index - starts[segment]

        ## get segment & subindex for start & end

        ## if startseg != endseg, sample spans file boundary
        ## sample startseg from startsubindex to end of file
        ## and endseg from beginning of file to endsubindex

        ## add method read so we can loop it easily?
        
        ## open filepaths[segment]
        ## read data from that segment

        ## result: dictionary of key = varname, value = array
        
        ## return result
        pass

    ## normalization (except for static) & structural transformation
    ## (unstacking z, concatenate to tensor) happen in parent class.
    ## Datamap just gets you data from a file.

    ## need to deal with input & target slicing
    
