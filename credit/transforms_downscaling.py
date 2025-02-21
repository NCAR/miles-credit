import os
import inspect
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import ClassVar
import netCDF4 as nc
import numpy as np

"""
transforms_downscaling.py
-------------------------------------------------------
"""



## Expand array by repeating x & y elements; equivalent to
## nearest-neighbor interpolation of coarse data for simplified
## single-funnel downscaling models

@dataclass
class Expand:
    by: int
    inverse: bool=False
    def __call__(self, x):
        if self.inverse:
            return x[..., ::self.by, ::self.by]
        else:
            n = len(x.shape)
            return x.repeat(selfby, axis=n-1).repeat(self,by, axis=n-2)
        

## Note: we don't want sklearn functions for normalization because
## they calcluate params from the data, and we want to use
## externally-specified param values.  We also need an inverse for
## each function.

def rescale(x, offset=0, scale=1, inverse=False):
    if inverse:
        return (x * scale) + offset
    else:
        return (x - offset) / scale

@dataclass
class Minmax:
    mmin: float
    mmax: float
    inverse: bool = False

    def __call__(self, x):
        return rescale(x, self.mmin, self.mmax-self.mmin, self.inverse)

@dataclass
class Zscore:
    mean: float = 0
    stdev: float = 1
    inverse: bool = False

    def __call(self, x):
        return rescale(x, self.mean, self.stdev, self.inverse)

## TODO: rename argument 'pow' - nameclash with function
@dataclass
class Power:
    pow: float
    inverse: bool=False
    
    def __call__(self, x):
        if self.inverse:
            return np.power(x, 1/self.pow)
        else:
            return np.power(x, self.pow)

    
## Inverse for clipping is the same as forward.  (If I didn't want
## precip < 0 on input, I also don't want it on output.)

@dataclass
class Clip:
    cmin: float=None
    cmax: float=None
    inverse: bool=False
    def __call__(self, x):
        return np.clip(x, a_min=self.cmin, a_max=self.cmax)

    
## the 'do-nothing' op
@dataclass
class Identity:
    inverse: bool=False
    def __call__(self, x):
        return x

#..# @Dataclass
#..# class Composition:
#..#     transforms: #list of callables
#..#     inverse: bool=False
#..# 
#..#     def __call__(self, x):
#..#         if self.inverse:
#..#             # call trasnforms in reverse order            
#..#             pass
#..#         else:
#..#             # call trasnforms on x in order
#..#             pass
#..# 

## inversion needs to be a switch that you flip

## Works for 2D!
## Need to make it work for 3D when pfile.variables[var][...].item() is a list


class DownscalingNormalizer:
    '''
    [insert more documentation here]
    '''
    xclassdict = {"expand": Expand,
                  "minmax": Minmax,
                  "zscore": Zscore,
                  "power": Power,
                  "clip": Clip,
                  None: Identity,  # do nothing if no transform defined
                  }
    
    def __init__(self, vardict, transdict, rootpath, inverse=False):
        ## TODO: make this take **kwargs instead so we can pass **conf?
        
        ## TODO: make this a setter property so it can flip all the transforms
        self.inverse = inverse

        ## flatten to get list of variables
        variables = [var for vlist in vardict.values() for var in vlist]
        
        ## get all parameter values stored in netcdf paramfiles
        params = defaultdict(dict)
        if 'paramfiles' in transdict:
            for par in transdict['paramfiles']:
                params[par] = {}
                pfile = nc.Dataset(rootpath + "/" + transdict['paramfiles'][par])
                for var in variables:
                    if var in pfile.variables:
                        params[par][var] = pfile.variables[var][...]  #.item()

        ## add explicitly specified parameter values, possibly overriding netcdf
        for var in variables:
            if var in transdict:
                tvdict = transdict[var]
                for xform in tvdict:
                    if tvdict[xform] != "paramfile":
                        for arg in tvdict[xform]:
                            params[arg][var] = tvdict[xform][arg]
                            
        # instantiate transforms for each var
        self.transforms = defaultdict(list)

        for var in variables:
            if var in transdict or 'default' in transdict:
                xkey = var if var in transdict else 'default'
                for xform in transdict[xkey]:
                    x = self.xclassdict[xform]
                    xparams = {}
                    for arg in inspect.signature(x).parameters.keys():
                        if var in params[arg]:
                            xparams[arg] = params[arg][var]                    
                    # instantiate new transform & append to list                    
                    self.transforms[var].append(x(**xparams))
            else:
                self.transforms[var].append(self.xclassdict[None]())
            
    
    def __call__(self, x):
        # x is a nested dict of [usage][var][time,(z),y,x]
        for usage in x:
            for var in x[usage]:
                
                pass
        # iterate over sample structure
        # call self.transforms[var] for each variable
        # (it's a list; iterate over it)
        # if self.inverse, iterate in reverse order
        # if variable not in list of transforms, do nothing to it
        return x


