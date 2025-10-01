import os
import inspect
import logging

from collections import defaultdict
from dataclasses import dataclass
import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)

# pylint: disable=no-else-return

"""
transforms_downscaling.py
-------------------------------------------------------
"""


# Expand array by repeating x & y elements; equivalent to
# nearest-neighbor interpolation of coarse data for simplified
# single-funnel downscaling models

@dataclass
class Expand:
    by: int

    def __call__(self, x, inverse=False):
        if inverse:
            return x[..., ::self.by, ::self.by]
        else:
            n = len(x.shape)
            return x.repeat(self.by, axis=n-1).repeat(self.by, axis=n-2)


@dataclass
class Pad:
    left:   int = 0
    right:  int = 0
    top:    int = 0
    bottom: int = 0
    mode:   str = "edge"

    def __call__(self, x, inverse=False):
        if inverse:
            nx = x.shape[-1]
            ny = x.shape[-2]
            return x[..., self.bottom:ny-self.top, self.left:nx-self.right]
        else:
            pad = ((self.bottom, self.top), (self.left, self.right))
            for _ in range(2, len(x.shape)):
                pad = ((0, 0), ) + pad   # no padding on other dimensions
            return np.pad(x, pad_width=pad, mode=self.mode)


# Note: we don't want to use sklearn functions for normalization
# because they calcluate params from the data, and we want to use
# externally- specified param values.  We also need an inverse for
# each function.

def rescale(x, offset=0, scale=1, inverse=False):
    if inverse:
        return (x * scale) + offset
    else:
        return (x - offset) / scale


@dataclass
class Minmax:
    mmin: float
    mmax: float

    def __call__(self, x, inverse=False):
        return rescale(x, self.mmin, self.mmax-self.mmin, inverse)


@dataclass
class Zscore:
    mean: float = 0
    stdev: float = 1

    def __call__(self, x, inverse=False):
        return rescale(x, self.mean, self.stdev, inverse)


@dataclass
class Power:
    exponent: float

    def __call__(self, x, inverse=False):
        if inverse:
            return np.power(x, 1/self.exponent)
        else:
            return np.power(x, self.exponent)


# Clipping limits data values to the range [cmin, cmax].  Inverse for
# clipping is the same as forward.  (If I didn't want precip < 0 on
# input, I also don't want it on output.)

@dataclass
class Clip:
    cmin: float = None
    cmax: float = None

    def __call__(self, x, inverse=False):
        return np.clip(x, a_min=self.cmin, a_max=self.cmax)


# the 'do-nothing' op
@dataclass
class Identity:
    def __call__(self, x, inverse=False, **kwargs):
        return x


# Note: if inverse proves to be more convenient as a state switch
# rather than an argument, add inverse=False to init args, drop from
# call and branch on self.inverse

class DataTransforms:
    '''
    [insert more documentation here]
    '''
    xclassdict = {"expand": Expand,
                  "minmax": Minmax,
                  "zscore": Zscore,
                  "power": Power,
                  "clip": Clip,
                  "pad": Pad,
                  "none": Identity,
                  }

    def __init__(self, vardict, transdict, rootpath, dim, zstride=1):

        # TODO: something if dim == 'static'

        if zstride != 1 and dim != '3D':
            raise ValueError("credit.transforms: zstride > 1 only allowed for dim=='3D'")

        # get flat list of variables
        variables = []
        for usage in vardict:
            if usage != 'unused':
                variables.extend(vardict[usage])

        # get parameters used by each transform (for transforms used)
        xformparams = {}
        for var in transdict:
            if var != "paramfiles" and transdict[var] != 'none':
                for xform in transdict[var]:
                    x = self.xclassdict[xform]
                    xformparams[xform] = list(inspect.signature(x).parameters.keys())

        # read in any parameter values stored in netcdf files
        if 'paramfiles' in transdict:

            fileparams = defaultdict(dict)

            for par in transdict['paramfiles']:
                ppath = os.path.join(rootpath, transdict['paramfiles'][par])
                pfile = nc.Dataset(ppath, mask_and_scale=False)
                for var in variables:
                    if var in pfile.variables:
                        v = pfile.variables[var]
                        if dim == '3D' and zstride != 1:
                            fileparams[var][par] = np.array(v[:, ::zstride, ...])
                        else:
                            fileparams[var][par] = np.array(v[...])

        # instantiate list of transforms for each variable
        self.transforms = {}
        for var in variables:
            self.transforms[var] = []
            if var in transdict or 'default' in transdict:
                xkey = var if var in transdict else 'default'
                if transdict[xkey] == "none":
                    self.transforms[var].append(Identity())
                else:
                    for xform in transdict[xkey]:
                        x = self.xclassdict[xform]
                        xargs = transdict[xkey][xform]
                        if xargs == 'paramfile':
                            xargs = {par: fileparams[var][par] for par in xformparams[xform]}
                        self.transforms[var].append(x(**xargs))
            else:
                # no tranform defined & no default for this variable
                self.transforms[var].append(Identity())

    def __call__(self, x, inverse=False):
        # x is a nested dict of [usage][var][time,(z),y,x]
        for usage in x:
            for var in x[usage]:
                if inverse:
                    for xform in reversed(self.transforms[var]):
                        x[usage][var] = xform(x[usage][var], inverse=True)
                else:
                    for xform in self.transforms[var]:
                        x[usage][var] = xform(x[usage][var], inverse=False)
        return x
