import numpy as np

"""
transforms_downscaling.py
-------------------------------------------------------
"""



## Expand array by repeating x & y elements; equivalent to
## nearest-neighbor interpolation of coarse data for simplified
## single-funnel downscaling models

def expand(x, by, inverse=False):
        if inverse:
            return x[...,::by,::by]
        else:
            n = len(x.shape)
            return x.repeat(by, axis=n-1).repeat(by, axis=n-2)


## Note: we don't want sklearn functions becaue they calcluate params
## from the data, and we want to use externally-specified param
## values.  We also need an inverse for each function.

def rescale(x, offset=0, scale=1, inverse=False):
    if inverse:
        return (x * scale) + offset
    else:
        return (x - offset) / scale

def minmax(x, lower, upper, inverse=False):
    return rescale(x, lower, upper-lower, inverse)

def zscore(x, mean, stdev, inverse=False):
    return rescale(x, mean, stdev, inverse)


## this needs a wrapper because numpy.power only takes positional
## args, plus we need the inverse option

def power(x, pow, inverse=False):
    if inverse:
        return np.power(x, 1/pow)
    else:
        return np.power(x, pow)

    
## wrapped to handle inverse, which is the same as forward.  (If I
## didn't want precip < 0 on input, I also don't want it on output.)

def clip(x, min=None, max=None, inverse=False):
    return np.clip(x, min, max)


## the 'do-nothing' op
def identity(x, inverse=False, **kwargs):
    return x

## map names to functions in dict to avoid insecure exec()
fundict = { "expand": expand,
            "rescale": rescale,
            "minmax": minmax,
            "zscore": zscore,
            "power": power,
            "clip": clip,
            None: identity,  # do nothing if no transform defined
           }


@dataclass
class DataTransform:  ## different name, since this is internal-use
    fname: str
    params: dict
    inverse: bool=False

    def __call__(x, **kwargs):
        return fundict[self.fname](x, inverse=self.inverse, **self.params)


##  Need a class that reads netcdf & instantiates all datatransforms
##  based on conf; then, given a dict of ndrrays, it applies all the
##  transforms to the relevant variables.  Invertible


## should inversion be an argument, or a switch that you flip?

class DownscalingNormalizer:
    ''' pass **conf['data'] as arguments to constructor                                   
    [insert more documentation here]
    '''
    def __init__(self, conf):
        self.config = conf
        # get params from config
        # set up organizational structure
        # read data from netcdf
        # creat transform objects
    
    def __call__(self, x):
        # iterate over sample structure
        # call transforms for each variabe
        return x


