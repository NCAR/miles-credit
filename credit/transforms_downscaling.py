"""
transforms_downscaling.py
-------------------------------------------------------
"""


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
        return numpy.power(x, 1/pow)
    else:
        return numpy.power(x, pow)

## the 'do-nothing' op
def identity(x, **kwargs):
    return x
    

fundict = { "rescale": rescale,
            "minmax": minmax,
            "zscore": zscore,
            "power": power,
            "clip": numpy.clip,
            "none": identity,
           }

# For the "clip" operation, the arguments are a_min and a_max.  You
# can define either or both.  It takes **kwargs, so you can provide
# extra arguments that it will ignore, like 'inverse'.  This is the
# right thing to do: inverse clip is the same as forward clip, because
# if I didn't want precip < 0 on input, I don't want precip < 0 on
# output, either.


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


