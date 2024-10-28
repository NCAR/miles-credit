from typing import Optional, Callable, TypedDict, Union, List
import os
from dataclasses import dataclass, field
from functools import reduce
from glob import glob
from itertools import repeat
from timeit import timeit
import numpy as np
import xarray as xr
import torch
import torch.utils.data


def get_forward_data(filename) -> xr.DataArray:
    """Lazily opens a Zarr store
    """
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.

    x = input (predictor) data (i.e, C404dataset[historical mask]
    y = target (predictand) data (i.e, C404dataset[forecast mask]

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    x: Array
    y: Array


def flatten(array):

    """ flattens a list-of-lists
    """
    return reduce(lambda a, b: a+b, array)


def lazymerge(zlist, rename=None):
    """ merges zarr stores opened lazily with get_forward_data()
    """
    zarrs = [get_forward_data(z) for z in zlist]
    if rename is not None:
        oldname = flatten([list(z.keys()) for z in zarrs])
        # ^^ this will break on multi-var zarr stores
        zarrs = [z.rename_vars({old: new}) for z, old, new in zip(zarrs, oldname, rename)]
    return xr.merge(zarrs)


# using dataclass decorator avoids lots of self.x=x and gets us free __repr__
@dataclass
class CONUS404Dataset(torch.utils.data.Dataset):
    """Each Zarr store for the CONUS-404 data contains one year of
    hourly data for one variable.

    When we're sampling data, we only want to load from a single zarr
    store; we don't want the samples to span zarr store boundaries.
    This lets us leave years out from across the entire span for
    validation during training.

    To do this, we segment the dataset by year.  We figure out how
    many samples we could have, then subtract all the ones that start
    in one year and end in another (or end past the end of the
    dataset).  Then we create an index of which segment each sample
    belongs to, and the number of that sample within the segment.

    Then, for the __getitem__ method, we look up which segment the
    sample is in and its numbering within the segment, then open the
    corresponding zarr store and read only the data we want with an
    isel() call.

    For multiple variables, we necessariy end up reading from multiple
    stores, but they're merged into a single xarray Dataset, so
    hopefully that won't cause a big performance hit.

    """

    zarrpath:     str = "/glade/campaign/ral/risc/DATA/conus404/zarr"
    varnames:     List[str] = field(default_factory=list)
    history_len:  int = 2
    forecast_len: int = 1
    transform:    Optional[Callable] = None
    seed:         int = 22
    skip_periods: int = None
    one_shot:     bool = False
    start:        str = None
    finish:       str = None

    def __post_init__(self):
        super().__init__()

        self.sample_len = self.history_len + self.forecast_len
        self.stride = 1 if self.skip_periods is None else self.skip_periods + 1

        self.rng = np.random.default_rng(self.seed)

        # CONUS404 data is organized into directories by variable,
        # with a set of yearly zarr stores for each variable
        if len(self.varnames) == 0:
            self.varnames = os.listdir(self.zarrpath)
        self.varnames = sorted(self.varnames)

        # get file paths
        zdict = {}
        for v in self.varnames:
            zdict[v] = sorted(glob(os.path.join(self.zarrpath, v, v+".*.zarr")))

        # check that lists align
        zlen = [len(z) for z in zdict.values()]
        assert all(zlen[i] == zlen[0] for i in range(len(zlen)))

        # transpose list-of-lists; sort by key to ensure var order constant
        zlol = list(zip(*sorted(zdict.values())))

        # lazy-load & merge zarr stores
        self.zarrs = [lazymerge(z, self.varnames) for z in zlol]

        # Name of time dimension may vary by dataset.  ERA5 is "time"
        # but C404 is "Time".  If dataset is CF-compliant, we
        # can/should look for a coordinate with the attribute
        # 'axis="T"', but C404 isn't CF, so instead we look for a dim
        # named "time" (any capitalization), and defer checking the
        # axis attribute until it actually comes up in practice.
        dnames = list(self.zarrs[0].dims)
        self.tdimname = dnames[[d.lower() for d in dnames].index("time")]

        # subset zarrs to constrain data to period defined by start
        # and finish.  Note that xarray subsetting includes finish.
        start, finish = self.start, self.finish
        if start is not None or finish is not None:
            if start is None:
                start = self.zarrs[0].coords[self.tdimname][0]
            if finish is None:
                start = self.zarrs[-1].coords[self.tdimname][-1]

            selection = {self.tdimname: slice(start, finish)}
            self.zarrs = [z.sel(selection) for z in self.zarrs]

        # construct indexing arrays
        zarrlen = [z.sizes[self.tdimname] for z in self.zarrs]
        whichseg = [list(repeat(s, z)) for s, z in zip(range(len(zarrlen)), zarrlen)]
        segindex = [list(range(z)) for z in zarrlen]

        # subset to samples that don't overlap a segment boundary
        # (sample size N = can't use last N-1 samples)
        N = self.sample_len - 1
        self.segments = flatten([s[:-N] for s in whichseg])
        self.zindex = flatten([i[:-N] for i in segindex])

        # precompute mask arrays for subsetting data for samples
        self.histmask = list(range(0, self.history_len, self.stride))
        foreind = list(range(self.sample_len))
        if self.one_shot:
            self.foremask = foreind[-1]
        else:
            self.foremask = foreind[slice(self.history_len, self.sample_len, self.stride)]

    def __len__(self):
        return len(self.zindex)

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index, do_transform=True):
        """like gets an element by index (as __getitem__ does), but
        with an optional argument to skip applying the normalization
        transform.
        """
        # Possible refactor: instead of an optional argument, make
        # do_transform a class attribute (i.e., a switch that can be
        # flipped) and push this all back to __getitem__

        time = self.tdimname
        first = self.zindex[index]
        last = first + self.sample_len
        seg = self.segments[index]
        subset = self.zarrs[seg].isel({time: slice(first, last)}).load()
        sample = Sample(
            x=subset.isel({time: self.histmask}),
            y=subset.isel({time: self.foremask}))

        if do_transform:
            if self.transform:
                sample = self.transform(sample)

        return sample


#    def get_time(self, index):
#        """ get time coordinate(s) associated with index
#        """
#        time = self.tdimname
#        first = self.zindex[index]
#        last = first + self.sample_len
#        seg = self.segments[index]
#        subset = self.zarrs[seg].isel({time: slice(first, last)}).load()
#        result = {}
#        result["x"] =subset.isel({time: self.histmask})
#        result["y"] =subset.isel({time: self.foremask})
#        return result


def testC4loader():
    """Test load speed of different number of vars & storage locs.
    Full load for C404 takes about 4 sec on campaign, 5 sec on scratch

    """
    zdirs = {
        "worktest": "/glade/work/mcginnis/ML/GWC/testdata/zarr",
        "scratch": "/glade/derecho/scratch/mcginnis/conus404/zarr",
        "campaign": "/glade/campaign/ral/risc/DATA/conus404/zarr"
        }
    for zk in zdirs.keys():
        src = zdirs[zk]
        print("######## "+zk+" ########")
        svars = os.listdir(src)
        for i in range(1, len(svars)+1):
            testvars = svars[slice(0, i)]
            print(testvars)
            cmd = 'c4 = CONUS404Dataset("'+src+'",varnames='+str(testvars)+')'
            print(cmd+"\t"+str(timeit(cmd, globals=globals(), number=1)))
