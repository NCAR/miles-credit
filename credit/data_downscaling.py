# system tools
import os
from copy import deepcopy
from typing import Dict #, TypedDict, Union
from dataclasses import dataclass, field

# data utils
import numpy as np
import xarray as xr

# Pytorch utils
import torch
import torch.utils.data
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler

from credit.datamap import DataMap

#Array = Union[np.ndarray, xr.DataArray]
#IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')
# 
# class Sample(TypedDict):
#     """Simple class for structuring data for the ML model.
# 
#     x = predictor (input) data
#     y = predictand (target) data
# 
#     Using typing.TypedDict gives us several advantages:
#       1. Single 'source of truth' for the type and documentation of each example.
#       2. A static type checker can check the types are correct.
# 
#     Instead of TypedDict, we could use typing.NamedTuple,
#     which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
#     so we cannot change the values in the transforms.
#     """
#     x: Array
#     y: Array


# # using dataclass decorator avoids lots of self.x=x and gets us free __repr__
# @dataclass
# class CONUS404Dataset(torch.utils.data.Dataset):
#     """Each Zarr store for the CONUS-404 data contains one year of
#     hourly data for one variable.
# 
#     When we're sampling data, we only want to load from a single zarr
#     store; we don't want the samples to span zarr store boundaries.
#     This lets us leave years out from across the entire span for
#     validation during training.
# 
#     To do this, we segment the dataset by year.  We figure out how
#     many samples we could have, then subtract all the ones that start
#     in one year and end in another (or end past the end of the
#     dataset).  Then we create an index of which segment each sample
#     belongs to, and the number of that sample within the segment.
# 
#     Then, for the __getitem__ method, we look up which segment the
#     sample is in and its numbering within the segment, then open the
#     corresponding zarr store and read only the data we want with an
#     isel() call.
# 
#     For multiple variables, we necessariy end up reading from multiple
#     stores, but they're merged into a single xarray Dataset, so
#     hopefully that won't cause a big performance hit.
# 
#     """
# 
#     zarrpath:     str = "/glade/campaign/ral/risc/DATA/conus404/zarr"
#     varnames:     List[str] = field(default_factory=list)
#     history_len:  int = 2
#     forecast_len: int = 1
#     transform:    Optional[Callable] = None
#     seed:         int = 22
#     skip_periods: int = None
#     one_shot:     bool = False
#     start:        str = None
#     finish:       str = None
# 
#     def __post_init__(self):
#         super().__init__()
# 
#         self.sample_len = self.history_len + self.forecast_len
#         self.stride = 1 if self.skip_periods is None else self.skip_periods + 1
# 
#         self.rng = np.random.default_rng(self.seed)
# 
#         # CONUS404 data is organized into directories by variable,
#         # with a set of yearly zarr stores for each variable
#         if len(self.varnames) == 0:
#             self.varnames = os.listdir(self.zarrpath)
#         self.varnames = sorted(self.varnames)
# 
#         # get file paths
#         zdict = {}
#         for v in self.varnames:
#             zdict[v] = sorted(glob(os.path.join(self.zarrpath, v, v+".*.zarr")))
# 
#         # check that lists align
#         zlen = [len(z) for z in zdict.values()]
#         assert all(zlen[i] == zlen[0] for i in range(len(zlen)))
# 
#         # transpose list-of-lists; sort by key to ensure var order constant
#         zlol = list(zip(*sorted(zdict.values())))
# 
#         # lazy-load & merge zarr stores
#         self.zarrs = [lazymerge(z, self.varnames) for z in zlol]
# 
#         # Name of time dimension may vary by dataset.  ERA5 is "time"
#         # but C404 is "Time".  If dataset is CF-compliant, we
#         # can/should look for a coordinate with the attribute
#         # 'axis="T"', but C404 isn't CF, so instead we look for a dim
#         # named "time" (any capitalization), and defer checking the
#         # axis attribute until it actually comes up in practice.
#         dnames = list(self.zarrs[0].dims)
#         self.tdimname = dnames[[d.lower() for d in dnames].index("time")]
# 
#         # subset zarrs to constrain data to period defined by start
#         # and finish.  Note that xarray subsetting includes finish.
#         start, finish = self.start, self.finish
#         if start is not None or finish is not None:
#             if start is None:
#                 start = self.zarrs[0].coords[self.tdimname][0]
#             if finish is None:
#                 start = self.zarrs[-1].coords[self.tdimname][-1]
# 
#             selection = {self.tdimname: slice(start, finish)}
#             self.zarrs = [z.sel(selection) for z in self.zarrs]
# 
#         # construct indexing arrays
#         zarrlen = [z.sizes[self.tdimname] for z in self.zarrs]
#         whichseg = [list(repeat(s, z)) for s, z in zip(range(len(zarrlen)), zarrlen)]
#         segindex = [list(range(z)) for z in zarrlen]
# 
#         # subset to samples that don't overlap a segment boundary
#         # (sample size N = can't use last N-1 samples)
#         N = self.sample_len - 1
#         self.segments = flatten([s[:-N] for s in whichseg])
#         self.zindex = flatten([i[:-N] for i in segindex])
# 
#         # precompute mask arrays for subsetting data for samples
#         self.histmask = list(range(0, self.history_len, self.stride))
#         foreind = list(range(self.sample_len))
#         if self.one_shot:
#             self.foremask = foreind[-1]
#         else:
#             self.foremask = foreind[slice(self.history_len, self.sample_len, self.stride)]
# 
#     def __len__(self):
#         return len(self.zindex)
# 
#     def __getitem__(self, index):
#         return self.get_data(index)
# 
#     def get_data(self, index, do_transform=True):
#         """like gets an element by index (as __getitem__ does), but
#         with an optional argument to skip applying the normalization
#         transform.
#         """
#         # Possible refactor: instead of an optional argument, make
#         # do_transform a class attribute (i.e., a switch that can be
#         # flipped) and push this all back to __getitem__
# 
#         time = self.tdimname
#         first = self.zindex[index]
#         last = first + self.sample_len
#         seg = self.segments[index]
#         subset = self.zarrs[seg].isel({time: slice(first, last)}).load()
#         sample = Sample(
#             x=subset.isel({time: self.histmask}),
#             y=subset.isel({time: self.foremask}))
# 
#         if do_transform:
#             if self.transform:
#                 sample = self.transform(sample)
# 
#         return sample
# 
# 
# #    def get_time(self, index):
# #        """ get time coordinate(s) associated with index
# #        """
# #        time = self.tdimname
# #        first = self.zindex[index]
# #        last = first + self.sample_len
# #        seg = self.segments[index]
# #        subset = self.zarrs[seg].isel({time: slice(first, last)}).load()
# #        result = {}
# #        result["x"] =subset.isel({time: self.histmask})
# #        result["y"] =subset.isel({time: self.foremask})
# #        return result
# 
# 
# def testC4loader():
#     """Test load speed of different number of vars & storage locs.
#     Full load for C404 takes about 4 sec on campaign, 5 sec on scratch
# 
#     """
#     zdirs = {
#         "worktest": "/glade/work/mcginnis/ML/GWC/testdata/zarr",
#         "scratch": "/glade/derecho/scratch/mcginnis/conus404/zarr",
#         "campaign": "/glade/campaign/ral/risc/DATA/conus404/zarr"
#         }
#     for zk in zdirs.keys():
#         src = zdirs[zk]
#         print("######## "+zk+" ########")
#         svars = os.listdir(src)
#         for i in range(1, len(svars)+1):
#             testvars = svars[slice(0, i)]
#             print(testvars)
#             cmd = 'c4 = CONUS404Dataset("'+src+'",varnames='+str(testvars)+')'
#             print(cmd+"\t"+str(timeit(cmd, globals=globals(), number=1)))


#####################

@dataclass
class DownscalingDataset(torch.utils.data.Dataset):
    ''' pass **conf['data'] as arguments to constructor
    [insert more documentation here]
    '''
    rootpath:     str
    history_len:  int = 2
    forecast_len: int = 1
    first_date:   str = None
    last_date:    str = None
    datasets: Dict = field(default_factory=dict)

    def __post_init__(self):
        super().__init__()

        ## replace the datasets dict (which holds configurations for
        ## the various DataMaps in the dataset) with actual DataMap
        ## objects intialized from those configurations.  Need to pop
        ## datasets from __dict__ because we need to update each one
        ## with the other class attributes (which are common to all
        ## datasets) first.
        
        dmap_configs = self.__dict__.pop("datasets")
        inherited = deepcopy(self.__dict__)

        ## error if length not > 1
        
        self.datasets = dict()
        for k in dmap_configs.keys():
            dmap_configs[k].update(inherited)
            # print(dmap_configs[k].keys())
            self.datasets[k] = DataMap(**dmap_configs[k])

            
        ## don't add anything to self above here, since we use
        ## self.__dict__ as kwargs to DataMap(), and dataclasses don't
        ## like extra args to __init__
            
        self.sample_len = self.history_len + self.forecast_len
        
        self._mode = "train"
            
        dlengths = [len(d) for d in self.datasets.values()]
        self.len = np.max(dlengths)
        # TODO: error if any dlengths != self.len or 1

        self.labels = dict()
        for dset in self.datasets.keys():
            lab = self.datasets[dset].label
            self.labels.setdefault(lab, []).append(dset)

    def __getitem__(self, index):

        hlen = self.history_len
        slen = self.sample_len
        
        items = {k:self.datasets[k][index] for k in self.datasets.keys()}

        ## items is a nested dict of {dataset: {use: {var: np.ndarray}}}.
        ## where use is one of (static, boundary, prognostic, diagnostic)
        
        ## need to rearrange that into a nested dict of
        ## {label: {input/target: {var: np.ndarray}}},
        ## where input is static & bound [hist_len] for 'infer', that
        ## plus prog [hist_len] for 'init' and 'train', and target is
        ## prog [fore_len] & diag[fore_len] for 'train'
        
        result = dict()
        for lab in self.labels.keys():
            result[lab] = dict()
            
            if self.mode in ("train", "init", "infer"):
                result[lab]["input"] = dict()
                for dset in self.labels[lab]:
                    for u in items[dset].keys():
                        if u in ("static", "boundary"):
                            result[lab]["input"].update(items[dset][u])
                                    
            if self.mode in ("train", "init"):
                for dset in self.labels[lab]:
                    for u in items[dset].keys():
                        if u in ("prognostic"):
                            result[lab]["input"].update(items[dset][u])

            if self.mode in ("train",):
                result[lab]["target"] = dict()
                for dset in self.labels[lab]:
                    for u in items[dset].keys():
                        if u in ("prognostic", "diagnostic"):
                            result[lab]["target"].update(items[dset][u])
                                
                ## time subset: modes 'infer' and 'init' return only
                ## historical timesteps.  For mode 'train', we need to
                ## split the data into histlen timesteps for 'input' and
                ## forelen timesteps for 'target'

                for v in result[lab]["input"].keys():
                    x = result[lab]["input"][v]
                    if len(x.shape) == 3:
                        result[lab]["input"][v] = x[0:hlen,:,:]
                        
                for v in result[lab]["target"].keys():
                    x = result[lab]["target"][v]
                    if len(x.shape) == 3:
                        result[lab]["target"][v] = x[hlen:slen,:,:]

                        
        # transform to tensor
        # applies normalization
        # (and any other transformations)
        return result

    # actually, does it do tensor transformation?  Or do we write a
    # ToTensor object that takes a dict of variables & does it?

 
    # the mode property determines which variables to return by use type:
    # "train" = all; "init" = all but diagnostic; "infer" = static + boundary

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if mode not in ('train', 'init', 'infer'):
            raise ValueError("invalid DataMap mode")
        self._mode = mode
        for d in self.datasets.values():
            d.mode = mode
