# system tools
from copy import deepcopy
from typing import Dict, TypedDict, Union
from dataclasses import dataclass, field
from inspect import signature

# data utils
import numpy as np
import xarray as xr

# Pytorch utils
import torch
import torch.utils.data

from credit.datamap import *
from credit.transforms_downscaling import *


Array = Union[np.ndarray, xr.DataArray]


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.

    x = predictor (input) data
    y = predictand (target) data

    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple, which would
    provide runtime checks, but the deal-breaker with Tuples is that
    they're immutable so we cannot change the values in the
    transforms.

    """
    input:  Array
    target: Array


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
#         print("#### "+zk+" ####")
#         svars = os.listdir(src)
#         for i in range(1, len(svars)+1):
#             testvars = svars[slice(0, i)]
#             print(testvars)
#             cmd = 'c4 = CONUS404Dataset("'+src+'",varnames='+str(testvars)+')'
#             print(cmd+"\t"+str(timeit(cmd, globals=globals(), number=1)))


###########

@dataclass
class DownscalingDataloader(torch.utils.data.Dataset):
    ''' pass **conf['data'] as arguments to constructor
    [insert more documentation here]
    '''
    rootpath:     str
    history_len:  int = 2
    forecast_len: int = 1
    first_date:   str = None
    last_date:    str = None
    #components: Dict = field(default_factory=dict)
    datasets: Dict = field(default_factory=dict)
    normalize:    bool= True
    _mode:        str = field(init=False, repr=False, default='train')
    # legal mode values: train, init, infer
    _output:      str = field(init=False, repr=False, default='by_io')
    # legal output values: by_dset, by_io, tensor

    def __post_init__(self):
        super().__init__()

        # replace the components dict (a nested dict of
        # configurations for the constituent datasets & their
        # transforms) with actual DataMap and DataTransform objects.

        dmap_sig = signature(DataMap).parameters
        norm_sig = signature(DownscalingNormalizer).parameters

        for dname, dconfig in self.datasets.items():
            print(dname)

            dm_args = {arg: val for arg,val in dconfig.items() if arg in dmap_sig}
            dm_args['variables'] = dconfig['variables']

            dt_args = {'rootpath': self.rootpath,
                       'vardict': dconfig['variables'],
                       'transdict': dconfig['transforms']
                       }

            self.datasets[dname] = {
                'datamap': DataMap(**dm_args),
                'transforms': DownscalingNormalizer(**dt_args)
            }

        self.sample_len = self.history_len + self.forecast_len

        dlengths = [len(d) for d in self.datasets.values()]
        self.len = np.max(dlengths)
        # TODO: error if any dlengths != self.len or 1

        # self.components = dict()
        # for dset in self.datasets.keys():
        #     comp = self.datasets[dset].component
        #     self.components.setdefault(comp, []).append(dset)


    def getdata(self, dset, index): #, normalize=self.normalize):
        raw = self.datasets[dset]['datamap'][index]
        if self.normalize:
            return self.datasets[dset]['transforms'](raw)
        else:
            return raw

    def rearrange(self, items):
        # based on mode, rearrange items{ dataset{ usage{ var to
        # sample{ input/target{ dset.var

        include = {"train": {"input":  ['boundary', 'prognostic', 'diagnostic'],
                             "target": [            'prognostic', 'diagnostic']},
                   "init":  {"input":  ['boundary', 'prognostic'],
                             "target": []},
                   "infer": {"input":  ['boundary',],
                             "target": []},
                   }

        result = {'input': {}, 'target': {}}

        hlen = self.history_len
        slen = self.sample_len

        for usage in ("boundary", "prognostic", "diagnostic"):
            for dim in ("static", "2D", "3D"):
                for dname, dset in self.datasets.items():
                    if dset['datamap'].dim == dim:
                        for part in ('input', 'target'):
                            if usage in include[self.mode][part]:
                                for var in dset['datamap'].variables[usage]:
                                    outname = dname + '.' + var
                                    data = items[dname][usage][var]
                                    if self.mode == 'train':
                                        if part == 'input':
                                            result[part][outname] = data[0:hlen, ...]
                                        if part == 'target':
                                            result[part][outname] = data[hlen:slen, ...]
                                    else:
                                        result[part][outname] = data

        # subsetting time dimension to hist / future is only needed
        # for training data; in mode init or infer, the datamaps only
        # return the historical part of the sample

        return result


    def __getitem__(self, index):

        items = {dset: self.getdata(dset, index) for dset in self.datasets}

        if self.output == 'by_dset':
            return items

        result = self.rearrange(items)

        if self.output == 'by_io':
            return result
        
        # if self.output = 'tensor':
        #    result = self.toTensor(result)

        return result


    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        print("_mode:", self._mode)
        print("self.mode:", self.mode)
        print("mode arg:", mode)
        if mode not in ('train', 'init', 'infer'):
            raise ValueError("invalid DataMap mode")
        self._mode = mode
        for d in self.datasets.values():
            d['datamap'].mode = mode

            
    @property
    def output(self) -> str:
        return self._output

    @output.setter
    def output(self, output: str):
        if output not in ('by_dset', 'by_io', 'tensor'):
            raise ValueError("invalid DataMap output")
        self._output = output
