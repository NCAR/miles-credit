# system tools
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
class DownscalingDataset(torch.utils.data.Dataset):
    ''' pass **conf['data'] as arguments to constructor
    [insert more documentation here]
    '''

    rootpath:     str
    history_len:  int = 2
    forecast_len: int = 1
    first_date:   str = None
    last_date:    str = None
    # components: Dict = field(default_factory=dict)
    datasets: Dict = field(default_factory=dict)
    transform:    bool = True
    _mode:        str = field(init=False, repr=False, default='train')
    # legal mode values: train, init, infer
    _output:      str = field(init=False, repr=False, default='tensor')
    # legal output values: by_dset, by_io, tensor

    def __post_init__(self):
        super().__init__()

        # replace the components dict (a nested dict of
        # configurations for the constituent datasets & their
        # transforms) with actual DataMap and DataTransform objects.

        dmap_sig = signature(DataMap).parameters
        norm_sig = signature(DownscalingNormalizer).parameters

        dsdict = {}

        for dname, dconfig in self.datasets.items():
            print(dname)

            dm_args = {arg: val for arg, val in dconfig.items() if arg in dmap_sig}
            dm_args['variables'] = dconfig['variables']

            dt_args = {arg: val for arg, val in dconfig.items() if arg in norm_sig}
            dt_args['vardict'] = dconfig['variables']
            dt_args['transdict'] = dconfig['transforms']

            dsdict[dname] = {
                'datamap': DataMap(**dm_args),
                'transforms': DownscalingNormalizer(**dt_args)
            }

        self.datasets = dsdict

        self.sample_len = self.history_len + self.forecast_len

        dlengths = [len(d) for d in self.datasets.values()]
        self.len = np.max(dlengths)
        # TODO: error if any dlengths != self.len or 1

        # self.components = dict()
        # for dset in self.datasets.keys():
        #     comp = self.datasets[dset].component
        #     self.components.setdefault(comp, []).append(dset)

        # end __post_init__

    def __getitem__(self, index):

        items = {dset: self.getdata(dset, index) for dset in self.datasets}

        if self.output == 'by_dset':
            return items

        result = self.rearrange(items)

        if self.output == 'by_io':
            return result

        if self.output == 'tensor':
            result = self.toTensor(result)

        return result
        # yield result    # change return to yield to make this lazy

    def getdata(self, dset, index):
        raw = self.datasets[dset]['datamap'][index]
        if self.transform:
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
                    if dset['datamap'].dim != dim:
                        continue
                    for part in ('input', 'target'):
                        if usage not in include[self.mode][part]:
                            continue
                        for var in dset['datamap'].variables[usage]:
                            outname = f"{dname}.{var}"
                            data = items[dname][usage][var]
                            if self.mode == 'train':
                                if dim == 'static':
                                    result[part][outname] = data
                                else:
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

    def toTensor(self, sample):
        # sample is nested dict {input{vars}, target{vars}}
        # arrays are dimensioned [T, Z, Y, X]
        # stack vars along z-dim (i.e., z-levels ~= variables)

        nt = {'input': self.history_len, 'target': self.forecast_len}

        for s in sample:
            for var, data in sample[s].items():
                if len(data.shape) == 2:
                    # static data; add time dimension & repeat along it
                    data = np.repeat(np.expand_dims(data, axis=0),
                                     repeats=nt[s], axis=0)

                if len(data.shape) == 3:
                    # add singleton var/z dimension
                    data = np.expand_dims(data, axis=1)

                sample[s][var] = data

            # concatenate along z/var dim
            sample[s] = np.concatenate(list(sample[s].values()), axis=1)

            # nparray to tensor
            sample[s] = torch.as_tensor(sample[s])

        return sample

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
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
