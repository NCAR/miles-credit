# system tools
from typing import Dict, TypedDict, Union
from dataclasses import dataclass, field
from inspect import signature

# data utils
import numpy as np
import xarray as xr
import pandas as pd

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
    history_len:  int = 1
    forecast_len: int = 1
    valid_history_len:  int = 1
    valid_forecast_len: int = 1
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

        # construct a pandas dataframe that defines the ordering used
        # by .rearrange() to go from nested dict [dataset][variable]
        # to nested dict [input/target][dataset.variable]

        dlist = list()
        for ds in self.datasets:
            dmap = self.datasets[ds]['datamap']
            dvar = dmap.variables
            varuse = [(var, use) for use, vlist in dvar.items() for var in vlist]
            df = pd.DataFrame(varuse, columns=['var','usage'])

            df.insert(0, 'dim', dmap.dim)
            df.insert(0, 'dataset', ds)

            df = df[df['usage'] != 'unused']

            dlist.append(df)

        rdf = pd.concat(dlist).reset_index(drop=True) # rdf = Rearrangement DataFrame

        rdf.insert(rdf.shape[1], "name",
                   [f"{d}.{v}" for d, v in zip(rdf['dataset'], rdf['var'])])

        # columns have to be Categorical to define a custom (non-alphabetical) sort order
        rdf['usage'] = pd.Categorical(rdf['usage'], ['boundary', 'prognostic', 'diagnostic'])
        rdf['dim']   = pd.Categorical(rdf['dim'],   ['static', '2D', '3D'])
        rdf['dataset'] = pd.Categorical(rdf['dataset'], self.datasets)

        # Sort order for variables:
        # first: input-only > input + output > output-only
        # then:  static > 2D > 3D
        # then:  order that datasets are defined in config
        # then:  alphabetical by variable name
        rdf = rdf.sort_values(by=['usage', 'dim', 'dataset', 'var']).reset_index(drop=True)

        self.arrangement = rdf

        # construct list of variable names corresponding to channels in (output) tensor
        # (Only done for mode=train, tensor=target at the moment)

        tarr = rdf[rdf['usage'].isin(['prognostic','diagnostic'])]
        self.tnames = list()
        for row in tarr.itertuples():
            if row.dim != "3D":
                self.tnames.append(row.name)
            else:
                dmap = self.datasets[row.dataset]['datamap']
                nlev = dmap.shape[0]
                zlevels = range(0, nlev, dmap.zstride)
                ## todo: record z-coord values in datamap, use those
                znames = [f"{row.name}.z{z}" for z in zlevels]
                self.tnames.extend(znames)
        
        # end __post_init__

    def __len__(self):
        return self.len

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

        for part in result:
            for row in self.arrangement.itertuples():
                
                if row.usage in include[self.mode][part]:
                    data = items[row.dataset][row.usage][row.var]
                    if self.mode == 'train':
                        if row.dim == 'static':
                            result[part][row.name] = data
                        else:
                            if part == 'input':
                                result[part][row.name] = data[0:hlen, ...]
                            if part == 'target':
                                result[part][row.name] = data[hlen:slen, ...]
                    else:
                        result[part][row.name] = data

        # subsetting time dimension to hist / future is only needed
        # for training data; in mode init or infer, the datamaps only
        # return the hist part of the sample

        return result

    def toTensor(self, sample):
        # sample is nested dict {input{vars}, target{vars}}
        # arrays are dimensioned [T, Z, Y, X]
        # stack vars along z-dim (i.e., z-levels ~= variables)
        # combine to tensor [V, T, Y, X]

        nt = {'input': self.history_len, 'target': self.forecast_len}

        for s in sample:
            if len(sample[s]) == 0:
                sample[s] = None
            else:
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
    
                # nparray[T,Z,Y,X] to tensor[V,T,Y,X]
                sample[s] = torch.as_tensor(sample[s]).permute(1, 0, 2, 3)

        # other code wants tensors named 'x' and 'y'
        sample['x'] = sample.pop('input')
        sample['y'] = sample.pop('target')

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
