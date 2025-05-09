# system tools
from typing import Dict, TypedDict, Union
from dataclasses import dataclass, field
from inspect import signature
import warnings

# data utils
import numpy as np
import xarray as xr
import pandas as pd

# Pytorch utils
import torch
import torch.utils.data

from credit.datamap import DataMap
from credit.transforms_downscaling import DataTransforms, Identity


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
    datasets:     Dict = field(default_factory=dict)
    image_width:  int = None
    image_height: int = None
    transform:    bool = True
    _mode:        str = field(init=False, repr=False, default='train')
    # legal mode values: train, init, infer
    _output:      str = field(init=False, repr=False, default='tensor')
    # legal output values: by_dset, by_io, tensor

    def __post_init__(self):
        super().__init__()

        self._setup_datasets()
        # Set up self.datasets[dataset]['datamap'|'transforms']
        # Also sets self.data_width and self.data_height

        self.sample_len = self.history_len + self.forecast_len

        dlengths = [len(d['datamap']) for d in self.datasets.values()]
        self.len = np.max(dlengths)
        # TODO: error if any dlengths != self.len or 1

        # create self.arrangement dataframe used by .rearrange()
        # and self.tnames list of names for tensor channels
        self._setup_arrangement()

    def _setup_datasets(self):
        '''Replace the `datasets` argument dict (a nested dict of
        configurations for the constituent datasets & their
        associated transforms) with DataMap and DataTransform
        objects created from those configurations.

        Datasets are allowed to be different sizes.  When sampling, we
        automatically resize them to (image_width x image_height) by
        adding Expand and/or Pad transforms to the sequence defined
        for each dataset.  To do this, we need to construct the
        self.datasets dict in two passes, because we need the shape of
        each dataset in order to figure out those resizing transforms.

        '''

        dmap_sig = signature(DataMap).parameters
        norm_sig = signature(DataTransforms).parameters

        dsdict = {}

        # create DataMaps
        for dname, dconfig in self.datasets.items():
            # collect config arguments needed by DataMap
            dm_args = {arg: val for arg, val in dconfig.items() if arg in dmap_sig}
            dm_args['variables'] = dconfig['variables']

            dsdict[dname] = {'datamap': DataMap(**dm_args)}

        # data_width (determined by the datasets) is the width of the
        # widest dataset; image_width (declared as an argument) is the
        # width of our output tensor (likewise for height).  If not
        # defined, image_width defaults to data_width.

        # To keep differently-sized datasets co-registered, we need to
        # Expand the smaller ones up to roughly data_width before
        # resizing them using Pad transforms.

        self.data_width  = max([dsdict[d]['datamap'].shape[-1] for d in dsdict])
        self.data_height = max([dsdict[d]['datamap'].shape[-2] for d in dsdict])

        if self.image_width is None:
            self.image_width = self.data_width
        if self.image_height is None:
            self.image_height = self.data_height

        if self.data_width > self.image_width or self.data_height > self.image_height:
            warnings.warn("model size is smaller than data size; "
                          "auto-shrinking not yet implemented, code may break.")
        # TODO: extend code below to handle this case, using subsampling
        # (inverse of Expand) and cropping (inverse of Pad)

        if (self.image_width  // self.data_width  > 1 or
            self.image_height // self.data_height > 1):
            warnings.warn("data size is much smaller than model size; "
                          "data will be mostly padding")

        # create DataTransforms
        for dname, dconfig in self.datasets.items():
            # collect config arguments needed by DataTransforms
            dt_args = {arg: val for arg, val in dconfig.items() if arg in norm_sig}
            dt_args['vardict'] = dconfig['variables']
            transdict = dconfig['transforms']

            if 'default' not in transdict or transdict['default'] == 'none':
                transdict['default'] = {}

            # add transforms for auto-resizing
            # first scale smaller datasets up
            dshape = dsdict[dname]['datamap'].shape
            if dshape[-1] != self.image_width or dshape[-2] != self.image_height:
                xscale = self.data_width // dshape[-1]
                yscale = self.data_height // dshape[-2]
                scale = min(xscale, yscale)
                if xscale != yscale:
                    warnings.warn(f"dataset {dname}: expansion ratio mismatch with data size "
                                  f"(x{xscale} x vs x{yscale} y); using x{scale}")
                if scale != 1:
                    for v in transdict:
                        if v != 'paramfiles':
                            transdict[v]['expand'] = {"by": scale}
                    dshape = (dshape[-2]*scale, dshape[-1]*scale)

            # then pad datasets to match image size
            # currently only padding along left/top of array
            # TODO: add options for centered or right/bottom padding
            xdelta = self.image_width - dshape[-1]
            ydelta = self.image_height - dshape[-2]

            if xdelta != 0 or ydelta != 0:
                for v in transdict:
                    if v != 'paramfiles':
                        transdict[v]['pad'] = {'top': ydelta, 'right': xdelta}

            if transdict['default'] == {}:
                transdict['default'] = 'none'
            dt_args['transdict'] = transdict

            dsdict[dname]['transforms'] = DataTransforms(**dt_args)

            # for static data, we need to run the transforms once and
            # then discard them to prevent the cached data from
            # repeatedly mutating in place, because python is too
            # object-oriented and stateful to be a functional language
            # (pun intended)

            dmap = dsdict[dname]['datamap']
            if dmap.dim == 'static':
                xforms = dsdict[dname]['transforms'].transforms
                for var in dmap.data:
                    for x in xforms[var]:
                        dmap.data[var] = x(dmap.data[var])
                    xforms[var] = [Identity()]

        self.datasets = dsdict

    def _setup_arrangement(self):
        '''construct a pandas dataframe that defines the ordering used
        by .rearrange() to go from a nested dict structured
        [dataset][variable] (the arrangement of output from
        .getdata()) to one structured [input/target][dataset.variable]
        (the arrangement needed by .to_tensor())

        Also creates self.tnames, a list of names for the channels in
        the output tensor formatted "<dataset>.<variable>[.z<level>"

        '''

        dlist = []

        # make a dataframe for each dataset with columns for
        # variable, usage, dataset, and dimensionality
        for ds in self.datasets:
            dmap = self.datasets[ds]['datamap']
            dvar = dmap.variables
            varuse = [(var, use) for use, vlist in dvar.items() for var in vlist]
            df = pd.DataFrame(varuse, columns=['var','usage'])

            df.insert(0, 'dim', dmap.dim)
            df.insert(0, 'dataset', ds)

            df = df[df['usage'] != 'unused']

            dlist.append(df)

        # concatenate all the dataframes together
        rdf = pd.concat(dlist).reset_index(drop=True) # rdf = Rearrangement DataFrame

        # add a 'name' column = "dataset.variable"
        rdf.insert(rdf.shape[1], "name",
                   [f"{d}.{v}" for d, v in zip(rdf['dataset'], rdf['var'])])

        # columns have to be Categorical to define a custom (non-alphabetical) sort order
        rdf['usage'] = pd.Categorical(rdf['usage'], ['boundary', 'prognostic', 'diagnostic'])
        rdf['dim']   = pd.Categorical(rdf['dim'],   ['static', '2D', '3D'])
        rdf['dataset'] = pd.Categorical(rdf['dataset'], self.datasets)

        # Sort order for variables:
        # first: input-only > input + output > output-only  (bound > prog > diag)
        # then:  static > 2D > 3D
        # then:  order that datasets are defined in config
        # then:  alphabetical by variable name
        rdf = rdf.sort_values(by=['usage', 'dim', 'dataset', 'var']).reset_index(drop=True)

        self.arrangement = rdf

        # construct list of variable names corresponding to channels in (output) tensor
        # (Only done for mode=train, tensor=target at the moment)

        tarr = rdf[rdf['usage'].isin(['prognostic','diagnostic'])]
        self.tnames = []
        for row in tarr.itertuples():
            if row.dim != "3D":
                self.tnames.append(row.name)
            else:
                dmap = self.datasets[row.dataset]['datamap']
                nlev = dmap.shape[0]
                zlevels = range(0, nlev, dmap.zstride)
                ## TODO: record z-coord values in datamap, use those
                znames = [f"{row.name}.z{z}" for z in zlevels]
                self.tnames.extend(znames)

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
            result = self.to_tensor(result)

        # add date to sample for tracking purposes.  Could be None for
        # static datasets, so iterate until we get an actual value
        for dset in self.datasets.values():
            result['date'] = dset['datamap'].sindex2date(index)
            if result['date'] is not None:
                break

        return result
        # yield result    # change return to yield to make this lazy


    def getdata(self, dset, index):
        # returns nested dict: items{ dataset{ usage { var
        raw = self.datasets[dset]['datamap'][index]
        if self.transform:
            return self.datasets[dset]['transforms'](raw)

        return raw


    def rearrange(self, items):
        # based on mode, rearrange items{ dataset{ usage{ var to
        # sample{ input/target{ dset.var

        include = {"train": {"input":  ['boundary', 'prognostic'],
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

    def to_tensor(self, sample):
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
                sample[s] = torch.as_tensor(sample[s]).permute(1, 0, 2, 3).unsqueeze(0)

        # other code wants tensors named 'x' and 'y'
        sample['x'] = sample.pop('input')
        sample['y'] = sample.pop('target')

        return sample

    def revert(self, prediction):
        pass
        # invert the steps of __getitem__:
        # convert tensor to dict of xarray ndarrays,
        # organize them by dataset,
        # invert any transformations,
        # return the results

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
