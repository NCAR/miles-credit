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

from credit.datasets.datamap import DataMap
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
    '''Class that wrangles a collection of DataMaps and their
    associated DataTransforms for use in ML.  Intended to be
    initialized with `**conf['data']`; see `config/downscaling.yml`
    for an example.

    Constructor Arguments:

        rootpath (str): pathway to the files

        history_len (int): number of input timesteps for training
        forecast_len (int): number of output timesteps for training
        valid_history_len (int): number of input timesteps for validation
        valid_forecast_len (int): number of output timesteps for validation
            all _history_len and _forecast_len arguments default to 1

        first_date (YYYY-MM-DD string): starting point of dataset
        last_date: (YYYY-MM-DD string): ending point of dataset
            Use first_date and last_date to define a subset of the dataset,
            e.g., to use 1980-2000 for training and 2001-2010 for testing.
            Default to first and last timesteps in the files, respectively.

        datasets (nested dict): dicts of parameters for initializing
            DataMaps and their corresponding DataTransforms via
            **kwargs.  These dicts are replaced with actual objects
            during initialization.

        image_width (int): width of dataset in gridcells
        image_height (int): height of dataset in gridcells
            image_width and _height are used to automatically resize
            different constituent datasets to the same size via
            expansion & padding.  E.g., if declared image size is
            120x120, a 55x55 dataset would be expanded by a factor of
            2 and padded by 10 to match.  Defaults to the width
            (height) of the widest (tallest) dataset.

        get_time_from (str): name of the dataset to pull time
            coordinates from when creating a sample; used when writing
            output to netcdf.  Defaults to first non-static dataset
            that has boundary variables.

        transform (bool): apply normalizing transforms to samples?
            Defaults to True

    Additional Attributes:

        sample_len: number of timesteps in a sample
            (sample_len = history_len + forecast_len)

        len: number of samples in the dataset, which will be less than
            the number of timesteps in the netcdf files, due to
            first_date/last_date subsetting and sample_len > 1.

        mode: determines which variables are returned when sampling.
            Valid values are 'train', 'test', and 'predict'.
            Defaults to 'train'.

            train: boundary & prognostic variables for input timesteps,
                prognostic and diagnostic for target timesteps
            test: boundary variables for input timesteps,
                prognostic and diagnostic for target timesteps
            predict: boundary variables for input timesteps,
                nothing for target timesteps

        output: determines the format of samples; see __getitem__ for
            details.  Valid values: 'by_dset', 'by_io', 'tensor'.
            Defaults to 'tensor'.

        arrangement: a pandas dataframe that defines how variables are
            ordered in the tensor.  Columns: dataset, dim, var, usage, name

        tnames: a list of names (structured `dataset.var[.z-level]`)
            corresponding to the channels in an output tensor (i.e.,
            prognostic and diagnostic variables only).

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
    get_time_from: str = None
    _mode:        str = field(init=False, repr=False, default='train')
    # legal mode values: train, init, infer
    _output:      str = field(init=False, repr=False, default='tensor')
    # legal output values: by_dset, by_io, tensor

    def __post_init__(self):
        super().__init__()

        # this needs to go before _setup_datasets
        if self.get_time_from is None:
            for d in self.datasets:
                if (self.datasets[d]['dim'] != 'static' and
                    'boundary' in self.datasets[d]['variables'] and
                    len(self.datasets[d]['variables']['boundary']) > 0):
                    self.get_time_from = d
                    break
            else:
                # Python for...else executes if loop exits w/o break
                raise(ValueError("No non-static datasets with boundary vars (needed for output time coords)"))
        
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
        by :meth:`~.rearrange` to go from the nested dict arrangement
        returned by :meth:`~.getdata` to the arrangement used as input
        by :meth:`~.to_tensor`.  See :meth:`~.__getitem__` for more
        details.

        Also creates self.tnames, a list of names for the channels in
        the output tensor formatted "<dataset>.<variable>[.z<level>]"

        '''

        dlist = []

        # make a dataframe for each dataset with columns for
        # variable, usage, dataset, and dimensionality
        for ds in self.datasets:
            dmap = self.datasets[ds]['datamap']
            dvar = dmap.variables
            varuse = [(var, use) for use, vlist in dvar.items() for var in vlist]
            df = pd.DataFrame(varuse, columns=['var', 'usage'])

            df.insert(0, 'dim', dmap.dim)
            df.insert(0, 'dataset', ds)

            df = df[df['usage'] != 'unused']

            dlist.append(df)

        # concatenate all the dataframes together
        rdf = pd.concat(dlist).reset_index(drop=True)  # rdf = Rearrangement DataFrame

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

        tarr = rdf[rdf['usage'].isin(['prognostic', 'diagnostic'])]
        self.tnames = []
        for row in tarr.itertuples():
            if row.dim != "3D":
                self.tnames.append(row.name)
            else:
                dmap = self.datasets[row.dataset]['datamap']
                nlev = dmap.shape[0]
                zlevels = range(0, nlev, dmap.zstride)
                # TODO: record z-coord values in datamap, use those
                znames = [f"{row.name}.z{z}" for z in zlevels]
                self.tnames.extend(znames)

    def __len__(self):
        '''Number of samples in the dataset.  Note that this is
        smaller than the number of timesteps in the files, both
        because `first_date` and `last_date` may not cover the full
        range, and because sample length = `history_len` +
        `forecast_len` and `~.__getitem__` only returns complete
        samples.

        '''
        return self.len

    def __getitem__(self, index):
        '''Gets the index'th sample from the dataset.  The value of the
        'output' attribute controls the format of the returned object.

        `~.output == 'by_dset'` returns a nested dict structured
        [dataset][usage][variable] (the format returned by
        :meth:`~.getdata`); the leaf elements are numpy ndarrays
        covering the full time period (history_len+forecast_len).

        `~.output == 'by_io'` splits the ndarrays into history and
        forecast periods and reorganizes the nested dict to
        [input/target][dataset.variable] (the format returned by
        :meth:`~.rearrange` and taken as input by
        :meth:`~.to_tensor`).  The variables are ordered:

            - first: boundary > prognostic > diagnostic (in-only > in-out > out-only)
            - then: static > 2D > 3D
            - then: order that datasets are defined in config
            - then: alphabetical by variable name

        `~.output == 'tensor'` stacks the ndarrays in the z-dimension
        and converts them to a pair of pyTorch tensors (input and
        target), returning them as a Sample.  It also includes the
        associated time coordinates in the sample

        '''

        items = {dset: self.getdata(dset, index) for dset in self.datasets}

        if self.output == 'by_dset':
            return items

        result = self.rearrange(items)

        if self.output == 'by_io':
            return result

        if self.output == 'tensor':
            result = self.to_tensor(result)

        # add date to sample for tracking purposes.
        result['dates'] = self.datasets[self.get_time_from]['datamap'].sindex2dates(index)

        return result
        # yield result    # change return to yield to make this lazy

    def getdata(self, dset, index):
        '''gets data for the index'th sample from dataset `dset`.
        Returns a nested dict organized [dataset][usage][variable].

        '''
        # returns nested dict: items{ dataset{ usage { var
        raw = self.datasets[dset]['datamap'][index]
        if self.transform:
            return self.datasets[dset]['transforms'](raw)

        return raw

    def rearrange(self, items):
        '''Rearranges a nested dict of data from [dataset][usage][var]
        to [input/target][dataset.var]. Elements returned depend on
        `~.mode`:

            - `train`: input contains boundary and prognostic
              variables, target contains prognostic and diagnostic
              variables.

            - `init`: input contains boundary and prognostic
              variables, target is empty.

            - `infer`: input contains boundary variables, `target` is
              empty.

        '''
        # TODO: change mode from train/init/infer to
        # train/test/predict.  test should contain only boundary vars
        # on input, prog and diag on target.  Initialize testing with
        # mode=train, then switch to test.  For predict, we have to
        # initialize the prognostic variables using some other data
        # source (since we have no high-res data for downscaling an
        # arbitrary GCM.  An interpolated version of the corresponding
        # coarse GCM data would be ideal, but requires wrangling and
        # entire extra dataset and not all varaibles may be available.
        # Could also init with climatology or [0,1] uniform noise
        # (post-transform) for those vars.

        # where: here, rollout_downscaling, datamap, datasets/load_dataset_and_dataloader

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
        '''Takes a nested dict organized [input/target][vars], with
        data arrays (numpy ndarrays) dimensioned [T, Z, Y, X].

        Stacks variables along the z-dimension (i.e., so different
        z-levels of a 3D variable are treated as different variables).

        Combines variable stacks into tensors ordered [V, T, Y, X] and
        returns a Sample where x is historical / input data and y is
        forecast / target data.

        '''
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
        '''Converts a tensor (ML model output) back into nested dict
        of numpy arrays (i.e, reverses the `getdata -> rearrange ->
        to_tensor` pipeline).

        '''

        # get rid of degenerate batch dimension
        assert(len(prediction.shape)==5 and prediction.shape[0]==1)
        prediction = prediction.squeeze(0)

        result = {dset: {} for dset in self.datasets}

        for i in range(len(self.tnames)):
            ndim = self.tnames[i].count('.') + 1
            if ndim == 2:
                dset, varname = self.tnames[i].split('.')
                # squeeze to get rid of degenerate variable dim
                result[dset][varname] = prediction[i, ...].squeeze().numpy()
            elif ndim == 3:
                dset, varname, zlev = self.tnames[i].split('.')
                if varname not in result[dset]:
                    result[dset][varname] = [prediction[i, ...]]
                else:
                    result[dset][varname].append(prediction[i, ...])
            else:
                raise ValueError(f"Tensor index name '{self.tnames[i]}' has no/too many '.' in it")

        result2 = {}
        for dset, vardict in result.items():
            if vardict:  # evaluates to False for empty directories
                for var, data in vardict.items():
                    if isinstance(data, list):
                        vardict[var] = np.concatenate(data)
                result2[dset] = vardict

        if self.transform:
            for dset in result2:
                xform = self.datasets[dset]['transforms']
                # dummy outer dict; xform wants a dict structured [usage][var]
                result2[dset] = xform({"_": result2[dset]}, inverse=True)["_"]

        return result2

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
