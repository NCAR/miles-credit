# system tools
import os
from copy import deepcopy
from typing import Dict, TypedDict, Union
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
     x: Array
     y: Array

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

        self.components = dict()
        for dset in self.datasets.keys():
            comp = self.datasets[dset].component
            self.components.setdefault(comp, []).append(dset)

    def __getitem__(self, index):

        hlen = self.history_len
        slen = self.sample_len
        
        items = {k:self.datasets[k][index] for k in self.datasets.keys()}

        ## items is a nested dict of {dataset: {usage: {var: np.ndarray}}}.
        ## where usage is one of (static, boundary, prognostic, diagnostic)

        ## Note that self.mode determines which usages we want to
        ## return.  The DataMap object understands mode and reads &
        ## returns only the necessary bits, so we don't need to worry
        ## about it here.
        
        ## We pass each item in the dict (1 item = 1 dataset) to the
        ## corresponding Normalizer object for normalization

        ## then we pass it to to dict<-> tensor converter, which
        ## unstacks 3D variables, converts all the arrays to tensors,
        ## and concatenates them together in the correct arrangment
        ## The code below all will move there:
        ## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        
        ## need to rearrange that into a nested dict of
        ## {component: {input/target: {var: np.ndarray}}},
        ## where input is static & bound [hist_len] for 'infer', that
        ## plus prog [hist_len] for 'init' and 'train', and target is
        ## prog [fore_len] & diag[fore_len] for 'train'
        
        result = dict()
        for comp in self.components.keys():
            result[comp] = dict()
            
            if self.mode in ("train", "init", "infer"):
                result[comp]["input"] = dict()
                for dset in self.components[comp]:
                    for u in items[dset].keys():
                        if u in ("static", "boundary"):
                            result[comp]["input"].update(items[dset][u])
                                    
            if self.mode in ("train", "init"):
                for dset in self.components[comp]:
                    for u in items[dset].keys():
                        if u in ("prognostic"):
                            result[comp]["input"].update(items[dset][u])

            if self.mode in ("train",):
                result[comp]["target"] = dict()
                for dset in self.components[comp]:
                    for u in items[dset].keys():
                        if u in ("prognostic", "diagnostic"):
                            result[comp]["target"].update(items[dset][u])

                
            ## time subset: modes 'infer' and 'init' return only
            ## historical timesteps.  For mode 'train', we need to
            ## split the data into histlen timesteps for 'input' and
            ## forelen timesteps for 'target'

            ## In the main code, toTensor() returns a dict of tensors
            ## and Trainer combines them.  This goes better here,
            ## though.

            if self.mode in ("train",):                
                for v in result[comp]["input"].keys():
                    x = result[comp]["input"][v]
                    if len(x.shape) == 3:
                        result[comp]["input"][v] = x[0:hlen,:,:]
                        
                for v in result[comp]["target"].keys():
                    x = result[comp]["target"][v]
                    if len(x.shape) == 3:
                        result[comp]["target"][v] = x[hlen:slen,:,:]

        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        return result


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
