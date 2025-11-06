from collections.abc import Iterable
from typing import Optional, List
import itertools

import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler

import logging
logger = logging.getLogger(__name__)


class MultiStepBatchSamplerSubset(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sampling_modes: Iterable,
        index_subset: Optional[List[int]] = None,  # if None, use entire dataset
    ) -> None:
        """
        The dataset is required to have attributes: 
         - init_times of valid init times, each time object compatible with:
         - timestep, the timestep to sample with

        This sampler will draw multistep batches with init indices drawn from index_subset. 
        if index_subset=None, it will draw init indices from the entire dataset.
        
        taking advantage of DistributedSampler class code with this dataset.
        can be used on its own as a non-distributed sampler with index_subset=None
        
        Args:
            data: list of data
        """

        self.sampling_modes = sampling_modes
    
        self.dataset = dataset
        self.num_forecast_steps = dataset.num_forecast_steps
        assert len(sampling_modes) == self.num_forecast_steps + 1

        for mode in self.sampling_modes:
            assert mode in self.dataset.valid_sampling_modes, f"{mode} is not a valid mode in {self.dataset.valid_sampling_modes}"

        self.init_times = dataset.init_times

        ### handle empty index subset ###
        if index_subset is not None and len(index_subset) > 0:
            # don't need to shuffle because distributed sampler shuffles for us.
            self.index_subset = torch.tensor(index_subset)
            # must all be valid starting times, this is given by the DistributedSampler Wrapper
        else:
            self.index_subset = torch.randperm(len(dataset))
        ###################################################

        self.batch_size = batch_size
       
        self.num_start_batches = (
            len(self.index_subset) + self.batch_size - 1
        ) // self.batch_size

    def __len__(self):
        # actual number of init batches
        return self.num_start_batches

    def __iter__(self):
        index_iter = iter(self.index_subset)

        batch = list(itertools.islice(index_iter, self.batch_size))
        logger.debug(f"batch indices: {batch}")
        
        while batch:
            # iterate through batches of valid starting times,
            # wrt self.num_forecast_steps
            batch_init_times = self.init_times[batch] # same as isel

            for i, mode in enumerate(self.sampling_modes):
                # for each batch of valid starting times,
                # iterate through subsequent valid forecast times

                sampling_times = batch_init_times + (i * self.dataset.timestep)

                yield [(t, mode) for t in sampling_times.values]

            batch = list(itertools.islice(index_iter, self.batch_size))

class DistributedMultiStepBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset,
                 batch_size: int,
                 sampling_modes: Iterable, # list of len forecast_steps, modes defined by dataset
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 ) -> None:
        
        super().__init__(dataset=dataset, num_replicas=num_replicas,
                         rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)

        self.batch_size = batch_size
        self.sampling_modes = sampling_modes
        self.num_forecast_steps = dataset.num_forecast_steps

    def __iter__(self):

        indices = list(super().__iter__())
        logger.debug(f"num indices: {len(indices)}")
        batch_sampler = MultiStepBatchSamplerSubset(self.dataset,
                                                    sampling_modes=self.sampling_modes,
                                                    index_subset=indices,
                                                    batch_size=self.batch_size,
                                                   )
        return iter(batch_sampler)

    def __len__(self) -> int:
        # self.num_samples is computed by super().__init__
        
        return (self.num_samples + self.batch_size - 1) // self.batch_size