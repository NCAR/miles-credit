from typing import Optional
import itertools
from torch.utils.data import Dataset, Sampler, DistributedSampler
import logging

logger = logging.getLogger(__name__)


class MultiStepBatchSamplerSubset(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        index_subset,
        num_forecast_steps: int,
    ) -> None:
        """Draw multi-step batches with init indices drawn from *index_subset*.

        The timestep ``dt`` is inferred from consecutive entries in
        ``dataset.datetimes``; no ``dt`` attribute is required on the dataset.

        Args:
            dataset: Dataset exposing a ``datetimes`` attribute
                (pd.DatetimeIndex of valid initialisation timestamps).
            batch_size: Number of init times per batch.
            index_subset: List of integer indices into ``dataset.datetimes``
                to draw from.
            num_forecast_steps: Number of rollout steps to yield per init time
                (``forecast_len + 1``).
        """
        self.dataset = dataset
        self.num_forecast_steps = num_forecast_steps
        self.init_times = dataset.datetimes
        self.dt = dataset.datetimes[1] - dataset.datetimes[0]
        self.index_subset = index_subset
        self.batch_size = batch_size
        self.num_start_batches = (len(self.index_subset) + self.batch_size - 1) // self.batch_size

    def __len__(self):
        # actual number of init batches
        return self.num_start_batches

    def __iter__(self):
        index_iter = iter(self.index_subset)
        batch = list(itertools.islice(index_iter, self.batch_size))
        logger.debug(f"batch indices: {batch}")

        while batch:
            # iterate through batches of valid starting times wrt self.num_forecast_steps
            batch_init_times = self.init_times[batch]

            # for each batch of valid starting times, iterate through subsequent valid forecast times
            for i, _ in enumerate(range(self.num_forecast_steps)):
                sampling_times = batch_init_times + (i * self.dt)

                yield [(t, i) for t in sampling_times.values]

            batch = list(itertools.islice(index_iter, self.batch_size))


class DistributedMultiStepBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_forecast_steps: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )

        self.batch_size = batch_size
        self.num_forecast_steps = num_forecast_steps

    def __iter__(self):
        index_subset = list(super().__iter__())
        logger.debug(f"num indices: {len(index_subset)}")
        batch_sampler = MultiStepBatchSamplerSubset(
            self.dataset,
            batch_size=self.batch_size,
            index_subset=index_subset,
            num_forecast_steps=self.num_forecast_steps,
        )

        return iter(batch_sampler)

    def __len__(self) -> int:
        # self.num_samples is computed by super().__init__
        return (self.num_samples + self.batch_size - 1) // self.batch_size
