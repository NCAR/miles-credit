"""Domain Parallel Manager - process group creation and coordination.

Creates a 2D logical mesh of (data_parallel, domain_parallel) process groups
from a flat world of GPUs. Domain-parallel ranks share the same data sample
but hold different spatial shards. Data-parallel ranks hold the same spatial
shard but process different data samples.

Example with 8 GPUs and domain_parallel_size=2:
    domain groups:  [0,1], [2,3], [4,5], [6,7]
    data-parallel groups: [0,2,4,6], [1,3,5,7]
"""

import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Module-level singleton
_MANAGER = None


class DomainParallelManager:
    """Manages process groups for domain parallelism.

    Args:
        world_size: Total number of GPUs.
        domain_parallel_size: Number of GPUs per domain-parallel group.
        shard_dim: Which spatial dimension to shard. -2 means latitude (H)
            in a (B, C, H, W) tensor.
    """

    def __init__(self, world_size, domain_parallel_size, shard_dim=-2):
        if world_size % domain_parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by domain_parallel_size ({domain_parallel_size})"
            )

        self.world_size = world_size
        self.domain_parallel_size = domain_parallel_size
        self.data_parallel_size = world_size // domain_parallel_size
        self.shard_dim = shard_dim

        global_rank = dist.get_rank()

        # Compute group memberships
        # Domain group index: which domain group this rank belongs to
        self._domain_group_idx = global_rank // domain_parallel_size
        # Domain rank: position within the domain group
        self._domain_rank = global_rank % domain_parallel_size
        # Data-parallel rank: position within the data-parallel group
        self._dp_rank = self._domain_group_idx

        # Create domain-parallel process groups
        # Each group contains ranks that share a data sample but have different spatial shards
        self._domain_group = None
        for i in range(self.data_parallel_size):
            ranks = list(range(i * domain_parallel_size, (i + 1) * domain_parallel_size))
            group = dist.new_group(ranks)
            if global_rank in ranks:
                self._domain_group = group

        # Create data-parallel process groups
        # Each group contains ranks with the same spatial shard but different data samples
        self._dp_group = None
        for j in range(domain_parallel_size):
            ranks = list(range(j, world_size, domain_parallel_size))
            group = dist.new_group(ranks)
            if global_rank in ranks:
                self._dp_group = group

        logger.info(
            f"DomainParallelManager: rank={global_rank}, "
            f"domain_rank={self._domain_rank}/{domain_parallel_size}, "
            f"dp_rank={self._dp_rank}/{self.data_parallel_size}, "
            f"shard_dim={shard_dim}"
        )

    @property
    def domain_group(self):
        """Process group for domain-parallel communication (halo exchange, reductions)."""
        return self._domain_group

    @property
    def data_parallel_group(self):
        """Process group for data-parallel communication (gradient sync)."""
        return self._dp_group

    @property
    def domain_rank(self):
        """Rank within the domain-parallel group (0 to domain_parallel_size-1)."""
        return self._domain_rank

    @property
    def domain_world_size(self):
        """Number of ranks in the domain-parallel group."""
        return self.domain_parallel_size

    @property
    def dp_rank(self):
        """Rank within the data-parallel group."""
        return self._dp_rank

    @property
    def dp_world_size(self):
        """Number of ranks in the data-parallel group."""
        return self.data_parallel_size

    @property
    def is_first_domain_rank(self):
        """True if this is the first rank in its domain group (north edge)."""
        return self._domain_rank == 0

    @property
    def is_last_domain_rank(self):
        """True if this is the last rank in its domain group (south edge)."""
        return self._domain_rank == self.domain_parallel_size - 1

    def neighbor_ranks(self):
        """Returns (prev_rank, next_rank) global ranks for halo exchange.

        Returns None for non-existent neighbors at edges.
        """
        global_rank = dist.get_rank()
        base = self._domain_group_idx * self.domain_parallel_size

        prev_rank = (base + self._domain_rank - 1) if self._domain_rank > 0 else None
        next_rank = (base + self._domain_rank + 1) if self._domain_rank < self.domain_parallel_size - 1 else None
        return prev_rank, next_rank


def initialize_domain_parallel(world_size, domain_parallel_size, shard_dim=-2):
    """Initialize the global DomainParallelManager singleton.

    Args:
        world_size: Total number of GPUs.
        domain_parallel_size: Number of GPUs per domain group.
        shard_dim: Spatial dimension to shard (-2 for lat in BCHW).

    Returns:
        DomainParallelManager instance.
    """
    global _MANAGER
    _MANAGER = DomainParallelManager(world_size, domain_parallel_size, shard_dim)
    return _MANAGER


def get_domain_parallel_manager():
    """Get the global DomainParallelManager singleton.

    Returns:
        DomainParallelManager or None if not initialized.
    """
    return _MANAGER
