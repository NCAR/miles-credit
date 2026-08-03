"""Domain parallelism for CREDIT weather models.

Shards high-resolution input data along spatial dimensions (latitude by default)
across multiple GPUs, enabling training on data too large for a single GPU.
Inspired by PhysicsNeMo's ShardTensor framework.

Key components:
- DomainParallelManager: Process group creation and coordination
- HaloExchange: Differentiable boundary communication for convolutions
- Domain-parallel layers: Conv2d, Conv3d, ConvTranspose2d, GroupNorm wrappers
- convert_to_domain_parallel: Automatic model conversion
- shard_tensor / gather_tensor: Input/output distribution

Usage:
    from credit.domain_parallel import (
        initialize_domain_parallel,
        get_domain_parallel_manager,
        convert_to_domain_parallel,
        shard_tensor,
        gather_tensor,
        shard_batch,
    )
"""

from credit.domain_parallel.convert import (
    convert_to_domain_parallel,
    validate_sharding_constraints,
)
from credit.domain_parallel.halo_exchange import HaloExchange
from credit.domain_parallel.layers import (
    DomainParallelConv2d,
    DomainParallelConv3d,
    DomainParallelConvTranspose2d,
    DomainParallelConvTranspose3d,
    DomainParallelGroupNorm,
    DomainParallelInterpolate,
    DomainParallelPeriodicConv2d,
)
from credit.domain_parallel.manager import (
    DomainParallelManager,
    get_domain_parallel_manager,
    initialize_domain_parallel,
)
from credit.domain_parallel.sharding import (
    gather_tensor,
    shard_batch,
    shard_tensor,
)

__all__ = [
    "DomainParallelConv2d",
    "DomainParallelConv3d",
    "DomainParallelConvTranspose2d",
    "DomainParallelConvTranspose3d",
    "DomainParallelGroupNorm",
    "DomainParallelInterpolate",
    "DomainParallelManager",
    "DomainParallelPeriodicConv2d",
    "HaloExchange",
    "convert_to_domain_parallel",
    "gather_tensor",
    "get_domain_parallel_manager",
    "initialize_domain_parallel",
    "shard_batch",
    "shard_tensor",
    "validate_sharding_constraints",
]
