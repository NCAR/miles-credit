"""Tensor sharding and gathering utilities for domain parallelism.

Provides functions to split tensors along a spatial dimension across
domain-parallel ranks, and to gather them back for metrics/output.
"""

import torch
import torch.distributed as dist

from credit.domain_parallel.manager import get_domain_parallel_manager


def shard_tensor(x, dim=-2, manager=None):
    """Shard a tensor along the given dimension across domain-parallel ranks.

    Splits the tensor into equal chunks and returns only this rank's chunk.

    Args:
        x: Input tensor.
        dim: Dimension to shard along (default: -2 for H in BCHW or BCTHW).
        manager: DomainParallelManager instance (uses global singleton if None).

    Returns:
        Local shard of the tensor.
    """
    if manager is None:
        manager = get_domain_parallel_manager()
    if manager is None or manager.domain_parallel_size <= 1:
        return x

    dim = dim % x.ndim
    size = x.shape[dim]
    n = manager.domain_parallel_size
    rank = manager.domain_rank

    if size % n != 0:
        raise ValueError(f"Cannot shard dim {dim} of size {size} evenly across {n} domain-parallel ranks")

    chunk_size = size // n
    start = rank * chunk_size
    return x.narrow(dim, start, chunk_size).contiguous()


def gather_tensor(x, dim=-2, manager=None):
    """Gather sharded tensor from all domain-parallel ranks.

    All-gathers the tensor along the sharding dimension and concatenates.

    Args:
        x: Local shard tensor.
        dim: Dimension that was sharded (default: -2).
        manager: DomainParallelManager instance (uses global singleton if None).

    Returns:
        Full (gathered) tensor.
    """
    if manager is None:
        manager = get_domain_parallel_manager()
    if manager is None or manager.domain_parallel_size <= 1:
        return x

    dim = dim % x.ndim
    n = manager.domain_parallel_size
    group = manager.domain_group

    # Prepare gather list
    gather_list = [torch.empty_like(x) for _ in range(n)]
    dist.all_gather(gather_list, x.contiguous(), group=group)

    return torch.cat(gather_list, dim=dim)


def shard_batch(batch, spatial_dims_5d=-2, spatial_dims_4d=-2, manager=None):
    """Shard all spatial tensors in a training batch dictionary.

    Handles both 5D (B, C, T, H, W) and 4D (B, C, H, W) tensors.
    Non-tensor entries and 1D/2D tensors are left unchanged.

    Args:
        batch: Dictionary of batch data (from dataloader).
        spatial_dims_5d: Dimension to shard in 5D tensors (default: -2 for H).
        spatial_dims_4d: Dimension to shard in 4D tensors (default: -2 for H).
        manager: DomainParallelManager instance.

    Returns:
        New dict with sharded tensors.
    """
    if manager is None:
        manager = get_domain_parallel_manager()
    if manager is None or manager.domain_parallel_size <= 1:
        return batch

    sharded = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            if val.ndim == 5:
                sharded[key] = shard_tensor(val, dim=spatial_dims_5d, manager=manager)
            elif val.ndim == 4:
                sharded[key] = shard_tensor(val, dim=spatial_dims_4d, manager=manager)
            else:
                # Scalars, indices, stop_forecast etc. — leave unchanged
                sharded[key] = val
        else:
            sharded[key] = val

    return sharded
