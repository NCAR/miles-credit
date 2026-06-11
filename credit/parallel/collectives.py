"""Shared gradient all-reduce machinery for the parallel package.

Used by both sync_domain_gradients (domain.py) and sync_replicated_gradients
(tensor_parallel.py) so the subtle DTensor handling lives in exactly one place.
"""

import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def all_reduce_avg(tensor, group=None) -> None:
    """In-place all_reduce average that also works on gloo.

    ReduceOp.AVG is NCCL-only; gloo (CPU multi-rank runs, --backend gloo)
    needs SUM + divide.
    """
    if dist.get_backend(group) == "nccl":
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=group)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(dist.get_world_size(group))


def allreduce_grads_avg(grads, group) -> None:
    """Average gradients across a process group with minimal NCCL calls.

    Plain dense grads are flattened into one bucket per (dtype, device) so the
    sync is a handful of large all_reduces instead of one per parameter.
    DTensor grads (FSDP2) are reduced in place on their local shards — shards
    can be 0-sized or oddly strided per rank, which breaks the
    flatten/unflatten round-trip — but the per-shard all_reduces are issued
    async and waited together, so they cost one latency, not one per param.

    Args:
        grads: iterable of gradient tensors (dense or DTensor).
        group: process group to average over.
    """
    # ReduceOp.AVG is NCCL-only; on gloo use SUM and divide afterwards.
    native_avg = dist.get_backend(group) == "nccl"
    op = dist.ReduceOp.AVG if native_avg else dist.ReduceOp.SUM
    world = dist.get_world_size(group)

    buckets = {}
    handles = []
    locals_reduced = []
    for grad in grads:
        # With FSDP2, grad is a DTensor. to_local() returns the local shard as
        # a view of the underlying storage (Shard placement), so an in-place
        # all_reduce on it updates the DTensor's data correctly.
        if hasattr(grad, "to_local"):
            local = grad.to_local()
            if local.numel() > 0:
                handles.append(dist.all_reduce(local, op=op, group=group, async_op=True))
                locals_reduced.append(local)
        else:
            buckets.setdefault((grad.dtype, grad.device), []).append(grad)

    for dense in buckets.values():
        flat = _flatten_dense_tensors(dense)
        dist.all_reduce(flat, op=op, group=group)
        if not native_avg:
            flat.div_(world)
        for g, synced in zip(dense, _unflatten_dense_tensors(flat, dense)):
            g.copy_(synced)

    for h in handles:
        h.wait()
    if not native_avg:
        for local in locals_reduced:
            local.div_(world)
