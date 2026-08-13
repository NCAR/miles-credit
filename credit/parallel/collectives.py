"""Shared gradient all-reduce machinery for the parallel package.

Used by both sync_domain_gradients (domain.py) and sync_replicated_gradients
(tensor_parallel.py) so the subtle DTensor handling lives in exactly one place.
Also home to the mixed-mesh-safe gradient clipping used by the gen2 trainer.
"""

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.tensor import DTensor


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


def _mesh_key(grad):
    """Grouping key for mixed-mesh grad collections: the DTensor's device
    mesh (hashable), or None for plain tensors."""
    return grad.device_mesh if isinstance(grad, DTensor) else None


def total_grad_norm(grads, norm_type=2.0):
    """Total p-norm of a set of gradients that may live on different meshes.

    torch.nn.utils.get_total_norm stacks every per-grad norm with foreach
    ops, and DTensor dispatch rejects operands on different meshes (and
    plain/DTensor mixes). With native TP composed under FSDP2, the TP'd
    projections carry grads on the 2D (dp, tp) mesh while every other param
    lives on the 1D dp mesh, so the stack raises. Here grads are grouped by
    mesh; each homogeneous group goes through get_total_norm unchanged, a
    DTensor group norm is made global with full_tensor() (the correct
    reduction over its sharded dims), and the group norms combine as
    (sum_i n_i**p)**(1/p) (max for p=inf) via vector_norm. With a single
    group this is exactly torch's computation.

    Args:
        grads: iterable of gradient tensors (dense or DTensor).
        norm_type: p-norm exponent (any float, or inf).

    Returns:
        The total norm as a plain scalar tensor (never a DTensor).
    """
    grads = list(grads)
    if not grads:
        return torch.tensor(0.0)
    groups = {}
    for g in grads:
        groups.setdefault(_mesh_key(g), []).append(g)
    norms = []
    for group in groups.values():
        norm = torch.nn.utils.get_total_norm(group, norm_type)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        norms.append(norm)
    if len(norms) == 1:
        return norms[0]
    device = norms[0].device
    return torch.linalg.vector_norm(torch.stack([n.to(device) for n in norms]), norm_type)


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """Mixed-mesh-safe drop-in for ``torch.nn.utils.clip_grad_norm_``.

    The total norm comes from total_grad_norm (mesh-grouped, see above);
    the scaling reuses ``torch.nn.utils.clip_grads_with_norm_`` per mesh group,
    because torch's foreach multiply also rejects mixed plain/DTensor
    lists. The clip coefficient depends only on the already-global total
    norm, so per-group scaling is exact. For a homogeneous parameter set
    (plain DDP tensors, or all DTensors on one mesh under FSDP2/domain)
    both steps reduce to exactly torch's ``clip_grad_norm_`` computation.

    Args:
        parameters: tensor or iterable of tensors whose .grad gets clipped
            in place.
        max_norm: max norm of the gradients (float or scalar tensor).
        norm_type: p-norm exponent (any float, or inf).

    Returns:
        The total norm of the gradients (before clipping) as a plain scalar
        tensor, matching the ``torch.nn.utils.clip_grad_norm_`` contract.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    groups = {}
    for p in parameters:
        if p.grad is not None:
            groups.setdefault(_mesh_key(p.grad), []).append(p)
    if not groups:
        return torch.tensor(0.0)
    total_norm = total_grad_norm([p.grad for group in groups.values() for p in group], norm_type)
    for group in groups.values():
        torch.nn.utils.clip_grads_with_norm_(group, max_norm, total_norm)
    return total_norm
