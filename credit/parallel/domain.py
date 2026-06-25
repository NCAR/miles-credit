"""Domain-parallel utility functions shared across all trainers."""

import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_domain_manager(model):
    m = model
    while m is not None:
        if hasattr(m, "_domain_parallel_manager"):
            return m._domain_parallel_manager
        m = getattr(m, "module", None)
    return None


def get_raw_model(model):
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


def shard_spatial(tensor, manager):
    if manager is None or manager.domain_parallel_size <= 1:
        return tensor
    from credit.domain_parallel.sharding import shard_tensor

    n = manager.domain_parallel_size
    h = tensor.shape[-2]
    if h % n != 0:
        raise ValueError(f"Domain parallel requires image_height ({h}) divisible by domain_parallel_size ({n}).")
    return shard_tensor(tensor, dim=-2, manager=manager)


def unpad_shard_interp(y_pred, padding_opt, manager, image_h, image_w):
    import torch.nn.functional as F

    squeezed = y_pred.dim() == 5
    if squeezed:
        y_pred = y_pred.squeeze(2)
    n = manager.domain_parallel_size
    pw = padding_opt.pad_WE
    if any(v > 0 for v in pw):
        end_w = -pw[1] if pw[1] > 0 else None
        y_pred = y_pred[..., pw[0] : end_w]
    ph = padding_opt.pad_NS
    pad_top = ph[0] if manager.is_first_domain_rank else 0
    pad_bot = ph[1] if manager.is_last_domain_rank else 0
    if pad_top > 0 or pad_bot > 0:
        end_h = -pad_bot if pad_bot > 0 else None
        y_pred = y_pred[..., pad_top:end_h, :]
    shard_h = image_h // n
    if y_pred.shape[-2] != shard_h or y_pred.shape[-1] != image_w:
        logger.warning(
            f"unpad_shard_interp: shape mismatch after unpadding "
            f"({y_pred.shape[-2]}×{y_pred.shape[-1]} vs {shard_h}×{image_w}); "
            "falling back to bilinear interpolation — check padding config"
        )
        y_pred = F.interpolate(y_pred, size=(shard_h, image_w), mode="bilinear", align_corners=False)
    if squeezed:
        y_pred = y_pred.unsqueeze(2)
    return y_pred


def shard_lat_weights(weights, target_h):
    """Match latitude weights to a (possibly domain-sharded) target height.

    Single source of truth for the lat-weight sharding used by the loss and
    both metrics classes. Returns weights unchanged when heights match;
    otherwise narrows to this rank's domain shard. Raises ValueError when the
    mismatch cannot be explained by domain sharding.
    """
    weights_h = weights.shape[-2]
    if weights_h == target_h:
        return weights
    from credit.domain_parallel.manager import get_domain_parallel_manager

    manager = get_domain_parallel_manager()
    if manager is None or manager.domain_parallel_size <= 1:
        raise ValueError(f"Latitude weights height ({weights_h}) does not match target height ({target_h}).")
    if weights_h % manager.domain_parallel_size != 0:
        raise ValueError(
            f"Latitude weights height ({weights_h}) is not divisible by "
            f"domain_parallel_size ({manager.domain_parallel_size})."
        )
    shard_h = weights_h // manager.domain_parallel_size
    if shard_h != target_h:
        raise ValueError(f"Latitude weights shard height ({shard_h}) does not match target height ({target_h}).")
    return weights.narrow(-2, manager.domain_rank * shard_h, shard_h).contiguous()


def gather_spatial(tensor, manager):
    """Inverse of shard_spatial: all-gather H-shards across the domain group.

    Used between autoregressive rollout steps — the next step's input is
    assembled at full height on every domain rank, then re-sharded by
    shard_spatial. Plain (non-differentiable) all_gather is correct here:
    Reconstruct detaches y_processed between steps, so no gradient flows
    through the gathered tensors.
    """
    if manager is None or manager.domain_parallel_size <= 1:
        return tensor
    import torch

    local = tensor.contiguous()
    shards = [torch.empty_like(local) for _ in range(manager.domain_parallel_size)]
    dist.all_gather(shards, local, group=manager.domain_group)
    return torch.cat(shards, dim=-2)


def sync_domain_gradients(model, manager):
    """Average gradients across the domain-parallel group.

    See credit.parallel.collectives.allreduce_grads_avg for the bucketing and
    DTensor handling.
    """
    if manager is None or manager.domain_parallel_size <= 1:
        return
    from credit.parallel.collectives import allreduce_grads_avg

    allreduce_grads_avg(
        (p.grad for p in model.parameters() if p.grad is not None),
        manager.domain_group,
    )
