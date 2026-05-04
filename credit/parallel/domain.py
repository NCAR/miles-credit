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


def sync_domain_gradients(model, manager):
    if manager is None or manager.domain_parallel_size <= 1:
        return
    group = manager.domain_group
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad
            # With FSDP2, p.grad is a DTensor. to_local() returns the local shard
            # as a view of the underlying storage (Shard placement), so the
            # in-place all_reduce updates the DTensor's local data correctly.
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=group)
