"""Ring-reduce ensemble CRPS loss for distributed training.

One ensemble member per data-parallel rank. Member diversity comes from the
per-dp-rank RNG seed (see train_gen2: seed + data_rank) acting on stochastic
model components (dropout, SKEBS, IC perturbations); every dp rank must
receive the SAME batch, so ring-crps training disables dp dataset sharding.
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def ring_crps_loss(y_pred, y, group=None):
    """Fair CRPS via ring communication — O(1) extra memory, no all_gather.

    Each dp rank holds 1 ensemble member of the same sample. K-1 ring shifts
    pass one member buffer at a time so rank r accumulates |x_r - x_j| for
    every j != r without ever materialising the full K-member ensemble on a
    single device.

    Gradient correctness (no cross-rank backward needed):
        d(CRPS)/d(x_r) = sign(x_r-y)/K  -  sum_{j!=r} sign(x_r-x_j) / (K*(K-1))
    Both terms are computed entirely from rank r's local graph. DDP averaging
    (1/K) then gives the correct model-parameter gradient.

    Local loss (scaled so DDP avg = d(CRPS)/d(params)):
        loss_r = |y_pred_r - y|  -  sum_{j!=r} |y_pred_r - x_j| / (K-1)

    Args:
        y_pred: Local member prediction, requires_grad. Same shape on every
            rank in the group.
        y:      Truth tensor, broadcastable against y_pred.
        group:  Process group holding one member per rank (the dp group).
            None falls back to WORLD. With K == 1 (or torch.distributed not
            initialized) the spread term vanishes and the loss reduces to MAE.

    Returns:
        Scalar loss; backward populates y_pred.grad correctly.
    """
    if not dist.is_initialized():
        return (y_pred - y).abs().mean()

    group = group if group is not None else dist.group.WORLD
    K = dist.get_world_size(group)
    if K == 1:
        return (y_pred - y).abs().mean()

    group_rank = dist.get_rank(group)
    send_peer = dist.get_global_rank(group, (group_rank + 1) % K)
    recv_peer = dist.get_global_rank(group, (group_rank - 1 + K) % K)

    skill = (y_pred - y).abs()

    buf = y_pred.detach().contiguous()
    spread = torch.zeros_like(y_pred)
    for _ in range(K - 1):
        next_buf = torch.empty_like(buf)
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, buf, send_peer, group=group),
                dist.P2POp(dist.irecv, next_buf, recv_peer, group=group),
            ]
        )
        for req in reqs:
            req.wait()
        buf = next_buf
        spread = spread + (y_pred - buf).abs()

    return (skill - spread / (K - 1)).mean() / K


class RingCRPSLoss(torch.nn.Module):
    """Criterion-compatible wrapper around ring_crps_loss.

    Matches the trainer call convention criterion(y, y_pred): target first,
    prediction second. Returns a scalar, so the trainer's .mean() is a no-op.

    The process group is resolved lazily at first forward (the dp group is
    registered by distributed_model_wrapper_gen2, which runs after load_loss).
    """

    def __init__(self, reduction="none", group=None):
        super().__init__()
        self._group = group
        self._group_resolved = group is not None

    def forward(self, target, pred):
        if not self._group_resolved:
            from credit.parallel.mesh import get_dp_group

            self._group = get_dp_group()
            self._group_resolved = True
        return ring_crps_loss(pred, target, group=self._group)
