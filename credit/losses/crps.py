"""Ring-reduce ensemble CRPS loss for distributed training.

One ensemble member per data-parallel rank. Member diversity comes from the
per-dp-rank RNG seed (see train_gen2: seed + data_rank) acting on stochastic
model components (dropout, SKEBS, IC perturbations); every dp rank must
receive the SAME batch, so ring-crps training disables dp dataset sharding.

Two entry points share the same ring exchange:

- flat criterion: ``loss: {training_loss: ring-crps}`` — RingCRPSLoss scores
  the whole normalized ``y_pred`` tensor at once.
- Gen 2 BaseLoss univariate: ``loss: {type: base, args: {training_loss:
  ring-crps, ...}}`` — each variable is scored elementwise in physical units
  and combined with BaseLoss's per-variable weighting (K-1 exchanges per
  variable per step). The launch contract is identical; train_gen2 resolves
  the nested name via ``credit.losses.effective_loss_name``.
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def ring_crps_elementwise(y_pred, y, group=None):
    """Elementwise fair-CRPS contribution via ring communication.

    Each dp rank holds 1 ensemble member of the same sample. K-1 ring shifts
    pass one member buffer at a time so rank r accumulates ``|x_r - x_j|`` for
    every j != r without ever materialising the full K-member ensemble on a
    single device.

    Gradient correctness (no cross-rank backward needed)::

        d(CRPS)/d(x_r) = sign(x_r-y)/K  -  sum_{j!=r} sign(x_r-x_j) / (K*(K-1))

    Both terms are computed entirely from rank r's local graph. DDP averaging
    (1/K) then gives the correct model-parameter gradient.

    Local elementwise contribution (scaled so DDP avg = d(CRPS)/d(params))::

        loss_r = [ |y_pred_r - y|  -  sum_{j!=r} |y_pred_r - x_j| / (K-1) ] / K

    The scaling survives any linear reduction applied downstream — a plain
    ``.mean()`` (``ring_crps_loss``) or a weighted mean such as BaseLoss's
    per-variable latitude/variance weighting.

    Args:
        y_pred: Local member prediction, requires_grad. Same shape on every
            rank in the group.
        y:      Truth tensor, broadcastable against y_pred.
        group:  Process group holding one member per rank (the dp group).
            None falls back to WORLD. With K == 1 (or torch.distributed not
            initialized) the spread term vanishes and the loss reduces to
            elementwise MAE.

    Returns:
        Elementwise tensor, same shape as ``y_pred``; backward populates
        ``y_pred.grad`` correctly.
    """
    if not dist.is_initialized():
        return (y_pred - y).abs()

    group = group if group is not None else dist.group.WORLD
    K = dist.get_world_size(group)
    if K == 1:
        return (y_pred - y).abs()

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

    return (skill - spread / (K - 1)) / K


def ring_crps_loss(y_pred, y, group=None):
    """Fair CRPS via ring communication, reduced to a scalar.

    ``ring_crps_elementwise(...).mean()`` — see that function for the member
    exchange, the local formula, and the gradient-correctness argument.

    Returns:
        Scalar loss; backward populates ``y_pred.grad`` correctly.
    """
    return ring_crps_elementwise(y_pred, y, group=group).mean()


class RingCRPSLoss(torch.nn.Module):
    """Criterion-compatible wrapper around the ring CRPS.

    Matches the trainer call convention criterion(y, y_pred): target first,
    prediction second.

    ``reduction`` follows the PyTorch convention: ``"mean"`` (default) returns
    a scalar; ``"none"`` returns the elementwise contribution, which is what
    the config loaders pass and what BaseLoss requires — the trainer's
    ``.mean()`` (or BaseLoss's weighted per-variable mean) then reduces it to
    the same scalar, with identical gradients (the ring scaling is linear, see
    ``ring_crps_elementwise``).

    Usable inside BaseLoss as a per-variable univariate loss
    (``loss: {type: base, args: {training_loss: ring-crps, ...}}``): the
    ensemble dimension lives across data-parallel ranks, not in the tensor, so
    each variable is scored elementwise locally. Requires the same launch
    contract as flat ring-crps training (one member per dp rank, every dp rank
    fed the SAME batch — train_gen2 arranges this when it detects ring-crps).
    Note this exchanges K-1 buffers per scored variable per step. For
    validation, prefer a deterministic ``validation_loss`` (e.g. ``mae``):
    validation keeps dp dataset sharding, so ranks hold different samples and
    a cross-rank spread term would be meaningless.

    The process group is resolved lazily at first forward (the dp group is
    registered by distributed_model_wrapper_gen2, which runs before load_loss).
    """

    # Ensemble CRPS across ranks, elementwise in the local tensor — BaseLoss
    # checks this attribute to exempt ring-crps from the univariate CRPS ban.
    supports_elementwise = True

    def __init__(self, reduction="mean", group=None):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"RingCRPSLoss reduction must be 'mean' or 'none'; got {reduction!r}")
        self.reduction = reduction
        self._group = group
        self._group_resolved = group is not None

    def forward(self, target, pred):
        if not self._group_resolved:
            from credit.parallel.mesh import get_dp_group

            self._group = get_dp_group()
            self._group_resolved = True
        elementwise = ring_crps_elementwise(pred, target, group=self._group)
        return elementwise if self.reduction == "none" else elementwise.mean()
