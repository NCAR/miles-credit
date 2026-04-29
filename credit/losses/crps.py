"""
Ensemble CRPS losses for distributed training.

Both losses assume one ensemble member per GPU under DDP.

- Gather / gather_tensor: all_gather with correct autograd (legacy, O(K) memory).
- ring_crps_loss: O(1)-memory ring-reduce CRPS (preferred).
"""

import torch
import torch.distributed as dist


class Gather(torch.autograd.Function):
    """all_gather with autograd — concatenates along dim 0, routes gradients back."""

    @staticmethod
    def forward(ctx, input):
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        gathered = [torch.zeros_like(input) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, input)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.chunk(ctx.world_size, dim=0)[ctx.rank]


def gather_tensor(tensor):
    """Gather tensor from all ranks preserving the autograd graph."""
    return Gather.apply(tensor)


def ring_crps_loss(y_pred, y, rank, world_size):
    """Fair CRPS via ring communication — O(1) extra memory, no all_gather.

    Each GPU holds 1 ensemble member. K-1 ring shifts pass one member buffer
    at a time so rank r accumulates |x_r - x_j| for every j != r without ever
    materialising the full K-member ensemble on a single device.

    Gradient correctness (no cross-rank backward needed):
        d(CRPS)/d(x_r) = sign(x_r-y)/K  -  sum_{j!=r} sign(x_r-x_j) / (K*(K-1))
    Both terms are computed entirely from rank r's local graph. DDP averaging
    (1/K) then gives the correct model-parameter gradient.

    Local loss (scaled so DDP avg = d(CRPS)/d(params)):
        loss_r = |y_pred_r - y|  -  sum_{j!=r} |y_pred_r - x_j| / (K-1)

    Args:
        y_pred:     (ens_per_gpu, C, T, H, W) — local predictions, requires_grad.
        y:          (1, C, T, H, W) — truth tensor.
        rank:       local rank index.
        world_size: K = total ensemble size (one member per GPU assumed).

    Returns:
        Scalar loss; backward populates y_pred.grad correctly.
    """
    K = world_size
    skill = (y_pred - y).abs()

    buf = y_pred.detach().contiguous()
    spread = torch.zeros_like(y_pred)
    for _ in range(K - 1):
        next_buf = torch.empty_like(buf)
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, buf, (rank + 1) % K),
                dist.P2POp(dist.irecv, next_buf, (rank - 1 + K) % K),
            ]
        )
        for req in reqs:
            req.wait()
        buf = next_buf
        spread = spread + (y_pred - buf).abs()

    return (skill - spread / (K - 1)).mean() / K
