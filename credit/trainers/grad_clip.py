"""Adaptive gradient clipping for ``trainer.grad_max_norm: 'dynamic'``.

A fixed ``grad_max_norm`` needs a threshold tuned to the model, the loss scale
and the training stage. ``'dynamic'`` instead clips each step to a multiple of
the recent typical gradient norm::

    trainer:
      grad_max_norm: 'dynamic'
      dynamic_grad_clip:        # optional; defaults shown
        factor: 2.0             # clip when the global norm exceeds factor x EMA
        ema_decay: 0.99         # EMA of the (post-clip) global grad norm
        warmup_steps: 50        # steps that only accumulate the EMA, no clipping

Normal steps pass through untouched; a spike (e.g. an outlier sample) is scaled
back to ``factor x EMA``. The EMA is updated with the post-clip norm, so a spike
cannot loosen the threshold for the steps after it. The state is saved next to
the checkpoint (``grad_clip_state.json``) so a resumed job does not repeat the
warmup.
"""

import json
import logging
import math
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

STATE_FILENAME = "grad_clip_state.json"


def global_grad_norm(parameters, distributed: bool) -> torch.Tensor:
    """Global L2 norm of the gradients across all ranks.

    Squared norms are summed across ranks before the square root (summing the
    norms themselves mixes units). DTensor grads (FSDP2 / native TP) go through
    the mesh-aware ``total_grad_norm``, whose reduction is already global. Plain
    grads are all-reduced; when ranks hold identical copies (DDP) that scales the
    norm by a constant sqrt(world size), which is harmless for EMA-relative
    clipping because the threshold carries the same factor.
    """
    from torch.distributed.tensor import DTensor

    from credit.parallel.collectives import total_grad_norm

    plain, sharded = [], []
    for p in parameters:
        if p.grad is not None:
            (sharded if isinstance(p.grad, DTensor) else plain).append(p.grad.detach())
    sq_terms = []
    if plain:
        local_sq = torch.stack([g.norm(2).float() for g in plain]).square().sum()
        if distributed:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        sq_terms.append(local_sq)
    if sharded:
        sq_terms.append(total_grad_norm(sharded, 2.0).float().square())
    if not sq_terms:
        return torch.tensor(0.0)
    return torch.stack([t.reshape(()).to(sq_terms[0].device) for t in sq_terms]).sum().sqrt()


class AdaptiveGradClipper:
    """Clip gradients to ``factor`` times an EMA of the global gradient norm.

    Args:
        factor: clip threshold as a multiple of the EMA norm (> 1).
        ema_decay: decay of the norm EMA, in (0, 1).
        warmup_steps: initial steps that only feed the EMA and never clip.
    """

    def __init__(self, factor: float = 2.0, ema_decay: float = 0.99, warmup_steps: int = 50):
        if factor <= 1.0:
            raise ValueError(f"dynamic_grad_clip.factor must be > 1, got {factor}")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"dynamic_grad_clip.ema_decay must be in (0, 1), got {ema_decay}")
        self.factor = float(factor)
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.ema = 0.0
        self.steps = 0  # finite norms folded into the EMA

    @classmethod
    def from_config(cls, trainer_conf: dict) -> "AdaptiveGradClipper":
        return cls(**(trainer_conf.get("dynamic_grad_clip") or {}))

    def threshold(self) -> float:
        """Current clip threshold (inf while warming up)."""
        if self.steps < max(self.warmup_steps, 1):
            return math.inf
        return self.factor * self.ema / (1.0 - self.ema_decay**self.steps)

    def clip_(self, parameters, distributed: bool) -> float:
        """Scale the gradients in place if their global norm exceeds the threshold.

        Returns the global norm before clipping. A non-finite norm is left alone
        (and kept out of the EMA) so the usual NaN/inf handling still sees it.
        """
        parameters = list(parameters)
        norm = float(global_grad_norm(parameters, distributed))
        if not math.isfinite(norm):
            return norm
        threshold = self.threshold()
        clipped = norm
        if norm > threshold:
            coef = threshold / (norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.detach().mul_(coef)
            clipped = threshold
        self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * clipped
        self.steps += 1
        return norm

    def state_dict(self) -> dict:
        return {"ema": self.ema, "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.ema = float(state["ema"])
        self.steps = int(state["steps"])

    def save(self, save_loc: str) -> None:
        with open(os.path.join(save_loc, STATE_FILENAME), "w") as f:
            json.dump(self.state_dict(), f)

    def load(self, save_loc: str) -> bool:
        path = os.path.join(save_loc, STATE_FILENAME)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.load_state_dict(json.load(f))
        return True
