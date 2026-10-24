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
warmup. The norm does not depend on the GPU count (except under tensor
parallelism, where a resume on a different world size restarts the warmup), so
the saved EMA stays valid when a job is resumed on more or fewer GPUs.
"""

import json
import logging
import math
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

STATE_FILENAME = "grad_clip_state.json"


def global_grad_norm(parameters, distributed: bool, replicated: bool = False) -> torch.Tensor:
    """Global L2 norm of the gradients across all ranks.

    Squared norms are summed across ranks before the square root (summing the
    norms themselves mixes units). DTensor grads (FSDP2 / native TP) go through
    the mesh-aware ``total_grad_norm``, whose reduction is already global. Plain
    grads are all-reduced. ``replicated`` means every rank holds the same copy of
    them (DDP, domain parallelism), so the sum counted that copy world-size times
    and is divided back; otherwise (FSDP1 shards) the sum is the true norm. The
    all-reduce is kept even for identical copies so every rank makes the same
    clipping decision bit for bit.
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
            if replicated:
                local_sq = local_sq / dist.get_world_size()
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

    Attributes:
        replicated_grads: plain grads are identical on every rank (DDP, domain
            parallelism); see ``global_grad_norm``. Set by ``from_config``.
        world_invariant: the norm does not depend on the world size, so a saved
            EMA is reused on any GPU count. False for tensor-parallel layouts,
            whose plain grads are only partly replicated; a world-size change
            there restarts the warmup instead.
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
        self.replicated_grads = False
        self.world_invariant = True
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    @classmethod
    def from_config(
        cls, trainer_conf: dict, replicated_grads: bool = False, world_invariant: bool = True
    ) -> "AdaptiveGradClipper":
        """Build from ``trainer.dynamic_grad_clip``; the layout flags come from the trainer, not the YAML."""
        clipper = cls(**(trainer_conf.get("dynamic_grad_clip") or {}))
        clipper.replicated_grads = replicated_grads
        clipper.world_invariant = world_invariant
        return clipper

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
        norm = float(global_grad_norm(parameters, distributed, self.replicated_grads))
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
        return {"ema": self.ema, "steps": self.steps, "world_size": self.world_size}

    def load_state_dict(self, state: dict) -> bool:
        """Restore a saved EMA; returns False (state left fresh) when it does not apply here."""
        saved_world_size = state.get("world_size")
        if saved_world_size is None and self.replicated_grads and self.world_size > 1:
            # Saved before replicated norms were divided by the world size: the
            # EMA is sqrt(world size) too large and would loosen the threshold.
            logger.warning("Dynamic grad clipping: state predates world-size normalization; restarting warmup.")
            return False
        if not self.world_invariant and saved_world_size != self.world_size:
            logger.warning(
                "Dynamic grad clipping: state was saved on %s ranks and this run has %s; with tensor "
                "parallelism the grad norm depends on the layout, so the warmup restarts.",
                saved_world_size,
                self.world_size,
            )
            return False
        self.ema = float(state["ema"])
        self.steps = int(state["steps"])
        return True

    def save(self, save_loc: str) -> None:
        with open(os.path.join(save_loc, STATE_FILENAME), "w") as f:
            json.dump(self.state_dict(), f)

    def load(self, save_loc: str) -> bool:
        path = os.path.join(save_loc, STATE_FILENAME)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            return self.load_state_dict(json.load(f))
