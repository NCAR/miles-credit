"""Fail-safe spectral-norm power iteration: ``weight_u`` / ``weight_v`` never stay zero or non-finite (#576).

With ``use_spectral_norm: true`` every Conv/Linear is wrapped in ``torch.nn.utils.spectral_norm``, whose forward
pre-hook runs one in-place power iteration per training forward on the per-rank buffers ``weight_u`` / ``weight_v``.
In long FSDP2 + activation-checkpointing runs, one module's u and v were seen to become exactly zero on a subset of
ranks in a single forward (#576). sigma = u . W v is then 0, the normalized weight W / sigma is 0/0 = NaN, and
because ``F.normalize(0) = 0`` the zero vectors are a fixed point: every later forward on those ranks is NaN.

:func:`install` replaces every ``torch.nn.utils.spectral_norm`` hook in a model with a :class:`GuardedSpectralNorm`
(same slot in ``_forward_pre_hooks``, same ``name`` / ``dim`` / ``eps`` / ``n_power_iterations``), so module
structure, parameter and buffer names, and the checkpoint format are unchanged. Each guarded hook:

* **Keeps u and v healthy through the power iteration.** It copies u and v before the iteration; afterwards, if
  ``|u|`` or ``|v|`` is below ``min_norm`` or non-finite, it puts the copies back in place. ``F.normalize`` returns
  a unit vector unless its input norm is below ``eps`` (1e-12), so a healthy vector has norm 1 and a collapsed one
  has norm in [0, 1); ``min_norm`` (default 0.5) catches exact zeros and partial collapses (which would inflate
  W / sigma) alike, with headroom for bf16 rounding. The check and the restore are branch-free tensor ops (no host
  sync per module), and when nothing degenerates the outputs, sigma and buffers are bit-identical to torch's hook.
  A forward whose own ``W`` reads as zero still gives sigma = 0 for that forward (its output is non-finite), but u
  and v survive it, so the next forward is finite again.
* **Repairs degenerate state it starts from** (e.g. a checkpoint saved after a collapse): on its first forward, and
  on the first forward after any ``load_state_dict`` into the module, it checks u and v once; a degenerate vector is
  re-initialized (seeded normal draw, normalized: the same draw on every rank) and ``reinit_iters`` power iterations
  run on the current ``W``. This happens inside the forward, where FSDP2 has gathered ``W``, so no collective is
  needed.
* **Counts recoveries** in one device tensor shared by the model's hooks, in the original forward only: the
  activation-checkpoint recompute (``CheckpointImpl.NO_REENTRANT`` re-runs the forward, hooks included, during
  backward) restores u / v too but does not count (:func:`in_backward`).

No collective is added. :meth:`SNGuard.poll` reads the counters (one small device-to-host copy; the trainer calls it
once per optimizer step), logs a WARNING per module with the step, module, rank and the bad norms (the first
``warn_first`` recoveries of a module, then every ``warn_every``-th), and keeps per-epoch and run totals that the gen2
trainer writes as ``train_sn_recoveries`` / ``train_sn_reinits``.

Config (``trainer.sn_guard``; on by default, ``sn_guard: false`` turns it off)::

    sn_guard: {enabled: true, min_norm: 0.5, reinit_iters: 20, warn_first: 3, warn_every: 100}
"""

from __future__ import annotations

import logging
import socket
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

logger = logging.getLogger(__name__)

DEFAULTS = {"enabled": True, "min_norm": 0.5, "reinit_iters": 20, "warn_first": 3, "warn_every": 100}
SEED = 20261115


def guard_config(conf: dict | None) -> dict:
    """``trainer.sn_guard`` merged over :data:`DEFAULTS` (``True`` / ``False`` / ``None`` accepted for the block)."""
    raw = ((conf or {}).get("trainer") or {}).get("sn_guard", {})
    if raw is None or raw is True:
        raw = {}
    if raw is False:
        raw = {"enabled": False}
    return {**DEFAULTS, **raw}


def degenerate(t: torch.Tensor, min_norm: float) -> torch.Tensor:
    """0-dim bool tensor: ``t`` is non-finite or its L2 norm is below ``min_norm`` (no host sync)."""
    n = torch.linalg.vector_norm(t.float())
    return ~torch.isfinite(n) | (n < min_norm)


def in_backward() -> bool:
    """True while autograd runs a backward pass on this thread, i.e. inside an activation-checkpoint recompute.

    A recovery is counted only in the original forward; the recompute still restores u / v if its own iteration
    degenerates but does not count it, so the counts (and the warnings driven by them) are per forward.
    """
    try:
        return torch._C._current_graph_task_id() != -1
    except AttributeError:  # older torch: count every pass
        return False


class GuardedSpectralNorm(SpectralNorm):
    """``torch.nn.utils.spectral_norm``'s hook with the u / v guard of the module docstring."""

    @classmethod
    def from_hook(cls, hook: SpectralNorm, guard: SNGuard, index: int, path: str) -> GuardedSpectralNorm:
        new = cls.__new__(cls)
        new.__dict__.update(hook.__dict__)  # name, dim, n_power_iterations, eps
        new.guard, new.index, new.path = guard, index, path
        new.checked = False
        return new

    def _power(self, weight_mat, u, v, iters):
        for _ in range(iters):
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)

    @torch.no_grad()
    def _repair(self, weight_mat, u, v) -> None:
        """Re-initialize a degenerate u / v and converge them on the current ``W`` (first forward after setup/load)."""
        g = self.guard
        bad_u, bad_v = bool(degenerate(u, g.min_norm)), bool(degenerate(v, g.min_norm))
        self.checked = True
        if not (bad_u or bad_v):
            return
        nu, nv = float(torch.linalg.vector_norm(u.float())), float(torch.linalg.vector_norm(v.float()))
        gen = torch.Generator(device=u.device).manual_seed(SEED + self.index)
        for t, bad in ((u, bad_u), (v, bad_v)):
            if bad:
                draw = torch.randn(t.shape, generator=gen, device=t.device, dtype=torch.float32)
                t.copy_(F.normalize(draw, dim=0, eps=self.eps).to(t.dtype))
        self._power(weight_mat, u, v, g.reinit_iters)
        g.reinits[self.index] += 1
        logger.warning(
            "sn_guard: rank %d host %s step %s: re-initialized degenerate spectral-norm u/v of %s "
            "(|u| %.4g, |v| %.4g) with %d power iterations on the current weight",
            g.rank,
            socket.gethostname(),
            g.step,
            self.path,
            nu,
            nv,
            g.reinit_iters,
        )

    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)
        if not self.checked:
            self._repair(weight_mat, u, v)
        if do_power_iteration:
            with torch.no_grad():
                u0, v0 = u.clone(), v.clone()
                self._power(weight_mat, u, v, self.n_power_iterations)
                g = self.guard
                g.to(u.device)
                bad = degenerate(u, g.min_norm) | degenerate(v, g.min_norm)
                if not in_backward():  # the activation-checkpoint recompute restores but does not count
                    i = self.index
                    g.bad_norms[i, 0] = torch.where(bad, torch.linalg.vector_norm(u.float()), g.bad_norms[i, 0])
                    g.bad_norms[i, 1] = torch.where(bad, torch.linalg.vector_norm(v.float()), g.bad_norms[i, 1])
                    g.counts[i] += bad.to(g.counts.dtype)
                u.copy_(torch.where(bad, u0, u))
                v.copy_(torch.where(bad, v0, v))
                if self.n_power_iterations > 0:  # as torch: use clones so a second forward can backprop
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return weight / sigma


def _uncheck(module, *args, **kwargs):
    """``load_state_dict`` pre-hook: re-check the module's u / v on its next forward."""
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, GuardedSpectralNorm):
            hook.checked = False


class SNGuard:
    """The model's guarded hooks, their shared recovery counters, and the per-step / per-epoch bookkeeping."""

    def __init__(self, hooks: list[tuple[str, Any, SpectralNorm]], cfg: dict, rank: int = 0):
        self.min_norm = float(cfg["min_norm"])
        self.reinit_iters = int(cfg["reinit_iters"])
        self.warn_first, self.warn_every = int(cfg["warn_first"]), max(int(cfg["warn_every"]), 1)
        self.rank, self.step = rank, None
        n = len(hooks)
        device = next((getattr(m, h.name + "_u").device for _, m, h in hooks), torch.device("cpu"))
        self.counts = torch.zeros(n, dtype=torch.int64, device=device)
        self.bad_norms = torch.zeros(n, 2, dtype=torch.float32, device=device)
        self.reinits = [0] * n
        self.paths = [p for p, _, _ in hooks]
        self.seen = [0] * n  # recoveries already polled, per module
        self.epoch_recoveries = self.total_recoveries = 0
        self.epoch_reinits_base = 0
        self.hooks: list[GuardedSpectralNorm] = []
        for i, (path, module, hook) in enumerate(hooks):
            new = GuardedSpectralNorm.from_hook(hook, self, i, path)
            for k, h in list(module._forward_pre_hooks.items()):
                if h is hook:
                    module._forward_pre_hooks[k] = new  # same slot: the hook order is unchanged
            module._register_load_state_dict_pre_hook(_uncheck, with_module=True)
            self.hooks.append(new)

    def __len__(self):
        return len(self.hooks)

    def to(self, device) -> None:
        """Keep the counters on the device the hooks run on (the model may move after :func:`install`)."""
        if self.counts.device != device:
            self.counts, self.bad_norms = self.counts.to(device), self.bad_norms.to(device)

    def poll(self, step: int | None = None) -> int:
        """Read the recovery counters (one device-to-host copy), log new recoveries, return how many were new."""
        if step is not None:
            self.step = step
        if not self.hooks:
            return 0
        counts = self.counts.tolist()
        new_total = 0
        norms = None
        for i, c in enumerate(counts):
            new = c - self.seen[i]
            if new <= 0:
                continue
            if norms is None:
                norms = self.bad_norms.tolist()
            for k in range(self.seen[i] + 1, c + 1):
                if k <= self.warn_first or k % self.warn_every == 0:
                    logger.warning(
                        "sn_guard: rank %d host %s step %s: spectral-norm u/v of %s degenerated in the power "
                        "iteration (|u| %.4g, |v| %.4g); kept the previous u/v (recovery %d of this module)",
                        self.rank,
                        socket.gethostname(),
                        self.step,
                        self.paths[i],
                        norms[i][0],
                        norms[i][1],
                        k,
                    )
                    break
            self.seen[i] = c
            new_total += new
        self.epoch_recoveries += new_total
        self.total_recoveries += new_total
        return new_total

    @property
    def total_reinits(self) -> int:
        return sum(self.reinits)

    def epoch_results(self) -> dict[str, list[float]]:
        """This rank's recoveries and re-initializations in the epoch; resets the per-epoch counts."""
        self.poll()
        reinits = self.total_reinits - self.epoch_reinits_base
        out = {"train_sn_recoveries": [float(self.epoch_recoveries)], "train_sn_reinits": [float(reinits)]}
        if self.epoch_recoveries or reinits:
            logger.warning(
                "sn_guard: rank %d host %s: %d spectral-norm u/v recoveries and %d re-initializations this epoch "
                "(%d recoveries in this job)",
                self.rank,
                socket.gethostname(),
                self.epoch_recoveries,
                reinits,
                self.total_recoveries,
            )
        self.epoch_recoveries, self.epoch_reinits_base = 0, self.total_reinits
        return out

    def summary(self) -> str:
        return (
            f"{self.epoch_recoveries} this epoch, {self.total_recoveries} in this job; "
            f"re-initialized {self.total_reinits}"
        )


def spectral_norm_hooks(model) -> list[tuple[str, Any, SpectralNorm]]:
    """``(module path, module, hook)`` for every unguarded ``torch.nn.utils.spectral_norm`` hook in ``model``."""
    out = []
    for path, m in model.named_modules():
        for h in m._forward_pre_hooks.values():
            if isinstance(h, SpectralNorm) and not isinstance(h, GuardedSpectralNorm):
                out.append((path, m, h))
    return out


def install(model, conf: dict | None = None, rank: int = 0) -> SNGuard | None:
    """Guard every spectral-norm hook of ``model`` (module docstring); ``None`` when disabled or there are none."""
    cfg = guard_config(conf)
    if not cfg["enabled"] or model is None:
        return None
    hooks = spectral_norm_hooks(model)
    if not hooks:
        return None
    guard = SNGuard(hooks, cfg, rank)
    if rank == 0:
        logger.info(
            "sn_guard: active on %d spectral-norm hooks (min |u|, |v| %.3g; %d power iterations on re-initialization)",
            len(guard),
            guard.min_norm,
            guard.reinit_iters,
        )
    return guard
