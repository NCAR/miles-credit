"""Tests for the ring-reduce ensemble CRPS loss (credit/losses/crps.py).

The multi-process tests run the real ring over gloo (CPU) and check the two
contracts that matter:
  1. The local loss value matches the documented formula
         loss_r = [ |x_r - y| - sum_{j!=r} |x_r - x_j| / (K-1) ].mean() / K
  2. The gradient w.r.t. each member equals the analytic fair-CRPS gradient
         d(CRPS)/d(x_r) = sign(x_r-y)/K - sum_{j!=r} sign(x_r-x_j)/(K*(K-1))
     (computed by autograd on a single-process reference with all members).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from credit.losses.crps import RingCRPSLoss, ring_crps_loss


def _members(K, shape=(2, 3, 8, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    members = [torch.randn(shape, generator=g) for _ in range(K)]
    y = torch.randn(shape, generator=g)
    return members, y


def _reference_fair_crps(members, y):
    """Fair CRPS via autograd-able reference: E|X-y| - sum_{i!=j}|x_i-x_j| / (2K(K-1))."""
    K = len(members)
    skill = torch.stack([(m - y).abs().mean() for m in members]).mean()
    spread = sum((members[i] - members[j]).abs().mean() for i in range(K) for j in range(K) if i != j)
    return skill - spread / (2 * K * (K - 1))


class TestSingleProcess:
    def test_no_dist_reduces_to_mae(self):
        members, y = _members(1)
        pred = members[0]
        loss = RingCRPSLoss()(y, pred)
        assert torch.allclose(loss, (pred - y).abs().mean())

    def test_gradient_no_dist(self):
        members, y = _members(1)
        pred = members[0].clone().requires_grad_(True)
        RingCRPSLoss()(y, pred).backward()
        expected = torch.sign(pred.detach() - y) / pred.numel()
        assert torch.allclose(pred.grad, expected)

    def test_registry(self):
        from credit.losses.base_losses import base_losses

        conf = {"loss": {"training_loss": "ring-crps"}}
        assert isinstance(base_losses(conf), RingCRPSLoss)

    def test_load_loss_rejects_weighted(self):
        from credit.losses import load_loss

        conf = {
            "data": {},
            "loss": {"training_loss": "ring-crps", "use_latitude_weights": True},
        }
        with pytest.raises(ValueError, match="ring-crps"):
            load_loss(conf)


class TestGen1Equivalence:
    """The gen1 ensemble trainer computes Gather(all members) -> KCRPS
    (biased=False, the fair kernel CRPS) per latitude slice. With the same
    members, its parameter gradient must equal the ring loss's: this test
    shows gen1's lat-sliced KCRPS == the fair-CRPS reference (value AND
    gradient), and test_ring_matches_reference shows ring gradient == the
    same reference. Values differ by design: the ring's detached spread
    counts each pair twice and carries a 1/K factor, so only the gradient
    (the training signal) is bit-comparable, not the logged loss.
    """

    def _gen1_loss(self, members, y):
        from credit.losses.kcrps import KCRPSLoss

        criterion = KCRPSLoss(reduction="none")  # biased=False default
        gathered = torch.cat(members, dim=0)  # Gather concatenates on dim 0
        lat_size = y.shape[3]
        total = 0
        for i in range(lat_size):  # the gen1 trainer's per-latitude loop
            pred_slice = gathered[:, :, :, i : i + 1].contiguous()
            y_slice = y[:, :, :, i : i + 1].contiguous()
            total = total + criterion(y_slice, pred_slice).mean() / lat_size
        return total

    def test_gen1_value_and_gradient_match_fair_crps(self):
        K = 3
        members, y = _members(K, shape=(1, 2, 4, 6, 8))
        members = [m.requires_grad_(True) for m in members]

        gen1_loss = self._gen1_loss(members, y)
        gen1_grads = torch.autograd.grad(gen1_loss, members)

        ref_members = [m.detach().clone().requires_grad_(True) for m in members]
        ref = _reference_fair_crps(ref_members, y)
        ref.backward()

        assert torch.allclose(gen1_loss, ref, atol=1e-6)
        for r in range(K):
            assert torch.allclose(gen1_grads[r], ref_members[r].grad, atol=1e-6), (
                f"gen1 gradient for member {r} differs from fair-CRPS reference"
            )


def _ring_worker(rank, K, init_file, value_out, grad_out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=K)
    try:
        members, y = _members(K)
        pred = members[rank].clone().requires_grad_(True)
        loss = ring_crps_loss(pred, y)
        loss.backward()
        value_out[rank] = loss.item()
        grad_out[rank] = pred.grad.clone()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("K", [2, 3])
def test_ring_matches_reference(tmp_path, K):
    init_file = str(tmp_path / "ring_init")
    manager = mp.Manager()
    value_out = manager.dict()
    grad_out = manager.dict()

    ctx = mp.spawn(_ring_worker, args=(K, init_file, value_out, grad_out), nprocs=K, join=True)
    del ctx

    members, y = _members(K)

    # 1. Per-rank loss value matches the documented local formula.
    for r in range(K):
        skill = (members[r] - y).abs()
        spread = sum((members[r] - members[j]).abs() for j in range(K) if j != r)
        expected = (skill - spread / (K - 1)).mean() / K
        assert abs(value_out[r] - expected.item()) < 1e-6, f"rank {r} loss value"

    # 2. Per-member gradient matches autograd on the fair-CRPS reference.
    ref_members = [m.clone().requires_grad_(True) for m in members]
    _reference_fair_crps(ref_members, y).backward()
    for r in range(K):
        assert torch.allclose(grad_out[r], ref_members[r].grad, atol=1e-6), (
            f"rank {r} gradient does not match fair-CRPS gradient"
        )


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    pytest.main([__file__, "-v"])
