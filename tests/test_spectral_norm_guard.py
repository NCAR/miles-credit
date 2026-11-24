"""Fail-safe spectral-norm power iteration (credit/models/spectral_norm_guard.py, #576)."""

import logging

import pytest
import torch
from torch.nn.utils import spectral_norm

from credit.models import spectral_norm_guard as S

CONF = {"trainer": {}}


class _Lin(torch.nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.lin = spectral_norm(torch.nn.Linear(n, n))

    def forward(self, x):
        return self.lin(x)


def _net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        spectral_norm(torch.nn.Linear(6, 5)),
        torch.nn.GELU(),
        spectral_norm(torch.nn.Conv2d(5, 4, 3, padding=1)),
        spectral_norm(torch.nn.ConvTranspose2d(4, 3, 2, stride=2)),
    )


def _run(net, xs):
    outs = []
    net.train()
    for x in xs:
        h = net[1](net[0](x)).reshape(2, 5, 1, 1).expand(2, 5, 3, 3)
        outs.append(net[3](net[2](h)))
    net.eval()
    outs.append(net[0](xs[0]))
    return outs


def _sigma(m):
    W = m.weight_orig.detach().float().reshape(m.weight_orig.shape[0], -1)
    return float(m.weight_u.float() @ (W @ m.weight_v.float()))


def _fwd(m, n=8):
    return m(torch.randn(2, n, dtype=m.lin.weight_orig.dtype))


def test_install_swaps_every_hook_in_place_and_keeps_names():
    net = _net()
    keys = list(net.state_dict())
    g = S.install(net, CONF)
    assert len(g) == 3 and all(isinstance(h, S.GuardedSpectralNorm) for h in g.hooks)
    assert [h.dim for h in g.hooks] == [0, 0, 1]  # ConvTranspose keeps dim 1
    assert list(net.state_dict()) == keys  # checkpoint format unchanged
    assert S.install(net, CONF) is None  # nothing left to guard
    assert S.install(_net(), {"trainer": {"sn_guard": False}}) is None
    assert S.install(torch.nn.Linear(2, 2), CONF) is None and S.install(None, CONF) is None


def test_forward_is_bit_identical_when_nothing_degenerates():
    ref, net = _net(), _net()
    S.install(net, CONF)
    torch.manual_seed(1)
    xs = [torch.randn(2, 6) for _ in range(4)]
    for a, b in zip(_run(ref, xs), _run(net, xs), strict=True):
        assert torch.equal(a, b)
    for k, v in ref.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k


def test_all_zero_weight_for_one_forward_keeps_u_and_v(caplog):
    caplog.set_level(logging.WARNING)
    m = _Lin()
    g = S.install(m, CONF)
    m.train()
    for _ in range(30):
        _fwd(m)
    u0, v0, s0 = m.lin.weight_u.clone(), m.lin.weight_v.clone(), _sigma(m.lin)
    w = m.lin.weight_orig.detach().clone()
    with torch.no_grad():
        m.lin.weight_orig.zero_()
    assert not torch.isfinite(_fwd(m)).all()  # that forward's own W is zero: sigma 0 for it
    assert torch.equal(m.lin.weight_u, u0) and torch.equal(m.lin.weight_v, v0)
    with torch.no_grad():
        m.lin.weight_orig.copy_(w)
    assert torch.isfinite(_fwd(m)).all() and _sigma(m.lin) == pytest.approx(s0, rel=1e-4)
    assert g.poll(step=77) == 1
    assert "sn_guard: rank 0" in caplog.text and "step 77" in caplog.text and "|u| 0, |v| 0" in caplog.text
    assert g.epoch_results() == {"train_sn_recoveries": [1.0], "train_sn_reinits": [0.0]}
    assert g.epoch_recoveries == 0 and g.total_recoveries == 1


def test_without_the_guard_a_zero_weight_is_permanent():
    """The failure in #576: torch's hook leaves u = v = 0 and every later forward is NaN."""
    m = _Lin()
    m.train()
    _fwd(m)
    w = m.lin.weight_orig.detach().clone()
    with torch.no_grad():
        m.lin.weight_orig.zero_()
    _fwd(m)
    with torch.no_grad():
        m.lin.weight_orig.copy_(w)
    assert float(m.lin.weight_u.norm()) == 0 and not torch.isfinite(_fwd(m)).all()


def test_nonfinite_weight_does_not_poison_u():
    m = _Lin()
    S.install(m, CONF)
    m.train()
    _fwd(m)
    u0 = m.lin.weight_u.clone()
    with torch.no_grad():
        m.lin.weight_orig[0, 0] = float("nan")
    _fwd(m)
    assert torch.equal(m.lin.weight_u, u0)


def test_degenerate_checkpoint_is_reinitialized_on_load(caplog):
    caplog.set_level(logging.WARNING)
    m = _Lin()
    m.train()
    for _ in range(30):
        _fwd(m)
    good = _sigma(m.lin)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    sd["lin.weight_u"].zero_()
    sd["lin.weight_v"].fill_(float("nan"))
    fresh = _Lin()
    g = S.install(fresh, CONF)
    _fwd(fresh.train())  # first forward: checked once
    fresh.load_state_dict(sd)  # a checkpoint saved after the collapse: same keys, same format
    assert float(fresh.lin.weight_u.norm()) == 0
    assert torch.isfinite(_fwd(fresh)).all()
    assert float(fresh.lin.weight_u.norm()) == pytest.approx(1) and float(fresh.lin.weight_v.norm()) == pytest.approx(1)
    assert _sigma(fresh.lin) == pytest.approx(good, rel=1e-3)
    assert g.total_reinits == 1 and "re-initialized degenerate spectral-norm u/v of lin" in caplog.text
    assert g.epoch_results()["train_sn_reinits"] == [1.0]
    _fwd(fresh)
    assert g.total_reinits == 1  # once per load


def test_reinitialization_is_the_same_on_every_rank():
    """Seeded draw: two ranks repairing the same degenerate state with the same W end with the same u / v."""
    a, b = _Lin(), _Lin()
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        S.install(m, CONF)
        with torch.no_grad():
            m.lin.weight_u.zero_()
            m.lin.weight_v.zero_()
        m.eval()
        _fwd(m)
    assert torch.equal(a.lin.weight_u, b.lin.weight_u) and torch.equal(a.lin.weight_v, b.lin.weight_v)


def test_degenerate_state_is_repaired_in_eval_too():
    m = _Lin()
    S.install(m, CONF)
    with torch.no_grad():
        m.lin.weight_u.zero_()
    m.eval()
    assert torch.isfinite(_fwd(m)).all()


def test_bf16_u_and_v():
    """FSDP2 with a bf16 param dtype casts u / v to bf16 (credit.parallel.fsdp2._fix_spectral_norm_dtype)."""
    torch.manual_seed(0)
    ref, m = _Lin().bfloat16(), _Lin().bfloat16()
    m.load_state_dict(ref.state_dict())
    g = S.install(m, CONF)
    ref.train()
    m.train()
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    for _ in range(5):
        assert torch.equal(ref(x), m(x))  # bit-identical in bf16 too
    assert m.lin.weight_u.dtype == torch.bfloat16
    u0, w = m.lin.weight_u.clone(), m.lin.weight_orig.detach().clone()
    with torch.no_grad():
        m.lin.weight_orig.zero_()
    m(x)
    assert torch.equal(m.lin.weight_u, u0) and g.poll() == 1
    with torch.no_grad():
        m.lin.weight_orig.copy_(w)
        m.lin.weight_u.zero_()
    m.lin.load_state_dict(m.lin.state_dict())  # load triggers the re-check
    assert torch.isfinite(m(x)).all() and m.lin.weight_u.dtype == torch.bfloat16
    assert float(m.lin.weight_u.float().norm()) == pytest.approx(1, abs=1e-2) and g.total_reinits == 1


def test_activation_checkpoint_recompute_restores_but_counts_once(caplog):
    """CREDIT's non-reentrant checkpoint_wrapper re-runs the hook in backward: one recovery per forward, not two."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    caplog.set_level(logging.WARNING)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = spectral_norm(torch.nn.Linear(8, 8))

        def forward(self, x):
            return torch.nn.functional.gelu(self.lin(x))

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        checkpoint_wrapper(Block(), checkpoint_impl=CheckpointImpl.NO_REENTRANT), torch.nn.Linear(8, 1)
    )
    g = S.install(net, CONF)
    assert len(g) == 1 and S.in_backward() is False
    lin = net[0]._checkpoint_wrapped_module.lin
    net.train()
    seen = []
    real = S.in_backward
    S.in_backward = lambda: seen.append(real()) or seen[-1]
    try:
        net(torch.randn(4, 8, requires_grad=True)).sum().backward()  # forward + recompute
        assert seen == [False, True] and g.poll(1) == 0
        with torch.no_grad():
            lin.weight_orig.zero_()  # W reads as zero in the forward and in its recompute
        net(torch.randn(4, 8, requires_grad=True)).sum().backward()
    finally:
        S.in_backward = real
    assert g.counts.tolist() == [1]  # counting in the recompute too would give [2]
    assert g.poll(2) == 1 and caplog.text.count("degenerated in the power iteration") == 1
    assert float(lin.weight_u.norm()) == pytest.approx(1)  # the recompute restored as well


def test_a_degeneration_only_in_the_recompute_is_restored_but_not_counted():
    m = _Lin()
    g = S.install(m, CONF)
    m.train()
    _fwd(m)
    u0 = m.lin.weight_u.clone()
    with torch.no_grad():
        m.lin.weight_orig.zero_()
    real = S.in_backward
    S.in_backward = lambda: True
    try:
        _fwd(m)
    finally:
        S.in_backward = real
    assert torch.equal(m.lin.weight_u, u0) and g.counts.tolist() == [0]


def test_zero_weight_at_repair_does_not_lock_in_zeros():
    """A first forward that repairs while W reads as zero keeps the seeded draw, not u = v = 0."""
    m = _Lin()
    g = S.install(m, CONF)
    w = m.lin.weight_orig.detach().clone()
    with torch.no_grad():
        m.lin.weight_u.zero_()
        m.lin.weight_orig.zero_()
    m.train()
    _fwd(m)  # repair + one power iteration, both on a zero W
    assert float(m.lin.weight_u.norm()) == pytest.approx(1) and float(m.lin.weight_v.norm()) == pytest.approx(1)
    assert g.total_reinits == 1 and g.poll(1) == 1
    with torch.no_grad():
        m.lin.weight_orig.copy_(w)
    for step in range(2, 5):
        assert torch.isfinite(_fwd(m)).all()
        assert g.poll(step) == 0  # the counter stops climbing once W is back


def test_degenerate_u_without_a_load_is_replaced_by_the_draw():
    """u / v zeroed in place after the first forward (no load_state_dict): the seeded draw replaces them."""
    m = _Lin()
    g = S.install(m, CONF)
    m.train()
    _fwd(m)
    with torch.no_grad():
        m.lin.weight_u.zero_()
    assert torch.isfinite(_fwd(m)).all()
    du, _ = m.lin._forward_pre_hooks[next(iter(m.lin._forward_pre_hooks))]._draw(m.lin.weight_u, m.lin.weight_v)
    assert torch.equal(m.lin.weight_u, du) and g.poll(1) == 1
    assert torch.isfinite(_fwd(m)).all() and g.poll(2) == 0


def test_gradients_are_bit_identical_under_activation_checkpointing():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    def build():
        torch.manual_seed(0)
        return torch.nn.Sequential(
            checkpoint_wrapper(_Lin(), checkpoint_impl=CheckpointImpl.NO_REENTRANT), torch.nn.Linear(8, 1)
        )

    ref, net = build(), build()
    S.install(net, CONF)
    ref.train()
    net.train()
    torch.manual_seed(1)
    for _ in range(3):
        x = torch.randn(4, 8)
        for m in (ref, net):
            m.zero_grad()
            m(x).sum().backward()
        for (k, a), (_, b) in zip(ref.named_parameters(), net.named_parameters(), strict=True):
            assert torch.equal(a.grad, b.grad), k
    for k, v in ref.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k


def test_guard_config():
    assert S.guard_config({"trainer": {}})["enabled"] is True and S.guard_config(None)["enabled"] is True
    assert S.guard_config({"trainer": {"sn_guard": None}})["min_norm"] == 0.5
    assert S.guard_config({"trainer": {"sn_guard": {"min_norm": 0.25}}})["min_norm"] == 0.25
    assert S.guard_config({"trainer": {"sn_guard": False}})["enabled"] is False


def test_warnings_are_rate_limited(caplog):
    caplog.set_level(logging.WARNING)
    m = _Lin()
    g = S.install(m, {"trainer": {"sn_guard": {"warn_first": 2, "warn_every": 5}}})
    m.train()
    _fwd(m)
    with torch.no_grad():
        m.lin.weight_orig.zero_()
    for step in range(1, 11):
        _fwd(m)
        g.poll(step)
    assert g.total_recoveries == 10
    assert caplog.text.count("degenerated in the power iteration") == 4  # recoveries 1, 2, 5, 10


def _two_rank_worker(rank, port, out):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(0)  # same init on both ranks, as after an FSDP2 all-gather
        m = _Lin()
        g = S.install(m, CONF, rank)
        m.train()
        x = torch.ones(2, 8)
        for _ in range(3):
            m(x)
        w = m.lin.weight_orig.detach().clone()
        if rank == 1:  # rank 1 sees a zero W for one forward, rank 0 the real one
            with torch.no_grad():
                m.lin.weight_orig.zero_()
        y = m(x)
        bad = torch.tensor([0.0 if torch.isfinite(y).all() else 1.0])
        dist.all_reduce(bad)  # stands in for the training step's collectives: both ranks stay in step
        with torch.no_grad():
            m.lin.weight_orig.copy_(w)
        y = m(x)
        ok = torch.tensor([1.0 if torch.isfinite(y).all() else 0.0])
        dist.all_reduce(ok)
        g.poll(4)
        epoch = g.epoch_results()  # every rank calls it: summed over ranks
        out[rank] = {
            "epoch": epoch,
            "bad": float(bad),
            "ok": float(ok),
            "u_norm": float(m.lin.weight_u.norm()),
            "u": m.lin.weight_u.tolist(),
            "recoveries": g.total_recoveries,
        }
    finally:
        dist.destroy_process_group()


def test_two_ranks_the_collapsed_rank_recovers():
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_two_rank_worker, args=(port, out), nprocs=2, join=True)
    r0, r1 = out[0], out[1]
    assert r0["bad"] == 1.0  # one rank's forward was non-finite at the zero-W step
    assert r0["ok"] == 2.0  # and the next forward is finite on both ranks
    assert r1["recoveries"] == 1 and r0["recoveries"] == 0
    for r in (r0, r1):  # rank 0 writes training_log.csv: it must see rank 1's recovery
        assert r["epoch"] == {"train_sn_recoveries": [1.0], "train_sn_reinits": [0.0]}
    assert r1["u_norm"] == pytest.approx(1, abs=1e-5)
    # rank 1 kept its pre-collapse u (one power iteration behind rank 0), so the two agree to the iteration's step
    assert r1["u"] == pytest.approx(r0["u"], abs=1e-2)
