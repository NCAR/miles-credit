"""Tests for credit.events.EventSink and the gen2 trainer hooks that feed it."""

import json
import logging
import os

import pytest

from credit import events


def _read(run_dir, job="j1"):
    path = os.path.join(run_dir, "events", f"{job}.proc.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("CREDIT_RUN_DIR", str(run_dir))
    monkeypatch.setenv("CREDIT_JOB_KEY", "j1")
    for var in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "PALS_RANKID", "SLURM_PROCID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(events, "_SINK", None)
    return str(run_dir)


def test_noop_without_run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RUN_DIR", raising=False)
    sink = events.EventSink()
    assert not sink.enabled
    sink.emit("alive")
    sink.train_progress(0, {"train_loss": [1.0]}, 1, 1)
    assert not list(tmp_path.iterdir())


def test_only_rank_zero_writes(run_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    sink = events.EventSink()
    assert not sink.enabled
    sink.emit("alive")
    assert _read(run_env) == []
    assert events.EventSink(rank=0).enabled


def test_envelope_and_nonfinite(run_env):
    sink = events.EventSink()
    sink.emit("train.epoch", {"epoch": 1, "row": {"train_loss": float("nan"), "lr": 1e-3}})
    (ev,) = _read(run_env)
    assert ev["v"] == 1 and ev["type"] == "train.epoch" and ev["ts"].endswith("Z")
    assert ev["src"] == {"writer": "proc", "job": "j1", "host": ev["src"]["host"], "rank": 0}
    assert ev["data"]["row"] == {"train_loss": "NaN", "lr": 1e-3}


def test_progress_is_throttled_with_window_stats(run_env, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(events.time, "monotonic", lambda: clock[0])
    sink = events.EventSink(conf={"events": {"progress_every_s": 5, "vars_every_s": 60}}, samples_per_batch=4)
    results = {"train_loss": [], "train_rmse": [], "train_rmse/ERA5/t": []}
    for batch in range(1, 11):
        results["train_loss"].append(float(batch))
        results["train_rmse"].append(0.5)
        results["train_rmse/ERA5/t"].append(float(batch))
        sink.train_progress(0, results, batch, 10, lr=0.1, grad_norm=2.5, display_metrics=("rmse", "acc"))
        clock[0] += 1.0
    progress = [e["data"] for e in _read(run_env) if e["type"] == "train.progress"]
    # first batch (nothing emitted yet), then every 5 s, then the last batch of the epoch
    assert [p["batch"] for p in progress] == [1, 6, 10]
    assert progress[1]["loss"] == {"mean": 4.0, "min": 2.0, "max": 6.0, "n": 5}
    assert progress[2]["loss"]["n"] == 4 and progress[2]["loss_epoch"] == 5.5
    assert progress[1]["metrics"] == {"rmse": {"mean": 0.5, "epoch": 0.5}}
    assert progress[1]["grad_norm"] == 2.5 and progress[1]["lr"] == 0.1
    assert progress[1]["samples_per_s"] == pytest.approx(5 * 4 / 5.0)
    kinds = [e["type"] for e in _read(run_env)]
    # vars: once at the start (with its schema) and on the last batch
    assert kinds.count("metrics.schema") == 1 and kinds.count("train.vars") == 2
    last_vars = [e["data"] for e in _read(run_env) if e["type"] == "train.vars"][-1]
    assert last_vars == {"epoch": 0, "batch": 10, "values": [5.5]}


def test_progress_window_resets_each_epoch(run_env):
    sink = events.EventSink(conf={"events": {"progress_every_s": 0}})
    sink.train_progress(0, {"train_loss": [1.0, 2.0]}, 2, 2)
    sink.train_progress(1, {"train_loss": [3.0]}, 1, 2)
    progress = [e["data"] for e in _read(run_env) if e["type"] == "train.progress"]
    assert progress[-1]["loss"] == {"mean": 3.0, "min": 3.0, "max": 3.0, "n": 1}


def test_io_errors_are_swallowed_and_logged_once(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "run"
    blocker.write_text("a file, not a directory")
    monkeypatch.setenv("CREDIT_RUN_DIR", str(blocker))
    sink = events.EventSink(rank=0)
    with caplog.at_level(logging.WARNING, logger="credit.events"):
        sink.emit("alive")
        sink.emit("alive")
    assert len([r for r in caplog.records if r.name == "credit.events"]) == 1


def test_init_process_started_log_handler_and_error(run_env):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        sink = events.init({"trainer": {"train_batch_size": 2}}, rank=0, world_size=4)
        assert sink.samples_per_batch == 8
        logging.getLogger("credit.something").warning("disk %s is slow", "/glade")
        logging.getLogger("credit.something").info("not forwarded")
        try:
            raise ValueError("boom")
        except ValueError as exc:
            sink.error(exc)
            sink.error(exc)  # only once
    finally:
        root.handlers[:] = before
        import sys

        sys.excepthook = sys.__excepthook__
    evs = _read(run_env)
    assert evs[0]["type"] == "process.started" and evs[0]["data"]["world_size"] == 4
    logs = [e["data"] for e in evs if e["type"] == "log"]
    assert logs == [{"level": "WARNING", "logger": "credit.something", "message": "disk /glade is slow"}]
    errors = [e["data"] for e in evs if e["type"] == "error"]
    assert len(errors) == 1 and errors[0]["exc_type"] == "ValueError" and "boom" in errors[0]["traceback_tail"]


def test_init_is_noop_without_env(monkeypatch):
    monkeypatch.delenv("CREDIT_RUN_DIR", raising=False)
    monkeypatch.setattr(events, "_SINK", None)
    handlers = list(logging.getLogger().handlers)
    sink = events.init({}, rank=0)
    assert not sink.enabled and logging.getLogger().handlers == handlers


def test_cadence_env_vars_and_config_override(run_env, monkeypatch):
    monkeypatch.setenv("CREDIT_EVENTS_PROGRESS_EVERY_S", "7")
    monkeypatch.setenv("CREDIT_EVENTS_ALIVE_EVERY_S", "30")
    sink = events.EventSink(conf={"events": {"progress_every_s": 2}})
    assert sink.cadence == {"progress_every_s": 2.0, "vars_every_s": 60.0, "alive_every_s": 30.0}


# ---------------------------------------------------------------------------
# Trainer hooks (BaseTrainer.fit / _log_batch_progress / _save_checkpoint)
# ---------------------------------------------------------------------------


def test_trainer_hooks_emit_progress_epoch_checkpoint_and_stop(run_env, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("credit.trainers.trainer_gen2")
    import tqdm
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import credit.trainers.base_trainer as bt

    monkeypatch.setenv("CREDIT_EVENTS_PROGRESS_EVERY_S", "3600")

    class Trainer(bt.BaseTrainer):
        def train_one_epoch(self, epoch, loader, optimizer, *args, **kwargs):
            results = {"train_loss": []}
            pbar = tqdm.tqdm(range(3), total=3, disable=True)
            for i in range(3):
                results["train_loss"].append(1.0 + i)
                self._grad_norm = torch.tensor(0.25)
                self._log_batch_progress(epoch, results, optimizer, pbar, phase="train")
            return results

        def validate(self, *args, **kwargs):
            return {"valid_loss": [0.5]}

    save_loc = tmp_path / "save"
    conf = {
        "save_loc": str(save_loc),
        "trainer": {"mode": "none", "epochs": 3, "num_epoch": 1, "amp": False, "use_scheduler": False},
    }
    monkeypatch.setattr(bt, "check_dataloader_startup", lambda *a, **k: None)
    monkeypatch.setattr(bt, "check_model_gpu_memory", lambda *a, **k: None)
    trainer = Trainer(nn.Linear(1, 1), rank=0, conf=conf)
    loader = DataLoader(TensorDataset(torch.zeros(1)))
    trainer.fit(
        conf,
        train_loader=loader,
        valid_loader=loader,
        optimizer=torch.optim.SGD(trainer.model.parameters(), lr=0.01),
        train_criterion=nn.MSELoss(),
        valid_criterion=nn.MSELoss(),
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        scheduler=None,
        metrics=None,
    )
    evs = _read(run_env)
    progress = [e["data"] for e in evs if e["type"] == "train.progress"]
    assert [p["batch"] for p in progress] == [1, 3]  # first, then the last batch of the epoch
    assert progress[-1]["batches"] == 3 and progress[-1]["grad_norm"] == 0.25 and progress[-1]["lr"] == 0.01
    (epoch_ev,) = [e["data"] for e in evs if e["type"] == "train.epoch"]
    assert epoch_ev["epoch"] == 0 and epoch_ev["row"]["train_loss"] == 2.0
    saved = {(e["data"]["kind"], os.path.basename(e["data"]["path"])) for e in evs if e["type"] == "checkpoint.saved"}
    assert saved == {("latest", "checkpoint.pt"), ("best", "best_checkpoint.pt")}
    (stopped,) = [e["data"] for e in evs if e["type"] == "train.stopped"]
    assert stopped == {"reason": "epochs_per_job", "epoch": 0}
    assert [p.name for p in save_loc.iterdir() if ".tmp" in p.name] == []  # CSV written via tmp + rename


def test_forked_copies_do_not_write(run_env):
    sink = events.EventSink()
    sink._pid = -1  # as seen from a forked DataLoader worker
    sink.emit("alive")
    assert _read(run_env) == []


def test_log_events_are_rate_limited(run_env):
    sink = events.EventSink()
    handler = events._LogHandler(sink, repeat_s=60, per_minute=5)
    log = logging.getLogger("credit.flood")
    log.addHandler(handler)
    log.propagate = False
    try:
        for _ in range(10):
            log.warning("same thing")
        for i in range(10):
            log.warning("distinct %d", i)
    finally:
        log.removeHandler(handler)
        log.propagate = True
    messages = [e["data"]["message"] for e in _read(run_env) if e["type"] == "log"]
    assert messages == ["same thing", "distinct 0", "distinct 1", "distinct 2", "distinct 3"]
