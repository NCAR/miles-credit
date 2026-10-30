"""Structured run events from inside CREDIT (rank 0), for ``credit status`` and the MCP / Studio.

When a batch script made by ``credit submit`` (or ``credit.remote submit``)
runs CREDIT, it exports ``CREDIT_RUN_DIR`` and ``CREDIT_JOB_KEY``.  The
:class:`EventSink` then appends one JSON line per event to
``$CREDIT_RUN_DIR/events/<job_key>.proc.jsonl``: training progress windows,
per-variable running means, epoch rows, checkpoints, the stop reason, warnings,
a crash's traceback tail, and an ``alive`` heartbeat.  The event catalog is in
``cmd/credit-mcp/CONTRACTS.md`` §4.

Without ``CREDIT_RUN_DIR`` (an interactive ``credit train``, a notebook, the
tests) every call is a no-op.  The sink only ever writes on rank 0, flushes each
line, is safe to call from any thread, and never raises into training: an I/O
error is logged once and later events are dropped.

Usage from training code::

    from credit import events

    events.init(conf, rank=rank, world_size=world_size)  # once per process
    events.sink().train_progress(epoch, results_dict, batch, batches, lr=lr)
    events.sink().emit("checkpoint.saved", {"path": p, "epoch": epoch, "kind": "latest"})

Stdlib only.
"""

import logging
import math
import os
import socket
import sys
import threading
import time
import traceback

from credit import runs

logger = logging.getLogger(__name__)

_DEFAULT_CADENCE = {"progress_every_s": 5.0, "vars_every_s": 60.0, "alive_every_s": 60.0}
_RANK_ENV = ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "PALS_RANKID", "SLURM_PROCID")


def _env_rank() -> int:
    for var in _RANK_ENV:
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return 0


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _float(value) -> float | None:
    """``float(value)`` (one GPU sync for a tensor), or None if it is unset or not a number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class EventSink:
    """Appends CONTRACTS §4 events to ``events/<job_key>.proc.jsonl``; a no-op unless enabled.

    Args:
        run_dir: Run directory (default ``$CREDIT_RUN_DIR``); the sink is disabled without one.
        job_key: Job key (default ``$CREDIT_JOB_KEY``).
        rank: Global rank (default from the launcher's env vars); only rank 0 writes.
        conf: Optional CREDIT config; an ``events:`` block overrides the
            ``CREDIT_EVENTS_*_EVERY_S`` env vars for the cadences.
        samples_per_batch: Global samples per optimizer batch, for ``samples_per_s``.
    """

    def __init__(self, run_dir=None, job_key=None, rank=None, conf=None, samples_per_batch=None):
        run_dir = os.environ.get("CREDIT_RUN_DIR") if run_dir is None else run_dir
        self.rank = _env_rank() if rank is None else int(rank)
        self.enabled = bool(run_dir) and self.rank == 0
        self.job_key = job_key or os.environ.get("CREDIT_JOB_KEY") or f"pid-{os.getpid()}"
        self.path = os.path.join(run_dir, "events", f"{self.job_key}.proc.jsonl") if run_dir else None
        self.samples_per_batch = samples_per_batch
        self.cadence = dict(_DEFAULT_CADENCE)
        for key in self.cadence:
            env = os.environ.get(f"CREDIT_EVENTS_{key.upper()}")
            if env:
                try:
                    self.cadence[key] = float(env)
                except ValueError:
                    pass
        for key, value in ((conf or {}).get("events") or {}).items():
            if key in self.cadence and value is not None:
                self.cadence[key] = float(value)
        self._src = {"writer": "proc", "job": self.job_key, "host": socket.gethostname(), "rank": self.rank}
        # A forked DataLoader worker inherits this object; only the creating process writes.
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._file = None
        self._failed = False
        self._errored = False
        self._alive = None
        # train.progress window state
        self._epoch = None
        self._win = 0  # index into results["train_loss"] where the current window starts
        self._win_t = None
        self._last_progress = -math.inf
        self._last_vars = -math.inf
        self._schema = None

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def emit(self, type_: str, data: dict | None = None) -> None:
        """Append one event line; never raises."""
        if not self.enabled or os.getpid() != self._pid:
            return
        try:
            event = {"v": runs.EVENT_VERSION, "ts": runs.now_ts(), "type": type_, "src": self._src, "data": data or {}}
            line = runs.encode_line(event).decode("utf-8")
            with self._lock:
                if self._file is None:
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    self._file = open(self.path, "a", encoding="utf-8")
                self._file.write(line)
                self._file.flush()
        except Exception as exc:  # noqa: BLE001 — events must never break training
            if not self._failed:
                self._failed = True
                logger.warning("credit.events: cannot write %s (%s); dropping further events.", self.path, exc)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def process_started(self, world_size: int = 1) -> None:
        torch = sys.modules.get("torch")
        gpus = None
        if torch is not None:
            try:
                gpus = torch.cuda.device_count()
            except Exception:  # noqa: BLE001
                gpus = None
        try:
            from importlib.metadata import version

            credit_version = version("miles-credit")
        except Exception:  # noqa: BLE001
            credit_version = None
        self.emit(
            "process.started",
            {
                "pid": os.getpid(),
                "world_size": world_size,
                "credit_version": credit_version,
                "torch_version": getattr(torch, "__version__", None),
                "gpus": gpus,
            },
        )

    def error(self, exc: BaseException) -> None:
        """Emit ``error`` once, with the last lines of the traceback."""
        if self._errored:
            return
        self._errored = True
        tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).splitlines()[-30:]
        self.emit("error", {"exc_type": type(exc).__name__, "message": str(exc), "traceback_tail": "\n".join(tail)})

    def start_alive(self) -> None:
        """Start the daemon thread that emits ``alive`` every ``alive_every_s``."""
        if not self.enabled or self._alive is not None:
            return

        def beat():
            while True:
                time.sleep(self.cadence["alive_every_s"])
                self.emit("alive", {})

        self._alive = threading.Thread(target=beat, name="credit-events-alive", daemon=True)
        self._alive.start()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_progress(
        self, epoch, results: dict, batch: int, batches: int | None, lr=None, grad_norm=None, display_metrics=()
    ) -> None:
        """Emit ``train.progress`` (and ``train.vars``) from the trainer's per-batch ``results`` lists.

        Cheap when not due: one clock read.  Emits at most every
        ``progress_every_s`` / ``vars_every_s``, and always on the last batch of
        an epoch.  ``loss`` covers the batches since the previous emit;
        ``loss_epoch`` is the running mean tqdm shows.  ``grad_norm`` may be a
        tensor; it is converted only when an emit is due.
        """
        if not self.enabled:
            return
        now = time.monotonic()
        losses = results.get("train_loss") or []
        if epoch != self._epoch or len(losses) < self._win:
            self._epoch, self._win, self._win_t = epoch, 0, None
        last = bool(batches) and batch >= batches
        if last or now - self._last_progress >= self.cadence["progress_every_s"]:
            window = losses[self._win :]
            finite = [v for v in window if math.isfinite(v)]
            metrics = {}
            for name in display_metrics:
                vals = results.get(f"train_{name}")
                if vals:
                    metrics[name] = {"mean": _mean(vals[self._win :]), "epoch": _mean(vals)}
            rate = None
            if self._win_t is not None and self.samples_per_batch and now > self._win_t:
                rate = (len(losses) - self._win) * self.samples_per_batch / (now - self._win_t)
            self.emit(
                "train.progress",
                {
                    "epoch": epoch,
                    "batch": batch,
                    "batches": batches,
                    "loss": {
                        "mean": _mean(window),
                        "min": min(finite) if finite else None,
                        "max": max(finite) if finite else None,
                        "n": len(window),
                    },
                    "loss_epoch": _mean(losses),
                    "metrics": metrics,
                    "lr": _float(lr),
                    "grad_norm": _float(grad_norm),
                    "samples_per_s": rate,
                },
            )
            self._win, self._win_t, self._last_progress = len(losses), now, now
        if last or now - self._last_vars >= self.cadence["vars_every_s"]:
            columns = [k for k in results if k.startswith("train_") and "/" in k]
            if columns:
                if columns != self._schema:
                    self._schema = columns
                    self.emit("metrics.schema", {"columns": columns})
                values = [_mean(results[c]) for c in columns]
                self.emit("train.vars", {"epoch": epoch, "batch": batch, "values": values})
            self._last_vars = now


class _LogHandler(logging.Handler):
    """Forwards WARNING-and-above log records as ``log`` events, rate-limited.

    The same (logger, message) is forwarded at most once per ``repeat_s``, and
    at most ``per_minute`` records per minute overall, so a warning inside the
    batch loop cannot flood the events file.
    """

    def __init__(self, sink: EventSink, repeat_s: float = 60.0, per_minute: int = 60):
        super().__init__(level=logging.WARNING)
        self.sink, self.repeat_s, self.per_minute = sink, repeat_s, per_minute
        self._seen = {}
        self._window = (0.0, 0)

    def _allow(self, key: tuple) -> bool:
        now = time.monotonic()
        if now - self._seen.get(key, -math.inf) < self.repeat_s:
            return False
        start, count = self._window
        if now - start >= 60.0:
            start, count = now, 0
        if count >= self.per_minute:
            return False
        self._window = (start, count + 1)
        if len(self._seen) > 1000:
            self._seen.clear()
        self._seen[key] = now
        return True

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == __name__:
            return  # the sink's own complaints must not loop back into it
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            message = str(record.msg)
        if self._allow((record.name, message[:200])):
            self.sink.emit("log", {"level": record.levelname, "logger": record.name, "message": message[:4000]})


_SINK = None


def sink() -> EventSink:
    """Return the process-wide sink (created from the environment on first use)."""
    global _SINK
    if _SINK is None:
        _SINK = EventSink()
    return _SINK


def init(conf: dict | None = None, rank: int | None = None, world_size: int = 1) -> EventSink:
    """Set up the process-wide sink for a CREDIT entry point.

    Emits ``process.started``, forwards WARNING+ log records, starts the
    ``alive`` heartbeat, and records an uncaught exception as ``error`` before
    the process exits.  A no-op (returning a disabled sink) unless
    ``CREDIT_RUN_DIR`` is set and this is rank 0.
    """
    global _SINK
    trainer = (conf or {}).get("trainer") or {}
    batch_size = trainer.get("train_batch_size")
    samples = int(batch_size) * int(world_size or 1) if isinstance(batch_size, int) else None
    _SINK = EventSink(rank=rank, conf=conf, samples_per_batch=samples)
    if not _SINK.enabled:
        return _SINK
    _SINK.process_started(world_size)
    logging.getLogger().addHandler(_LogHandler(_SINK))
    _SINK.start_alive()
    previous = sys.excepthook

    def excepthook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            _SINK.error(exc)
        previous(exc_type, exc, tb)

    sys.excepthook = excepthook
    return _SINK
