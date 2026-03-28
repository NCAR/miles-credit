"""preflight.py
--------------
Pre-training checks that run before the epoch loop starts.

The goal is to catch silent hangs and OOM conditions early and emit
clear, actionable error messages rather than letting jobs hang on the
cluster for hours.

Public API
----------
estimate_dataloader_memory_gb(conf) -> float
    Pure function. Computes the expected peak DataLoader CPU RAM footprint
    from trainer and data config. No IO, fully testable.

check_dataloader_startup(conf, loader, rank, timeout_s) -> None
    Fetches one batch from *loader* with a timeout. Raises RuntimeError
    with a user-friendly message if the fetch hangs or if estimated
    memory looks dangerous.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_dataloader_memory_gb(conf: dict) -> float:
    """Estimate peak CPU RAM used by the DataLoader (GB).

    Formula::

        workers × prefetch_factor × batch_size × sample_bytes

    where sample_bytes = H × W × total_channels × 4 (float32).
    Input and target tensors are counted separately (×2).

    Args:
        conf: Full training config dict.

    Returns:
        Estimated peak DataLoader RAM in GB. Returns 0.0 if config is
        missing required keys (non-fatal — estimation is best-effort).
    """
    try:
        trainer_conf = conf.get("trainer", {})
        data_conf = conf.get("data", {})
        model_conf = conf.get("model", {})
        src = next(iter(data_conf.get("source", {}).values()), {})
        v = src.get("variables", {})
        prog = v.get("prognostic") or {}
        diag = v.get("diagnostic") or {}

        n_levels = len(src.get("levels", []))
        n_vars_3d = len(prog.get("vars_3D", []))
        n_vars_2d = len(prog.get("vars_2D", []))
        n_diag_2d = len(diag.get("vars_2D", []))
        total_ch = n_vars_3d * n_levels + n_vars_2d + n_diag_2d

        if total_ch == 0:
            return 0.0

        H = model_conf.get("image_height", 721)
        W = model_conf.get("image_width", 1440)

        bytes_per_sample = H * W * total_ch * 4  # float32
        bytes_per_sample *= 2  # input + target

        workers = trainer_conf.get("thread_workers", 4)
        prefetch = trainer_conf.get("prefetch_factor", 4)
        batch_size = trainer_conf.get("train_batch_size", 1)

        total_bytes = workers * prefetch * batch_size * bytes_per_sample
        return total_bytes / 1e9

    except Exception:
        return 0.0


def _available_ram_gb() -> float:
    """Return available system RAM in GB, or 0 if psutil is not installed."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return 0.0


# ---------------------------------------------------------------------------
# First-batch timeout check
# ---------------------------------------------------------------------------


def _fetch_one_batch(loader):
    """Return the first batch from *loader*, or raise on error."""
    it = iter(loader)
    return next(it)


def check_dataloader_startup(
    conf: dict,
    loader,
    rank: int = 0,
    timeout_s: float = 300.0,
) -> None:
    """Run pre-training data loading checks (rank-0 only).

    1. Logs estimated DataLoader memory and warns if it looks dangerous.
    2. Attempts to fetch the first batch within *timeout_s* seconds.
       Raises RuntimeError with a clear, actionable message if it hangs.

    Args:
        conf:      Full training config dict.
        loader:    Training DataLoader.
        rank:      Global rank. Checks only run on rank 0.
        timeout_s: Seconds to wait for the first batch before failing.
    """
    if rank != 0:
        return

    trainer_conf = conf.get("trainer", {})
    workers = trainer_conf.get("thread_workers", 4)
    prefetch = trainer_conf.get("prefetch_factor", 4)
    batch = trainer_conf.get("train_batch_size", 1)

    # ---- Memory estimate ----
    est_gb = estimate_dataloader_memory_gb(conf)
    avail_gb = _available_ram_gb()

    if est_gb > 0:
        logger.info(
            "DataLoader memory estimate: %.1f GB (workers=%d × prefetch=%d × batch=%d)",
            est_gb,
            workers,
            prefetch,
            batch,
        )
        if avail_gb > 0:
            pct = 100 * est_gb / avail_gb
            if pct > 80:
                logger.warning(
                    "DataLoader may use ~%.1f GB (%.0f%% of %.1f GB available RAM). "
                    "Training could hang or OOM. Consider reducing "
                    "thread_workers (currently %d) or prefetch_factor (currently %d) "
                    "in your trainer config.",
                    est_gb,
                    pct,
                    avail_gb,
                    workers,
                    prefetch,
                )
            elif pct > 50:
                logger.info(
                    "DataLoader memory (%.1f GB) is %.0f%% of available RAM (%.1f GB). "
                    "Looks OK, but watch memory if you increase workers or batch size.",
                    est_gb,
                    pct,
                    avail_gb,
                )

    # ---- First-batch timeout ----
    result: dict = {}
    exc: dict = {}

    def _fetch():
        try:
            result["batch"] = _fetch_one_batch(loader)
        except Exception as e:
            exc["e"] = e

    logger.info("Preflight: fetching first training batch (timeout=%.0fs) …", timeout_s)
    t0 = time.monotonic()
    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    elapsed = time.monotonic() - t0

    if thread.is_alive():
        # Build a helpful diagnostic message
        mem_hint = (
            f"Estimated DataLoader RAM: {est_gb:.1f} GB "
            f"({100 * est_gb / avail_gb:.0f}% of {avail_gb:.1f} GB available). "
            if est_gb > 0 and avail_gb > 0
            else ""
        )
        raise RuntimeError(
            f"\n\n{'=' * 70}\n"
            f"DATA LOADING HANG DETECTED — first batch took >{timeout_s:.0f}s\n"
            f"{'=' * 70}\n"
            f"{mem_hint}\n"
            f"Common causes and fixes:\n"
            f"  1. Too many DataLoader workers using too much RAM:\n"
            f"       thread_workers: {workers}  →  try thread_workers: 1\n"
            f"  2. prefetch_factor too high:\n"
            f"       prefetch_factor: {prefetch}  →  try prefetch_factor: 1\n"
            f"  3. Data files not accessible or very slow filesystem:\n"
            f"       Check your data paths are readable from this node.\n"
            f"  4. Workers forking inside a distributed job (common on Derecho):\n"
            f"       Try setting thread_workers: 0 to use the main process only.\n"
            f"{'=' * 70}\n"
        )

    if "e" in exc:
        raise exc["e"]

    logger.info("Preflight: first batch ready in %.1fs.", elapsed)
