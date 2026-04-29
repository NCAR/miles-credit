"""preflight.py
--------------
Pre-training checks that run before the epoch loop starts.

The goal is to catch silent hangs and OOM conditions early and emit
clear, actionable error messages rather than letting jobs hang on the
cluster for hours.

Public API
----------
estimate_dataloader_memory_gib(conf) -> float
    Pure function. Computes the expected peak DataLoader CPU RAM footprint
    from trainer and data config. No IO, fully testable.

check_dataloader_startup(conf, loader, rank, timeout_s) -> None
    Fetches one batch from *loader* with a timeout. Raises RuntimeError
    with a user-friendly message if the fetch hangs or if estimated
    memory looks dangerous.

check_model_gpu_memory(conf, model, optimizer, rank) -> None
    Runs a synthetic forward/backward/optimizer step and logs peak VRAM.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_dataloader_memory_gib(conf: dict) -> float:
    """Estimate peak CPU RAM used by the DataLoader (GiB).

    Formula::

        workers × prefetch_factor × batch_size × sample_bytes

    where sample_bytes = H × W × total_channels × 4 (float32).
    Input and target tensors are counted separately (×2).

    Args:
        conf: Full training config dict.

    Returns:
        Estimated peak DataLoader RAM in GiB. Returns 0.0 if config is
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

        bytes_per_sample = H * W * total_ch * 4  # float32 4 bytes
        bytes_per_sample *= 2  # input + target

        workers = trainer_conf.get("thread_workers", 4)
        prefetch = trainer_conf.get("prefetch_factor", 4)
        batch_size = trainer_conf.get("train_batch_size", 1)

        total_bytes = workers * prefetch * batch_size * bytes_per_sample
        return total_bytes / 2**30

    except Exception:
        return 0.0


def _available_ram_gib() -> float:
    """Return available system RAM in GiB, or 0 if psutil is not installed."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
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
    est_gib = estimate_dataloader_memory_gib(conf)
    avail_gib = _available_ram_gib()

    if est_gib > 0:
        logger.info(
            "DataLoader memory estimate: %.1f GiB (workers=%d × prefetch=%d × batch=%d)",
            est_gib,
            workers,
            prefetch,
            batch,
        )
        if avail_gib > 0:
            pct = 100 * est_gib / avail_gib
            if pct > 80:
                logger.warning(
                    "DataLoader may use ~%.1f GiB (%.0f%% of %.1f GiB available RAM). "
                    "Training could hang or OOM. Consider reducing "
                    "thread_workers (currently %d) or prefetch_factor (currently %d) "
                    "in your trainer config.",
                    est_gib,
                    pct,
                    avail_gib,
                    workers,
                    prefetch,
                )
            elif pct > 50:
                logger.info(
                    "DataLoader memory (%.1f GiB) is %.0f%% of available RAM (%.1f GiB). "
                    "Looks OK, but watch memory if you increase workers or batch size.",
                    est_gib,
                    pct,
                    avail_gib,
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
            f"Estimated DataLoader RAM: {est_gib:.1f} GiB "
            f"({100 * est_gib / avail_gib:.0f}% of {avail_gib:.1f} GiB available). "
            if est_gib > 0 and avail_gib > 0
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


# ---------------------------------------------------------------------------
# GPU memory check
# ---------------------------------------------------------------------------


def check_model_gpu_memory(conf: dict, model, optimizer, rank: int = 0) -> None:
    """Run a synthetic forward/backward/optimizer step and log peak VRAM.

    Creates a zero-filled batch of the expected input shape, runs it through
    the model, backprops, and steps the optimizer. Logs peak VRAM so users
    can verify their model fits on the target GPU before a long training run.

    Input channel count is inferred from the model config:
        frames × (channels × levels + surface_channels + input_only_channels)

    Skips silently if:
      - rank != 0 (only report from rank 0)
      - CUDA is not available
      - input channels cannot be inferred (returns 0)
      - any exception occurs during the synthetic pass

    Args:
        conf:      Full training config dict.
        model:     The model (possibly DDP/FSDP wrapped).
        optimizer: The optimizer (used to test a full optimizer step).
        rank:      Global rank. Check only runs on rank 0.
    """
    import torch

    if rank != 0 or not torch.cuda.is_available():
        return

    model_conf = conf.get("model", {})
    trainer_conf = conf.get("trainer", {})

    H = model_conf.get("image_height", 721)
    W = model_conf.get("image_width", 1440)
    B = trainer_conf.get("train_batch_size", 1)

    frames = model_conf.get("frames", 1)
    channels_3d = model_conf.get("channels", 0)
    levels = model_conf.get("levels", 0)
    surface_ch = model_conf.get("surface_channels", 0)
    input_only_ch = model_conf.get("input_only_channels", 0)
    C_in = frames * (channels_3d * levels + surface_ch + input_only_ch)

    if C_in == 0:
        logger.warning("check_model_gpu_memory: cannot infer input channels from model config; skipping GPU check.")
        return

    try:
        device = next(model.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)

        x = torch.zeros(B, C_in, H, W, device=device)
        y_hat = model(x)
        loss = y_hat.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
        total_mib = torch.cuda.get_device_properties(device).total_memory / 2**20
        logger.info(
            "GPU memory check: peak %.0f MiB / %.0f MiB (%.0f%%) for batch_size=%d, H=%d, W=%d, C_in=%d",
            peak_mib,
            total_mib,
            100 * peak_mib / total_mib,
            B,
            H,
            W,
            C_in,
        )

        del x, y_hat, loss
        torch.cuda.empty_cache()

    except Exception as e:
        logger.warning("check_model_gpu_memory: synthetic GPU pass failed (%s); skipping.", e)
