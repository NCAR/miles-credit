"""Lightweight CLI-argument helpers shared by ``credit rollout`` and ``rollout_gen2.py``.

Kept free of torch/model imports so the ``credit`` parser can validate flags
before the slow rollout imports, and so the override logic is unit-testable on CPU.
"""

import argparse

import pandas as pd


def init_time_arg(value: str) -> str:
    """Argparse ``type=`` for ``--init-time``: parse ISO 8601 and normalize to naive UTC.

    Init times are naive UTC, matching the dataset time coordinates. A ``Z`` or
    ``+00:00`` suffix is accepted and stripped; any other UTC offset is rejected
    rather than silently shifted.

    Returns:
        The init time as a naive ISO 8601 string, e.g. ``2020-06-01T00:00:00``.
    """
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid init time {value!r}: expected ISO 8601 such as 2020-06-01T00 ({exc})"
        ) from exc
    if ts is pd.NaT:
        raise argparse.ArgumentTypeError(f"invalid init time {value!r}: expected ISO 8601 such as 2020-06-01T00")
    if ts.tzinfo is not None:
        if ts.utcoffset() != pd.Timedelta(0):
            raise argparse.ArgumentTypeError(
                f"init time {value!r} has a non-UTC offset; give the time in UTC without an offset"
            )
        ts = ts.tz_localize(None)
    return ts.isoformat()


def check_rollout_flags(run_mode: str | None, init_time: str | None) -> None:
    """Raise ``ValueError`` if ``--run-mode`` and ``--init-time`` contradict each other."""
    if init_time is not None and run_mode == "batch":
        raise ValueError(
            "--init-time runs a single forecast and conflicts with --run-mode batch; "
            "drop --run-mode or use --run-mode single."
        )


def apply_cli_overrides(inf_conf: dict, args: argparse.Namespace) -> dict:
    """Apply ``credit rollout`` / ``rollout_gen2.py`` CLI overrides to the ``inference:`` block in place.

    Only flags that were given (not ``None``) override the config. ``--init-time``
    implies ``run_mode: single``.

    Args:
        inf_conf: The config's ``inference`` dict (mutated in place).
        args: Parsed arguments with ``run_mode``, ``save_dir``, ``init_time``,
            ``mode`` and ``noise_scale`` attributes (missing attributes count as unset).

    Returns:
        The same ``inf_conf`` dict.
    """
    run_mode = getattr(args, "run_mode", None)
    init_time = getattr(args, "init_time", None)
    check_rollout_flags(run_mode, init_time)

    if run_mode is not None:
        inf_conf["run_mode"] = run_mode
    save_dir = getattr(args, "save_dir", None)
    if save_dir is not None:
        inf_conf["save_forecast"] = save_dir
    if init_time is not None:
        inf_conf.setdefault("single_forecast", {})["start_datetime"] = init_time
        inf_conf["run_mode"] = "single"
    mode = getattr(args, "mode", None)
    if mode is not None:
        inf_conf["mode"] = mode
    noise_scale = getattr(args, "noise_scale", None)
    if noise_scale is not None:
        inf_conf["noise_scale"] = noise_scale
    return inf_conf
