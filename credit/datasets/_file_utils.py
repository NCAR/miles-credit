"""
_file_utils.py
--------------
Shared file-mapping helpers for ERA5 and MRMS dataset classes.

Provides strftime-based filename parsing and binary-search timestamp-to-file
lookup, supporting any temporal file granularity (annual, monthly, daily, etc.).
"""

from __future__ import annotations

import bisect
import os
import re
from datetime import datetime as dt_cls

import pandas as pd


# Maps strftime format codes to non-capturing regex fragments
_STRFTIME_TO_REGEX: dict[str, str] = {
    "%Y": r"\d{4}",
    "%y": r"\d{2}",
    "%m": r"\d{2}",
    "%d": r"\d{2}",
    "%H": r"\d{2}",
    "%M": r"\d{2}",
    "%S": r"\d{2}",
    "%j": r"\d{3}",
}

# Ordered finest → coarsest; first match wins
_STRFTIME_TO_FREQ: list[tuple[str, str]] = [
    ("%S", "s"),
    ("%M", "min"),
    ("%H", "h"),
    ("%j", "D"),
    ("%d", "D"),
    ("%m", "M"),
]


def _strftime_to_regex(fmt: str) -> re.Pattern:
    """Convert a strftime format string to a compiled regex.

    The returned pattern matches the date substring in a filename; use
    ``m.group(0)`` together with the original *fmt* and ``strptime`` to
    recover the datetime.

    Args:
        fmt: strftime format string (e.g. ``"%Y"``, ``"%Y%m%d-%H%M%S"``).

    Returns:
        Compiled regex pattern matching the date portion of a filename.
    """
    pattern = re.escape(fmt)
    for code, repl in _STRFTIME_TO_REGEX.items():
        pattern = pattern.replace(re.escape(code), repl)
    return re.compile(pattern)


def _infer_period_freq(fmt: str) -> str:
    """Return the finest ``pd.Period`` frequency implied by a strftime format.

    Args:
        fmt: strftime format string.

    Returns:
        pd.Period frequency string (e.g. ``"h"``, ``"D"``, ``"M"``, ``"Y"``).
    """
    for code, freq in _STRFTIME_TO_FREQ:
        if code in fmt:
            return freq
    return "Y"  # annual default


def _map_files(
    file_list: list[str],
    time_fmt: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Build a sorted list of ``(start, end, path)`` intervals.

    For a single file the interval covers all representable time so no
    date parsing is attempted. For multiple files, *time_fmt* (a strftime
    format string) is used to extract the date from each filename's
    basename; ``pd.Period`` then determines the exact coverage window.

    Args:
        file_list: Sorted list of file paths returned by glob.
        time_fmt: strftime format string, e.g. ``"%Y"``, ``"%Y%m%d-%H%M%S"``.

    Returns:
        List of ``(start, end, path)`` tuples sorted by start time.

    Raises:
        ValueError: If *time_fmt* does not match the basename of any file
            in *file_list*.
    """
    if len(file_list) == 1:
        return [(pd.Timestamp.min, pd.Timestamp.max, file_list[0])]

    pattern = _strftime_to_regex(time_fmt)
    freq = _infer_period_freq(time_fmt)

    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for f in file_list:
        basename = os.path.basename(f)
        m = pattern.search(basename)
        if m is None:
            raise ValueError(
                f"filename_time_format '{time_fmt}' did not match "
                f"filename '{basename}'. Verify that the format matches "
                "the date portion of your filenames."
            )
        parsed = dt_cls.strptime(m.group(0), time_fmt)
        period = pd.Period(parsed, freq)
        intervals.append((period.start_time, period.end_time, f))

    return sorted(intervals, key=lambda x: x[0])


def _find_file(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    t: pd.Timestamp,
) -> str:
    """Binary-search for the file whose interval covers *t*.

    Args:
        intervals: Sorted list of ``(start, end, path)`` tuples.
        t: Timestamp to look up.

    Returns:
        Path to the file covering *t*.

    Raises:
        KeyError: If no interval covers *t*.
    """
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(starts, t) - 1
    if idx >= 0 and t <= intervals[idx][1]:
        return intervals[idx][2]
    raise KeyError(f"No file found covering timestamp {t}. Check that your data files span the requested time range.")
