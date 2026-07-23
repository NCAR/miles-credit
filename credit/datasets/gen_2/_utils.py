"""
_utils.py
--------------
Shared file-mapping and calendar helpers for CREDIT dataset classes.

Provides strftime-based filename parsing, binary-search timestamp-to-file
lookup (any temporal file granularity: annual, monthly, daily, etc.), and
the calendar layer that lets the gen2 pipeline run on non-standard CF
calendars (e.g. noleap CESM output).

Calendar design (see gen2-calendar-plan): the master clock and every
timestamp flowing between sampler and dataset are *calendar-native* objects —
``pd.Timestamp`` for standard calendars, ``cftime.datetime`` otherwise.  The
sampler needs no changes because ``xr.CFTimeIndex`` supports the same
indexing/arithmetic API as ``pd.DatetimeIndex``.  For standard calendars every
helper below reduces exactly to the pre-existing pandas code path, so no
cftime objects are ever created and behavior is bit-identical.
"""

from __future__ import annotations

import bisect
import cftime
import logging
import re
from datetime import datetime as dt_cls

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Calendar helpers
#
# ``standard`` family calendars are handled with plain pandas objects (the
# identity path).  Non-standard CF calendars use cftime objects end-to-end.
# ``360_day`` is explicitly unsupported: its dates (e.g. Feb 30) cannot be
# converted field-wise to any other calendar, which breaks both cross-source
# label conversion and filename-based file lookup.

_STANDARD_CALENDARS = frozenset({"standard", "gregorian", "proleptic_gregorian"})
_UNSUPPORTED_CALENDARS = frozenset({"360_day"})
# cftime accepts these aliases; normalize so equality checks work
_CALENDAR_ALIASES = {"365_day": "noleap", "366_day": "all_leap"}
_EPOCH_US = "microseconds since 1970-01-01T00:00:00"


def normalize_calendar(calendar: str | None) -> str:
    """Lower-case a CF calendar name and resolve aliases; ``None`` → ``"standard"``."""
    if calendar is None:
        return "standard"
    cal = str(calendar).lower()
    return _CALENDAR_ALIASES.get(cal, cal)


def is_standard_calendar(calendar: str | None) -> bool:
    """True when *calendar* is handled by plain pandas datetimes."""
    return normalize_calendar(calendar) in _STANDARD_CALENDARS


def _check_supported(calendar: str) -> str:
    cal = normalize_calendar(calendar)
    if cal in _UNSUPPORTED_CALENDARS:
        raise NotImplementedError(
            f"Calendar '{calendar}' is not supported: its dates cannot be converted "
            "field-wise to other calendars (e.g. Feb 30), which the file-mapping and "
            "multi-source label logic rely on. Supported: standard family, julian, "
            "noleap/365_day, all_leap/366_day."
        )
    return cal


def to_calendar(t, calendar: str | None):
    """Convert a timestamp to *calendar* by date fields (year, month, day, ...).

    The conversion preserves the calendar *label* (2000-03-01 stays 2000-03-01),
    not elapsed time.  Raises ``ValueError`` loudly when the date does not exist
    in the target calendar (e.g. Gregorian Feb 29 → noleap): a silent snap to a
    neighboring date would corrupt training pairs.

    Args:
        t: ``pd.Timestamp``, ``cftime.datetime``, ``datetime64``, or parseable str.
        calendar: target CF calendar name (``None`` == standard).

    Returns:
        ``pd.Timestamp`` when *calendar* is standard, else ``cftime.datetime``.
    """
    if is_standard_calendar(calendar):
        if isinstance(t, cftime.datetime):
            try:
                return pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)
            except ValueError as exc:
                raise ValueError(
                    f"Date {t} (calendar '{t.calendar}') does not exist in the standard calendar."
                ) from exc
        return pd.Timestamp(t)

    cal = _check_supported(calendar)
    if isinstance(t, cftime.datetime):
        if normalize_calendar(t.calendar) == cal:
            return t
        ts = t
    else:
        ts = pd.Timestamp(t)
    try:
        return cftime.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond, calendar=cal)
    except ValueError as exc:
        raise ValueError(
            f"Date {ts} does not exist in calendar '{calendar}'. This usually means a "
            "timestamp was generated on a less restrictive calendar (e.g. a Gregorian "
            "Feb 29 requested from noleap data); check start/end datetimes and init times."
        ) from exc


def build_time_index(start, end, freq, calendar: str | None = "standard"):
    """Build the sampling clock for *calendar*.

    Standard calendars return exactly ``pd.date_range(start, end, freq=freq)``
    (the identity path); other calendars return an ``xr.CFTimeIndex``, on which
    positional indexing and ``+ pd.Timedelta`` arithmetic behave like a
    ``pd.DatetimeIndex`` — so samplers work unchanged.

    Args:
        start / end: range bounds (converted to *calendar* by date fields).
        freq: ``pd.Timedelta`` or pandas frequency string.
        calendar: CF calendar name.
    """
    if is_standard_calendar(calendar):
        return pd.date_range(start, end, freq=freq)

    cal = _check_supported(calendar)
    import xarray as xr  # deferred: keeps the standard path free of the import

    freq_str = pd.tseries.frequencies.to_offset(freq).freqstr if isinstance(freq, pd.Timedelta) else freq
    return xr.date_range(to_calendar(start, cal), to_calendar(end, cal), freq=freq_str, calendar=cal, use_cftime=True)


def encode_time(t) -> int:
    """Encode a timestamp as int nanoseconds since 1970-01-01 *in its own calendar*.

    For ``pd.Timestamp`` this is exactly ``int(t.value)`` (unix ns).  For cftime
    objects the epoch offset is computed in the timestamp's calendar, so
    differencing two encoded values always yields true elapsed model time (a
    noleap year is 365 days of nanoseconds).  The ``(int, calendar)`` pair is
    the full identity — decode with the same calendar (see ``decode_time``);
    datasets advertise the calendar via ``static_metadata``.
    """
    if isinstance(t, cftime.datetime):
        return int(round(float(cftime.date2num(t, _EPOCH_US, calendar=t.calendar)))) * 1000
    return int(pd.Timestamp(t).value)


def decode_time(value: int, calendar: str | None = "standard"):
    """Inverse of ``encode_time``: int ns-since-epoch (+ calendar) → timestamp object."""
    if is_standard_calendar(calendar):
        return pd.Timestamp(int(value))
    cal = _check_supported(calendar)
    return cftime.num2date(int(value) // 1000, _EPOCH_US, calendar=cal)


def most_restrictive_calendar(calendars) -> str:
    """Pick the master-clock calendar from per-source calendars.

    Standard-family calendars impose no restriction; a single non-standard
    calendar wins (its dates are a subset of the standard ones we support).
    Two *different* non-standard calendars cannot share one master clock.
    """
    non_standard = {normalize_calendar(c) for c in calendars if not is_standard_calendar(c)}
    if not non_standard:
        return "standard"
    if len(non_standard) > 1:
        raise ValueError(
            f"Sources use multiple non-standard calendars {sorted(non_standard)}; "
            "a single master clock cannot represent both. Split the run or convert one source."
        )
    return next(iter(non_standard))


def _time_label(t) -> tuple:
    """Calendar-agnostic identity of a timestamp: its date/time fields."""
    return (t.year, t.month, t.day, t.hour, t.minute, t.second)


def filter_index_by_labels(index, other):
    """Keep elements of *index* whose (Y,M,D,h,m,s) label occurs in *other*.

    Calendar-agnostic replacement for ``index.isin(other)`` when the two sides
    may be a mix of ``pd.DatetimeIndex`` and ``CFTimeIndex``: comparing across
    those types is always unequal element-wise, which would silently empty the
    master clock.
    """
    labels = {_time_label(t) for t in other}
    mask = [_time_label(t) in labels for t in index]
    return index[mask]


# Set of all recognised strftime codes — used for path-template detection
_STRFTIME_CODES: frozenset[str] = frozenset(
    {
        "%Y",
        "%y",
        "%m",
        "%d",
        "%H",
        "%M",
        "%S",
        "%j",
        "%f",
    }
)

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
    "%f": r"\d",  # single fractional-second digit
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


def _path_template_to_glob(template: str) -> str:
    """Replace strftime codes in *template* with ``*`` to produce a glob pattern.

    Args:
        template: Path string that may contain strftime codes, e.g.
            ``"/data/%Y/%m/era5_*.nc"``.

    Returns:
        Glob-compatible pattern, e.g. ``"/data/*/*/era5_*.nc"``.
    """
    result = template
    for code in _STRFTIME_CODES:
        result = result.replace(code, "*")
    return result


def _extract_time_fmt(template: str) -> str:
    """Extract the strftime format substring from a path template.

    Returns the slice of *template* from the first strftime code to the end of
    the last one, preserving any literal characters between them.

    Example::

        _extract_time_fmt("/data/%Y/%m/era5_*.nc")  # "%Y/%m"
        _extract_time_fmt("/data/era5_%Y%m%d.nc")   # "%Y%m%d"

    Args:
        template: Path template containing at least one strftime code.

    Returns:
        The strftime format string (suitable for ``strptime``).
    """
    first_pos = len(template)
    last_pos = 0
    for code in _STRFTIME_CODES:
        idx = 0
        while True:
            pos = template.find(code, idx)
            if pos == -1:
                break
            first_pos = min(first_pos, pos)
            last_pos = max(last_pos, pos + len(code))
            idx = pos + 1
    return template[first_pos:last_pos] if last_pos > 0 else template


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
    path_template: str | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Build a sorted list of ``(start, end, path)`` intervals.

    For a single file the interval covers all representable time so no
    date parsing is attempted. For multiple files, *time_fmt* (a strftime
    format string) is used to extract the date from each filename's
    basename; ``pd.Period`` then determines the exact coverage window.

    When *path_template* is supplied the regex is anchored to the position of
    the strftime codes within the full template, preventing false matches when
    literal digits appear before the date placeholder (e.g.
    ``branch_1980_%Y_data.zarr`` where a bare ``\\d{4}`` would match ``1980``
    instead of the actual year).

    Args:
        file_list: Sorted list of file paths returned by glob.
        time_fmt: strftime format string extracted from the path template,
            e.g. ``"%Y"``, ``"%Y/%m"``.
        path_template: Original path template containing the strftime codes
            (e.g. ``"/data/run_1980_%Y_output.zarr"``). When provided, the
            full template is used to build an anchored regex so the date is
            extracted from the correct position in each filename.

    Returns:
        List of ``(start, end, path)`` tuples sorted by start time.

    Raises:
        ValueError: If *time_fmt* does not match any file in *file_list*.
    """
    if len(file_list) == 1:
        return [(pd.Timestamp.min, pd.Timestamp.max, file_list[0])]

    if path_template is not None:
        # Build a date-regex string from the time_fmt (without compiling yet)
        date_pat = re.escape(time_fmt)
        for code, repl in _STRFTIME_TO_REGEX.items():
            date_pat = date_pat.replace(re.escape(code), repl)
        # Escape the full template and splice the date portion in as a named
        # capture group so the match is anchored to the right field.
        anchored = re.escape(path_template).replace(re.escape(time_fmt), f"(?P<date>{date_pat})")
        # Templates may also contain glob wildcards (e.g. "..._%Y*.zarr");
        # re.escape made them literal, so translate them to their regex
        # equivalents (glob * and ? never cross a path separator).
        anchored = anchored.replace(re.escape("*"), r"[^/]*").replace(re.escape("?"), r"[^/]")
        pattern = re.compile(anchored)
        group_key: str | int = "date"
    else:
        pattern = _strftime_to_regex(time_fmt)
        group_key = 0

    freq = _infer_period_freq(time_fmt)

    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for f in file_list:
        m = pattern.search(f)
        if m is None:
            raise ValueError(
                f"Time format '{time_fmt}' did not match path '{f}'. "
                "Verify that your path contains strftime codes covering the date portion."
            )
        parsed = dt_cls.strptime(m.group(group_key), time_fmt)
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
    # File coverage windows are date *labels* parsed from filenames, held as
    # pandas intervals; normalize cftime queries to the same label space.
    if isinstance(t, cftime.datetime):
        t = to_calendar(t, "standard")
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(starts, t) - 1
    if idx >= 0 and t <= intervals[idx][1]:
        return intervals[idx][2]
    raise KeyError(f"No file found covering timestamp {t}. Check that your data files span the requested time range.")


def _to_cftime(ts: pd.Timestamp, calendar: str) -> cftime.datetime:
    """Convert a pandas Timestamp to a cftime.datetime.

    Unlike ``to_calendar`` this always returns a ``cftime.datetime``, even for
    standard-family calendar names (needed when a file's time coordinate is
    cftime-typed regardless of calendar).

    Args:
        ts: Pandas Timestamp to convert.
        calendar: cftime calendar string read from the dataset
            (e.g. ``"noleap"``, ``"gregorian"``, ``"proleptic_gregorian"``).

    Returns:
        cftime.datetime with the specified calendar.
    """
    return cftime.datetime(
        ts.year,
        ts.month,
        ts.day,
        ts.hour,
        ts.minute,
        ts.second,
        calendar=calendar,
    )


def _start_s3_fs():
    """Lazily initialize an anonymous ``s3fs.S3FileSystem`` instance.

    Called automatically on the first ``__extract_field__`` (called within ``__getitem__``)
    invocation when ``mode`` is ``"remote"``. The filesystem object is cached in ``_fs``
    for re-use across later calls.

    """

    try:
        import s3fs
    except ImportError as exc:
        raise ImportError("s3fs is required for remote dataset access. Install it with: pip install s3fs") from exc
    fs_config = {
        "anon": True,
        "token": "anon",
        "default_block_size": 8**20,
    }
    return s3fs.S3FileSystem(**fs_config)


def _start_s3_obstore(s3_bucket_name: str):
    """Lazily initialize an anonymous ``obstore.store.S3S3Store`` instance.

    Called automatically on the first ``__extract_field__`` (called within ``__getitem__``)
    invocation when ``mode`` is ``"remote"``. The filesystem object is cached in ``_obstore``
    for re-use across later calls.

    """

    try:
        import obstore
    except ImportError as exc:
        raise ImportError("s3fs is required for remote dataset access. Install it with: pip install obstore") from exc

    # Skip signature is important to create an anonymous client!
    return obstore.store.S3Store(s3_bucket_name, skip_signature=True)
