"""``credit check`` — validate a config by resolving everything it names.

Static by default: every registry key is resolved, every block's ``args`` is
bound against its constructor signature, and the channel layout derived from
``data.source`` is cross-checked against the model geometry.  ``--deep`` also
instantiates the model, pre/postblocks, loss, and metrics, which needs the
scaler JSON to exist on disk.

Nothing here touches the network or reads training data, so it is safe to run
on a login node before submitting a job.

Findings carry a severity and, wherever the fix is unambiguous, the YAML to
apply.  Errors mean "this config will raise"; warnings mean "this will run but
probably not do what you meant".
"""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import os
import sys
from dataclasses import dataclass, field

import yaml

from ._common import _PBS_QUEUES

# Field types that reach the model as input but are never predicted — together
# they make up model.input_only_channels.
_INPUT_ONLY_FIELD_TYPES = ("dynamic_forcing", "static")

# Postblocks that synthesise a NEW variable rather than transforming existing
# ones.  BaseLoss scores these only if the same block also ran on the target
# twin — see _check_loss_pipeline.
_DIAGNOSTIC_POSTBLOCKS = frozenset({"geopotential_diagnostic", "mslp_diagnostic", "pressure_interp_diagnostic"})

# Postblocks that change a variable's units.  Whatever runs on the prediction
# must also run on the target twin or the loss compares mismatched units.
_UNIT_POSTBLOCKS = frozenset({"bridgescaler_transform", "exp_transform", "square_transform"})

_GEN1_TRAINERS = frozenset({"era5", "era5-gen1"})

_KNOWN_TOP_LEVEL = frozenset(
    {
        "save_loc",
        "seed",
        "data",
        "data_valid",
        "validation_data",
        "preblocks",
        "postblocks",
        "trainer",
        "model",
        "loss",
        "metrics",
        "predict",
        "inference",
        "pbs",
        "slurm",
        "custom_objects",
    }
)

_ERROR, _WARN, _INFO = "error", "warning", "info"


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass
class _Finding:
    severity: str
    where: str  # dotted config path, e.g. "postblocks.per_step.scaler"
    message: str
    fix: str | None = None


@dataclass
class _Report:
    config_path: str
    findings: list[_Finding] = field(default_factory=list)

    def error(self, where: str, message: str, fix: str | None = None) -> None:
        self.findings.append(_Finding(_ERROR, where, message, fix))

    def warn(self, where: str, message: str, fix: str | None = None) -> None:
        self.findings.append(_Finding(_WARN, where, message, fix))

    def info(self, where: str, message: str, fix: str | None = None) -> None:
        self.findings.append(_Finding(_INFO, where, message, fix))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def exit_code(self, strict: bool = False) -> int:
        if self.count(_ERROR):
            return 1
        if strict and self.count(_WARN):
            return 1
        return 0

    def to_dict(self) -> dict:
        return {
            "config": self.config_path,
            "errors": self.count(_ERROR),
            "warnings": self.count(_WARN),
            "findings": [
                {"severity": f.severity, "where": f.where, "message": f.message, "fix": f.fix} for f in self.findings
            ],
        }

    def render(self, stream=None) -> None:
        stream = stream or sys.stdout
        order = {_ERROR: 0, _WARN: 1, _INFO: 2}
        label = {_ERROR: "ERROR  ", _WARN: "WARNING", _INFO: "INFO   "}
        print(f"\ncredit check — {self.config_path}\n", file=stream)
        if not self.findings:
            print("  Everything resolves. No problems found.\n", file=stream)
            return
        for f in sorted(self.findings, key=lambda f: (order[f.severity], f.where)):
            print(f"  {label[f.severity]}  {f.where}", file=stream)
            for line in f.message.splitlines():
                print(f"            {line}", file=stream)
            if f.fix:
                fix_lines = f.fix.splitlines()
                print(f"       fix: {fix_lines[0]}", file=stream)
                for line in fix_lines[1:]:
                    print(f"            {line}", file=stream)
            print("", file=stream)
        n_err, n_warn = self.count(_ERROR), self.count(_WARN)
        verdict = "FAIL" if n_err else "OK"
        print(f"  {verdict}: {n_err} error(s), {n_warn} warning(s)\n", file=stream)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _suggest(key, candidates) -> str:
    """Return a ' Did you mean ...?' clause for a misspelled registry key."""
    matches = difflib.get_close_matches(str(key), [str(c) for c in candidates], n=1, cutoff=0.6)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _accepted_params(cls) -> list[str]:
    """Named (non-varargs) constructor parameters, minus ``self``."""
    sig = inspect.signature(cls.__init__)
    return [
        p.name for p in sig.parameters.values() if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _bind_error(cls, args: dict | None) -> str | None:
    """Return the TypeError text if ``args`` do not bind to ``cls.__init__``, else None."""
    try:
        inspect.signature(cls.__init__).bind(None, **(args or {}))
    except TypeError as exc:
        return str(exc)
    return None


def _get(conf: dict, *path, default=None):
    """Nested lookup that treats an explicit YAML ``null`` as missing."""
    cur = conf
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def _block_output_key(block: dict) -> str:
    """Which entry of the batch dict a postblock writes to.

    ``reconstruct`` names it ``out_key``; every transforming block names it
    ``key``.  Both default to ``y_processed``.
    """
    args = block.get("args") or {}
    return args.get("out_key") or args.get("key") or "y_processed"


def _awaits_preprocess(args: dict | None) -> str | None:
    """Return the scaler path if these args point at a scaler that is not fitted yet.

    Deep construction of such a block would fail with a FileNotFoundError that
    says nothing the missing-file warning has not already said, so callers skip
    it instead of emitting a second, noisier copy.
    """
    path = (args or {}).get("scaler_path")
    if isinstance(path, str) and path and not os.path.exists(os.path.expandvars(path)):
        return os.path.expandvars(path)
    return None


def _n_levels_for_source(src_conf: dict, model_levels) -> int | None:
    """Resolve a source's level count exactly as ChannelSchema.from_config does."""
    src_levels = src_conf.get("levels")
    return len(src_levels) if src_levels else model_levels


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: str, rep: _Report) -> dict | None:
    if not os.path.isfile(path):
        rep.error("config", f"File not found: {path}")
        return None
    try:
        # FullLoader (not safe_load) matches what train_gen2.py uses, so merge
        # keys and any other FullLoader-only syntax validate the same way here.
        with open(path) as handle:
            conf = yaml.load(handle, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        where = ""
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            where = f" at line {mark.line + 1}, column {mark.column + 1}"
        rep.error("config", f"YAML failed to parse{where}: {exc}")
        return None
    if not isinstance(conf, dict):
        rep.error("config", f"Top level must be a mapping, got {type(conf).__name__}.")
        return None
    return conf


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _is_data_fragment(conf: dict) -> bool:
    """True for a data-schema example that was never meant to train.

    Several configs under ``config/gen_2/examples`` document just the data
    section.  Reporting every missing training section on those is noise, so
    they get one note instead.
    """
    return "data" in conf and "model" not in conf and "trainer" not in conf


def _check_top_level(conf: dict, rep: _Report) -> None:
    if _is_data_fragment(conf):
        rep.info(
            "config",
            "Data-only fragment: no 'model' or 'trainer' section, so this cannot train as-is. "
            "Validating the data and pre/postblock sections only.",
        )
        return
    for key in ("save_loc", "data", "model", "trainer", "loss"):
        if key not in conf:
            rep.error(key, f"Required top-level section '{key}' is missing.")
    for key in conf:
        if key not in _KNOWN_TOP_LEVEL:
            rep.warn(key, f"Unrecognised top-level key '{key}' — it will be ignored.{_suggest(key, _KNOWN_TOP_LEVEL)}")


def _detect_generation(conf: dict, rep: _Report) -> str:
    trainer_type = _get(conf, "trainer", "type")
    if trainer_type in _GEN1_TRAINERS:
        rep.warn(
            "trainer.type",
            f"'{trainer_type}' is a gen1 trainer. credit check validates the gen2 pipeline; "
            "gen1-specific checks are skipped.",
            fix="Migrate with `credit convert -c <config>`, or keep gen1 and validate manually.",
        )
        return "gen1"
    return "gen2"


def _check_registry_keys(conf: dict, rep: _Report) -> None:
    """Resolve every ``type`` / ``dataset_type`` string against its registry."""
    from credit.datasets.gen_2.multi_source import _SOURCE_REGISTRY
    from credit.losses import _LOSS_REGISTRY
    from credit.models import _MODEL_REGISTRY
    from credit.trainers import _TRAINER_REGISTRY

    # The loss section has two spellings: new-style `type:` + `args:`, and the
    # legacy flat form with `training_loss:` and no `type:`.  load_loss accepts
    # both, so require one of them rather than `type` specifically.
    loss_conf = conf.get("loss") or {}
    loss_type = loss_conf.get("type") or loss_conf.get("training_loss")
    loss_where = "loss.type" if "type" in loss_conf else "loss.training_loss"

    for where, value, registry in (
        ("model.type", _get(conf, "model", "type"), _MODEL_REGISTRY),
        ("trainer.type", _get(conf, "trainer", "type"), _TRAINER_REGISTRY),
        (loss_where, loss_type, _LOSS_REGISTRY),
    ):
        if value is None:
            rep.error(where, f"'{where}' is not set.", fix=f"Add one of: {', '.join(sorted(registry))}")
        elif value not in registry:
            rep.error(
                where,
                f"Unknown value '{value}'.{_suggest(value, registry)}",
                fix=f"Valid: {', '.join(sorted(registry))}",
            )

    for name, src in (_get(conf, "data", "source", default={}) or {}).items():
        where = f"data.source.{name}.dataset_type"
        dtype = (src or {}).get("dataset_type")
        if dtype is None:
            rep.error(where, "Source has no 'dataset_type'.", fix=f"Valid: {', '.join(sorted(_SOURCE_REGISTRY))}")
        elif dtype not in _SOURCE_REGISTRY:
            rep.error(
                where,
                f"Unknown dataset_type '{dtype}'.{_suggest(dtype, _SOURCE_REGISTRY)}",
                fix=f"Valid: {', '.join(sorted(_SOURCE_REGISTRY))}",
            )


def _check_data_sources(conf: dict, rep: _Report) -> None:
    from typing import get_args

    from credit.datasets.gen_2.base_dataset import VALID_FIELD_TYPES

    valid_field_types = set(get_args(VALID_FIELD_TYPES))
    sources = _get(conf, "data", "source", default={}) or {}
    if not sources:
        rep.error("data.source", "No data sources defined.")
        return

    for name, src in sources.items():
        src = src or {}
        base = f"data.source.{name}"

        # An empty list is NOT "all levels" — the dataset checks for the key's
        # presence, so `levels: []` silently selects zero levels.
        if "levels" in src and isinstance(src["levels"], list) and not src["levels"]:
            rep.error(
                f"{base}.levels",
                "`levels: []` selects ZERO levels. The dataset tests whether the 'levels' key is "
                "present, not whether it is non-empty, so an empty list is not the same as omitting it.",
                fix="Remove the key to take the dataset default, or list the levels explicitly.",
            )

        variables = src.get("variables") or {}
        if not variables:
            rep.error(f"{base}.variables", "Source defines no variables.")
        for field_type in variables:
            if field_type not in valid_field_types:
                rep.error(
                    f"{base}.variables.{field_type}",
                    f"Unknown field type '{field_type}'.{_suggest(field_type, valid_field_types)}",
                    fix=f"Valid: {', '.join(sorted(valid_field_types))}",
                )
        for field_type, grp in variables.items():
            if grp is None:
                continue  # explicit null disables the field
            # _register_field raises unless the group names at least one of these.
            if not grp.get("vars_3D") and not grp.get("vars_2D"):
                rep.error(
                    f"{base}.variables.{field_type}",
                    "Field defines neither vars_3D nor vars_2D.",
                    fix="Add vars_3D and/or vars_2D, or set the field to null to disable it.",
                )
            # `path` is read by _get_file_source for local datasets, and dataset
            # subclasses may read further keys of their own — so anything
            # unrecognised is a warning, not an error.
            for dim_key in grp:
                if dim_key not in ("vars_3D", "vars_2D", "path"):
                    rep.warn(
                        f"{base}.variables.{field_type}.{dim_key}",
                        f"Unrecognised key '{dim_key}' — ignored unless this dataset type reads it."
                        f"{_suggest(dim_key, ['vars_3D', 'vars_2D', 'path'])}",
                    )
            # 3D variables need a vertical coordinate name to select against.
            if (grp.get("vars_3D") or []) and "level_coord" not in src:
                rep.warn(
                    f"{base}.level_coord",
                    "Source has 3D variables but no 'level_coord'; the dataset may not know which "
                    "coordinate to select levels from.",
                    fix="Add level_coord (e.g. 'level' for pressure levels, 'hybrid' for model levels).",
                )

    # gen2 is 1-indexed: forecast_len 0 is the gen1 spelling of a single step.
    fl = _get(conf, "data", "forecast_len")
    if fl == 0:
        rep.error(
            "data.forecast_len",
            "forecast_len: 0 is gen1 semantics. In gen2, 1 means a single step.",
            fix="forecast_len: 1",
        )
    elif fl is None:
        rep.error("data.forecast_len", "Not set; the gen2 trainer reads it with no default.", fix="forecast_len: 1")

    frames = _get(conf, "model", "frames")
    history_len = _get(conf, "data", "history_len", default=1)
    if frames is not None and history_len is not None and frames != history_len:
        rep.warn(
            "model.frames",
            f"model.frames ({frames}) != data.history_len ({history_len}); the model expects as many "
            "input time steps as the dataset supplies.",
            fix=f"model.frames: {history_len}",
        )


def _check_validation_data(conf: dict, rep: _Report) -> None:
    """load_dataset rejects a validation block that is missing keys ``data`` has."""
    valid = conf.get("validation_data") or conf.get("data_valid")
    if not valid:
        return
    if "data_valid" in conf and "validation_data" not in conf:
        rep.warn(
            "data_valid",
            "The gen2 loader reads 'validation_data'; 'data_valid' alone is ignored and validation "
            "will silently reuse the training range.",
            fix="Rename the section to 'validation_data'.",
        )
        return
    missing = sorted(set(conf.get("data") or {}) - set(valid) - {"source"})
    if missing:
        rep.error(
            "validation_data",
            f"Missing key(s) that data defines: {missing}. Partial validation blocks are rejected "
            "rather than silently merged.",
            fix=f"Add {missing} to validation_data, or delete the section to reuse data.",
        )


def _check_blocks(conf: dict, rep: _Report, deep: bool) -> None:
    """Resolve and signature-check every pre/postblock."""
    from credit.postblock import _POSTBLOCK_REGISTRY, _load_postblock_entry
    from credit.preblock import _PREBLOCK_REGISTRY, _load_preblock_entry

    specs = (
        ("preblocks", ("ic_only", "per_step"), _PREBLOCK_REGISTRY, _load_preblock_entry),
        ("postblocks", ("per_step", "post_rollout"), _POSTBLOCK_REGISTRY, _load_postblock_entry),
    )
    for top, valid_sections, registry, loader in specs:
        cfg = conf.get(top) or {}
        if not isinstance(cfg, dict):
            rep.error(top, f"Expected a mapping of sections, got {type(cfg).__name__}.")
            continue
        for section in cfg:
            if section not in valid_sections:
                rep.error(
                    f"{top}.{section}",
                    f"Unknown section '{section}'.{_suggest(section, valid_sections)} "
                    "(The old flat block format is no longer accepted.)",
                    fix=f"Move these blocks under one of: {', '.join(valid_sections)}",
                )
        for section in valid_sections:
            for name, block in (cfg.get(section) or {}).items():
                where = f"{top}.{section}.{name}"
                if not isinstance(block, dict) or "type" not in block:
                    rep.error(where, "Block has no 'type'.")
                    continue
                btype = block["type"]
                if btype not in registry:
                    rep.error(
                        where,
                        f"Unknown {top[:-1]} type '{btype}'.{_suggest(btype, registry)}",
                        fix=f"Valid: {', '.join(sorted(registry))}",
                    )
                    continue
                try:
                    cls = loader(btype)
                except ImportError as exc:
                    rep.error(where, f"'{btype}' could not be imported (missing optional dependency): {exc}")
                    continue
                err = _bind_error(cls, block.get("args"))
                if err:
                    rep.error(
                        where,
                        f"'{btype}' {err}",
                        fix=f"Accepted args: {', '.join(_accepted_params(cls))}",
                    )
                elif deep and not _awaits_preprocess(block.get("args")):
                    try:
                        cls(**(block.get("args") or {}))
                    except Exception as exc:  # noqa: BLE001 — surfacing any construction failure is the point
                        rep.error(where, f"'{btype}' failed to construct: {type(exc).__name__}: {exc}")


def _check_model(conf: dict, rep: _Report, deep: bool) -> None:
    from credit.models import _MODEL_REGISTRY, _load_model_entry

    mconf = dict(conf.get("model") or {})
    mtype = mconf.pop("type", None)
    if mtype not in _MODEL_REGISTRY:
        return  # already reported by _check_registry_keys
    mconf.pop("post_conf", None)  # gen1 leftover; train_gen2 strips it with a warning
    try:
        cls, _msg = _load_model_entry(mtype)
    except ImportError as exc:
        rep.error("model.type", f"'{mtype}' could not be imported: {exc}")
        return
    err = _bind_error(cls, mconf)
    if err:
        rep.error("model", f"'{mtype}' {err}", fix=f"Accepted keys: {', '.join(_accepted_params(cls))}")
        return
    if deep:
        try:
            cls(**mconf)
        except Exception as exc:  # noqa: BLE001
            rep.error("model", f"'{mtype}' failed to construct: {type(exc).__name__}: {exc}")


def _check_model_geometry(conf: dict, rep: _Report) -> None:
    """Check that the padded grid survives every downsample + window partition.

    CrossFormer-family models stride the grid down once per stage and then
    partition each stage into local and global attention windows.  A grid that
    is not divisible at some stage fails deep inside the forward pass with an
    opaque reshape error, so it is worth catching here.
    """
    mconf = conf.get("model") or {}
    height, width = mconf.get("image_height"), mconf.get("image_width")
    strides = mconf.get("cross_embed_strides")
    if not strides:
        return

    # CrossFormer-family models are hard-wired to 4 stages: each per-stage list
    # is broadcast with cast_tuple(x, 4), which leaves a wrong-length list alone
    # and then trips a bare `assert len(...) == 4` with no message.
    for key in (
        "dim",
        "depth",
        "global_window_size",
        "local_window_size",
        "cross_embed_kernel_sizes",
        "cross_embed_strides",
    ):
        value = mconf.get(key)
        if isinstance(value, list) and len(value) != 4:
            rep.error(
                f"model.{key}",
                f"model.{key} has {len(value)} entries; CrossFormer-family models require exactly 4 "
                "(one per stage). A scalar is broadcast to all 4, but a list is not padded.",
                fix=f"Give {key} 4 entries, or a single scalar to use for every stage.",
            )

    # Both decoder branches build their upsample blocks as
    # UpBlock*(..., out_ch=last_dim // 2**k, num_groups=dim[0]) for k = 1, 2, 3,
    # and nn.GroupNorm requires num_groups to divide the channel count.  The
    # tightest case is the narrowest block, last_dim // 8.
    dim = mconf.get("dim")
    dim = [dim] * 4 if isinstance(dim, int) else dim
    if isinstance(dim, list) and len(dim) == 4 and all(isinstance(d, int) for d in dim):
        narrowest = dim[-1] // 8
        if narrowest == 0 or narrowest % dim[0]:
            rep.error(
                "model.dim",
                f"The decoder normalises its narrowest upsample block ({dim[-1]} // 8 = {narrowest} "
                f"channels) into dim[0] = {dim[0]} GroupNorm groups, but {dim[0]} does not divide "
                f"{narrowest}. The model will not construct.",
                fix=(
                    f"Use a pyramid dim so dim[0] divides dim[-1]//8 — e.g. "
                    f"[{max(dim[-1] // 8, 1)}, {max(dim[-1] // 4, 1)}, {max(dim[-1] // 2, 1)}, {dim[-1]}]."
                ),
            )

    if not (isinstance(height, int) and isinstance(width, int)):
        return

    pad = mconf.get("padding_conf") or {}
    pad_lat = list(pad.get("pad_lat") or [0, 0]) if pad.get("activate") else [0, 0]
    pad_lon = list(pad.get("pad_lon") or [0, 0]) if pad.get("activate") else [0, 0]
    padded_h, padded_w = height + sum(pad_lat), width + sum(pad_lon)

    global_ws = mconf.get("global_window_size") or []
    local_ws = mconf.get("local_window_size")

    kernels = mconf.get("cross_embed_kernel_sizes") or []
    bad = _first_indivisible_stage(padded_h, padded_w, strides, kernels, global_ws, local_ws)
    if bad is None:
        return
    stage, stage_h, stage_w, wname, wsize = bad

    # Search for the smallest symmetric extra padding that clears every stage.
    # Cheaper and more reliable than deriving a closed form, since the stage
    # sizes come from a floor-chain rather than exact division.
    suggestion = None
    for extra_h in range(0, 1024):
        for extra_w in range(0, 1024):
            if (
                _first_indivisible_stage(padded_h + extra_h, padded_w + extra_w, strides, kernels, global_ws, local_ws)
                is None
            ):
                suggestion = (extra_h, extra_w)
                break
        if suggestion:
            break

    fix = None
    if suggestion:
        extra_h, extra_w = suggestion
        parts = []
        for pad_key, raw, cur, extra in (
            ("pad_lat", height, pad_lat, extra_h),
            ("pad_lon", width, pad_lon, extra_w),
        ):
            if extra:
                total = sum(cur) + extra
                parts.append(f"{pad_key}: [{total // 2}, {total - total // 2}]  # {raw} + {total}")
        if not pad.get("activate") and parts:
            parts.append("padding_conf.activate: true")
        fix = "\n".join(parts) or None

    rep.error(
        "model.padding_conf",
        f"Stage {stage} of the encoder is {stage_h}x{stage_w}, which is not divisible by "
        f"{wname} = {wsize}. Window attention reshapes that stage with einops and will fail with "
        f'"can\'t divide axis of length {stage_h if stage_h % wsize else stage_w} in chunks of {wsize}". '
        f"(Grid {height}x{width} padded to {padded_h}x{padded_w}; each stage is "
        "floor((n + 2*((k-s)//2) - k)/s) + 1, not an exact halving.)",
        fix=fix,
    )


def _first_indivisible_stage(height, width, strides, kernels, global_ws, local_ws):
    """Walk the encoder stages; return the first that a window size cannot tile.

    Stage sizes follow the real ``CrossEmbedLayer`` convolutions
    (``padding=(kernel - stride) // 2``), which floor rather than halve exactly —
    so an odd input can still land on a clean stage size.

    Returns ``(stage_index, h, w, window_name, window_size)`` or None.
    """
    cur_h, cur_w = height, width
    for i, stride in enumerate(strides):
        stage_kernels = kernels[i] if i < len(kernels) else None
        if isinstance(stage_kernels, int):
            stage_kernels = [stage_kernels]
        if not stage_kernels:
            stage_kernels = [stride * 2]  # only the stride matters when unspecified
        kernel = sorted(stage_kernels)[0]
        pad = (kernel - stride) // 2
        cur_h = (cur_h + 2 * pad - kernel) // stride + 1
        cur_w = (cur_w + 2 * pad - kernel) // stride + 1
        if cur_h < 1 or cur_w < 1:
            return (i, cur_h, cur_w, "stage size", 1)
        for wname, wsize in (
            ("global_window_size", global_ws[i] if i < len(global_ws) else None),
            ("local_window_size", local_ws),
        ):
            if isinstance(wsize, int) and wsize > 1 and (cur_h % wsize or cur_w % wsize):
                return (i, cur_h, cur_w, wname, wsize)
    return None


def _check_channel_counts(conf: dict, rep: _Report) -> None:
    """Cross-check model channel counts against the variables actually configured.

    CrossFormer computes ``channels*levels + surface_channels + input_only_channels``
    for its input width, so a mismatch here is a guaranteed shape error at the
    first forward pass.
    """
    mconf = conf.get("model") or {}
    sources = _get(conf, "data", "source", default={}) or {}
    if not sources or not mconf:
        return
    model_levels = mconf.get("levels")

    level_counts, n_prog_3d, n_prog_2d, n_input_only, n_diagnostic = set(), 0, 0, 0, 0
    for src in sources.values():
        src = src or {}
        variables = src.get("variables") or {}
        n_lev = _n_levels_for_source(src, model_levels)
        prog = variables.get("prognostic") or {}
        if prog.get("vars_3D"):
            if not n_lev:
                return  # unresolvable level count — _check_data_sources already flagged it
            level_counts.add(n_lev)
        n_prog_3d += len(prog.get("vars_3D") or [])
        n_prog_2d += len(prog.get("vars_2D") or [])
        for field_type in _INPUT_ONLY_FIELD_TYPES:
            grp = variables.get(field_type) or {}
            n_input_only += len(grp.get("vars_3D") or []) * (n_lev or 0) + len(grp.get("vars_2D") or [])
        diag = variables.get("diagnostic") or {}
        n_diagnostic += len(diag.get("vars_3D") or []) * (n_lev or 0) + len(diag.get("vars_2D") or [])

    if len(level_counts) == 1:
        (resolved,) = level_counts
        if model_levels is not None and resolved != model_levels:
            rep.error(
                "model.levels",
                f"model.levels is {model_levels} but the data sources resolve to {resolved} level(s).",
                fix=f"model.levels: {resolved}  (or change data.source.<name>.levels)",
            )
    elif len(level_counts) > 1:
        rep.info("model.levels", f"Sources use differing level counts {sorted(level_counts)}; skipping the check.")

    for key, expected, explanation in (
        ("channels", n_prog_3d, "3D prognostic variables"),
        ("surface_channels", n_prog_2d, "2D prognostic variables"),
        ("input_only_channels", n_input_only, "dynamic_forcing + static channels"),
        ("output_only_channels", n_diagnostic, "diagnostic channels"),
    ):
        configured = mconf.get(key)
        if configured is not None and configured != expected:
            rep.error(
                f"model.{key}",
                f"model.{key} is {configured} but the config defines {expected} {explanation}.",
                fix=f"{key}: {expected}",
            )


def _check_loss_pipeline(conf: dict, rep: _Report) -> None:
    """Check the postblock chain against what BaseLoss needs to score.

    BaseLoss compares ``y_processed`` (prediction chain) with
    ``y_target_processed`` (target twin).  The twin has to exist and has to
    apply the same unit-changing blocks, or the two sides are not comparable.
    """
    if _get(conf, "loss", "type") != "base":
        return
    per_step = _get(conf, "postblocks", "per_step", default={}) or {}

    pred_chain, twin_chain = {}, {}
    for name, block in per_step.items():
        if not isinstance(block, dict) or "type" not in block:
            continue
        (twin_chain if _block_output_key(block) == "y_target_processed" else pred_chain)[name] = block

    if not twin_chain:
        rep.error(
            "postblocks.per_step",
            "BaseLoss scores y_processed against y_target_processed, but no postblock produces y_target_processed.",
            fix=(
                "reconstruct_target:\n"
                "  type: reconstruct\n"
                "  args: {in_key: 'y', out_key: 'y_target_processed'}\n"
                "then mirror the prediction chain with key: 'y_target_processed'."
            ),
        )
    else:
        # The twin must apply the same unit transforms, else e.g. a log-space
        # target is compared against an exponentiated prediction.
        pred_units = {b["type"] for b in pred_chain.values() if b["type"] in _UNIT_POSTBLOCKS}
        twin_units = {b["type"] for b in twin_chain.values() if b["type"] in _UNIT_POSTBLOCKS}
        for missing in sorted(pred_units - twin_units):
            rep.error(
                "postblocks.per_step",
                f"'{missing}' runs on the prediction but not on the target twin, so the two sides "
                "of the loss end up in different units.",
                fix=f"Add a '{missing}' block with the same args plus key: 'y_target_processed'.",
            )

    # detach defaults to True, which silently severs the graph: training runs,
    # the loss is finite, and no gradient ever reaches the model.
    for name, block in pred_chain.items():
        if block["type"] == "reconstruct" and (block.get("args") or {}).get("detach", True):
            rep.error(
                f"postblocks.per_step.{name}",
                "reconstruct defaults to detach: true. BaseLoss scores the reconstructed output, so "
                "the gradient never reaches the model — training runs but learns nothing.",
                fix="args:\n  detach: false",
            )

    # Computed diagnostics are scored only if the same block also ran on the twin.
    include_computed = _get(conf, "loss", "args", "include_computed_diagnostics", default=True)
    if include_computed:
        twin_types = {b["type"] for b in twin_chain.values()}
        for name, block in pred_chain.items():
            if block["type"] in _DIAGNOSTIC_POSTBLOCKS and block["type"] not in twin_types:
                rep.error(
                    f"postblocks.per_step.{name}",
                    f"'{block['type']}' adds a computed diagnostic to y_processed, and "
                    "loss.args.include_computed_diagnostics is true, so BaseLoss will look for the "
                    "same variable in y_target_processed and raise a KeyError.",
                    fix=(
                        f"Add a second '{block['type']}' block with key: 'y_target_processed', "
                        "or set loss.args.include_computed_diagnostics: false."
                    ),
                )

    if _get(conf, "loss", "args", "var_weighting", default="inverse_variance") in ("inverse_variance", "learnable"):
        if not _get(conf, "loss", "args", "scaler_path"):
            rep.error(
                "loss.args.scaler_path",
                "var_weighting needs per-variable variances from a fitted scaler, but scaler_path is unset.",
                fix="Set loss.args.scaler_path, or use var_weighting: none / manual.",
            )
    if _get(conf, "loss", "args", "use_latitude_weights", default=False) and not _get(
        conf, "loss", "args", "latitude_weights"
    ):
        rep.error(
            "loss.args.latitude_weights",
            "use_latitude_weights is true but no latitude_weights file is given.",
            fix="Set loss.args.latitude_weights to a dataset with a 'latitude' coordinate.",
        )


def _check_trainer(conf: dict, rep: _Report) -> None:
    from credit.parallel.mesh import parse_parallelism_conf

    try:
        parse_parallelism_conf(conf)
    except (ValueError, KeyError) as exc:
        rep.error(
            "trainer.parallelism",
            str(exc),
            fix="parallelism:\n  data: fsdp2   # fsdp2 | ddp | none\n  tensor: 1\n  domain: 1",
        )

    if _get(conf, "trainer", "use_scheduler", default=False):
        sched = dict(_get(conf, "trainer", "scheduler", default={}) or {})
        stype = sched.pop("scheduler_type", None)
        builders = {
            "lambda": None,
            "plateau": ("torch.optim.lr_scheduler", "ReduceLROnPlateau"),
            "cosine-annealing": ("torch.optim.lr_scheduler", "CosineAnnealingLR"),
            "cosine-annealing-restarts": ("credit.scheduler", "CosineAnnealingWarmupRestarts"),
            "linear-warmup-cosine": ("credit.scheduler", "LinearWarmupCosineScheduler"),
        }
        if stype not in builders:
            rep.error(
                "trainer.scheduler.scheduler_type",
                f"Unknown scheduler_type '{stype}'.{_suggest(stype, builders)}",
                fix=f"Valid: {', '.join(sorted(builders))}",
            )
        elif builders[stype] is not None:
            import importlib

            module_path, class_name = builders[stype]
            cls = getattr(importlib.import_module(module_path), class_name)
            # The optimizer is supplied positionally at build time; only the
            # config-provided keyword arguments are checked here.
            unknown = sorted(set(sched) - set(_accepted_params(cls)))
            if unknown:
                rep.error(
                    "trainer.scheduler",
                    f"'{stype}' does not accept {unknown}.",
                    fix=f"Accepted: {', '.join(p for p in _accepted_params(cls) if p != 'optimizer')}",
                )

    epochs = _get(conf, "trainer", "epochs")
    num_epoch = _get(conf, "trainer", "num_epoch")
    if epochs is not None and num_epoch is not None and num_epoch > epochs:
        rep.warn(
            "trainer.num_epoch",
            f"num_epoch ({num_epoch}) exceeds epochs ({epochs}); `credit submit` chains jobs from "
            "their ratio and would submit a single job.",
        )


def _check_paths(conf: dict, rep: _Report) -> None:
    """Existence checks for every file the config names."""
    save_loc = os.path.expandvars(conf.get("save_loc") or "")
    if save_loc:
        parent = os.path.dirname(save_loc.rstrip("/")) or "."
        if not os.path.isdir(save_loc) and not os.path.isdir(parent):
            rep.warn("save_loc", f"Neither {save_loc} nor its parent exists; training will try to create it.")
        if "$" in (conf.get("save_loc") or "") and "$" in save_loc:
            rep.warn(
                "save_loc",
                f"Unexpanded variable in save_loc after expansion: {save_loc}",
                fix="Check that the environment variable is set in the job script.",
            )

    seen = set()
    for where, path in _iter_config_paths(conf):
        expanded = os.path.expandvars(path)
        if expanded in seen:
            continue
        seen.add(expanded)
        if os.path.exists(expanded):
            continue
        if "scaler" in where:
            rep.warn(
                where,
                f"Scaler file does not exist yet: {expanded}",
                fix="Run `credit preprocess -c <config>` to fit it before training.",
            )
        else:
            rep.error(where, f"File not found: {expanded}")

    # level_info_file resolves against credit/metadata when given as a bare name.
    from credit.metadata import get_meta_file_path

    for section in ("per_step", "post_rollout"):
        for name, block in (_get(conf, "postblocks", section, default={}) or {}).items():
            if not isinstance(block, dict):
                continue
            meta_file = (block.get("args") or {}).get("level_info_file")
            if meta_file and not os.path.exists(get_meta_file_path(meta_file)):
                rep.error(
                    f"postblocks.{section}.{name}.level_info_file",
                    f"Metadata file not found: {meta_file} (resolved to {get_meta_file_path(meta_file)})",
                )


def _iter_config_paths(conf: dict):
    """Yield ``(where, path)`` for the file-valued settings worth existence-checking."""
    for section in ("preblocks", "postblocks"):
        for phase, blocks in (conf.get(section) or {}).items():
            for name, block in (blocks or {}).items():
                if not isinstance(block, dict):
                    continue
                for key in ("scaler_path", "latitude_weights"):
                    value = (block.get("args") or {}).get(key)
                    if isinstance(value, str) and value:
                        yield f"{section}.{phase}.{name}.{key}", value
    for key in ("scaler_path", "latitude_weights"):
        for top in ("loss", "metrics"):
            value = _get(conf, top, "args", key)
            if isinstance(value, str) and value:
                yield f"{top}.args.{key}", value


def _check_channel_schema(conf: dict, rep: _Report) -> None:
    """Derive the channel schema, and compare it with any saved one."""
    from credit.datasets.gen_2.channel_utils import DEFAULT_SCHEMA_FILENAME, ChannelSchema

    try:
        schema = ChannelSchema.from_config(conf)
    except (KeyError, ValueError) as exc:
        rep.error(
            "data.source",
            f"Channel schema cannot be derived from this config: {exc}",
            fix="Set data.source.<name>.levels or model.levels.",
        )
        return

    saved_path = os.path.join(os.path.expandvars(conf.get("save_loc") or ""), DEFAULT_SCHEMA_FILENAME)
    if not os.path.isfile(saved_path):
        return
    try:
        saved = ChannelSchema.load(saved_path)
    except Exception as exc:  # noqa: BLE001
        rep.warn("save_loc", f"Could not read the saved channel schema {saved_path}: {exc}")
        return
    if saved.input_layout != schema.input_layout or saved.target_layout != schema.target_layout:
        rep.error(
            "save_loc",
            f"The channel schema saved at {saved_path} disagrees with this config. Training loads the "
            "SAVED schema, so your config edits will not take effect and may corrupt reconstruction.",
            fix="Delete the saved schema to re-derive it, or train into a fresh save_loc.",
        )


def _check_deep_loads(conf: dict, rep: _Report) -> None:
    """Actually construct the loss and metrics (blocks/model handled in place)."""
    from credit.losses import load_loss
    from credit.metrics import load_metric

    skipped = False
    for label, fn in (("loss", load_loss), ("metrics", load_metric)):
        pending = _awaits_preprocess(_get(conf, label, "args", default={}))
        if pending:
            skipped = True
            continue
        try:
            fn(conf)
        except Exception as exc:  # noqa: BLE001
            rep.error(label, f"Failed to load: {type(exc).__name__}: {exc}")
    if skipped:
        rep.info(
            "--deep",
            "Skipped constructing everything that reads a scaler that has not been fitted yet; "
            "re-run --deep after `credit preprocess` to check those too.",
        )


def _check_pbs(conf: dict, rep: _Report) -> None:
    pbs = conf.get("pbs") or {}
    if not pbs:
        return
    # gpu_type is only used by the Casper template; leaving it unset requests any
    # NVIDIA GPGPU, which is a valid (and usually faster-queuing) choice.
    if pbs.get("queue") in ("casper", "gpgpu") and "gpu_type" not in pbs:
        rep.info(
            "pbs.gpu_type",
            "Unset; Casper jobs will run on any available NVIDIA GPGPU. "
            "Set gpu_type (a100_80gb, h100, v100, ...) to pin a specific model.",
        )
    # The config names a queue but not a cluster, so the queue is what says which
    # machine this block was written for.  Say so — reusing a Derecho block on
    # Casper (or vice versa) is rejected by `credit submit`.
    queue = pbs.get("queue")
    if queue:
        name = str(queue).split("@", 1)[0].strip().lower()
        targets = sorted(c for c, queues in _PBS_QUEUES.items() if name in queues)
        if not targets:
            rep.warn(
                "pbs.queue",
                f"'{name}' is not a known Casper or Derecho queue.",
                fix="queue: casper   # or 'main' on Derecho",
            )
        elif len(targets) == 1:
            rep.info(
                "pbs.queue",
                f"'{name}' is a {targets[0].capitalize()} queue; "
                f"`credit submit --cluster {targets[0]}` is the matching target.",
            )
    for key in ("project", "job_name", "walltime", "queue"):
        if key not in pbs:
            rep.warn(f"pbs.{key}", f"'{key}' is unset; `credit submit` will fall back to a built-in default.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_checks(conf: dict, rep: _Report, deep: bool = False) -> None:
    _check_top_level(conf, rep)
    if _detect_generation(conf, rep) == "gen1":
        return

    # Custom classes registered by the config must be importable before any
    # registry lookup, or a valid custom type looks like a typo.
    try:
        from credit.registry import load_custom_objects

        load_custom_objects(conf)
    except Exception as exc:  # noqa: BLE001
        rep.error("custom_objects", f"Failed to load: {type(exc).__name__}: {exc}")

    # A data-only fragment has nothing to say about models, losses, or training.
    checks = [_check_data_sources, _check_validation_data, _check_paths]
    if not _is_data_fragment(conf):
        checks += [
            _check_registry_keys,
            _check_channel_counts,
            _check_model_geometry,
            _check_loss_pipeline,
            _check_trainer,
            _check_channel_schema,
            _check_pbs,
        ]
    for check in checks:
        try:
            check(conf, rep)
        except Exception as exc:  # noqa: BLE001 — a crashing check must not hide the others
            rep.warn(check.__name__, f"Check itself failed: {type(exc).__name__}: {exc}")

    block_checks = [_check_blocks] if _is_data_fragment(conf) else [_check_blocks, _check_model]
    for check in block_checks:
        try:
            check(conf, rep, deep)
        except Exception as exc:  # noqa: BLE001
            rep.warn(check.__name__, f"Check itself failed: {type(exc).__name__}: {exc}")

    if deep:
        try:
            _check_deep_loads(conf, rep)
        except Exception as exc:  # noqa: BLE001
            rep.warn("_check_deep_loads", f"Check itself failed: {type(exc).__name__}: {exc}")


def _check(args: argparse.Namespace) -> None:
    rep = _Report(args.config)
    conf = _load_config(args.config, rep)
    if conf is not None:
        _run_checks(conf, rep, deep=getattr(args, "deep", False))
    if getattr(args, "json", False):
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        rep.render()
    sys.exit(rep.exit_code(strict=getattr(args, "strict", False)))
