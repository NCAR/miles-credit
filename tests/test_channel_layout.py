"""Unit tests for build_channel_layout / update_x (multi-source channel slicing).

The authority these tests check against is ``ChannelSchema``, because that is
what ``ConcatToTensor`` validates the real batch against at runtime
(``validate_channel_map``).  If the layout agrees with the schema, it agrees
with the tensors the model actually sees.
"""

import pytest
import torch

from credit.datasets.gen_2.channel_utils import (
    ChannelGroup,
    ChannelSchema,
    build_channel_layout,
    update_x,
)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _source(levels=None, prognostic=None, static=None, dynamic_forcing=None, diagnostic=None):
    src = {"dataset_type": "local", "variables": {}}
    if levels is not None:
        src["levels"] = levels
    for name, grp in (
        ("prognostic", prognostic),
        ("static", static),
        ("dynamic_forcing", dynamic_forcing),
        ("diagnostic", diagnostic),
    ):
        src["variables"][name] = grp
    return src


def _conf(sources, model_levels=None, history_len=1):
    conf = {"data": {"source": sources, "history_len": history_len, "forecast_len": 1}}
    if model_levels is not None:
        conf["model"] = {"levels": model_levels}
    return conf


@pytest.fixture
def single_source():
    """One source: 2x3D + 1x2D prognostic, 1 static, 1 dyn forcing, 1 diagnostic."""
    return _conf(
        {
            "ERA5": _source(
                levels=[1, 2, 3],
                prognostic={"vars_3D": ["T", "Q"], "vars_2D": ["SP"]},
                static={"vars_2D": ["lsm"]},
                dynamic_forcing={"vars_2D": ["tisr"]},
                diagnostic={"vars_2D": ["TP"]},
            )
        }
    )


@pytest.fixture
def two_sources():
    """The arco shape: a big ERA5 source plus a forcing-only SOLAR source."""
    return _conf(
        {
            "ERA5": _source(
                levels=[1, 2],
                prognostic={"vars_3D": ["T"], "vars_2D": ["SP"]},
                static={"vars_2D": ["lsm", "z"]},
                diagnostic={"vars_2D": ["TP"]},
            ),
            "SOLAR": _source(dynamic_forcing={"vars_2D": ["tisr"]}),
        }
    )


@pytest.fixture
def interleaved():
    """Two sources that BOTH predict, each with its own diagnostics.

    This is the case the old sequential implementation got wrong: the second
    source's prognostic block in y_pred is pushed along by the first source's
    diagnostics, while in x it is pushed along by the first source's statics.
    """
    return _conf(
        {
            "A": _source(
                levels=[1, 2],
                prognostic={"vars_3D": ["T"]},  # 2 ch
                static={"vars_2D": ["lsm"]},  # 1 ch
                diagnostic={"vars_2D": ["TP", "SF"]},  # 2 ch (y only)
            ),
            "B": _source(
                levels=[1, 2, 3],
                prognostic={"vars_2D": ["SP"]},  # 1 ch
                dynamic_forcing={"vars_2D": ["tisr"]},  # 1 ch
            ),
        }
    )


def _schema_group_spans(channel_map):
    """Collapse a per-variable channel map into (source/field_type) -> (start, stop)."""
    spans = {}
    for var_key, entry in channel_map.items():
        source, field_type = var_key.split("/")[0], var_key.split("/")[1]
        key = f"{source}/{field_type}"
        start, stop = entry["slice"].start, entry["slice"].stop
        spans[key] = (spans[key][0], stop) if key in spans else (start, stop)
    return spans


# ---------------------------------------------------------------------------
# Agreement with ChannelSchema — the runtime-validated authority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["single_source", "two_sources", "interleaved"])
def test_x_layout_matches_channel_schema(fixture, request):
    conf = request.getfixturevalue(fixture)
    groups, _ = build_channel_layout(conf)
    expected = _schema_group_spans(ChannelSchema.from_config(conf).input_channel_map())
    actual = {k: (g.x_slice.start, g.x_slice.stop) for k, g in groups.items()}
    assert actual == expected


@pytest.mark.parametrize("fixture", ["single_source", "two_sources", "interleaved"])
def test_prognostic_src_slices_match_target_layout(fixture, request):
    """src_slice must index y_pred, whose layout is the schema's TARGET map."""
    conf = request.getfixturevalue(fixture)
    groups, _ = build_channel_layout(conf)
    expected = _schema_group_spans(ChannelSchema.from_config(conf).target_channel_map())
    for key, group in groups.items():
        if group.field_type == "prognostic":
            assert (group.src_slice.start, group.src_slice.stop) == expected[key]


def test_x_slices_tile_without_gaps_or_overlap(two_sources):
    groups, _ = build_channel_layout(two_sources)
    cursor = 0
    for group in groups.values():
        assert group.x_slice.start == cursor
        cursor = group.x_slice.stop
    assert cursor == 2 + 1 + 2 + 1  # ERA5 prog(2+1) + static(2) + SOLAR dyn(1)


# ---------------------------------------------------------------------------
# Multi-source specifics
# ---------------------------------------------------------------------------


def test_all_sources_are_represented(two_sources):
    """Regression: only the first source used to be read, dropping SOLAR/tisr."""
    groups, _ = build_channel_layout(two_sources)
    assert set(groups) == {"ERA5/prognostic", "ERA5/static", "SOLAR/dynamic_forcing"}
    assert max(g.x_slice.stop for g in groups.values()) == 6  # not 5


def test_second_source_prognostic_is_offset_by_earlier_diagnostics(interleaved):
    groups, _ = build_channel_layout(interleaved)
    a, b = groups["A/prognostic"], groups["B/prognostic"]
    assert (a.x_slice.start, a.x_slice.stop) == (0, 2)
    assert (a.src_slice.start, a.src_slice.stop) == (0, 2)
    # In x, B follows A's prognostic (2) + A's static (1) -> 3.
    # In y_pred, B follows A's prognostic (2) + A's diagnostics (2) -> 4.
    assert (b.x_slice.start, b.x_slice.stop) == (3, 4)
    assert (b.src_slice.start, b.src_slice.stop) == (4, 5)


def test_dynamic_forcing_offsets_are_source_major(interleaved):
    groups, _ = build_channel_layout(interleaved)
    assert groups["B/dynamic_forcing"].src_slice == slice(0, 1)


def test_sources_may_have_different_level_counts(interleaved):
    groups, _ = build_channel_layout(interleaved)
    assert groups["A/prognostic"].x_slice.stop - groups["A/prognostic"].x_slice.start == 2  # 1 var x 2 lev
    assert groups["B/prognostic"].x_slice.stop - groups["B/prognostic"].x_slice.start == 1  # 1 x 2D var


def test_levels_fall_back_to_model_levels():
    conf = _conf({"S": _source(prognostic={"vars_3D": ["T"]})}, model_levels=5)
    groups, n_pred = build_channel_layout(conf)
    assert n_pred == 5


def test_static_has_no_src_slice(two_sources):
    groups, _ = build_channel_layout(two_sources)
    assert groups["ERA5/static"].src_slice is None


def test_diagnostic_never_appears_in_x(single_source):
    groups, _ = build_channel_layout(single_source)
    assert all(g.field_type != "diagnostic" for g in groups.values())


def test_empty_and_null_groups_are_skipped():
    conf = _conf(
        {"S": _source(levels=[1], prognostic={"vars_2D": ["SP"]}, static=None, dynamic_forcing={"vars_2D": []})}
    )
    groups, _ = build_channel_layout(conf)
    assert set(groups) == {"S/prognostic"}


def test_n_pred_sums_prognostics_across_sources(interleaved):
    _, n_pred = build_channel_layout(interleaved)
    assert n_pred == 3  # A: 1 var x 2 lev, B: 1 x 2D var


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_history_len_greater_than_one_raises():
    conf = _conf({"S": _source(levels=[1], prognostic={"vars_2D": ["SP"]})}, history_len=2)
    with pytest.raises(ValueError, match="history_len"):
        build_channel_layout(conf)


def test_unresolvable_levels_raises():
    conf = _conf({"S": _source(prognostic={"vars_3D": ["T"]})})  # no levels, no model.levels
    with pytest.raises(ValueError, match="3D variables"):
        build_channel_layout(conf)


# ---------------------------------------------------------------------------
# update_x
# ---------------------------------------------------------------------------


def _sentinel_tensors(conf):
    """x filled with -1; y_pred[c] = 1000+c; x_dynfrc[c] = 2000+c."""
    groups, _ = build_channel_layout(conf)
    schema = ChannelSchema.from_config(conf)
    n_x = max(g.x_slice.stop for g in groups.values())
    n_y = max(e["slice"].stop for e in schema.target_channel_map().values())
    dyn = [g for g in groups.values() if g.field_type == "dynamic_forcing"]
    n_d = max((g.src_slice.stop for g in dyn), default=1)

    def ramp(base, n):
        return (base + torch.arange(n).float()).view(1, n, 1, 1, 1).expand(1, n, 1, 2, 2).contiguous()

    return groups, schema, torch.full((1, n_x, 1, 2, 2), -1.0), ramp(2000, n_d), ramp(1000, n_y)


@pytest.mark.parametrize("fixture", ["single_source", "two_sources", "interleaved"])
def test_update_x_places_every_channel_correctly(fixture, request):
    """Each x channel must receive the value the ChannelSchema says belongs there."""
    conf = request.getfixturevalue(fixture)
    groups, schema, x, dyn, y = _sentinel_tensors(conf)
    out = update_x(x, dyn, y, groups)[0, :, 0, 0, 0].tolist()

    input_map, target_map = schema.input_channel_map(), schema.target_channel_map()
    # Recompute dynamic-forcing offsets from the schema, independently of `groups`.
    dyn_offset, offsets = 0, {}
    for key, entry in input_map.items():
        if key.split("/")[1] == "dynamic_forcing":
            group_key = "/".join(key.split("/")[:2])
            offsets.setdefault(group_key, dyn_offset)
            dyn_offset += entry["slice"].stop - entry["slice"].start

    for var_key, entry in input_map.items():
        source, field_type = var_key.split("/")[0], var_key.split("/")[1]
        group_key = f"{source}/{field_type}"
        for i, channel in enumerate(range(entry["slice"].start, entry["slice"].stop)):
            if field_type == "prognostic":
                expected = 1000 + target_map[var_key]["slice"].start + i
            elif field_type == "dynamic_forcing":
                expected = 2000 + offsets[group_key] + i
            else:
                expected = -1.0  # static: carried forward untouched
            assert out[channel] == expected, f"{var_key} channel {channel}"


def test_update_x_leaves_x_prev_unmodified(two_sources):
    groups, _schema, x, dyn, y = _sentinel_tensors(two_sources)
    before = x.clone()
    update_x(x, dyn, y, groups)
    assert torch.equal(x, before)


def test_update_x_ignores_group_iteration_order(interleaved):
    """Correctness must come from the slices, not from dict ordering."""
    groups, _schema, x, dyn, y = _sentinel_tensors(interleaved)
    shuffled = dict(reversed(list(groups.items())))
    assert torch.equal(update_x(x, dyn, y, groups), update_x(x, dyn, y, shuffled))


def test_channel_group_is_immutable():
    group = ChannelGroup("S", "prognostic", slice(0, 1), slice(0, 1))
    with pytest.raises(Exception):
        group.x_slice = slice(0, 2)
