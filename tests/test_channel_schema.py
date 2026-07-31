"""Tests for credit.datasets.gen_2.channel_utils.ChannelSchema and its ConcatToTensor integration."""

import pytest
import torch
import yaml

from credit.datasets.gen_2.channel_utils import DEFAULT_SCHEMA_FILENAME, ChannelSchema
from credit.preblock import attach_channel_schema, build_preblocks
from credit.preblock.concat import ConcatToTensor
from credit.postblock.reconstruct import Reconstruct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# CAMulator-shaped config: no source-level `levels` list (resolved from
# model.levels), 3D+2D prognostics, 2D diagnostics, forcing and statics.
CONF = {
    "save_loc": "/tmp/does-not-exist-schema-test",
    "data": {
        "source": {
            "CESM": {
                "dataset_type": "local",
                "variables": {
                    "prognostic": {"vars_3D": ["U", "T"], "vars_2D": ["PS"]},
                    "diagnostic": {"vars_2D": ["PRECT", "FLUT"]},
                    "dynamic_forcing": {"vars_2D": ["SOLIN"]},
                    "static": {"vars_2D": ["z_norm"]},
                },
            }
        },
    },
    "model": {"levels": 3},
}


def make_batch(n_levels=3, H=4, W=5, with_target=True):
    """Synthetic nested batch matching CONF, in dataset insertion order."""
    t3 = lambda: torch.randn(1, n_levels, 1, H, W)  # noqa: E731
    t2 = lambda: torch.randn(1, 1, 1, H, W)  # noqa: E731
    batch = {
        "input": {
            "CESM": {
                # deliberately shuffled — ConcatToTensor sorts the input
                "CESM/dynamic_forcing/2d/SOLIN": t2(),
                "CESM/prognostic/3d/U": t3(),
                "CESM/static/2d/z_norm": t2(),
                "CESM/prognostic/3d/T": t3(),
                "CESM/prognostic/2d/PS": t2(),
            }
        },
    }
    if with_target:
        # dataset target insertion order: prognostic (3d then 2d), then diagnostic
        batch["target"] = {
            "CESM": {
                "CESM/prognostic/3d/U": t3(),
                "CESM/prognostic/3d/T": t3(),
                "CESM/prognostic/2d/PS": t2(),
                "CESM/diagnostic/2d/PRECT": t2(),
                "CESM/diagnostic/2d/FLUT": t2(),
            }
        }
    return batch


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_target_layout_order_and_levels(self):
        """Target layout: prognostic 3d → 2d → diagnostic, levels from model.levels."""
        schema = ChannelSchema.from_config(CONF)
        keys = [e["var_key"] for e in schema.target_layout]
        assert keys == [
            "CESM/prognostic/3d/U",
            "CESM/prognostic/3d/T",
            "CESM/prognostic/2d/PS",
            "CESM/diagnostic/2d/PRECT",
            "CESM/diagnostic/2d/FLUT",
        ]
        assert schema.target_layout[0]["n_levels"] == 3
        assert schema.target_layout[2]["n_levels"] == 1

    def test_input_layout_order(self):
        """Input layout: prognostic < static < dynamic_forcing, 3d before 2d; no diagnostics."""
        schema = ChannelSchema.from_config(CONF)
        keys = [e["var_key"] for e in schema.input_layout]
        assert keys == [
            "CESM/prognostic/3d/U",
            "CESM/prognostic/3d/T",
            "CESM/prognostic/2d/PS",
            "CESM/static/2d/z_norm",
            "CESM/dynamic_forcing/2d/SOLIN",
        ]

    def test_source_levels_take_priority_over_model_levels(self):
        conf = yaml.safe_load(yaml.safe_dump(CONF))  # deep copy
        conf["data"]["source"]["CESM"]["levels"] = [500, 850]
        schema = ChannelSchema.from_config(conf)
        assert schema.target_layout[0]["n_levels"] == 2

    def test_raises_when_3d_vars_and_no_levels(self):
        conf = yaml.safe_load(yaml.safe_dump(CONF))
        del conf["model"]["levels"]
        with pytest.raises(ValueError, match="neither data.source.CESM.levels nor model.levels"):
            ChannelSchema.from_config(conf)

    def test_multi_source_config_order(self):
        """Sources appear in config key order in both layouts."""
        conf = yaml.safe_load(yaml.safe_dump(CONF))
        conf["data"]["source"]["OBS"] = {
            "dataset_type": "local",
            "variables": {"prognostic": {"vars_2D": ["refl"]}},
        }
        schema = ChannelSchema.from_config(conf)
        target_keys = [e["var_key"] for e in schema.target_layout]
        assert target_keys.index("CESM/diagnostic/2d/FLUT") < target_keys.index("OBS/prognostic/2d/refl")

    def test_channel_map_slices(self):
        """Slices are contiguous, in order, sized n_levels * n_time."""
        cmap = ChannelSchema.from_config(CONF).target_channel_map()
        assert cmap["CESM/prognostic/3d/U"]["slice"] == slice(0, 3)
        assert cmap["CESM/prognostic/3d/T"]["slice"] == slice(3, 6)
        assert cmap["CESM/prognostic/2d/PS"]["slice"] == slice(6, 7)
        assert cmap["CESM/diagnostic/2d/PRECT"]["slice"] == slice(7, 8)
        assert cmap["CESM/diagnostic/2d/FLUT"]["slice"] == slice(8, 9)
        assert cmap["CESM/prognostic/3d/U"]["orig_shape"] == (3, 1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        schema = ChannelSchema.from_config(CONF)
        path = str(tmp_path / DEFAULT_SCHEMA_FILENAME)
        schema.save(path)
        loaded = ChannelSchema.load(path)
        assert loaded.input_layout == schema.input_layout
        assert loaded.target_layout == schema.target_layout

    def test_load_or_from_config_prefers_saved_file(self, tmp_path):
        """A saved schema wins over config derivation."""
        schema = ChannelSchema.from_config(CONF)
        schema.target_layout = schema.target_layout[:-1]  # make it distinguishable
        schema.save(str(tmp_path / DEFAULT_SCHEMA_FILENAME))
        loaded = ChannelSchema.load_or_from_config(CONF, save_loc=str(tmp_path))
        assert len(loaded.target_layout) == len(schema.target_layout)

    def test_load_or_from_config_falls_back_to_config(self, tmp_path):
        loaded = ChannelSchema.load_or_from_config(CONF, save_loc=str(tmp_path))
        assert [e["var_key"] for e in loaded.target_layout] == [
            e["var_key"] for e in ChannelSchema.from_config(CONF).target_layout
        ]

    def test_load_or_from_config_returns_none_when_unresolvable(self, tmp_path):
        conf = yaml.safe_load(yaml.safe_dump(CONF))
        del conf["model"]["levels"]
        assert ChannelSchema.load_or_from_config(conf, save_loc=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# validate_channel_map
# ---------------------------------------------------------------------------


class TestValidate:
    def test_matching_map_passes(self):
        schema = ChannelSchema.from_config(CONF)
        schema.validate_channel_map(schema.target_channel_map(), which="target")

    def test_reordered_map_raises(self):
        schema = ChannelSchema.from_config(CONF)
        cmap = schema.target_channel_map()
        reordered = dict(reversed(list(cmap.items())))
        with pytest.raises(ValueError, match="ChannelSchema mismatch"):
            schema.validate_channel_map(reordered, which="target")

    def test_wrong_shape_raises(self):
        schema = ChannelSchema.from_config(CONF)
        cmap = schema.target_channel_map()
        cmap["CESM/prognostic/3d/U"] = {"slice": slice(0, 4), "orig_shape": (4, 1)}
        with pytest.raises(ValueError, match="ChannelSchema mismatch"):
            schema.validate_channel_map(cmap, which="target")


# ---------------------------------------------------------------------------
# ConcatToTensor integration — the ordering contract, end to end
# ---------------------------------------------------------------------------


class TestConcatIntegration:
    def test_schema_matches_data_derived_target_map(self):
        """The config-derived schema reproduces the map ConcatToTensor builds from real targets."""
        concat = ConcatToTensor(to_device=False)
        _, _, meta = concat(make_batch(with_target=True))
        schema = ChannelSchema.from_config(CONF)
        schema.validate_channel_map(meta["target"]["_channel_map"], which="target")

    def test_schema_matches_data_derived_input_map(self):
        concat = ConcatToTensor(to_device=False)
        _, _, meta = concat(make_batch(with_target=True))
        schema = ChannelSchema.from_config(CONF)
        schema.validate_channel_map(meta["input"]["_channel_map"], which="input")

    def test_no_target_no_schema_drops_diagnostics(self):
        """Legacy fallback: without a schema, the inference map is prognostic-only."""
        concat = ConcatToTensor(to_device=False)
        _, meta = concat(make_batch(with_target=False))
        keys = list(meta["target"]["_channel_map"])
        assert "CESM/diagnostic/2d/PRECT" not in keys

    def test_no_target_with_schema_covers_diagnostics(self):
        """With a schema attached, the inference map covers all predicted variables."""
        concat = ConcatToTensor(to_device=False)
        concat.set_schema(ChannelSchema.from_config(CONF))
        _, meta = concat(make_batch(with_target=False))
        keys = list(meta["target"]["_channel_map"])
        assert keys == [
            "CESM/prognostic/3d/U",
            "CESM/prognostic/3d/T",
            "CESM/prognostic/2d/PS",
            "CESM/diagnostic/2d/PRECT",
            "CESM/diagnostic/2d/FLUT",
        ]

    def test_target_present_validates_against_schema(self):
        """A schema that disagrees with the data-derived target map fails the forward pass."""
        conf = yaml.safe_load(yaml.safe_dump(CONF))
        conf["model"]["levels"] = 5  # wrong level count → layout mismatch
        concat = ConcatToTensor(to_device=False)
        concat.set_schema(ChannelSchema.from_config(conf))
        with pytest.raises(ValueError, match="ChannelSchema mismatch"):
            concat(make_batch(with_target=True))

    def test_reconstruct_recovers_diagnostics_at_inference(self):
        """End to end: y_pred sliced by the schema map yields all diagnostics."""
        H, W = 4, 5
        concat = ConcatToTensor(to_device=False)
        concat.set_schema(ChannelSchema.from_config(CONF))
        _, meta = concat(make_batch(with_target=False))

        n_out = 3 + 3 + 1 + 1 + 1  # U(3) T(3) PS PRECT FLUT
        batch_dict = {"y_pred": torch.randn(1, n_out, H, W), "metadata": meta}
        out = Reconstruct(detach=True)(batch_dict)

        y_processed = out["y_processed"]["CESM"]
        assert set(y_processed) == {
            "CESM/prognostic/3d/U",
            "CESM/prognostic/3d/T",
            "CESM/prognostic/2d/PS",
            "CESM/diagnostic/2d/PRECT",
            "CESM/diagnostic/2d/FLUT",
        }
        assert y_processed["CESM/prognostic/3d/U"].shape == (1, 3, 1, H, W)
        assert y_processed["CESM/diagnostic/2d/FLUT"].shape == (1, 1, 1, H, W)
        # values line up with the flat tensor slices
        assert torch.equal(y_processed["CESM/diagnostic/2d/PRECT"][:, 0, 0], batch_dict["y_pred"][:, 7])

    def test_attach_channel_schema_helper(self):
        """attach_channel_schema reaches ConcatToTensor blocks inside a built group."""
        preblocks = build_preblocks(
            {"preblocks": {"per_step": {"concat": {"type": "concat", "args": {"to_device": False}}}}}, "per_step"
        )
        print(preblocks)
        attach_channel_schema(preblocks, ChannelSchema.from_config(CONF))
        assert preblocks["concat"]._schema is not None
        attach_channel_schema(preblocks, None)  # no-op, must not raise
