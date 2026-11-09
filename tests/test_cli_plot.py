"""Unit tests for credit plot helpers: _build_channel_map and the postblock-based --denorm path.

All tests run on CPU with no real data files; the bridgescaler file is synthesized in tmp_path.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------


def _make_conf(vars_3d=None, vars_2d=None, diag_2d=None, n_levels=3):
    levels = list(range(n_levels))
    return {
        "data": {
            "source": {
                "ERA5": {
                    "level_coord": "level",
                    "levels": levels,
                    "variables": {
                        "prognostic": {
                            "vars_3D": vars_3d or [],
                            "vars_2D": vars_2d or [],
                        },
                        "diagnostic": {
                            "vars_2D": diag_2d or [],
                        },
                    },
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# _build_channel_map
# ---------------------------------------------------------------------------


class TestBuildChannelMap:
    def test_only_2d_variables(self):
        from credit.cli import _build_channel_map

        conf = _make_conf(vars_2d=["SP", "VAR_2T"], n_levels=3)
        cm = _build_channel_map(conf)
        assert cm == {"SP": [0], "VAR_2T": [1]}

    def test_only_3d_variables(self):
        from credit.cli import _build_channel_map

        conf = _make_conf(vars_3d=["U", "V"], n_levels=4)
        cm = _build_channel_map(conf)
        # U occupies indices 0-3, V occupies 4-7
        assert cm["U"] == [0, 1, 2, 3]
        assert cm["V"] == [4, 5, 6, 7]

    def test_3d_then_2d_ordering(self):
        from credit.cli import _build_channel_map

        conf = _make_conf(vars_3d=["T"], vars_2d=["SP"], n_levels=2)
        cm = _build_channel_map(conf)
        # T: channels 0-1, SP: channel 2
        assert cm["T"] == [0, 1]
        assert cm["SP"] == [2]

    def test_diagnostic_appended_after_prognostic(self):
        from credit.cli import _build_channel_map

        conf = _make_conf(vars_3d=["T"], vars_2d=["SP"], diag_2d=["precip"], n_levels=2)
        cm = _build_channel_map(conf)
        assert cm["T"] == [0, 1]
        assert cm["SP"] == [2]
        assert cm["precip"] == [3]

    def test_total_channel_count(self):
        from credit.cli import _build_channel_map

        conf = _make_conf(vars_3d=["U", "V"], vars_2d=["SP", "VAR_2T"], diag_2d=["precip", "evap"], n_levels=5)
        cm = _build_channel_map(conf)
        all_channels = sorted(c for chans in cm.values() for c in chans)
        # 2 × 5 = 10 for 3D, 2 × 1 = 2 for prog 2D, 2 × 1 = 2 diag → 14 total
        assert all_channels == list(range(14))

    def test_empty_config(self):
        from credit.cli import _build_channel_map

        conf = _make_conf()
        cm = _build_channel_map(conf)
        assert cm == {}


# ---------------------------------------------------------------------------
# --denorm: postblock-based denormalisation
# ---------------------------------------------------------------------------

T_KEY = "ERA5/prognostic/3d/T"
SP_KEY = "ERA5/prognostic/2d/SP"
MEAN_T, VAR_T = [250.0, 260.0, 270.0], [4.0, 4.0, 4.0]
MEAN_SP, VAR_SP = [101325.0], [4.0e6]


def _make_scaler(tmp_path):
    from bridgescaler import save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = {}
    for key, mean, var in ((T_KEY, MEAN_T, VAR_T), (SP_KEY, MEAN_SP, VAR_SP)):
        s = DStandardScalerTensor(channels_last=False)
        s.mean_x_ = torch.tensor(mean)
        s.var_x_ = torch.tensor(var)
        s.x_columns_ = list(range(len(mean)))
        s.n_ = 100
        s._fit = True
        scalers[key] = s
    path = str(tmp_path / "scaler.json")
    save_scaler_dict({"target": {"ERA5": scalers}}, path)
    return path


def _denorm_conf(scaler_path):
    return {
        "postblocks": {
            "per_step": {
                "reconstruct": {"type": "reconstruct", "args": {"detach": False}},
                "scaler": {
                    "type": "bridgescaler_transform",
                    "args": {"scaler_path": scaler_path, "variables": [], "method": "inverse_transform"},
                },
                # physics fixer: must be skipped (its args would not even construct)
                "fixer": {"type": "global_mass_fixer", "args": {"bogus": True}},
                # target twin: must be skipped
                "reconstruct_target": {"type": "reconstruct", "args": {"in_key": "y", "out_key": "y_target_processed"}},
                "scaler_target": {
                    "type": "bridgescaler_transform",
                    "args": {"scaler_path": scaler_path, "variables": [], "key": "y_target_processed"},
                },
            }
        }
    }


@pytest.fixture
def metadata():
    return {
        "target": {
            "_channel_map": {
                T_KEY: {"slice": slice(0, 3), "orig_shape": (3, 1)},
                SP_KEY: {"slice": slice(3, 4), "orig_shape": (1, 1)},
            }
        }
    }


class TestDenormBlockConfs:
    def test_keeps_only_prediction_side_unit_blocks(self, tmp_path):
        from credit.cli import _denorm_block_confs

        blocks = _denorm_block_confs(_denorm_conf(_make_scaler(tmp_path)))
        assert list(blocks) == ["reconstruct", "scaler"]

    def test_no_inverse_scaler_raises_with_fix(self):
        from credit.cli import _denorm_block_confs

        conf = {"postblocks": {"per_step": {"reconstruct": {"type": "reconstruct"}}}}
        with pytest.raises(ValueError, match="bridgescaler_transform"):
            _denorm_block_confs(conf)

    def test_forward_only_scaler_is_not_an_inverse(self, tmp_path):
        from credit.cli import _denorm_block_confs

        conf = _denorm_conf(_make_scaler(tmp_path))
        conf["postblocks"]["per_step"]["scaler"]["args"]["method"] = "transform"
        with pytest.raises(ValueError):
            _denorm_block_confs(conf)

    def test_missing_postblocks_raises(self):
        from credit.cli import _denorm_block_confs

        with pytest.raises(ValueError):
            _denorm_block_confs({})


class TestDenormalize:
    def test_inverse_scales_to_physical_units(self, tmp_path, metadata):
        from credit.cli import _build_denorm_postblocks, _denormalize

        chain = _build_denorm_postblocks(_denorm_conf(_make_scaler(tmp_path)))
        # Flat model-space tensor (B=1, C=4, H=2, W=3): all zeros -> physical value == mean
        out = _denormalize(chain, metadata, torch.zeros(1, 4, 2, 3))
        assert out[T_KEY].shape == (1, 3, 1, 2, 3)
        for lev, mean in enumerate(MEAN_T):
            assert torch.allclose(out[T_KEY][0, lev, 0], torch.full((2, 3), mean))
        assert torch.allclose(out[SP_KEY][0, 0, 0], torch.full((2, 3), MEAN_SP[0]))

    def test_unit_normalised_value_scales_by_std(self, tmp_path, metadata):
        from credit.cli import _build_denorm_postblocks, _denormalize

        chain = _build_denorm_postblocks(_denorm_conf(_make_scaler(tmp_path)))
        out = _denormalize(chain, metadata, torch.ones(1, 4, 2, 3))
        assert torch.allclose(out[T_KEY][0, 0, 0], torch.full((2, 3), MEAN_T[0] + 2.0))  # std = sqrt(4)
        assert torch.allclose(out[SP_KEY][0, 0, 0], torch.full((2, 3), MEAN_SP[0] + 2000.0))  # std = sqrt(4e6)

    def test_accepts_5d_model_output_with_singleton_time(self, tmp_path, metadata):
        from credit.cli import _build_denorm_postblocks, _denormalize

        chain = _build_denorm_postblocks(_denorm_conf(_make_scaler(tmp_path)))
        out = _denormalize(chain, metadata, torch.zeros(1, 4, 1, 2, 3))
        assert out[SP_KEY].shape == (1, 1, 1, 2, 3)


class TestResolveField:
    KEYS = [T_KEY, SP_KEY, "ERA5/diagnostic/2d/TP"]

    def test_short_name(self):
        from credit.cli import _resolve_field

        assert _resolve_field("SP", self.KEYS) == SP_KEY

    def test_full_key(self):
        from credit.cli import _resolve_field

        assert _resolve_field(T_KEY, self.KEYS) == T_KEY

    def test_unknown_returns_none(self):
        from credit.cli import _resolve_field

        assert _resolve_field("NOPE", self.KEYS) is None
