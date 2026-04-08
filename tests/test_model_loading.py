"""Unit tests for credit.models factory functions.

Covers:
  - load_model()                        — missing/invalid type error paths; conf immutability
  - load_model_name()                   — missing/invalid type error paths; conf immutability
  - load_fsdp_or_checkpoint_policy()    — unsupported type (always); model-specific sets (guarded)

Tests that require legacy model imports (crossformer, swin, fuxi, etc.) are skipped
when those imports are unavailable (e.g. numba/NumPy conflict in CI).
"""

import pytest

try:
    from credit.models import _LEGACY_MODELS_AVAILABLE
except ImportError:
    _LEGACY_MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# load_model — error paths
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_missing_type_key_raises(self):
        from credit.models import load_model

        with pytest.raises(ValueError, match="type"):
            load_model({"model": {}})

    def test_invalid_type_raises(self):
        from credit.models import load_model

        with pytest.raises(ValueError, match="not supported"):
            load_model({"model": {"type": "nonexistent-model-xyz"}})

    def test_original_conf_not_mutated(self):
        """load_model deep-copies conf internally; the caller's dict must be unchanged."""
        from credit.models import load_model

        conf = {"model": {"type": "nonexistent-model-xyz", "extra_key": True}}
        with pytest.raises(ValueError):
            load_model(conf)
        assert conf["model"]["type"] == "nonexistent-model-xyz"
        assert conf["model"]["extra_key"] is True


# ---------------------------------------------------------------------------
# load_model_name — error paths
# ---------------------------------------------------------------------------


class TestLoadModelName:
    def test_missing_type_key_raises(self):
        from credit.models import load_model_name

        with pytest.raises(ValueError, match="type"):
            load_model_name({"model": {}}, "checkpoint.pt")

    def test_invalid_type_raises(self):
        from credit.models import load_model_name

        with pytest.raises(ValueError, match="not supported"):
            load_model_name({"model": {"type": "nonexistent-model-xyz"}}, "checkpoint.pt")

    def test_original_conf_not_mutated(self):
        from credit.models import load_model_name

        conf = {"model": {"type": "nonexistent-model-xyz"}}
        with pytest.raises(ValueError):
            load_model_name(conf, "checkpoint.pt")
        assert conf["model"]["type"] == "nonexistent-model-xyz"


# ---------------------------------------------------------------------------
# load_fsdp_or_checkpoint_policy
# ---------------------------------------------------------------------------


class TestLoadFSDPPolicy:
    def _conf(self, model_type):
        return {"model": {"type": model_type}}

    def test_unsupported_type_raises_os_error(self):
        """A type with no defined FSDP policy must raise OSError without importing legacy deps."""
        from credit.models import load_fsdp_or_checkpoint_policy

        with pytest.raises(OSError):
            load_fsdp_or_checkpoint_policy(self._conf("nonexistent-model-xyz"))

    @pytest.mark.skipif(not _LEGACY_MODELS_AVAILABLE, reason="legacy model deps unavailable in this environment")
    def test_crossformer_returns_nonempty_set(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        result = load_fsdp_or_checkpoint_policy(self._conf("crossformer"))
        assert isinstance(result, set) and len(result) > 0

    @pytest.mark.skipif(not _LEGACY_MODELS_AVAILABLE, reason="legacy model deps unavailable in this environment")
    def test_unet_returns_nonempty_set(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        result = load_fsdp_or_checkpoint_policy(self._conf("unet"))
        assert isinstance(result, set) and len(result) > 0

    @pytest.mark.skipif(not _LEGACY_MODELS_AVAILABLE, reason="legacy model deps unavailable in this environment")
    def test_fuxi_returns_nonempty_set(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        result = load_fsdp_or_checkpoint_policy(self._conf("fuxi"))
        assert isinstance(result, set) and len(result) > 0

    @pytest.mark.skipif(not _LEGACY_MODELS_AVAILABLE, reason="legacy model deps unavailable in this environment")
    def test_wrf_returns_nonempty_set(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        result = load_fsdp_or_checkpoint_policy(self._conf("wrf"))
        assert isinstance(result, set) and len(result) > 0

    @pytest.mark.skipif(not _LEGACY_MODELS_AVAILABLE, reason="legacy model deps unavailable in this environment")
    def test_swin_returns_nonempty_set(self):
        from credit.models import load_fsdp_or_checkpoint_policy

        result = load_fsdp_or_checkpoint_policy(self._conf("swin"))
        assert isinstance(result, set) and len(result) > 0
