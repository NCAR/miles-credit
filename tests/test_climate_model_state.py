"""climate/Model_State.py must apply the conservation fixers its config asks for.

Model_State imported the fixers from ``credit.postblock``, which exports the gen2
classes and no ``GlobalEnergyFixer``. The ImportError was swallowed, so
``Quick_Climate.py`` and the coupled-CESM ``camulator_server.py`` rolled out with
no mass/water/energy fixers and only a warning.
"""

import importlib.util
from pathlib import Path

import pytest

CLIMATE_DIR = Path(__file__).resolve().parents[1] / "climate"


@pytest.fixture
def model_state(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIMATE_DIR))  # Model_State does `from WindPP import ...`
    spec = importlib.util.spec_from_file_location("climate_model_state_under_test", CLIMATE_DIR / "Model_State.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post_conf(active):
    return {
        "activate": True,
        "global_mass_fixer": {"activate": active},
        "global_water_fixer": {"activate": active},
        "global_energy_fixer": {"activate": active},
    }


def _bare_stepper(module, post_conf):
    """A CAMulatorStepper with only the config, so _setup_postprocessing runs without a model."""
    stepper = module.CAMulatorStepper.__new__(module.CAMulatorStepper)
    stepper.conf = {"model": {"post_conf": post_conf}}
    return stepper


def test_imports_the_gen1_fixers(model_state):
    from credit.postblock import gen1

    assert model_state.POSTBLOCK_AVAILABLE
    assert model_state.GlobalMassFixer is gen1.GlobalMassFixer
    assert model_state.GlobalWaterFixer is gen1.GlobalWaterFixer
    assert model_state.GlobalEnergyFixer is gen1.GlobalEnergyFixer


def test_builds_activated_fixers_from_post_conf(model_state, monkeypatch):
    built = []

    def stub(name):
        def make(post_conf):
            built.append((name, post_conf))
            return name

        return make

    for name in ("GlobalMassFixer", "GlobalWaterFixer", "GlobalEnergyFixer"):
        monkeypatch.setattr(model_state, name, stub(name))

    post_conf = _post_conf(active=True)
    stepper = _bare_stepper(model_state, post_conf)
    stepper._setup_postprocessing()

    assert (stepper.flag_mass, stepper.flag_water, stepper.flag_energy) == (True, True, True)
    assert [name for name, _ in built] == ["GlobalMassFixer", "GlobalWaterFixer", "GlobalEnergyFixer"]
    assert all(conf is post_conf for _, conf in built)  # gen1 signature: the whole post_conf


def test_activated_fixers_fail_loudly_when_unavailable(model_state, monkeypatch):
    monkeypatch.setattr(model_state, "POSTBLOCK_AVAILABLE", False)
    with pytest.raises(ImportError, match="activates conservation fixers"):
        _bare_stepper(model_state, _post_conf(active=True))._setup_postprocessing()


def test_inactive_fixers_do_not_need_the_import(model_state, monkeypatch):
    monkeypatch.setattr(model_state, "POSTBLOCK_AVAILABLE", False)
    stepper = _bare_stepper(model_state, _post_conf(active=False))
    stepper._setup_postprocessing()
    assert not (stepper.flag_mass or stepper.flag_water or stepper.flag_energy)
