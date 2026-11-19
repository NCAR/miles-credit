"""Model and rollout modules must import without SKEBS's dependencies.

``credit.skebs`` needs ``torch_harmonics`` and ``segmentation_models_pytorch``,
but SKEBS only runs when ``post_conf.skebs.activate`` is set. Downstream
packages (e.g. earth2studio) import the architecture and post-blocks for
inference only, so importing a model must not require either package.

Each import runs in a fresh interpreter with both packages blocked, because
they are installed in the test environment and may already be imported.
"""

import subprocess
import sys

import pytest

_BLOCK_AND_IMPORT = """
import importlib
import importlib.abc
import sys

BLOCKED = ("torch_harmonics", "segmentation_models_pytorch")


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)", name=name)
        return None


sys.meta_path.insert(0, _Block())
importlib.import_module(sys.argv[1])
"""

MODULES = [
    "credit.postblock.gen1",
    "credit.models.camulator",
    "credit.models.crossformer",
    "credit.models.wxformer.crossformer",
    "credit.models.fuxi",
    "credit.models.swin",
    "credit.applications.rollout_gen2",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports_without_skebs_dependencies(module):
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_AND_IMPORT, module],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"importing {module} needs a SKEBS dependency:\n{result.stderr}"


def test_postblock_still_builds_skebs_when_activated(monkeypatch):
    """The SKEBS import moved inside PostBlock.__init__; it must still be used when active."""
    import credit.skebs
    from torch import nn

    from credit.postblock.gen1 import PostBlock

    class _StubSKEBS(nn.Module):
        def __init__(self, post_conf):
            super().__init__()

    monkeypatch.setattr(credit.skebs, "SKEBS", _StubSKEBS)
    off = {"activate": False}
    post_conf = {
        "skebs": {"activate": True},
        "tracer_fixer": off,
        "global_mass_fixer": off,
        "global_water_fixer": off,
        "global_energy_fixer": off,
    }
    block = PostBlock(post_conf)
    assert [type(op) for op in block.operations] == [_StubSKEBS]
