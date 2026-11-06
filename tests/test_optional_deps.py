"""Core CREDIT must import without the optional 'spectral' extra (issue #519).

torch-harmonics and segmentation-models-pytorch are only needed by SKEBS, SphericalNoise, the
pole/diffusion filter, and spectral verification. Each check runs in a fresh interpreter with those
modules blocked in ``sys.modules``, so it passes or fails the same whether or not they are installed.
"""

import subprocess
import sys
import textwrap

import pytest

_BLOCK_OPTIONAL = textwrap.dedent(
    """
    import sys
    for name in ("torch_harmonics", "segmentation_models_pytorch"):
        sys.modules[name] = None  # makes `import name` raise ImportError
    """
)


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_OPTIONAL + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


@pytest.mark.parametrize(
    "statement",
    [
        "from credit.models.camulator import Camulator",
        "from credit.models.crossformer import CrossFormer",
        "from credit.models.wxformer.crossformer import CrossFormer",
        "from credit.models.fuxi import Fuxi",
        "from credit.models.swin import SwinTransformerV2Cr",
        "from credit.postblock.gen1 import PostBlock, GlobalMassFixer",
        "import credit.skebs",
        "import credit.pol_lapdiff_filt",
        "import credit.ensemble.spherical",
        "import credit.verification.standard",
    ],
)
def test_import_without_spectral_extra(statement):
    _run(statement)


def test_postblock_without_skebs_builds():
    _run(
        """
        from credit.postblock.gen1 import PostBlock
        conf = {
            "skebs": {"activate": False},
            "tracer_fixer": {"activate": False},
            "global_mass_fixer": {"activate": False},
            "global_water_fixer": {"activate": False},
            "global_energy_fixer": {"activate": False},
        }
        PostBlock(conf)
        """
    )


def test_skebs_without_extra_raises_actionable_error():
    out = _run(
        """
        from credit.skebs import load_premade_encoder_model
        import torch
        from credit.ensemble.spherical import SphericalNoise
        for build in (
            lambda: load_premade_encoder_model({"name": "unet"}),
            lambda: SphericalNoise(amplitude=0.1, smoothness=2.0, length_scale=3.0)(torch.zeros(1, 2, 1, 16, 32)),
        ):
            try:
                build()
            except ImportError as err:
                assert "miles-credit[spectral]" in str(err), err
                print("ok")
        """
    )
    assert out.split() == ["ok", "ok"]
