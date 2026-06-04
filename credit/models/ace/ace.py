"""
ACE2 (AI2 Climate Emulator v2) — CREDIT port.

Watt-Meyer et al. 2024.  arXiv:2411.11268
Reference code: https://github.com/ai2cm/ace  (Apache-2.0)

Architecture
------------
ACE2 uses a Spherical Fourier Neural Operator (SFNO, Bonev et al. 2023) with one
key difference from a generic SFNO: it takes **two consecutive input time steps**
concatenated along the channel dimension and outputs a single next time step.
This gives the model explicit access to the time tendency (T - T_prev) so it can
learn stable long-rollout dynamics without drift.

  Input : (B, 2*C, H, W)  — two consecutive states concatenated along channels
  Output: (B, C_out, H, W)

In CREDIT the two-step input is handled automatically via history_len=2 in the
data config; the model receives (B, C, T=2, H, W) which is reshaped to (B, 2C, H, W).

Default hyperparameters match the ACE2-ERA5 configuration:
  1° global grid (181×360), no patching (patch_size=1)
  embed_dim=512, depth=12 blocks, spectral truncation at 90 modes

ACE2-ERA5 pretrained weights require the fme package from AI2 and are not
directly loadable without weight remapping.  This implementation is suitable
for training from scratch on CREDIT-format ERA5 data with the same architecture.
"""

import os

import torch
import torch.nn as nn

from credit.models.sfno.sfno import SFNOModel, _HARMONICS_AVAILABLE

import logging

logger = logging.getLogger(__name__)


class CREDITACE(nn.Module):
    """ACE2 climate emulator architecture for CREDIT training.

    Wraps SFNOModel with two-step input handling.  Set ``history_len: 2`` in
    the data config so each sample contains two consecutive time steps.

    Parameters
    ----------
    in_channels : int
        Prognostic channel count **per time step**.  The model internally
        processes ``in_channels * 2`` channels (two concatenated steps).
    out_channels : int
        Output channels (single next time step).
    img_size : (H, W)
        Lat/lon grid size.
    embed_dim : int
        Internal SFNO embedding dimension.
    depth : int
        Number of SFNO blocks.
    n_modes : int, optional
        Spectral truncation (default: ``min(H, W//2+1)``).
    mlp_ratio : float
        MLP expansion ratio inside each SFNO block.
    drop_rate : float
        Dropout probability.
    use_harmonics : bool
        Use spherical harmonic transforms when torch-harmonics is installed.
    """

    def __init__(
        self,
        in_channels: int = 76,
        out_channels: int = 76,
        img_size: tuple = (181, 360),
        embed_dim: int = 512,
        depth: int = 12,
        n_modes: int = 90,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        use_harmonics: bool = True,
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W

        # ACE concatenates 2 time steps → 2× the channel count seen by the SFNO
        self.model = SFNOModel(
            img_size=(H, W),
            patch_size=1,
            in_channels=in_channels * 2,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            n_modes=n_modes,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            use_harmonics=use_harmonics,
        )

        if not _HARMONICS_AVAILABLE and use_harmonics:
            logger.warning(
                "torch-harmonics not installed; ACE SFNO falling back to rfft2. "
                "Install with: pip install torch-harmonics"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T=2, H, W) or (B, 2C, H, W)
            Two consecutive time steps.  The 5-D form is expected when
            history_len=2 in the CREDIT data config.

        Returns
        -------
        (B, C_out, 1, H, W)
        """
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)
        out = self.model(x)
        return out.unsqueeze(2)

    @classmethod
    def load_model(cls, conf):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "model_checkpoint.pt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def load_model_name(cls, conf, model_name):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        ckpt = os.path.join(os.path.expandvars(conf["save_loc"]), model_name)
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model


if __name__ == "__main__":
    import sys

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, _root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CREDITACE(
        in_channels=8,
        out_channels=8,
        img_size=(32, 64),
        embed_dim=64,
        depth=4,
        n_modes=16,
    ).to(device)

    x = torch.randn(1, 8, 2, 32, 64, device=device)
    y = model(x)
    assert y.shape == (1, 8, 1, 32, 64), f"unexpected shape {y.shape}"
    y.mean().backward()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CREDITACE OK — input {x.shape} → output {y.shape}, {n_params:.1f}M params, device {device}")
