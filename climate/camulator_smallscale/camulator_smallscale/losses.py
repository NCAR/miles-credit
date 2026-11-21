"""Losses for the small-scale experiments, built on camulator_gen2's ``HFLoss``.

- :class:`AMSEHFLoss` (``amse_hf_loss``): HF's training loss plus the AMSE adjustment of Subich et al. (2025),
  "Fixing the double penalty in data-driven weather forecasting through a modified spherical harmonic loss
  function" (arXiv:2501.19374).
"""

import torch

from camulator_gen2.losses import HFLoss
from camulator_smallscale.spectral import PARSEVAL, SphericalSpectra


class AMSEHFLoss(HFLoss):
    """``HFLoss`` plus the AMSE adjustment in spherical-harmonic space.

    By total wavenumber ``l``, the area-weighted MSE between prediction ``x`` and truth ``y`` splits into an
    amplitude and a decorrelation term, ``MSE_l = (sqrt(Px) - sqrt(Py))**2 + 2 sqrt(Px Py) (1 - coh_l)``. AMSE
    replaces ``sqrt(Px Py)`` with ``max(Px, Py)``, so a prediction that has decorrelated from the truth at some scale
    can no longer lower the loss by damping that scale. AMSE = MSE + the adjustment

        adj_l = 2 [max(Px, Py) - sqrt(Px Py)] (1 - coh_l) >= 0.

    This loss is ``HFLoss`` plus ``amse_weight * sum_l adj_l / (4 pi)`` per channel (``1 / (4 pi)`` puts a spectral
    sum in the units of a grid mean with mean-1 area weights), times HF's channel weight, averaged over channels like
    HF's channel losses:

    - in training: ``HFLoss`` (MSE on the post-mass-fixer prediction, cos(lat)**0.4, channel weights, water
      penalty) plus the adjustment;
    - in validation (under ``torch.no_grad()``, as the gen2 trainer validates): exactly ``HFLoss``, so validation
      curves compare directly with the null fine-tune.

    The adjustment is zero when prediction and truth have the same spectra, so the loss then equals ``HFLoss`` (bit
    for bit when the two fields are equal). With ``amse_weight`` 1, a prediction whose phases have fully decorrelated
    from the truth's is best at the truth's amplitude, where ``HFLoss`` alone would damp it.

    The constraint is on the global spectrum only: it does not fix where (in latitude) the variance sits. With HF's
    cos(lat)**0.4 weights, which put more weight on high latitudes than area weighting does, the MSE's damping
    incentive survives partly there (in a synthetic test: full amplitude in the tropics, 0.8 poleward of 30 deg).

    Args:
        latitude_weights: NetCDF file with ``latitude`` / ``longitude`` (as for ``HFLoss``).
        amse_weight: multiplier on the adjustment (1 is AMSE).
        rel_eps: floor on ``Px Py`` relative to the truth's own power at each ``l`` (``rel_eps * Py**2``); keeps the
            coherence and its gradient finite where the prediction has no power.
        lmax, mmax: spherical-harmonic truncation (default: triangular at the zonal Nyquist).
        amse_variables: variable names that get the adjustment (default: all target variables).
        **hf_kwargs: the ``HFLoss`` arguments (``train_loss`` must be ``"mse"``).
    """

    def __init__(
        self,
        latitude_weights,
        amse_weight=1.0,
        rel_eps=1e-6,
        lmax=None,
        mmax=None,
        amse_variables=None,
        **hf_kwargs,
    ):
        super().__init__(latitude_weights, **hf_kwargs)
        if self.train_loss != "mse":
            raise ValueError(f"AMSEHFLoss adds the AMSE adjustment to an MSE; got train_loss={self.train_loss!r}")
        self.amse_weight = float(amse_weight)
        self.rel_eps = float(rel_eps)
        self.amse_variables = list(amse_variables) if amse_variables else None
        self._channel_mask = None
        nlat, nlon = self.lat_weights.shape
        self.spectra = SphericalSpectra(nlat, nlon, lmax=lmax, mmax=mmax)

    def forward(self, full_data_dict: dict) -> torch.Tensor:
        loss = super().forward(full_data_dict)
        if not torch.is_grad_enabled() or self.amse_weight == 0:
            return loss
        pred = full_data_dict[self.pred_key]
        target = full_data_dict[self.target_key].to(pred.dtype)
        adjustment = self.amse_weight * self.channel_adjustment(pred, target)  # (C,)
        weights = self._weights_per_channel(full_data_dict, pred.shape[1])
        if weights is not None:
            adjustment = adjustment * weights.to(device=adjustment.device, dtype=adjustment.dtype)
        mask = self._mask_per_channel(full_data_dict, pred.shape[1])
        if mask is not None:
            adjustment = adjustment * mask.to(adjustment.device)
        per_variable = self._per_variable(adjustment.detach(), full_data_dict)
        self.last_var_losses.update({f"{name}_amse": value for name, value in per_variable.items()})
        return loss + adjustment.mean()

    def channel_adjustment(self, pred, target):
        """Per-channel AMSE adjustment ``sum_l adj_l / (4 pi)``, averaged over every dim but the channel (dim 1)."""
        spectra = self.spectra
        if spectra.m_weights.device != pred.device:
            spectra.to(pred.device)
        cx, cy = spectra.transform(pred), spectra.transform(target)
        px, py = spectra.power_from(cx), spectra.power_from(cy)
        cross = spectra.cross_from(cx, cy)
        eps = self.rel_eps * py.detach() ** 2 + torch.finfo(py.dtype).tiny
        geometric = torch.sqrt(px * py + eps)
        # Both factors are >= 0 in exact arithmetic; the clamps keep it so (and make it exactly 0 when pred == target).
        adj = 2 * (torch.maximum(px, py) - geometric).clamp(min=0) * (1 - cross / geometric).clamp(min=0)
        adj = PARSEVAL * adj.sum(-1)  # (B, C, ...)
        return adj.transpose(0, 1).reshape(adj.shape[1], -1).mean(-1)

    def _mask_per_channel(self, full_data_dict, n_channels):
        """1 for channels of ``amse_variables`` and 0 elsewhere; ``None`` when every variable gets the adjustment."""
        if self.amse_variables is None:
            return None
        if self._channel_mask is not None:
            return self._channel_mask
        channel_map = (full_data_dict.get("metadata") or {}).get("target", {}).get("_channel_map")
        if not channel_map:
            raise KeyError("AMSEHFLoss: amse_variables needs metadata['target']['_channel_map'] in the batch dict.")
        mask = torch.zeros(n_channels)
        names = set()
        for var_key, info in channel_map.items():
            name = var_key.split("/")[-1]
            names.add(name)
            if name in self.amse_variables:
                mask[info["slice"]] = 1.0
        unknown = set(self.amse_variables) - names
        if unknown:
            raise KeyError(f"AMSEHFLoss: amse_variables names variables that are not targets: {sorted(unknown)}")
        self._channel_mask = mask
        return mask
