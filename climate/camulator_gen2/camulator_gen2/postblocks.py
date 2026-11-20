"""Postblocks that reproduce the gen1 CAMulator fixers inside the gen2 chain.

Reference implementation: ``credit/postblock/_postblock.py`` and
``credit/trainers/trainerERA5.py`` on ``WillyChap/miles-credit@camulator_huggingface``
(the code behind the Hugging Face checkpoints), called "HF" below.

The gen2 mass, water and energy fixers in ``credit.postblock.conservation``
already match HF on CAMulator's hybrid-sigma grid. Two things are missing:

- HF's tracer fixer runs inside the model, so before the wind filter and the
  other fixers, and it works in normalized space with per-channel statistics
  from the gen1 mean/std NetCDF files. For a 3-D tracer such as Qtot it uses
  the level-0 statistics for every level (see ``replicate_level0_stats_bug``).
- HF's trainer adds a water-conservation penalty to the loss, computed by the
  water fixer before it corrects precipitation.
"""

import os

import numpy as np
import torch
import xarray as xr

from credit.postblock.base import BasePostblock
from credit.postblock.conservation import GlobalWaterFixer


def _per_var(value, n, name):
    """Broadcast a scalar to ``n`` entries, or check a list has ``n`` entries."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"HFTracerFixer: {name} has {len(value)} entries for {n} tracer_vars.")
        return list(value)
    return [value] * n


class HFTracerFixer(BasePostblock):
    """Clip tracers the way the HF CAMulator ``TracerFixer`` does.

    For each tracer channel the HF fixer de-normalizes with the channel's
    mean/std, clips to ``[tracer_thres, tracer_thres_max]`` in physical units,
    and re-normalizes. This block does the same float32 arithmetic on the
    normalized ``y_processed`` that ``reconstruct`` writes, so it must sit
    directly after ``reconstruct`` and before ``wind_artifact_filter`` and the
    inverse scaler, which is where HF ran it (inside ``Camulator.forward``).

    The statistics come from the gen1 mean/std NetCDF files, as in HF. HF built
    its per-channel list from the variable name alone and took
    ``values.flatten()[0]``, so every level of a 3-D tracer used the level-0
    (model-top) mean and std. With the shipped statistics that turns the Qtot
    clip at 0 into a clip at a normalized value of about -10.8 on every level,
    which leaves levels 1-31 effectively unclipped. ``replicate_level0_stats_bug``
    keeps that behavior (the default, to reproduce HF); set it to ``False`` to
    use each level's own statistics.

    Args:
        mean_path: gen1 mean NetCDF (e.g. ``mean_6h_..._Mixed_Modal.nc``).
        std_path: gen1 std NetCDF.
        tracer_vars: ``y_processed`` keys to clip, e.g. ``CESM/prognostic/3d/Qtot``.
        tracer_thres: lower bound(s) in physical units; a scalar or one per var.
        tracer_thres_max: upper bound(s) in physical units; ``None``, or one per
            var with ``None`` for no upper bound.
        replicate_level0_stats_bug: use level-0 statistics for every level of a
            3-D tracer, as HF did.
        key: batch-dict key holding the nested prediction.
    """

    def __init__(
        self,
        mean_path,
        std_path,
        tracer_vars,
        tracer_thres=0.0,
        tracer_thres_max=None,
        replicate_level0_stats_bug=True,
        key="y_processed",
    ):
        super().__init__()
        self.tracer_vars = list(tracer_vars)
        n = len(self.tracer_vars)
        self.tracer_thres = [float(v) for v in _per_var(tracer_thres, n, "tracer_thres")]
        thres_max = _per_var(tracer_thres_max, n, "tracer_thres_max")
        self.tracer_thres_max = [None if v is None else float(v) for v in thres_max]
        self.replicate_level0_stats_bug = bool(replicate_level0_stats_bug)
        self.key = key

        mean_ds = xr.open_dataset(os.path.expandvars(mean_path))
        std_ds = xr.open_dataset(os.path.expandvars(std_path))
        self.means, self.stds = [], []
        for var_key in self.tracer_vars:
            name = var_key.split("/")[-1]
            m = np.array(mean_ds[name].values).flatten()
            s = np.array(std_ds[name].values).flatten()
            if self.replicate_level0_stats_bug:
                m = np.full(m.shape, m[0])
                s = np.full(s.shape, s[0])
            # HF: Python floats -> torch.tensor(...).float(), i.e. float64 values rounded to float32.
            self.means.append(torch.tensor([float(v) for v in m]).float())
            self.stds.append(torch.tensor([float(v) for v in s]).float())
        mean_ds.close()
        std_ds.close()

    def forward(self, batch_dict: dict) -> dict:
        prediction = batch_dict[self.key]
        for i, var_key in enumerate(self.tracer_vars):
            source = var_key.split("/")[0]
            tensor = prediction[source][var_key]  # (B, levels, T, ...spatial), normalized
            n_levels = tensor.shape[1]
            if self.means[i].numel() != n_levels:
                raise ValueError(
                    f"HFTracerFixer: {var_key} has {n_levels} levels but its statistics have "
                    f"{self.means[i].numel()} entries."
                )
            view = (1, n_levels) + (1,) * (tensor.dim() - 2)
            m = self.means[i].to(tensor.device).view(view)
            s = self.stds[i].to(tensor.device).view(view)

            # HF: chan = y.float() * s + m; chan[chan < lo] = lo; chan[chan >= hi] = hi; y = (chan - m) / s
            chan = tensor.float() * s + m
            lo = self.tracer_thres[i]
            chan = torch.where(chan < lo, torch.full_like(chan, lo), chan)
            hi = self.tracer_thres_max[i]
            if hi is not None:
                chan = torch.where(chan >= hi, torch.full_like(chan, hi), chan)
            prediction[source][var_key] = ((chan - m) / s).to(tensor.dtype)
        return batch_dict


class HFWaterFixer(GlobalWaterFixer, BasePostblock):
    """The gen2 water fixer, plus the HF trainer's water-conservation penalty.

    HF's ``GlobalWaterFixer`` returned ``(P_correct_ratio - 1)**2`` averaged over
    the batch, computed from the prediction *before* the precipitation
    correction, and its trainer added ``conservation_loss_weight`` times that to
    the loss. This block computes the same quantity, keeping its gradient, and
    stores it under ``penalty_key`` for :class:`~camulator_gen2.losses.HFLoss`.
    The correction itself is the unchanged gen2 ``GlobalWaterFixer``.

    Args:
        penalty_key: batch-dict key the penalty is written to.
        **kwargs: arguments of :class:`credit.postblock.conservation.GlobalWaterFixer`.
    """

    def __init__(self, penalty_key="water_conservation_loss", **kwargs):
        super().__init__(**kwargs)
        self.penalty_key = penalty_key

    def _prediction(self, batch_dict, var_key):
        return batch_dict["y_processed"][var_key.split("/")[0]][var_key]

    def precip_correction_ratio(self, batch_dict: dict) -> torch.Tensor:
        """Per-sample ``P_correct_ratio``, computed exactly as ``GlobalWaterFixer.forward`` does."""
        q_pred = self._prediction(batch_dict, self.q_var)[:, :, 0, ...]
        sp_pred = self._prediction(batch_dict, self.sp_var)[:, 0, 0, ...]
        precip = self._prediction(batch_dict, self.precip_var)[:, 0, 0, ...]
        evapor = self._prediction(batch_dict, self.evapor_var)[:, 0, 0, ...]
        q_input = self._input(batch_dict, self.q_var)[:, :, -1, ...]
        sp_input = self._input(batch_dict, self.sp_var)[:, 0, -1, ...]
        device = q_pred.device

        precip_flux = precip * self.precip_scale
        evapor_flux = evapor * self.evapor_scale

        TWC_input = self.core.total_column_water(q_input.to(device), sp_input.to(device))
        TWC_pred = self.core.total_column_water(q_pred, sp_pred)
        dTWC_dt = (TWC_pred - TWC_input) / self.N_seconds

        TWC_sum = self.core.weighted_sum(dTWC_dt, axis=(-2, -1))
        E_sum = self.core.weighted_sum(evapor_flux, axis=(-2, -1))
        P_sum = self.core.weighted_sum(precip_flux, axis=(-2, -1))

        residual = -TWC_sum - E_sum - P_sum
        return (P_sum + residual) / P_sum

    def forward(self, batch_dict: dict) -> dict:
        if not self.flag_sigma:
            raise NotImplementedError("HFWaterFixer supports the hybrid-sigma grid only.")
        ratio = self.precip_correction_ratio(batch_dict)
        batch_dict[self.penalty_key] = (ratio - 1.0).pow(2).mean()
        return super().forward(batch_dict)
