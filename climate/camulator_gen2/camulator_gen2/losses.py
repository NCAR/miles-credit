"""The HF CAMulator training loss for the gen2 trainer.

HF (``trainerERA5.py`` on ``WillyChap/miles-credit@camulator_huggingface``)
scored each step with the gen1 ``VariableTotalLoss2D``: a loss in normalized
space, weighted by latitude and by variable, averaged per channel and then over
channels. With ``conservation_loss_weight > 0`` (0.1 for the run that produced
the released checkpoint) training scored the MSE of the prediction saved
*before* the water and energy fixers (after the tracer and mass fixers) and
added the water penalty. Validation scored the fully fixed prediction at the
last step with the same weights, without the penalty, and with an L1 loss: the
gen1 loss falls back to ``torch.nn.L1Loss`` when ``loss.validation_loss`` is
not set, and HF's config did not set it.

In the gen2 chain this means two ``flatten_to_tensor`` blocks: one after the
mass fixer writing ``y_pred_hf`` (what training scores) and one at the end
writing ``y_pred`` (what validation scores). See ``configs/camulator_hf.yml``.
"""

import os

import torch
import xarray as xr
from torch import nn

from credit.losses.base import BaseLoss


_ELEMENTWISE = {
    "mse": lambda target, pred: (target - pred) ** 2,
    "mae": lambda target, pred: (target - pred).abs(),
}


class HFLoss(BaseLoss):
    """Weighted MSE on the post-mass-fixer prediction plus the water penalty; weighted MAE in validation.

    This subclasses :class:`credit.losses.base.BaseLoss` only so the gen2
    trainer passes it the whole batch dict (it checks ``isinstance(criterion,
    BaseLoss)``); none of BaseLoss's behavior is used. ``load_loss`` builds a
    custom loss from its ``args`` alone, without a ``validation`` flag, so
    validation is detected from autograd: the trainer validates under
    ``torch.no_grad()``.

    Args:
        latitude_weights: NetCDF file with a ``latitude`` coordinate (HF used the
            ``f.e21...statics...nc`` file).
        pred_key: flat normalized prediction scored in training.
        valid_pred_key: flat normalized prediction scored in validation.
        target_key: flat normalized target.
        penalty_key: batch-dict key holding the water penalty (``HFWaterFixer``).
        water_penalty_weight: weight of the penalty in training (HF: 0.1);
            ``0`` disables it.
        latitude_weight_power: exponent on the cosine weights (HF: 0.4).
        variable_weights: ``{variable name: weight}``, a float or, for a 3-D
            variable, one weight per level; unlisted variables get 1. ``None``
            turns variable weighting off.
        train_loss: ``"mse"`` (HF's ``training_loss``).
        valid_loss: ``"mae"`` (HF's fallback without ``validation_loss``) or ``"mse"``.
    """

    def __init__(
        self,
        latitude_weights,
        pred_key="y_pred_hf",
        valid_pred_key="y_pred",
        target_key="y",
        penalty_key="water_conservation_loss",
        water_penalty_weight=0.1,
        latitude_weight_power=0.4,
        variable_weights=None,
        train_loss="mse",
        valid_loss="mae",
    ):
        nn.Module.__init__(self)  # BaseLoss.__init__ needs a channel schema this loss does not use
        for name, value in (("train_loss", train_loss), ("valid_loss", valid_loss)):
            if value not in _ELEMENTWISE:
                raise ValueError(f"HFLoss: {name} must be one of {sorted(_ELEMENTWISE)}, got {value!r}")
        self.pred_key = pred_key
        self.valid_pred_key = valid_pred_key
        self.target_key = target_key
        self.penalty_key = penalty_key
        self.water_penalty_weight = float(water_penalty_weight)
        self.variable_weights = dict(variable_weights) if variable_weights else None
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self._channel_weights = None  # built from the batch's channel map on the first call
        # attributes the gen2 trainer / train_gen2 read from a BaseLoss
        self.log_variance = None
        self.last_var_losses = {}

        ds = xr.open_dataset(os.path.expandvars(latitude_weights))
        lat = torch.from_numpy(ds["latitude"].values).float()
        lon_dim = ds["longitude"].shape[0]
        ds.close()
        # HF get_lat_weights: the clamp matters, because the CAMulator grid includes the poles.
        weights = torch.cos(torch.deg2rad(lat)).clamp(min=1e-4) ** latitude_weight_power
        weights = weights / weights.mean()
        self.register_buffer("lat_weights", weights.unsqueeze(1).expand(-1, lon_dim).contiguous(), persistent=False)

    def forward(self, full_data_dict: dict) -> torch.Tensor:
        training = torch.is_grad_enabled()
        pred_key = self.pred_key if training else self.valid_pred_key
        if pred_key not in full_data_dict:
            raise KeyError(
                f"HFLoss: '{pred_key}' is not in the batch dict. Add a flatten_to_tensor postblock "
                f"with out_key: {pred_key} (after the mass fixer for '{self.pred_key}', at the end of "
                f"the chain for '{self.valid_pred_key}')."
            )
        pred = full_data_dict[pred_key]
        target = full_data_dict[self.target_key].to(pred.dtype)

        # HF VariableTotalLoss2D: elementwise loss, latitude weight, variable weight, mean per channel,
        # mean over channels.
        elementwise = _ELEMENTWISE[self.train_loss if training else self.valid_loss](target, pred)
        lat_weights = self.lat_weights.to(elementwise.device)
        var_weights = self._weights_per_channel(full_data_dict, elementwise.shape[1])
        channel_losses = []
        for i in range(elementwise.shape[1]):
            var_loss = torch.mul(elementwise[:, i], lat_weights)
            if var_weights is not None:
                # A 0-dim float64 weight times a float32 tensor computes in float32 anyway; casting first gives the
                # same result and works on Apple MPS, which has no float64.
                var_loss = var_loss * var_weights[i].to(device=var_loss.device, dtype=var_loss.dtype)
            channel_losses.append(var_loss.mean())
        channel_losses = torch.stack(channel_losses)
        loss = torch.mean(channel_losses)

        if training and self.water_penalty_weight > 0:
            if self.penalty_key not in full_data_dict:
                raise KeyError(
                    f"HFLoss: water_penalty_weight is {self.water_penalty_weight} but '{self.penalty_key}' "
                    "is not in the batch dict. Use the HFWaterFixer postblock, or set water_penalty_weight: 0."
                )
            loss = loss + self.water_penalty_weight * full_data_dict[self.penalty_key]

        self.last_var_losses = self._per_variable(channel_losses.detach(), full_data_dict)
        return loss

    def _weights_per_channel(self, full_data_dict, n_channels):
        """One weight per target channel, in the batch's channel order (HF used float64 weights)."""
        if self.variable_weights is None:
            return None
        if self._channel_weights is not None:
            return self._channel_weights
        channel_map = (full_data_dict.get("metadata") or {}).get("target", {}).get("_channel_map")
        if not channel_map:
            raise KeyError("HFLoss: variable_weights needs metadata['target']['_channel_map'] in the batch dict.")
        weights = torch.ones(n_channels, dtype=torch.float64)
        names = set()
        for var_key, info in channel_map.items():
            name = var_key.split("/")[-1]
            names.add(name)
            if name not in self.variable_weights:
                continue
            value = self.variable_weights[name]
            n = info["slice"].stop - info["slice"].start
            values = torch.as_tensor(value if isinstance(value, (list, tuple)) else [value] * n, dtype=torch.float64)
            if values.numel() != n:
                raise ValueError(f"HFLoss: variable_weights['{name}'] has {values.numel()} values for {n} channels.")
            weights[info["slice"]] = values
        unknown = set(self.variable_weights) - names
        if unknown:
            raise KeyError(f"HFLoss: variable_weights names variables that are not targets: {sorted(unknown)}")
        self._channel_weights = weights
        return weights

    @staticmethod
    def _per_variable(channel_losses, full_data_dict):
        """Mean channel loss per variable, for the trainer's per-variable log columns."""
        channel_map = (full_data_dict.get("metadata") or {}).get("target", {}).get("_channel_map")
        if not channel_map:
            return {}
        return {
            var_key.split("/")[-1]: channel_losses[info["slice"]].mean().item()
            for var_key, info in channel_map.items()
            if info["slice"].stop <= channel_losses.shape[0]
        }
