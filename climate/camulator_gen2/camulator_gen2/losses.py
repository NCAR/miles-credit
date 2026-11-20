"""The HF CAMulator training loss for the gen2 trainer.

HF (``trainerERA5.py`` on ``WillyChap/miles-credit@camulator_huggingface``)
scored each step with the gen1 ``VariableTotalLoss2D``: an MSE in normalized
space, latitude-weighted, averaged per channel and then over channels. With
``conservation_loss_weight > 0`` (0.05 for the released checkpoints) it scored
the prediction saved *before* the water and energy fixers (after the tracer and
mass fixers) and added the water penalty. Validation scored the fully fixed
prediction, without the penalty.

In the gen2 chain this means two ``flatten_to_tensor`` blocks: one after the
mass fixer writing ``y_pred_hf`` (what training scores) and one at the end
writing ``y_pred`` (what validation scores). See ``configs/camulator_hf.yml``.
"""

import os

import torch
import xarray as xr
from torch import nn

from credit.losses.base import BaseLoss


class HFLoss(BaseLoss):
    """Latitude-weighted MSE on the post-mass-fixer prediction, plus the water penalty.

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
        water_penalty_weight: weight of the penalty in training (HF: 0.05);
            ``0`` disables it.
        latitude_weight_power: exponent on the cosine weights (HF default 1).
    """

    def __init__(
        self,
        latitude_weights,
        pred_key="y_pred_hf",
        valid_pred_key="y_pred",
        target_key="y",
        penalty_key="water_conservation_loss",
        water_penalty_weight=0.05,
        latitude_weight_power=1.0,
    ):
        nn.Module.__init__(self)  # BaseLoss.__init__ needs a channel schema this loss does not use
        self.pred_key = pred_key
        self.valid_pred_key = valid_pred_key
        self.target_key = target_key
        self.penalty_key = penalty_key
        self.water_penalty_weight = float(water_penalty_weight)
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

        # HF VariableTotalLoss2D: MSELoss(reduction="none"), weight, mean per channel, mean over channels.
        squared = (target - pred) ** 2
        lat_weights = self.lat_weights.to(squared.device)
        channel_losses = torch.stack([torch.mul(squared[:, i], lat_weights).mean() for i in range(squared.shape[1])])
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
