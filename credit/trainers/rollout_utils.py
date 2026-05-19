"""rollout_utils.py — utilities for autoregressive multi-step rollout."""

import logging

logger = logging.getLogger(__name__)


def assemble_rollout_batch(
    corrected_pred: dict,
    ic_preprocessed: dict,
    curr_batch: dict,
) -> dict:
    """Assemble a batch dict for the rollout preblock pass at autoregressive step t > 0.

    Constructs a dataset-schema batch by routing each variable from the
    appropriate source:

    - **prognostic / diagnostic** channels: from ``corrected_pred`` — the
      postblock-corrected prediction in physical units from the previous step.
    - **dynamic_forcing** channels: from ``curr_batch["input"]`` — the
      current step's time-varying forcing loaded by the dataset.
    - **static** (and any other non-predicted) channels: from
      ``ic_preprocessed["input"]`` — the t=0 batch after IC-only preblocks
      (e.g. regrid), so statics are already on the model grid.

    The assembled dict is passed to ``apply_step_preblocks()``
    which handles per-step operations (dynamic-forcing regrid, log_transform,
    concat).  ``curr_batch["target"]`` is forwarded so preblocks normalize the
    training target in the same pass.

    Args:
        corrected_pred: Nested ``{source: {var_key: tensor(B, n_lev, n_time, H, W)}}``
            in physical units — output of the postblock chain at the previous
            step.  Must be a dict; if postblocks are not configured, include at
            least ``Reconstruct`` so this function receives the expected type.
        ic_preprocessed: t=0 batch after IC-only preblocks.
            Provides time-invariant static variables already mapped to the model
            grid, and the authoritative list of all input variable keys.
        curr_batch: Current step's raw batch from the dataset.  Provides the
            dynamic forcing fields for this step and the training target.

    Returns:
        dict with keys ``"input"`` (nested source→var dict) and ``"target"``
        (from ``curr_batch``), ready for ``apply_step_preblocks()``.

    Raises:
        TypeError: if ``corrected_pred`` is not a dict, which usually means
            ``Reconstruct`` was not included in the postblock chain.
    """
    if not isinstance(corrected_pred, dict):
        raise TypeError(
            "assemble_rollout_batch: corrected_pred must be a nested dict "
            "{source: {var_key: tensor}}. "
            "For multi-step rollout, 'Reconstruct' must be the first postblock. "
            f"Got {type(corrected_pred).__name__}."
        )

    assembled_input: dict = {}

    for source, source_vars in ic_preprocessed["input"].items():
        assembled_input[source] = {}
        curr_source = curr_batch.get("input", {}).get(source, {})
        pred_source = corrected_pred.get(source, {})

        for var_key, ic_tensor in source_vars.items():
            parts = var_key.split("/")
            field_type = parts[1] if len(parts) > 1 else ""

            if field_type in ("prognostic", "diagnostic"):
                if var_key in pred_source:
                    assembled_input[source][var_key] = pred_source[var_key]
                else:
                    logger.warning(
                        "assemble_rollout_batch: '%s' not in corrected_pred; carrying forward from ic_preprocessed.",
                        var_key,
                    )
                    assembled_input[source][var_key] = ic_tensor

            elif field_type == "dynamic_forcing":
                if var_key in curr_source:
                    assembled_input[source][var_key] = curr_source[var_key]
                else:
                    logger.warning(
                        "assemble_rollout_batch: dynamic_forcing '%s' not in curr_batch; "
                        "carrying forward from ic_preprocessed.",
                        var_key,
                    )
                    assembled_input[source][var_key] = ic_tensor

            else:
                # static and any other non-predicted field: carry forward from ic_preprocessed
                # (already on the model grid after IC-only preblocks)
                assembled_input[source][var_key] = ic_tensor

    return {
        "input": assembled_input,
        "target": curr_batch.get("target"),
    }
