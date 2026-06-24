"""rollout_utils.py — utilities for autoregressive multi-step rollout."""

import logging

logger = logging.getLogger(__name__)


def assemble_rollout_batch(full_data_dict: dict, curr_batch: dict, history_len: int = 1) -> dict:
    """Assemble a batch dict for the rollout preblock pass at autoregressive step t > 0.

    Constructs a dataset-schema batch by routing each variable from the
    appropriate source:

    - **prognostic / diagnostic** channels: from ``full_data_dict["y_processed"]`` —
      the postblock-processed prediction from the previous step.
    - **dynamic_forcing** channels: from ``curr_batch["input"]`` — the
      current step's time-varying forcing loaded by the dataset.
    - **static** (and any other non-predicted) channels: from
      ``full_data_dict["ic_preprocessed"]["input"]`` — the t=0 raw batch after
      IC-only preblocks, so statics are already on the model grid.

    The assembled dict is passed to ``apply_preblocks(step_preblocks, ...)``
    which handles per-step operations (log_transform, concat).
    ``curr_batch["target"]`` is forwarded so preblocks normalize the training
    target in the same pass.

    When ``history_len > 1`` the carried-forward (static / non-predicted) tensors
    taken from ``ic_preprocessed`` still carry the t=0 history window along the
    time dimension (dim=2). The newly-routed prognostic and dynamic_forcing
    tensors are single-step, so this function slices the carried-forward tensors
    to their newest step (``[..., -1:, ...]``) to keep every variable single-step.
    The trainer then slides the full ``history_len``-step input window itself
    (drop oldest, append this newest step). With ``history_len == 1`` no slicing
    happens and behaviour is identical to the original.

    Args:
        full_data_dict: the rollout state dict.  Must contain:
            ``"y_processed"`` — nested ``{source: {var_key: tensor}}`` from the
            previous step's postblock chain (output of ``Reconstruct`` + fixers).
            ``"ic_preprocessed"`` — t=0 raw batch after IC-only preblocks,
            providing the authoritative variable key list and static tensors.
        curr_batch: current step's raw batch from the dataset.  Provides
            dynamic forcing fields and the training target.
        history_len: length of the model's input time window. When > 1, static /
            non-predicted tensors are sliced to their newest time step so the
            assembled batch is single-step along the time dimension.

    Returns:
        dict with keys ``"input"`` (nested source→var dict) and ``"target"``
        (from ``curr_batch``), ready for ``apply_preblocks(step_preblocks, ...)``.

    Raises:
        TypeError: if ``full_data_dict["y_processed"]`` is not a dict, which
            usually means ``Reconstruct`` was not included in the postblock chain.
    """
    corrected_pred = full_data_dict["y_processed"]
    ic_preprocessed = full_data_dict["ic_preprocessed"]

    if not isinstance(corrected_pred, dict):
        raise TypeError(
            "assemble_rollout_batch: full_data_dict['y_processed'] must be a nested dict "
            "{source: {var_key: tensor}}. "
            "For multi-step rollout, 'Reconstruct' must be the first postblock. "
            f"Got {type(corrected_pred).__name__}."
        )

    def _newest_step(tensor):
        # Slice the time dimension (dim=2) to its newest step when running a
        # multi-step history, so carried-forward statics line up single-step
        # with the freshly-routed prognostic / dynamic_forcing tensors.
        if history_len > 1 and tensor.dim() >= 3 and tensor.shape[2] > 1:
            return tensor[:, :, -1:, ...]
        return tensor

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
                        "assemble_rollout_batch: '%s' not in y_processed; carrying forward from ic_preprocessed.",
                        var_key,
                    )
                    assembled_input[source][var_key] = _newest_step(ic_tensor)

            elif field_type == "dynamic_forcing":
                if var_key in curr_source:
                    assembled_input[source][var_key] = curr_source[var_key]
                else:
                    logger.warning(
                        "assemble_rollout_batch: dynamic_forcing '%s' not in curr_batch; "
                        "carrying forward from ic_preprocessed.",
                        var_key,
                    )
                    assembled_input[source][var_key] = _newest_step(ic_tensor)

            else:
                # static and any other non-predicted field: carry forward from ic_preprocessed
                assembled_input[source][var_key] = _newest_step(ic_tensor)

    return {
        "input": assembled_input,
        "target": curr_batch.get("target"),
    }
