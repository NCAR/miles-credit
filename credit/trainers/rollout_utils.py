"""rollout_utils.py — utilities for autoregressive multi-step rollout."""

import torch


def build_rollout_input(
    prev_full_dict: dict,
    curr_dyn_input: torch.Tensor,
    ic_channel_map: dict,
) -> torch.Tensor:
    """Build the next-step model input tensor using the IC channel map.

    Combines three sources:
    - **Prognostic** channels: from ``prev_full_dict["predicted"]``.
      Supports both a nested variable dict (after ``Reconstruct`` postblock)
      and a raw flat tensor (when no postblocks are configured).
    - **Dynamic forcing** channels: from ``curr_dyn_input`` (current dataloader
      batch), written in the same relative order as the dynamic_forcing entries
      in ``ic_channel_map``.
    - **Static** (and any other) channels: copied as-is from the previous
      input tensor.

    No full clone of the previous input is needed; only static channels are
    copied, so peak extra memory is proportional to the static channel count
    rather than the full tensor size.

    Args:
        prev_full_dict: ``full_data_dict`` from the previous rollout step.
            Must contain:
              ``["preprocessed"]["input"]`` — previous full ``x`` tensor
              ``["preprocessed"]["metadata"]["target"]["_channel_map"]`` — output
              channel map used for the flat-tensor fallback path
              ``["predicted"]`` — either a nested dict
              ``{source: {var_key: tensor(B, n_lev, T, H, W)}}`` (after
              ``Reconstruct``) or a raw flat tensor ``(B, C_pred, H, W)``.
        curr_dyn_input: flat tensor ``(B, C_dyn, H, W)`` of dynamic_forcing
            channels from the current dataloader batch.  Must be in the same
            relative order as the dynamic_forcing entries in ``ic_channel_map``.
        ic_channel_map: per-variable channel map cached from ``t=1``, covering
            ALL input variable groups.  Keys are ``source/field_type/dim/varname``
            strings; values are ``{"slice": slice, "orig_shape": (n_lev, T)}``.

    Returns:
        Tensor ``(B, C_in, H, W)``: assembled input for the next forward pass.
    """
    x_prev = prev_full_dict["preprocessed"]["input"]
    prediction = prev_full_dict["predicted"]

    x_new = torch.empty_like(x_prev)
    dyn_cursor = 0

    for var_key, info in ic_channel_map.items():
        sl = info["slice"]
        n = sl.stop - sl.start
        parts = var_key.split("/")
        field_type = parts[1] if len(parts) > 1 else ""

        if field_type == "dynamic_forcing":
            x_new[:, sl, ...] = curr_dyn_input[:, dyn_cursor : dyn_cursor + n, ...]
            dyn_cursor += n

        elif field_type == "prognostic":
            if isinstance(prediction, dict):
                # After Reconstruct: nested dict {source: {var_key: (B, n_lev, T, H, W)}}
                # Reconstruct always produces 5D var tensors.  If x_prev is 4D we need to
                # flatten (n_lev, T) → n_lev*T; if x_prev is 5D assign directly.
                source = parts[0]
                if source in prediction and var_key in prediction[source]:
                    var_tensor = prediction[source][var_key]  # (B, n_lev, T, H, W)
                    if x_prev.dim() == 5:
                        x_new[:, sl, ...] = var_tensor
                    else:
                        x_new[:, sl, ...] = var_tensor.flatten(1, 2)
                else:
                    x_new[:, sl, ...] = x_prev[:, sl, ...]
            else:
                # Flat tensor fallback (no Reconstruct in postblocks).
                # prediction has the same dimensionality as x_prev, so the slice
                # naturally yields a compatible shape — no flatten needed.
                output_map = prev_full_dict["preprocessed"]["metadata"]["target"]["_channel_map"]
                if var_key in output_map:
                    out_sl = output_map[var_key]["slice"]
                    x_new[:, sl, ...] = prediction[:, out_sl, ...]
                else:
                    x_new[:, sl, ...] = x_prev[:, sl, ...]

        else:
            # Static and any other non-updated channels: preserve from previous x.
            x_new[:, sl, ...] = x_prev[:, sl, ...]

    return x_new
