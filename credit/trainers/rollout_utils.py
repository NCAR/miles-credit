import logging

import pandas as pd
import torch
from tqdm import tqdm
import os
from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.postblock import apply_postblocks
from credit.preblock import apply_preblocks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def parse_length(length_str: str, timestep: str) -> int:
    """Convert a duration string to a number of autoregressive steps.

    Example: "10d" with timestep "6h" → 40 steps.
    """
    total = pd.Timedelta(length_str)
    step = pd.Timedelta(timestep)
    n = int(total / step)
    if n <= 0:
        raise ValueError(f"Inference length '{length_str}' is not positive for timestep '{timestep}'.")
    return n


def batch_init_times(batch_conf: dict) -> list[pd.Timestamp]:
    """Generate the ordered list of init timestamps from inference.batch_forecast."""
    start = pd.Timestamp(batch_conf["first_init_date"])
    end = pd.Timestamp(batch_conf["last_init_date"])
    interval = pd.Timedelta(batch_conf["init_interval"])

    init_times = []
    current = start
    while current <= end:
        init_times.append(current)
        current += interval

    return sorted(set(init_times))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_inference(conf: dict, device: torch.device) -> torch.nn.Module:
    """Load and optionally wrap the model for the requested inference mode.

    Reads ``conf["inference"]["mode"]`` and delegates to the appropriate loader:
    - ``none``: uses the canonical ``load_model(conf, load_weights=True)``
    - ``ddp``:  wraps with DDP then loads checkpoint into ``model.module``
    - ``fsdp``: wraps with FSDP then restores sharded state via ``load_model_state``
    """
    from credit.distributed import distributed_model_wrapper

    mode = conf["inference"]["mode"]
    save_loc = os.path.expandvars(conf["save_loc"])

    # distributed_model_wrapper reads conf["trainer"]["mode"], so patch it.
    conf.setdefault("trainer", {})["mode"] = mode
    conf["trainer"].setdefault("activation_checkpoint", False)

    if mode == "none":
        return load_model(conf, load_weights=True).to(device)

    if mode == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(conf, model, device)
        # Buffers are static once the checkpoint is loaded below (eval mode, no
        # training), so there's nothing to sync. DDP's default broadcast_buffers=True
        # makes every forward() a collective requiring all ranks to call it the same
        # number of times — but rollout ranks can cover an uneven number of init
        # times, so disable it to avoid a mid-loop hang.
        model.broadcast_buffers = False
        # Mirror BaseModel.load_model: prefer model_checkpoint.pt, fall back to checkpoint.pt
        ckpt_path = os.path.join(save_loc, "model_checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(save_loc, "checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        key = "model_state_dict" if "model_state_dict" in ckpt else None
        load_state_dict_error_handler(model.module.load_state_dict(ckpt[key] if key else ckpt, strict=False))
        return model

    if mode == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        return load_model_state(conf, model, device)

    raise ValueError(f"Unsupported inference mode: {mode!r}. Choose none | ddp | fsdp.")


# ---------------------------------------------------------------------------
# Core rollout loop
# ---------------------------------------------------------------------------


def run_forecast(
    conf: dict,
    n_steps: int,
    save_dir: str,
    ic_preblocks,
    step_preblocks,
    step_postblocks,
    rollout_postblocks,
    model: torch.nn.Module,
    batch_iter,
    device: torch.device,
    pool,
    save_output_fn,
    verbose: bool = True,  # False on non-rank-0 workers in DDP to suppress tqdm and save-path output
) -> None:
    """Run one autoregressive forecast, consuming n_steps batches from batch_iter.

    Mirrors trainer_gen2.py: full_data_dict carries the super-dict through
    preblocks → model → postblocks → assemble_rollout_batch at every step.

    The first batch consumed is the IC (step index 0, returns all variable groups).
    Steps 1..n_steps-1 each consume one dynamic-forcing-only batch from the iterator.
    The final step n_steps runs the model on state assembled from the previous step's
    output and does not load an additional forcing batch.

    Args:
        conf: full config dict.
        n_steps: number of autoregressive steps.
        save_dir: output directory passed to save_output_fn.
        ic_preblocks: preblocks built for phase ``"ic_only"``.
        step_preblocks: preblocks built for phase ``"per_step"``.
        step_postblocks: postblocks built for phase ``"per_step"``.
        rollout_postblocks: postblocks built for phase ``"post_rollout"``.
        model: the loaded (and optionally wrapped) model.
        batch_iter: iterator over a DataLoader whose sampler yields IC then
            forcing batches in step order for one init time.
        device: target device.
        pool: multiprocessing pool passed through to save_output_fn.
        save_output_fn: callable(y_processed, init_time, step, fhr_per_step,
            save_dir, pool) invoked after each model step.
    """
    dt = pd.Timedelta(conf["data"]["timestep"])
    fhr_per_step = int(dt.total_seconds() / 3600)
    history_len = int(conf["data"].get("history_len", 1))

    full_data_dict: dict = {}

    # ── Step 0 (IC): load initial condition, run preblocks ──────────────────
    ic_batch = next(batch_iter)

    # Extract init_time from the IC batch metadata (stored as nanoseconds since epoch).
    source_name = next(iter(ic_batch["metadata"]))
    init_time = pd.Timestamp(ic_batch["metadata"][source_name]["input_datetime"][0].item())
    init_str = init_time.strftime("%Y-%m-%dT%HZ")

    full_data_dict["ic_raw"] = ic_batch.get("input", {})
    full_data_dict["ic_preprocessed"] = apply_preblocks(ic_preblocks, ic_batch, device=device)
    full_data_dict.update(apply_preblocks(step_preblocks, full_data_dict["ic_preprocessed"], device=device))

    # Multi-step history window fed to the model. At the IC it comes straight
    # from the dataset (history_len steps stacked on the time dim). At each
    # rollout step below we slide it forward (drop oldest, append the newest
    # assembled step) so the model keeps seeing history_len steps, mirroring the
    # trainer's rollout. With history_len == 1 there is nothing to carry.
    history_x = full_data_dict["x"] if history_len > 1 else None

    logger.info("Forecast init: %s  steps: %d  fhr_max: %dh", init_str, n_steps, n_steps * fhr_per_step)

    # ── Autoregressive loop ──────────────────────────────────────────────────
    with torch.no_grad():
        for step in tqdm(
            range(1, n_steps + 1), desc=f"Rollout {init_str}", disable=not verbose
        ):  # one bar on rank 0 only
            full_data_dict["y_pred"] = model(full_data_dict["x"])

            # postblocks: Reconstruct → inverse_scaler → physics fixers
            full_data_dict = apply_postblocks(step_postblocks, full_data_dict)

            save_output_fn(full_data_dict["y_processed"], init_time, step, fhr_per_step, save_dir, pool)

            if step < n_steps:
                # Load dynamic forcing for the next step from the shared iterator.
                frc_batch = next(batch_iter)

                # route predictions → prognostic/diagnostic, new forcing → dynamic_forcing,
                # IC statics → static
                next_batch = assemble_rollout_batch(full_data_dict, frc_batch, history_len)

                # drop None target so ConcatToTensor doesn't trip over it
                next_batch = {k: v for k, v in next_batch.items() if v is not None}

                full_data_dict.update(apply_preblocks(step_preblocks, next_batch, device=device))

                if history_len > 1:
                    # assemble_rollout_batch produced a single-step x; slide the
                    # previous history_len-step window forward by one step: drop
                    # the oldest time step and append this newest one (dim=2 is
                    # time). Same free-running slide the trainer uses in rollout.
                    history_x = torch.cat(
                        [history_x[:, :, 1:, ...], full_data_dict["x"][:, :, -1:, ...]], dim=2
                    )
                    full_data_dict["x"] = history_x

    # post_rollout postblocks (e.g. global physics fixers applied once)
    apply_postblocks(rollout_postblocks, full_data_dict)

    # Make sure all output files are fully written before starting the next forecast.
    if hasattr(save_output_fn, "flush"):
        save_output_fn.flush()
    logger.info("Done: %s", init_str)


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
