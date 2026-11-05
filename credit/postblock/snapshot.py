from credit.postblock.base import BasePostblock


class Snapshot(BasePostblock):
    """Copy the current state of a nested prediction dict to a second key.

    Placed in the ``per_step`` chain, it records the prediction as it stands at
    that point, so the blocks after it can keep modifying ``key`` while
    ``out_key`` holds the earlier version. The typical use is to score the loss
    on the prediction *before* the global budget fixers::

        tracer_fixer:   {type: tracer_fixer, args: {...}}
        snapshot_raw:   {type: snapshot, args: {key: y_processed, out_key: y_processed_raw}}
        mass_fixer:     {type: global_mass_fixer, args: {...}}
        energy_fixer:   {type: global_energy_fixer_net, args: {...}}

    with ``loss.args.pred_key: y_processed_raw``. A global fixer that rescales
    or shifts a field hides that field's global mean (or scale) from a loss
    computed on its output, so nothing in training pulls the raw prediction's
    global mean toward the truth; scoring the snapshot restores that.

    Only the dict structure is copied; the tensors are shared, so the snapshot
    costs no memory and stays on the autograd graph. This is safe because
    postblocks replace tensors in the dict rather than modifying them in place.

    Args:
        key: nested ``{source: {var_key: tensor}}`` dict to copy (default ``y_processed``).
        out_key: key the copy is written to (default ``y_processed_raw``).
    """

    def __init__(self, key: str = "y_processed", out_key: str = "y_processed_raw"):
        super().__init__()
        if key == out_key:
            raise ValueError(f"Snapshot: key and out_key must differ (both are {key!r}).")
        self.key = key
        self.out_key = out_key

    def forward(self, batch_dict: dict) -> dict:
        if self.key not in batch_dict:
            raise KeyError(
                f"Snapshot: '{self.key}' is not in the batch; place the snapshot after the reconstruct postblock."
            )
        batch_dict[self.out_key] = {source: dict(variables) for source, variables in batch_dict[self.key].items()}
        return batch_dict
