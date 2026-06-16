import logging

import torch

from credit.preblock.base import BasePreblock
from ._utils import _parse_variable_selection

logger = logging.getLogger(__name__)

# Explicit torch functions rather than Python operators so the lookup in forward()
# is a single dict access with no branching.
_OPS = {
    "eq": torch.eq,   # x == search
    "ne": torch.ne,   # x != search
    "lt": torch.lt,   # x <  search
    "le": torch.le,   # x <= search
    "gt": torch.gt,   # x >  search
    "ge": torch.ge,   # x >= search
}


class FillValues(BasePreblock):
    """Replace values matching a set of rules with constant fill values for selected variables.

    Walks a nested batch dict of the form ``batch[data_type][source][var_key]``
    and applies each rule as a search-and-replace pass. All masks are computed
    on the **original** tensor before any replacement (simultaneous semantics),
    so earlier rules do not affect later rules' matches.

    Each rule is a dict with:

    - ``search``: the string ``"nan"`` (matches NaN) or a float (used with ``op``).
    - ``op``: comparison operator — ``"eq"``, ``"ne"``, ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``
      (default ``"eq"``; ignored when ``search`` is ``"nan"``).
    - ``fill``: the replacement value.

    Numeric ops (``search`` is a float) never match NaN positions — use ``search: "nan"``
    to explicitly target NaN values.

    If two rules match the same position, the last rule in the list wins. A
    warning is logged on the first forward pass if any overlap is detected.

    Config example::

        type: "fill_values"
        args:
            rules:
                - search: nan        # NaN       → -1.0
                  fill: -1.0
                - search: 0.0        # == 0.0    → 1.0e-4
                  op: eq
                  fill: 1.0e-4
                - search: 0.0        # < 0.0     → 0.0  (clamp negatives)
                  op: lt
                  fill: 0.0
            variables:               # optional — defaults to all variables
                - "era5/prognostic/3d/Q"
            data_types:              # optional — defaults to ['input', 'target']
                - "input"
                - "target"
    """

    def __init__(
        self,
        rules: list[dict],
        variables: list[str] = None,
        data_types: list[str] = None,
    ):
        super().__init__()

        # --- state ---
        self.rules = rules
        self.variables = variables or []                # empty list means "all variables" (expanded on first forward)
        self.variables_expanded = False                 # expanded lazily on first forward pass once batch structure is known
        self._overlap_checked = False                   # overlap check runs once on the first forward pass
        self.data_types = data_types or ["input", "target"]  # which batch splits to apply rules to

        # --- validation ---
        # reject unknown data_types early so errors surface at config load, not mid-training
        invalid_dt = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid_dt:
            raise ValueError(f"Invalid data_types {invalid_dt}. Valid options are {self.VALID_DATA_TYPES}.")

        # validate each rule at init so bad configs fail immediately
        for rule in rules:
            # both keys are required; 'op' is optional (defaults to "eq")
            if "search" not in rule or "fill" not in rule:
                raise ValueError(f"Each rule must have 'search' and 'fill' keys, got: {rule}")
            if rule["search"] != "nan":
                # 'search' must be a number when not "nan"
                if not isinstance(rule["search"], (int, float)):
                    raise ValueError(f"Rule 'search' must be 'nan' or a number, got: {rule['search']!r}")
                op = rule.get("op", "eq")
                # op is only validated for numeric searches; it's ignored for "nan"
                if op not in _OPS:
                    raise ValueError(f"Rule 'op' must be one of {sorted(_OPS)}, got: {op!r}")

    def _check_overlaps(self, masks: list[tuple]) -> None:
        """Warn if any two masks overlap (both True at the same position).

        Called once on the first forward pass. When rules overlap, the last
        matching rule wins because replacements are applied sequentially.
        """
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if (masks[i][0] & masks[j][0]).any():
                    logger.warning(
                        "FillValues: rules %d and %d overlap "
                        "(search=%r, op=%r and search=%r, op=%r). "
                        "Rule %d (the later one) wins at overlapping positions.",
                        i, j,
                        self.rules[i]["search"], self.rules[i].get("op", "eq"),
                        self.rules[j]["search"], self.rules[j].get("op", "eq"),
                        j,
                    )

    def forward(self, batch: dict) -> dict:
        # --- lazy variable expansion ---
        # deferred to forward so we can inspect the actual batch keys;
        # an empty self.variables at init means "expand to all variables"
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            self.variables_expanded = True

        # --- apply rules ---
        for var_key in self.variables:
            source = var_key.split("/")[0]  # e.g. "era5" from "era5/prognostic/3d/Q"

            for data_type in self.data_types:
                # skip gracefully if this data_type / source / var is absent in the batch
                # (e.g. a GOES-only rule applied to a batch that also contains ERA5)
                if data_type not in batch:
                    continue
                if source not in batch[data_type]:
                    continue
                if var_key not in batch[data_type][source]:
                    continue

                x = batch[data_type][source][var_key]

                # Compute all masks on the original tensor before any replacement.
                # Numeric ops exclude NaN positions (NaN comparisons are unreliable);
                # use search="nan" to explicitly target NaN values.
                not_nan = ~torch.isnan(x)
                masks = []
                for rule in self.rules:
                    if rule["search"] == "nan":
                        mask = torch.isnan(x)
                    else:
                        op_fn = _OPS[rule.get("op", "eq")]
                        # AND with not_nan so NaN positions are never matched by numeric ops
                        mask = not_nan & op_fn(x, rule["search"])
                    masks.append((mask, rule["fill"]))

                # check for overlapping rules on the first forward pass only;
                # uses the first variable encountered as a representative sample
                if not self._overlap_checked:
                    self._check_overlaps(masks)
                    self._overlap_checked = True

                # apply replacements in rule order; if rules overlap the last rule wins
                for mask, fill in masks:
                    # dtype and device must match x to avoid implicit cast or device mismatch in torch.where
                    x = torch.where(mask, torch.tensor(fill, dtype=x.dtype, device=x.device), x)

                batch[data_type][source][var_key] = x

        return batch
