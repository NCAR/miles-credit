"""CAMulator models for the experiments.

- :class:`MPSSafeHFCamulator` (``mps_safe_hf_camulator``): camulator_gen2's ``HFCamulator`` with the MPS large-kernel
  conv workaround (:mod:`camulator_smallscale.mps`) applied when it is built. The local configs use it; on CPU and
  CUDA it computes exactly what ``HFCamulator`` does, so the Derecho configs keep ``hf_camulator``.
"""

import logging

from camulator_gen2.models import HFCamulator

from camulator_smallscale.mps import patch_mps_large_kernel_convs

logger = logging.getLogger(__name__)


class MPSSafeHFCamulator(HFCamulator):
    """``HFCamulator`` that stays correct on Apple MPS (see :mod:`camulator_smallscale.mps`).

    Args:
        **kwargs: passed to :class:`camulator_gen2.models.HFCamulator`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = patch_mps_large_kernel_convs(self)
        logger.info("MPSSafeHFCamulator: %d large-kernel Conv2d layer(s) chunked when run on MPS", n)
