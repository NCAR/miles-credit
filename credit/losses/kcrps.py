import torch


class KCRPSLoss(torch.nn.Module):
    """Adapted from Nvidia Modulus
    pred : Tensor
        Tensor containing the ensemble predictions. The ensemble dimension
        is assumed to be the leading dimension
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    biased :
        When False, uses the unbiased estimators described in (Zamo and Naveau, 2018)::

            E|X-y|/m - 1/(2m(m-1)) sum_(i,j=1)|x_i - x_j|

        Unlike ``crps`` this is fair for finite ensembles. Non-fair ``crps`` favors less
        dispersive ensembles since it is biased high by E|X- X'|/ m where m is the
        ensemble size.

    Estimate the CRPS from a finite ensemble

    Computes the local Continuous Ranked Probability Score (CRPS) by using
    the kernel version of CRPS. The cost is O(m log m).

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Approximates:

    .. math::
        CRPS(X, y) = E[X - y] - 0.5 E[X-X']

    with

    .. math::
        sum_i=1^m |X_i - y| / m - 1/(2m^2) sum_i,j=1^m |x_i - x_j|

    """

    def __init__(self, reduction, biased: bool = False):
        super().__init__()
        self.biased = biased
        self.batched_forward = torch.vmap(self.single_sample_forward)

    def forward(self, target, pred):
        # integer division but will error out next op if there is a remainder
        ensemble_size = pred.shape[0] // target.shape[0] + pred.shape[0] % target.shape[0]
        pred = pred.view(target.shape[0], ensemble_size, *target.shape[1:])  # b, ensemble, c, t, lat, lon
        # apply single_sample_forward to each dim
        target = target.unsqueeze(1)
        return self.batched_forward(target, pred).squeeze(1)

    def single_sample_forward(self, target, pred):
        """
        Forward pass for KCRPS loss for a single sample.

        Args:
            target (torch.Tensor): Target tensor.
            pred (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: CRPS loss values at each lat/lon
        """
        pred = torch.movedim(pred, 0, -1)
        return self._kernel_crps_implementation(pred, target, self.biased)

    def _kernel_crps_implementation(self, pred: torch.Tensor, obs: torch.Tensor, biased: bool) -> torch.Tensor:
        """An O(m log m) implementation of the kernel CRPS formulas"""
        skill = torch.abs(pred - obs[..., None]).mean(-1)
        pred, _ = torch.sort(pred)

        # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
        # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
        #                           = 2 sum_(i <= j) |x_i -x_j|
        #                           = 2 sum_(i <= j) (x_j - x_i)
        #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
        #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
        #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
        m = pred.size(-1)
        i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
        denom = m * m if biased else m * (m - 1)
        factor = (2 * i - m - 1) / denom
        spread = torch.sum(factor * pred, dim=-1)
        return skill - spread
