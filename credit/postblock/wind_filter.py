import logging
import torch
import torch.nn.functional as F
from torch import nn
from credit.postblock.conservation import _pred, _set_pred

logger = logging.getLogger(__name__)


def _compute_blend_mask(
    u: torch.Tensor,
    v: torch.Tensor,
    speed_threshold: float,
    dilation_zonal: int,
    dilation_meridional: int,
    falloff_sigma: float,
    smooth_sigma: float,
    smooth_sigma_zonal: float | None = None,
    smooth_sigma_meridional: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the smooth 0-1 blend mask and the smoothing kernel from one level of U, V.

    Args:
        u, v: wind components at the mask level, shape (B, H, W), float32.

    Returns:
        smooth_blend_mask: (B, 1, H, W) in [0, 1] — how strongly to blend
            toward the smoothed field at each point.
        gaussian_2d: (1, 1, k_lat, k_lon) data-smoothing kernel. Separable
            Gaussian, anisotropic when the per-axis sigmas are set: sigma
            along longitude (zonal) kills the zonally-oscillating stripe
            artifact, while a small sigma along latitude (meridional) spares
            the jet's sharp latitude profile. Falls back to isotropic
            ``smooth_sigma`` when the per-axis sigmas are None.
    """
    device, dtype = u.device, u.dtype
    u = u.unsqueeze(1)  # (B, 1, H, W)
    v = v.unsqueeze(1)

    wind_speed = torch.sqrt(u**2 + v**2)
    high_speed_mask = (wind_speed > speed_threshold).to(dtype=dtype)

    # Anisotropic dilation: wider zonally (longitude) than meridionally
    # (latitude), matching jet-stream geometry.
    dilation_kernel = torch.ones(1, 1, dilation_meridional, dilation_zonal, device=device, dtype=dtype)
    expanded_mask = F.conv2d(high_speed_mask, dilation_kernel, padding=(dilation_meridional // 2, dilation_zonal // 2))
    expanded_mask = torch.clamp(expanded_mask, 0, 1)

    # Anisotropic Gaussian falloff (also wider zonally) turns the hard
    # dilated mask into a smooth 0-1 blend weight.
    falloff_kernel_size_lat = int(2 * falloff_sigma * 2 + 1) | 1  # round up to odd
    falloff_kernel_size_lon = int(2 * falloff_sigma * 4 + 1) | 1

    x_lat = torch.arange(falloff_kernel_size_lat, dtype=dtype, device=device) - falloff_kernel_size_lat // 2
    gaussian_1d_lat = torch.exp(-0.5 * (x_lat / falloff_sigma) ** 2)
    gaussian_1d_lat = gaussian_1d_lat / gaussian_1d_lat.sum()

    x_lon = torch.arange(falloff_kernel_size_lon, dtype=dtype, device=device) - falloff_kernel_size_lon // 2
    gaussian_1d_lon = torch.exp(-0.5 * (x_lon / (falloff_sigma * 2)) ** 2)
    gaussian_1d_lon = gaussian_1d_lon / gaussian_1d_lon.sum()

    gaussian_2d_falloff = (gaussian_1d_lat.unsqueeze(1) * gaussian_1d_lon.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    smooth_blend_mask = F.conv2d(
        expanded_mask,
        gaussian_2d_falloff,
        padding=(falloff_kernel_size_lat // 2, falloff_kernel_size_lon // 2),
    )

    # Data-smoothing kernel (separable Gaussian), per-axis sigmas with
    # isotropic fallback — mirrors climate/WindPP.py's simple_wind_artifact_filter.
    sig_lat = smooth_sigma if smooth_sigma_meridional is None else smooth_sigma_meridional
    sig_lon = smooth_sigma if smooth_sigma_zonal is None else smooth_sigma_zonal

    def _gauss1d(sig):
        ks = int(2 * sig * 3 + 1) | 1  # round up to odd
        xx = torch.arange(ks, dtype=dtype, device=device) - ks // 2
        g = torch.exp(-0.5 * (xx / sig) ** 2)
        return g / g.sum()

    g_lat = _gauss1d(sig_lat)  # along latitude  (kernel height)
    g_lon = _gauss1d(sig_lon)  # along longitude (kernel width)
    gaussian_2d = (g_lat.unsqueeze(1) * g_lon.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # (1, 1, k_lat, k_lon)

    return smooth_blend_mask, gaussian_2d


def _blend_smoothed(
    field: torch.Tensor,
    gaussian_2d: torch.Tensor,
    smooth_blend_mask: torch.Tensor,
    preserve_amplitude: bool = False,
) -> torch.Tensor:
    """Blend *field* with its Gaussian-smoothed version, weighted by smooth_blend_mask.

    When ``preserve_amplitude`` is set, the smoothed field is first rescaled so
    its mask-weighted RMS amplitude matches the original: the low-pass removes
    grid-scale (near-2dx) energy but also shaves the jet's genuine peak; since
    the smoothed field carries no grid-scale energy, scaling it back up restores
    the jet strength without re-injecting the artifact. The scale factor is
    computed per batch element (a global sum would couple unrelated samples in
    a batch; identical to the original single-sample behavior at batch size 1)
    and clamped at 4x to guard against pathological over-amplification.

    Args:
        field: (B, 1, H, W), float32.
        gaussian_2d: (1, 1, k_lat, k_lon) smoothing kernel; padding is derived
            from its (possibly rectangular) shape.

    Returns:
        (B, 1, H, W) blended field, same dtype as input.
    """
    k_lat, k_lon = gaussian_2d.shape[-2], gaussian_2d.shape[-1]
    field_smooth = F.conv2d(field, gaussian_2d, padding=(k_lat // 2, k_lon // 2))

    if preserve_amplitude:
        w = smooth_blend_mask
        num = (w * field**2).sum(dim=(1, 2, 3), keepdim=True)
        den = (w * field_smooth**2).sum(dim=(1, 2, 3), keepdim=True)
        alpha = torch.sqrt(num / (den + 1e-12))
        alpha = torch.clamp(alpha, max=4.0)
        field_smooth = alpha * field_smooth

    return smooth_blend_mask * field_smooth + (1 - smooth_blend_mask) * field


class WindArtifactFilter(nn.Module):
    """Smooth spurious grid-scale wind artifacts near the jet stream.

    Detects anomalously high wind speed at ``mask_level`` (from ``u_var``,
    ``v_var``), builds an anisotropic smooth blend mask around those points
    (wider zonally, matching jet geometry), then blends every field in
    ``target_vars`` toward a Gaussian-smoothed version of itself at each
    level in ``target_levels``, weighted by that mask. Points far from a
    detected high-wind-speed region are left unchanged (blend weight ~0);
    points at/near one are pulled toward the smoothed field (blend weight ~1).

    Args:
        u_var: variable key for the zonal wind (e.g. ``CESM/prognostic/3d/U``).
        v_var: variable key for the meridional wind.
        target_vars: variable keys to apply the filter to (must be 3D fields
            sharing u_var/v_var's level axis, e.g. ``[u_var, v_var, T_var, q_var]``).
        mask_level: level index used to detect the high-wind-speed region.
        target_levels: level indices the filter is applied to.
        speed_threshold: wind speed above which a point is flagged. UNIT-SENSITIVE:
            this postblock does no scaling itself — it operates on whatever units
            u_var/v_var happen to be in at its position in the postblocks.per_step
            chain. Placed *before* an inverse-scale/bridgescaler step, values are
            normalized; placed *after*, values are physical (m/s). The default here
            (3.0193274566643846) is climate/WindPP.py's original tuned value,
            calibrated against *normalized* model output in that pipeline — reusing
            it against physical wind speed silently flags nearly the whole domain
            instead of a localized jet region (verified: ~98% of grid points at
            physical scale, vs. the intended sparse/localized detection). If you
            move this block to run after an inverse-scale step, recalibrate
            speed_threshold from real physical wind-speed statistics first.
        smooth_sigma: std dev (grid points) of the smoothing kernel applied to
            flagged fields (isotropic fallback when the per-axis sigmas are None).
        smooth_sigma_zonal / smooth_sigma_meridional: optional per-axis std devs
            for anisotropic data smoothing. The residual artifact is a meridional
            stripe oscillating in longitude (high zonal wavenumber, ~zero
            meridional wavenumber) while the jet is zonally uniform and
            meridionally sharp — so smooth strongly zonally to kill the stripe
            and barely meridionally to preserve the jet's latitude profile.
            None (default) falls back to isotropic ``smooth_sigma``.
        dilation_zonal / dilation_meridional: size (grid points) of the
            rectangular dilation kernel expanding the raw detected region.
        falloff_sigma: std dev (grid points) of the Gaussian falloff that
            turns the dilated hard mask into a smooth blend weight.
        preserve_amplitude: if True, rescale the smoothed field inside the
            flagged region so its mask-weighted RMS matches the original —
            removes the grid-scale wiggle without shaving a genuine jet's peak
            strength (see ``_blend_smoothed``).
    """

    def __init__(
        self,
        u_var: str,
        v_var: str,
        target_vars: list,
        mask_level: int = 14,
        target_levels: list = tuple(range(9, 21)),
        speed_threshold: float = 3.0193274566643846,
        smooth_sigma: float = 1.0,
        smooth_sigma_zonal: float = None,
        smooth_sigma_meridional: float = None,
        dilation_zonal: int = 13,
        dilation_meridional: int = 5,
        falloff_sigma: float = 4.0,
        preserve_amplitude: bool = False,
    ):
        super().__init__()
        self.u_var = u_var
        self.v_var = v_var
        self.target_vars = list(target_vars)
        self.mask_level = mask_level
        self.target_levels = set(target_levels)
        self.speed_threshold = speed_threshold
        self.smooth_sigma = smooth_sigma
        self.smooth_sigma_zonal = smooth_sigma_zonal
        self.smooth_sigma_meridional = smooth_sigma_meridional
        self.dilation_zonal = dilation_zonal
        self.dilation_meridional = dilation_meridional
        self.falloff_sigma = falloff_sigma
        self.preserve_amplitude = preserve_amplitude

    def forward(self, batch_dict: dict) -> dict:
        u = _pred(batch_dict, self.u_var)  # (B, L, 1, H, W)
        v = _pred(batch_dict, self.v_var)
        orig_dtype = u.dtype

        u_mask = u[:, self.mask_level, 0, ...].float()  # (B, H, W)
        v_mask = v[:, self.mask_level, 0, ...].float()

        smooth_blend_mask, gaussian_2d = _compute_blend_mask(
            u_mask,
            v_mask,
            self.speed_threshold,
            self.dilation_zonal,
            self.dilation_meridional,
            self.falloff_sigma,
            self.smooth_sigma,
            self.smooth_sigma_zonal,
            self.smooth_sigma_meridional,
        )

        for var_key in self.target_vars:
            tensor = _pred(batch_dict, var_key)  # (B, L, 1, H, W)
            n_levels = tensor.shape[1]
            out_of_range = [lev for lev in self.target_levels if lev >= n_levels]
            if out_of_range:
                logger.warning(
                    "WindArtifactFilter: target level(s) %s exceed available levels (%d) for '%s'; skipping them.",
                    out_of_range,
                    n_levels,
                    var_key,
                )

            level_slices = []
            for lev in range(n_levels):
                field = tensor[:, lev, 0, ...]  # (B, H, W)
                if lev in self.target_levels:
                    filtered = _blend_smoothed(
                        field.float().unsqueeze(1), gaussian_2d, smooth_blend_mask, self.preserve_amplitude
                    )
                    field = filtered.squeeze(1).to(dtype=orig_dtype)
                level_slices.append(field)

            # (B, L, H, W) -> (B, L, 1, H, W), rebuilding the singleton time dim.
            filtered_tensor = torch.stack(level_slices, dim=1).unsqueeze(2)
            _set_pred(batch_dict, var_key, filtered_tensor)

        return batch_dict
