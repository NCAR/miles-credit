"""Spherical-harmonic spectra on CAMulator's grid: power, cross-spectrum and coherence by total wavenumber.

CAMulator's grid (CAM's 0.9x1.25 finite-volume grid, 192 x 288) is equally spaced in latitude and includes both
poles, which is torch-harmonics' ``equiangular`` grid (Clenshaw-Curtis quadrature). The transforms are truncated
triangularly at the zonal Nyquist, ``lmax = mmax = nlon // 2 + 1`` (145 on this grid, so wavenumbers up to 144,
about 280 km).

With ``nlat`` nodes the quadrature is exact for polynomial degree ``nlat - 1``, so coefficient ``l`` is exact only
for input content of degree ``<= nlat - 1 - l``; higher content leaks in slightly. For red (atmospheric) spectra
the leak is negligible: band powers of a synthetic ``l**-3`` field are recovered to about 1e-4 up to ``l = 144``.
An untruncated transform (``lmax = 192``) adds its top two degrees, which collect most of the leak.

Conventions:

- ``power(x)[..., l] = sum_m w_m |c_lm|^2`` with ``w_0 = 1`` and ``w_m = 2`` for ``m > 0`` (a real field's ``-m``
  modes), so ``power(x).sum(-1)`` is the integral of ``x**2`` over the sphere (Parseval, torch-harmonics'
  orthonormal norm).
- ``area_mean_square(x)`` is the grid mean of ``A * x**2``, with ``A`` the quadrature weights normalized to mean 1.
  For a band-limited field it equals ``power(x).sum(-1) / (4 pi)`` (``PARSEVAL``).
- Latitude order does not matter: a north-south flip multiplies ``c_lm`` by ``(-1)**(l+m)``, which leaves power,
  cross-spectra and coherence unchanged. CREDIT stores latitude south to north; torch-harmonics assumes north to
  south; fields are used as stored.
"""

import math

import torch
from torch import nn

PARSEVAL = 1.0 / (4.0 * math.pi)


def total_wavenumber_to_km(wavenumber, radius_km=6371.0):
    """Wavelength (km) of total wavenumber ``l``: ``2 pi R / l``."""
    return 2.0 * math.pi * radius_km / wavenumber


class SphericalSpectra(nn.Module):
    """Spherical-harmonic power, cross-spectrum and coherence by total wavenumber.

    Args:
        nlat, nlon: grid size (latitude equally spaced, poles included).
        lmax, mmax: truncation; defaults to triangular at the zonal Nyquist, ``nlon // 2 + 1``.
        dtype: precision of the precomputed weights. float32 (default) works everywhere, including Apple MPS, which
            has no float64; float64 is exact to round-off on CPU. Casting later (``.double()``) cannot restore
            precision lost to a float32 build.
    """

    def __init__(self, nlat=192, nlon=288, lmax=None, mmax=None, dtype=torch.float32):
        super().__init__()
        # Imported here: torch-harmonics is a CREDIT dependency today, but PR #547 makes it optional.
        from torch_harmonics import RealSHT
        from torch_harmonics.quadrature import clenshaw_curtiss_weights

        self.nlat, self.nlon = int(nlat), int(nlon)
        self.mmax = int(mmax or self.nlon // 2 + 1)
        self.lmax = int(lmax or self.mmax)
        self.sht = RealSHT(self.nlat, self.nlon, lmax=self.lmax, mmax=self.mmax, grid="equiangular", norm="ortho")
        m_weights = torch.full((self.mmax,), 2.0)
        m_weights[0] = 1.0
        self.register_buffer("m_weights", m_weights, persistent=False)
        _, quad = clenshaw_curtiss_weights(self.nlat, -1.0, 1.0)
        quad = torch.as_tensor(quad, dtype=torch.float64)
        self.register_buffer("area_weights", quad / quad.mean(), persistent=False)  # (nlat,), mean 1
        # torch-harmonics precomputes its Legendre weights in float64, which Apple MPS cannot hold.
        self.to(dtype)

    def transform(self, x):
        """Coefficients ``c_lm`` of ``x`` (``..., nlat, nlon``): complex, ``(..., lmax, mmax)``."""
        return self.sht(x)

    def power_from(self, coeffs):
        """Power by total wavenumber, ``(..., lmax)``, from coefficients."""
        return (coeffs.real**2 + coeffs.imag**2) @ self.m_weights.to(coeffs.real.dtype)

    def cross_from(self, coeffs_a, coeffs_b):
        """Real cross-spectrum ``sum_m w_m Re(a_lm conj(b_lm))`` by total wavenumber, ``(..., lmax)``."""
        real = coeffs_a.real * coeffs_b.real + coeffs_a.imag * coeffs_b.imag
        return real @ self.m_weights.to(real.dtype)

    def power(self, x):
        """Power by total wavenumber of ``x`` (``..., nlat, nlon``), ``(..., lmax)``."""
        return self.power_from(self.transform(x))

    def coherence(self, x, y, rel_eps=1e-12):
        """Coherence by total wavenumber, ``cross / sqrt(Px Py + eps)``, in ``[-1, 1]``.

        ``eps = rel_eps * mean_l(Px) * mean_l(Py)``: a floor relative to each field's own power, so the small
        high-wavenumber power of a red spectrum is not swamped, and gradients stay finite where a field has none.
        """
        cx, cy = self.transform(x), self.transform(y)
        px, py = self.power_from(cx), self.power_from(cy)
        eps = rel_eps * px.mean(-1, keepdim=True) * py.mean(-1, keepdim=True) + torch.finfo(px.dtype).tiny
        return self.cross_from(cx, cy) / torch.sqrt(px * py + eps)

    def area_mean_square(self, x):
        """Grid mean of ``A * x**2`` over the last two dims (``A``: quadrature weights with mean 1)."""
        return (self.area_weights.to(x.dtype)[:, None] * x**2).mean(dim=(-2, -1))


def band_sums(spectrum, bands):
    """Sum ``spectrum`` (``..., lmax``) over inclusive wavenumber bands ``[(lo, hi), ...]``: ``(..., n_bands)``."""
    return torch.stack([spectrum[..., lo : hi + 1].sum(-1) for lo, hi in bands], dim=-1)
