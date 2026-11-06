"""Red-in-time (AR(1)) noise for stochastic emulators.

Two generators, shared by the SDL model (``model.red_noise``, see
``credit.models.wxformer.crossformer_ensemble``) and the ``targeted_red_noise``
postblock (``credit.postblock.red_noise``):

- :class:`AR1Noise`: standard-normal noise of any shape, AR(1) in time and
  white in space.
- :class:`SphericalAR1Pattern`: a unit-variance, isotropic Gaussian random
  field on an equiangular lat-lon grid (poles included), with correlation
  ``~exp(-d**2 / (2 L**2))`` at great-circle distance ``d``, AR(1) in time.
  The same construction as SPPT/SKEBS patterns: random spherical-harmonic
  coefficients, inverse-transformed with torch-harmonics.

Both start from their stationary distribution: the first draw after a reset is
plain N(0, 1), and each later draw is
``phi * previous + sqrt(1 - phi**2) * innovation`` with ``phi = exp(-dt / tau)``.
Every step therefore has the same N(0, 1) marginal as white noise. That is why
single-step CRPS training cannot tell red from white (or learn tau), and why a
white-trained SDL checkpoint can run red noise without retraining.

The state lives in plain attributes, never in registered buffers: DDP's
``broadcast_buffers`` would copy rank 0's state to every rank and collapse a
ring-crps ensemble, and checkpoints stay unchanged. It is built only from fresh
``randn`` draws, so it never carries an autograd graph from one step to the
next. Each batch row is its own stream, i.e. one ensemble member.

Call :func:`reset_noise_state` at the start of every trajectory: the gen2
trainer does so at ``t == 1`` of every batch and ``run_forecast`` at the start
of every forecast.
"""

import math
from numbers import Real

import torch

from credit.physics_constants import RAD_EARTH

RED_NOISE_TARGETS = ("latent", "pixel")
_RED_NOISE_KEYS = {"tau_hours", "timestep_hours", "apply_to"}
_N_DECODER_POINTS = 3


def ar1_phi(tau_hours, timestep_hours: float) -> float:
    """Lag-one autocorrelation of an AR(1) process with e-folding time ``tau_hours``.

    Returns ``exp(-timestep_hours / tau_hours)``, or 0.0 (white) when ``tau_hours`` is None or 0.
    """
    if not tau_hours:
        return 0.0
    return math.exp(-timestep_hours / tau_hours)


def reset_noise_state(*modules) -> None:
    """Restart every red-noise stream in ``modules`` from its stationary distribution.

    Calls ``reset_noise_state()`` on every submodule that defines it. Accepts
    models (wrapped or not), postblock ``ModuleDict``s and ``None``; anything
    that is not an ``nn.Module`` is skipped. A no-op for white noise, so the
    training and rollout loops call it unconditionally.
    """
    for module in modules:
        if not isinstance(module, torch.nn.Module):
            continue
        for m in module.modules():
            reset = getattr(m, "reset_noise_state", None)
            if callable(reset):
                reset()


class AR1Noise:
    """Standard-normal noise that is AR(1) in time: ``s_t = phi * s_{t-1} + sqrt(1 - phi**2) * e_t``.

    The first draw after a reset (or after the shape or device changes) is the
    innovation itself, so every draw is N(0, 1). With ``phi = 0`` this is white
    noise.

    Args:
        phi: lag-one autocorrelation, in [0, 1).
    """

    def __init__(self, phi: float):
        if not 0.0 <= phi < 1.0:
            raise ValueError(f"AR(1) phi must be in [0, 1), got {phi}.")
        self.phi = float(phi)
        self.innovation_scale = math.sqrt(1.0 - self.phi**2)
        self.state = None

    def reset(self) -> None:
        self.state = None

    def advance(self, innovation: torch.Tensor) -> torch.Tensor:
        """Step the process with a unit-variance ``innovation`` and return the new state."""
        if self.state is None or self.state.shape != innovation.shape or self.state.device != innovation.device:
            self.state = innovation
        else:
            self.state = self.phi * self.state + self.innovation_scale * innovation
        return self.state

    def sample(self, shape, device) -> torch.Tensor:
        """Draw the next step. Uses exactly one ``torch.randn(shape)`` call, as white noise would."""
        return self.advance(torch.randn(shape, device=device))


def _spectrum(n_modes: int, length_scale_km: float, radius_km: float) -> torch.Tensor:
    """Angular power ``exp(-l(l+1) L**2 / (2 R**2))`` for l = 0 .. n_modes-1, without the l = 0 (global-mean) mode.

    This is the heat-kernel spectrum, whose correlation is ``~exp(-d**2 / (2 L**2))`` for ``L << R``.
    """
    l_ = torch.arange(n_modes, dtype=torch.float64)
    spectrum = torch.exp(-l_ * (l_ + 1) * (length_scale_km / radius_km) ** 2 / 2)
    spectrum[0] = 0.0
    return spectrum


def spectral_truncation_fraction(nlat: int, nlon: int, length_scale_km: float, radius_km: float = RAD_EARTH / 1000):
    """Fraction of a length-``L`` pattern's variance beyond the wavenumbers an ``nlat`` x ``nlon`` grid resolves.

    Above about 0.01, ``length_scale_km`` is too short for the grid: the pattern
    loses its smallest scales and its correlation length grows.
    """
    n_modes = min(nlat, nlon // 2)
    l_ = torch.arange(8 * n_modes, dtype=torch.float64)
    weight = _spectrum(len(l_), length_scale_km, radius_km) * (2 * l_ + 1)
    total = weight.sum()
    return float(weight[n_modes:].sum() / total) if total > 0 else 1.0


class SphericalAR1Pattern:
    """Unit-variance isotropic random pattern on an equiangular lat-lon grid, AR(1) in time.

    Spherical-harmonic coefficients ``a_lm`` are drawn with variance
    proportional to ``exp(-l(l+1) L**2 / (2 R**2))`` for every ``m <= l``
    (triangular truncation), and inverse-transformed with torch-harmonics. By
    the addition theorem the grid-point variance is then the same everywhere,
    poles included; the spectrum is normalized so that it is 1. The global-mean
    mode (l = 0) is left out.

    The grid must be equiangular with both poles (CAM's FV grid, ERA5 0.25 deg):
    torch-harmonics' "equiangular" nodes are exactly those latitudes. The field
    is isotropic, so the latitude order (north-south or south-north) does not
    matter.

    Args:
        nlat, nlon: grid size.
        length_scale_km: correlation length L.
        phi: lag-one autocorrelation of the AR(1) process (0 = white).
        radius_km: sphere radius.
    """

    def __init__(self, nlat: int, nlon: int, length_scale_km: float, phi: float, radius_km: float = RAD_EARTH / 1000):
        truncated = spectral_truncation_fraction(nlat, nlon, length_scale_km, radius_km)
        if truncated > 0.01:
            raise ValueError(
                f"length_scale_km={length_scale_km} is too short for a {nlat} x {nlon} grid: "
                f"{truncated:.1%} of the pattern's variance lies beyond the resolved wavenumbers."
            )
        self.nlat, self.nlon = nlat, nlon
        # m < nlon / 2 keeps every zonal mode a full complex pair (the Nyquist mode is not one).
        self.n_modes = min(nlat, nlon // 2)
        l_ = torch.arange(self.n_modes, dtype=torch.float64)
        spectrum = _spectrum(self.n_modes, length_scale_km, radius_km)
        # Orthonormal harmonics: sum_m |Y_lm|^2 = (2l + 1) / (4 pi) at every point.
        variance = (spectrum * (2 * l_ + 1)).sum() / (4 * math.pi)
        amplitude = torch.sqrt(spectrum / variance)
        # A real field counts each m > 0 twice (m and -m): give its real and imaginary
        # parts variance amplitude^2 / 2 each; m = 0 is real (its imaginary part is dropped).
        m_ = torch.arange(self.n_modes)
        per_m = torch.where(m_ == 0, 1.0, math.sqrt(0.5)).double()
        coef_std = amplitude[:, None] * per_m[None, :] * (m_[None, :] <= l_[:, None])
        self.coef_std = coef_std.float()  # (l, m)
        self.noise = AR1Noise(phi)
        self._transforms = {}

    def reset(self) -> None:
        self.noise.reset()

    def _transform(self, device):
        key = str(device)
        if key not in self._transforms:
            from torch_harmonics import InverseRealSHT

            self._transforms[key] = InverseRealSHT(
                self.nlat, self.nlon, lmax=self.n_modes, mmax=self.n_modes, grid="equiangular", norm="ortho"
            ).to(device)
        return self._transforms[key]

    def innovation(self, batch: int, device) -> torch.Tensor:
        """A fresh, temporally independent pattern, ``(batch, nlat, nlon)``."""
        device = torch.device(device)
        # torch-harmonics' complex Legendre/FFT path is not supported on MPS: transform on the CPU.
        work = torch.device("cpu") if device.type == "mps" else device
        coef = torch.randn(batch, self.n_modes, self.n_modes, 2, device=work) * self.coef_std.to(work)[..., None]
        return self._transform(work)(torch.view_as_complex(coef)).to(device)

    def sample(self, batch: int, device) -> torch.Tensor:
        """The next step of the pattern, ``(batch, nlat, nlon)``."""
        return self.noise.advance(self.innovation(batch, device))


def red_noise_errors(red_noise, encoder_noise: bool = False, correlated: bool = False) -> list[str]:
    """Problems with a ``model.red_noise`` block (empty list if it is valid).

    Shared by ``CrossFormerWithNoise`` (which raises on them) and ``credit check``.
    """
    if not isinstance(red_noise, dict):
        return [f"must be a mapping with keys {sorted(_RED_NOISE_KEYS)}, got {type(red_noise).__name__}"]
    errors = []
    unknown = set(red_noise) - _RED_NOISE_KEYS
    if unknown:
        errors.append(f"unknown keys {sorted(unknown)} (accepted: {sorted(_RED_NOISE_KEYS)})")

    def bad_tau(value):
        return value is not None and (isinstance(value, bool) or not isinstance(value, Real) or value < 0)

    tau = red_noise.get("tau_hours")
    if tau is None:
        errors.append("tau_hours is required (remove the red_noise block for white noise)")
    elif isinstance(tau, (list, tuple)):
        if len(tau) != _N_DECODER_POINTS or any(bad_tau(t) for t in tau):
            errors.append(
                f"tau_hours as a list needs {_N_DECODER_POINTS} entries, one per decoder injection point "
                f"(coarsest first), each a number >= 0 or null; got {tau}"
            )
        if encoder_noise:
            errors.append("tau_hours as a list covers the decoder only; use one number with encoder_noise: true")
        if correlated:
            errors.append("tau_hours as a list needs correlated: false (correlated shares one latent)")
    elif bad_tau(tau):
        errors.append(f"tau_hours must be a number >= 0, got {tau!r}")

    dt = red_noise.get("timestep_hours")
    if isinstance(dt, bool) or not isinstance(dt, Real) or dt <= 0:
        errors.append(f"timestep_hours must be a positive number (the model step, e.g. 6), got {dt!r}")

    apply_to = red_noise.get("apply_to", list(RED_NOISE_TARGETS))
    if not isinstance(apply_to, (list, tuple)) or not apply_to or set(apply_to) - set(RED_NOISE_TARGETS):
        errors.append(f"apply_to must be a non-empty subset of {list(RED_NOISE_TARGETS)}, got {apply_to!r}")
    return errors
