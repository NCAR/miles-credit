"""Start-state perturbations for twin runs and initial-condition ensembles.

:class:`StartStatePerturbation` (``start_state_perturbation``) adds a seeded perturbation, in physical units, to the
start state of a rollout. Configure it under ``preblocks.ic_only``, which runs once at t = 0 of each forecast (the
trainer does not use that section). Kinds:

- ``"white"``: independent Gaussian noise at every grid point, standard deviation ``amplitude``.
- ``"band"``: Gaussian noise restricted to total wavenumbers ``lmin..lmax`` on the sphere (spherical-harmonic
  coefficients outside the band zeroed), scaled to an area-weighted RMS of ``amplitude``. For small-scale-only
  perturbations that test whether error moves upscale.
- ``"file"``: a precomputed perturbation field (e.g. a scaled difference of two same-season CESM states) read from a
  NetCDF file by short variable name, times ``amplitude``.

``amplitude`` is a float for every variable or ``{short name: float}``. The noise depends only on ``seed`` and, with
``per_init_seed`` (default), on the forecast's start time, so every model arm sees the same perturbation for the
same start date, and different start dates get different perturbations.
"""

import hashlib

import torch
import xarray as xr

from credit.preblock._utils import _parse_variable_selection
from credit.preblock.base import BasePreblock

KINDS = ("white", "band", "file")


class StartStatePerturbation(BasePreblock):
    """Seeded start-state perturbation (see the module docstring).

    Args:
        variables: full or partial variable keys, e.g. ``["CESM/prognostic/3d/T"]``.
        amplitude: perturbation size in the variable's physical units; a float or ``{short name: float}``.
        kind: ``"white"``, ``"band"`` or ``"file"``.
        seed: base seed.
        per_init_seed: mix the forecast start time into the seed.
        lmin, lmax: wavenumber band for ``"band"`` (inclusive).
        path: NetCDF file for ``"file"``; each variable's field is ``(lat, lon)`` or ``(level, lat, lon)``.
        data_types: which parts of the batch to perturb (``"input"``: the start state).
    """

    def __init__(
        self,
        variables,
        amplitude,
        kind="white",
        seed=0,
        per_init_seed=True,
        lmin=None,
        lmax=None,
        path=None,
        data_types=("input",),
    ):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if kind == "band" and (lmin is None or lmax is None or not 0 <= lmin <= lmax):
            raise ValueError("kind 'band' needs 0 <= lmin <= lmax")
        if kind == "file" and not path:
            raise ValueError("kind 'file' needs path")
        self.variables = list(variables)
        self.variables_expanded = False
        self.amplitude = amplitude
        self.kind = kind
        self.seed = int(seed)
        self.per_init_seed = bool(per_init_seed)
        self.lmin, self.lmax = lmin, lmax
        self.path = path
        self.data_types = list(data_types)
        self._fields = None
        self._band_transforms = {}

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            self.variables_expanded = True
        batch = self._copy_batch(batch)
        base_seed = self._seed(batch)
        for index, var_key in enumerate(self.variables):
            amplitude = self._amplitude(var_key)
            if amplitude == 0:
                continue
            source = var_key.split("/")[0]
            for data_type in self.data_types:
                fields = batch.get(data_type, {}).get(source, {})
                if var_key not in fields:
                    continue
                x = fields[var_key]
                generator = torch.Generator().manual_seed((base_seed + 7919 * (index + 1)) % 2**63)
                fields[var_key] = x + amplitude * self._unit_perturbation(var_key, x, generator).to(x.device, x.dtype)
        return batch

    def _amplitude(self, var_key):
        if isinstance(self.amplitude, dict):
            return float(self.amplitude.get(var_key.split("/")[-1], 0.0))
        return float(self.amplitude)

    def _seed(self, batch):
        if not self.per_init_seed:
            return self.seed
        stamps = []
        for meta in (batch.get("metadata") or {}).values():
            if isinstance(meta, dict) and "input_datetime" in meta:
                stamps.append(int(torch.as_tensor(meta["input_datetime"]).flatten()[0]))
        if not stamps:
            return self.seed
        digest = hashlib.sha256(f"{self.seed}:{stamps[0]}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % 2**63

    def _unit_perturbation(self, var_key, x, generator):
        if self.kind == "white":
            return torch.randn(x.shape, generator=generator, dtype=torch.float64)
        if self.kind == "band":
            return self._band_noise(x.shape, generator)
        return self._file_field(var_key, x)

    def _band_noise(self, shape, generator):
        """Gaussian noise limited to total wavenumbers lmin..lmax, area-weighted RMS 1 per 2-D field."""
        from torch_harmonics import InverseRealSHT

        from camulator_smallscale.spectral import SphericalSpectra

        nlat, nlon = shape[-2], shape[-1]
        if (nlat, nlon) not in self._band_transforms:
            spectra = SphericalSpectra(nlat, nlon, dtype=torch.float64)
            inverse = InverseRealSHT(nlat, nlon, lmax=spectra.lmax, mmax=spectra.mmax, grid="equiangular").double()
            if self.lmax >= spectra.lmax:
                raise ValueError(f"lmax {self.lmax} must be below the truncation {spectra.lmax}")
            self._band_transforms[(nlat, nlon)] = (spectra, inverse)
        spectra, inverse = self._band_transforms[(nlat, nlon)]
        coeffs = spectra.transform(torch.randn(shape, generator=generator, dtype=torch.float64))
        keep = torch.zeros(spectra.lmax, 1, dtype=torch.float64)
        keep[self.lmin : self.lmax + 1] = 1.0
        noise = inverse(coeffs * keep)
        rms = spectra.area_mean_square(noise).sqrt()[..., None, None]
        return noise / rms.clamp(min=torch.finfo(torch.float64).tiny)

    def _file_field(self, var_key, x):
        if self._fields is None:
            with xr.open_dataset(self.path) as ds:
                self._fields = {name: torch.as_tensor(ds[name].values, dtype=torch.float64) for name in ds.data_vars}
        name = var_key.split("/")[-1]
        if name not in self._fields:
            raise KeyError(f"StartStatePerturbation: {self.path} has no field {name!r}")
        field = self._fields[name]
        if field.ndim == 3 and x.ndim == 5:  # (level, lat, lon) into (batch, level, time, lat, lon)
            field = field[None, :, None]
        return torch.broadcast_to(field, x.shape)
