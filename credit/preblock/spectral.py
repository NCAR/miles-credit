"""Spherical-harmonic synthesis preblock.

Transforms variables stored as ECMWF spherical-harmonic coefficient vectors
(e.g. the ARCO ERA5 cloud-optimized ``co/`` stores: temperature, vorticity,
divergence, vertical velocity at T639) into grid-point fields, on the fly,
inside the gen2 preblock chain. Winds are *derived*: ERA5 never stores model
level u/v in spectral form — they are computed from vorticity and divergence
via the streamfunction/velocity-potential relations.

Coefficient conventions (verified against the ARCO analysis-ready 0.25°
product to ~0.1 K RMSE for model-level temperature):

- Packing is m-major: ``for m in 0..T: for l in m..T: (Re, Im)`` —
  ``(T+1)(T+2)/2`` complex numbers, ``(T+1)(T+2)`` float32 values
  (T639 -> 410,240).
- IFS normalization has ``Y_0^0 = 1`` (the (0,0) coefficient is the global
  mean), no Condon-Shortley phase. Relative to torch-harmonics'
  orthonormal basis this is a single factor of ``sqrt(4*pi)``.

Synthesis targets:

- ``target_grid: "reduced_gaussian"`` (default) — evaluate on the native
  reduced Gaussian grid (ring latitudes = Gaussian nodes, per-ring inverse
  real FFTs), producing the same flat point vector layout as the
  grid-point ``co/`` stores, so *all* variables can then go through one
  ``regrid`` preblock (see ``credit.reduced_gaussian`` for building the
  bilinear weight file). Requires ``grid_file``.
- ``target_grid: "equiangular"`` — synthesize directly on an equiangular
  lat/lon grid (``nlat`` x ``nlon``, poles included, N->S) with no
  intermediate interpolation. For derived winds the two pole rows are
  copied from the adjacent rows (u, v are ill-defined at the poles).

Config example (ARCO ERA5 co/ -> N320 -> 0.25°)::

    preblocks:
      per_step:
        to_device: {type: to_device, args: {device: cuda}}
        spectral:
          type: "spectral_to_grid"
          args:
            grid_file: "$SCRATCH/N320_grid.nc"
            scalar_vars:
              "ERA5/prognostic/3d/temperature": "ERA5/prognostic/3d/temperature"
            vector_vars:
              - vorticity: "ERA5/prognostic/3d/vorticity"
                divergence: "ERA5/prognostic/3d/divergence"
                u: "ERA5/prognostic/3d/u_component_of_wind"
                v: "ERA5/prognostic/3d/v_component_of_wind"
        regrid:
          type: "regrid"
          args: {weight_file: "$SCRATCH/n320_to_0p25_bilinear.nc", variables: [...]}

Variables absent from a data type are skipped silently, so at rollout steps
t > 1 (where prognostics are re-fed from the model output and are already
grid-point fields) the block is a no-op.
"""

from __future__ import annotations

import logging
import math
from os.path import expandvars

import torch

from credit.preblock.base import BasePreblock

logger = logging.getLogger(__name__)

_VALID_TARGET_GRIDS = ("reduced_gaussian", "equiangular")


def spectral_packing_index(truncation: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(l, m) degree/order of each packed complex coefficient, ECMWF m-major order."""
    ls, ms = [], []
    for m in range(truncation + 1):
        for l in range(m, truncation + 1):  # noqa: E741
            ls.append(l)
            ms.append(m)
    return torch.tensor(ls, dtype=torch.long), torch.tensor(ms, dtype=torch.long)


def pack_spectral(coef: torch.Tensor, truncation: int) -> torch.Tensor:
    """Inverse of the unpacking done in the preblock — mainly for tests.

    Args:
        coef: complex tensor ``(..., lmax, mmax)`` with ``lmax = mmax = truncation + 1``.

    Returns:
        float tensor ``(..., (truncation+1)*(truncation+2))`` in ECMWF packing.
    """
    l_idx, m_idx = spectral_packing_index(truncation)
    packed = coef[..., l_idx, m_idx]
    return torch.view_as_real(packed).flatten(-2)


class SpectralToGrid(BasePreblock):
    """Synthesize ECMWF spherical-harmonic variables to grid-point fields.

    Args:
        scalar_vars: mapping ``{spectral_var_key: output_var_key}`` for scalar
            fields (temperature, vertical velocity, log surface pressure, ...).
            A plain list means "synthesize in place" (output key = input key).
        vector_vars: list of dicts, each with keys ``vorticity``, ``divergence``,
            ``u``, ``v`` naming the two spectral input var keys and the two
            grid-point wind output var keys.
        exp_vars: output var keys to exponentiate after synthesis
            (e.g. log surface pressure -> surface pressure).
        truncation: spectral truncation T (ERA5 model levels: 639).
        target_grid: ``"reduced_gaussian"`` (default) or ``"equiangular"``.
        grid_file: ring-description NetCDF written by
            ``credit.reduced_gaussian.ReducedGaussianGrid.save`` (or by the
            ``arco_era5_co`` dataset). Required for ``reduced_gaussian``.
        nlat: destination latitudes for ``equiangular`` (default 721).
        nlon: destination longitudes for ``equiangular`` (default 1440).
        radius: sphere radius in meters used in the wind derivation
            (IFS value 6,371,229 m).
        data_types: batch splits to process; defaults to ``["input", "target"]``.
        delete_inputs: remove consumed spectral inputs (vorticity/divergence
            keys not claimed by ``scalar_vars``) from the batch. Default True.
    """

    def __init__(
        self,
        scalar_vars: dict[str, str] | list[str] | None = None,
        vector_vars: list[dict[str, str]] | None = None,
        exp_vars: list[str] | None = None,
        truncation: int = 639,
        target_grid: str = "reduced_gaussian",
        grid_file: str | None = None,
        nlat: int = 721,
        nlon: int = 1440,
        radius: float = 6371229.0,
        data_types: list[str] | None = None,
        delete_inputs: bool = True,
    ):
        super().__init__()
        if isinstance(scalar_vars, (list, tuple)):
            scalar_vars = {k: k for k in scalar_vars}
        self.scalar_vars: dict[str, str] = scalar_vars or {}
        self.vector_vars: list[dict[str, str]] = list(vector_vars or [])
        for spec in self.vector_vars:
            missing = {"vorticity", "divergence", "u", "v"} - set(spec)
            if missing:
                raise ValueError(f"vector_vars entry {spec} is missing keys {sorted(missing)}.")
        self.exp_vars = list(exp_vars or [])
        if not self.scalar_vars and not self.vector_vars:
            raise ValueError("SpectralToGrid: at least one of scalar_vars / vector_vars must be given.")
        if target_grid not in _VALID_TARGET_GRIDS:
            raise ValueError(f"target_grid must be one of {_VALID_TARGET_GRIDS}, got '{target_grid}'.")
        self.target_grid = target_grid
        self.truncation = int(truncation)
        self.radius = float(radius)
        self.data_types = data_types or ["input", "target"]
        invalid = set(self.data_types) - set(self.VALID_DATA_TYPES)
        if invalid:
            raise ValueError(f"Invalid data_types {invalid}. Valid options are {self.VALID_DATA_TYPES}.")
        self.delete_inputs = delete_inputs
        # Per-device cache of moved constant tensors (the Legendre table is
        # ~1 GB at T639 — it must not be re-copied host->device every call).
        self._dev_cache: dict = {}

        T = self.truncation
        self.lmax = T + 1  # number of degrees for scalar coefficients
        self.mmax = T + 1
        self.n_coef = (T + 1) * (T + 2) // 2
        self.n_values = 2 * self.n_coef

        l_idx, m_idx = spectral_packing_index(T)
        self.register_buffer("l_idx", l_idx, persistent=False)
        self.register_buffer("m_idx", m_idx, persistent=False)

        # eps[l, m] = sqrt((l^2 - m^2) / (4 l^2 - 1)), 0 where m > l or l == 0;
        # rows go up to l = lmax (needed by the H-recurrence coefficient shift).
        lg = torch.arange(self.lmax + 1, dtype=torch.float64).unsqueeze(1)
        mg = torch.arange(self.mmax, dtype=torch.float64).unsqueeze(0)
        eps = torch.sqrt(torch.clamp((lg**2 - mg**2) / (4.0 * lg**2 - 1.0), min=0.0))
        eps[0] = 0.0
        eps = eps * (mg <= lg)
        self.register_buffer("eps", eps.to(torch.float32), persistent=False)

        # -l(l+1)/a^2 inverse (psi = -a^2 zeta / (l(l+1))), 0 at l=0.
        lv = torch.arange(self.lmax, dtype=torch.float64)
        invlap = torch.zeros(self.lmax, dtype=torch.float64)
        invlap[1:] = -(self.radius**2) / (lv[1:] * (lv[1:] + 1.0))
        self.register_buffer("inv_laplace", invlap.to(torch.float32), persistent=False)

        if self.target_grid == "reduced_gaussian":
            if grid_file is None:
                raise ValueError("SpectralToGrid: grid_file is required when target_grid='reduced_gaussian'.")
            from credit.reduced_gaussian import ReducedGaussianGrid

            grid = ReducedGaussianGrid.from_file(expandvars(grid_file))
            self.n_points = grid.n_points
            self.n_rings = grid.n_rings
            ring_lats = torch.from_numpy(grid.ring_lats)
            theta = torch.deg2rad(90.0 - ring_lats)  # colatitude, N->S
            self.register_buffer("cos_phi", torch.sin(theta).to(torch.float32), persistent=False)
            self._init_ring_groups(grid)
            # Associated Legendre table at the ring latitudes, inverse-normalized
            # (includes the (2 - delta_m0) real-synthesis factor), no CS phase,
            # with the extra degree row needed for the wind H-terms:
            # shape (mmax, lmax + 1, n_rings).
            from torch_harmonics.legendre import legpoly

            pct = legpoly(self.mmax, self.lmax + 1, torch.cos(theta), norm="ortho", inverse=True, csphase=False)
            self.register_buffer("pct", pct.to(torch.float32), persistent=False)
            self.isht = None
        else:
            from torch_harmonics import InverseRealSHT

            self.nlat = int(nlat)
            self.nlon = int(nlon)
            self.n_points = self.nlat * self.nlon
            # One transform sized for the wind terms (lmax + 1); scalar
            # coefficients are zero-padded by one degree row.
            self.isht = InverseRealSHT(
                self.nlat,
                self.nlon,
                lmax=self.lmax + 1,
                mmax=self.mmax,
                grid="equiangular",
                norm="ortho",
                csphase=False,
            )
            lats = torch.linspace(90.0, -90.0, self.nlat, dtype=torch.float64)
            self.register_buffer("cos_phi", torch.cos(torch.deg2rad(lats)).to(torch.float32), persistent=False)

    # ------------------------------------------------------------------
    # Grid bookkeeping
    # ------------------------------------------------------------------

    def _init_ring_groups(self, grid) -> None:
        """Group rings sharing the same nlon so each group runs one batched irfft."""
        import numpy as np

        self._ring_groups: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        nlon_arr = grid.ring_nlon
        for nlon_g in np.unique(nlon_arr):
            rings = np.flatnonzero(nlon_arr == nlon_g)
            # Flat destination indices of every point in this ring group.
            pts = (grid.ring_offset[rings][:, None] + np.arange(nlon_g)[None, :]).ravel()
            self._ring_groups.append(
                (
                    int(nlon_g),
                    torch.from_numpy(rings.astype(np.int64)),
                    torch.from_numpy(pts.astype(np.int64)),
                )
            )

    def _on(self, name: str, device: torch.device) -> torch.Tensor:
        """Return buffer *name* on *device*, moving and caching it on first use."""
        t = getattr(self, name)
        if t.device == device:
            return t
        key = (device, name)
        if key not in self._dev_cache:  # setdefault would re-copy host->device on every call
            self._dev_cache[key] = t.to(device)
        return self._dev_cache[key]

    def _groups_on(self, device: torch.device) -> list:
        key = (device, "_ring_groups")
        if key not in self._dev_cache:
            self._dev_cache[key] = [(n, rings.to(device), pts.to(device)) for n, rings, pts in self._ring_groups]
        return self._dev_cache[key]

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def _unpack(self, x: torch.Tensor, extra_degree_row: bool = False) -> torch.Tensor:
        """Packed float ``(N, n_values)`` -> dense real-view ``(N, L, M, 2)``.

        Includes the IFS -> orthonormal ``sqrt(4 pi)`` rescaling.
        """
        n_deg = self.lmax + 1 if extra_degree_row else self.lmax
        coef = x.new_zeros((x.shape[0], n_deg, self.mmax, 2))
        coef[:, self._on("l_idx", x.device), self._on("m_idx", x.device)] = x.view(x.shape[0], self.n_coef, 2)
        return coef * math.sqrt(4.0 * math.pi)

    def _legendre_fourier(self, coef_r: torch.Tensor) -> torch.Tensor:
        """Dense real-view coefficients ``(N, L, M, 2)`` -> complex Fourier
        coefficients per ring, ``(N, n_rings, M)``."""
        pct = self._on("pct", coef_r.device)
        n_deg = coef_r.shape[1]
        f = torch.einsum("nlmr,mlj->njmr", coef_r, pct[:, :n_deg])
        return torch.view_as_complex(f.contiguous())

    def _rings_to_points(self, fm: torch.Tensor) -> torch.Tensor:
        """Per-ring Fourier coefficients ``(N, n_rings, M)`` -> flat reduced-grid
        point values ``(N, n_points)`` via grouped inverse real FFTs."""
        out = torch.empty((fm.shape[0], self.n_points), dtype=torch.float32, device=fm.device)
        for nlon_g, rings, pts in self._groups_on(fm.device):
            m_keep = nlon_g // 2 + 1
            fg = fm.index_select(1, rings)
            if m_keep <= self.mmax:
                fg = fg[..., :m_keep].contiguous()
            else:
                pad = fg.new_zeros((*fg.shape[:-1], m_keep - self.mmax))
                fg = torch.cat([fg, pad], dim=-1)
            # Real-synthesis hygiene (mirrors torch-harmonics): the mean and
            # Nyquist Fourier modes must be purely real.
            fg[..., 0].imag = 0.0
            if nlon_g % 2 == 0 and nlon_g // 2 < fg.shape[-1]:
                fg[..., nlon_g // 2].imag = 0.0
            vals = torch.fft.irfft(fg, n=nlon_g, dim=-1, norm="forward")
            out[:, pts] = vals.reshape(vals.shape[0], -1).to(torch.float32)
        return out

    def _shift_for_h(self, coef_r: torch.Tensor) -> torch.Tensor:
        """Coefficient shift that turns the derivative sum ``sum_l c_lm H_lm``
        into a plain Legendre sum ``sum_l c'_lm P_lm``, using
        ``H_lm = -l eps_{l+1,m} P_{l+1,m} + (l+1) eps_{l,m} P_{l-1,m}``.

        Input ``(N, L, M, 2)`` -> output ``(N, L+1, M, 2)``.
        """
        eps = self._on("eps", coef_r.device)
        L = self.lmax
        out = coef_r.new_zeros((coef_r.shape[0], L + 1, self.mmax, 2))
        lv = torch.arange(1, L + 1, dtype=coef_r.dtype, device=coef_r.device).view(1, -1, 1, 1)
        # c'_{l} += -(l - 1) eps_{l} c_{l-1}
        out[:, 1:] += -(lv - 1.0) * eps[1 : L + 1].unsqueeze(-1) * coef_r[:, :L]
        # c'_{l} += (l + 2) eps_{l+1} c_{l+1}
        lv2 = torch.arange(0, L - 1, dtype=coef_r.dtype, device=coef_r.device).view(1, -1, 1, 1)
        out[:, : L - 1] += (lv2 + 2.0) * eps[1:L].unsqueeze(-1) * coef_r[:, 1:L]
        return out

    def _synthesize_scalar(self, x: torch.Tensor) -> torch.Tensor:
        lead_shape = x.shape[:-1]
        coef_r = self._unpack(x.reshape(-1, self.n_values))
        if self.target_grid == "reduced_gaussian":
            fm = self._legendre_fourier(coef_r)
            out = self._rings_to_points(fm)
            return out.reshape(*lead_shape, self.n_points)
        # equiangular: pad the extra degree row expected by the shared isht
        pad = coef_r.new_zeros((coef_r.shape[0], 1, self.mmax, 2))
        coef = torch.view_as_complex(torch.cat([coef_r, pad], dim=1).contiguous())
        out = self._isht_on(coef, x.device)
        return out.reshape(*lead_shape, self.nlat, self.nlon)

    def _synthesize_vector(self, vo: torch.Tensor, div: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lead_shape = vo.shape[:-1]
        if div.shape != vo.shape:
            raise ValueError(f"vorticity/divergence shapes differ: {vo.shape} vs {div.shape}.")
        psi = self._unpack(vo.reshape(-1, self.n_values))
        chi = self._unpack(div.reshape(-1, self.n_values))
        invlap = self._on("inv_laplace", psi.device).view(1, -1, 1, 1)
        psi = psi * invlap  # streamfunction coefficients
        chi = chi * invlap  # velocity potential coefficients

        # d/dlambda: multiply by i m -> (re, im) -> (-m im, m re)
        m = torch.arange(self.mmax, dtype=psi.dtype, device=psi.device).view(1, 1, -1, 1)
        im_chi = torch.cat([-m[..., 0:1] * chi[..., 1:2], m[..., 0:1] * chi[..., 0:1]], dim=-1)
        im_psi = torch.cat([-m[..., 0:1] * psi[..., 1:2], m[..., 0:1] * psi[..., 0:1]], dim=-1)

        psi_h = self._shift_for_h(psi)
        chi_h = self._shift_for_h(chi)
        pad = psi.new_zeros((psi.shape[0], 1, self.mmax, 2))
        # u cos(phi) = (1/a) [ sum i m chi P  -  sum psi H ]
        # v cos(phi) = (1/a) [ sum i m psi P  +  sum chi H ]
        u_coef = torch.cat([im_chi, pad], dim=1) - psi_h
        v_coef = torch.cat([im_psi, pad], dim=1) + chi_h

        if self.target_grid == "reduced_gaussian":
            cos_phi = self._on("cos_phi", psi.device)
            outs = []
            for coef_r in (u_coef, v_coef):
                fm = self._legendre_fourier(coef_r)
                fm = fm / (self.radius * cos_phi).view(1, -1, 1)
                outs.append(self._rings_to_points(fm).reshape(*lead_shape, self.n_points))
            return outs[0], outs[1]

        outs = []
        cos_phi = torch.clamp(self._on("cos_phi", psi.device), min=1e-8).view(1, -1, 1)
        for coef_r in (u_coef, v_coef):
            coef = torch.view_as_complex(coef_r.contiguous())
            field = self._isht_on(coef, vo.device) / (self.radius * cos_phi)
            # u, v are ill-defined at the poles: copy the adjacent rows.
            field[:, 0, :] = field[:, 1, :]
            field[:, -1, :] = field[:, -2, :]
            outs.append(field.reshape(*lead_shape, self.nlat, self.nlon))
        return outs[0], outs[1]

    def _isht_on(self, coef: torch.Tensor, device) -> torch.Tensor:
        if next(iter(self.isht.buffers())).device != device:
            self.isht = self.isht.to(device)
        return self.isht(coef)

    # ------------------------------------------------------------------
    # Preblock interface
    # ------------------------------------------------------------------

    def _check_spectral_shape(self, key: str, x: torch.Tensor) -> None:
        if x.shape[-1] != self.n_values:
            raise ValueError(
                f"SpectralToGrid: '{key}' has last dim {x.shape[-1]}, expected "
                f"{self.n_values} packed T{self.truncation} spectral values. "
                "Is this variable really stored as spherical-harmonic coefficients?"
            )

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)  # shallow copy — avoids mutating the caller's dict
        scalar_outputs = set(self.scalar_vars.values())
        for data_type in self.data_types:
            if data_type not in batch:
                continue  # data type absent in this batch (e.g. no "target" during inference)

            # Vector (wind) derivation first, so a vorticity/divergence key that
            # is *also* requested as a scalar channel is still spectral here.
            for spec in self.vector_vars:
                source = spec["vorticity"].split("/")[0]
                src_dict = batch[data_type].get(source)
                if src_dict is None or spec["vorticity"] not in src_dict or spec["divergence"] not in src_dict:
                    continue
                vo, div = src_dict[spec["vorticity"]], src_dict[spec["divergence"]]
                self._check_spectral_shape(spec["vorticity"], vo)
                self._check_spectral_shape(spec["divergence"], div)
                u, v = self._synthesize_vector(vo, div)
                # Seat the derived winds in the dict slots their spectral inputs
                # vacated. The dataset emits vorticity/divergence where u/v were
                # declared, and ChannelSchema is built from that declared order, so
                # appending the winds instead would reorder the channel layout and
                # trip the concat preblock's schema validation.
                substitutions = {}
                for consumed, out_key, field in (
                    (spec["vorticity"], spec["u"], u),
                    (spec["divergence"], spec["v"], v),
                ):
                    keep = consumed in self.scalar_vars or consumed in scalar_outputs
                    if self.delete_inputs and not keep:
                        substitutions[consumed] = (out_key, field)
                    else:
                        src_dict[out_key] = field  # input is kept, so the wind is appended

                if substitutions:
                    rebuilt = {}
                    for key, field in src_dict.items():
                        if key in substitutions:
                            out_key, wind = substitutions[key]
                            rebuilt[out_key] = wind
                        else:
                            rebuilt[key] = field
                    src_dict.clear()
                    src_dict.update(rebuilt)

            for raw_key, out_key in self.scalar_vars.items():
                source = raw_key.split("/")[0]
                src_dict = batch[data_type].get(source)
                if src_dict is None or raw_key not in src_dict:
                    continue
                self._check_spectral_shape(raw_key, src_dict[raw_key])
                src_dict[out_key] = self._synthesize_scalar(src_dict[raw_key])
                if out_key != raw_key and self.delete_inputs:
                    del src_dict[raw_key]

            for key in self.exp_vars:
                source = key.split("/")[0]
                src_dict = batch[data_type].get(source)
                if src_dict is not None and key in src_dict:
                    src_dict[key] = torch.exp(src_dict[key])

        return batch
