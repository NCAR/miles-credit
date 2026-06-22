"""
Refactored SKEBS module.

SpectralNoisePattern  — standalone red-noise spherical-harmonic generator
                        (no neural-network dependency; fully reusable).
SKEBS                 — full backscatter scheme; owns a SpectralNoisePattern
                        and a backscatter neural network.
"""


import torch
from torch import nn
import torch_harmonics as harmonics

import numpy as np

from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from credit.physics_constants import (
    RAD_EARTH,
    PI
)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reusable spectral noise pattern
# ---------------------------------------------------------------------------

class SpectralNoisePattern(nn.Module):
    """
    Red-noise spherical-harmonic pattern generator.

    Implements the temporally-correlated spectral stochastic pattern from
    Berner et al. (2009).  The class is self-contained: it owns the SHT
    objects, the Laplacian / inverse-Laplacian buffers, and all trainable
    pattern parameters (alpha, variance, p, dE, r, and the spectral filter).
    It does *not* reference a backscatter network or any data-loading logic,
    so it can be reused wherever a red-noise perturbation field is needed.

    Args:
        nlat (int):  number of latitude grid points.
        nlon (int):  number of longitude grid points.
        lmax (int):  [optional] maximum total wave-number.
        mmax (int):  [optional] maximum zonal wave-number.
        grid (str):  grid type understood by torch_harmonics (e.g. "equiangular").
        levels (int): number of vertical levels (used only for the top-of-model
                      filter buffer).
        zero_out_levels_top_of_model (int): how many top levels to zero out.
        tropics_only_dissipation (bool): restrict pattern to tropical latitudes.
        multistep (bool): whether alpha should be a trainable multi-step param.
        alpha_init (float): initial value of alpha.
        max_pattern_wavenum (int):   cut-off wave-number for pattern filter.
        pattern_filter_anneal_start (int): wave-number where filter starts to
                                           roll off.
        freeze_pattern_weights (bool): freeze all parameters of this module.

    References:
        Berner et al. (2009). J. Atmos. Sci., 66(3), 603-626.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int = None,
        mmax: int = None,
        mode: str = "streamfunction", #other mode is scalar
        multistep: bool = False,
        alpha_init: float = 0.125,
        max_pattern_wavenum: int = 60,
        pattern_filter_anneal_start: int = 40,
        freeze_pattern_weights: bool = False,
    ):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.multistep = multistep
        self.eps = 1e-12

        grid = "equiangular" # generalize this later
        self._initialize_sht(lmax, mmax, grid)
        self._initialize_parameters(multistep, alpha_init)
        self._initialize_filters(max_pattern_wavenum, pattern_filter_anneal_start)

        # Pattern state (need to lazy-initialise before first forward call!)
        self.spec_coef: torch.Tensor | None = None
        self.multivariateNormal: MultivariateNormal | None = None
        self.is_initialized = False
        if mode == "streamfunction":
            self.inverse_transform = self.get_streamfunction_perturb
        elif mode == "scalar":
            self.inverse_transform = self.get_scalar_perturb

        if freeze_pattern_weights:
            logger.warning("freezing all SpectralNoisePattern weights")
            for param in self.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_sht(self, lmax: int, mmax: int, grid):
        """Set up spherical-harmonic transform objects and related buffers."""
        self.sht  = harmonics.RealSHT(self.nlat, self.nlon, lmax, mmax, grid, csphase=False)
        # self.vsht  = harmonics.RealVectorSHT(self.nlat, self.nlon, lmax, mmax, grid, csphase=False)
        self.isht = harmonics.InverseRealSHT(self.nlat, self.nlon, lmax, mmax, grid, csphase=False)
        self.ivsht = harmonics.InverseRealVectorSHT(
            self.nlat, self.nlon, lmax, mmax, grid, csphase=False
        )

        # Overwrite with actual sizes reported by the transform objects
        self.lmax = self.ivsht.lmax
        self.mmax = self.ivsht.mmax

        # Equiangular grid quadrature
        cost, _ = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)
        self.lats = -torch.arcsin(cost)
        self.lons = torch.linspace(0, 2 * PI, self.nlon + 1, dtype=torch.float64)[: self.nlon]

        l_arr = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l_arr = l_arr.expand(self.lmax, self.mmax)

        self.register_buffer("lap",    -l_arr * (l_arr + 1) / RAD_EARTH**2, persistent=False)
        self.register_buffer("invlap", -(RAD_EARTH**2) / l_arr / (l_arr + 1), persistent=False)
        self.invlap[0] = 0.0  # avoid division by zero at l=0

        self.register_buffer(
            "lrange",
            torch.arange(1, self.lmax + 1).unsqueeze(1),
            persistent=False,
        )

        logging.info(f"SpectralNoisePattern — lmax: {self.lmax}, mmax: {self.mmax}")

    def _initialize_parameters(self, multistep: bool, alpha_init: float):
        """Register trainable / fixed pattern parameters."""
        if multistep:
            logger.info("multi-step SpectralNoisePattern")
            self.alpha = Parameter(torch.tensor(alpha_init, requires_grad=True))
        else:
            logger.info("single-step SpectralNoisePattern, no need to tune AR(1) parameter alpha")
            self.alpha = Parameter(torch.tensor(1.0, requires_grad=False))
            self.alpha.requires_grad = False # doubly sure this happens

        self.variance = Parameter(torch.tensor(0.083,  requires_grad=True))
        self.p        = Parameter(torch.tensor(-1.27,  requires_grad=True))
        self.dE       = Parameter(torch.tensor(1e-4,   requires_grad=True))

    def _initialize_filters(
        self,
        max_pattern_wavenum: int,
        pattern_filter_anneal_start: int,
    ):
        """Build spectral pattern filter and spatial masks."""
        # Spectral roll-off filter for the noise pattern
        spectral_filter = torch.cat([
                                torch.ones(pattern_filter_anneal_start),
                                torch.linspace(1., 0.1, max_pattern_wavenum - pattern_filter_anneal_start),
                                torch.zeros(self.lmax - max_pattern_wavenum)
                                ]).view(1,1,1,self.lmax, 1)
        self.spectral_pattern_filter = Parameter(spectral_filter, requires_grad=False)
    
    # ------------------------------------------------------------------
    # Pattern lifecycle
    # ------------------------------------------------------------------

    def initialize_pattern(self, reference_tensor: torch.Tensor):
        """
        Allocate spectral coefficients and spin up the AR(1) pattern.
        ! run this before the first forward pass !

        Args:
            reference_tensor: any tensor on the target device whose first
                              dimension gives the batch size.
        """
        b = reference_tensor.shape[0]
        device = reference_tensor.device

        self.spec_coef = torch.zeros(
            (b, 1, 1, self.lmax, self.mmax),
            dtype=torch.cfloat,
            device=device,
        )
        self.multivariateNormal = MultivariateNormal(
            torch.zeros(2, device=device),
            torch.eye(2, device=device),
        )

        spin_up_iters = 10
        logger.debug(f"SpectralNoisePattern: spinning up with {spin_up_iters} iterations")
        for _ in range(spin_up_iters):
            self.spec_coef = self.cycle_pattern(self.spec_coef)

    def _clip_parameters(self):
        """ clip the trainable parameters so that they are always physical"""
        self.alpha.data = self.alpha.data.clamp(self.eps, 1.)
        self.variance.data = self.variance.clamp(self.eps, 10.)
        self.p.data = self.p.data.clamp(-10, -self.eps)
        self.dE.data = self.dE.data.clamp(self.eps, 1.)
        self.spectral_pattern_filter.data = self.spectral_pattern_filter.data.clamp(0., 1.)


    def cycle_pattern(self, spec_coef: torch.Tensor) -> torch.Tensor:
        """
        Advance the AR(1) spectral pattern by one time step.

        Args:
            spec_coef: complex tensor of shape (b, 1, 1, lmax, mmax).

        Returns:
            Updated complex tensor of the same shape.
        """
        self._clip_parameters() # clip parameters before cycling
        Gamma = torch.sum(
            self.lrange * (self.lrange + 1.0)
            * (2 * self.lrange + 1.0)
            * self.lrange ** (2.0 * self.p)
        )
        b = torch.sqrt(
            (4.0 * PI * RAD_EARTH**2.0) / (self.variance * Gamma) * self.alpha * self.dE
        )
        g_n = b * self.lrange ** self.p  # (lmax, 1)

        cmplx_noise = torch.view_as_complex(self.multivariateNormal.sample(spec_coef.shape))
        noise       = self.variance * cmplx_noise
        new_coef    = (1.0 - self.alpha) * spec_coef + g_n * torch.sqrt(self.alpha) * noise
        return new_coef * self.spectral_pattern_filter

    # ------------------------------------------------------------------
    # Spectral / grid utilities
    # ------------------------------------------------------------------

    def spec2grid(self, uspec: torch.Tensor) -> torch.Tensor:
        """Transform spectral coefficients to grid space."""
        return self.isht(uspec)

    def getuv(self, vrtdivspec: torch.Tensor) -> torch.Tensor:
        """Compute wind vector from spectral coefficients of vorticity and divergence."""
        return self.ivsht(self.invlap * vrtdivspec / RAD_EARTH).unsqueeze(1).unsqueeze(1)

    def getgrad(self, chispec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the non-divergent (u, v) perturbation from a streamfunction
        given in spectral space.

        Args:
            chispec: complex tensor of shape (b, lmax, mmax).

        Returns:
            u_chi, v_chi: real tensors of shape (b, 1, 1, nlat, nlon).
        """
        vrtspec = self.lap * chispec
        uv = self.getuv(
            torch.stack( # non-divergent component only
                (vrtspec, torch.zeros_like(vrtspec).to(vrtspec.device)),
                # ( torch.zeros_like(vrtspec).to(vrtspec.device), vrtspec),
                dim=-3,
            )
        )
        return uv[..., 0, :, :], uv[..., 1, :, :]

    def get_scalar_perturb(self, coef):
        return self.spec2grid(coef) / 1e6 * 2
    
    def get_streamfunction_perturb(self, coef):
        u_chi, v_chi = self.getgrad(coef) 
        return u_chi / torch.sqrt(self.dE), v_chi / torch.sqrt(self.dE)
    
    # ------------------------------------------------------------------
    # Forward pass: advance pattern and return (u_chi, v_chi) or a scalar
    # ------------------------------------------------------------------

    def forward(
        self,
        detach: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance the pattern by one step and return grid-space perturbations.
        ! use initialize_pattern before first forward call !

        Args:
            detach: whether to detach ``spec_coef`` before cycling (set False
                    to retain the compute graph).

        Returns:
            u_chi, v_chi — non-divergent wind perturbations on the grid,
            shape (b, 1, 1, nlat, nlon).
        """
        if detach:
            self.spec_coef = self.spec_coef.detach()

        self.spec_coef = self.cycle_pattern(self.spec_coef)
        spec_coef_squeezed = self.spec_coef.squeeze()
        return self.inverse_transform(spec_coef_squeezed)


    # ------------------------------------------------------------------
    # Tiny helpers
    # ------------------------------------------------------------------

    def _should_write_debug(self) -> bool:
        return (
            (self.write_rollout_debug_files and not self.is_training)
            or (self.write_train_debug_files and self.iteration % self.write_every == 0)
        )
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nlat = 200
    nlon = 300
    mode = "scalar"
    nsteps = 3
    x = torch.rand((2,nlat, nlon))

    noise_gen = SpectralNoisePattern(
        nlat,
        nlon,
        max_pattern_wavenum=100,
        pattern_filter_anneal_start=50,
        mode=mode,
    )
    noise_gen.initialize_pattern(x)
    for step in range(nsteps):
        perturb = noise_gen()

        # plt.imshow(torch.sqrt(u_chi[0] ** 2 + v_chi[0] ** 2).squeeze().detach().numpy())
        perturb = perturb[0].squeeze().detach().numpy()
        plt.figure(figsize=(12, 6))
        plt.imshow(perturb)
        plt.savefig(f"skebs_scalar_{step}.png")
        print(np.max(perturb))

        ### histogram
        plt.figure(figsize=(12,6))
        plt.hist(perturb)
        plt.savefig(f"skebs_hist_{step}.png")


    # for step in range(nsteps):
    #     u_chi, v_chi = noise_gen()

    #     # plt.imshow(torch.sqrt(u_chi[0] ** 2 + v_chi[0] ** 2).squeeze().detach().numpy())
    #     u_np = u_chi[0].squeeze().detach().numpy()
    #     plt.figure(figsize=(12, 6))

    #     plt.imshow(u_np)
    #     plt.savefig(f"skebs_u_{step}.png")
    #     v_np = v_chi[0].squeeze().detach().numpy()
    #     plt.figure(figsize=(12, 6))

    #     plt.imshow(v_np)

    #     plt.savefig(f"skebs_v_{step}.png")
        
    #     print(np.max(u_np), np.max(v_np))


    #     ### histogram
    #     plt.figure(figsize=(12,6))
    #     plt.hist(u_np)
    #     plt.savefig(f"skebs_hist_{step}.png")

    #     ### barbs
    #     # grid coordinates
    #     lon = np.arange(nlon)
    #     lat = np.arange(nlat)

    #     LON, LAT = np.meshgrid(lon, lat)
        
    #     skip = 8

    #     plt.figure(figsize=(12,6))
    #     plt.quiver(
    #         lon,
    #         lat,
    #         u_np,
    #         v_np,
    #         pivot="middle"
    #     )
    #     plt.title("Wind vectors")
    #     plt.savefig(f"skebs_quiver_{step}.png")
