import numpy as np
import torch
import torch_harmonics as harmonics
import copy
import gc


def polfiltT(D, inddo):
    if len(D.shape) == 2:
        for ii in torch.concatenate(
            [torch.arange(-inddo, 0), torch.arange(1, inddo + 1)]
        ):
            # print(ii)
            ts_Udo = copy.deepcopy(D[ii, :])
            Z = torch.fft.fft(ts_Udo)
            Yfft = Z / ts_Udo.size()[0]
            freq = torch.fft.rfftfreq(len(ts_Udo))
            perd = 1 / freq[1:]
            val_1d, ind_1d = find_nearest(perd, value=100)
            Ck2 = 2.0 * torch.abs(Yfft[0 : int(ts_Udo.size()[0] / 2) + 1]) ** 2
            A = Ck2 / torch.sum(Ck2)
            A[ind_1d:] = 0.0
            Zlow = torch.clone(Z)
            Zlow[ind_1d:-ind_1d] = 0.0
            X_filtered11 = torch.real(torch.fft.ifft(Zlow))
            D[ii, :] = X_filtered11
        return D

    if len(D.shape) == 3:
        for jj in range(D.shape[0]):
            for ii in torch.concatenate(
                [torch.arange(-inddo, 0), torch.arange(1, inddo + 1)]
            ):
                # print(ii)
                ts_Udo = copy.deepcopy(D[jj, ii, :])
                Z = torch.fft.fft(ts_Udo)
                Yfft = Z / ts_Udo.size()[0]
                freq = torch.fft.rfftfreq(len(ts_Udo))
                perd = 1 / freq[1:]
                val_1d, ind_1d = find_nearest(perd, value=100)
                Ck2 = 2.0 * torch.abs(Yfft[0 : int(ts_Udo.size()[0] / 2) + 1]) ** 2
                A = Ck2 / torch.sum(Ck2)
                A[ind_1d:] = 0.0
                Zlow = torch.clone(Z)
                Zlow[ind_1d:-ind_1d] = 0.0
                X_filtered11 = torch.real(torch.fft.ifft(Zlow))
                D[jj, ii, :] = X_filtered11
        return D


def create_sigmoid_ramp_function(array_length, ramp_length):
    """
    Creates an array of specified length with a sigmoid ramp up and down.

    Args:
        array_length: The length of the output array.
        ramp_length: The length of the ramp up and down.

    Returns:
        An array of the specified length with the described ramp up and down using a sigmoid function.
    """

    # Calculate the positions for ramp start and end
    ramp_up_end = ramp_length
    ramp_down_start = array_length - ramp_length

    # Initialize the array with zeros
    array = torch.ones(array_length)

    # Calculate the ramp up using a sigmoid function
    x_up = torch.linspace(-6, 6, ramp_up_end)
    sigmoid_up = 1 / (1 + torch.exp(-x_up))
    array[:ramp_up_end] = sigmoid_up

    # Calculate the ramp down using a reversed sigmoid function
    x_down = torch.linspace(-6, 6, ramp_length)
    sigmoid_down = 1 / (1 + torch.exp(-x_down))
    array[ramp_down_start:] = torch.flip(sigmoid_down, dims=(0,))

    # Return the modified array
    return array


def find_nearest(array, value):
    """
    find nearest index in array
    """
    array = torch.asarray(array)
    idx = (torch.abs(array - value)).argmin()
    return array[idx], idx


class Diffusion_and_Pole_Filter:
    """
    A class designed to encapsulate operations related to diffusion and pole filtering
    with spherical harmonics transformations
    """

    def __init__(
        self,
        nlat,
        nlon,
        device="cpu",
        lmax=None,
        mmax=None,
        grid="legendre-gauss",
        radius=6.37122e6,
        omega=7.292e-5,
        gravity=9.80616,
        havg=10.0e3,
        hamp=120.0,
    ):
        """
        Initialize the Diffusion_and_Pole_Filter object with provided parameters.

        Parameters:
            nlat: The number of latitude points in the grid.
            nlon: The number of longitude points in the grid.
            device: The computing device, default is 'cpu'.
            lmax: maximum spherical harmonic degree
            mmax: The maximum spherical harmonic order.
            grid: Type of grid used for spherical harmonics, default is 'legendre-gauss'.
            radius: radius of the earth
            omega: rotation rate
            gravity:  gravity.
            havg: average height
            hamp: Average amplitude for height calculations.
        """
        self.nlat = nlat
        self.nlon = nlon
        self.device = device
        self.lmax = lmax
        self.mmax = mmax
        self.grid = grid
        self.radius = radius
        self.omega = omega
        self.gravity = gravity
        self.havg = havg
        self.hamp = hamp
        self.indpol = 10
        self.sigmoid = create_sigmoid_ramp_function(self.nlat, self.indpol).to(
            self.device
        )

        self.initialize_sht()
        self.initialize_other_properties()

    def load_data(self, filename):
        return np.load(self.dirnpy + filename)

    def initialize_sht(self):
        """
        Initialize spherical harmonics and inverse spherical harmonics transformations
        for both scalar and vector fields.
        """
        # Initialize spherical harmonics transformation objects
        self.sht = harmonics.RealSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        ).to(self.device)
        self.isht = harmonics.InverseRealSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        ).to(self.device)
        self.vsht = harmonics.RealVectorSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        ).to(self.device)
        self.ivsht = harmonics.InverseRealVectorSHT(
            self.nlat, self.nlon, self.lmax, self.mmax, self.grid, csphase=False
        ).to(self.device)
        self.lmax = self.sht.lmax
        self.mmax = self.sht.mmax

    def initialize_other_properties(self):
        """
        Initialize additional properties required for diffusion and pole filtering,
        including latitude and longitude arrays, Laplacian operators, and Coriolis effect.
        """
        # Compute quadrature weights and cosine of latitudes for the grid
        cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(
            self.nlat, -1, 1
        )
        self.lats = -torch.as_tensor(np.arcsin(cost))
        self.lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[
            : self.nlon
        ]

        l_arr = torch.arange(0, self.lmax).reshape(self.lmax, 1).double()
        l_arr = l_arr.expand(self.lmax, self.mmax)
        self.lap = (-l_arr * (l_arr + 1) / self.radius**2).to(self.device)
        self.invlap = (-(self.radius**2) / l_arr / (l_arr + 1)).to(self.device)
        self.invlap[0] = 0.0  # Adjusting the first element to avoid division by zero
        self.coriolis = 2 * self.omega * torch.sin(self.lats).reshape(self.nlat, 1)

    def grid2spec(self, ugrid):
        """
        spectral coefficients from spatial data
        """
        return self.sht(ugrid)

    def spec2grid(self, uspec):
        """
        spatial data from spectral coefficients
        """
        return self.isht(uspec)

    # ugrid is 2d:
    def vrtdivspec(self, ugrid):
        """spatial data from spectral coefficients"""
        vrtdivspec = self.lap * self.radius * self.vsht(ugrid)
        return vrtdivspec

    def getuv(self, vrtdivspec):
        """
        compute wind vector from spectral coeffs of vorticity and divergence
        """
        return self.ivsht(self.invlap * vrtdivspec / self.radius)

    def getgrad(self, chispec):
        """
        compute vector gradient on grid given complex spectral coefficients.

        Args:
            chispec: rank 1 or 2 or 3 tensor complex array with shape
        `(ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt)` containing
        complex spherical harmonic coefficients (where ntrunc is the
        triangular truncation limit and nt is the number of spectral arrays
        to be transformed). If chispec is rank 1, nt is assumed to be 1.

        Returns:
            C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
        gridded zonal and meridional components of the vector gradient.
        Shapes are either (nlat,nlon) or (nlat,nlon,nt).
        """
        idim = chispec.ndim

        if (
            len(chispec.shape) != 1
            and len(chispec.shape) != 2
            and len(chispec.shape) != 3
        ):
            msg = "getgrad needs rank one or two arrays!"
            raise ValueError(msg)

        ntrunc = int(
            -1.5
            + 0.5
            * torch.sqrt(
                9.0 - 8.0 * (1.0 - torch.tensor(self.spec2grid(chispec).shape[0]))
            )
        )

        if len(chispec.shape) == 1:
            chispec = torch.reshape(chispec, ((ntrunc + 1) * (ntrunc + 2) // 2, 1))

        divspec2 = self.lap * chispec

        if idim == 1:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]).to(
                            self.device
                        ),
                        divspec2,
                    )
                )
            )
            return torch.squeeze(uchi), torch.squeeze(vchi)
        elif idim == 2:
            uchi, vchi = self.getuv(
                torch.stack(
                    (
                        torch.zeros([divspec2.shape[0], divspec2.shape[1]]).to(
                            self.device
                        ),
                        divspec2,
                    )
                )
            )
            return uchi, vchi
        elif idim == 3:
            new_shape = (divspec2.shape[0], 2, *divspec2.shape[1:])
            stacked_divspec = torch.zeros(
                new_shape, dtype=torch.complex64, device=self.device
            )
            # Copy the original data into the second slice of the new dimension
            stacked_divspec[:, 1, :, :] = divspec2
            backy = self.getuv(stacked_divspec)
            uchi = backy[:, 0, :, :]
            vchi = backy[:, 1, :, :]
            return uchi, vchi
        else:
            print("nothing happening here")

    def polefilt_lap2d_V2(self, U, V, substeps):
        """
        Enhances the characteristics of a two-dimensional (2D) vector field by applying a
        combination of pole filtering and Laplacian-based correction. This function is
        designed to refine the input vector field by selectively suppressing the influence
        of specific poles and adjusting the field to better reflect physical constraints
        and properties. It achieves this through a sequence of operations in the spectral
        domain, focusing on the field's divergence, vorticity, and Laplacian characteristics.

        The process involves initial pole filtering to mitigate the effects of unwanted
        features followed by a detailed correction phase that leverages the field's
        Laplacian to enforce smoothness and continuity. The correction phase is further
        enhanced by considering the field's divergence and vorticity, ensuring that the
        final vector field adheres more closely to the expected physical behavior.

        Args:
            U (Tensor): The x-component of the velocity or vector field. This tensor should
            represent one of the two dimensions of the field, with spatial dimensions that
            match those of the V component.
            V (Tensor): The y-component of the velocity or vector field. This tensor complements the U component
            by representing the second dimension of the field.
            substeps (int): The number of iterations for the correction process. This parameter
            controls the depth of the refinement process, with more substeps leading to a more
            pronounced adjustment of the vector field.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the modified x and y components (U, V) of the
            vector field after the pole filtering and Laplacian-based correction have been applied.
            These components will have undergone adjustments to suppress specified poles and to
            refine their characteristics based on divergence, vorticity, and Laplacian considerations.
        """
        U = polfiltT(U.clone().detach(), self.indpol).to(self.device)
        V = polfiltT(V.clone().detach(), self.indpol).to(self.device)

        if len(U.shape) == 2:
            for suby in range(substeps):
                # print(suby)
                # the hard shit:
                ugrid = torch.stack((U, V)).to(self.device)
                vrt, div = self.vrtdivspec(ugrid)
                ddiv_dx, ddiv_dy = self.getgrad(div)
                ddx_dx2, ddx_dy2 = self.getgrad(self.grid2spec(ddiv_dx))
                ddy_dx2, ddy_dy2 = self.getgrad(self.grid2spec(ddiv_dy))
                lappy = ddx_dx2 + ddy_dy2
                dlapdx, dlapdy = self.getgrad(self.grid2spec(lappy))
                U = U - (dlapdx * self.sigmoid[:, None] * 2e16)
                V = V - (dlapdy * self.sigmoid[:, None] * 2e16)
            return U, V
        if len(U.shape) == 3:
            for suby in range(substeps):
                ugrid = torch.stack((U, V), dim=1).to(self.device)
                pp = self.vrtdivspec(ugrid)
                ddiv_dx, ddiv_dy = self.getgrad(pp[:, 1, :, :].to(self.device))
                ddx_dx2, ddx_dy2 = self.getgrad(self.grid2spec(ddiv_dx))
                ddy_dx2, ddy_dy2 = self.getgrad(self.grid2spec(ddiv_dy))
                lappy = ddx_dx2 + ddy_dy2
                dlapdx, dlapdy = self.getgrad(self.grid2spec(lappy))
                U = U - (dlapdx * self.sigmoid[:, None] * 2e16)
                V = V - (dlapdy * self.sigmoid[:, None] * 2e16)
            return U, V

    def polefilt_lap2d_V1(self, T, substeps):
        """
        Applies a pole filtering transformation followed by a Laplacian-based correction
        to a scalar in 2D space.

        The function aims to modify the scalar field to suppress
        features associated with specified poles, and to adjust the field based on
        its divergence, vorticity, and Laplacian properties through a series of
        spectral domain operations.

        Args:
            T (Tensor): scalar-component of the velocity or vector field.
            substeps: number of substeps

        Returns:
            Tuple of Tensors: The modified T components of the scalar field.
        """
        T = polfiltT(T.clone().detach(), self.indpol)

        for suby in range(substeps):
            # the hard shit:
            ugrid = T
            dT_dx, dT_dy = self.getgrad(self.grid2spec(ugrid))
            ddx_dx2, ddx_dy2 = self.getgrad(self.grid2spec(dT_dx))
            ddy_dx2, ddy_dy2 = self.getgrad(self.grid2spec(dT_dy))
            lappy = ddx_dx2 + ddy_dy2
            T = T + (lappy * self.sigmoid[:, None].to(self.device) * 1e8)
        return T

    def polefilt_lap2d_QV1(self, T, substeps):
        """
        Applies a pole filtering transformation followed by a Laplacian-based correction
        to a scalar in 2D space.

        The function aims to modify the scalar field to suppress
        features associated with specified poles, and to adjust the field based on
        its divergence, vorticity, and Laplacian properties through a series of
        spectral domain operations.

        Args:
            T (Tensor): scalar-component of the velocity or vector field.
            ind_pol (list/int): Index/indices specifying poles for the filtering process.

        Returns:
            Tuple of Tensors: The modified T components of the scalar field.
        """
        T = polfiltT(T.clone().detach(), self.indpol)

        for suby in range(substeps):
            # print(suby)
            # the hard shit:
            ugrid = T
            dT_dx, dT_dy = self.getgrad(self.grid2spec(ugrid))
            ddx_dx2, ddx_dy2 = self.getgrad(self.grid2spec(dT_dx))
            ddy_dx2, ddy_dy2 = self.getgrad(self.grid2spec(dT_dy))
            lappy = ddx_dx2 + ddy_dy2
            T = T + (lappy * self.sigmoid[:, None] * 0.5e8)
        return T

    def diff_lap2d_filt(self, BB2_tensor):
        # Create a new tensor 'BBfix' as a copy of 'BB2_tensor' but ensure it's only copied once.
        BBfix = BB2_tensor.clone().to(
            self.device
        )  # Ensure it's on the same device as needed

        # Apply filters directly on slices of 'BBfix' without additional unnecessary cloning.
        BBfix[:16], BBfix[16:32] = self.polefilt_lap2d_V2(
            BBfix[:16], BBfix[16:32], substeps=6
        )
        BBfix[32:48] = self.polefilt_lap2d_V1(BBfix[32:48], substeps=5)
        BBfix[48:64] = self.polefilt_lap2d_QV1(BBfix[48:64], substeps=8)

        # Since BB2_tensor[61], BB2_tensor[63], and BB2_tensor[62] are mentioned,
        # it seems there might have been a typo in the original indices given.
        # Adjusting indices based on the sequence and assuming BB2_tensor should not skip indices
        # Also, correcting indices based on the order and avoiding overwriting without reading first
        SP = self.polefilt_lap2d_V1(BB2_tensor[64].clone().to(self.device), substeps=5)
        T2m = self.polefilt_lap2d_V1(BB2_tensor[65].clone().to(self.device), substeps=5)
        U500, V500 = self.polefilt_lap2d_V2(
            BB2_tensor[67].clone().to(self.device),
            BB2_tensor[66].clone().to(self.device),
            substeps=6,
        )
        T500 = self.polefilt_lap2d_V1(
            BB2_tensor[68].clone().to(self.device), substeps=5
        )
        # Assuming Q500 should use BB2_tensor[64] if we're following sequential order
        Q500 = self.polefilt_lap2d_QV1(
            BB2_tensor[69].clone().to(self.device), substeps=4
        )

        # Update BBfix with the results of filtering operations
        BBfix[64] = SP
        BBfix[65] = T2m
        # Ensure the order of assignment does not overwrite any needed value
        BBfix[67] = U500
        BBfix[66] = V500
        BBfix[68] = T500
        BBfix[69] = Q500  # Assuming the corrected index is 64 for Q500

        return BBfix


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dirnpy = "/glade/u/home/schreck/schreck/repos/global/miles-credit/results/spectral_norm/forecasts/"
    BB1 = np.load(dirnpy + "0_1527822000_4_pred.npy")
    BB2 = np.load(dirnpy + "0_1527825600_5_pred.npy")
    U = BB2[8, :, :]
    V = BB2[8, :, :]
    nlat = U.shape[0]
    nlon = U.shape[1]
    BB2_tensor = torch.tensor(BB2).clone().detach().to(device)
    # instantiate this with the original model:
    DPF = Diffusion_and_Pole_Filter(nlat=nlat, nlon=nlon, device=device)

    # this fixes the divergent modes ... might eb doing too much to Q
    BBfix = DPF.diff_lap2d_filt(BB2_tensor)

    print("done")
    torch.cuda.empty_cache()
    gc.collect()
