import numpy as np
from scipy import ndimage
import xarray as xr
import torch
from torch_harmonics import RealSHT, RealVectorSHT


import logging

logger = logging.getLogger(__name__)


def radial_fft_spectrum(da, dx, dim=("latitude", "longitude")):
    """
    Compute radially averaged 2D FFT power spectrum for each lat-lon slice.

    Parameters
    ----------
    da : xarray.DataArray
        Input array with shape (..., latitude, longitude)
    dx : float
        Grid spacing in km (assumes equal spacing in both dimensions)
    dim : tuple of str
        Names of the latitude and longitude dimensions

    Returns
    -------
    spectrum : xarray.DataArray
        Radially averaged power spectrum with new dimensions 'wavenumber' and 'wavelength'
    """
    lat_dim, lon_dim = dim

    # Get the size of lat-lon grid
    nlat = da.sizes[lat_dim]
    nlon = da.sizes[lon_dim]

    # Compute 2D FFT along lat-lon dimensions
    fft_2d = xr.apply_ufunc(
        np.fft.fft2,
        da,
        input_core_dims=[[lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim]],
        vectorize=True,
    )

    # Compute power spectrum
    power = np.abs(fft_2d) ** 2

    # Shift zero frequency to center
    power_shifted = xr.apply_ufunc(
        np.fft.fftshift,
        power,
        input_core_dims=[[lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim]],
        vectorize=True,
    )

    # Radially average the power spectrum
    def radial_average(power_2d):
        """Average 2D power spectrum into radial bins"""
        wc = nlon // 2
        hc = nlat // 2
        min_c = min(wc, hc)

        # create an array of integer radial distances from the center
        Y, X = np.ogrid[0:nlat, 0:nlon]
        r = np.hypot(X - wc, Y - hc).astype(int)

        # SUM all psd2D pixels with label 'r' for 0<=r<=min_c
        # NOTE: this will miss power contributions in 'corners' r>min_c
        psd1D = ndimage.sum(power_2d, r, index=np.arange(0, min_c))

        return psd1D

    min_dim = min(nlon, nlat)
    min_c = min_dim // 2

    # Apply radial averaging to each slice
    spectrum = xr.apply_ufunc(
        radial_average,
        power_shifted,
        input_core_dims=[[lat_dim, lon_dim]],
        output_core_dims=[["wavenumber"]],
        vectorize=True,
        output_sizes={"wavenumber": min_c},
    )

    # Create coordinate for wavenumbers and wavelengths
    wavenumbers = np.arange(min_c)
    spectrum = spectrum.assign_coords(wavenumber=wavenumbers)
    wavelengths = 1 / np.fft.rfftfreq(min_c * 2, d=dx)[:min_c]
    spectrum = spectrum.assign_coords(wavelength=("wavenumber", wavelengths))

    return spectrum


def average_zonal_spectrum(da, grid, norm="ortho"):
    """
    takes the average of all spectra in da

    input: Torch Tensor with dim  (..., wavenumber)
    output: numpy array with dim (wavenumber)

    """

    spectrum_raw = zonal_spectrum(da, grid, norm)
    average_spectrum = spectrum_raw.mean(dim=list(range(len(spectrum_raw.shape) - 1)))
    return average_spectrum.detach().numpy()


def zonal_spectrum(da, grid, norm="ortho"):
    """
    Returns the zonal energy spectrum of a dataarray with dimensions

    input: DataArray with backing array with dim (..., lat, lon)
    output: Torch Tensor with dim  (..., nlat // 2 + 1)

    """

    nlat, nlon = len(da.latitude), len(da.longitude)
    lmax = nlat + 1  # Maximum degree for the transform
    with torch.no_grad():
        sht = RealSHT(nlat, nlon, lmax=lmax, grid=grid, norm=norm)

        data = torch.tensor(da.values, dtype=torch.float64)
        coeffs = sht(data)

        ### compute zonal spectra
        # square then multiply everything but l=0 by 2
        times_two = 2.0 * torch.ones(coeffs.shape[-1])
        times_two[0] = 1.0
        # sum over l of coeffs matrix with dim l,m
        spectrum = (torch.abs(coeffs**2) * times_two).sum(dim=-2)

        return spectrum


def average_div_rot_spectrum(ds, grid, wave_spec="n", norm="ortho"):
    """
    takes the average of all divergence and rotational spectra in da

    input: Torch Tensor with dim  (..., n, m), total_wavenum x ...
    output: numpy array with dim (wavenumber)

    """

    reduce_dim = -1 if wave_spec == "n" else -2  # which wavenumber spectrum to compute

    vrt, div = div_rot_spectrum(ds, grid, norm)  # (..., n, m)

    # square then multiply everything but index l=0 by 2 then sum
    times_two = 2.0 * torch.ones(vrt.shape[-1])
    times_two[0] = 1.0

    vrt_spectrum = (torch.abs(vrt**2) * times_two).sum(dim=reduce_dim)
    div_spectrum = (torch.abs(div**2) * times_two).sum(dim=reduce_dim)
    logger.info(f"vrt:{vrt_spectrum.shape}")

    # average over all batch dimensions
    dims_for_avg = list(range(len(vrt_spectrum.shape) - 1))
    avg_vrt_spectrum = vrt_spectrum.mean(dim=dims_for_avg)
    avg_div_spectrum = div_spectrum.mean(dim=dims_for_avg)
    logger.info(avg_vrt_spectrum.shape)

    return (
        avg_vrt_spectrum.detach().numpy().flatten(),
        avg_div_spectrum.detach().numpy().flatten(),
    )


def div_rot_spectrum(ds, grid, norm="ortho"):
    """
    Returns the spectrum of the divergent and rotational components of a flow

    input: Dataset with variables U,V each with backing array with dim (..., lat, lon)
    output: Torch Tensor with dim  (..., nlat // 2 + 1)

    """
    nlat, nlon = len(ds.latitude), len(ds.longitude)
    lmax = nlat + 1  # Maximum degree for the transform
    with torch.no_grad():  # for speed: don't want to keep track of gradients
        vsht = RealVectorSHT(nlat, nlon, lmax=lmax, grid=grid, norm=norm, csphase=False)

        U = torch.tensor(ds.U.values, dtype=torch.float64).unsqueeze(-3)
        V = torch.tensor(ds.V.values, dtype=torch.float64).unsqueeze(-3)
        uv = torch.cat((U, V), dim=-3)  # concat so -3 dim corresponds to U, V

        vrtdivspec = vsht(uv)

        vrt = vrtdivspec[..., 0, :, :]
        div = vrtdivspec[..., 1, :, :]

        return vrt, div
