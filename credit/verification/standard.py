import numpy as np
import xarray as xr
import torch
from torch_harmonics import RealSHT, RealVectorSHT

import logging

logger = logging.getLogger(__name__)

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
    lmax = nlat + 1 # Maximum degree for the transform
    with torch.no_grad():
        sht = RealSHT(nlat, nlon, lmax=lmax, grid=grid, norm=norm)

        data = torch.tensor(da.values, dtype=torch.float64)
        coeffs = sht(data)

        ### compute zonal spectra
        # square then multiply everything but l=0 by 2
        times_two = 2. * torch.ones(coeffs.shape[-1])
        times_two[0] = 1.
        # sum over l of coeffs matrix with dim l,m
        spectrum = ((torch.abs(coeffs ** 2) * times_two).sum(dim=-2))
        
        return spectrum

def average_div_rot_spectrum(ds, grid, wave_spec="n", norm="ortho"):
    """
    takes the average of all divergence and rotational spectra in da

    input: Torch Tensor with dim  (..., n, m), total_wavenum x ...
    output: numpy array with dim (wavenumber)

    """

    reduce_dim = -1 if wave_spec == "n" else -2 # which wavenumber spectrum to compute

    vrt, div = div_rot_spectrum(ds, grid, norm) # (..., n, m)

    # square then multiply everything but index l=0 by 2 then sum
    times_two = 2. * torch.ones(vrt.shape[-1])
    times_two[0] = 1.

    vrt_spectrum = ((torch.abs(vrt ** 2) * times_two).sum(dim=reduce_dim))
    div_spectrum = ((torch.abs(div ** 2) * times_two).sum(dim=reduce_dim))
    logger.info(f"vrt:{vrt_spectrum.shape}")

    # average over all batch dimensions
    dims_for_avg = list(range(len(vrt_spectrum.shape) - 1))
    avg_vrt_spectrum = vrt_spectrum.mean(dim=dims_for_avg)
    avg_div_spectrum = div_spectrum.mean(dim=dims_for_avg)
    logger.info(avg_vrt_spectrum.shape)

    return avg_vrt_spectrum.detach().numpy().flatten(), avg_div_spectrum.detach().numpy().flatten()

def div_rot_spectrum(ds, grid, norm="ortho"):
    """
    Returns the spectrum of the divergent and rotational components of a flow 

    input: Dataset with variables U,V each with backing array with dim (..., lat, lon)
    output: Torch Tensor with dim  (..., nlat // 2 + 1)

    """
    nlat, nlon = len(ds.latitude), len(ds.longitude)
    lmax = nlat + 1 # Maximum degree for the transform
    with torch.no_grad(): # for speed: don't want to keep track of gradients
        vsht = RealVectorSHT(nlat, nlon, lmax=lmax, grid=grid, norm=norm, csphase=False)

        U = torch.tensor(ds.U.values, dtype=torch.float64).unsqueeze(-3)
        V = torch.tensor(ds.V.values, dtype=torch.float64).unsqueeze(-3)
        uv = torch.cat((U, V), dim=-3) # concat so -3 dim corresponds to U, V

        vrtdivspec = vsht(uv)

        vrt = vrtdivspec[..., 0, :, :]
        div = vrtdivspec[..., 1, :, :]

        return vrt, div