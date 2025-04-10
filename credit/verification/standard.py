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

def average_div_rot_spectrum(ds, grid, norm="ortho"):
    """
    takes the average of all divergence and rotational spectra in da

    input: Torch Tensor with dim  (..., wavenumber)
    output: numpy array with dim (wavenumber)

    """

    vrt_raw, div_raw = div_rot_spectrum(ds, grid, norm)

    dims_for_avg = list(range(len(vrt_raw.shape) - 1))
    avg_vrt = vrt_raw.mean(dim=dims_for_avg)
    avg_div = div_raw.mean(dim=dims_for_avg)

    return avg_vrt.detach().numpy(), avg_div.detach().numpy()

def div_rot_spectrum(ds, grid, norm="ortho"):
    """
    Returns the spectrum of the divergent and rotational components of a flow 

    input: Dataset with variables U,V each with backing array with dim (..., lat, lon)
    output: Torch Tensor with dim  (..., nlat // 2 + 1)

    """
    nlat, nlon = len(ds.latitude), len(ds.longitude)
    lmax = nlat + 1 # Maximum degree for the transform
    with torch.no_grad():
        vsht = RealVectorSHT(nlat, nlon, lmax=lmax, grid=grid, norm=norm, csphase=False)

        uv = torch.cat([torch.tensor(ds.U, dtype=torch.float64).unsqueeze(-3),
                        torch.tensor(ds.V, dtype=torch.float64).unsqueeze(-3)],
                        dim = -3)

        vrt, div = vsht(uv)

        return vrt, div