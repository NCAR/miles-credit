import numpy as np
import xarray as xr
import torch
from torch_harmonics import RealSHT

import logging

logger = logging.getLogger(__name__)

def average_zonal_spectrum(da, grid, norm="ortho"):
    spectrum_raw = zonal_spectrum(da, grid, norm)
    average_spectrum = spectrum_raw.mean(dim=list(range(len(spectrum_raw.shape) - 1)))
    return average_spectrum.detach().numpy()

def zonal_spectrum(da, grid, norm="ortho"):
    """
    last two dims of da need to be lat, lon
    """

    nlat, nlon = len(da.latitude), len(da.longitude)
    lmax = nlat + 1 # Maximum degree for the transform

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