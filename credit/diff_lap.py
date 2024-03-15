import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
# from scipy.ndimage import convolve
#import windspharm
#from windspharm.standard import VectorWind
from torch_harmonics import *
import torch_harmonics as harmonics
import copy



def grid2spec(ugrid):
    """
    spectral coefficients from spatial data
    """
    return sht(ugrid)

def spec2grid(uspec):
    """
    spatial data from spectral coefficients
    """
    return isht(uspec)

#ugrid is 2d: 
def vrtdivspec(ugrid):
    """spatial data from spectral coefficients"""
    vrtdivspec = lap * radius * vsht(ugrid)
    return vrtdivspec

def getuv(vrtdivspec):
    """
    compute wind vector from spectral coeffs of vorticity and divergence
    """
    return ivsht(invlap * vrtdivspec /radius)

def getgrad(chispec):

    """
     compute vector gradient on grid given complex spectral coefficients.
    
     @param chispec: rank 1 or 2 numpy complex array with shape
     (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt) containing
     complex spherical harmonic coefficients (where ntrunc is the
     triangular truncation limit and nt is the number of spectral arrays
     to be transformed). If chispec is rank 1, nt is assumed to be 1.
    
     @return: C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
     gridded zonal and meridional components of the vector gradient.
     Shapes are either (nlat,nlon) or (nlat,nlon,nt).
    """
    chispec.to(device)
    idim = chispec.ndim

    if len(chispec.shape) !=1 and len(chispec.shape) !=2:
        msg = 'getgrad needs rank one or two arrays!'
        raise ValueError(msg)

    ntrunc = int(-1.5 + 0.5*torch.sqrt(9.-8.*(1.-torch.tensor(grid2spec(U).shape[0]))))

    if len(chispec.shape) == 1:
        nt = 1
        chispec = torch.reshape(chispec, ((ntrunc+1)*(ntrunc+2)//2,1))
    else:
        nt = chispec.shape[1]

    divspec2 = (lap*chispec).to(device)
    uchi, vchi = getuv(torch.stack((torch.zeros([divspec2.shape[0],divspec2.shape[1]]).to(device),divspec2)))

    if idim == 1:
        return torch.squeeze(uchi), torch.squeeze(vchi)
    else:
        return uchi, vchi


def getgradspec(chispec):

    """
     compute vector gradient on grid given complex spectral coefficients.
    
     @param chispec: rank 1 or 2 numpy complex array with shape
     (ntrunc+1)*(ntrunc+2)/2 or ((ntrunc+1)*(ntrunc+2)/2,nt) containing
     complex spherical harmonic coefficients (where ntrunc is the
     triangular truncation limit and nt is the number of spectral arrays
     to be transformed). If chispec is rank 1, nt is assumed to be 1.
    
     @return: C{B{uchi, vchi}} - rank 2 or 3 numpy float32 arrays containing
     gridded zonal and meridional components of the vector gradient.
     Shapes are either (nlat,nlon) or (nlat,nlon,nt).
    """
    chispec.to(device)
    idim = chispec.ndim

    if len(chispec.shape) !=1 and len(chispec.shape) !=2:
        msg = 'getgrad needs rank one or two arrays!'
        raise ValueError(msg)

    ntrunc = int(-1.5 + 0.5*torch.sqrt(9.-8.*(1.-torch.tensor(grid2spec(U).shape[0]))))

    if len(chispec.shape) == 1:
        nt = 1
        chispec = torch.reshape(chispec, ((ntrunc+1)*(ntrunc+2)//2,1))
    else:
        nt = chispec.shape[1]

    divspec2 = (lap*chispec).to(device)

    if idim == 1:
        return torch.squeeze(uchi), torch.squeeze(vchi)
    else:
        return divspec2

def polfilt(D,inddo):
    for ii in np.concatenate([np.arange(-inddo,0),np.arange(1,inddo+1)]):
        ts_Udo = copy.deepcopy(D[ii,:])
        Z = np.fft.fft(ts_Udo)
        Yfft = Z/np.size(ts_Udo)
        freq = np.fft.rfftfreq(len(ts_Udo))
        perd = 1/freq[1:]
        val_1d, ind_1d = find_nearest(perd,value=100)
        Ck2 = 2.*np.abs(Yfft[0:int(np.size(ts_Udo)/2)+1])**2 
        var_actual = np.var(ts_Udo)
        a = Yfft[np.arange(0,int(np.size(ts_Udo)/2)+1)]
        s=np.sum(a[1::] * np.conj(a[1::]))  # on't want to include the mean, since this is not in the variance calculation
        var_spectrum = np.real(2 * s)  #multiply by two in order to conserve variance
        A = Ck2/np.sum(Ck2)
        A[ind_1d:] = 0.
        Zlow = np.copy(Z)
        Zlow[ind_1d:-ind_1d] = 0.0
        X_filtered11 = np.real(np.fft.ifft(Zlow))
        D[ii,:]=X_filtered11
    return D

def polfiltT(D,inddo):
    for ii in torch.concatenate([torch.arange(-inddo,0),torch.arange(1,inddo+1)]):
        # print(ii)
        ts_Udo = copy.deepcopy(D[ii,:])
        Z = torch.fft.fft(ts_Udo)
        Yfft = Z/ts_Udo.size()[0]
        freq = torch.fft.rfftfreq(len(ts_Udo))
        perd = 1/freq[1:]
        val_1d, ind_1d = find_nearest(perd,value=100)
        Ck2 = 2.*torch.abs(Yfft[0:int(ts_Udo.size()[0]/2)+1])**2 
        var_actual = torch.var(ts_Udo)
        a = Yfft[torch.arange(0,int(ts_Udo.size()[0]/2)+1)]
        s=torch.sum(a[1::] * torch.conj(a[1::]))  # on't want to include the mean, since this is not in the variance calculation
        var_spectrum = np.real(2 * s)  #multiply by two in order to conserve variance
        A = Ck2/torch.sum(Ck2)
        A[ind_1d:] = 0.
        Zlow = torch.clone(Z)
        Zlow[ind_1d:-ind_1d] = 0.0
        X_filtered11 = torch.real(torch.fft.ifft(Zlow))
        D[ii,:]=X_filtered11
    return D

def find_nearest(array, value):
    array = torch.asarray(array)
    idx = (torch.abs(array - value)).argmin()
    return array[idx],idx


def create_sigmoid_ramp_function(array_length, ramp_length):
    """
    Creates an array of specified length with a sigmoid ramp up and down.
    
    Parameters:
    - array_length: The length of the output array.
    - ramp_length: The length of the ramp up and down.
    
    Returns:
    - An array of the specified length with the described ramp up and down using a sigmoid function.
    """
    import numpy as np
    
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


def polefilt_lap2d_V2(U,V,ind_pol,device,sigmoid_ramp_array):
    """
    Applies a pole filtering transformation followed by a Laplacian-based correction
    to two components of a velocity field (or similar vector field) in 2D space.

    The function aims to modify the vector field to suppress
    features associated with specified poles, and to adjust the field based on
    its divergence, vorticity, and Laplacian properties through a series of
    spectral domain operations.

    Parameters:
    - U (Tensor): x-component of the velocity or vector field.
    - V (Tensor): y-component of the velocity or vector field.
    - ind_pol (list/int): Index/indices specifying poles for the filtering process.

    Returns:
    - Tuple of Tensors: The modified x and y components of the vector field.
    """
    
    #U = polfiltT(torch.tensor(U).to(device),ind_pol).to(device)
    #V = polfiltT(torch.tensor(V).to(device),ind_pol).to(device)

    U = polfiltT(U.clone().detach(),ind_pol).to(device)
    V = polfiltT(V.clone().detach(),ind_pol).to(device)

    for suby in range(7):
        #print(suby)
        #the hard shit: 
        ugrid = torch.stack((U, V)).to(device)
        vrt,div = vrtdivspec(ugrid)
        ddiv_dx,ddiv_dy=getgrad(div)
        ddx_dx2,ddx_dy2 =getgrad(grid2spec(ddiv_dx))
        ddy_dx2,ddy_dy2 = getgrad(grid2spec(ddiv_dy))
        lappy = ddx_dx2+ddy_dy2
        dlapdx,dlapdy = getgrad(grid2spec(lappy))
        U = U-(dlapdx*sigmoid_ramp_array[:,None]*2e16)
        V = V-(dlapdy*sigmoid_ramp_array[:,None]*2e16)
    return U,V


def polefilt_lap2d_V1(T,ind_pol,device,sigmoid_ramp_array):
    """
    Applies a pole filtering transformation followed by a Laplacian-based correction
    to a scalar in 2D space.

    The function aims to modify the scalar field to suppress
    features associated with specified poles, and to adjust the field based on
    its divergence, vorticity, and Laplacian properties through a series of
    spectral domain operations.

    Parameters:
    - T (Tensor): scalar-component of the velocity or vector field.
    - ind_pol (list/int): Index/indices specifying poles for the filtering process.

    Returns:
    - Tuple of Tensors: The modified T components of the scalar field.
    """
    T = polfiltT(T.clone().detach(),ind_pol)

    for suby in range(7):
        #print(suby)
        #the hard shit: 
        ugrid = T
        dT_dx,dT_dy=getgrad(grid2spec(ugrid))
        ddx_dx2,ddx_dy2 =getgrad(grid2spec(dT_dx))
        ddy_dx2,ddy_dy2 = getgrad(grid2spec(dT_dy))
        lappy = ddx_dx2+ddy_dy2
        T = T+(lappy*sigmoid_ramp_array[:,None].to(device)*1e8)
    return T

def polefilt_lap2d_QV1(T,ind_pol,device,sigmoid_ramp_array):
    """
    Applies a pole filtering transformation followed by a Laplacian-based correction
    to a scalar in 2D space.

    The function aims to modify the scalar field to suppress
    features associated with specified poles, and to adjust the field based on
    its divergence, vorticity, and Laplacian properties through a series of
    spectral domain operations.

    Parameters:
    - T (Tensor): scalar-component of the velocity or vector field.
    - ind_pol (list/int): Index/indices specifying poles for the filtering process.

    Returns:
    - Tuple of Tensors: The modified T components of the scalar field.
    """
    T=T.clone()
    T = polfiltT(T.clone().detach(),ind_pol)

    for suby in range(5):
        #print(suby)
        #the hard shit: 
        ugrid = T
        dT_dx,dT_dy=getgrad(grid2spec(ugrid))
        ddx_dx2,ddx_dy2 =getgrad(grid2spec(dT_dx))
        ddy_dx2,ddy_dy2 = getgrad(grid2spec(dT_dy))
        lappy = ddx_dx2+ddy_dy2
        T = T+(lappy*sigmoid_ramp_array[:,None]*0.5e8)
    return T



def diff_lap2d_filt(BB2):
    ind_pol=10
    sigmoid_ramp_array = create_sigmoid_ramp_function(nlat, ind_pol)
    sigmoid_ramp_array = sigmoid_ramp_array.to(device)

    #send to device, torch tensorize:
    BB2=torch.tensor(BB2).to(device)
    BB2_cp = torch.zeros_like(BB2).to(device)

    #U,V
    for ii in range(0,15):
        BB2_cp[ii,:,:],BB2_cp[ii+15,:,:]= polefilt_lap2d_V2(BB2[ii,:,:],BB2[ii+15,:,:],ind_pol,device,sigmoid_ramp_array.to(device))

    #T
    for ii in range(0,15):
        BB2_cp[ii+30,:,:]= polefilt_lap2d_V1(BB2[ii+30,:,:],ind_pol,device,sigmoid_ramp_array.to(device))

    #Q
    for ii in range(0,15):
        BB2_cp[ii+45,:,:]= polefilt_lap2d_QV1(BB2[ii+45,:,:],ind_pol,device,sigmoid_ramp_array.to(device))

    #T500? 
    BB2_cp[61,:,:]=polefilt_lap2d_V1(BB2[61,:,:],ind_pol,device,sigmoid_ramp_array.to(device))

    #Q500? 
    BB2_cp[66,:,:]=polefilt_lap2d_QV1(BB2[66,:,:],ind_pol,device,sigmoid_ramp_array.to(device))

    return BB2_cp



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dirnpy= '/glade/u/home/schreck/schreck/repos/global/miles-credit/results/spectral_norm/forecasts/'
    BB1 = np.load(dirnpy+'0_1527818400_3_pred.npy')
    BB2 = np.load(dirnpy+'0_1527818400_3_pred.npy')
    U = BB2[8,:,:]
    V = BB2[8+15,:,:]
    nlat = U.shape[0]
    nlon = U.shape[1]
    grid = 'legendre-gauss'
    lmax=None
    mmax=None
    radius=6.37122E6
    omega=7.292E-5
    gravity=9.80616
    havg=10.e3
    hamp=120.
    
    sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False).to(device)
    isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False).to(device)
    vsht = harmonics.RealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False).to(device)
    ivsht = harmonics.InverseRealVectorSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid, csphase=False).to(device)
    lmax = sht.lmax
    mmax = sht.mmax
    cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(nlat, -1, 1)
    lats = -torch.as_tensor(np.arcsin(cost))
    lons = torch.linspace(0, 2*np.pi, nlon+1, dtype=torch.float64)[:nlon]
    
    U = torch.tensor(U).to(device)
    V = torch.tensor(V).to(device)
    ugrid = torch.stack((U, V)).to(device)
    
    #laplacian and inverse laplacian operators::::
    l = torch.arange(0, lmax).reshape(lmax, 1).double()
    l = l.expand(lmax, mmax)
    lap = (- l * (l + 1) / radius**2).to(device)
    invlap = (- radius**2 / l / (l + 1)).to(device)
    invlap[0] = 0.

    ind_pol = 10
    BB2_filt=diff_lap2d_filt(BB2)
    print('done filtering')





