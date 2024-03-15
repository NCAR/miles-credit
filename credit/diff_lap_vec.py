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

    if len(chispec.shape) != 1 and len(chispec.shape) != 2 and len(chispec.shape) != 3:
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
        uchi, vchi = getuv(torch.stack((torch.zeros([divspec2.shape[0],divspec2.shape[1]]).to(device),divspec2)))
        return torch.squeeze(uchi), torch.squeeze(vchi)
    elif idim == 2:
        uchi, vchi = getuv(torch.stack((torch.zeros([divspec2.shape[0],divspec2.shape[1]]).to(device),divspec2)))
        return uchi, vchi
    elif idim == 3:
        new_shape = (divspec2.shape[0], 2, *divspec2.shape[1:])
        stacked_divspec = torch.zeros(new_shape,dtype=torch.complex64, device=device)
        # Copy the original data into the second slice of the new dimension
        stacked_divspec[:, 1, :, :] = divspec2
        backy = getuv(stacked_divspec)
        uchi = backy[:,0,:,:]
        vchi = backy[:,1,:,:] 
        return uchi, vchi
    else:
        print('nothing happening here')


def polfiltT(D,inddo):

    if len(D.shape)==2:
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

    if len(D.shape)==3:
        for jj in range(D.shape[0]):
            for ii in torch.concatenate([torch.arange(-inddo,0),torch.arange(1,inddo+1)]):
                # print(ii)
                ts_Udo = copy.deepcopy(D[jj,ii,:])
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
                D[jj,ii,:]=X_filtered11
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


def polefilt_lap2d_V2(U,V,ind_pol,device,sigmoid_ramp_array,substeps):
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
    
    Parameters:
    - U (Tensor): The x-component of the velocity or vector field. This tensor should
      represent one of the two dimensions of the field, with spatial dimensions that
      match those of the V component.
    - V (Tensor): The y-component of the velocity or vector field. This tensor complements
      the U component by representing the second dimension of the field.
    - ind_pol (list/int): Index or indices specifying the poles to be filtered out from
      the vector field. These indices target specific features or regions within the
      field for suppression.
    - device (torch.device): The computational device (CPU or GPU) where the operations
      will be performed. This parameter ensures that tensors are appropriately allocated
      for efficient computation.
    - sigmoid_ramp_array (Tensor): An array used to modulate the intensity of the Laplacian
      correction applied to the vector field. This array typically represents a spatially
      varying factor that adjusts the correction strength across different regions of the field.
    - substeps (int): The number of iterations for the correction process. This parameter
      controls the depth of the refinement process, with more substeps leading to a more
      pronounced adjustment of the vector field.
    
    Returns:
    - Tuple[Tensor, Tensor]: A tuple containing the modified x and y components (U, V) of the
      vector field after the pole filtering and Laplacian-based correction have been applied.
      These components will have undergone adjustments to suppress specified poles and to
      refine their characteristics based on divergence, vorticity, and Laplacian considerations.
    """
    U = polfiltT(U.clone().detach(),ind_pol).to(device)
    V = polfiltT(V.clone().detach(),ind_pol).to(device)

    if len(U.shape) == 2:
        for suby in range(substeps):
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
        return U, V
    if len(U.shape) == 3:
        for suby in range(substeps):
            ugrid = torch.stack((U,V),dim=1).to(device)
            pp = vrtdivspec(ugrid)
            ddiv_dx,ddiv_dy=getgrad(pp[:,1,:,:].to(device))
            ddx_dx2,ddx_dy2 =getgrad(grid2spec(ddiv_dx))
            ddy_dx2,ddy_dy2 = getgrad(grid2spec(ddiv_dy))
            lappy = ddx_dx2+ddy_dy2
            dlapdx,dlapdy = getgrad(grid2spec(lappy))
            U = U-(dlapdx*sigmoid_ramp_array[:,None]*2e16)
            V = V-(dlapdy*sigmoid_ramp_array[:,None]*2e16)
        return U, V



def polefilt_lap2d_V1(T,ind_pol,device,sigmoid_ramp_array,substeps):
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

    for suby in range(substeps):
        #the hard shit: 
        ugrid = T
        dT_dx,dT_dy=getgrad(grid2spec(ugrid))
        ddx_dx2,ddx_dy2 =getgrad(grid2spec(dT_dx))
        ddy_dx2,ddy_dy2 = getgrad(grid2spec(dT_dy))
        lappy = ddx_dx2+ddy_dy2
        T = T+(lappy*sigmoid_ramp_array[:,None].to(device)*1e8)
    return T


def polefilt_lap2d_QV1(T,ind_pol,device,sigmoid_ramp_array,substeps):
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

    for suby in range(substeps):
        #print(suby)
        #the hard shit: 
        ugrid = T
        dT_dx,dT_dy=getgrad(grid2spec(ugrid))
        ddx_dx2,ddx_dy2 =getgrad(grid2spec(dT_dx))
        ddy_dx2,ddy_dy2 = getgrad(grid2spec(dT_dy))
        lappy = ddx_dx2+ddy_dy2
        T = T+(lappy*sigmoid_ramp_array[:,None]*0.5e8)
    return T


def diff_lap2d_filt(BB2_tensor):
    ind_pol=10
    sigmoid_ramp_array = create_sigmoid_ramp_function(nlat, ind_pol)
    sigmoid_ramp_array = sigmoid_ramp_array.to(device)

    U = BB2_tensor[:15].clone().to(device)
    V = BB2_tensor[15:30].clone().to(device)
    T = BB2_tensor[31:45].clone().to(device)
    Q = BB2_tensor[45:60].clone().to(device)
    
    indpol=10
    # Example usage with the specified parameters
    array_length = 640
    indpol = 10
    ramp_length=10
    sigmoid_ramp_array = create_sigmoid_ramp_function(nlat, indpol)
    Ufit,Vfit=polefilt_lap2d_V2(U,V,ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=6)
    Tfit = polefilt_lap2d_V1(T,ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=5)
    Qfit = polefilt_lap2d_QV1(Q,ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=8)
    
    T2m = polefilt_lap2d_V1(BB2_tensor[61].clone().to(device),ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=5)
    U500,V500 =polefilt_lap2d_V2(BB2_tensor[63].clone().to(device),BB2_tensor[62].clone().to(device),ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=6)
    T500 = polefilt_lap2d_V1(BB2_tensor[64].clone().to(device),ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=5)
    Q500 = polefilt_lap2d_QV1(BB2_tensor[66].clone().to(device),ind_pol=10,device=device,sigmoid_ramp_array=sigmoid_ramp_array.clone().to(device),substeps=4)

    return Ufit,Vfit,Tfit,Qfit,T2m,U500,V500,T500,Q500



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
    print('...start filtering...')
    BB2_tensor=torch.tensor(BB2).clone().detach()
    diff_lap2d_filt(BB2_tensor)
    print('...done filtering...')





