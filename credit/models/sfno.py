import torch
import torch.nn as nn
from torch.cuda import amp
from torch_harmonics import RealSHT, InverseRealSHT
from functools import partial

import torch.fft
from torch.utils.checkpoint import checkpoint
import math
import torch
import warnings
from tltorch.factorized_tensors.core import FactorizedTensor
import tensorly as tl
tl.set_backend('pytorch')
warnings.filterwarnings("ignore")

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
cmap = 'twilight_shifted'


"""
Contains complex contractions wrapped into jit for harmonic layers
"""


@torch.jit.script
def contract_diagonal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixy->bkxy", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def contract_dhconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kix->bkxy", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def contract_blockdiag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixyz->bkxz", ac, bc)
    return torch.view_as_real(res)

# Helper routines for the non-linear FNOs (Attention-like)


@torch.jit.script
def compl_mul1d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixs,ior->srbox", a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return res


@torch.jit.script
def compl_mul1d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul1d_fwd(a, b) + c
    return res


@torch.jit.script
def compl_muladd1d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul1d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)

# Helper routines for FFT MLPs


@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ior->srboxy", a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return res


@torch.jit.script
def compl_mul2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = compl_mul2d_fwd(a, b) + c
    return res


@torch.jit.script
def compl_muladd2d_fwd_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul2d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def real_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.einsum("bixy,io->boxy", a, b)
    return out


@torch.jit.script
def real_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return compl_mul2d_fwd_c(a, b) + c


def _contract_dense(x, weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        weight_syms.insert(-1, einsum_symbols[order+1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == 'driscoll-healy':
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


def _contract_cp(x, cp_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)

    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym, out_sym+rank_sym]  # in, out

    factor_syms += [xs+rank_sym for xs in x_syms[2:]]  # x, y, ...

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        out_syms[-1] = einsum_symbols[order+2]
        factor_syms += [out_syms[-1] + rank_sym]
    elif operator_type == 'driscoll-healy':
        factor_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]]  # out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]  # x, y, ...

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        raise NotImplementedError(f"Operator type {operator_type} not implemented for Tucker")
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size

    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        weight_syms.insert(-1, einsum_symbols[order+1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == 'driscoll-healy':
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    rank_syms = list(einsum_symbols[order+2:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation='reconstructed', separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == 'reconstructed':
        return _contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == 'complexdense':
                return _contract_dense
            elif weight.name.lower() == 'complextucker':
                return _contract_tucker
            elif weight.name.lower() == 'complextt':
                return _contract_tt
            elif weight.name.lower() == 'complexcp':
                return _contract_cp
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


# complex activation functions


class ComplexCardioid(nn.Module):
    """
    Complex Cardioid activation function
    """
    def __init__(self):
        super(ComplexCardioid, self).__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = 0.5 * (1. + torch.cos(z.angle())) * z
        return out


class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """
    def __init__(self, negative_slope=0., mode="real", bias_shape=None, scale=1.):
        super(ComplexReLU, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(scale * torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)

        elif self.mode == "cardioid":
            out = 0.5 * (1. + torch.cos(z.angle())) * z

        # elif self.mode == "halfplane":
        #     # bias is an angle parameter in this case
        #     modified_angle = torch.angle(z) - self.bias
        #     condition = torch.logical_and( (0. <= modified_angle), (modified_angle < torch.pi/2.) )
        #     out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            raise NotImplementedError

        return out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        ll = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * ll - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution
    a: the minimum cutoff value
    b: the maximum cutoff value
    Examples:
    >>> w = torch.empty(3, 5)
    >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 output_bias=False,
                 drop_rate=0.,
                 checkpointing=False,
                 gain=1.0):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Fist dense layer
        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0., std=scale)
        if fc1.bias is not None:
            nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # output layer
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0., std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.:
            drop = nn.Dropout2d(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x)

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nlat,
                 nlon,
                 lmax=None,
                 mmax=None):
        super(RealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        y = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        y = torch.cat((y[..., :math.ceil(self.lmax/2), :self.mmax], y[..., -math.floor(self.lmax/2):, :self.mmax]), dim=-2)
        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nlat,
                 nlon,
                 lmax=None,
                 mmax=None):
        super(InverseRealFFT2, self).__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):
        return torch.fft.irfft2(x, dim=(-2, -1), s=(self.nlat, self.nlon), norm="ortho")


class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 in_channels,
                 out_channels,
                 gain=2.,
                 operator_type="driscoll-healy",
                 lr_scale_exponent=0,
                 bias=False):
        super(SpectralConvS2, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # remember factorization details
        self.operator_type = operator_type

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [out_channels, in_channels]

        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
            _contract = contract_diagonal
        elif self.operator_type == "block-diagonal":
            weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
            _contract = contract_blockdiag
        elif self.operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
            # from .contractions import contract_dhconv as _contract
            _contract = contract_dhconv
        else:
            raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

        # form weight tensors
        scale = math.sqrt(gain / in_channels) * torch.ones(self.modes_lat, 2)
        scale[0] *= math.sqrt(2)
        self.weight = nn.Parameter(scale * torch.view_as_real(torch.randn(*weight_shape, dtype=torch.complex64)))
        # self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))

        # get the right contraction function
        self._contract = _contract

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x

        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        x = torch.view_as_real(x)
        x = self._contract(x, self.weight)
        x = torch.view_as_complex(x)

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        return x, residual


class FactorizedSpectralConvS2(nn.Module):
    """
    Factorized version of SpectralConvS2. Uses tensorly-torch to keep the weights factorized
    """

    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 in_channels,
                 out_channels,
                 gain=2.,
                 operator_type="driscoll-healy",
                 rank=0.2,
                 factorization=None,
                 separable=False,
                 implementation="factorized",
                 decomposition_kwargs=dict(),
                 bias=False):
        super(SpectralConvS2, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [out_channels]

        if not self.separable:
            weight_shape += [in_channels]

        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
        elif self.operator_type == "block-diagonal":
            weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
        elif self.operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
        else:
            raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

        # form weight tensors
        self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, factorization=factorization,
                                           fixed_rank_modes=False, **decomposition_kwargs)

        # initialization of weights
        scale = math.sqrt(gain / in_channels)
        self.weight.normal_(0, scale)

        # get the right contraction function
        self._contract = get_contract_fun(self.weight, implementation=implementation, separable=separable)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x

        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        x = self._contract(x, self.weight, separable=self.separable, operator_type=self.operator_type)

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        return x, residual


class SpectralFilterLayer(nn.Module):
    """
    Fourier layer. Contains the convolution part of the FNO/SFNO
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        input_dim,
        output_dim,
        gain=2.,
        operator_type="diagonal",
        hidden_size_factor=2,
        factorization=None,
        separable=False,
        rank=1e-2,
        bias=True
    ):
        super(SpectralFilterLayer, self).__init__()

        if factorization is None:
            self.filter = SpectralConvS2(forward_transform,
                                         inverse_transform,
                                         input_dim,
                                         output_dim,
                                         gain=gain,
                                         operator_type=operator_type,
                                         bias=bias)

        elif factorization is not None:
            self.filter = FactorizedSpectralConvS2(forward_transform,
                                                   inverse_transform,
                                                   input_dim,
                                                   output_dim,
                                                   gain=gain,
                                                   operator_type=operator_type,
                                                   rank=rank,
                                                   factorization=factorization,
                                                   separable=separable,
                                                   bias=bias)

        else:
            raise NotImplementedError

    def forward(self, x):
        return self.filter(x)


class SphericalFourierNeuralOperatorBlock(nn.Module):
    """
    Helper module for a single SFNO/FNO block. Can use both FFTs and SHTs to represent either FNO or SFNO blocks.
    """
    def __init__(
            self,
            forward_transform,
            inverse_transform,
            input_dim,
            output_dim,
            operator_type="driscoll-healy",
            mlp_ratio=2.,
            drop_rate=0.,
            drop_path=0.,
            act_layer=nn.ReLU,
            norm_layer=nn.Identity,
            factorization=None,
            separable=False,
            rank=128,
            inner_skip="linear",
            outer_skip=None,
            use_mlp=True):
        super(SphericalFourierNeuralOperatorBlock, self).__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        # convolution layer
        self.filter = SpectralFilterLayer(forward_transform,
                                          inverse_transform,
                                          input_dim,
                                          output_dim,
                                          gain=gain_factor,
                                          operator_type=operator_type,
                                          hidden_size_factor=mlp_ratio,
                                          factorization=factorization,
                                          separable=separable,
                                          rank=rank,
                                          bias=True)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor/input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        self.act_layer = act_layer()

        # first normalisation layer
        self.norm0 = norm_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.

        if use_mlp:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(in_features=output_dim,
                           out_features=input_dim,
                           hidden_features=mlp_hidden_dim,
                           act_layer=act_layer,
                           drop_rate=drop_rate,
                           checkpointing=False,
                           gain=gain_factor)

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor/input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        # second normalisation layer
        self.norm1 = norm_layer()

    # def init_weights(self, scale):
    #     if hasattr(self, "inner_skip") and isinstance(self.inner_skip, nn.Conv2d):
    #         gain_factor = 1.
    #         scale = (gain_factor / embed_dim)**0.5
    #         nn.init.normal_(self.inner_skip.weight, mean=0., std=scale)
    #         self.filter.filter.init_weights(scale)
    #     else:
    #         gain_factor = 2.
    #         scale = (gain_factor / embed_dim)**0.5
    #         self.filter.filter.init_weights(scale)

    def forward(self, x):

        x, residual = self.filter(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SphericalFourierNeuralOperator module. Can use both FFTs and SHTs to represent either FNO or SFNO,
    both linear and non-linear variants.

    Parameters
    ----------
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    operator_type : str, optional
        Type of operator to use ('driscoll-healy', 'diagonal'), by default "driscoll-healy"
    img_shape : tuple, optional
        Shape of the input channels, by default (128, 256)
    scale_factor : int, optional
        Scale factor to use, by default 3
    in_chans : int, optional
        Number of input channels, by default 3
    out_chans : int, optional
        Number of output channels, by default 3
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 4
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    use_mlp : int, optional
        Whether to use MLPs in the SFNO blocks, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding (frequency cutoff) to apply, by default 1.0
    big_skip : bool, optional
        Whether to add a single large skip connection, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    rank : (int, Tuple[int]), optional
        If a factorization is used, which rank to use. Argument is passed to tensorly
    pos_embed : bool, optional
        Whether to use positional embedding, by default True

    Example:
    --------
    >>> model = SphericalFourierNeuralOperatorNet(
    ...         img_shape=(128, 256),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=4,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 128, 256)).shape
    torch.Size([1, 2, 128, 256])
    """

    def __init__(
            self,
            spectral_transform="sht",
            operator_type="driscoll-healy",
            img_size=(128, 256),
            grid="equiangular",
            scale_factor=3,
            in_chans=3,
            out_chans=3,
            embed_dim=256,
            num_layers=4,
            activation_function="relu",
            encoder_layers=1,
            use_mlp=True,
            mlp_ratio=2.,
            drop_rate=0.,
            drop_path_rate=0.,
            normalization_layer="none",
            hard_thresholding_fraction=1.0,
            use_complex_kernels=True,
            big_skip=False,
            factorization=None,
            separable=False,
            rank=128,
            levels=16,
            pos_embed=False):

        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.spectral_transform = spectral_transform
        self.operator_type = operator_type
        self.img_size = img_size
        self.grid = grid
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip
        self.factorization = factorization
        self.separable = separable,
        self.rank = rank

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # compute downsampled image size
        self.h = self.img_size[0] // scale_factor
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # # encoder
        # encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        # encoder = MLP(in_features = self.in_chans,
        #               out_features = self.embed_dim,
        #               hidden_features = encoder_hidden_dim,
        #               act_layer = self.activation_function,
        #               drop_rate = drop_rate,
        #               checkpointing = False)
        # self.encoder = encoder

        # construct an encoder with num_encoder_layers
        num_encoder_layers = 1
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.in_chans
        encoder_layers = []
        for _ in range(num_encoder_layers-1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2. / current_dim)
            nn.init.normal_(fc.weight, mean=0., std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1. / current_dim)
        nn.init.normal_(fc.weight, mean=0., std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # prepare the spectral transform
        if self.spectral_transform == "sht":

            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int(self.w//2 * self.hard_thresholding_fraction)
            modes_lat = modes_lon = min(modes_lat, modes_lon)

            self.trans_down = RealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
            self.itrans_up = InverseRealSHT(*self.img_size, lmax=modes_lat, mmax=modes_lon, grid=self.grid).float()
            self.trans = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
            self.itrans = InverseRealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

        elif self.spectral_transform == "fft":

            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

            self.trans_down = RealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up = InverseRealFFT2(*self.img_size, lmax=modes_lat, mmax=modes_lon).float()
            self.trans = RealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans = InverseRealFFT2(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()

        else:
            raise ValueError("Unknown spectral transform")

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers-1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "identity"

            if first_layer:
                norm_layer = norm_layer1
            elif last_layer:
                norm_layer = norm_layer0
            else:
                norm_layer = norm_layer1

            block = SphericalFourierNeuralOperatorBlock(forward_transform,
                                                        inverse_transform,
                                                        self.embed_dim,
                                                        self.embed_dim,
                                                        operator_type=self.operator_type,
                                                        mlp_ratio=mlp_ratio,
                                                        drop_rate=drop_rate,
                                                        drop_path=dpr[i],
                                                        act_layer=self.activation_function,
                                                        norm_layer=norm_layer,
                                                        inner_skip=inner_skip,
                                                        outer_skip=outer_skip,
                                                        use_mlp=use_mlp,
                                                        factorization=self.factorization,
                                                        separable=self.separable,
                                                        rank=self.rank)

            self.blocks.append(block)

        # # decoder
        # decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        # self.decoder = MLP(in_features = self.embed_dim + self.big_skip*self.in_chans,
        #                    out_features = self.out_chans,
        #                    hidden_features = decoder_hidden_dim,
        #                    act_layer = self.activation_function,
        #                    drop_rate = drop_rate,
        #                    checkpointing = False)

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip*self.in_chans
        decoder_layers = []
        for _ in range(num_decoder_layers-1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2. / current_dim)
            nn.init.normal_(fc.weight, mean=0., std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1. / current_dim)
        nn.init.normal_(fc.weight, mean=0., std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):

        if len(x.shape) == 5:
            x = x.squeeze(2)

        if self.big_skip:
            residual = x

        print("A", residual.shape)

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.forward_features(x)

        if self.big_skip:
            print("B", x.shape, residual.shape)
            x = torch.cat((x, residual), dim=1)

            print("C", x.shape)

        x = self.decoder(x)

        return x.unsqueeze(2)


if __name__ == "__main__":
    image_height = 192  # 640, 192
    image_width = 288  # 1280, 288
    levels = 16
    frames = 1
    channels = 4
    surface_channels = 7
    input_only_channels = 3
    frame_patch_size = 2

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    in_channels = channels * levels + surface_channels + input_only_channels

    input_tensor = torch.randn(1, in_channels, frames, image_height, image_width).to(device)

    model = SphericalFourierNeuralOperatorNet(
        spectral_transform='sht',
        operator_type='driscoll-healy',
        img_size=(image_height, image_width),
        scale_factor=1,
        grid="equiangular",
        big_skip=True,
        pos_embed="lat",
        use_mlp=True,
        in_chans=in_channels,
        out_chans=in_channels-input_only_channels,
        num_layers=8,
        embed_dim=256,
        normalization_layer="none"
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    y_pred = model(input_tensor.to("cuda"))
    print("Predicted shape:", y_pred.shape)

    # ACE settings
    # embed_dim 256
    # filter_type linear
    # num_layers 8
    # operator_type dhconv
    # scale_factor 1
    # spectral_layers 3
