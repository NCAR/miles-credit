"""WXFormer-Simple: a CrossFormer U-Net that sizes itself from the data.

``wxformer_simple`` keeps the WXFormer architecture (``credit/models/wxformer/crossformer.py``)
but drops every data-shaped constructor argument -- ``levels``, ``channels``, ``surface_channels``,
``input_only_channels``, ``output_only_channels``, ``frames``, ``image_height``/``image_width`` and
``padding_conf`` (see https://github.com/NCAR/miles-credit/issues/331). The config only names the
architecture; the data-dependent layers are built by :meth:`WXFormerSimple.materialize` the first
time the model sees a batch.

Weights must exist before the trainer compiles, wraps (DDP/FSDP2/TP/domain), builds the optimizer and
snapshots the EMA, so ``train_gen2`` materializes the model on one real preprocessed sample right
after ``load_model`` (under the shared init seed). A bare ``forward`` on an unbuilt model materializes
too, which is handy in notebooks, but then the output width must be given as ``output_channels``.

The resolved hyperparameters -- defaults and measured values included -- are written to
``save_loc/model_hparams.yml``. Reloading for rollout rebuilds from that file, not from the config,
so a later change to the defaults here cannot silently change a trained model.

The building blocks below are copied from ``crossformer.py`` rather than imported so later edits there
cannot change this model. ``CrossEmbedConvBranch`` is the one exception: domain-parallel conversion
(``credit/domain_parallel/convert.py``) recognizes that exact class.
"""

import copy
import logging
import math
import os
import warnings

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn

from credit.boundary_padding import TensorPadding
from credit.models.base_model import BaseModel
from credit.models.wxformer.crossformer import CrossEmbedConvBranch

logger = logging.getLogger(__name__)

HPARAMS_FILENAME = "model_hparams.yml"

# Config keys the other WXFormers need but this one measures from the data.
DATA_DERIVED_KEYS = (
    "frames",
    "image_height",
    "image_width",
    "levels",
    "channels",
    "surface_channels",
    "input_only_channels",
    "output_only_channels",
    "padding_conf",
    "patch_height",
    "patch_width",
    "interp",
    "upsample_with_ps",
)

_PADDING_MODES = ("earth", "mirror")

# --------------------------------------------------------------------------------------- #
# Building blocks copied from credit/models/wxformer/crossformer.py (legacy-checkpoint hooks
# dropped: no pre-fix checkpoints of this model exist).
# --------------------------------------------------------------------------------------- #


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if module.weight.numel() > 0:
                nn.utils.spectral_norm(module)


def icnr_init_(weight, scale, init=nn.init.kaiming_normal_):
    """ICNR init for a sub-pixel conv feeding nn.PixelShuffle (Aitken et al. 2017).

    Initializes the conv weight so that, immediately after PixelShuffle(scale), the
    output equals a nearest-neighbor upsample of a single initialized sub-kernel.
    All scale**2 sub-pixel channels start identical, which removes the checkerboard
    grid pattern present at initialization with default init.

    Args:
        weight: conv weight of shape (out_ch * scale**2, in_ch, kh, kw).
        scale: PixelShuffle upscale factor.
        init: in-place initializer applied to the sub-kernel.
    """
    out_ch = weight.shape[0] // (scale**2)
    sub = torch.zeros(out_ch, *weight.shape[1:], device=weight.device, dtype=weight.dtype)
    init(sub)
    # PixelShuffle consumes channels in contiguous (r**2) blocks per output channel,
    # so each sub-kernel must be repeated contiguously along the channel dim.
    sub = sub.repeat_interleave(scale**2, dim=0)
    with torch.no_grad():
        weight.copy_(sub)


class UpBlockPS(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups, scale=2, num_residuals=2, fsdp2_shard=True):
        super().__init__()
        # FSDP2 per-block sharding / activation-checkpointing opt-in
        self._fsdp2_shard = fsdp2_shard
        # sub-pixel conv at low res (ICNR init removes checkerboard at initialization)
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, stride=1, padding=1)
        icnr_init_(self.conv.weight, scale)
        nn.init.zeros_(self.conv.bias)
        self.ps = nn.PixelShuffle(scale)
        # sharpening branch (identity init)
        self.sharp = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.xavier_normal_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        # residual stack
        blk = []
        for _ in range(num_residuals):
            blk += [nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(num_groups, out_ch), nn.SiLU()]
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.ps(self.conv(x))  # upsample+conv at low res
        x = x + self.sharp(x)  # sharpen residual
        sc = x
        x = self.b(x)
        return x + sc


# cross embed layer


def crossembed_pad_total(kernel: int, stride: int) -> int:
    """Total zero-padding a CrossEmbedLayer conv branch applies per spatial dim."""
    return kernel - stride


def crossembed_out_size(size: int, kernel: int, stride: int) -> int:
    """Spatial output size of one CrossEmbedLayer conv branch.

    Standard conv arithmetic with the layer's asymmetric zero-padding; with
    pad_total = kernel - stride this is kernel-independent (floor(size/stride)),
    which is why the cat() across kernel sizes works. Shared with the
    `credit begin` wizard's grid-spec search so the two cannot drift apart.
    """
    return (size + crossembed_pad_total(kernel, stride) - kernel) // stride + 1


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=2):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            # Symmetric padding = (kernel - stride) // 2 only gives a true "same"
            # shape (out = ceil(in / stride)) when (kernel - stride) is even. For
            # an even kernel at stride=1 that's odd, so // 2 rounds down and drops
            # one cell (e.g. kernel=4, stride=1: padding=1, 480 in -> 479 out). All
            # kernel branches lose the same cell, so the cat() still succeeds --
            # silently, at the wrong size. Use explicit asymmetric zero-padding
            # instead so every (kernel, stride) combination gets the exact "same"
            # shape; this is a no-op vs. the old formula whenever (kernel - stride)
            # is even (the previously-working case, e.g. every default stride=2 config).
            pad_total = crossembed_pad_total(kernel, stride)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            self.convs.append(
                CrossEmbedConvBranch(
                    nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
                    nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=0),
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


# dynamic positional bias


class DynamicPositionBias(nn.Module):
    def __init__(self, dim):
        super(DynamicPositionBias, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            Rearrange("... () -> ..."),
        )

    def forward(self, x):
        return self.layers(x)


# transformer classes


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, tp_col="layers.1", tp_row="layers.4"):
        super(FeedForward, self).__init__()
        # Tensor-parallel opt-in: dotted paths to the column-parallel layer
        # (Conv2d dim → dim*mult, output channels sharded) and the row-parallel
        # layer (Conv2d dim*mult → dim, input channels sharded + all_reduce).
        self._tp_col = tp_col
        self._tp_row = tp_row
        self.layers = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    """
    Attention module for the CrossFormer model.

    Tensor parallelism opt-in: ``to_qkv`` is column-parallel (output sharded),
    ``to_out`` is row-parallel (input sharded, all_reduce).

    This module performs either short-range or long-range attention on the input tensor.
    It uses a dynamic positional bias to incorporate relative positional information.

    Args:
        dim (int): Input dimension.
        attn_type (str): Type of attention, either "short" or "long".
        window_size (int): Size of the attention window.
        dim_head (int, optional): Dimension of each attention head. Defaults to 32.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    @staticmethod
    def _tp_constraints(instance, tp_size):
        if instance.heads % tp_size != 0:
            raise ValueError(
                f"Attention TP: heads={instance.heads} not divisible by tp_size={tp_size}. "
                f"Choose a TP degree that divides {instance.heads}, or increase dim_head."
            )

    def __init__(self, dim, attn_type, window_size, dim_head=32, dropout=0.0, tp_col="to_qkv", tp_row="to_out"):
        super().__init__()
        # Tensor-parallel opt-in: to_qkv is column-parallel (output channels
        # sharded), to_out is row-parallel (input sharded + all_reduce).
        self._tp_col = tp_col
        self._tp_row = tp_row
        assert attn_type in {
            "short",
            "long",
        }, "attention type must be one of local or distant"
        if dim < dim_head:
            raise ValueError(
                f"Attention: dim={dim} is smaller than dim_head={dim_head}; "
                f"set dim_head <= {dim} or increase the smallest dim in the model."
            )
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        # positions

        self.dpb = DynamicPositionBias(dim // 4)

        # calculate and store indices for retrieving bias

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        *_, height, width, heads, wsz, device = (
            *x.shape,
            self.heads,
            self.window_size,
            x.device,
        )

        # prenorm

        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == "short":
            x = rearrange(x, "b d (h s1) (w s2) -> (b h w) d s1 s2", s1=wsz, s2=wsz)
        elif self.attn_type == "long":
            x = rearrange(x, "b d (l1 h) (l2 w) -> (b h w) d l1 l2", l1=wsz, l2=wsz)

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=heads), (q, k, v))
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add dynamic positional bias

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rearrange(rel_pos, "c i j -> (i j) c")
        rel_pos = rel_pos.to(x.dtype)
        biases = self.dpb(rel_pos)
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=wsz, y=wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == "short":
            out = rearrange(
                out,
                "(b h w) d s1 s2 -> b d (h s1) (w s2)",
                h=height // wsz,
                w=width // wsz,
            )
        elif self.attn_type == "long":
            out = rearrange(
                out,
                "(b h w) d l1 l2 -> b d (l1 h) (l2 w)",
                h=height // wsz,
                w=width // wsz,
            )

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth=4,
        dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
        fsdp2_shard=True,
    ):
        super().__init__()
        # FSDP2 per-block sharding / activation-checkpointing opt-in
        self._fsdp2_shard = fsdp2_shard
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            attn_type="short",
                            window_size=local_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                        Attention(
                            dim,
                            attn_type="long",
                            window_size=global_window_size,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


# --------------------------------------------------------------------------------------- #
# Geometry: the padding every grid needs to survive the encoder.
# --------------------------------------------------------------------------------------- #


def _per_stage(value, name):
    """Broadcast a scalar to the 4 encoder stages and check list lengths."""
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise ValueError(
                f"WXFormerSimple: {name} has {len(value)} entries; it needs exactly 4 (one per stage), "
                "or a single scalar to use for every stage."
            )
        return list(value)
    return [value] * 4


def required_divisor(cross_embed_strides, local_window_size, global_window_size) -> int:
    """Smallest number the padded height and width must both be multiples of.

    Each CrossEmbedLayer divides the grid exactly by its stride (its padding is
    ``kernel - stride``), so stage ``k`` is ``n / prod(strides[:k+1])`` cells wide.
    That size must tile into both attention windows at stage ``k``, and the decoder's
    skip connections need every stage to be an exact division. The answer is
    ``lcm_k(prod(strides[:k+1]) * lcm(local[k], global[k]))``.
    """
    strides = _per_stage(cross_embed_strides, "cross_embed_strides")
    local_ws = _per_stage(local_window_size, "local_window_size")
    global_ws = _per_stage(global_window_size, "global_window_size")
    divisor, scale = 1, 1
    for stride, local_w, global_w in zip(strides, local_ws, global_ws):
        scale *= int(stride)
        divisor = math.lcm(divisor, scale * math.lcm(int(local_w), int(global_w)))
    return divisor


def _normalize_padding(padding) -> dict:
    """Fill in the padding policy: ``{mode, min_pad_lat, min_pad_lon}``."""
    if padding is None:
        padding = {}
    elif padding is False:
        padding = {"mode": "none"}
    elif isinstance(padding, str):
        padding = {"mode": padding}
    elif not isinstance(padding, dict):
        raise TypeError(f"WXFormerSimple: padding must be a dict, a mode string, or false; got {padding!r}")
    unknown = set(padding) - {"mode", "min_pad_lat", "min_pad_lon"}
    if unknown:
        raise ValueError(
            f"WXFormerSimple: unknown padding key(s) {sorted(unknown)}; "
            "accepted keys are mode, min_pad_lat, min_pad_lon (pad sizes themselves are computed)."
        )
    policy = {
        "mode": padding.get("mode", "earth"),
        "min_pad_lat": int(padding.get("min_pad_lat", 0)),
        "min_pad_lon": int(padding.get("min_pad_lon", 0)),
    }
    if policy["mode"] not in (*_PADDING_MODES, "none"):
        raise ValueError(
            f"WXFormerSimple: padding.mode must be one of {(*_PADDING_MODES, 'none')}, got {policy['mode']!r}"
        )
    if policy["min_pad_lat"] < 0 or policy["min_pad_lon"] < 0:
        raise ValueError("WXFormerSimple: padding.min_pad_lat / min_pad_lon must be >= 0")
    return policy


def resolve_padding(height: int, width: int, divisor: int, padding=None) -> tuple[list[int], list[int]]:
    """Smallest padding that makes ``height`` and ``width`` multiples of ``divisor``.

    Each side gets at least ``min_pad_lat`` / ``min_pad_lon`` cells; any extra is split
    as evenly as possible, with the odd cell on the bottom/right side.

    Args:
        height: grid cells south-north.
        width: grid cells west-east.
        divisor: from :func:`required_divisor`.
        padding: the padding policy (see :class:`WXFormerSimple`).

    Returns:
        ``(pad_lat, pad_lon)`` as ``[top, bottom]`` and ``[left, right]`` lists.

    Raises:
        ValueError: if ``mode: none`` but the grid does not divide, or the latitude pad
            is too large for the pole reflection.
    """
    policy = _normalize_padding(padding)

    def _solve(size, min_side):
        total = 2 * min_side
        total += -(size + total) % divisor
        return [total // 2, total - total // 2]

    if policy["mode"] == "none":
        if height % divisor or width % divisor:
            raise ValueError(
                f"WXFormerSimple: padding.mode is 'none' but the {height}x{width} grid is not a multiple of "
                f"{divisor} (required by the strides and window sizes). Use mode 'earth' or 'mirror' to pad "
                "automatically, or pick windows/strides that divide the grid."
            )
        return [0, 0], [0, 0]

    pad_lat = _solve(height, policy["min_pad_lat"])
    pad_lon = _solve(width, policy["min_pad_lon"])
    if max(pad_lat) >= height:
        raise ValueError(
            f"WXFormerSimple: the {height}x{width} grid needs pad_lat={pad_lat} to reach a multiple of {divisor}, "
            f"but the pole reflection can pad at most {height - 1} rows per side. Use smaller window sizes or "
            "strides (a smaller divisor), or a lower min_pad_lat."
        )
    return pad_lat, pad_lon


def _to_builtin(value):
    """Tuples to lists (recursively) so hyperparameters dump to plain YAML."""
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------------------- #


class WXFormerSimple(BaseModel):
    """WXFormer whose data-shaped layers are built from the first batch it sees.

    Args:
        dim (tuple): channels of each encoder stage. Must be a halving pyramid
            (``dim[k] == dim[-1] // 2**(3-k)``) because the decoder concatenates skip
            connections of those widths.
        depth (tuple): attention blocks per encoder stage.
        dim_head (int): channels per attention head.
        global_window_size (tuple): long-range attention window per stage.
        local_window_size (int or tuple): short-range attention window (per stage if a tuple).
        cross_embed_kernel_sizes (tuple): kernel sizes of each stage's CrossEmbedLayer.
        cross_embed_strides (tuple): downsampling stride of each stage.
        attn_dropout (float): attention dropout.
        ff_dropout (float): feed-forward dropout.
        use_spectral_norm (bool): spectral-normalize every conv and linear layer.
        padding (dict, str or bool): padding policy, not pad sizes. ``mode`` is ``earth``
            (default: pole reflection + circular longitude), ``mirror`` or ``none``;
            ``min_pad_lat`` / ``min_pad_lon`` set a minimum pad per side. The sizes are
            computed so the padded grid divides through every stage and window.
        output_channels (int, optional): output channels per output frame. Only needed
            when the first call is a bare ``forward`` (no target to measure); ``train_gen2``
            measures it from the target.
        output_frames (int): output time steps; measured from the target when available.
        **kwargs: data-shaped keys from other WXFormer configs (``levels``, ``channels``,
            ``image_height``, ...) are ignored with a warning, but checked against the
            measured shapes when the model is built.
    """

    def __init__(
        self,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size: tuple = (8, 4, 2, 1),
        local_window_size: int = 4,
        cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides: tuple = (2, 2, 2, 2),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_spectral_norm: bool = True,
        padding=None,
        output_channels: int = None,
        output_frames: int = 1,
        **kwargs,
    ):
        super().__init__()

        dim = [int(d) for d in _per_stage(dim, "dim")]
        kernels = cross_embed_kernel_sizes
        if isinstance(kernels, (list, tuple)) and kernels and not isinstance(kernels[0], (list, tuple)):
            kernels = [kernels] * 4  # one kernel list shared by every stage
        self.arch = {
            "dim": dim,
            "depth": [int(d) for d in _per_stage(depth, "depth")],
            "dim_head": int(dim_head),
            "global_window_size": [int(w) for w in _per_stage(global_window_size, "global_window_size")],
            "local_window_size": [int(w) for w in _per_stage(local_window_size, "local_window_size")],
            "cross_embed_kernel_sizes": [list(k) for k in _per_stage(kernels, "cross_embed_kernel_sizes")],
            "cross_embed_strides": [int(s) for s in _per_stage(cross_embed_strides, "cross_embed_strides")],
            "attn_dropout": float(attn_dropout),
            "ff_dropout": float(ff_dropout),
            "use_spectral_norm": bool(use_spectral_norm),
            "padding": _normalize_padding(padding),
        }
        self._check_arch()
        self.divisor = required_divisor(
            self.arch["cross_embed_strides"], self.arch["local_window_size"], self.arch["global_window_size"]
        )

        self.output_channels_hint = output_channels
        self.output_frames_hint = int(output_frames)
        self.data_shape = None  # filled in by materialize()
        self._expected = {}

        kwargs.pop("type", None)
        post_conf = kwargs.pop("post_conf", None)
        if post_conf and post_conf.get("activate", False):
            warnings.warn("WXFormerSimple does not run Gen 1 postblocks (model.post_conf); use Gen 2 postblocks.")
        legacy = {k: kwargs.pop(k) for k in list(kwargs) if k in DATA_DERIVED_KEYS}
        if legacy:
            logger.warning(
                "WXFormerSimple measures %s from the data; the configured values are ignored (but checked "
                "against the data when the model is built). They can be removed from the model block.",
                ", ".join(sorted(legacy)),
            )
            self._expected = legacy
        if kwargs:
            logger.warning("WXFormerSimple: ignoring unrecognized model option(s) %s", ", ".join(sorted(kwargs)))

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _check_arch(self):
        dim, dim_head = self.arch["dim"], self.arch["dim_head"]
        if dim_head > min(dim):
            raise ValueError(
                f"WXFormerSimple: dim_head={dim_head} is larger than the smallest stage dim {min(dim)}; "
                f"set dim_head <= {min(dim)}."
            )
        pyramid = [dim[-1] // 8, dim[-1] // 4, dim[-1] // 2, dim[-1]]
        if dim != pyramid:
            raise ValueError(
                f"WXFormerSimple: dim={dim} must halve stage by stage (the decoder concatenates skip "
                f"connections of those widths); for dim[-1]={dim[-1]} use dim={pyramid}."
            )
        if pyramid[0] == 0 or pyramid[0] % dim[0]:
            raise ValueError(f"WXFormerSimple: dim[-1]={dim[-1]} is too small; use at least 8 * dim[0].")

    @property
    def needs_materialize(self) -> bool:
        """True until the data-dependent layers have been built."""
        return self.data_shape is None

    @property
    def resolved_hparams(self) -> dict:
        """Every hyperparameter the model was built with, defaults and measured values included."""
        hparams = {"type": "wxformer_simple", **copy.deepcopy(self.arch)}
        if self.data_shape is not None:
            hparams.update(copy.deepcopy(self.data_shape))
        return _to_builtin(hparams)

    def materialize(self, x, y=None, output_channels=None):
        """Measure the data shapes and build every layer.

        Args:
            x: model input, ``(B, C, T, H, W)``; a 4D ``(B, C, H, W)`` input means ``T = 1``.
            y: optional target, ``(B, C_out, T_out, H, W)`` or ``(B, C_out, H, W)``. Gives
                the output width; otherwise ``output_channels`` (or the constructor's) is used.
            output_channels: output channels per output frame, if there is no ``y``.

        Returns:
            self
        """
        if not self.needs_materialize:
            raise RuntimeError("WXFormerSimple.materialize: the model is already built.")
        if x.ndim not in (4, 5):
            raise ValueError(f"WXFormerSimple: expected a (B, C, T, H, W) input, got shape {tuple(x.shape)}")
        channels, frames = x.shape[1], (x.shape[2] if x.ndim == 5 else 1)
        height, width = x.shape[-2:]

        output_frames = self.output_frames_hint
        if y is not None:
            if tuple(y.shape[-2:]) != (height, width):
                raise ValueError(
                    f"WXFormerSimple: input grid {height}x{width} and target grid "
                    f"{tuple(y.shape[-2:])} differ; this model predicts on its input grid."
                )
            output_channels = y.shape[1]
            output_frames = y.shape[2] if y.ndim == 5 else 1
        output_channels = output_channels or self.output_channels_hint
        if not output_channels:
            raise ValueError(
                "WXFormerSimple cannot tell how many channels to predict from the input alone. Call "
                "model.materialize(x, y) with a target batch, or set model.output_channels."
            )
        self._build(
            input_channels=int(channels),
            frames=int(frames),
            output_channels=int(output_channels),
            output_frames=int(output_frames),
            image_height=int(height),
            image_width=int(width),
        )
        return self.to(x.device)

    def materialize_from_hparams(self, hparams: dict):
        """Build the layers from a saved ``model_hparams.yml`` (no data needed)."""
        if not self.needs_materialize:
            raise RuntimeError("WXFormerSimple.materialize_from_hparams: the model is already built.")
        self._build(
            input_channels=int(hparams["input_channels"]),
            frames=int(hparams["frames"]),
            output_channels=int(hparams["output_channels"]),
            output_frames=int(hparams["output_frames"]),
            image_height=int(hparams["image_height"]),
            image_width=int(hparams["image_width"]),
        )
        for key in ("pad_lat", "pad_lon"):
            if key in hparams and list(hparams[key]) != self.data_shape[key]:
                raise ValueError(
                    f"WXFormerSimple: saved {key}={hparams[key]} differs from the {key}={self.data_shape[key]} "
                    "this version computes; the saved weights were trained on a different padded grid."
                )
        return self

    def _check_expected(self, measured: dict):
        """Compare configured (legacy) data keys against what the data actually has."""
        exp = self._expected
        mismatches = []
        for key in ("frames", "image_height", "image_width"):
            if key in exp and exp[key] != measured[key]:
                mismatches.append(f"{key}: configured {exp[key]}, data has {measured[key]}")
        per_level = ("channels", "levels", "surface_channels")
        if all(k in exp for k in per_level):
            prognostic = exp["channels"] * exp["levels"] + exp["surface_channels"]
            if "input_only_channels" in exp:
                n_in = prognostic + exp["input_only_channels"]
                if n_in != measured["input_channels"]:
                    mismatches.append(f"input channels: configured {n_in}, data has {measured['input_channels']}")
            if "output_only_channels" in exp:
                n_out = prognostic + exp["output_only_channels"]
                if n_out != measured["output_channels"]:
                    mismatches.append(f"output channels: configured {n_out}, data has {measured['output_channels']}")
        if mismatches:
            raise ValueError(
                "WXFormerSimple: the model block disagrees with the data -- "
                + "; ".join(mismatches)
                + ". Delete these keys from the model block; this model measures them."
            )

    def _build(self, input_channels, frames, output_channels, output_frames, image_height, image_width):
        pad_lat, pad_lon = resolve_padding(image_height, image_width, self.divisor, self.arch["padding"])
        measured = {
            "input_channels": input_channels,
            "frames": frames,
            "output_channels": output_channels,
            "output_frames": output_frames,
            "image_height": image_height,
            "image_width": image_width,
            "pad_lat": pad_lat,
            "pad_lon": pad_lon,
        }
        self._check_expected(measured)

        arch = self.arch
        dim = arch["dim"]
        last_dim = dim[-1]

        # Attributes shared with CrossFormer: the gen2 trainer's domain-parallel path
        # reads padding_opt / image_height / image_width / use_padding / use_interp.
        self.image_height, self.image_width = image_height, image_width
        self.frames, self.output_frames = frames, output_frames
        self.input_channels = input_channels * frames
        self.base_output_channels = output_channels
        self.output_channels = output_channels * output_frames
        self.use_padding = any(pad_lat) or any(pad_lon)
        self.padding_opt = TensorPadding(mode=arch["padding"]["mode"], pad_lat=pad_lat, pad_lon=pad_lon)
        self.use_interp = True  # only acts when stage-0 stride != 2 leaves the decoder short

        dims = [self.input_channels, *dim]
        self.layers = nn.ModuleList([])
        for (dim_in, dim_out), num_layers, global_wsize, local_wsize, kernel_sizes, stride in zip(
            zip(dims[:-1], dims[1:]),
            arch["depth"],
            arch["global_window_size"],
            arch["local_window_size"],
            arch["cross_embed_kernel_sizes"],
            arch["cross_embed_strides"],
        ):
            self.layers.append(
                nn.ModuleList(
                    [
                        CrossEmbedLayer(dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernel_sizes, stride=stride),
                        Transformer(
                            dim=dim_out,
                            local_window_size=local_wsize,
                            global_window_size=global_wsize,
                            depth=num_layers,
                            dim_head=arch["dim_head"],
                            attn_dropout=arch["attn_dropout"],
                            ff_dropout=arch["ff_dropout"],
                        ),
                    ]
                )
            )

        # Decoder: sub-pixel conv + PixelShuffle, ICNR-initialized with zero bias (see the
        # checkerboard notes in crossformer.py).
        self.up_block1 = UpBlockPS(1 * last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlockPS(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlockPS(2 * (last_dim // 4), last_dim // 8, dim[0])
        scale = 2
        ps_conv = nn.Conv2d(2 * (last_dim // 8), self.output_channels * (scale**2), kernel_size=3, stride=1, padding=1)
        icnr_init_(ps_conv.weight, scale)
        nn.init.zeros_(ps_conv.bias)
        self.up_block4 = nn.Sequential(
            ps_conv,
            nn.PixelShuffle(upscale_factor=scale),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
        )

        if arch["use_spectral_norm"]:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        self.data_shape = measured
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "WXFormerSimple built from the data: %d input channels x %d frame(s) -> %d output channels x %d "
            "frame(s) on a %dx%d grid padded by lat %s / lon %s to multiples of %d; %s parameters.",
            input_channels,
            frames,
            output_channels,
            output_frames,
            image_height,
            image_width,
            pad_lat,
            pad_lon,
            self.divisor,
            f"{n_params:,}",
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x):
        if self.needs_materialize:
            self.materialize(x)

        if x.ndim == 4:
            x = x.unsqueeze(2)
        if self.use_padding:
            x = self.padding_opt.pad(x)

        b, c, t, h, w = x.shape
        if c * t != self.input_channels:
            raise ValueError(
                f"WXFormerSimple was built for {self.input_channels} input channels x frames but got "
                f"{c} x {t} = {c * t}; the model is tied to the data layout it was built from."
            )
        x = x.reshape(b, c * t, h, w)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.up_block4(x)

        # The decoder upsamples 2x past stage 0, so with a stage-0 stride other than 2
        # it lands off the padded grid; resize back before cutting the padding off.
        if self.use_interp and tuple(x.shape[-2:]) != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear")

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        b, _, h, w = x.shape
        return x.view(b, self.base_output_channels, self.output_frames, h, w)

    # ------------------------------------------------------------------ #
    # Saving and loading
    # ------------------------------------------------------------------ #

    def save_hparams(self, save_loc: str) -> str:
        """Write :attr:`resolved_hparams` to ``save_loc/model_hparams.yml``.

        An existing file is kept (it describes the weights already in ``save_loc``);
        a warning is logged if it disagrees with this model.

        Returns:
            The path of the hyperparameter file.
        """
        if self.needs_materialize:
            raise RuntimeError("WXFormerSimple.save_hparams: build the model (materialize) first.")
        path = os.path.join(os.path.expandvars(save_loc), HPARAMS_FILENAME)
        hparams = self.resolved_hparams
        if os.path.isfile(path):
            existing = load_hparams(os.path.dirname(path))
            if existing != hparams:
                logger.warning(
                    "%s already exists and differs from this model; keeping the existing file. Differences: %s",
                    path,
                    {
                        k: (existing.get(k), hparams.get(k))
                        for k in set(existing) | set(hparams)
                        if existing.get(k) != hparams.get(k)
                    },
                )
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            yaml.safe_dump(hparams, f, sort_keys=False)
        os.replace(tmp, path)
        logger.info("WXFormerSimple hyperparameters saved to %s", path)
        return path

    @classmethod
    def from_hparams(cls, hparams: dict) -> "WXFormerSimple":
        """Rebuild a model (layers included, weights random) from saved hyperparameters."""
        arch_keys = (
            "dim",
            "depth",
            "dim_head",
            "global_window_size",
            "local_window_size",
            "cross_embed_kernel_sizes",
            "cross_embed_strides",
            "attn_dropout",
            "ff_dropout",
            "use_spectral_norm",
            "padding",
        )
        model = cls(**{k: hparams[k] for k in arch_keys if k in hparams})
        return model.materialize_from_hparams(hparams)

    @classmethod
    def _build_for_loading(cls, conf: dict) -> "WXFormerSimple":
        """Rebuild from ``save_loc/model_hparams.yml`` so reloads ignore later config edits."""
        hparams = load_hparams(conf["save_loc"])
        configured = {k: v for k, v in (conf.get("model") or {}).items() if k != "type"}
        saved_arch = cls.from_hparams(hparams)
        current_arch = cls(**configured).arch
        changed = {k: (hparams[k], v) for k, v in _to_builtin(current_arch).items() if hparams.get(k) != v}
        if changed:
            logger.warning(
                "WXFormerSimple: the config's model block differs from the saved %s for %s; "
                "using the saved values (they match the checkpoint).",
                HPARAMS_FILENAME,
                ", ".join(f"{k} (saved {s}, config {c})" for k, (s, c) in changed.items()),
            )
        return saved_arch


def load_hparams(save_loc: str) -> dict:
    """Read ``save_loc/model_hparams.yml``.

    Raises:
        FileNotFoundError: with a hint, if the file is missing.
    """
    path = os.path.join(os.path.expandvars(save_loc), HPARAMS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"WXFormerSimple needs {path} to rebuild a trained model; it is written by credit train next to "
            "the checkpoint. Copy it along with the checkpoint if you moved save_loc."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def materialize_from_data(model, conf: dict, dataset) -> None:
    """Build a model's data-dependent layers from one real, preprocessed sample.

    Runs the first sample of ``dataset`` through the same preblock chain the gen2
    trainer uses at ``t = 1`` (``ic_only``, rollout renames, ``per_step``) and hands
    the resulting ``x`` and ``y`` to ``model.materialize``.

    Args:
        model: an unbuilt model with a ``materialize(x, y)`` method.
        conf: the full config.
        dataset: the gen2 training dataset (indexed by ``(timestamp, step)``).
    """
    from torch.utils.data import default_collate

    from credit.preblock import apply_preblocks, build_preblocks
    from credit.trainers.rollout_utils import apply_rollout_renames

    sample = dataset[(dataset.datetimes[0], 0)]
    batch = default_collate([sample])
    ic_preblocks = build_preblocks(conf, phase="ic_only")
    step_preblocks = build_preblocks(conf, phase="per_step")
    out = apply_preblocks(ic_preblocks, batch)
    out = apply_rollout_renames(out, step_preblocks)
    out = apply_preblocks(step_preblocks, out)
    if "x" not in out:
        raise ValueError("materialize_from_data: the preblock chain did not produce 'x'; it must end with concat.")
    with torch.no_grad():
        model.materialize(out["x"], out.get("y"))
