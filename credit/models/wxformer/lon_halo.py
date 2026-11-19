"""Per-layer longitude halo refresh for WXFormer on a global lat-lon grid.

WXFormer pads its input once (``TensorPadding``, ``mode: earth``: ``pad_lon`` columns copied circularly onto each
side) and then runs every convolution with zero padding at the outer edge of the padded grid. Those zeros spread
inward through the stride-2 cross-embed layers, the window-aligned attention and the decoder, roughly 150-160
full-resolution columns for the four-stage 0.25 deg model, so a narrower halo leaves a seam at the first/last column
(0 deg E on an ERA5 grid). In autoregressive rollouts that seam grows: with ``pad_lon: [16, 16]`` the jump across
0 deg E reached 10-30 times the typical neighbour-to-neighbour jump within a few days.

:func:`install_lon_halo_refresh` makes the longitude halo exact at every layer instead:

* each ``CrossEmbedConvBranch`` (``ZeroPad2d`` + strided ``Conv2d``) and each 3 x 3, stride-1 ``Conv2d`` with
  ``padding=1`` runs on a periodic extension of the real (interior) columns: the halo and the conv's own padding come
  from the other side of the globe. Latitude keeps its zero padding;
* each ``Attention`` block refreshes its input's halo from the interior first, so the windows see exact copies.

The halo at a layer is ``pad_lon * W / padded_width`` columns for a tensor of width ``W``; it must be a whole number
at every stage, so ``pad_lon`` and the image width must both divide by ``2 ** n_stages``. Pointwise layers need
nothing. Not changed: the long-range (dilated) attention and GroupNorm statistics still span the whole padded grid,
and latitude, whose rows do not halve evenly through the stages, keeps the pole padding.

Implemented with forward pre-hooks; parameters and ``state_dict`` keys are unchanged, so trained checkpoints load as
before and the refresh can be switched on for inference only. Not compatible with domain parallelism, which reads
each conv's ``padding`` to size its own halos (``convert_to_domain_parallel`` refuses a refreshed model).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Set on a model by :func:`install_lon_halo_refresh` (checked by ``convert_to_domain_parallel``).
REFRESH_ATTR = "_lon_halo_refresh"


def _halo(width: int, pad: int, padded_width: int) -> int:
    h = pad * width / padded_width
    if h != int(h) or 2 * int(h) >= width:
        raise ValueError(
            f"lon halo refresh: a tensor of width {width} would need a halo of {h} columns "
            f"(pad_lon {pad}, padded width {padded_width}); pad_lon and the image width must divide by 2**n_stages"
        )
    return int(h)


def periodic_extend(x: torch.Tensor, halo: int, left: int, right: int) -> torch.Tensor:
    """Interior columns of ``x`` (halo ``halo`` on each side) extended periodically by ``halo + left`` / ``halo + right``."""
    xi = x[..., halo : x.shape[-1] - halo] if halo else x
    a, b = halo + left, halo + right
    parts = ([xi[..., xi.shape[-1] - a :]] if a else []) + [xi] + ([xi[..., :b]] if b else [])
    return torch.cat(parts, dim=-1)


def install_lon_halo_refresh(model: nn.Module, pad_lon: int, padded_width: int, roots=None) -> nn.Module:
    """Install the per-layer longitude halo refresh on ``model`` in place (see the module docstring).

    Args:
        model: a WXFormer ``CrossFormer`` (or any module built from the same blocks).
        pad_lon: longitude halo on each side of the padded input (symmetric).
        padded_width: width of the padded input, ``image_width + 2 * pad_lon``.
        roots: submodules to patch (default ``[model]``). ``CrossFormer`` passes its encoder and decoder only, so
            layers that run on the unpadded grid (e.g. a postblock) are left alone.

    Returns:
        ``model``, with the hook handles in ``model._lon_halo_handles``.
    """
    from credit.models.wxformer.crossformer import Attention, CrossEmbedConvBranch

    if getattr(model, REFRESH_ATTR, False):
        return model
    handles = []
    counts = {"cross_embed": 0, "conv3x3": 0, "attention": 0}

    roots = [model] if roots is None else list(roots)
    mods = [m for r in roots for m in r.modules()]
    for mod in mods:
        if isinstance(mod, CrossEmbedConvBranch):
            zp = mod[0]
            left, right, top, bottom = zp.padding
            zp.padding = (0, 0, top, bottom)  # latitude stays zero-padded; longitude comes from the hook

            def hook(_m, args, lr=(left, right)):
                x = args[0]
                return (periodic_extend(x, _halo(x.shape[-1], pad_lon, padded_width), *lr),) + tuple(args[1:])

            handles.append(zp.register_forward_pre_hook(hook))
            counts["cross_embed"] += 1
        elif isinstance(mod, Attention):

            def ahook(_m, args):
                x = args[0]
                return (periodic_extend(x, _halo(x.shape[-1], pad_lon, padded_width), 0, 0),) + tuple(args[1:])

            handles.append(mod.register_forward_pre_hook(ahook))
            counts["attention"] += 1

    for mod in mods:
        if (
            isinstance(mod, nn.Conv2d)
            and mod.kernel_size == (3, 3)
            and mod.stride == (1, 1)
            and mod.padding == (1, 1)
            and mod.padding_mode == "zeros"
        ):
            mod.padding = (0, 0)

            def chook(_m, args):
                x = args[0]
                x = periodic_extend(x, _halo(x.shape[-1], pad_lon, padded_width), 1, 1)
                return (F.pad(x, (0, 0, 1, 1)),) + tuple(args[1:])

            handles.append(mod.register_forward_pre_hook(chook))
            counts["conv3x3"] += 1

    model._lon_halo_handles = handles
    setattr(model, REFRESH_ATTR, True)
    model._lon_halo_counts = counts
    return model
