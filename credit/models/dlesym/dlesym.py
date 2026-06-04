"""
DLESyM — Deep Learning Earth System Model (CREDIT port).

Faithful port of NVIDIA PhysicsNeMo HEALPixRecUNet + supporting layers.
Hydra/OmegaConf/physicsnemo dependencies removed; architecture is identical.

Architecture (atmosphere backbone, default DLESyM-v1 settings):
  Encoder: ConvNeXtBlock × 3 levels, n_channels=[136,68,34], dilations=[1,2,4]
           AvgPool2d downsampling between levels; no recurrent blocks.
  Decoder: ConvNeXtBlock × 3 levels, n_channels=[34,68,136], dilations=[4,2,1]
           TransposedConvUpsample + skip concat + ConvGRUBlock per level.
  Output:  1×1 Conv → prognostic channels; residual connection.

Paper: https://arxiv.org/abs/2409.16247
Reference: NVIDIA/physicsnemo  physicsnemo/models/dlwp_healpix/  (Apache-2.0)

CREDIT integration
------------------
CREDITDLESyM accepts (B, C, T, H, W) lat/lon tensors:
  1. Flatten T → C: (B, C*T, H, W)
  2. Lat/lon → 12 HEALPix faces: (B*12, C*T, nside, nside)
  3. HEALPixRecUNet forward (single integration step)
  4. Faces → lat/lon + residual: (B, C_out, 1, H, W)
"""

import logging
import os

import torch
import torch.nn as nn

from credit.models.healpix.healpix import (
    _build_index_buffers,
)

try:
    import importlib.util

    _HEALPY = importlib.util.find_spec("healpy") is not None
except Exception:
    _HEALPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HEALPix primitives (fold/unfold/padding/layer)
# These mirror physicsnemo/nn/module/hpx exactly, without jaxtyping/physicsnemo deps.
# ---------------------------------------------------------------------------


class HEALPixFoldFaces(nn.Module):
    """(B, 12, C, H, W) → (B*12, C, H, W)"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = x.shape
        return x.reshape(B * F, C, H, W)


class HEALPixUnfoldFaces(nn.Module):
    """(B*12, C, H, W) → (B, 12, C, H, W)"""

    def __init__(self, num_faces: int = 12):
        super().__init__()
        self.num_faces = num_faces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BF, C, H, W = x.shape
        return x.reshape(BF // self.num_faces, self.num_faces, C, H, W)


class HEALPixPadding(nn.Module):
    """Topologically-correct padding for 12-face HEALPix tensors.
    Faithful port of physicsnemo HEALPixPadding (padding.py).
    Input/output: (B*12, C, H, W)
    """

    def __init__(self, padding: int):
        super().__init__()
        self.p = padding
        self.d = (-2, -1)
        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.unfold(data)
        f = [data[:, i] for i in range(12)]

        p00 = self.pn(f[0], f[1], f[2], f[3], f[3], f[4], f[8], f[5], f[1])
        p01 = self.pn(f[1], f[2], f[3], f[0], f[0], f[5], f[9], f[6], f[2])
        p02 = self.pn(f[2], f[3], f[0], f[1], f[1], f[6], f[10], f[7], f[3])
        p03 = self.pn(f[3], f[0], f[1], f[2], f[2], f[7], f[11], f[4], f[0])
        p04 = self.pe(f[4], f[0], self.tl(f[0], f[3]), f[3], f[7], f[11], self.br(f[11], f[8]), f[8], f[5])
        p05 = self.pe(f[5], f[1], self.tl(f[1], f[0]), f[0], f[4], f[8], self.br(f[8], f[9]), f[9], f[6])
        p06 = self.pe(f[6], f[2], self.tl(f[2], f[1]), f[1], f[5], f[9], self.br(f[9], f[10]), f[10], f[7])
        p07 = self.pe(f[7], f[3], self.tl(f[3], f[2]), f[2], f[6], f[10], self.br(f[10], f[11]), f[11], f[4])
        p08 = self.ps(f[8], f[5], f[0], f[4], f[11], f[11], f[10], f[9], f[9])
        p09 = self.ps(f[9], f[6], f[1], f[5], f[8], f[8], f[11], f[10], f[10])
        p10 = self.ps(f[10], f[7], f[2], f[6], f[9], f[9], f[8], f[11], f[11])
        p11 = self.ps(f[11], f[4], f[3], f[7], f[10], f[10], f[9], f[8], f[8])

        stacked = torch.stack([p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11], dim=1)
        return self.fold(stacked)

    def pn(self, c, t, tl, lft, bl, b, br, rgt, tr):
        p, d = self.p, self.d
        c = torch.cat([t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]], dim=-2)
        left = torch.cat([tl.rot90(2, d)[..., -p:, -p:], lft.rot90(-1, d)[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def pe(self, c, t, tl, lft, bl, b, br, rgt, tr):
        p = self.p
        c = torch.cat([t[..., -p:, :], c, b[..., :p, :]], dim=-2)
        left = torch.cat([tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def ps(self, c, t, tl, lft, bl, b, br, rgt, tr):
        p, d = self.p, self.d
        c = torch.cat([t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]], dim=-2)
        left = torch.cat([tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]], dim=-2)
        right = torch.cat([tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]], dim=-2)
        return torch.cat([left, c, right], dim=-1)

    def tl(self, top, lft):
        p = self.p
        ret = torch.zeros_like(top)[..., :p, :p]
        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]
        for i in range(1, p):
            ret[..., -i - 1, -i:] = top[..., -i - 1, :i]
            ret[..., -i:, -i - 1] = lft[..., :i, -i - 1]
            ret[..., -i - 1, -i - 1] = 0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
        return ret

    def br(self, b, r):
        p = self.p
        ret = torch.zeros_like(b)[..., :p, :p]
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]
        for i in range(1, p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]
        return ret


class HEALPixLayer(nn.Module):
    """Wraps any 2D layer with HEALPix-aware padding for Conv2d.
    Mirrors physicsnemo HEALPixLayer (layers.py).
    For Conv2d with kernel_size>1: prepend HEALPixPadding, set padding=0.
    For kernel_size=1 or non-conv layers: pass through directly.
    """

    def __init__(self, layer, **kwargs):
        super().__init__()
        layers = []
        kwargs.pop("enable_nhwc", None)
        kwargs.pop("enable_healpixpad", None)

        is_conv = issubclass(layer, nn.modules.conv._ConvNd)
        kernel_size = kwargs.get("kernel_size", 3)
        if is_conv and kernel_size > 1:
            kwargs["padding"] = 0
            dilation = kwargs.get("dilation", 1)
            pad = ((kernel_size - 1) // 2) * dilation
            layers.append(HEALPixPadding(pad))

        layers.append(layer(**kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Building blocks — faithful ports of physicsnemo healpix_blocks.py
# ---------------------------------------------------------------------------


class ConvGRUBlock(nn.Module):
    """Convolutional GRU with 1×1 kernel on folded HEALPix faces.
    Output: inputs + h_next  (residual form, as in DLESyM).
    Stores hidden state internally; call reset() between sequences.
    Mirrors physicsnemo ConvGRUBlock exactly.
    """

    def __init__(self, in_channels: int, kernel_size: int = 1):
        super().__init__()
        self.channels = in_channels
        self.conv_gates = HEALPixLayer(
            nn.Conv2d,
            in_channels=in_channels + in_channels,
            out_channels=2 * in_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv_can = HEALPixLayer(
            nn.Conv2d,
            in_channels=in_channels + in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.register_buffer("h", torch.zeros(1, 1, 1, 1), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape != self.h.shape:
            self.h = torch.zeros_like(inputs)
        combined = torch.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        combined = torch.cat([inputs, reset_gate * self.h], dim=1)
        cnm = torch.tanh(self.conv_can(combined))
        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next
        return inputs + h_next

    def reset(self):
        self.h = torch.zeros_like(self.h)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with dilated HEALPix-aware convolutions.
    Mirrors physicsnemo ConvNeXtBlock exactly:
      Conv3×3(in → latent*4, dil=d) + act
      Conv3×3(latent*4 → latent*4, dil=d) + act
      Conv1×1(latent*4 → out)
      + skip (identity or 1×1 projection)
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        activation: nn.Module = None,
        **kwargs,  # absorb n_layers, enable_nhwc, enable_healpixpad from encoder/decoder
    ):
        super().__init__()
        hidden = int(latent_channels * upscale_factor)
        act = activation if activation is not None else nn.LeakyReLU(0.15)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = HEALPixLayer(nn.Conv2d, in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.net = nn.Sequential(
            HEALPixLayer(
                nn.Conv2d, in_channels=in_channels, out_channels=hidden, kernel_size=kernel_size, dilation=dilation
            ),
            act,
            HEALPixLayer(
                nn.Conv2d, in_channels=hidden, out_channels=hidden, kernel_size=kernel_size, dilation=dilation
            ),
            act,
            HEALPixLayer(nn.Conv2d, in_channels=hidden, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.net(x)


class AvgPoolBlock(nn.Module):
    """2×2 average pool downsampling (mirrors physicsnemo avg_pool config)."""

    def __init__(self, pooling: int = 2, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class TransposedConvUpsample(nn.Module):
    """ConvTranspose2d 2× upsampling (mirrors physicsnemo TransposedConvUpsample)."""

    def __init__(self, in_channels: int, out_channels: int, upsampling: int = 2, **kwargs):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upsampling, stride=upsampling, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class OutputLayer(nn.Module):
    """1×1 linear output projection."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = HEALPixLayer(nn.Conv2d, in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder and decoder — faithful ports of healpix_encoder.py / healpix_decoder.py
# ---------------------------------------------------------------------------


class UNetEncoder(nn.Module):
    """HEALPix U-Net encoder.
    Mirrors physicsnemo UNetEncoder with ConvNeXtBlock + AvgPool + optional ConvGRU.
    n_channels: channel count at each scale (level 0 = finest).
    """

    def __init__(
        self,
        input_channels: int,
        n_channels: tuple = (136, 68, 34),
        n_layers: tuple = (2, 2, 1),
        dilations: tuple = (1, 2, 4),
    ):
        super().__init__()
        self.n_channels = n_channels
        dilations = dilations or [1] * len(n_channels)

        self.encoder = nn.ModuleList()
        old_ch = input_channels
        for n, ch in enumerate(n_channels):
            mods = []
            if n > 0:
                mods.append(AvgPoolBlock())
            mods.append(
                ConvNeXtBlock(
                    in_channels=old_ch,
                    latent_channels=ch,
                    out_channels=ch,
                    dilation=dilations[n],
                    n_layers=n_layers[n],
                )
            )
            self.encoder.append(nn.Sequential(*mods))
            old_ch = ch

    def forward(self, x: torch.Tensor) -> list:
        outputs = []
        for layer in self.encoder:
            x = layer(x)
            outputs.append(x)
        return outputs

    def reset(self):
        pass


class UNetDecoder(nn.Module):
    """HEALPix U-Net decoder with ConvGRU recurrent blocks.
    Mirrors physicsnemo UNetDecoder exactly:
      Level 0: no upsample; ConvNeXtBlock(curr → next); ConvGRU(next)
      Level n: upsample(curr→curr); cat(curr+skip=2*curr); ConvNeXtBlock(2*curr→next); ConvGRU(next)
      Final: OutputLayer(last → output_channels)
    """

    def __init__(
        self,
        output_channels: int,
        n_channels: tuple = (34, 68, 136),
        n_layers: tuple = (1, 2, 2),
        dilations: tuple = (4, 2, 1),
    ):
        super().__init__()
        self.channel_dim = 1
        dilations = dilations or [1] * len(n_channels)

        self.decoder = nn.ModuleList()
        for n, curr_ch in enumerate(n_channels):
            up = None if n == 0 else TransposedConvUpsample(curr_ch, curr_ch)
            next_ch = n_channels[n + 1] if n < len(n_channels) - 1 else n_channels[-1]
            in_ch = curr_ch * 2 if n > 0 else curr_ch
            conv = ConvNeXtBlock(
                in_channels=in_ch,
                latent_channels=curr_ch,
                out_channels=next_ch,
                dilation=dilations[n],
                n_layers=n_layers[n],
            )
            rec = ConvGRUBlock(in_channels=next_ch)
            self.decoder.append(nn.ModuleDict({"upsamp": up, "conv": conv, "recurrent": rec}))

        self.output_layer = OutputLayer(n_channels[-1], output_channels)

    def forward(self, inputs: list) -> torch.Tensor:
        x = inputs[-1]
        for n, layer in enumerate(self.decoder):
            if layer["upsamp"] is not None:
                up = layer["upsamp"](x)
                x = torch.cat([up, inputs[-1 - n]], dim=self.channel_dim)
            x = layer["conv"](x)
            x = layer["recurrent"](x)
        return self.output_layer(x)

    def reset(self):
        for layer in self.decoder:
            layer["recurrent"].reset()


# ---------------------------------------------------------------------------
# HEALPixRecUNet — the DLESyM core (faithful port of physicsnemo HEALPixRecUNet)
# ---------------------------------------------------------------------------


class HEALPixRecUNet(nn.Module):
    """Recurrent HEALPix U-Net.
    Faithful port of physicsnemo HEALPixRecUNet without hydra/physicsnemo.
    Operates on folded HEALPix tensors (B*12, C, H, W).

    Forward input: flat (B*12, C_total, H, W) where C_total =
        input_time_dim * (input_channels + decoder_input_channels) + n_constants
    Residual connection on first input_channels * input_time_dim channels.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        n_constants: int,
        decoder_input_channels: int,
        input_time_dim: int,
        output_time_dim: int,
        n_channels: tuple = (136, 68, 34),
        dilations: tuple = (1, 2, 4),
        n_layers_enc: tuple = (2, 2, 1),
        n_layers_dec: tuple = (1, 2, 2),
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels

        self.is_diagnostic = output_time_dim == 1 and input_time_dim > 1
        self.integration_steps = max(output_time_dim // input_time_dim, 1)

        total_in = input_time_dim * (input_channels + decoder_input_channels) + n_constants
        total_out = (1 if self.is_diagnostic else input_time_dim) * output_channels

        self.encoder = UNetEncoder(
            input_channels=total_in,
            n_channels=n_channels,
            n_layers=n_layers_enc,
            dilations=dilations,
        )
        self.decoder = UNetDecoder(
            output_channels=total_out,
            n_channels=tuple(reversed(n_channels)),
            n_layers=n_layers_dec,
            dilations=tuple(reversed(dilations)),
        )

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces()

    def _reshape_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """(B*12, C, H, W) → (B, 12, T, C_out, H, W)"""
        outputs = self.unfold(outputs)
        B, F, C, H, W = outputs.shape
        T = 1 if self.is_diagnostic else self.input_time_dim
        return outputs.reshape(B, F, T, -1, H, W)

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*12, C_total, H, W) — already folded, concat of all inputs
        returns: (B*12, C_out * T, H, W)
        """
        self.reset()
        # residual channels = first input_channels * input_time_dim
        n_res = self.input_channels * self.input_time_dim
        encodings = self.encoder(x)
        decoded = self.decoder(encodings)
        return x[:, :n_res] + decoded


# ---------------------------------------------------------------------------
# CREDIT wrapper — lat/lon ↔ HEALPix + CREDIT (B,C,T,H,W) interface
# ---------------------------------------------------------------------------


class CREDITDLESyM(nn.Module):
    """CREDIT wrapper for the DLESyM atmospheric backbone (HEALPixRecUNet).

    Accepts flat (B, C, T, H, W) or (B, C, H, W) lat/lon tensors.
    Reprojects to HEALPix, runs the recurrent U-Net, reprojects back.

    The full CREDIT channel vector (prognostic + forcing + static) is fed
    directly to the model; the residual connection uses the first
    `out_channels` channels of the input.

    Parameters
    ----------
    in_channels : int
        Prognostic input channels per time step (also used for residual).
    out_channels : int
        Output channels.
    img_size : (H, W)
        Lat/lon grid size.
    frames : int
        Input time steps (input_time_dim).  DLESyM uses 4.
    nside : int
        HEALPix resolution.  nside=64 ≈ 1°.
    n_channels : list[int]
        Encoder channel counts per level.  Default [136, 68, 34] (DLESyM).
    dilations : list[int]
        Encoder dilation per level.  Default [1, 2, 4].
    n_layers_enc : list[int]
        ConvNeXt blocks per encoder level.
    n_layers_dec : list[int]
        ConvNeXt blocks per decoder level.
    forcing_channels : int
        Extra dynamic/static channels per time step beyond in_channels.
        These are concatenated into the flat encoder input but excluded
        from the residual.  Equivalent to decoder_input_channels + n_constants.
    lat_range : (float, float)
        Latitude extents — (north, south) i.e. (90, -90) for global.
    lon_range : (float, float)
        Longitude extents — (0, 360) for global.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        img_size: tuple = (181, 360),
        frames: int = 4,
        nside: int = 64,
        n_channels: tuple = (136, 68, 34),
        dilations: tuple = (1, 2, 4),
        n_layers_enc: tuple = (2, 2, 1),
        n_layers_dec: tuple = (1, 2, 2),
        forcing_channels: int = 0,
        lat_range: tuple = (90.0, -90.0),
        lon_range: tuple = (0.0, 360.0),
    ):
        super().__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.nside = nside
        self.out_channels = out_channels
        self.frames = frames

        if not _HEALPY:
            logger.warning("healpy not installed — HEALPix grid is approximate. Install with: pip install healpy")

        hp_to_ll, ll_to_hp = _build_index_buffers(nside, H, W, lat_range, lon_range)
        self.register_buffer("hp_to_ll", hp_to_ll)
        self.register_buffer("ll_to_hp", ll_to_hp)

        # total input channels per time step = in_channels + extra forcing per step
        # forcing_channels covers decoder_input_channels + n_constants collapsed into flat
        total_per_step = in_channels + forcing_channels
        self.model = HEALPixRecUNet(
            input_channels=in_channels,
            output_channels=out_channels,
            n_constants=0,
            decoder_input_channels=forcing_channels,
            input_time_dim=frames,
            output_time_dim=frames,
            n_channels=tuple(n_channels),
            dilations=tuple(dilations),
            n_layers_enc=tuple(n_layers_enc),
            n_layers_dec=tuple(n_layers_dec),
        )

        self.fold = HEALPixFoldFaces()
        self.unfold = HEALPixUnfoldFaces()

    # ------------------------------------------------------------------
    # Lat/lon ↔ HEALPix reprojection
    # ------------------------------------------------------------------

    def _ll_to_faces(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B*12, C, nside, nside)"""
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W)
        idx = self.hp_to_ll.reshape(-1)
        hp_flat = x_flat[:, :, idx]
        faces = hp_flat.reshape(B, C, 12, self.nside, self.nside)
        return self.fold(faces.permute(0, 2, 1, 3, 4))

    def _faces_to_ll(self, x: torch.Tensor) -> torch.Tensor:
        """(B*12, C, nside, nside) → (B, C, H, W)"""
        B12, C, NS, _ = x.shape
        B = B12 // 12
        faces = self.unfold(x)
        hp_flat = faces.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        ll_flat = hp_flat[:, :, self.ll_to_hp]
        return ll_flat.reshape(B, C, self.H, self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.reshape(B, C * T, H, W)

        faces = self._ll_to_faces(x)  # (B*12, C*T, nside, nside)
        out = self.model(faces)  # (B*12, C_out*T, nside, nside)
        # take only the last time step's channels
        out = out[:, -self.out_channels :]
        ll = self._faces_to_ll(out)  # (B, C_out, H, W)
        return ll.unsqueeze(2)  # (B, C_out, 1, H, W)

    @classmethod
    def load_model(cls, conf):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "model_checkpoint.pt")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def load_model_name(cls, conf, model_name):
        model = cls(**{k: v for k, v in conf["model"].items() if k != "type"})
        ckpt = os.path.join(os.path.expandvars(conf["save_loc"]), model_name)
        checkpoint = torch.load(ckpt, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
        return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, _root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DLESyM-like config: 8 atmo vars, 4 time steps, nside=16 for speed
    model = CREDITDLESyM(
        in_channels=8,
        out_channels=8,
        img_size=(32, 64),
        frames=4,
        nside=16,
        n_channels=(136, 68, 34),
        dilations=(1, 2, 4),
        n_layers_enc=(2, 2, 1),
        n_layers_dec=(1, 2, 2),
        forcing_channels=0,
    ).to(device)

    x = torch.randn(1, 8, 4, 32, 64, device=device)
    y = model(x)
    assert y.shape == (1, 8, 1, 32, 64), f"unexpected output shape {y.shape}"
    y.mean().backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    healpy_str = "healpy" if _HEALPY else "approx"
    print(
        f"CREDITDLESyM OK ({healpy_str}) — input {x.shape} → output {y.shape}, params {n_params:.1f}M, device {device}"
    )
