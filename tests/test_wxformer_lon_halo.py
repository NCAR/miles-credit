"""Tests for the pole vector flip (TensorPadding) and the per-layer longitude halo refresh (WXFormer)."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from credit.boundary_padding import TensorPadding
from credit.models.wxformer.crossformer import CrossEmbedLayer, CrossFormer
from credit.models.wxformer.lon_halo import install_lon_halo_refresh, periodic_extend

# ---------------------------------------------------------------------------- pole flip


def test_pole_flip_negates_listed_channels_in_pole_rows():
    x = torch.randn(1, 3, 1, 19, 36)
    plain = TensorPadding(mode="earth", pad_lat=[4, 5], pad_lon=[2, 2]).pad(x)
    flip = TensorPadding(
        mode="earth", pad_lat=[4, 5], pad_lon=[2, 2], pole_flip_channels=[1, 2], pole_flip_offsets=[0.5, 0.0]
    ).pad(x)
    assert flip.shape == plain.shape
    top, bot = slice(0, 4), slice(4 + 19, None)
    for rows in (top, bot):
        torch.testing.assert_close(flip[:, 0, :, rows], plain[:, 0, :, rows])  # scalar channel untouched
        torch.testing.assert_close(flip[:, 1, :, rows], -plain[:, 1, :, rows] - 0.5)  # z -> -z - 2 mu / sigma
        torch.testing.assert_close(flip[:, 2, :, rows], -plain[:, 2, :, rows])
    torch.testing.assert_close(flip[:, :, :, 4 : 4 + 19], plain[:, :, :, 4 : 4 + 19])  # real rows untouched
    torch.testing.assert_close(TensorPadding(mode="earth", pad_lat=[4, 5], pad_lon=[2, 2]).unpad(flip), x)


def test_pole_flip_is_the_physical_reflection_for_standardized_input():
    # a uniform flow over the north pole, in physical units, standardized with a non-zero mean
    nlat, nlon, pad = 9, 72, 3
    lat = torch.linspace(90, 50, nlat).deg2rad().view(-1, 1)
    lon = torch.arange(nlon).mul(360 / nlon).deg2rad().view(1, -1)
    X, Y = 10.0, -4.0  # pole-plane components of the flow
    u = (-X * torch.sin(lon) + Y * torch.cos(lon)).expand(nlat, nlon)
    v = (-(X * torch.cos(lon) + Y * torch.sin(lon)) / torch.sin(lat)).expand(nlat, nlon)
    mu, sd = torch.tensor([3.0, -1.0]), torch.tensor([7.0, 5.0])
    z = ((torch.stack([u, v]) - mu.view(2, 1, 1)) / sd.view(2, 1, 1)).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, H, W)
    tp = TensorPadding(
        mode="earth",
        pad_lat=[pad, pad],
        pad_lon=[0, 0],
        pole_flip_channels=[0, 1],
        pole_flip_offsets=(2 * mu / sd).tolist(),
    )
    out = tp.pad(z)[0, :, 0, :pad]  # padded rows: the rows just "beyond" the pole, i.e. rows 1..pad seen from across
    phys = out * sd.view(2, 1, 1) + mu.view(2, 1, 1)
    # beyond the pole at (lat_j, lon + 180) the field is the same uniform flow, evaluated at that point
    lon_s = lon + torch.pi
    for k in range(pad):
        j = pad - 1 - k  # torch.flip puts row pad-1 first
        uu = -X * torch.sin(lon_s) + Y * torch.cos(lon_s)
        vv = -(X * torch.cos(lon_s) + Y * torch.sin(lon_s)) / torch.sin(lat[j])
        # the reflected row lies across the pole: east and north both reverse
        torch.testing.assert_close(phys[0, k], -uu.squeeze(0), atol=1e-4, rtol=0)
        torch.testing.assert_close(phys[1, k], -vv.squeeze(0), atol=1e-4, rtol=0)


def test_pole_flip_validation():
    with pytest.raises(ValueError):
        TensorPadding(mode="earth", pole_flip_channels=[1, 2], pole_flip_offsets=[0.1])
    with pytest.raises(ValueError):
        TensorPadding(mode="mirror", pole_flip_channels=[1])


# ---------------------------------------------------------------------------- longitude halo refresh

PAD, NLON, NLAT = 32, 256, 20
WIDTH = NLON + 2 * PAD


def _padded(xi):
    return torch.cat([xi[..., -PAD:], xi, xi[..., :PAD]], -1)


def test_periodic_extend():
    xi = torch.arange(10.0).view(1, 1, 1, 10)
    x = torch.cat([xi[..., -2:], xi, xi[..., :2]], -1)
    out = periodic_extend(x, 2, 1, 3)
    assert out[0, 0, 0].tolist() == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]


def test_refresh_matches_periodic_convolution():
    torch.manual_seed(0)
    xi = torch.randn(1, 4, NLAT, NLON)
    holder = nn.Module()
    holder.c3 = nn.Conv2d(4, 4, 3, padding=1)
    holder.ce = CrossEmbedLayer(4, 8, [4, 8, 16], stride=2)
    ref3 = F.conv2d(F.pad(F.pad(xi, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1)), holder.c3.weight, holder.c3.bias)
    big = F.pad(xi, (64, 64, 0, 0), mode="circular")  # far from the outer edge the zero padding does not matter
    refce = holder.ce(big)[..., 32 : 32 + NLON // 2]
    install_lon_halo_refresh(holder, PAD, WIDTH)
    out3 = holder.c3(_padded(xi))
    torch.testing.assert_close(out3[..., PAD:-PAD], ref3)
    torch.testing.assert_close(out3[..., :PAD], out3[..., NLON : NLON + PAD])  # halo stays an exact copy
    ce = holder.ce(_padded(xi))
    torch.testing.assert_close(ce[..., PAD // 2 : -PAD // 2], refce)
    torch.testing.assert_close(ce[..., : PAD // 2], ce[..., NLON // 2 : NLON // 2 + PAD // 2])


def test_refresh_rejects_widths_that_do_not_halve_evenly():
    holder = nn.Module()
    holder.c3 = nn.Conv2d(2, 2, 3, padding=1)
    install_lon_halo_refresh(holder, 10, 100)
    assert holder.c3(torch.randn(1, 2, 8, 30)).shape == (1, 2, 8, 30)  # halo 10 * 30 / 100 = 3 columns
    with pytest.raises(ValueError, match="halo"):
        holder.c3(torch.randn(1, 2, 8, 45))  # halo would be 4.5 columns


def _small_wxformer(**pad_extra):
    padding_conf = {"activate": True, "mode": "earth", "pad_lat": [6, 6], "pad_lon": [16, 16], **pad_extra}
    torch.manual_seed(0)
    return CrossFormer(
        image_height=52,
        image_width=128,
        frames=1,
        channels=2,
        surface_channels=1,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(32, 64, 128, 256),
        depth=(1, 1, 1, 1),
        global_window_size=(4, 2, 1, 1),
        local_window_size=2,
        cross_embed_kernel_sizes=((4, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        padding_conf=padding_conf,
    )


def test_wxformer_refresh_keeps_state_dict_and_shapes():
    base = _small_wxformer()
    ref = _small_wxformer(refresh_lon_halo=True, pole_flip_channels=[2, 3, 4, 5])
    assert list(base.state_dict()) == list(ref.state_dict())
    ref.load_state_dict(base.state_dict())
    assert getattr(ref, "_lon_halo_refresh", False)
    assert ref._lon_halo_counts["cross_embed"] > 0 and ref._lon_halo_counts["attention"] > 0
    x = torch.randn(1, 2 * 2 + 1 + 1, 1, 52, 128)
    with torch.no_grad():
        y0, y1 = base(x), ref(x)
    assert y1.shape == y0.shape and torch.isfinite(y1).all()
    assert not torch.allclose(y0, y1)  # the refresh changes the edge columns


def test_wxformer_refresh_config_validation():
    with pytest.raises(ValueError, match="symmetric pad_lon"):
        _small_wxformer(refresh_lon_halo=True, pad_lon=[16, 32])
    with pytest.raises(ValueError, match="divide by 2"):
        _small_wxformer(refresh_lon_halo=True, pad_lon=[8, 8])  # 8 does not divide by 2**4


def test_domain_parallel_refuses_refreshed_model():
    from credit.domain_parallel.convert import convert_to_domain_parallel

    with pytest.raises(ValueError, match="refresh_lon_halo"):
        convert_to_domain_parallel(_small_wxformer(refresh_lon_halo=True), manager=None)
