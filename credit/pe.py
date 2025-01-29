import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class SurfacePosEmb2D(nn.Module):
    def __init__(
        self,
        image_height,
        image_width,
        patch_height,
        patch_width,
        dim,
        temperature=10000,
        cls_token=False,
    ):
        super(SurfacePosEmb2D, self).__init__()
        y, x = torch.meshgrid(
            torch.arange(image_height // patch_height),
            torch.arange(image_width // patch_width),
            indexing="ij",
        )

        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        # Add an additional row for the CLS token
        if cls_token:
            cls_pe = torch.zeros(1, pe.size(1))
            pe = torch.cat((cls_pe, pe), dim=0)

        self.embedding = nn.Parameter(pe)

    def forward(self, x):
        return x + self.embedding.to(dtype=x.dtype, device=x.device)


class PosEmb3D(nn.Module):
    def __init__(
        self,
        frames,
        image_height,
        image_width,
        frame_patch_size,
        patch_height,
        patch_width,
        dim,
        temperature=10000,
    ):
        super(PosEmb3D, self).__init__()
        z, y, x = torch.meshgrid(
            torch.arange(frames // frame_patch_size),
            torch.arange(image_height // patch_height),
            torch.arange(image_width // patch_width),
            indexing="ij",
        )

        fourier_dim = dim // 6
        omega = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1.0 / (temperature**omega)

        z = z.flatten()[:, None] * omega[None, :]
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]

        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
        pe = F.pad(
            pe, (0, dim - (fourier_dim * 6))
        )  # pad if feature dimension not cleanly divisible by 6
        self.embedding = pe

    def forward(self, x):
        return x + self.embedding.to(dtype=x.dtype, device=x.device)


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, T, C, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        return x
