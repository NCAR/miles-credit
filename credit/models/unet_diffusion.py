import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from credit.diffusion import ModifiedGaussianDiffusion, default
import math

import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from credit.models.base_model import BaseModel
from credit.attend import Attend
from credit.boundary_padding import TensorPadding
from credit.diffusion_utils import cast_tuple, exists, divisible_by
from einops.layers.torch import Rearrange


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


# sinusoidal positional embeds


class SinusoidalPosEmb(Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# class PeriodicBoundaryConv(nn.Module):
#     def __init__(self, input_channels, init_dim, kernel_size=7, conditional_dimensions=0):
#         super(PeriodicBoundaryConv, self).__init__()
#         self.kernel_size = kernel_size
#         # Compute padding based on the kernel size (integer division for "same" padding)
#         self.padding = kernel_size // 2

#         # Initialize Conv2d layer with the variable kernel size
#         self.init_conv = nn.Conv2d(input_channels + conditional_dimensions, init_dim, kernel_size, padding=0)  # Set padding=0 as we handle it manually

#     def forward(self, x):
#         # x shape is [batch, variable, latitude, longitude]

#         # 1. Circular padding for longitude (last dimension)
#         x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')  # Pad last dimension (longitude) by self.padding

#         # 2. Zero padding (or reflect padding) for latitude (second-to-last dimension)
#         x = F.pad(x, (0, 0, self.padding, self.padding), mode='reflect')  # Pad second-last dimension (latitude) by self.padding

#         # 3. Apply convolution
#         x = self.init_conv(x)

#         return x


class PeriodicConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=1):
        super(PeriodicConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size, padding=0
        )  # Padding is handled manually

    def forward(self, x):
        # Apply circular padding (periodic boundary condition) on longitude
        x = F.pad(
            x, (self.padding, self.padding, 0, 0), mode="circular"
        )  # Apply padding to the last (longitude) dimension

        # Reflect padding or zero padding on latitude (2nd-to-last dimension)
        x = F.pad(
            x, (0, 0, self.padding, self.padding), mode="reflect"
        )  # Reflect padding for latitude

        # Apply convolution
        return self.conv(x)


# model


class UnetDiffusion(BaseModel):
    def __init__(
        self,
        image_height: int = 640,
        image_width: int = 1280,
        init_dim=None,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        padding_conf: dict = None,
        post_conf: dict = None,
        dim_mults: tuple = (1, 2, 4, 8),
        conditional_dimensions: int = 0,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        dropout: float = 0.0,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        full_attn: dict = None,
        flash_attn: bool = False,
        self_condition: bool = False,
        condition: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        # determine dimensions

        # output channels
        output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = output_channels

        print("output channels", output_channels)

        self.channels = channels
        self.channels_out = output_channels
        self.pre_out_dim = surface_channels + (channels * levels)  ## c
        self.self_condition = self_condition
        self.conditional_dimensions = conditional_dimensions
        self.condition = condition
        self.image_height = image_height
        self.image_width = image_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = post_conf["activate"]

        self.use_padding = padding_conf["activate"]

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # input channels
        self.input_only_channels = input_only_channels
        input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = input_channels

        # output channels
        output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = output_channels

        init_dim = default(init_dim, dim[0])

        # Old code (deprecated usage of padding):

        if self.condition:
            self.init_conv = nn.Conv2d(
                output_channels + input_channels,
                init_dim,
                kernel_size=7,
                padding="same",
            )
        else:
            self.init_conv = nn.Conv2d(
                output_channels, init_dim, kernel_size=7, padding="same"
            )

        dims = [init_dim, *map(lambda m: dim[0] * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim[0] * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim[0], theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim[0]

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(
                            dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else PeriodicConv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(
            mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else PeriodicConv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # default_out_dim = self.channels_out * (1 if not learned_variance else 2)
        # self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.output_channels, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None, x_cond=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        # print("▸ before condition cat:", x.shape)
        if self.condition:
            x = torch.cat((x, x_cond), dim=1)

        if self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:  # case where only using one time-step as input
            x = x.squeeze(2)

        # print("▸ before padding:", x.shape)
        if self.use_padding:
            x = self.padding_opt.pad(x)
        # print("▸ after padding:", x.shape)
        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(time)

        h = []
        # print("▸ before downs:", x.shape)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            # print("    ↓ splitting:", x.shape[-2:])
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)
            # print("▸ after upsample:", x.shape)

        # print("▸ after final upsample:", x.shape)
        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        x = self.final_conv(x)
        # print("▸ after final conv:", x.shape)
        if self.use_padding:
            x = self.padding_opt.unpad(x)

        x = x.unsqueeze(2)

        # print("exiting Unet", x.shape)
        return x


def create_model(config, self_condition=True):
    """Initialize and return the CrossFormer model using a config dictionary."""
    return UnetDiffusion(**config).to("cuda")


def create_diffusion(model, config):
    """Initialize and return the Gaussian Diffusion process."""
    return ModifiedGaussianDiffusion(model, **config)


if __name__ == "__main__":
    ##### unet diffusion settings:
    unet_config = {
        "frames": 1,  # number of input states (default: 1)
        "image_height": 192,  # number of latitude grids (default: 640)
        "image_width": 288,  # number of longitude grids (default: 1280)
        "levels": 32,  # number of upper-air variable levels (default: 15)
        "channels": 4,  # upper-air variable channels
        "surface_channels": 3,  # surface variable channels
        "input_only_channels": 3,  # dynamic forcing, forcing, static channels
        "output_only_channels": 16,  # diagnostic variable channels
        "dim_mults": (1, 2, 4, 8),
        "conditional_dimensions": 0,
        "self_condition": False,
        "condition": False,
        "learned_variance": False,
        "learned_sinusoidal_cond": False,
        "random_fourier_features": False,
        "learned_sinusoidal_dim": 16,
        "sinusoidal_pos_emb_theta": 10000,
        "dropout": 0.0,
        "attn_dim_head": 32,
        "attn_heads": 4,
        "full_attn": None,  # defaults to full attention only for inner most layer
        "flash_attn": True,
        "padding_conf": {
            "activate": True,
            "mode": "earth",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
    }

    diffusion_config = {
        "image_size": (192, 288),
        "timesteps": 100,
        "sampling_timesteps": None,
        "objective": "pred_v",
        "beta_schedule": "sigmoid",
        "schedule_fn_kwargs": dict(),
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
        "offset_noise_strength": 0.0,
        "min_snr_loss_weight": False,
        "min_snr_gamma": 5,
        "immiscible": False,
    }

    model = create_model(unet_config).to("cuda")
    diffusion = create_diffusion(model, diffusion_config).to("cuda")

    noise_tensor = torch.randn(
        1,
        unet_config["channels"] * unet_config["levels"]
        + unet_config["surface_channels"]
        + unet_config["output_only_channels"],
        unet_config["frames"],
        unet_config["image_height"],
        unet_config["image_width"],
    ).to("cuda")

    conditional_tensor = torch.randn(
        1,
        unet_config["channels"] * unet_config["levels"]
        + unet_config["surface_channels"]
        + unet_config["input_only_channels"],
        unet_config["frames"],
        unet_config["image_height"],
        unet_config["image_width"],
    ).to("cuda")

    print(
        unet_config["channels"],
        unet_config["levels"],
        unet_config["channels"] * unet_config["levels"],
        noise_tensor.shape,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    model_out, loss = diffusion(noise_tensor, x_cond=conditional_tensor)  # lazy test
    loss = loss.item()

    print("Loss:", loss)

    sampled_images = diffusion.sample(conditional_tensor, batch_size=1)

    print("Predicted shape:", sampled_images.shape)

    # Extract the last color channel (index -1 for the last channel)
    last_channel = sampled_images[0, -1, 0, :, :]

    import matplotlib.pyplot as plt

    # Plot and save the image
    plt.imshow(last_channel.cpu().numpy(), cmap="gray")  # Display in grayscale
    plt.axis("off")  # Turn off the axis
    plt.savefig("last_channel.png", bbox_inches="tight", pad_inches=0)
    plt.close()
