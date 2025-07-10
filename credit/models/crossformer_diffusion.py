import torch
import torch.nn as nn
from credit.diffusion import ModifiedGaussianDiffusion
from credit.models.base_model import BaseModel
from credit.postblock import PostBlock
from credit.boundary_padding import TensorPadding
from credit.models.crossformer import (
    Attention,
    FeedForward,
    CrossEmbedLayer,
    CubeEmbedding,
    apply_spectral_norm,
    cast_tuple,
)

import torch.nn.functional as F
from collections import namedtuple
import logging


logger = logging.getLogger(__name__)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


# mp fourier embeds


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2, emb_dim=None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.output_channels = out_chans

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

        # Optional embedding projection
        self.to_emb = None
        if emb_dim is not None:
            self.to_emb = nn.Sequential(nn.Linear(emb_dim, out_chans), nn.SiLU())

    def forward(self, x, emb=None):
        x = self.conv(x)
        shortcut = x

        x = self.b(x)

        if self.to_emb is not None and emb is not None:
            scale = self.to_emb(emb) + 1
            x = x * scale[:, :, None, None]  # reshape for broadcasting

        return x + shortcut


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
    ):
        super().__init__()
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

    def forward(self, x, t_emb=None):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x

            if t_emb is not None:
                # assume t_emb is [B, C] — reshape and add to [B, C, H, W]
                t_add = t_emb[:, :, None, None]
                x = x + t_add  # inject time into spatial tokens

            x = long_ff(x) + x

        return x


class CrossFormerDiffusion(BaseModel):
    def __init__(
        self,
        self_condition: bool = False,
        condition: bool = True,
        image_height: int = 640,
        patch_height: int = 1,
        image_width: int = 1280,
        patch_width: int = 1,
        frames: int = 2,
        channels: int = 4,
        surface_channels: int = 7,
        input_only_channels: int = 3,
        output_only_channels: int = 0,
        levels: int = 15,
        dim: tuple = (64, 128, 256, 512),
        depth: tuple = (2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size: tuple = (5, 5, 2, 1),
        local_window_size: int = 10,
        cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides: tuple = (4, 2, 2, 2),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_spectral_norm: bool = True,
        interp: bool = True,
        padding_conf: dict = None,
        post_conf: dict = None,
        **kwargs,
    ):
        """
        CrossFormer is the base architecture for the WXFormer model. It uses convolutions and long and short distance
        attention layers in the encoder layer and then uses strided transpose convolution blocks for the decoder
        layer.

        Args:
            image_height (int): number of grid cells in the south-north direction.
            patch_height (int): number of grid cells within each patch in the south-north direction.
            image_width (int): number of grid cells in the west-east direction.
            patch_width (int): number of grid cells within each patch in the west-east direction.
            frames (int): number of time steps being used as input
            channels (int): number of 3D variables. Default is 4 for our ERA5 configuration (U, V, T, and Q)
            surface_channels (int): number of surface (single-level) variables.
            input_only_channels (int): number of variables only used as input to the ML model (e.g., forcing variables)
            output_only_channels (int):number of variables that are only output by the model (e.g., diagnostic variables).
            levels (int): number of vertical levels for each 3D variable (should be the same across frames)
            dim (tuple): output dimensions of hidden state of each conv/transformer block in the encoder
            depth (tuple): number of attention blocks per encoder layer
            dim_head (int): dimension of each attention head.
            global_window_size (tuple): number of grid cells between cells in long range attention
            local_window_size (tuple): number of grid cells between cells in short range attention
            cross_embed_kernel_sizes (tuple): width of the cross embed kernels in each layer
            cross_embed_strides (tuple): stride of convolutions in each block
            attn_dropout (float): dropout rate for attention layout
            ff_dropout (float): dropout rate for feedforward layers.
            use_spectral_norm (bool): whether to use spectral normalization
            interp (bool): whether to use interpolation
            padding_conf (dict): padding configuration
            post_conf (dict): configuration for postblock processing
            **kwargs:
        """
        super().__init__()

        dim = tuple(dim)
        depth = tuple(depth)
        global_window_size = tuple(global_window_size)
        cross_embed_kernel_sizes = tuple([tuple(_) for _ in cross_embed_kernel_sizes])
        cross_embed_strides = tuple(cross_embed_strides)

        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frames = frames
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels
        self.use_spectral_norm = use_spectral_norm
        self.use_interp = interp
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = post_conf["activate"]

        # input channels
        self.input_only_channels = input_only_channels
        input_channels = channels * levels + surface_channels + input_only_channels
        self.input_channels = input_channels

        # output channels
        output_channels = channels * levels + surface_channels + output_only_channels
        self.output_channels = output_channels

        self.total_input_channels = self.input_channels
        if kwargs.get("diffusion"):
            # do stuff
            self.total_input_channels = self.input_channels + self.output_channels

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # dimensions
        last_dim = dim[-1]
        first_dim = self.total_input_channels if (patch_height == 1 and patch_width == 1) else dim[0]
        dims = [first_dim, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))
        self.condition = condition
        self.self_condition = self_condition

        # allocate cross embed layers
        self.layers = nn.ModuleList([])

        # loop through hyperparameters
        for (
            dim_in,
            dim_out,
        ), num_layers, global_wsize, local_wsize, kernel_sizes, stride in zip(
            dim_in_and_out,
            depth,
            global_window_size,
            local_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            # create CrossEmbedLayer
            cross_embed_layer = CrossEmbedLayer(
                dim_in=dim_in, dim_out=dim_out, kernel_sizes=kernel_sizes, stride=stride
            )

            # create Transformer
            transformer_layer = Transformer(
                dim=dim_out,
                local_window_size=local_wsize,
                global_window_size=global_wsize,
                depth=num_layers,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )

            # append everything
            self.layers.append(nn.ModuleList([cross_embed_layer, transformer_layer]))

        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # define embedding layer using adjusted sizes
        # if the original sizes were good, adjusted sizes should == original sizes
        self.cube_embedding = CubeEmbedding(
            (frames, image_height, image_width),
            (frames, patch_height, patch_width),
            input_channels,
            dim[0],
        )

        time_emb_dim = dim[0]
        self.time_to_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.time_emb_proj = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, d)) for d in dim])

        # =================================================================================== #

        self.up_block1 = UpBlock(1 * last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlock(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlock(2 * (last_dim // 4), last_dim // 8, dim[0])
        self.up_block4 = nn.ConvTranspose2d(2 * (last_dim // 8), output_channels, kernel_size=4, stride=2, padding=1)

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            apply_spectral_norm(self)

        if self.use_post_block:
            # freeze base model weights before postblock init
            if "skebs" in post_conf.keys():
                if post_conf["skebs"].get("activate", False) and post_conf["skebs"].get(
                    "freeze_base_model_weights", False
                ):
                    logger.warning("freezing all base model weights due to skebs config")
                    for param in self.parameters():
                        param.requires_grad = False

            logger.info("using postblock")
            self.postblock = PostBlock(post_conf)

    def forward(self, x, t, x_self_cond=False, x_cond=None):
        x_copy = None

        if self.self_condition:
            # input_channels = self.channels * self.levels + self.surface_channels + self.input_only_channels
            # input_channels = self.output_channels
            x_self_cond = default(
                x_self_cond,
                torch.zeros(x.shape[0], self.output_channels, x.shape[2], x.shape[3], x.shape[4], device=x.device),
            )
            x = torch.cat((x_self_cond[:, : self.output_channels], x), dim=1)

        if self.condition:
            x = torch.cat([x, x_cond], dim=1)

        if self.use_post_block:
            x_copy = x.clone().detach()

        if self.use_padding:
            x = self.padding_opt.pad(x)

        t = t.view(-1, 1).float()  # Make sure t is [B, 1]
        t_emb = self.time_to_emb(t)  # [B, time_emb_dim]

        # Project time embedding for each level of transformer
        t_emb_levels = [proj(t_emb) for proj in self.time_emb_proj]  # List of [B, dim[i]] per level

        # Patch embedding or 3D temporal pooling
        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        elif self.frames > 1:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
        else:
            x = x.squeeze(2)

        # Time embedding
        t = t[:, None]  # [B] -> [B, 1]
        t_emb = self.time_to_emb(t)  # [B, model_dim]

        encodings = []
        for (cross_embed_layer, transformer_layer), t_emb_proj in zip(self.layers, t_emb_levels):
            # Apply cross embedding layer (convolution)
            x = cross_embed_layer(x)  # Shape: [B, dim_out, H', W']

            # Add time embedding to the transformer
            # The time embedding is added to the transformer layers (through the cross attention mechanism)
            x = transformer_layer(x) + t_emb_proj.unsqueeze(-1).unsqueeze(
                -1
            )  # Broadcast time embedding over spatial dimensions
            encodings.append(x)

        # Decoder (Upsampling blocks + skip connections), all with t_emb
        x = self.up_block1(x, t_emb)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x, t_emb)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x, t_emb)
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding_opt.unpad(x)

        if self.use_interp:
            x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

        x = x.unsqueeze(2)

        if self.use_post_block:
            x = {
                "y_pred": x,
                "x": x_copy,
            }
            x = self.postblock(x)

        return x


# class CrossFormerDiffusion(CrossFormer):
#     def __init__(self, self_condition, condition, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.pre_out_dim = kwargs.get("surface_channels") + (
#             kwargs.get("channels") * kwargs.get("levels")
#         )  ## channels=total number of output vars out of the wxformer when input+condition is in line.

#         self.dim = kwargs.get("dim", (64, 128, 256, 512))  # Default value as in CrossFormer
#         self.self_condition = self_condition
#         self.condition = condition

#         if self.self_condition and self.condition:
#             logging.warning("Both self-conditioning and standard conditioning on the manifold via x are not simultanously supported. Exiting")
#             sys.exit(0)

#         # Adding timestep embedding layer for diffusion
#         self.time_mlp = nn.Sequential(nn.Linear(1, self.dim[0]), nn.SiLU(), nn.Linear(self.dim[0], self.dim[-1]))

#         self.final_conv = nn.Conv3d(
#             self.pre_out_dim, self.output_channels, 1
#         )  # used to ensure that only noise channels are left at the end; channels=total number of output vars.

#     def forward(self, x, timestep, x_self_cond=False, x_cond=None):
#         x_copy = None

#         if self.self_condition:
#             # input_channels = self.channels * self.levels + self.surface_channels + self.input_only_channels
#             # input_channels = self.output_channels
#             x_self_cond = default(
#                 x_self_cond,
#                 torch.zeros(x.shape[0], self.output_channels, x.shape[2], x.shape[3], x.shape[4], device=x.device)
#             )
#             x = torch.cat((x_self_cond[:, :self.output_channels], x), dim=1)

#         if self.condition:
#             x = torch.cat([x, x_cond], dim = 1)

#         if self.use_post_block:
#             x_copy = x.clone().detach()

#         if self.use_padding:
#             x = self.padding_opt.pad(x)

#         if self.patch_width > 1 and self.patch_height > 1:
#             x = self.cube_embedding(x)
#         elif self.frames > 1:
#             x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)
#         else:
#             x = x.squeeze(2)

#         encodings = []
#         for cel, transformer in self.layers:
#             x = cel(x)
#             x = transformer(x)
#             encodings.append(x)

#         # Add timestep embedding to the feature maps
#         t_embed = self.time_mlp(timestep.view(-1, 1).float())  # (B, dim[0])
#         t_embed = t_embed[:, :, None, None]  # Reshape to (B, dim[0], 1, 1)
#         t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to (B, dim[0], H, W)
#         x = x + t_embed

#         x = self.up_block1(x)
#         x = torch.cat([x, encodings[2]], dim=1)
#         x = self.up_block2(x)
#         x = torch.cat([x, encodings[1]], dim=1)
#         x = self.up_block3(x)
#         x = torch.cat([x, encodings[0]], dim=1)
#         x = self.up_block4(x)

#         if self.use_padding:
#             x = self.padding_opt.unpad(x)

#         if self.use_interp:
#             x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")

#         x = x.unsqueeze(2)

#         x = self.final_conv(x)

#         if self.use_post_block:
#             x = {"y_pred": x, "x": x_copy}
#             x = self.postblock(x)

#         return x


def create_model(config, self_condition=True):
    """Initialize and return the CrossFormer model using a config dictionary."""
    return CrossFormerDiffusion(**config).to("cuda")


def create_diffusion(model, config):
    """Initialize and return the Gaussian Diffusion process."""
    return ModifiedGaussianDiffusion(model, **config)


if __name__ == "__main__":
    crossformer_config = {
        "type": "crossformer",
        "frames": 1,  # Number of input states
        "image_height": 192,  # Number of latitude grids
        "image_width": 288,  # Number of longitude grids
        "levels": 16,  # Number of upper-air variable levels
        "channels": 4,  # Upper-air variable channels
        "surface_channels": 7,  # Surface variable channels
        "input_only_channels": 3,  # Dynamic forcing, forcing, static channels
        "output_only_channels": 0,  # Diagnostic variable channels
        "patch_width": 1,  # Number of latitude grids in each 3D patch
        "patch_height": 1,  # Number of longitude grids in each 3D patch
        "frame_patch_size": 1,  # Number of input states in each 3D patch
        "dim": [32, 64, 128, 256],  # Dimensionality of each layer
        "depth": [2, 2, 2, 2],  # Depth of each layer
        "global_window_size": [8, 4, 2, 1],  # Global window size for each layer
        "local_window_size": 8,  # Local window size
        "cross_embed_kernel_sizes": [  # Kernel sizes for cross-embedding
            [4, 8, 16, 32],
            [2, 4],
            [2, 4],
            [2, 4],
        ],
        "cross_embed_strides": [2, 2, 2, 2],  # Strides for cross-embedding
        "attn_dropout": 0.0,  # Dropout probability for attention layers
        "ff_dropout": 0.0,  # Dropout probability for feed-forward layers
        "use_spectral_norm": True,  # Whether to use spectral normalization
        "padding_conf": {
            "activate": True,
            "mode": "earth",
            "pad_lat": [32, 32],
            "pad_lon": [48, 48],
        },
        "interp": False,
        "condition": True,
        "self_condition": False,
        "pretrained_weights": "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/model_levels/single_step/checkpoint.pt",
    }

    diffusion_config = {
        "image_size": (192, 288),
        "timesteps": 1000,
        "sampling_timesteps": 1000,
        "objective": "pred_v",
        "beta_schedule": "linear",
        "schedule_fn_kwargs": dict(),
        "ddim_sampling_eta": 0.0,
        "auto_normalize": True,
        "offset_noise_strength": 0.0,
        "min_snr_loss_weight": False,
        "min_snr_gamma": 5,
        "immiscible": False,
    }

    crossformer_config["diffusion"] = diffusion_config

    model = create_model(crossformer_config).to("cuda")
    diffusion = create_diffusion(model, diffusion_config).to("cuda")

    conditional_tensor = torch.randn(
        1,
        crossformer_config["channels"] * crossformer_config["levels"]
        + crossformer_config["surface_channels"]
        + crossformer_config["input_only_channels"],
        crossformer_config["frames"],
        crossformer_config["image_height"],
        crossformer_config["image_width"],
    ).to("cuda")

    noise_tensor = torch.randn(
        1,
        crossformer_config["channels"] * crossformer_config["levels"]
        + crossformer_config["surface_channels"]
        + crossformer_config["output_only_channels"],
        crossformer_config["frames"],
        crossformer_config["image_height"],
        crossformer_config["image_width"],
    ).to("cuda")

    print(conditional_tensor.shape)

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
