import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import os
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import logging


# helpers


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# cube embedding

class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, T, C, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x.squeeze(2)


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


# cross embed layer

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride=2
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


# dynamic positional bias

def DynamicPositionBias(dim):
    return nn.Sequential(
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
        Rearrange('... () -> ...')
    )


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


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        attn_type,
        window_size,
        dim_head=32,
        dropout=0.
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
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
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        *_, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

        # prenorm

        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == 'short':
            x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1=wsz, s2=wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1=wsz, l2=wsz)

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add dynamic positional bias

        pos = torch.arange(-wsz, wsz + 1, device=device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=wsz, y=wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h=height // wsz, w=width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h=height // wsz, w=width // wsz)

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
        attn_dropout=0.,
        ff_dropout=0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_type='short', window_size=local_window_size, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
                Attention(dim, attn_type='long', window_size=global_window_size, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


# classes

class CrossFormer(nn.Module):
    def __init__(
        self,
        image_height=640,
        patch_height=2,
        image_width=1280,
        patch_width=2,
        frames=2,
        frame_patch_size=2,
        channels=4,
        surface_channels=7,
        levels=15,
        dim=(64, 128, 256, 512),
        depth=(2, 2, 8, 2),
        global_window_size=(10, 5, 2, 1),
        local_window_size=10,
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(4, 2, 2, 2),
        attn_dropout=0.,
        ff_dropout=0.,
    ):
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
        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels
        input_channels = channels * levels + surface_channels

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
        first_dim = input_channels if (patch_height == 1 and patch_width == 1) else dim[0]
        dims = [first_dim, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        self.layers = nn.ModuleList([])

        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
            self.layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            ]))

        self.cube_embedding = CubeEmbedding(
            (frames, image_height, image_width),
            (frames, patch_height, patch_width),
            input_channels,
            dim[0]
        )

        self.up = UpBlock(last_dim, last_dim // 2, dim[0])
        self.fc = nn.Linear(last_dim // 2, input_channels)

    def forward(self, x):

        if self.patch_width > 1 and self.patch_height > 1:
            x = self.cube_embedding(x)
        else:
            x = F.avg_pool3d(x, kernel_size=(2, 1, 1)).squeeze(2)

        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
        x = self.up(x).permute(0, 2, 3, 1)
        x = self.fc(x).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(self.image_height, self.image_width), mode="bilinear")
        return x.unsqueeze(2)

    def concat_and_reshape(self, x1, x2):
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3], x1.shape[4], x1.shape[5])
        x_concat = torch.cat((x1, x2), dim=2)
        return x_concat.permute(0, 2, 1, 3, 4)

    def split_and_reshape(self, tensor):
        tensor1 = tensor[:, :int(self.channels * self.levels), :, :, :]
        tensor2 = tensor[:, -int(self.surface_channels):, :, :, :]
        tensor1 = tensor1.view(tensor1.shape[0], channels, self.levels, tensor1.shape[2], tensor1.shape[3], tensor1.shape[4])
        return tensor1, tensor2

    @classmethod
    def load_model(cls, conf):
        save_loc = conf['save_loc']
        ckpt = os.path.join(f"{save_loc}", "checkpoint.pt")

        if conf["trainer"]["mode"] == "ddp":
            if not os.path.isfile(ckpt):
                ckpt = os.path.join(f"{save_loc}", "checkpoint_cuda:0.pt")

        if not os.path.isfile(ckpt):
            raise ValueError(
                "No saved checkpoint exists. You must train a model first. Exiting."
            )

        logging.info(
            f"Loading a model with pre-trained weights from path {ckpt}"
        )

        checkpoint = torch.load(ckpt)

        if "type" in conf["model"]:
            del conf["model"]["type"]

        model_class = cls(**conf["model"])

        if conf["trainer"]["mode"] == "fsdp":
            FSDP.set_state_dict_type(
                model_class,
                StateDictType.SHARDED_STATE_DICT,
            )
            state_dict = {
                "model_state_dict": model_class.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model_class.load_state_dict(state_dict["model_state_dict"])
        else:
            model_class.load_state_dict(checkpoint["model_state_dict"])

        return model_class

    def save_model(self, conf):
        save_loc = conf['save_loc']
        state_dict = {
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, f"{save_loc}/checkpoint.pt")


if __name__ == "__main__":
    image_height = 640  # 640
    image_width = 1280  # 1280
    levels = 15
    frames = 2
    channels = 4
    surface_channels = 7
    patch_height = 2
    patch_width = 2
    frame_patch_size = 2

    input_tensor = torch.randn(2, channels * levels + surface_channels, frames, image_height, image_width).to("cuda")

    model = CrossFormer(
        image_height=image_height,
        patch_height=patch_height,
        image_width=image_width,
        patch_width=patch_width,
        frames=frames,
        frame_patch_size=frame_patch_size,
        channels=channels,
        surface_channels=surface_channels,
        levels=levels,
        dim=(64, 128, 256, 512),
        depth=(2, 2, 8, 2),
        global_window_size=(5, 5, 2, 1),
        local_window_size=10,
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(4, 2, 2, 2),
        attn_dropout=0.,
        ff_dropout=0.,
    ).to("cuda")

    y_pred = model(input_tensor.to("cuda"))

    print("Predicted shape:", y_pred.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")