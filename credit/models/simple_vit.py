import os
import logging

import torch
import torch.fft
import torch.nn.functional as F
from torch import nn

import xarray as xr

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from credit.pe import SurfacePosEmb2D
from credit.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., rotary_emb=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rotary_emb = rotary_emb

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.dr = dropout

        if self.rotary_emb is not None:
            self.rotary_emb = RotaryEmbedding(dim=dim, freqs_for="pixel", max_freq=1280)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = self.attend(dots)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)

        with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True  # should be false but bug somewhere on my end
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dr
            )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, rotary_emb=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rotary_emb=rotary_emb),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(BaseModel):
    def __init__(
            self,
            image_height,
            patch_height,
            image_width,
            patch_width,
            frames,
            frame_patch_size,
            dim=32,
            channels=60,
            surface_channels=7,
            depth=4,
            heads=8,
            dim_head=32,
            mlp_dim=32,
            dropout=0.0,
            use_last_conv_layers=False,
            use_rotary=True,
            use_registers=False,
            num_register_tokens=0,
            static_variables=None
    ):

        super().__init__()

        self.channels = channels
        self.surface_channels = surface_channels
        self.frames = frames
        self.use_last_conv_layers = use_last_conv_layers
        self.static_variables = static_variables

        # Encoder-decoder layers

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            rotary_emb=use_rotary
        )

        # Input/output dimensions
        self.num_patches = int((image_height // patch_height) * (image_width // patch_width))
        input_channels = channels * frames + surface_channels
        input_dim = input_channels * patch_height * patch_width

        # Encoder layers
        self.encoder_embed = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                       p1=patch_height,
                                       p2=patch_width)

        if self.static_variables is not None:
            self.encoder_linear = nn.Linear(
                (len(self.static_variables) + input_channels) * patch_height * patch_width,
                dim
            )
        else:
            self.encoder_linear = nn.Linear(input_dim, dim)
        self.encoder_layer_norm = nn.LayerNorm(dim)

        # Decoder layers
        self.decoder_linear = nn.Linear(dim, input_dim)
        self.decoder_rearrange = Rearrange('b (h w) (p1 p2 c) -> b c (p1 h) (w p2)',
                                           h=(image_height // patch_height),
                                           w=(image_width // patch_width),
                                           p1=patch_height,
                                           p2=patch_width)

        # Positional embeddings
        # self.pos_embedding_enc = TokenizationAggregation(channels, patch_height, patch_width, dim)
        self.pos_embedding_enc = SurfacePosEmb2D(
            image_height, image_width, patch_height, patch_width, dim, cls_token=False
        )

        # Conv smoothing layer for decoder
        if self.use_last_conv_layers:
            self.conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1
            )

        # Vision Transformers Need Registers, https://arxiv.org/abs/2309.16588
        self.use_registers = use_registers
        if self.use_registers:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # static inputs

        if self.static_variables is not None:
            self.static_vars = torch.from_numpy(
                xr.open_dataset(self.static_variables["cos_lat"])["coslat"].values
            ).float().unsqueeze(0).unsqueeze(0)

        logger.info(f"... loaded a simple ViT. Rotary embedding: {use_rotary}")

    def forward(self, x):

        # add grid here to the inputs
        if self.static_variables is not None:
            x = torch.cat([x, self.static_vars.expand(x.size(0), -1, -1, -1).to(x.device)], dim=1)

        # encode
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)
        x = self.encoder_layer_norm(x)

        # Add PE
        x = self.pos_embedding_enc(x)

        if self.use_registers:
            r = repeat(self.register_tokens_enc, 'n d -> b n d', b=x.shape[0])
            x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        if self.use_registers:
            x, _ = unpack(x, ps, 'b * d')

        x = self.decoder_linear(x)
        x = self.decoder_rearrange(x)

        # Add a convolutional layer
        if self.use_last_conv_layers:
            x = self.conv(x)

        return x


if __name__ == "__main__":
    image_height = 640  # 640
    patch_height = 64
    image_width = 1280  # 1280
    patch_width = 64
    frames = 15
    frame_patch_size = 3

    channels = 4
    surface_channels = 7
    dim = 64
    dim_head = 64
    mlp_dim = 64
    heads = 4
    depth = 2

    input_tensor = torch.randn(1, channels * frames + surface_channels, image_height, image_width)

    model = SimpleViT(
        image_height,
        patch_height,
        image_width,
        patch_width,
        frames,
        frame_patch_size,
        dim,
        channels,
        surface_channels,
        depth,
        heads,
        dim_head,
        mlp_dim,
    ).to("cuda")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=0, world_size=1)

    wrapped_model = FSDP(model, use_orig_params=True)

    y_pred = wrapped_model(input_tensor.to("cuda"))

    print("Predicted shape:", y_pred.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
