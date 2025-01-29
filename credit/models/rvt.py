from torch import nn, einsum

import os
import torch
import torch.fft
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import torch.nn.functional as F

from credit.pe import SurfacePosEmb2D
from credit.models.base_model import BaseModel
from vector_quantize_pytorch import VectorQuantize
import logging
import xarray as xr
from rotary_embedding_torch import RotaryEmbedding


logger = logging.getLogger(__name__)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


# helper classes


class SpatialConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, bias=False):
        super().__init__()
        self.conv = DepthWiseConv2d(
            dim_in, dim_out, kernel, padding=kernel // 2, bias=False
        )  # lucid rains padding
        self.cls_proj = (
            nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x, fmap_dims):
        cls_token, x = x[:, :1], x[:, 1:]
        x = rearrange(x, "b (h w) d -> b d h w", **fmap_dims)
        x = self.conv(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        cls_token = self.cls_proj(cls_token)
        return torch.cat((cls_token, x), dim=1)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_rotary=True,
        use_ds_conv=True,
        conv_query_kernel=5,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.use_ds_conv = use_ds_conv

        self.to_q = (
            SpatialConv(dim, inner_dim, conv_query_kernel, bias=False)
            if use_ds_conv
            else nn.Linear(dim, inner_dim, bias=False)
        )

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.pos_emb = RotaryEmbedding(dim=dim_head, freqs_for="pixel", max_freq=1280)

    def forward(self, x, fmap_dims):
        _, _, _, h = *x.shape, self.heads

        to_q_kwargs = {"fmap_dims": fmap_dims} if self.use_ds_conv else {}

        x = self.norm(x)

        q = self.to_q(x, **to_q_kwargs)

        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), qkv)

        q = self.pos_emb.rotate_queries_or_keys(q)
        k = self.pos_emb.rotate_queries_or_keys(k)

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        image_size,
        dropout=0.0,
        use_rotary=True,
        use_ds_conv=True,
        use_glu=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_rotary=use_rotary,
                            use_ds_conv=use_ds_conv,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                    ]
                )
            )

    def forward(self, x, fmap_dims):
        for attn, ff in self.layers:
            x = attn(x, fmap_dims=fmap_dims) + x
            x = ff(x) + x
        return x


class PatchDropout(torch.nn.Module):
    # https://github.com/yueliukth/PatchDropout/blob/main/scripts/patchdropout.py
    """
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    """

    def __init__(self, keep_rate=0.5, sampling="uniform", token_shuffling=False):
        super().__init__()
        assert 0 < keep_rate <= 1, "The keep_rate must be in (0,1]"

        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop:
            return x
        if self.keep_rate == 1:
            return x

        # batch, length, dim
        N, L, D = x.shape

        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x, patch_mask

    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        else:
            return NotImplementedError(
                f"PatchDropout does ot support {self.sampling} sampling"
            )

    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L - 1  # patch lenght (without CLS)

        keep = int(_L * self.keep_rate)
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask


# Rotary Vision Transformer


class RViT(BaseModel):
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
        emb_dropout=0.0,
        use_rotary=True,
        use_ds_conv=True,
        use_glu=True,
        use_decoder_conv_layers=False,
        use_cls_tokens=False,
        use_registers=False,
        num_register_tokens=0,
        token_dropout=0.0,
        use_codebook=False,
        vq_codebook_dim=32,
        vq_codebook_size=128,
        vq_decay=0.1,
        vq_commitment_weight=1.0,
        vq_kmeans_init=True,
        vq_use_cosine_sim=True,
        rk4_integration=False,
        use_visual_ssl=False,  # not implemented
        visual_ssl_weight=0.0,  # not implemented
        transformer_type="lucidrains",
        static_variables=None,
    ):
        super().__init__()

        self.channels = channels
        self.surface_channels = surface_channels
        self.frames = frames
        self.use_codebook = use_codebook
        self.use_cls_tokens = use_cls_tokens
        self.rk4_integration = rk4_integration
        self.use_decoder_conv_layers = use_decoder_conv_layers
        self.static_variables = static_variables

        # Encoder-decoder layers

        self.transformer_encoder = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            mlp_dim=mlp_dim,
            image_size=image_width,
            dropout=dropout,
            use_rotary=use_rotary,
            use_ds_conv=use_ds_conv,
            use_glu=use_glu,
        )
        self.transformer_decoder = Transformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            mlp_dim=mlp_dim,
            image_size=image_width,
            dropout=dropout,
            use_rotary=use_rotary,
            use_ds_conv=use_ds_conv,
            use_glu=use_glu,
        )

        # Input/output dimensions
        self.num_patches = int(
            (image_height // patch_height) * (image_width // patch_width)
        )
        self.fmap_dims = {
            "h": image_height // patch_height,
            "w": image_width // patch_width,
        }
        input_channels = channels * frames + surface_channels
        input_dim = input_channels * patch_height * patch_width

        # Encoder layers
        self.encoder_embed = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width
        )

        if self.static_variables is not None:
            self.encoder_linear = nn.Linear(
                (len(self.static_variables) + input_channels)
                * patch_height
                * patch_width,
                dim,
            )
        else:
            self.encoder_linear = nn.Linear(input_dim, dim)
        self.encoder_layer_norm = nn.LayerNorm(dim)

        # Decoder layers
        self.decoder_linear_1 = nn.Linear(dim, dim * 4)
        self.decoder_linear_2 = nn.Linear(dim * 4, input_dim)
        self.decoder_layer_norm_1 = nn.LayerNorm(4 * dim)
        self.decoder_layer_norm_2 = nn.LayerNorm(input_dim)
        self.decoder_rearrange = Rearrange(
            "b (h w) (p1 p2 c) -> b c (p1 h) (w p2)",
            h=(image_height // patch_height),
            w=(image_width // patch_width),
            p1=patch_height,
            p2=patch_width,
        )

        # Positional embeddings
        self.pos_embedding_enc = SurfacePosEmb2D(
            image_height,
            image_width,
            patch_height,
            patch_width,
            dim,
            cls_token=self.use_cls_tokens,
        )
        self.pos_embedding_dec = SurfacePosEmb2D(
            image_height,
            image_width,
            patch_height,
            patch_width,
            dim,
            cls_token=self.use_cls_tokens,
        )

        # Conv smoothing layer for decoder
        if self.use_decoder_conv_layers:
            self.conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                padding=1,
            )

        # CLS paramters
        if self.use_cls_tokens:
            self.cls_token_enc = nn.Parameter(torch.randn(1, 1, dim))
            self.cls_token_dec = nn.Parameter(torch.randn(1, 1, dim))

        # Token / patch drop
        self.token_dropout_prob = token_dropout
        self.token_dropout = (
            PatchDropout(1.0 - token_dropout) if token_dropout > 0.0 else nn.Identity()
        )

        # Vision Transformers Need Registers, https://arxiv.org/abs/2309.16588
        self.use_registers = use_registers
        if self.use_registers:
            self.register_tokens_enc = nn.Parameter(
                torch.randn(num_register_tokens, dim)
            )
            self.register_tokens_dec = nn.Parameter(
                torch.randn(num_register_tokens, dim)
            )

        # codebook
        self.use_codebook = use_codebook
        if self.use_codebook:
            self.vq = VectorQuantize(
                dim=vq_codebook_dim,
                codebook_size=vq_codebook_size,  # codebook size
                decay=vq_decay,  # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight=vq_commitment_weight,  # the weight on the commitment loss
                kmeans_init=vq_kmeans_init,
                use_cosine_sim=vq_use_cosine_sim,
            )

        if self.static_variables is not None:
            self.static_vars = (
                torch.from_numpy(
                    xr.open_dataset(self.static_variables["cos_lat"])["coslat"].values
                )
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        logger.info("... loaded a ViT with a rotary embedding in each attention layer")

    def encode(self, x):
        # encode
        x = self.encoder_embed(x)
        x = self.encoder_linear(x)
        x = self.encoder_layer_norm(x)

        # add encoder CLS tokens
        if self.use_cls_tokens:
            cls_token = self.cls_token_enc.repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)

        # Add PE
        x = self.pos_embedding_enc(x)

        if self.token_dropout_prob > 0.0 and self.training:
            x, pred_mask = self.token_dropout(x)
        else:
            pred_mask = None

        if self.use_registers:
            r = repeat(self.register_tokens_enc, "n d -> b n d", b=x.shape[0])
            x, ps = pack([x, r], "b * d")

        x = self.transformer_encoder(x, fmap_dims=self.fmap_dims)

        if self.use_registers:
            x, _ = unpack(x, ps, "b * d")

        # excise CLS tokens
        if self.use_cls_tokens:
            x = x[:, 1:]
            pred_mask = pred_mask[:, 1:] if pred_mask is not None else pred_mask

        return x, pred_mask

    def decode(self, x, patch_mask=None):
        if patch_mask is not None:
            # create a tensor of zeros with the original shape
            x_zeros = torch.zeros(
                x.shape[0],
                self.num_patches + 1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            # scatter the original patches back into the tensor of zeros
            x_zeros.scatter_(1, patch_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]), x)
            # replace x with the new tensor that has zero tokens inserted
            x = x_zeros[:, 1:]  # ignore the CLS-token placeholder dimension

        # add decoder CLS tokens
        if self.use_cls_tokens:
            cls_token = self.cls_token_dec.repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)

        # Add PE
        x = self.pos_embedding_dec(x)

        if self.use_registers:
            r = repeat(self.register_tokens_dec, "n d -> b n d", b=x.shape[0])
            x, ps = pack([x, r], "b * d")
        x = self.transformer_decoder(x, fmap_dims=self.fmap_dims)
        if self.use_registers:
            x, _ = unpack(x, ps, "b * d")

        x = self.decoder_linear_1(x)
        # x = F.tanh(x)
        x = self.decoder_layer_norm_1(x)
        x = self.decoder_linear_2(x)
        x = self.decoder_layer_norm_2(x)

        # excise CLS tokens
        if self.use_cls_tokens:
            x = x[:, 1:]

        x = self.decoder_rearrange(x)

        # Add a convolutional layer
        if self.use_decoder_conv_layers:
            x = self.conv(x)

        return x

    def codebook(self):
        if self.use_codebook:
            return self.vq.codebook
        return None

    def forward(self, x):
        # add grid here to the inputs
        if self.static_variables is not None:
            x = torch.cat(
                [x, self.static_vars.expand(x.size(0), -1, -1, -1).to(x.device)], dim=1
            )

        if self.rk4_integration:
            x, commit_loss = self.rk4(x)
            if self.use_codebook:
                return x, commit_loss
            else:
                return x

        z, mask = self.encode(x)

        if self.use_codebook:
            z, indices, commit_loss = self.vq(z)
            x = self.decode(z, mask)
            return x, commit_loss

        x = self.decode(z, mask)

        return x

    def rk4(self, x):
        total_commit_loss = 0

        def integrate_step(x, k, factor):
            if self.use_codebook:
                z, m = self.encode(x + k * factor)
                z, _, cm_loss = self.vq(z)
                nonlocal total_commit_loss
                total_commit_loss += cm_loss
                result = self.decode(z, m)
            else:
                z, m = self.encode(x + k * factor)
                result = self.decode(z, m)
            return result

        z, mask = self.encode(x)
        k1 = self.decode(z, mask)
        k2 = integrate_step(x, k1, 0.5)
        k3 = integrate_step(x, k2, 0.5)
        k4 = integrate_step(x, k3, 1.0)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6, total_commit_loss


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

    input_tensor = torch.randn(
        1, channels * frames + surface_channels, image_height, image_width
    )

    model = RViT(
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

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=0, world_size=1)

    wrapped_model = FSDP(model)

    y_pred = wrapped_model(input_tensor.to("cuda"))

    print("Predicted shape:", y_pred.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
