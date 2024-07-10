import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
import logging

from credit.models.base_model import BaseModel


logger = logging.getLogger(__name__)


def apply_spectral_norm(model):
    '''
    add spectral norm to all the conv and linear layers
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


def circular_pad1d(x, pad):
    '''
    Repeat the edge the map to fill the padded area
    '''
    return torch.cat((x[..., -pad:], x, x[..., :pad]), dim=-1)


def get_pad3d(input_resolution, window_size):
    """
    Estimate the size of padding based on the given window size and the original input size.

    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()

        # input size
        self.img_size = img_size

        # number of patches after embedding (T_num, Lat_num, Lon_num)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.patches_resolution = patches_resolution

        # number of embedded dimension after patching
        self.embed_dim = embed_dim

        # Conv3d-based patching
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # layer norm
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):

        # example size: [Batch, 67, 2, 640, 1280]
        B, T, C, Lat, Lon = x.shape

        # Conv3d-based patching and embedding
        # output size: [B, 1024, 1, 40, 80]
        x = self.proj(x)

        # combine T, Lat, Lon dimensions
        # output size: [B, 1024, 3200]
        x = x.reshape(B, self.embed_dim, -1)

        # switch to channel-last for normalization
        # output size: [B, 3200, 1024]
        x = x.transpose(1, 2)  # B T*Lat*Lon C

        # Layer norm (channel last)
        if self.norm is not None:
            x = self.norm(x)

        # switch back to channel first
        # output size: [B, 1024, 3200]
        x = x.transpose(1, 2)

        # recover T, Lat, Lon dimensions
        # output size: [B, 1024, 1, 40, 80]
        x = x.reshape(B, self.embed_dim, *self.patches_resolution)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()

        # down-sampling with Conv2d
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        # blocks of residual path
        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

    def forward(self, x):

        # down-sampling
        x = self.conv(x)

        # skip-connection
        shortcut = x

        # residual path
        x = self.b(x)

        # additive residual connection
        return x + shortcut


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()

        # down-sampling with Transpose Conv
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        # blocks of residual path
        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())
        self.b = nn.Sequential(*blk)

    def forward(self, x):
        # up-sampling
        x = self.conv(x)

        # skip-connection
        shortcut = x

        # residual path
        x = self.b(x)

        # additive residual connection
        return x + shortcut


class UTransformer(nn.Module):
    """U-Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        num_groups (int | tuple[int]): number of groups to separate the channels into.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """
    def __init__(self, embed_dim, num_groups, input_resolution, num_heads, window_size, depth):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)  # convert window_size[int] to tuple

        # padding input tensors so they are divided by the window size
        padding = get_pad2d(input_resolution, window_size)  # <--- Accepts tuple only
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)

        # input resolution after padding
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right

        # down-sampling block
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])

        # SwinT block
        self.layer = SwinTransformerV2Stage(embed_dim,
                                            embed_dim,
                                            input_resolution,
                                            depth, num_heads,
                                            window_size[0])  # <--- window_size[0] get window_size[int] from tuple

        # up-sampling block
        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.down(x)
        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop
        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon
        x = self.up(x)
        return x


class Fuxi(BaseModel):
    """
    Args:
        img_size (Sequence[int], optional): T, Lat, Lon.
        patch_size (Sequence[int], optional): T, Lat, Lon.
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """
    def __init__(self,
                 image_height=640,  # 640
                 patch_height=16,
                 image_width=1280,  # 1280
                 patch_width=16,
                 levels=15,
                 frames=2,
                 frame_patch_size=2,
                 dim=1536,
                 num_groups=32,
                 channels=4,
                 surface_channels=7,
                 static_channels=0,
                 diagnostic_channels=0,
                 num_heads=8,
                 depth=48,
                 window_size=7,
                 pad_lon=80,
                 pad_lat=80,
                 use_spectral_norm=False):

        super().__init__()
        # input tensor size (time, lat, lon)
        img_size = (frames, image_height + 2 * pad_lat, image_width + 2 * pad_lon)

        # the size of embedded patches
        patch_size = (frame_patch_size, patch_height, patch_width)

        # number of channels = levels * varibales per level + surface variables
        #in_chans = out_chans = levels * channels + surface_channels
        
        in_chans = channels * levels + surface_channels + static_channels
        out_chans = channels * levels + surface_channels + diagnostic_channels
        
        # input resolution = number of embedded patches / 2
        # divide by two because "u_trasnformer" has a down-sampling block
        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)

        # FuXi cube embedding layer
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, dim)

        # Downsampling --> SwinTransformerV2 stacks --> Upsampling
        self.u_transformer = UTransformer(dim, num_groups, input_resolution, num_heads, window_size, depth=depth)

        # dense layer applied on channel dmension
        # channel * patch_size beucase dense layer recovers embedded dimensions to the input dimensions
        self.fc = nn.Linear(dim, out_chans * patch_size[1] * patch_size[2])

        # Hyperparameters
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size

        self.channels = channels
        self.surface_channels = surface_channels
        self.levels = levels

        self.pad_lon = pad_lon
        self.pad_lat = pad_lat
        self.use_spectral_norm = use_spectral_norm

        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Move the model to the device
            self.to(device)
            apply_spectral_norm(self)

        if self.pad_lon > 0:
            logger.info(f"Padding each longitudinal boundary with {self.pad_lon} pixels from the other side")
        if self.pad_lat > 0:
            logger.info(f"Padding each pole using a reflection with {self.pad_lat} pixels")

    def forward(self, x: torch.Tensor):

        if self.pad_lon > 0:
            x = circular_pad1d(x, pad=self.pad_lon)

        if self.pad_lat > 0:
            x_shape = x.shape

            # Reshape the tensor to (B, C*2, lat, lon) using the values from x_shape
            x = torch.reshape(x, (x_shape[0], x_shape[1] * x_shape[2], x_shape[3], x_shape[4]))

            # Pad the tensor with reflect mode
            x = F.pad(x, (0, 0, self.pad_lat, self.pad_lat), mode='reflect')

            # Reshape the tensor back to (B, C, 2, new lat, lon) using the values from x_shape
            x = torch.reshape(x, (x_shape[0], x_shape[1], x_shape[2], x_shape[3] + 2 * self.pad_lat, x_shape[4]))

        # Tensor dims: Batch, Variables, Time, Lat grids, Lon grids
        B, _, _, _, _ = x.shape

        _, patch_lat, patch_lon = self.patch_size

        # Get the number of patches after embedding
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2

        # Cube Embedding and squeese the time dimension
        # (the model produce single forecast lead time only)

        # x: input size = (Batch, Variables, Time, Lat grids, Lon grids)
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        # x: output size = (Batch, Embedded dimension, time, number of patches, number of patches)

        # u_transformer stage
        # the size of x does notchange
        x = self.u_transformer(x)

        # recover embeddings to lat/lon grids with dense layer and reshape operation.
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C
        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        img_size = list(self.img_size)
        if self.pad_lon > 0:
            # Slice to original size
            x = x[..., self.pad_lon:-self.pad_lon]
            img_size[2] -= 2 * self.pad_lon

        if self.pad_lat > 0:
            # Slice to original size
            x = x[..., self.pad_lat:-self.pad_lat, :]
            img_size[1] -= 2 * self.pad_lat

        # bilinear interpolation
        # if lat/lon grids (i.e., img_size) cannot be divided by the patche size completely
        # this will preserve the output size
        x = F.interpolate(x, size=img_size[1:], mode="bilinear")

        # unfold the time dimension
        return x.unsqueeze(2)


if __name__ == "__main__":

    # ============================================================= #
    # hyperparam examples
    image_height = 640    # Image height (default: 640)
    patch_height = 4     # Patch height (default: 16)
    image_width = 1280    # Image width (default: 1280)
    patch_width = 4      # Patch width (default: 16)
    levels = 15           # Number of levels (default: 15)
    frames = 2            # Number of frames (default: 2)
    frame_patch_size = 2  # Frame patch size (default: 2)
    dim = 1024            # Dimension (default: 1536)
    num_groups = 32       # Number of groups (default: 32)
    channels = 4          # Channels (default: 4)
    surface_channels = 7  # Surface channels (default: 7)
    static_channels = 2
    diagnostic_channels = 0
    num_heads = 8         # Number of heads (default: 8)
    window_size = 7       # Window size (default: 7)
    depth = 8            # Depth of the swin transformer (default: 48)
    use_spectral_norm = True
    pad_lon = 80
    pad_lat = 80
    
    # ============================================================= #
    # build the model
    img_size = (frames, image_height, image_width)
    patch_size = (frames, patch_height, patch_width)
    
    model = Fuxi(
        channels=channels,
        surface_channels=surface_channels,
        static_channels=static_channels,
        diagnostic_channels=diagnostic_channels,
        levels=levels,
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        patch_height=patch_height,
        patch_width=patch_width,
        frame_patch_size=frame_patch_size,
        dim=dim,
        pad_lat=pad_lat,
        pad_lon=pad_lon,
        use_spectral_norm=use_spectral_norm
    ).to("cuda")

    # ============================================================= #
    # test the model
    
    # pass an input tensor to test the graph
    input_tensor = torch.randn(2, channels * levels + surface_channels + static_channels, 
                               frames, image_height, image_width).to("cuda")    
    
    y_pred = model(input_tensor.to("cuda"))

    print('Input shape: {}'.format(input_tensor.shape))
    print("Predicted shape: {}".format(y_pred.shape))

    # print the number of params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
