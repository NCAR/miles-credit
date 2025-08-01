import torch
import torch.nn as nn


def load_unet_attention(attention_type, out_chans, reduction=32, spatial_kernel=7):
    """
    Factory function to load different attention mechanisms for UNet.

    Args:
        attention_type (str): Type of attention mechanism
            - 'coordinate': CoordinateAttention - Best for rectangular images like 128x256
            - 'eca': ECABlock - Most efficient for high channel counts
            - 'spatial': LightSpatialAttention - Minimal overhead, spatial focus
            - 'scse_optimized': OptimizedSCSEBlock with depthwise separable convs
            - 'scse_standard': OptimizedSCSEBlock with standard convs
            - 'mixed': EfficientMixedAttention - Balance of channel + spatial
            - None or 'none': No attention
        out_chans (int): Number of output channels
        reduction (int): Reduction ratio for channel attention mechanisms (default: 32)
        spatial_kernel (int): Kernel size for spatial attention (default: 7)

    Returns:
        nn.Module or None: Attention module instance or None

    Raises:
        ValueError: If attention_type is not recognized
    """

    if attention_type is None or attention_type.lower() == "none":
        return None

    attention_type = attention_type.lower()

    if attention_type == "coordinate":
        return CoordinateAttention(out_chans, reduction)
    elif attention_type == "eca":
        return ECABlock(out_chans)
    elif attention_type == "spatial":
        return LightSpatialAttention(spatial_kernel)
    elif attention_type == "scse_optimized":
        return SCSEAttention(out_chans, reduction, use_depthwise=True)
    elif attention_type == "scse_standard":
        return SCSEAttention(out_chans, reduction, use_depthwise=False)
    elif attention_type == "mixed":
        return EfficientMixedAttention(out_chans, reduction, spatial_kernel)
    else:
        available_types = ["coordinate", "eca", "spatial", "scse_optimized", "scse_standard", "mixed", "none"]
        raise ValueError(f"Unknown attention_type '{attention_type}'. Available options: {available_types}")


# Efficient Channel Attention (ECA) - Lightweight and effective for high channels
class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size based on channel dimension
        k = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# Coordinate Attention - Great for rectangular images like 128x256
class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.size()

        # Pool along height and width dimensions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention weights
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return x * a_h * a_w


# Lightweight Spatial Attention - Minimal overhead
class LightSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


# Optimized SCSE - Best variation of SCSE for high channels
class SCSEAttention(nn.Module):
    def __init__(self, channels, reduction=16, use_depthwise=True):
        super().__init__()

        # Optimized Channel Squeeze & Excitation
        # Use smaller reduction ratio and more efficient design
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Optimized Spatial Squeeze & Excitation
        if use_depthwise:
            # Use depthwise separable convolution for efficiency
            self.sse = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),  # Depthwise
                nn.Conv2d(channels, 1, kernel_size=1, bias=False),  # Pointwise
                nn.Sigmoid(),
            )
        else:
            # Standard spatial attention but with efficiency improvements
            self.sse = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        # Channel attention
        cse_out = self.cse(x) * x

        # Spatial attention
        sse_out = self.sse(x) * x

        # Combine both attentions
        return cse_out + sse_out


# Efficient Mixed Attention - Balance of channel and spatial
class EfficientMixedAttention(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        # Lightweight channel attention
        self.channel_attention = ECABlock(channels)
        # Lightweight spatial attention
        self.spatial_attention = LightSpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
