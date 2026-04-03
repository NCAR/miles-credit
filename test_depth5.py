import sys

sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")
from credit.models.wxformer.wxformer_v2 import CrossFormer

configs = {
    "depth-4 (current)": dict(
        dim=(128, 256, 512, 1024),
        depth=(2, 2, 8, 2),
        global_window_size=(10, 5, 2, 1),
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
    ),
    "depth-5 small (128->1024)": dict(
        dim=(64, 128, 256, 512, 1024),
        depth=(2, 2, 6, 2, 2),
        global_window_size=(10, 5, 2, 2, 1),
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(4, 2, 2, 2, 2),  # stride-4 at L0 for 160x320
    ),
    "depth-5 large (128->2048)": dict(
        dim=(128, 256, 512, 1024, 2048),
        depth=(2, 2, 6, 2, 2),
        global_window_size=(10, 5, 2, 2, 1),
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2, 2),
    ),
}

for name, kwargs in configs.items():
    m = CrossFormer(
        image_height=640,
        image_width=1280,
        levels=16,
        channels=4,
        surface_channels=7,
        input_only_channels=3,
        deformable_start_level=1,
        **kwargs,
    )
    p = sum(pp.numel() for pp in m.parameters()) / 1e6
    print(f"{name}: {p:.1f}M params")
print("OK")
