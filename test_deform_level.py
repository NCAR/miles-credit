import sys

sys.path.insert(0, "/glade/work/schreck/repos/miles-credit-main")

from credit.models.wxformer.wxformer_v2 import CrossFormer

m = CrossFormer(
    image_height=640,
    image_width=1280,
    levels=16,
    channels=4,
    surface_channels=7,
    input_only_channels=3,
    dim=(128, 256, 512, 1024),
    depth=(2, 2, 8, 2),
    cross_embed_strides=(2, 2, 2, 2),
    deformable_start_level=1,
)
total = sum(p.numel() for p in m.parameters()) / 1e6
print(f"Model params: {total:.1f}M")
print("Level 0 use_deformable:", m.layers[0][1].use_deformable)
print("Level 1 use_deformable:", m.layers[1][1].use_deformable)
print("Level 2 use_deformable:", m.layers[2][1].use_deformable)
print("OK")
