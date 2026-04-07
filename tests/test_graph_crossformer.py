import torch
from credit.models.wxformer.graph_crossformer import GraphCrossFormer


def create_cube_sphere_mock_edges(c=96):
    """
    Creates a mock edge_index for a C96 cube sphere.
    For simplicity in this test, we treat it as a (6*c) x c grid.
    Total nodes = 6 * c * c = 55,296 for c=96.
    """
    height = 6 * c
    width = c

    # Pre-allocate edge arrays for speed
    # Internal nodes have 4 edges, boundary nodes have fewer
    # We will build two lists of src and dst
    src = []
    dst = []

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x < width - 1:
                src.extend([idx, idx + 1])
                dst.extend([idx + 1, idx])
            if y < height - 1:
                src.extend([idx, idx + width])
                dst.extend([idx + width, idx])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def test_graph_crossformer_c96():
    """
    Tests the GraphCrossFormer model with a synthetic C96 cube sphere graph.
    """
    c = 96
    N = 6 * c * c
    frames = 2
    output_frames = 1
    channels = 2
    levels = 4
    surface_channels = 1
    input_only_channels = 1

    base_in = channels * levels + surface_channels + input_only_channels
    base_out = channels * levels + surface_channels

    edge_index = create_cube_sphere_mock_edges(c)

    # Instantiate the model with scaled-down dimensions for a fast unit test
    model = GraphCrossFormer(
        frames=frames,
        output_frames=output_frames,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        output_only_channels=0,
        levels=levels,
        dim=(16, 32, 64, 128),
        depth=(1, 1, 2, 1),
        dim_head=8,
        heads=2,
    )

    # Batch size of 2
    B = 2
    x = torch.randn(B, base_in, frames, N)

    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)

    expected_shape = (B, base_out, output_frames, N)

    assert out.shape == expected_shape, f"Expected shape {expected_shape}, but got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert not torch.isinf(out).any(), "Output contains Infs"
