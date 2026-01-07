from credit.models.crossformer import CrossFormer
import torch


def test_crossformer():
    image_height = 128  # 640, 192
    image_width = 128  # 1280, 288
    levels = 3
    frames = 1
    output_frames = 1
    channels = 4
    surface_channels = 1
    input_only_channels = 4
    frame_patch_size = 0
    upsample_v_conv = True
    # attention_type = "scse_standard"
    padding_conf = {
        "activate": True,
        "mode": "earth",
        "pad_lat": [64, 64],
        "pad_lon": [64, 64],
    }

    input_tensor = torch.randn(
        1,
        channels * levels + surface_channels + input_only_channels,
        frames,
        image_height,
        image_width,
    )

    model = CrossFormer(
        image_height=image_height,
        image_width=image_width,
        frames=frames,
        output_frames=output_frames,
        frame_patch_size=frame_patch_size,
        channels=channels,
        surface_channels=surface_channels,
        input_only_channels=input_only_channels,
        levels=levels,
        upsample_v_conv=upsample_v_conv,
        # attention_type=attention_type,
        dim=(32, 64, 128, 256),
        depth=(2, 2, 4, 2),
        global_window_size=(16, 8, 4, 2),
        local_window_size=4,
        cross_embed_kernel_sizes=((4, 8, 16), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        attn_dropout=0.0,
        ff_dropout=0.0,
        padding_conf=padding_conf,
    )

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Number of parameters is negative"
    y_pred = model(input_tensor)
    assert (
        y_pred.shape[1] == input_tensor.shape[1] - input_only_channels
    ), "Num channels do not match"
    assert (
        y_pred.shape[2:] == input_tensor.shape[2:]
    ), "Output dimensions do not match input dimensions"
    assert ~torch.any(torch.isnan(y_pred)), "NaNs in prediction"
