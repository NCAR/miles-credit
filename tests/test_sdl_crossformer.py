from credit.models.wxformer.crossformer_ensemble import CrossFormerWithNoise
import torch


def test_sdl_crossformer():
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
    noise_latent_dim = 128
    encoder_noise_factor = 0.05
    decoder_noise_factor = 0.275
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

    model = CrossFormerWithNoise(
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
        dim=(32, 64, 128, 256),
        depth=(2, 2, 4, 2),
        global_window_size=(16, 8, 4, 2),
        local_window_size=4,
        cross_embed_kernel_sizes=((4, 8, 16), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        attn_dropout=0.0,
        ff_dropout=0.0,
        padding_conf=padding_conf,
        noise_latent_dim=noise_latent_dim,
        encoder_noise_factor=encoder_noise_factor,
        decoder_noise_factor=decoder_noise_factor,
    )

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Number of parameters is negative"
    y_pred = model(input_tensor)
    assert y_pred.shape[1] == input_tensor.shape[1] - input_only_channels, "Num channels do not match"
    assert y_pred.shape[2:] == input_tensor.shape[2:], "Output dimensions do not match input dimensions"
    assert ~torch.any(torch.isnan(y_pred)), "NaNs in prediction"


def test_sdl_loads_wxformer_checkpoint():
    # SDL fine-tunes on top of a trained ``model.type: wxformer`` checkpoint, so every
    # base weight must load into CrossFormerWithNoise: only the new noise layers may be
    # missing, nothing unexpected, and no shape mismatch (load_state_dict raises on one
    # even with strict=False).
    from credit.models.wxformer.crossformer import CrossFormer

    arch = dict(
        image_height=32,
        image_width=64,
        frames=1,
        channels=2,
        surface_channels=1,
        input_only_channels=1,
        levels=2,
        dim=(16, 32, 64, 128),
        dim_head=8,
        depth=(1, 1, 1, 1),
        global_window_size=(4, 2, 1, 1),
        local_window_size=2,
        cross_embed_kernel_sizes=((4, 8), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        padding_conf={"activate": True, "mode": "earth", "pad_lat": [16, 16], "pad_lon": [16, 16]},
    )
    base = CrossFormer(**arch)
    model = CrossFormerWithNoise(**arch, noise_latent_dim=8, encoder_noise=True, freeze=True)
    msg = model.load_state_dict(base.state_dict(), strict=False)

    def is_noise(name):
        return name.startswith(("noise_inject", "encoder_noise_layers"))

    assert msg.unexpected_keys == []
    assert [k for k in msg.missing_keys if not is_noise(k)] == []
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(is_noise(n) for n in trainable)

    x = torch.randn(2, 2 * 2 + 1 + 1, 1, 32, 64)
    y = model(x)
    assert y.shape == (2, 2 * 2 + 1, 1, 32, 64)
    y.pow(2).mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)
