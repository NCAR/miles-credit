import pytest
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


# CAMulator trains its decoder ``sharp`` convs without spectral norm; the wxformer
# CrossFormer underneath CrossFormerWithNoise normalizes them unless told not to.
_CAM_ARCH = dict(
    image_height=32,
    image_width=64,
    frames=1,
    channels=2,
    surface_channels=1,
    input_only_channels=1,
    output_only_channels=1,
    levels=2,
    dim=(32, 64, 128, 256),
    depth=(1, 1, 1, 1),
    global_window_size=(4, 2, 1, 1),
    local_window_size=2,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    padding_conf={"activate": True, "mode": "earth", "pad_lat": [16, 16], "pad_lon": [16, 16]},
)


def _camulator_and_input():
    from credit.models.camulator import Camulator

    torch.manual_seed(0)
    cam = Camulator(**_CAM_ARCH)
    x = torch.randn(2, 2 * 2 + 1 + 1, 1, 32, 64)
    with torch.no_grad():  # converge the spectral-norm power iterations so outputs are well scaled
        for _ in range(20):
            cam(x)
    return cam.eval(), x


def test_sdl_loads_camulator_checkpoint():
    cam, x = _camulator_and_input()
    sdl = CrossFormerWithNoise(**_CAM_ARCH, noise_latent_dim=8, encoder_noise=False, spectral_norm_sharp=False).eval()
    msg = sdl.load_state_dict(cam.state_dict(), strict=False)
    assert msg.unexpected_keys == []
    assert [k for k in msg.missing_keys if not k.startswith("noise_inject")] == []

    for module in sdl.modules():  # zero noise: the SDL model must reproduce CAMulator exactly
        if hasattr(module, "noise_factor"):
            module.noise_factor.data.zero_()
    with torch.no_grad():
        torch.testing.assert_close(sdl(x), cam(x), rtol=1e-4, atol=1e-4)


def test_sharp_spectral_norm_mismatch_raises():
    from credit.models.wxformer.crossformer import migrate_legacy_state_dict

    cam, _ = _camulator_and_input()
    sdl = CrossFormerWithNoise(**_CAM_ARCH, noise_latent_dim=8, encoder_noise=False)
    with pytest.raises(RuntimeError, match="spectral_norm_sharp: false"):
        sdl.load_state_dict(cam.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="spectral_norm_sharp: false"):  # the FSDP2 path
        migrate_legacy_state_dict(sdl, dict(cam.state_dict()))

    # and the reverse: a default (normalized-sharp) checkpoint into a plain-sharp model
    plain = CrossFormerWithNoise(**_CAM_ARCH, noise_latent_dim=8, encoder_noise=False, spectral_norm_sharp=False)
    with pytest.raises(RuntimeError, match="with spectral norm"):
        plain.load_state_dict(sdl.state_dict(), strict=False)
