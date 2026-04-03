#!/usr/bin/env python
"""
CREDIT model smoke test — forward + backward pass for every registered model.
Also loads pretrained weights for wxformer_1h, wxformer_6h, and fuxi_6h.

Run on casper29 (V100-32GB):
    conda run -n credit-main-casper python model_smoke_test.py
"""

import os
import time
import traceback
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

PASS, FAIL, SKIP = "✅ PASS", "❌ FAIL", "⚠️  SKIP"
results = {}


def run(name, fn):
    torch.cuda.empty_cache()
    t0 = time.time()
    try:
        status, info = fn()
        dt = time.time() - t0
        results[name] = {"status": status, "dt": dt, **info}
        icon = status
        detail = "  ".join(f"{k}={v}" for k, v in info.items())
        print(f"  {icon}  {name:<42s} {dt:5.1f}s  {detail}")
    except Exception as e:
        dt = time.time() - t0
        results[name] = {"status": FAIL, "dt": dt, "error": str(e)[:120]}
        print(f"  {FAIL}  {name:<42s} {dt:5.1f}s  {e}")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()


def fwd_bwd(model, *inputs, extra_info=None):
    """Run forward + backward, return (status, info dict)."""
    model = model.to(DEVICE).train()
    inputs = [x.to(DEVICE) for x in inputs]
    out = model(*inputs)
    if isinstance(out, dict):
        out = out.get("y_pred", next(iter(out.values())))
    loss = out.float().sum()
    loss.backward()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    info = {"params_M": f"{nparams:.1f}", "out": str(tuple(out.shape))}
    if extra_info:
        info.update(extra_info)
    model.cpu()
    return PASS, info


# ─────────────────────────────────────────────────────────────────────────────
# Shared small-model dims (H=192, W=288 from test config)
H, W = 192, 288
CH, SURF, IN_ONLY, LVLS, FRAMES = 4, 7, 3, 16, 1
IN_C = CH * LVLS + SURF + IN_ONLY  # 74
OUT_C = CH * LVLS + SURF  # 71
CROSS_CONF = dict(
    image_height=H,
    image_width=W,
    frames=FRAMES,
    channels=CH,
    surface_channels=SURF,
    input_only_channels=IN_ONLY,
    output_only_channels=0,
    levels=LVLS,
    patch_height=1,
    patch_width=1,
    frame_patch_size=1,
    dim=(64, 128, 256, 512),
    depth=(2, 2, 8, 2),
    global_window_size=(4, 4, 2, 1),
    local_window_size=3,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    attn_dropout=0.0,
    ff_dropout=0.0,
    interp=True,
    padding_conf={"activate": True, "mode": "earth", "pad_lat": [48, 48], "pad_lon": [48, 48]},
)


def X():
    return torch.randn(1, IN_C, FRAMES, H, W)


# ─────────────────────────────────────────────────────────────────────────────
print("\n── CrossFormer variants ──────────────────────────────────────────")


def test_crossformer():
    from credit.models.crossformer import CrossFormer

    return fwd_bwd(CrossFormer(**CROSS_CONF), X())


run("crossformer", test_crossformer)


def test_wxformer():
    from credit.models.wxformer.crossformer import CrossFormer as WXFormer

    return fwd_bwd(WXFormer(**CROSS_CONF), X())


run("wxformer (crossformer alias)", test_wxformer)


def test_camulator():
    from credit.models.camulator import Camulator

    return fwd_bwd(Camulator(**CROSS_CONF), X())


run("camulator", test_camulator)


def test_ensemble():
    from credit.models.wxformer.crossformer_ensemble import CrossFormerWithNoise

    return fwd_bwd(
        CrossFormerWithNoise(
            **CROSS_CONF,
            noise_latent_dim=64,
            encoder_noise_factor=0.05,
            decoder_noise_factor=0.275,
            encoder_noise=True,
            freeze=False,
            correlated=False,
        ),
        X(),
    )


run("crossformer-ensemble", test_ensemble)


def test_downscaling_cf():
    from credit.models.wxformer.crossformer_downscaling import DownscalingCrossFormer

    channels = {"boundary": IN_C, "prognostic": OUT_C, "diagnostic": 0}
    conf = {**CROSS_CONF}
    conf.pop("channels")
    conf.pop("surface_channels")
    conf.pop("input_only_channels")
    conf.pop("output_only_channels")
    conf.pop("levels")
    return fwd_bwd(DownscalingCrossFormer(channels=channels, **conf), torch.randn(1, IN_C + OUT_C, FRAMES, H, W))


run("crossformer_downscaling", test_downscaling_cf)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── UNet variants ─────────────────────────────────────────────────")


def test_unet():
    from credit.models.unet import SegmentationModel

    arch = {"name": "unet", "encoder_name": "resnet34", "encoder_weights": None}
    return fwd_bwd(
        SegmentationModel(
            image_height=H,
            image_width=W,
            frames=FRAMES,
            channels=CH,
            surface_channels=SURF,
            input_only_channels=IN_ONLY,
            output_only_channels=0,
            levels=LVLS,
            architecture=arch,
        ),
        X(),
    )


run("unet (SegmentationModel)", test_unet)


def test_unet_downscaling():
    from credit.models.unet.unet_downscaling import DownscalingSegmentationModel

    channels = {"boundary": IN_C, "prognostic": OUT_C, "diagnostic": 0}
    arch = {"name": "unet", "encoder_name": "resnet34", "encoder_weights": None}
    return fwd_bwd(
        DownscalingSegmentationModel(
            channels=channels, image_height=H, image_width=W, frames=FRAMES, architecture=arch
        ),
        torch.randn(1, IN_C + OUT_C, FRAMES, H, W),
    )


run("unet_downscaling", test_unet_downscaling)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Fuxi ──────────────────────────────────────────────────────────")


def test_fuxi():
    from credit.models.fuxi import Fuxi

    # patch_height=32 needs H divisible by 32: 192/32=6 ✓
    # frame_patch_size=2, frames=2: 2/2=1 ✓
    return fwd_bwd(
        Fuxi(
            image_height=H,
            image_width=W,
            frames=2,
            channels=CH,
            surface_channels=SURF,
            input_only_channels=IN_ONLY,
            output_only_channels=0,
            levels=LVLS,
            patch_height=32,
            patch_width=32,
            frame_patch_size=2,
            dim=128,
            depth=2,
            num_groups=2,
            num_heads=8,
            window_size=7,
            use_spectral_norm=True,
            interp=True,
            padding_conf={"activate": True, "mode": "mirror", "pad_lat": [40, 40], "pad_lon": [40, 40]},
        ),
        torch.randn(1, IN_C, 2, H, W),
    )


run("fuxi", test_fuxi)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Swin ──────────────────────────────────────────────────────────")


def test_swin():
    from credit.models.swin import SwinTransformerV2Cr

    return fwd_bwd(
        SwinTransformerV2Cr(
            img_size=(H, W),
            frames=1,
            channels=CH,
            surface_channels=SURF,
            input_only_channels=IN_ONLY,
            output_only_channels=0,
            levels=LVLS,
            patch_size=4,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
        ),
        X(),
    )


run("swin (SwinTransformerV2Cr)", test_swin)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── WRF / Regional ────────────────────────────────────────────────")


def test_wrf():
    from credit.models.swin_wrf import WRFTransformer

    HI, WI = 64, 64  # interior domain (small)
    HO, WO = 64, 64  # exterior domain
    CH_I, CH_O = 3, 2
    SURF_I, SURF_O = 4, 3
    IN_ONLY_I = 2
    pi = dict(
        image_height=HI,
        image_width=WI,
        levels=4,
        frames=1,
        frame_patch_size=1,
        patch_height=4,
        patch_width=4,
        channels=CH_I,
        surface_channels=SURF_I,
        input_only_channels=IN_ONLY_I,
        output_only_channels=0,
        dim=128,
    )
    po = dict(
        image_height=HO,
        image_width=WO,
        levels=3,
        frames=1,
        frame_patch_size=1,
        patch_height=4,
        patch_width=4,
        channels=CH_O,
        surface_channels=SURF_O,
        dim=128,
    )
    model = WRFTransformer(pi, po, depth=2, num_groups=2, num_heads=4, window_size=4, interp=True)
    in_c = CH_I * 4 + SURF_I + IN_ONLY_I  # 18
    out_c = CH_O * 3 + SURF_O  #  9
    x = torch.randn(1, in_c, 1, HI, WI)
    x_outside = torch.randn(1, out_c, 1, HO, WO)
    time_enc = torch.randn(1, 12)
    return fwd_bwd(model, x, x_outside, time_enc)


run("wrf (WRFTransformer)", test_wrf)


def test_dscale():
    from credit.models.dscale_wrf import DscaleTransformer

    return fwd_bwd(
        DscaleTransformer(
            total_input_channels=IN_C,
            total_target_channels=OUT_C,
            image_height=H,
            image_width=W,
            frames=1,
            patch_height=4,
            patch_width=4,
            frame_patch_size=1,
            dim=128,
            depth=2,
            num_groups=2,
            num_heads=4,
            window_size=4,
            time_encode_dim=4,
            interp=True,
        ),
        X(),
        torch.randn(1, 4),
    )


run("dscale (DscaleTransformer)", test_dscale)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Graph ─────────────────────────────────────────────────────────")


def test_graph():
    # Edge file lives on Derecho scratch, not mounted on Casper.
    # TODO: ask DJ (dgagne) to copy grid_edge_pairs_125_onedeg.nc to campaign storage.
    edge_path = "/glade/derecho/scratch/dgagne/credit_scalers/grid_edge_pairs_125_onedeg.nc"
    if not os.path.exists(edge_path):
        return SKIP, {"reason": "edge file only on Derecho — copy to /glade/campaign to test on Casper (ask DJ)"}
    from credit.models.graph import GraphResTransfGRU

    return fwd_bwd(
        GraphResTransfGRU(
            n_variables=CH,
            n_surface_variables=SURF,
            n_static_variables=IN_ONLY,
            levels=LVLS,
            history_len=FRAMES,
            hidden_size=64,
            dim_head=16,
            n_blocks=2,
            edge_path=edge_path,
        ),
        X(),
    )


run("graph (GraphResTransfGRU)", test_graph)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Debugger ──────────────────────────────────────────────────────")


def test_debugger():
    from credit.models.debugger_model import DebuggerModel

    return fwd_bwd(
        DebuggerModel(
            image_height=H,
            image_width=W,
            frames=FRAMES,
            channels=CH,
            surface_channels=SURF,
            input_only_channels=IN_ONLY,
            output_only_channels=0,
            levels=LVLS,
        ),
        X(),
    )


run("debugger", test_debugger)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Pretrained weight loading ─────────────────────────────────────")

PRETRAINED = {
    "wxformer_1h": {
        "ckpt": "/glade/campaign/cisl/aiml/credit/pretrained_weights/wxformer_1h_hf/finetune_final/best_model_checkpoint.pt",
        "type": "crossformer",
        "shape": (1, 74, 1, 640, 1280),
        "conf": dict(
            image_height=640,
            image_width=1280,
            frames=1,
            channels=4,
            surface_channels=7,
            input_only_channels=3,
            output_only_channels=0,
            levels=16,
            patch_height=1,
            patch_width=1,
            frame_patch_size=1,
            dim=(128, 256, 512, 1024),
            depth=(2, 2, 8, 2),
            global_window_size=(10, 5, 2, 1),
            local_window_size=10,
            cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
            cross_embed_strides=(2, 2, 2, 2),
            attn_dropout=0.0,
            ff_dropout=0.0,
            interp=False,
            padding_conf={"activate": True, "mode": "earth", "pad_lat": [80, 80], "pad_lon": [80, 80]},
        ),
    },
    "wxformer_6h": {
        "ckpt": "/glade/campaign/cisl/aiml/credit/pretrained_weights/wxformer_6h_hf/finetune_final/best_model_checkpoint.pt",
        "type": "crossformer",
        "shape": (1, 74, 1, 640, 1280),
        "conf": dict(
            image_height=640,
            image_width=1280,
            frames=1,
            channels=4,
            surface_channels=7,
            input_only_channels=3,
            output_only_channels=0,
            levels=16,
            patch_height=1,
            patch_width=1,
            frame_patch_size=1,
            dim=(128, 256, 512, 1024),
            depth=(2, 2, 8, 2),
            global_window_size=(10, 5, 2, 1),
            local_window_size=10,
            cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
            cross_embed_strides=(2, 2, 2, 2),
            attn_dropout=0.0,
            ff_dropout=0.0,
            interp=False,
            padding_conf={"activate": True, "mode": "earth", "pad_lat": [80, 80], "pad_lon": [80, 80]},
        ),
    },
    "fuxi_6h": {
        "ckpt": "/glade/campaign/cisl/aiml/credit/pretrained_weights/fuxi_6h_hf/best_model_checkpoint.pt",
        "type": "fuxi",
        "shape": (1, 74, 2, 640, 1280),
        "conf": dict(
            image_height=640,
            image_width=1280,
            frames=2,
            channels=4,
            surface_channels=7,
            input_only_channels=3,
            output_only_channels=0,
            levels=16,
            patch_height=4,
            patch_width=4,
            frame_patch_size=2,
            dim=1024,
            depth=16,
            num_groups=32,
            num_heads=8,
            window_size=7,
            use_spectral_norm=True,
            interp=False,
            padding_conf={"activate": True, "mode": "mirror", "pad_lat": [80, 80], "pad_lon": [80, 80]},
        ),
    },
}


def test_pretrained(name, spec):
    if not os.path.exists(spec["ckpt"]):
        return SKIP, {"reason": "checkpoint not found"}
    if spec["type"] == "crossformer":
        from credit.models.crossformer import CrossFormer as Model
    elif spec["type"] == "fuxi":
        from credit.models.fuxi import Fuxi as Model
    model = Model(**spec["conf"])
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        x = torch.randn(*spec["shape"]).to(DEVICE)
        out = model(x)
        if isinstance(out, dict):
            out = out.get("y_pred", next(iter(out.values())))
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    model.cpu()
    return PASS, {
        "params_M": f"{nparams:.1f}",
        "out": str(tuple(out.shape)),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


for name, spec in PRETRAINED.items():
    run(f"pretrained/{name}", lambda s=spec: test_pretrained(name, s))

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Kyle's WRF pretrained (CONUS_FULL_clean, Mar 2026) ────────────")


def test_pretrained_wrf():
    ckpt = "/glade/work/ksha/DWC_runs/CONUS_FULL_clean/best_model_checkpoint.pt"
    if not os.path.exists(ckpt):
        return SKIP, {"reason": "checkpoint not found"}
    from credit.models.swin_wrf import WRFTransformer

    # CONUS_FULL_clean config (336×336, 12 interior levels, 6 boundary levels)
    pi = dict(
        frames=1,
        image_height=336,
        image_width=336,
        levels=12,
        channels=5,
        surface_channels=6,
        input_only_channels=2,
        output_only_channels=0,
        patch_height=4,
        patch_width=4,
        frame_patch_size=1,
        dim=1536,
    )
    po = dict(
        frames=2,
        image_height=336,
        image_width=336,
        levels=6,
        channels=4,
        surface_channels=4,
        patch_height=4,
        patch_width=4,
        frame_patch_size=2,
        dim=1536,
    )
    model = WRFTransformer(
        pi,
        po,
        time_encode_dim=16,
        depth=36,
        num_groups=32,
        num_heads=8,
        window_size=7,
        use_spectral_norm=True,
        interp=False,
        padding_conf={"activate": False},
        post_conf={
            "activate": True,
            "tracer_fixer": {
                "activate": True,
                "denorm": True,
                "tracer_name": ["WRF_Q_tot_05", "WRF_PWAT_05"],
                "tracer_thres": [0, 0],
            },
        },
    )
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    # interior: 5*12+6+2=68 channels, 1 frame, 336×336
    # outside:  4*6+4=28 channels, 2 frames, 336×336
    # time_enc: 16-dim
    x_int = torch.randn(1, 68, 1, 336, 336).to(DEVICE)
    x_out = torch.randn(1, 28, 2, 336, 336).to(DEVICE)
    x_time = torch.randn(1, 16).to(DEVICE)
    with torch.no_grad():
        out = model(x_int, x_out, x_time)
        if isinstance(out, dict):
            out = out.get("y_pred", next(iter(out.values())))
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    model.cpu()
    return PASS, {
        "params_M": f"{nparams:.1f}",
        "out": str(tuple(out.shape)),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


run("pretrained/wrf_conus_full (Kyle, Mar-2026)", test_pretrained_wrf)

# ─────────────────────────────────────────────────────────────────────────────
print("\n── Summary ───────────────────────────────────────────────────────")
passes = sum(1 for r in results.values() if r["status"] == PASS)
fails = sum(1 for r in results.values() if r["status"] == FAIL)
skips = sum(1 for r in results.values() if r["status"] == SKIP)
print(f"  {passes} passed  {fails} failed  {skips} skipped  ({len(results)} total)")
if fails:
    print("\n  Failed models:")
    for name, r in results.items():
        if r["status"] == FAIL:
            print(f"    {name}: {r.get('error', '')}")
