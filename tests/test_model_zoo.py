"""
Smoke test for all model-zoo models.
Run on a GPU node: python tests/test_model_zoo.py
Prints a markdown table of results.
"""

import sys
import os
import time
import traceback

import torch

# Ensure repo root is on path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

B = 1
H, W = 32, 64  # small spatial size for speed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Model factories — each returns (model, x, expected_out_channels)
# ---------------------------------------------------------------------------


def make_stormer():
    from credit.models.stormer.stormer import CREDITStormer

    C_IN, C_OUT = 20, 18
    m = CREDITStormer(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        embed_dim=128,
        depth=2,
        num_heads=4,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_climax():
    from credit.models.climax.climax import CREDITClimaX

    C_IN, C_OUT = 20, 18
    m = CREDITClimaX(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        patch_size=2,
        embed_dim=128,
        depth=2,
        num_heads=4,
        agg_depth=1,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_fourcastnet():
    from credit.models.fourcastnet.afno import CREDITFourCastNet

    C_IN, C_OUT = 20, 18
    m = CREDITFourCastNet(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        patch_size=2,
        embed_dim=128,
        depth=2,
        n_blocks=4,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_sfno():
    from credit.models.sfno.sfno import CREDITSfno

    C_IN, C_OUT = 20, 18
    m = CREDITSfno(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        patch_size=2,
        embed_dim=128,
        depth=2,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_swinrnn():
    from credit.models.swinrnn.swinrnn import CREDITSwinRNN

    C_IN, C_OUT = 20, 18
    m = CREDITSwinRNN(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        embed_dim=64,
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        window_size=4,
        patch_size=2,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_fengwu():
    from credit.models.fengwu.fengwu import CREDITFengWu

    C_IN, C_OUT = 20, 18
    m = CREDITFengWu(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        patch_size=4,
        embed_dim=128,
        encoder_depth=1,
        fuser_depth=2,
        decoder_depth=1,
        num_heads=4,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_graphcast():
    from credit.models.graphcast.graphcast import CREDITGraphCast

    C_IN, C_OUT = 20, 18
    m = CREDITGraphCast(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        latent_dim=64,
        edge_dim=32,
        processor_depth=2,
        k_neighbours=4,
        mlp_hidden=128,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_healpix():
    from credit.models.healpix.healpix import CREDITHEALPix

    C_IN, C_OUT = 20, 18
    m = CREDITHEALPix(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        nside=16,
        embed_dim=32,
        depth=1,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_fourcastnet3():
    from credit.models.fourcastnet3.fcn3 import CREDITFourCastNetV3

    C_IN, C_OUT = 20, 18
    m = CREDITFourCastNetV3(
        in_channels=C_IN,
        out_channels=C_OUT,
        img_size=(H, W),
        base_dim=32,
        depth=2,
        n_stages=2,
    ).to(device)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_aurora():
    from credit.models.aurora.model import CREDITAurora

    # 2 surf + 2 atmos * 2 levels + 1 static = 7 in, 6 out
    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITAurora(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=atmos_levels,
        n_lat=H,
        n_lon=W,
    ).to(device)
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_pangu():
    from credit.models.pangu.pangu import CREDITPangu

    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITPangu(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=atmos_levels,
    ).to(device)
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_aifs():
    from credit.models.aifs.aifs import CREDITAifs

    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITAifs(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        atmos_levels=atmos_levels,
    ).to(device)
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


def make_wxformer():
    from credit.models.wxformer.crossformer import CrossFormer as WXFormer

    # minimal config: 2 atmos vars × 2 levels + 2 surface + 1 input-only = 7 in, 6 out
    m = WXFormer(
        image_height=H,
        image_width=W,
        frames=2,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(16, 32),
        depth=(2, 2),
        global_window_size=(2, 1),
        local_window_size=4,
        cross_embed_kernel_sizes=((2, 4), (2, 4)),
        cross_embed_strides=(2, 2),
        use_spectral_norm=True,
    ).to(device)
    C_IN = 2 * 2 + 2 + 1  # channels*levels + surface + input_only
    C_OUT = 2 * 2 + 2  # channels*levels + surface
    return m, torch.randn(B, C_IN, H, W, device=device), (B, C_OUT, H, W)


MODELS = {
    "wxformer": make_wxformer,
    "stormer": make_stormer,
    "climax": make_climax,
    "fourcastnet": make_fourcastnet,
    "sfno": make_sfno,
    "swinrnn": make_swinrnn,
    "fengwu": make_fengwu,
    "graphcast": make_graphcast,
    "healpix": make_healpix,
    "fourcastnet3": make_fourcastnet3,
    "aurora": make_aurora,
    "pangu": make_pangu,
    "aifs": make_aifs,
}


def run_one(name, factory):
    try:
        t0 = time.perf_counter()
        model, x, expected_shape = factory()
        n_params = _params(model)

        y = model(x)

        assert y.shape == torch.Size(expected_shape), f"shape {tuple(y.shape)} != {expected_shape}"
        assert not torch.isnan(y).any(), "NaN in output"

        y.mean().backward()
        fwd_bwd_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": "PASS",
            "params_M": f"{n_params:.1f}",
            "fwd_bwd_ms": f"{fwd_bwd_ms:.0f}",
            "err": "",
        }
    except Exception:
        return {
            "status": "FAIL",
            "params_M": "—",
            "fwd_bwd_ms": "—",
            "err": traceback.format_exc().strip().splitlines()[-1],
        }


def main():
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Grid   : ({H}, {W})\n")

    rows = []
    for name, factory in MODELS.items():
        print(f"  testing {name} ...", flush=True)
        result = run_one(name, factory)
        rows.append((name, result))

    # Markdown table
    header = f"{'Model':<16} {'Status':<8} {'Params (M)':<12} {'Fwd+Bwd (ms)':<14} {'Error'}"
    sep = f"{'-' * 16} {'-' * 8} {'-' * 12} {'-' * 14} {'-' * 60}"
    print(f"\n{header}")
    print(sep)
    for name, r in rows:
        err_str = r["err"][:70] if r["err"] else ""
        print(f"{name:<16} {r['status']:<8} {r['params_M']:<12} {r['fwd_bwd_ms']:<14} {err_str}")

    n_pass = sum(1 for _, r in rows if r["status"] == "PASS")
    print(f"\n{n_pass}/{len(rows)} passed")


if __name__ == "__main__":
    main()
