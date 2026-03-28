"""
Short training sanity test for all model-zoo models.

Runs N gradient steps on fixed random (x, y) pairs to verify:
  1. Loss decreases (model can absorb gradient signal)
  2. No NaN / Inf at any step
  3. Forward + backward work consistently

Run on a GPU node: python tests/test_model_zoo_train.py
"""

import sys
import os
import time
import traceback

import torch
import torch.nn as nn

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

N_STEPS = 50
LR = 1e-3
B = 1
H, W = 32, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Model factories (same as smoke test)
# ---------------------------------------------------------------------------


def make_stormer():
    from credit.models.stormer.stormer import CREDITStormer

    C_IN, C_OUT = 20, 18
    return (
        CREDITStormer(in_channels=C_IN, out_channels=C_OUT, img_size=(H, W), embed_dim=128, depth=2, num_heads=4).to(
            device
        ),
        C_IN,
        C_OUT,
    )


def make_climax():
    from credit.models.climax.climax import CREDITClimaX

    C_IN, C_OUT = 20, 18
    return (
        CREDITClimaX(
            in_channels=C_IN,
            out_channels=C_OUT,
            img_size=(H, W),
            patch_size=2,
            embed_dim=128,
            depth=2,
            num_heads=4,
            agg_depth=1,
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_fourcastnet():
    from credit.models.fourcastnet.afno import CREDITFourCastNet

    C_IN, C_OUT = 20, 18
    return (
        CREDITFourCastNet(
            in_channels=C_IN, out_channels=C_OUT, img_size=(H, W), patch_size=2, embed_dim=128, depth=2, n_blocks=4
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_sfno():
    from credit.models.sfno.sfno import CREDITSfno

    C_IN, C_OUT = 20, 18
    return (
        CREDITSfno(in_channels=C_IN, out_channels=C_OUT, img_size=(H, W), patch_size=2, embed_dim=128, depth=2).to(
            device
        ),
        C_IN,
        C_OUT,
    )


def make_swinrnn():
    from credit.models.swinrnn.swinrnn import CREDITSwinRNN

    C_IN, C_OUT = 20, 18
    return (
        CREDITSwinRNN(
            in_channels=C_IN,
            out_channels=C_OUT,
            img_size=(H, W),
            embed_dim=64,
            depths=[2, 2, 2],
            num_heads=[2, 4, 8],
            window_size=4,
            patch_size=2,
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_fengwu():
    from credit.models.fengwu.fengwu import CREDITFengWu

    C_IN, C_OUT = 20, 18
    return (
        CREDITFengWu(
            in_channels=C_IN,
            out_channels=C_OUT,
            img_size=(H, W),
            patch_size=4,
            embed_dim=128,
            encoder_depth=1,
            fuser_depth=2,
            decoder_depth=1,
            num_heads=4,
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_graphcast():
    from credit.models.graphcast.graphcast import CREDITGraphCast

    C_IN, C_OUT = 20, 18
    return (
        CREDITGraphCast(
            in_channels=C_IN,
            out_channels=C_OUT,
            img_size=(H, W),
            latent_dim=64,
            edge_dim=32,
            processor_depth=2,
            k_neighbours=4,
            mlp_hidden=128,
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_healpix():
    from credit.models.healpix.healpix import CREDITHEALPix

    C_IN, C_OUT = 20, 18
    return (
        CREDITHEALPix(
            in_channels=C_IN, out_channels=C_OUT, img_size=(H, W), nside=16, embed_dim=32, depth=1, n_stages=2
        ).to(device),
        C_IN,
        C_OUT,
    )


def make_fourcastnet3():
    from credit.models.fourcastnet3.fcn3 import CREDITFourCastNetV3

    C_IN, C_OUT = 20, 18
    return (
        CREDITFourCastNetV3(in_channels=C_IN, out_channels=C_OUT, img_size=(H, W), base_dim=32, depth=2, n_stages=2).to(
            device
        ),
        C_IN,
        C_OUT,
    )


def make_aurora():
    from credit.models.aurora.model import CREDITAurora

    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITAurora(
        surf_vars=surf_vars, atmos_vars=atmos_vars, static_vars=static_vars, atmos_levels=atmos_levels, n_lat=H, n_lon=W
    ).to(device)
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, C_IN, C_OUT


def make_pangu():
    from credit.models.pangu.pangu import CREDITPangu

    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITPangu(surf_vars=surf_vars, atmos_vars=atmos_vars, static_vars=static_vars, atmos_levels=atmos_levels).to(
        device
    )
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, C_IN, C_OUT


def make_aifs():
    from credit.models.aifs.aifs import CREDITAifs

    surf_vars = ["2t", "10u"]
    atmos_vars = ["z", "t"]
    static_vars = ["lsm"]
    atmos_levels = [500, 850]
    m = CREDITAifs(surf_vars=surf_vars, atmos_vars=atmos_vars, static_vars=static_vars, atmos_levels=atmos_levels).to(
        device
    )
    C_IN = len(surf_vars) + len(atmos_vars) * len(atmos_levels) + len(static_vars)
    C_OUT = len(surf_vars) + len(atmos_vars) * len(atmos_levels)
    return m, C_IN, C_OUT


def make_wxformer():
    from credit.models.wxformer.crossformer import CrossFormer as WXFormer

    # CrossFormer I/O: (B, C_per_frame, T, H, W) → (B, C_out, 1, H, W)
    FRAMES = 2
    C_IN = 2 * 2 + 2 + 1  # channels*levels + surface + input_only
    C_OUT = 2 * 2 + 2  # channels*levels + surface
    m = WXFormer(
        image_height=H,
        image_width=W,
        frames=FRAMES,
        channels=2,
        surface_channels=2,
        input_only_channels=1,
        output_only_channels=0,
        levels=2,
        dim=(4, 8, 16, 32),
        depth=(2, 2, 2, 2),
        global_window_size=(2, 2, 1, 1),
        local_window_size=4,
        cross_embed_kernel_sizes=((2, 4), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        use_spectral_norm=True,
    ).to(device)
    return m, C_IN, C_OUT, FRAMES


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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one(name, factory, n_steps=N_STEPS):
    try:
        result = factory()
        model, C_IN, C_OUT = result[0], result[1], result[2]
        frames = result[3] if len(result) == 4 else None
        n_params = _params(model)

        # fixed random target so we're fitting one sample
        torch.manual_seed(42)
        x = torch.randn(B, C_IN, frames, H, W, device=device) if frames else torch.randn(B, C_IN, H, W, device=device)
        y_tgt = torch.randn(B, C_OUT, 1, H, W, device=device) if frames else torch.randn(B, C_OUT, H, W, device=device)

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.MSELoss()

        losses = []
        t0 = time.perf_counter()
        for step in range(n_steps):
            opt.zero_grad()
            y_pred = model(x)
            loss = crit(y_pred, y_tgt)
            if torch.isnan(loss) or torch.isinf(loss):
                return {
                    "status": "NaN/Inf",
                    "params_M": f"{n_params:.1f}",
                    "loss_0": "—",
                    "loss_N": "—",
                    "pct_drop": "—",
                    "ms_per_step": "—",
                    "err": f"loss={loss.item()} at step {step}",
                }
            loss.backward()
            opt.step()
            losses.append(loss.item())

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ms_per_step = elapsed_ms / n_steps
        pct_drop = 100.0 * (losses[0] - losses[-1]) / (losses[0] + 1e-8)
        converging = "YES" if losses[-1] < losses[0] else "NO"

        return {
            "status": converging,
            "params_M": f"{n_params:.1f}",
            "loss_0": f"{losses[0]:.4f}",
            "loss_N": f"{losses[-1]:.4f}",
            "pct_drop": f"{pct_drop:.1f}%",
            "ms_per_step": f"{ms_per_step:.0f}",
            "err": "",
        }
    except Exception:
        return {
            "status": "FAIL",
            "params_M": "—",
            "loss_0": "—",
            "loss_N": "—",
            "pct_drop": "—",
            "ms_per_step": "—",
            "err": traceback.format_exc().strip().splitlines()[-1][:70],
        }


def main():
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Grid   : ({H}, {W})   Steps: {N_STEPS}   LR: {LR}\n")

    rows = []
    for name, factory in MODELS.items():
        print(f"  training {name} ...", flush=True)
        r = train_one(name, factory)
        rows.append((name, r))

    hdr = f"{'Model':<16} {'Conv?':<6} {'Params(M)':<10} {'Loss[0]':<10} {'Loss[N]':<10} {'Drop%':<8} {'ms/step':<10} {'Error'}"
    sep = "-" * 90
    print(f"\n{hdr}\n{sep}")
    for name, r in rows:
        print(
            f"{name:<16} {r['status']:<6} {r['params_M']:<10} "
            f"{r['loss_0']:<10} {r['loss_N']:<10} {r['pct_drop']:<8} "
            f"{r['ms_per_step']:<10} {r['err']}"
        )

    n_conv = sum(1 for _, r in rows if r["status"] == "YES")
    n_fail = sum(1 for _, r in rows if r["status"] == "FAIL")
    print(f"\n{n_conv}/{len(rows)} converging   {n_fail} failed")


if __name__ == "__main__":
    main()
