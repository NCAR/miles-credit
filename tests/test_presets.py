"""
End-to-end preset test: instantiate each preset, load pretrained weights,
run a forward pass where the model fits in GPU memory.

Run via scripts/casper_preset_test.sh
"""

import sys
import os
import torch
import traceback

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from credit.models import _resolve_preset, load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU   : {props.name}  {props.total_memory / 1e9:.1f} GB")
print()

results = {}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def try_load(preset_name, do_forward=False, x_shape=None):
    print(f"{'=' * 60}")
    print(f"Preset: {preset_name}")
    try:
        conf = {"model": {"preset": preset_name}, "save_loc": "/tmp"}
        resolved = _resolve_preset(conf)
        m = resolved["model"]
        ckpt_path = m.get("pretrained_weights", "")
        print(f"  type             : {m['type']}")
        print(f"  pretrained_weights: {ckpt_path}")
        print(f"  checkpoint exists : {os.path.isfile(ckpt_path)}")

        model = load_model(resolved)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  params           : {n_params:.1f} M")

        # Count key match
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        model_keys = set(k for k, _ in model.named_parameters())
        matched = [k for k in state if k in model_keys]
        print(f"  keys matched     : {len(matched)} / {len(model_keys)} model params  ({len(state)} in ckpt)")

        if do_forward and x_shape is not None:
            model = model.to(device)
            model.eval()
            x = torch.randn(*x_shape, device=device)
            with torch.no_grad():
                y = model(x)
            print(f"  forward OK       : {tuple(x.shape)} -> {tuple(y.shape)}")
            del model, x, y
            torch.cuda.empty_cache()

        results[preset_name] = "PASS"
        print("  RESULT: PASS")

    except Exception as e:
        results[preset_name] = f"FAIL: {e}"
        print("  RESULT: FAIL")
        traceback.print_exc()
    print()


# ---------------------------------------------------------------------------
# WXFormer presets — instantiate + load; skip forward (640x1280 is too big
# for a single H100 with batch > 1; just verify key loading)
# ---------------------------------------------------------------------------

try_load("wxformer-v2-025deg-6h", do_forward=False)
try_load("wxformer-025deg-1h", do_forward=False)
try_load("wxformer-1deg-6h", do_forward=False)
try_load("fuxi-025deg-6h", do_forward=False)

# ---------------------------------------------------------------------------
# Stormer — small enough for a forward pass on GPU
# (128x256 grid, patch_size=2, depth=24, embed_dim=1024)
# ---------------------------------------------------------------------------

try_load(
    "stormer-1.40625deg",
    do_forward=True,
    x_shape=(1, 69, 128, 256),
)

# ---------------------------------------------------------------------------
# ClimaX — similar size to Stormer
# ---------------------------------------------------------------------------

try_load(
    "climax-1.40625deg",
    do_forward=True,
    x_shape=(1, 69, 128, 256),
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = True
for name, result in results.items():
    status = "✓" if result == "PASS" else "✗"
    print(f"  {status} {name:35s}  {result}")
    if result != "PASS":
        all_pass = False

print()
if all_pass:
    print("All presets PASSED.")
    sys.exit(0)
else:
    print("Some presets FAILED.")
    sys.exit(1)
