"""Experiment configs built from camulator_gen2's ``camulator_hf.yml``.

Every arm is the HF training config (the stage that produced the released epoch 69), then the shared fine-tune
protocol, then only the arm's own changes, so two arms differ exactly where intended.

Targets:

- ``derecho``: the glade paths of ``camulator_hf.yml``, with one ``save_loc`` per arm.
- ``local``: every glade path swapped for a local copy under ``--local-root``, run on 1981 and sized for a laptop.
  The root is laid out like a miles-credit checkout's ``climate/`` folder: ``data/`` holds the per-year zarrs
  (which also carry the dynamic forcing) and the scaler JSON; ``assets/`` holds the checkpoints, the mean/std files
  and the statics.

Example::

    python -m camulator_smallscale.make_configs --arm null --target local \\
        --local-root ~/PycharmProjects/miles-credit/climate --stage clone

``--stage clone`` puts the released checkpoint in ``save_loc`` as ``checkpoint.pt`` (an APFS clone on macOS, so it
costs no disk space). Use ``--stage link`` only for rollouts: training writes ``checkpoint.pt`` in place, which
through a symlink would overwrite the original.
"""

import argparse
import copy
import os
import posixpath
import shutil
import subprocess
import sys

import yaml

import camulator_gen2

HF_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(camulator_gen2.__file__), "..", "configs", "camulator_hf.yml")
)

DERECHO_RUNS = "/glade/derecho/scratch/$USER/CREDIT_runs/camulator_smallscale"
LOCAL_RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_runs"))

# The local per-year zarrs carry the dynamic forcing too, so the forcing zarr pattern maps onto them.
LOCAL_YEAR_ZARR = "b.e21.CREDIT_climate_branch_1980_%Y_zmdata_ERA5scaled_zmdata_Qtot.zarr"
LOCAL_YEAR = 1981
LOCAL_CHECKPOINT = "checkpoint.pt00069.pt"

# The shared fine-tune protocol (plan D4 and D10): continue from the released weights with a fresh AdamW and HF's
# final-stage settings, count epochs from 0, keep every epoch's checkpoint for the probes. An epoch is HF's 300
# updates; 4 epochs is the default budget, extended one epoch at a time while the probes say so.
FINETUNE = {
    "trainer": {
        "load_weights": True,
        "load_optimizer": False,
        "load_scaler": False,
        "load_scheduler": False,
        "reload_epoch": False,
        "start_epoch": 0,
        "num_epoch": 4,
        "epochs": 4,
        "use_scheduler": False,
        "learning_rate": 2.5e-05,
        "save_every_epoch": True,
        "scheduler": {"T_max": 4},
    }
}

# What each arm changes on top of FINETUNE. The null arm is the HF loss as it is.
ARMS = {
    "null": {},
}

# The local target: 1981 only (the year with full 3-D truth on the laptop), one sample per update, a few updates.
MPS_SAFE_MODEL = {
    "object_type": "model",
    "module_path": "camulator_smallscale.models",
    "module_name": "MPSSafeHFCamulator",
}

LOCAL = {
    "data": {"start_datetime": "1981-01-01", "end_datetime": "1981-12-31"},
    "validation_data": {"start_datetime": "1981-12-01", "end_datetime": "1981-12-31"},
    "trainer": {
        "parallelism": {"data": "none", "tensor": 1, "domain": 1},
        "train_batch_size": 1,
        "valid_batch_size": 1,
        "batches_per_epoch": 5,
        "valid_batches_per_epoch": 2,
        "num_epoch": 1,
        "epochs": 1,
        "scheduler": {"T_max": 1},
        "thread_workers": 0,
        "valid_thread_workers": 0,
        "prefetch_factor": None,
        "use_tensorboard": False,
    },
    "inference": {
        "single_forecast": {"forecast_length": "10d", "start_datetime": "1981-01-01T00:00Z"},
        # credit.metadata's CAM file, resolved by bare name; the HF config's copy lives on glade.
        "output": {"metadata": "cam.yaml"},
    },
}


def deep_update(base, overrides):
    """Return a copy of ``base`` with ``overrides`` merged in (dicts merge, anything else replaces)."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def iter_strings(node):
    """Yield every string in a nested config."""
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_strings(value)
    elif isinstance(node, str):
        yield node


def _local_path(glade_path, root):
    name = posixpath.basename(glade_path)
    if name.endswith("%Y_forcing.zarr"):
        name = LOCAL_YEAR_ZARR
    for sub in ("data", "assets"):
        candidate = os.path.join(root, sub, name)
        if os.path.exists(candidate.replace("%Y", str(LOCAL_YEAR))):
            return candidate
    raise FileNotFoundError(f"No local copy of {glade_path} under {root}/data or {root}/assets")


def localize(node, root):
    """Swap every glade file path for its local copy under ``root``; run directories ($USER) are left alone."""
    if isinstance(node, dict):
        return {key: localize(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [localize(value, root) for value in node]
    if isinstance(node, str) and node.startswith("/glade/") and "$USER" not in node:
        return _local_path(node, root)
    return node


def build_config(arm, target, local_root=None, runs_root=None, base=HF_CONFIG, batches_per_epoch=None):
    """Build one arm's config for ``target`` ("derecho" or "local")."""
    if arm not in ARMS:
        raise KeyError(f"Unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    with open(base) as f:
        conf = yaml.safe_load(f)
    conf = deep_update(conf, FINETUNE)
    conf = deep_update(conf, ARMS[arm])

    if target == "derecho":
        conf["save_loc"] = posixpath.join(runs_root or DERECHO_RUNS, arm)
        conf["inference"]["save_forecast"] = posixpath.join(conf["save_loc"], "rollout")
    elif target == "local":
        if not local_root:
            raise ValueError("The local target needs local_root (or $CAMULATOR_LOCAL_ROOT).")
        root = os.path.abspath(os.path.expanduser(local_root))
        # LOCAL first: it replaces glade-only files (the output metadata YAML) before the rest is localized.
        conf = deep_update(conf, LOCAL)
        conf = localize(conf, root)
        if batches_per_epoch is not None:
            conf["trainer"]["batches_per_epoch"] = int(batches_per_epoch)
        for section in ("data", "validation_data"):
            conf[section]["source"]["CESM"]["level_coord"] = "level"
        conf.pop("pbs", None)
        if conf["model"]["type"] == "hf_camulator":
            # Correct on Apple MPS, identical to hf_camulator elsewhere (camulator_smallscale.mps).
            conf["custom_objects"]["mps_safe_hf_camulator"] = dict(MPS_SAFE_MODEL)
            conf["model"]["type"] = "mps_safe_hf_camulator"
        conf["save_loc"] = os.path.join(os.path.abspath(runs_root or LOCAL_RUNS), arm)
        conf["inference"]["save_forecast"] = os.path.join(conf["save_loc"], "rollout")
        leftover = sorted({s for s in iter_strings(conf) if "/glade/" in s})
        if leftover:
            raise ValueError(f"Glade paths left in the local config: {leftover}")
    else:
        raise ValueError(f"Unknown target {target!r}; use 'derecho' or 'local'.")
    return conf


class _NoAliasDumper(yaml.SafeDumper):
    """Write repeated blocks out in full instead of as &id001 anchors."""

    def ignore_aliases(self, data):
        return True


def write_config(conf, path, arm, target, base=HF_CONFIG):
    """Write ``conf`` to ``path`` with a header that says how it was made."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = (
        f"# Generated by camulator_smallscale.make_configs (arm: {arm}, target: {target})\n"
        f"# from {base}.\n"
        "# Edit the generator, not this file.\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(conf, f, Dumper=_NoAliasDumper, sort_keys=False, default_flow_style=False, width=120)
    return path


def stage_checkpoint(save_loc, checkpoint, mode):
    """Put ``checkpoint`` in ``save_loc`` as ``checkpoint.pt``: ``clone`` (copy-on-write if possible) or ``link``."""
    if mode == "none":
        return None
    dest = os.path.join(save_loc, "checkpoint.pt")
    if os.path.lexists(dest):
        print(f"{dest} exists; leaving it alone.")
        return dest
    os.makedirs(save_loc, exist_ok=True)
    if mode == "link":
        os.symlink(os.path.abspath(checkpoint), dest)
    elif mode == "clone":
        if sys.platform == "darwin" and subprocess.run(["cp", "-c", checkpoint, dest]).returncode == 0:
            return dest
        shutil.copy2(checkpoint, dest)
    else:
        raise ValueError(f"Unknown stage mode {mode!r}; use clone, link or none.")
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--target", required=True, choices=["local", "derecho"])
    parser.add_argument("--local-root", default=os.environ.get("CAMULATOR_LOCAL_ROOT"))
    parser.add_argument("--runs-root", default=None, help="Parent of the per-arm save_loc directories.")
    parser.add_argument(
        "--out", default=None, help="Config path (default: <save_loc>/<arm>.yml locally, ./<arm>_derecho.yml)."
    )
    parser.add_argument("--stage", default="none", choices=["none", "clone", "link"])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to stage (default: the local epoch 69).")
    parser.add_argument("--batches-per-epoch", type=int, default=None, help="Local target only.")
    parser.add_argument("--base", default=HF_CONFIG)
    args = parser.parse_args(argv)

    conf = build_config(args.arm, args.target, args.local_root, args.runs_root, args.base, args.batches_per_epoch)
    if args.target == "local":
        out = args.out or os.path.join(conf["save_loc"], f"{args.arm}.yml")
    else:
        out = args.out or f"{args.arm}_derecho.yml"
    print(write_config(conf, out, args.arm, args.target, args.base))

    if args.stage != "none":
        if args.target != "local":
            parser.error("--stage only applies to the local target")
        checkpoint = args.checkpoint or os.path.join(os.path.expanduser(args.local_root), "assets", LOCAL_CHECKPOINT)
        print(stage_checkpoint(conf["save_loc"], checkpoint, args.stage))


if __name__ == "__main__":
    main()
