#!/usr/bin/env python
"""Check that a killed-and-resumed run matches an uninterrupted one.

Compares the epoch rows of the two runs' training_log.csv (train_loss and
learning rate) and the final checkpoint.pt model weights.

GPU kernels and NCCL reductions are not bit-reproducible by default, so values
are compared with a relative tolerance. A wrong resume position (sample order,
LR schedule, RNG) diverges well beyond it within a few steps.

Usage:
  python check_mid_epoch_resume.py REF_SAVE_LOC RESUMED_SAVE_LOC [--rtol 1e-3]

Exit code 0 on match, 1 on mismatch.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref")
    parser.add_argument("resumed")
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    ok = True
    ref_log = pd.read_csv(os.path.join(args.ref, "training_log.csv"))
    run_log = pd.read_csv(os.path.join(args.resumed, "training_log.csv"))
    if list(ref_log["epoch"]) != list(run_log["epoch"]):
        print(f"FAIL epochs differ: {list(ref_log['epoch'])} vs {list(run_log['epoch'])}")
        return 1
    for col in ("train_loss", "lr"):
        if col not in ref_log:
            continue
        a, b = ref_log[col].to_numpy(float), run_log[col].to_numpy(float)
        if not np.allclose(a, b, rtol=args.rtol, atol=0):
            print(f"FAIL {col}: ref={a.tolist()} resumed={b.tolist()}")
            ok = False
        else:
            print(f"ok   {col}: {a.tolist()}")

    ref_ckpt = torch.load(os.path.join(args.ref, "checkpoint.pt"), map_location="cpu", weights_only=False)
    run_ckpt = torch.load(os.path.join(args.resumed, "checkpoint.pt"), map_location="cpu", weights_only=False)
    if run_ckpt.get("batch_in_epoch", 0) != 0:
        print(f"FAIL resumed checkpoint.pt is still mid-epoch (batch {run_ckpt['batch_in_epoch']})")
        ok = False
    worst = 0.0
    for key, ref_t in ref_ckpt["model_state_dict"].items():
        run_t = run_ckpt["model_state_dict"][key]
        if not torch.is_floating_point(ref_t):
            continue
        diff = (ref_t.float() - run_t.float()).abs().max().item()
        scale = ref_t.float().abs().max().item() or 1.0
        worst = max(worst, diff / scale)
    if worst > args.rtol:
        print(f"FAIL weights: max relative diff {worst:.3g} > rtol {args.rtol}")
        ok = False
    else:
        print(f"ok   weights: max relative diff {worst:.3g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
