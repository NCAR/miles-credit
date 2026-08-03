#!/usr/bin/env python
"""Compare per-step train losses between two parallelism-mode smoke logs.

Modes that share the same data-parallel layout (same dp_size, same seed) must
produce IDENTICAL loss trajectories: the parallelism changes how the
computation is distributed, never the math. This is the strongest cheap check
that the dp-rank dataloader contract, halo exchange, and TP all_reduce are
wired correctly — a broken wiring still "trains" but the trajectories diverge.

Known-good pairs from the smoke matrix (both fixed seed 1000):
  tp_domain (4 GPU, dp=1)  vs  fsdp2_tp_domain (4 GPU, dp=1; FSDP2 self-skips)
      -> EXACT match expected (identical code path executes).
  ddp (2 GPU, dp=2)        vs  domain (4 GPU, dp=2 x domain=2)
      -> match to ~1% only: with interp=true the serial model interpolates its
         padded output back to grid size, while the domain path crops via
         unpad_shard_interp. Same data, slightly different output operator.
         Use --rtol 0.05; a wiring bug (wrong samples per rank) diverges far
         beyond that within a few steps.

Usage:
  python check_parity.py logA.log logB.log [--rtol 0]

Exit code 0 on parity, 1 on mismatch.
"""

import argparse
import re
import sys


def loss_sequence(path: str) -> list[str]:
    """Extract the deduplicated train_loss sequence from a smoke log.

    Multi-rank tqdm output prints each loss once per rank; consecutive
    duplicates are collapsed so the sequence is one entry per step.
    """
    txt = open(path, errors="ignore").read()
    raw = re.findall(r"train_loss: ([\d.]+)", txt)
    out: list[str] = []
    for v in raw:
        if not out or out[-1] != v:
            out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--rtol", type=float, default=0.0, help="relative tolerance; 0 = require exact printed equality")
    args = ap.parse_args()

    a = loss_sequence(args.log_a)
    b = loss_sequence(args.log_b)

    if not a or not b:
        print(f"PARITY ERROR: no train_loss entries found ({len(a)} vs {len(b)})")
        return 1

    n = min(len(a), len(b))
    bad = []
    for i in range(n):
        if args.rtol > 0:
            va, vb = float(a[i]), float(b[i])
            if abs(va - vb) > args.rtol * max(abs(va), abs(vb)):
                bad.append((i, a[i], b[i]))
        elif a[i] != b[i]:
            bad.append((i, a[i], b[i]))

    if bad:
        print(f"PARITY FAIL: {len(bad)}/{n} steps differ")
        for i, va, vb in bad[:5]:
            print(f"  step {i}: {va} vs {vb}")
        return 1

    print(f"PARITY OK: {n} steps identical ({args.log_a} vs {args.log_b})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
