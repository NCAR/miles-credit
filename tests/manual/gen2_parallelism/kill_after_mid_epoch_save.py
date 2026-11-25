#!/usr/bin/env python
"""Run train_gen2, but hard-exit every rank right after a given mid-epoch checkpoint.

Simulates a job killed by walltime/preemption at a known point, so the
multi-rank kill-and-resume check in run_mid_epoch_resume.pbs is deterministic.

Usage (same arguments as train_gen2.py, plus --kill-after-batch):
  torchrun --nproc-per-node=2 tests/manual/gen2_parallelism/kill_after_mid_epoch_save.py \
      --kill-after-batch 3 -c config.yml

Exits with code 137 (like SIGKILL) on every rank once the checkpoint for
``batch_in_epoch == N`` is on disk.
"""

import argparse
import os
import sys

import torch

from credit.applications.train_gen2 import main_cli
from credit.trainers.trainer_gen2 import TrainerERA5Gen2


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kill-after-batch", type=int, required=True)
    args, rest = parser.parse_known_args()
    kill_after = args.kill_after_batch

    real_save = TrainerERA5Gen2._save_checkpoint

    def save_then_die(self, *a, resume_state=None, **kw):
        real_save(self, *a, resume_state=resume_state, **kw)
        if resume_state and resume_state.get("batch_in_epoch") == kill_after:
            # Rank 0 writes the file after the collective gathers; wait for it
            # so the kill never lands mid-write (that case is _atomic_save's job).
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            print(f"[rank {self.rank}] killed after mid-epoch checkpoint at batch {kill_after}", flush=True)
            os._exit(137)

    TrainerERA5Gen2._save_checkpoint = save_then_die
    sys.argv = [sys.argv[0], *rest]
    main_cli()


if __name__ == "__main__":
    main()
