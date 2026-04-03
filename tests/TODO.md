# Tests to Implement

## Trainer
- `test_trainer_era5.py` — unit test for `Trainer.__init__`, `train_one_epoch`, and `validate` using a
  tiny toy model and a mock `ERA5_MultiStep_Batcher`-style dataloader (no GPU, `amp=False`, `mode=none`).
  Verifies all `self.xxx` refs resolve and both loops run end-to-end without crashing.
- `test_base_trainer_init.py` — verify `BaseTrainer.__init__` correctly extracts every field from `conf`
  (spot-check `self.epochs`, `self.amp`, `self.scheduler_type`, `self.ema`, etc.) with a minimal conf dict.

## Scheduler
- `test_scheduler_warmup.py` — verify `LinearWarmupCosineScheduler` lr values: ramps linearly during
  warmup, peaks at `base_lr`, decays to `min_lr` at `total_steps`, and never undershoots `min_lr`.

## EMA
- `test_ema_tracker.py` — verify `EMATracker.update` moves shadow toward param values, `swap` exchanges
  weights correctly and is idempotent when called twice, and `state_dict`/`load_state_dict` round-trips.
