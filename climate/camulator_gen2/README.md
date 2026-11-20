# camulator_gen2: the Hugging Face CAMulator in the gen2 pipeline

CREDIT custom objects and a config that reproduce how the released CAMulator checkpoints
([huggingface.co/willychap/camulator](https://huggingface.co/willychap/camulator)) were trained and run, using
the gen2 trainer and `rollout_gen2` without changing CREDIT itself.

The checkpoints come from the gen1 code on `WillyChap/miles-credit@camulator_huggingface` ("HF" below): run
`NEW_CLI_JOHN_CASPER_extended_v2`, whose config is `climate/camulator_config.yml` at `5102a3b9^` on that branch.

## Install

The package is not part of the `miles-credit` wheel, so `pip install miles-credit` does not include it.

```bash
pip install -e climate/camulator_gen2            # from a clone of miles-credit
pip install -e climate/camulator_gen2 --no-deps  # on NCAR's shared conda envs
pip install "git+https://github.com/NCAR/miles-credit.git#subdirectory=climate/camulator_gen2"  # without a clone
```

## Use

```bash
python -m camulator_gen2.make_scaler --mean mean_6h_..._Mixed_Modal.nc --std std_6h_..._Mixed_Modal.nc --out scaler.json
credit check -c climate/camulator_gen2/configs/camulator_hf.yml
credit train -c climate/camulator_gen2/configs/camulator_hf.yml
credit rollout -c climate/camulator_gen2/configs/camulator_hf.yml   # checkpoint in save_loc as checkpoint.pt
```

The config points at the NCAR glade copies of the data, statistics and statics. The mean/std and statics files also
ship with the HF assets.

## What reproduces what

HF training step (`credit/trainers/trainerERA5.py` on the HF branch) and its gen2 equivalent:

| HF | gen2 (`configs/camulator_hf.yml`) |
|---|---|
| model, with `TracerFixer` inside it (normalized space) | `reconstruct` + `hf_tracer_fixer`, before anything else |
| mass fixer, outside the model | `inverse_scale` + `global_mass_fixer` |
| MSE scored on the prediction saved here | `flatten_hf` writes `y_pred_hf`; `hf_loss` scores it |
| water fixer, which also returns `(P_ratio - 1)^2` | `hf_water_fixer` (stock fixer + the penalty) |
| energy fixer (up/down fluxes) | `global_energy_fixer_updown` |
| loss = lat-weighted MSE + 0.05 x penalty | `hf_loss`, `water_penalty_weight: 0.05` |
| validation scores the fully fixed prediction, no penalty | `flatten` writes `y_pred`; `hf_loss` scores it under `no_grad` |
| fully fixed state, detached, is the next input | gen2 trainer default |
| `forecast_len: 2` (3 steps), backward on each step | `forecast_len: 3` (gen2 is 1-indexed), trainer default |
| AdamW (0.9, 0.95), lr 1e-4, no weight decay, grad clip 1.0, no AMP/EMA/scheduler | trainer settings |
| Mixed_Modal mean/std, PS per gridpoint | bridgescaler JSON from `make_scaler`, `spatial_variables: [PS]` |

HF rollout (`climate/Model_State.py` on the HF branch): model (tracer fixer inside), wind filter, mass, water,
energy. That is `inference.postblocks` in the config; the wind filter is not in the training chain, as in HF.

### The custom objects

- **`HFTracerFixer`** (`hf_tracer_fixer`): HF's tracer fixer, same float32 arithmetic, on the normalized
  `y_processed` right after `reconstruct`. `replicate_level0_stats_bug: True` (default) keeps HF's bug: every Qtot
  level is clipped with the level-0 (model-top) mean/std, which leaves levels 1-31 effectively unclipped
  (`CAMULATOR_QTOT_TRACER_BUG.md`). Set it to `False` to clip each level at 0.
- **`HFWaterFixer`** (`hf_water_fixer`): the stock gen2 `GlobalWaterFixer`, plus the penalty
  `mean((P_correct_ratio - 1)^2)` computed before the correction and kept in the graph.
- **`HFLoss`** (`hf_loss`): HF's `VariableTotalLoss2D` (MSE, latitude weights `cos(lat).clamp(min=1e-4)`,
  per-channel mean, then mean over channels). The clamp matters: the grid includes both poles. It subclasses
  `BaseLoss` only so the gen2 trainer hands it the whole batch dict.
- **`make_scaler`**: writes the Mixed_Modal statistics into a bridgescaler JSON. It reproduces
  `scaler_spatial_PS.json` byte for byte with bridgescaler 0.8.6 and 0.9.0.

## Known differences from HF

- **Not bit-identical training.** Data order, GPU count, nondeterministic kernels and float operation order differ,
  as between any two HF runs. The arithmetic of each step is the same up to float32 rounding (bridgescaler
  normalizes in float32; gen1 normalized the xarray data in float64).
- **Mass-fixer baseline.** In training, HF's mass fixer took the dry-air mass of the *first* step's input as the
  target for every step; gen2 uses the current step's input. On CAMulator's hybrid-sigma grid the mass fixer only
  rescales PS, the water fixer only PRECT and the energy fixer only T, so each step's input already has the first
  step's dry mass (to float precision). No custom block is needed.
- **Validation detection.** `HFLoss` is built from its `args` only (`load_loss` passes no `validation` flag), so
  it treats `torch.no_grad()` as validation, which is how the gen2 trainer validates.
- **Training years.** HF meant to train on 1980-2012, but the gen1 year filter matched "1980" in every CESM file
  name, so it trained on 1980-2014, including its validation years. The config reproduces that; set
  `data.end_datetime: "2012-12-31"` for a held-out 2013-2014.

## Tests

Run from this directory (the parent `climate/` package does not import, so pytest has to start here):

```bash
cd climate/camulator_gen2 && pytest
```

The tracer and loss tests compare against line-by-line ports of the HF code and expect bitwise equality. To also
check `make_scaler` against the released JSON, set `CAMULATOR_MEAN_NC`, `CAMULATOR_STD_NC` and
`CAMULATOR_SCALER_JSON`.
