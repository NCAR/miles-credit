# camulator_smallscale: CAMulator fine-tuning experiments on small-scale variability

CREDIT custom objects and configs for fine-tuning the released CAMulator (epoch 69) with losses meant to restore the
small-scale variability that MSE training smooths out: AMSE, SDL noise with CRPS, and a wavelet loss. Each arm is
compared with a "null" fine-tune that keeps the original loss.

Everything builds on [`camulator_gen2`](../camulator_gen2/README.md), which reproduces the Hugging Face CAMulator
training setup in the gen2 pipeline. Each arm is that setup, then the shared fine-tune protocol, then one change.

## Install

```bash
pip install -e climate/camulator_gen2 -e climate/camulator_smallscale            # from a clone of miles-credit
pip install -e climate/camulator_gen2 -e climate/camulator_smallscale --no-deps  # on NCAR's shared conda envs
```

## Configs

`make_configs` builds each arm's config from `camulator_gen2/configs/camulator_hf.yml`:

```bash
# Derecho: glade paths, one save_loc per arm under /glade/derecho/scratch/$USER/CREDIT_runs/camulator_smallscale
python -m camulator_smallscale.make_configs --arm null --target derecho

# A laptop: glade paths swapped for local copies, 1981 only, batch 1, a few updates.
# --local-root is laid out like a miles-credit checkout's climate/ folder (data/ and assets/).
python -m camulator_smallscale.make_configs --arm null --target local \
    --local-root ~/PycharmProjects/miles-credit/climate --stage clone
credit check -c climate/camulator_smallscale/local_runs/null/null.yml
credit rollout -c climate/camulator_smallscale/local_runs/null/null.yml
credit train -c climate/camulator_smallscale/local_runs/null/null.yml
```

`--stage clone` puts the released checkpoint in `save_loc` as `checkpoint.pt`. On macOS that is a copy-on-write
clone, so it costs no disk space. Use `--stage link` only for rollouts: training writes `checkpoint.pt` in place,
which would overwrite the original through a symlink. Local runs go to `local_runs/` (git-ignored).

The shared fine-tune protocol (`FINETUNE` in `make_configs.py`):
- continue from epoch 69 with a fresh AdamW, using HF's final-stage settings (decoder `up_block1-4`, 3-step rollouts,
  constant lr 2.5e-5);
- count epochs from 0 and keep every epoch's checkpoint;
- default to 4 epochs of 300 updates.

Arms so far: `null` (HFLoss unchanged) and `amse`.

## Custom objects

| Name | Type | What it does |
|---|---|---|
| `amse_hf_loss` (`losses.AMSEHFLoss`) | loss | HFLoss plus the AMSE adjustment of Subich et al. (2025); see below |
| `start_state_perturbation` (`preblocks.StartStatePerturbation`) | preblock | Seeded start-state perturbation for twin runs and ensembles; put it under `preblocks.ic_only` |
| `mps_safe_hf_camulator` (`models.MPSSafeHFCamulator`) | model | `hf_camulator` that stays correct on Apple MPS (local configs only) |

### AMSE loss

By total wavenumber, the area-weighted MSE splits into an amplitude term and a decorrelation term weighted by
`sqrt(Px Py)`. AMSE weights the decorrelation term by `max(Px, Py)` instead. That way a scale that no longer matches
the truth cannot lower the loss by losing amplitude.

`AMSEHFLoss` is HFLoss plus the difference, `sum_l 2 [max(Px,Py) - sqrt(Px Py)] (1 - coh_l) / (4 pi)` per channel.
HF's channel weights apply on top.
- It is zero when the spectra match, and the loss then equals HFLoss (bit for bit when prediction equals truth).
- Validation is exactly HFLoss, so validation curves compare directly with the null arm.
- The constraint is on the global spectrum only, so variance can shift between latitudes. With HF's cos(lat)^0.4
  weights, part of the damping incentive remains at high latitudes.
- Spectra come from `spectral.SphericalSpectra`, truncated at total wavenumber 144 (about 280 km).

### Start-state perturbations

`kind: white` (Gaussian, std = `amplitude`), `band` (wavenumbers `lmin..lmax`, area-weighted RMS = `amplitude`) or
`file` (a precomputed field times `amplitude`). The noise depends only on `seed` and the forecast start time, so every
arm gets the same perturbation for the same start date:

```yaml
custom_objects:
  start_state_perturbation:
    object_type: preblock
    module_path: camulator_smallscale.preblocks
    module_name: StartStatePerturbation
preblocks:
  ic_only:
    perturb:
      type: start_state_perturbation
      args: {variables: [CESM/prognostic/3d/T], amplitude: 0.01, kind: white, seed: 1}
```

## Converting CESM output (`convert`)

`python -m camulator_smallscale.convert` turns raw CAM 6-hourly history (`cam.h1`, one file per day) into per-year
CREDIT zarrs, one time step per chunk. It needs only xarray and numpy, so it runs in NCAR's `npl` environment. The
recipes reproduce the training data: two days of 1981 match the existing training zarr for all 27 model variables to
float32 rounding. The module docstring lists every recipe.

```bash
python -m camulator_smallscale.convert --hist-dir <case>/atm/hist --case <case> --years 2015 2025 --out-dir <dir>
python -m camulator_smallscale.convert --hist-dir <case>/atm/hist --case <case> --years 2026 2048 --out-dir <dir> --forcing-only
python -m camulator_smallscale.convert verify --converted <zarr> --reference <zarr>
```

## Apple MPS

On MPS, `Conv2d` silently returns wrong values once `in_channels * kH * kW` exceeds about 2**16, still the case on
torch 2.13. CAMulator's first layer crosses it, and a forward pass on MPS then differs from CPU by up to ~10% of the
output range. The local configs therefore use `mps_safe_hf_camulator` (`camulator_smallscale.models`). It is
`hf_camulator` with those convolutions split into input-channel chunks when, and only when, they run on MPS. With it,
MPS matches CPU to 6e-5 (relative 7e-6). On CPU and CUDA it is bit-identical to `hf_camulator`, which the Derecho
configs keep.

## Tests

```bash
cd climate/camulator_smallscale && pytest
```
