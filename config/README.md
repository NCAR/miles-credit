# CREDIT Config Directory

## Start here

| File | What it is |
|------|-----------|
| `wxformer_1dg_6hr_v2.yml` | **ERA5 1° WXFormer — recommended starting point** |
| `wxformer_025deg_6hr_v2.yml` | ERA5 0.25° WXFormer (full-res, more GPU memory) |
| `starter_v2.yml` | Minimal template with `# USER SETTINGS` comments |
| `example-v2026.1.0.yml` | Fully annotated reference — every option explained |

All four use `trainer.type: era5-gen2` and the current data schema.
To generate a fresh config from a template: `credit init --grid 1deg -o my_config.yml`
To convert a v1 config: `credit convert -c old_model.yml`

## Subdirectories

```
config/
  applications/
    realtime/       GFS-initialized real-time forecast configs
    ensemble/       Ensemble training and evaluation (CAM, ERA5)
    downscaling/    Statistical downscaling, CONUS404
    diffusion/      Diffusion model variants
    other_models/   FuXi, Swin, UNet, GraphCast, Samudra
    climate/        Long climate rollout configs
    physics/        Physics post-block examples (mass/water/energy conservation)
    ic_opt/         Initial condition optimization
  data/             Dataset-specific configs (ERA5 schema variants, multi-source)
  dev/              Development / CI test configs (not for production use)
  archive/
    v1/             Superseded v1 configs (kept for reference)
    old_configs/    Pre-2024 configs
    arXiv_2024/     Configs used in the 2024 arXiv paper (reproducibility)
```

## PBS / job submission

Set your allocation and cluster settings in the `pbs:` block of your config:

```yaml
pbs:
    project: "YOUR_ACCOUNT"   # PBS -A charge code
    conda: "credit-derecho"   # conda env name or full path
    walltime: "12:00:00"
    nodes: 1
    ngpus: 4
    ncpus: 64
    mem: '480GB'
    queue: 'main'
```

Then submit with: `credit submit --cluster derecho -c my_config.yml`

CLI flags override config values. See `credit submit --help`.
