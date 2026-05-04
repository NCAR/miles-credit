# CREDIT Config Directory

## Start here

| File | What it is |
|------|-----------|
| `example-v2026.2.yml` | **Fully annotated gen2 reference — every option explained** |
| `example-v2026.1.0.yml` | Fully annotated gen1 reference (superseded) |

For working starting points, see `gen_2/examples/` below.

## Directory structure

```
config/
  gen_2/                    Current generation (era5-gen2 trainer, nested data schema)
    examples/
      example-v2026.2.yml           Annotated reference config (CrossFormer, 1° ERA5, 6h)
      wxformer_era5_025deg_6hr.yml  WXFormer, 0.25° ERA5 pressure-level, 6h
      wxformer_npj_era5_028deg.yml  WXFormer, model-level ERA5, 0.28°
      multi_source_data.yaml        Multi-source data configuration example
    smoke/                  CI smoke-test configs (not for production use)

  gen_1/                    Previous generation configs (kept for reference)
    applications/
      climate/        Long climate rollout
      diffusion/      Diffusion model variants
      domain_parallel/ Domain-decomposition training
      downscaling/    Statistical downscaling, CONUS404
      ensemble/       Ensemble training and evaluation (CAM, ERA5)
      ic_opt/         Initial condition optimization
      other_models/   FuXi, Swin, UNet, GraphCast, Samudra
      physics/        Physics post-block examples (mass/water/energy conservation)
    example/          Annotated gen1 reference config
    arXiv_2024/       Configs used in the 2024 arXiv paper (reproducibility)
    archive/
      v1/             Superseded v1 configs
    old_configs/      Pre-2024 configs
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
