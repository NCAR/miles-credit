# Model Presets

Named presets let you load a pretrained model — and run inference — with almost no configuration.
For NCAR campaign models, the preset also bundles a **default config** (data paths, normalization
stats, variable lists) so a two-line config is enough to run a full rollout.

---

## Usage

### Minimal config (NCAR campaign models)

For presets that include a `default_config`, only `model: preset:` and your `save_loc` are required:

```yaml
# my_config.yml
model:
  preset: wxformer-v2-025deg-6h

save_loc: /glade/derecho/scratch/$USER/my_run
```

```bash
credit_rollout_realtime --config my_config.yml   # inference
credit_train            --config my_config.yml   # fine-tune from the pretrained checkpoint
```

The preset fills in the architecture parameters, pretrained checkpoint path, and the full
`data:` / `predict:` / `trainer:` blocks from the canonical campaign config automatically.

### Overriding defaults

Any key you provide overrides the preset (and its default_config).  User-supplied values always win:

```yaml
model:
  preset: wxformer-v2-025deg-6h
  pretrained_weights: /path/to/my_finetuned.pt   # use a different checkpoint
  ff_dropout: 0.05                                # change dropout

data:
  forecast_len: 40   # override just one data key; rest come from default_config
```

### Adapting to different channels or resolution

The backbone (transformer blocks) still transfers even if you change the channel count or grid size.
Only the embedding and head layers train from scratch:

```yaml
model:
  preset: stormer-1.40625deg
  in_channels: 100   # your channel count — embedding trains from scratch
  out_channels: 99
  img_size: [192, 288]
```

---

## Available Presets

| Preset name | Model | Resolution | Default config | Key match |
|-------------|-------|:----------:|:--------------:|:---------:|
| `wxformer-v2-025deg-6h` | WXFormer v2 CrossFormer | 0.25° 6h | ✓ | 100% |
| `wxformer-025deg-1h` | WXFormer v2 CrossFormer | 0.25° 1h | ✓ | 100% |
| `wxformer-1deg-6h` | WXFormer CrossFormer | 1° 6h | — | 100% |
| `fuxi-025deg-6h` | FuXi U-Transformer | 0.25° 6h | ✓ | 100% |
| `stormer-1.40625deg` | Stormer ViT | 1.40625° 6h | — | 194/295 (24-block ViT backbone) |
| `climax-1.40625deg` | ClimaX ViT | 1.40625° | — | 105/130 (8-block ViT backbone) |
| `aurora-0.1deg` | Aurora Perceiver3D | 0.1° 6h | — | ~74% (full Perceiver3D+Swin3D) |

**Default config ✓** means the preset bundles a full working config (data paths, variable lists,
normalization stats) — a two-line user config is sufficient for a rollout on Casper/Derecho.

---

## How It Works

Preset files are plain YAML stored in [`credit/models/presets/`](../../credit/models/presets/).
When `preset:` is detected in your config, CREDIT merges three layers (lowest → highest priority):

1. **`default_config`** — a full working config bundled with the checkpoint (data paths, variables, etc.)
2. **Preset YAML** — architecture parameters and `pretrained_weights` path
3. **Your config** — whatever you wrote; always wins

```
default_config  <  preset model section  <  your config
```

No code changes are required to add a new preset — just drop a `.yml` file in `credit/models/presets/`.

### Example preset file (with default_config)

```yaml
# credit/models/presets/wxformer-v2-025deg-6h.yml
default_config: /glade/campaign/cisl/aiml/credit/pretrained_weights/wxformer_6h_hf/finetune_final/model_predict.yml
type: crossformer
pretrained_weights: /glade/campaign/cisl/aiml/credit/pretrained_weights/wxformer_6h_hf/finetune_final/best_model_checkpoint.pt
frames: 1
image_height: 640
image_width: 1280
levels: 16
channels: 4
surface_channels: 7
dim: [128, 256, 512, 1024]
depth: [2, 2, 8, 2]
# ... etc
```

---

## Adding a New Preset

1. Train a model (or locate pretrained weights).
2. Create `credit/models/presets/<your-preset-name>.yml` with the arch params and `pretrained_weights:` path.
3. Optionally add `default_config: /path/to/model_predict.yml` to bundle a full working config.
4. That's it — the preset is immediately available with `model: preset: <your-preset-name>`.
5. Add a row to the table above and to [`credit/models/MODELS.md`](../../credit/models/MODELS.md).
