# FuXi-ENS

**Paper:** Chen et al., "FuXi-ENS: A machine learning model for medium-range ensemble weather forecasting," Science Advances 2025.
https://arxiv.org/abs/2405.05925

## Architecture

FuXi-ENS adds a **VAE bottleneck** to the latent space of a deterministic weather model so that sampling different latent vectors `z` produces diverse but physically consistent ensemble members — without any post-hoc perturbation scheme.

CREDIT simplification: the original Swin-based encoder/decoder is replaced by a plain ViT (same building blocks as CREDITStormer) to keep the code self-contained.

```
(B, C_in, H, W)
  → PatchEmbed (Conv2d, patch_size)  →  (B, N, embed_dim)
  → positional embedding
  → depth/2 × Block  [encoder]
  →  VAEBottleneck (optional)
       Linear(embed_dim → 2*z_dim) → mu, logvar
       z = mu + eps * exp(0.5 * logvar)   [train: stochastic]
       z = mu                             [eval:  deterministic]
       Linear(z_dim → embed_dim) added to h
       kl_loss = -0.5 * mean(1 + logvar - mu² - exp(logvar))
  →  depth/2 × Block  [decoder]
  → LayerNorm
  → Linear(embed_dim → C_out * patch_size²)
  → fold patches                         →  (B, C_out, H, W)
```

Setting `use_vae=False` disables the bottleneck entirely, yielding a **plain deterministic ViT** with no extra overhead.

## CREDIT implementation notes

- `kl_loss` is stored as `model.last_kl_loss` (a scalar tensor) after every forward pass. Trainers can add it to the prediction loss without modifying the forward signature:
  ```python
  pred = model(x)
  loss = mse(pred, target) + kl_weight * model.last_kl_loss
  ```
- `last_kl_loss` is always `0.0` when `use_vae=False`.
- To draw **stochastic ensemble members** at inference, call `model.train()` before sampling, then `model.eval()` to restore deterministic behaviour. Or set `model.vae.training = True` directly.
- Spatial padding is applied internally so any `(H, W)` divisible by `patch_size` is accepted.
- 5-D trainer input `(B, C, T, H, W)` is handled automatically (T channels merged into C).

## Validation status

**Architectural smoke test only** (correct shape, no NaN, gradients flow, KL > 0 in training mode).
Not yet trained to convergence or compared against published spread/skill scores.

## CREDIT config

```yaml
model:
  type: fuxi_ens
  in_channels: 70        # total input channels (atmos + surface + static)
  out_channels: 69       # total output channels (no static)
  img_size: [192, 288]   # (H, W) of your lat/lon grid
  patch_size: 4          # spatial patch size
  embed_dim: 512         # ViT width
  depth: 8               # total transformer depth (split evenly enc/dec)
  num_heads: 8           # attention heads
  mlp_ratio: 4.0
  drop_rate: 0.0
  use_vae: true          # set false for deterministic mode
  z_dim: 64              # VAE latent dimension
```

Suggested production config: `embed_dim=1024, depth=12, num_heads=16, z_dim=128`.

Trainer KL weight recommendation: start at `1e-5` and anneal up to `1e-3` over the first epoch to avoid posterior collapse.

## Known caveats

- **Posterior collapse**: if KL weight is too large early in training the VAE ignores the latent code and all ensemble members are identical. Use KL annealing.
- Global attention is O(N²) in patch count. At 1° with `patch_size=4` you get N≈3456 tokens — manageable on A100-80GB at batch 1–2. Use `patch_size ≥ 4`.
- The CREDIT ViT backbone differs from the paper's Swin encoder; expect different skill/spread tradeoffs. The VAE mechanism itself is identical.
