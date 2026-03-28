# AIFS — Artificial Intelligence Forecasting System

**Paper:** Lang et al., "AIFS – ECMWF's data-driven forecasting system," 2024.
https://arxiv.org/abs/2406.01465

**Original code:** ECMWF Anemoi — https://github.com/ecmwf/anemoi-models (Apache 2.0)
Also NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo (Apache 2.0)

## Architecture

**Graph transformer** operating on the lat/lon grid.
Key innovation: Fourier-based positional encoding and a scalable attention
processor (GNN or transformer, depending on config).

```
(B, C, H, W) flat
→ split: surf + atmos×levels + static
→ node features: flatten to (B, H*W, n_in)
→ positional encoding (Fourier features of lat/lon)
→ node encoder (Linear + LayerNorm)
→ N × processor blocks (self-attention + MLP)
→ node decoder (Linear + LayerNorm)
→ reshape → (B, C_out, H, W)
```

AIFS uses a GNN on a reduced Gaussian grid in production; this implementation
uses a lat/lon grid with a transformer processor for simplicity.

## CREDIT implementation

`aifs.py` — translated from PhysicsNeMo / Anemoi reference.
`CREDITAifs` wraps with CREDIT's flat `(B, C, H, W)` interface.
Channel layout: `[surf_vars | atmos_vars × levels | static_vars]`

`**aifs_kwargs` forwarded to `AIFSProcessor(...)` for architectural control.

Significant simplifications:
- **No graph structure** — uses full self-attention on flattened grid nodes
  (production AIFS uses a sparse GNN on a reduced Gaussian grid).
- Fourier positional encoding of lat/lon instead of graph edge features.
- No multi-scale graph hierarchy.

## Validation status

**Architectural smoke test only** at small variable config.
The lat/lon transformer is a reasonable approximation of the AIFS processing
stage but is not structurally identical to ECMWF's production system.
Not weight-compatible with ECMWF's checkpoint (also proprietary/restricted).

## CREDIT config

AIFS uses named variable lists, not `in_channels`. Channel layout is
`[surf_vars | atmos_vars × levels | static_vars]`.

```yaml
model:
  type: aifs
  surf_vars: ["2t", "10u", "10v", "msl"]
  atmos_vars: ["z", "u", "v", "t", "q"]
  static_vars: ["lsm", "z", "slt"]
  atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  # aifs_kwargs forwarded to AIFSProcessor(...):
  aifs_kwargs:
    hidden_dim: 512
    num_layers: 16
    num_heads: 16
    mlp_ratio: 4.0
    window_size: 0        # 0 = full attention; set >0 for windowed (memory efficient)
```

**Memory warning**: full self-attention is O(N²) in the number of grid nodes.
At 1° resolution (N ≈ 55k) this is ~12 GB per batch element. Use a
coarser grid or set `window_size > 0` for windowed attention.

## Known caveats

- Production AIFS runs on ECMWF's proprietary reduced Gaussian grid with a
  sparse GNN; this version is a lat/lon transformer approximation.
- ECMWF's pretrained weights are not publicly available.
- Memory scales as O(N²) with N = H×W at global resolution. For 1° ERA5
  (N ≈ 65k), full self-attention is impractical — use a smaller `img_size`
  or replace the processor with a windowed variant.
