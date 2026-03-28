# GraphCast

**Paper:** Lam et al., "GraphCast: Learning skillful medium-range global weather forecasting," Science 2023.
https://www.science.org/doi/10.1126/science.adi2336

**Original code:** Google DeepMind — https://github.com/google-deepmind/graphcast (Apache 2.0)
Also NVIDIA PhysicsNeMo: https://github.com/NVIDIA/physicsnemo

## Architecture

GNN encoder-processor-decoder operating on a **kNN lat/lon graph**:

```
Grid (B, C, H, W)
→ flatten to nodes (B, H*W, C)
→ node_encoder (Linear → LayerNorm)
→ edge_encoder (Haversine features: Δlat, Δlon, dist → Linear → LayerNorm)
→ N × MessagePassingLayer
    EdgeMLP: edge_feat + src_node + dst_node → new edge_feat
    scatter_add: aggregate edges to nodes
    NodeMLP: node_feat + aggregated → new node_feat
→ node_decoder (Linear → LayerNorm)
→ reshape to (B, C_out, H, W)
```

kNN graph is **precomputed at construction** from the lat/lon grid and stored
as a buffer (moves to GPU with `.to(device)`).

## CREDIT implementation

`graphcast.py` — written from scratch following the paper and DeepMind reference.
Significant simplifications relative to production GraphCast:
- **Single-resolution lat/lon graph** (no multi-mesh icosahedral refinement).
  Production GraphCast uses a 6-level icosahedral multi-mesh; this is a
  single kNN graph on the rectilinear lat/lon grid.
- kNN neighbours via Haversine pairwise distance (O(N²) at construction,
  precomputed once).
- Edge features: Δlat, Δlon, Haversine distance only (no vector components).
- No separate encoder/decoder mesh — grid nodes are processed directly.
- No pretrained weights.

## Validation status

**Architectural smoke test only.** The multi-mesh is the core innovation in
production GraphCast; this implementation captures the message-passing spirit
but is structurally simpler. Expect it to train and converge but do not expect
it to match production GraphCast skill.

## CREDIT config

```yaml
model:
  type: graphcast
  in_channels: 70
  out_channels: 69
  img_size: [192, 288]
  latent_dim: 512        # node/edge feature dimension
  edge_dim: 256
  processor_depth: 16    # message-passing iterations
  k_neighbours: 6        # kNN connectivity; 6 ≈ icosahedral mesh
  mlp_hidden: 512
```

**Warning**: at 192×288 the pairwise distance matrix is (192×288)² = 3×10⁹ floats
— far too large. The kNN graph is built in chunks but still slow at full resolution.
Consider using a coarser grid or pre-saving the edge index to disk.

## Known caveats

- kNN graph construction is O(N²) in node count. At 1° global resolution
  (N = 181×360 ≈ 65k nodes) this is ~4 GB of pairwise distances. Construction
  is done once at init; cache the model after first load.
- `k_neighbours=6` approximates the connectivity of an icosahedral grid.
- CREDIT's flat tensor interface requires unflattening nodes at the end — the
  reshape assumes the lat/lon grid is regular.
