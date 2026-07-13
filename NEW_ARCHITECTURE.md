# GUNTAM v2 — edge-classification architecture (July 2026)

This document describes the v2 overhaul of the GUNTAM seeding model and its data
pipeline. The legacy path (`SeedTransformer`, `Train.py`, attention losses, CSV/h5
input) is untouched and still works; v2 is a parallel stack.

```
G4 cache (ROOT, sparse ids 0..45610)
   └─ dump_spacepoints_parquet.py            [CPU node, ACTS digitization on the fly]
        └─ spacepoints/particles parquet chunks   (/shared, ~107 GB, 37,050 events)
             └─ PrepareTensor (input_format=parquet, sort_by_m, edge_truth)
                  └─ per-bin tensors (.pt)
                       └─ TrainEdge.py  →  SeedEdgeTransformer + edge_bce_loss   [GPU]
                            └─ ONNX (hits, padding_mask, bin_phi_center) → (output, triplets)
                                 └─ batched beam search → seeds (unchanged consumer)
```

## 1. Data / preprocessing changes

| aspect | v1 | v2 |
|---|---|---|
| input format | per-event ACTS CSVs → `space_points_small*.csv` | parquet chunk pairs (`spacepoints_chunk*` + `particles_chunk*`), read by `GUNTAM/IO/Read_Parquet.py` |
| truth linking | CSV measurement-simhit maps | `measurement_particles_map` read on the ACTS whiteboard; particle kinematics via uproot keyed by **true event id** (ACTS `RootParticleReader` mis-indexes sparse id spaces — see `guntam_v2_parquet_pipeline/README.md`) |
| sort order | `r` | **`m = (R + ρ)/2`**, `R=√(x²+y²)`, `ρ=√(x²+y²+z²)`. The sort order *defines* "forward" for the candidate mask, the truth edges and beam search; the inference-side seeder must sort identically (`--sort_by_m`) |
| truth targets | all same-particle pairs in a bin (symmetric) | **layer-aware directed edges** (`--edge_truth`): per particle, hits are grouped by detector layer (`layer_key = volume<<12 \| layer`), occupied layers ordered by mean m, consecutive layers connected bipartitely, directed low-m → high-m. Same-layer pairs are never edges. Built by `PrepareTensor._build_truth_edge_tensors` |
| min truth hits | 9 hits/particle | **4 spacepoints/particle** (`min_sp_per_particle`, applied in `Read_Parquet`) |
| hit features | `x y z r phi eta` | `x y z r phi eta m layer_key` (locked layout, contract with the model) |
| pair weights | pv_weight / z0 brackets | unchanged, carried onto edges |

Binning (neighbor, φ) is unchanged, but note the **effective** bin width is
`2π / n_bins` with `n_bins = ceil(2π / bin_width)` — bin centers must be computed
from `n_bins`, not the nominal width.

## 2. Model — `GUNTAM/Seed/SeedEdgeTransformer.py`

Inputs (also the ONNX interface): `hits [B,N,8]`, `padding_mask [B,N(,1)]`,
**`bin_phi_center [B]`** (new — the rotation is in-graph so ONNX carries it).

### 2.1 Per-bin canonical frame (in-graph)

```
phi_rel = atan2(sin(φ−φc), cos(φ−φc));  x' = r·cos(phi_rel);  y' = r·sin(phi_rel)
```

Every bin looks like the same wedge, so the transformer does not relearn the same
physics at every angle. x' and y' get **distinct normalisation scales**: x' spans
the radial range (≤500 mm) while y' only spans the half-bin transverse offset
(~13 mm at bin_width 0.05), so the network resolves fine transverse distances.

### 2.2 Embedding (locked)

Fourier features with k = 0..14 on five renormalised coordinates plus a raw block:

```
[ {sin,cos}(2π·2^k · x'/500)   k=0..14
  {sin,cos}(2π·2^k · y'/13)
  {sin,cos}(2π·2^k · z/1000)
  {sin,cos}(2π·2^k · r/500)
  {sin,cos}(2π·2^k · phi_rel/1.5)
  phi_rel, r, m, z, η ]                    → Linear → dim_embedding (128)
```

φ and η are now fully Fourier-encoded (v1 passed only cosφ, sinφ, η raw). The raw
passthrough of φ_rel, r, m, z, η is kept (bin-invariant quantities only; absolute φ
is deliberately excluded — it would reintroduce the bin dependence the rotation
removes). Precision: the k=14 band needs more than fp32 phase accuracy, so the
argument reduction (mod 2π) runs in **float64** and only sin/cos run in fp32
(onnxruntime's CPU provider has no float64 Sin/Cos kernel; this split is exact).

### 2.3 Encoder

Unchanged transformer encoder blocks (4 layers, 2 heads, dim 128, FF ratio 2).

### 2.4 Edge head (replaces the pairwise matching attention)

Dense masked N×N MLP over candidate edges:

* Pair features: `[h_i, h_j, Δm, Δz, Δr, Δφ_rel]`. The first MLP layer is
  **factorised** (`W_src h_i + W_tgt h_j + W_δ δ`) so the largest intermediate is
  `[B,N,N,64]`, never `[B,N,N,2·128]`.
* **Candidate rules computed in-graph** (export cleanly to ONNX):
  `m_j − m_i > 5 mm` (forward direction + minimum edge length, from the
  seed_filter_dashboard `basic_5mm` study: keeps 99.9 % of truth edges) ∧
  `layer_key_i ≠ layer_key_j` (VIP distinct-layer) ∧ neither hit padded.
  Everything else is masked to **−inf**; `score = sigmoid(logit)` therefore gives
  masked edges an exact 0. ~60 %+ of the N² pairs die to the forward rule alone;
  in PU200 bins ~72 % of all pairs are masked overall.
* Output: strict **top-5** `(source, target, score)` triplets per hit — fixed ONNX
  shapes, *no* extra confidence threshold (we do not prune the current ~95 %
  efficiency; the beam-search `att_threshold` remains the only runtime knob).

### 2.5 Loss — `SeedLoss.edge_bce_loss`

Per-edge binary cross-entropy over the candidate set, replacing the attention
cross-entropies. Positives = layer-aware truth edges inside the candidate mask
(pruned-and-counted if the in-graph cuts make them impossible — training never
sees them); negatives = all other candidates. Class-balanced (each side sums to
~1), positive weights scaled by the pv/z0 scheme. Multiple valid targets per
source are natural here — no softmax competition between sibling truth edges,
which was the failure mode of `attention_next_loss` on same-layer duplicates.

### 2.6 Reconstruction

`batched_beam_search_seed_reconstruction` is unchanged — it consumes the same
`[B,N,5,3]` triplet format; scores are now sigmoid probabilities instead of
softmax rows.

## 3. Training — `GUNTAM/Seed/TrainEdge.py`

Mirrors `Train.py`'s file/event/bin-batch loop; requires `--edge_truth
--sort_by_m --hit_features x y z r phi eta m layer_key`. Per-epoch checkpoints,
tensorboard (`edge_bce/*`, `edges/pruned_fraction`), quick seed-level eval
(beam search + ≥3-common-hit truth match). `--export_onnx` exports the trained
model. GPU environment: `source /shared/elmenard/acts-work/activate_guntam_gpu.sh`
(dedicated CUDA venv; do NOT use the LCG/ACTS env for training).

## 4. Validation status

* `guntam_v2_parquet_pipeline/preflight_validation.py`: 30/30 checks (schema,
  min-SP filter, m-sort, layer-aware edges, −inf mask vs independent numpy
  reconstruction, BCE mapping incl. pruning, ONNX export + onnxruntime parity).
* 80-step optimisation smoke on real PU200 bins: edge BCE 2.96 → 0.28; mean
  truth-edge score 0.976 vs 0.039 for the average candidate.
* Dataset audit (`v2_20260703b`): 37,050 events / 2.31 B spacepoints, particle
  truth uniform (~11,956/event) across the whole sparse id range.

## 5. Open items

* φ_rel Fourier scale is the locked literal `1.5`; if it was meant as *1.5
  bin-widths* (0.075 rad), change `EdgeTransformerConfig.fourier_scales[4]`.
* ACTS-side seeder update for inference: sort by m, pass bin centers, consume the
  new ONNX I/O.
* Upstream ACTS fix for `RootParticleReader`/`RootVertexReader` ordinal indexing
  on sparse event-id spaces.
* AMP (mixed precision) in TrainEdge for ~2× throughput — not yet enabled.
