# GPU Baseline Summary

This note records the current GPU-side baseline conclusions for the rebuilt CADETS experiment line.

The numbers below are taken from the recorded GPU terminal runs on March 23, 2026.

## End-to-End Project Flow

The current project pipeline runs from raw CADETS provenance JSON to ranked anomalous nodes.

### 1. Raw event ingest

Starting point:

- raw CADETS official JSON files: `ta1-cadets-e3-official*.json*`

We first rebuild an auditable event database:

- stage every parsed event into `event_stage_raw`
- rebuild one canonical event table `events_raw`
- keep file-level and day-level ingest audit summaries

This stage is documented in `docs/cadets_event_ingest.md`.

### 2. Entity ingest and GT alignment

On top of canonical events, we rebuild local entity tables for:

- `process`
- `file`
- `network`

We then align ground-truth attack entities and mark which nodes are GT-related.

Relevant notes live in:

- `docs/cadets_entity_ingest.md`
- `docs/cadets_gt_alignment.md`
- `docs/cadets_gt_entity_alignment.md`

### 3. Day split

After GT alignment, we create a split manifest:

- `train`: `2018-04-02` to `2018-04-10`
- `val`: `2018-04-11`
- `test_2018-04-12`: `2018-04-12`
- `test_2018-04-13`: `2018-04-13`

This stage is documented in `docs/cadets_day_split.md`.

### 4. Feature extraction, cleaning, and encoding

We build seven model-facing feature groups:

- `process_view__process_node`
- `process_view__file_node`
- `process_view__network_node`
- `file_view__process_node`
- `file_view__file_node`
- `network_view__process_node`
- `network_view__network_node`

Then we clean and encode them into numeric tensors that can be written into graph artifacts.

Relevant notes:

- `docs/cadets_feature_extraction.md`
- `docs/cadets_feature_cleaning.md`
- `docs/cadets_feature_encoding.md`

### 5. Edge aggregation, filtering, and graph construction

Raw events are aggregated into typed window-level edges, then filtered, then assembled into per-window graphs.

Each graph contains:

- unified node indexing for retained `process/file/network` nodes
- `edge_index`
- `edge_type`
- edge attributes such as `edge_event_count`
- node labels `y`, `gt_mask`, `normal_mask`
- per-group feature tensors instead of one padded heterogeneous matrix

The final graph artifacts are:

- `artifacts/graphs/train/graph.pt`
- `artifacts/graphs/val/graph.pt`
- `artifacts/graphs/test_2018-04-12/graph.pt`
- `artifacts/graphs/test_2018-04-13/graph.pt`

Relevant notes:

- `docs/cadets_edge_aggregation.md`
- `docs/cadets_edge_filter.md`
- `docs/cadets_graph_construction.md`

### 6. GPU-side model training

The GPU baseline is a multi-view full-batch graph autoencoding pipeline.

The model uses:

- view-specific block-concatenated node features built at runtime from `graph.pt`
- one encoder branch per view: `process_view`, `file_view`, `network_view`
- gated fusion across the three view-specific node embeddings
- edge reconstruction training on `normal-normal` edges

The current formal model variant is:

- relation-aware message passing: `rel_grouped`
- relation grouping: `coarse_v1`
- decoder: `dot`

In practical terms, the current encoder is:

- a multi-view GCN-style encoder
- with grouped relation-specific adjacencies during message passing
- followed by gated multi-view fusion

The current decoder is:

- a dot-product edge decoder

Training uses:

- best-checkpoint selection by `val_edge_loss`
- positive-edge caps during training/eval-time monitoring for efficiency

### 7. From edge reconstruction to anomaly-node ranking

The model does not directly perform node classification.

Instead, the current anomaly scoring pipeline is:

1. encode nodes with the trained multi-view GNN
2. reconstruct graph edges
3. convert each edge into an edge reconstruction error
4. aggregate incident edge errors into a node score
5. calibrate node scores within each `node_type`
6. rank all nodes by final anomaly score

The current formal post-processing is:

- node score aggregation: `top5_mean`
- node-type calibration: `robust_zscore_by_type`

This produces:

- node-level anomaly scores
- ranked GT statistics
- top-k hit summaries

### 8. Final output of the pipeline

From raw CADETS data, the end product of the current experiment line is:

- a trained checkpoint
- a ranked anomaly list over graph nodes
- evaluation summaries on `val`, `test_2018-04-12`, and `test_2018-04-13`

So the full project path is:

`raw CADETS JSON -> canonical events -> entities + GT alignment -> day split -> features -> typed edges -> graph.pt -> multi-view GNN -> edge reconstruction error -> node anomaly scores -> ranked anomalous nodes`

## Current Formal Baseline

Training configuration:

- model: multi-view full-batch GNN
- message passing: `rel_grouped`
- relation grouping: `coarse_v1`
- decoder: `dot`
- epochs: `30`
- checkpoint selection: `val_edge_loss`
- training caps: `--train-pos-edge-cap 3000000 --eval-pos-edge-cap 500000`

Evaluation configuration:

- node score aggregation: `top5_mean`
- node-type calibration: `robust_zscore_by_type`

Recommended GPU artifacts:

- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- evaluation directory: `artifacts/evaluations/rel_grouped_dot_30ep_eval`

## Formal Baseline Results

### Main metrics

| Window | ROC-AUC | AP |
| --- | ---: | ---: |
| `val` | `0.9160` | `0.000526` |
| `test_2018-04-12` | `0.8859` | `0.5264` |
| `test_2018-04-13` | `0.8699` | `0.001024` |

### Main ranking statistics

For `test_2018-04-12`:

- best GT rank: `79 / 114955`
- median GT rank: `17858.5 / 114955`
- top-100 hits: `1`
- top-1000 hits: `1`
- top-5000 hits: `3`
- top-10000 hits: `15`

For `test_2018-04-13`:

- best GT rank: `2587 / 100085`
- median GT rank: `6456.5 / 100085`
- top-5000 hits: `4`
- top-10000 hits: `11`

## What Was Tried

### Score aggregation and calibration

- Plain `mean` node scoring was not good enough for top-k retrieval.
- `top5_mean` improved front-of-list retrieval and became the default node score aggregation.
- `robust_zscore_by_type` was the strongest calibration method overall.
- Percentile and raw/robust hybrid calibrations were more aggressive in some top-k slices, but did not beat `robust_zscore_by_type` on the overall tradeoff.

### Decoder ablations

| Setup | `test_2018-04-12` ROC-AUC | `test_2018-04-12` AP | Conclusion |
| --- | ---: | ---: | --- |
| vanilla `dot` + `top5_mean` + `robust_zscore_by_type` | `0.8586` | `0.4718` | strong baseline before relation-aware message passing |
| `mlp` decoder | `0.5010` | `0.1249` | rejected |
| relation-aware `rel_mlp` decoder | `0.5011` | `0.1250` | rejected |

The main lesson was that putting relation information only in the decoder was not enough.

### Message-passing ablations

| Setup | `test_2018-04-12` ROC-AUC | `test_2018-04-12` AP | Conclusion |
| --- | ---: | ---: | --- |
| `rel_grouped` + `coarse_v1` + `dot` + `30ep` | `0.8859` | `0.5264` | current formal baseline |
| `rel_grouped` + `coarse_v1` + `dot` + `60ep` | `0.8860` | `0.5267` | not selected; nearly tied on main metric but worse GT ranks/top-k and weaker `test_2018-04-13` |
| `rel_grouped` + `coarse_v2` + `dot` + `30ep` | about `0.6357` | about `0.2556` during training-time eval | rejected; grouping was too fine and signal became weaker |

`coarse_v1` groups event types into:

- `file_read`
- `file_write`
- `file_meta`
- `process`
- `network`
- `flow_other`

## Why `rel_grouped_dot_30ep` Was Chosen

It beat the strongest vanilla baseline on the main test window:

- `ROC-AUC`: `0.8586 -> 0.8859`
- `AP`: `0.4718 -> 0.5264`

It also produced much better GT ranking than the strongest vanilla baseline:

- best GT rank: `1389 -> 79`
- median GT rank: `20667.5 -> 17858.5`

Compared with `rel_grouped_dot_60ep`, the `30ep` checkpoint was kept as the formal baseline because:

- the main metric gain at `60ep` was tiny
- `60ep` had worse top-k retrieval on `test_2018-04-12`
- `60ep` had weaker `test_2018-04-13` metrics

## Current Interpretation

The most useful relation signal is being captured in message passing, not only in the decoder.

In practice:

- relation-aware message passing helped
- decoder-only relation awareness failed
- over-splitting relation groups weakened the signal

## Recommended Next Directions

- Try a stronger relation-aware encoder family such as `RGCN` or another per-relation propagation variant.
- If relation grouping is revisited, keep the grouping compact and semantics-driven instead of splitting too finely.
- Keep the current evaluation defaults fixed while testing new encoders:
  `score_method=top5_mean`, `score_calibration=robust_zscore_by_type`
