# CADETS Graph Construction

This stage assembles the final graph artifacts used for training and evaluation.

## Inputs

- `artifacts/features_model_ready/feature_manifest.json`
- `artifacts/node_retention/retention_manifest.json`
- `artifacts/edges_filtered/edge_manifest.json`

## Outputs

- `artifacts/graphs/graph_manifest.json`
- per-window graph folders:
  - `artifacts/graphs/train/`
  - `artifacts/graphs/val/`
  - `artifacts/graphs/test_2018-04-12/`
  - `artifacts/graphs/test_2018-04-13/`

Each window folder contains:

- `graph.pt`
- `nodes.tsv`
- `summary.json`

## Design

Each window is built as a separate graph.

The graph artifact contains:

- unified node indexing across all retained `process/file/network` nodes in that window
- `edge_index`
- `edge_type`
- edge attributes:
  - `edge_event_count`
  - `edge_first_timestamp_ns`
  - `edge_last_timestamp_ns`
- node labels:
  - `y`
  - `gt_mask`
  - `normal_mask`

## Feature Storage

To avoid forcing heterogeneous node types into one padded feature matrix, features are stored per feature group:

- `process_view__process_node`
- `process_view__file_node`
- `process_view__network_node`
- `file_view__process_node`
- `file_view__file_node`
- `network_view__process_node`
- `network_view__network_node`

For each group, `graph.pt` stores:

- `node_ids`: global node ids for rows present in that group
- `x`: model-ready numeric feature tensor

This keeps raw view-specific dimensionality intact and fits the later type-specific projection design.

## Metadata

`nodes.tsv` stores:

- `node_id`
- `node_uuid`
- `node_type`
- `node_type_id`
- `is_gt`
- `decision_reason`

This keeps string metadata out of `graph.pt` while preserving exact alignment for debugging and case analysis.
