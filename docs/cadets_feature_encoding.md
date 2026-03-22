# CADETS Feature Encoding

This stage converts retained-node cleaned features into graph-ready model inputs.

## Goal

The cleaned feature tables still contain a small number of text columns. GNN input features must be numeric, so this stage:

- keeps only a minimal set of metadata columns for alignment
- one-hot encodes low-cardinality categorical columns
- leaves high-cardinality raw strings out of the model feature matrix

## Inputs

- `artifacts/features_cleaned/feature_manifest.json`
- per-window cleaned TSV files under `artifacts/features_cleaned/{window}/`

## Outputs

- `artifacts/features_model_ready/feature_manifest.json`
- per-window encoded TSV files under `artifacts/features_model_ready/{window}/`
- per-window `metadata.json`

## Metadata Columns

These columns are retained only for alignment and graph construction:

- `node_uuid`
- `node_type`

They are not counted as numeric model features.

## Encoded Categorical Columns

Low-cardinality categorical vocabularies are fit on the `train` window only.

- `process_view__file_node`
  - `file_type`
- `file_view__file_node`
  - `file_type`
- `process_view__network_node`
  - `local_port_bucket`
  - `remote_port_bucket`
  - `external_remote_ip_flag`
- `network_view__network_node`
  - `local_port_bucket`
  - `remote_port_bucket`
  - `external_remote_ip_flag`
- `file_view__process_node`
  - `subject_type` if non-constant on train
- `network_view__process_node`
  - `subject_type` if non-constant on train

If a candidate column has only one category on train, it is dropped instead of encoded.

## Not Encoded

These raw text fields are intentionally excluded from model features in v1:

- `host_id`
- `local_address`
- `remote_address`

Reason:

- they are either high-cardinality or weakly stable across windows
- they are more useful as metadata/debug context than as first-pass numeric features

## Numeric Feature Handling

All numeric columns come from the previous cleaning stage:

- retained-node rows only
- missing numeric values filled with `0`
- standardized with train-window mean and standard deviation

This stage does not re-standardize numeric columns.

## Result

The final model-ready TSV for each feature group contains:

- metadata columns: `node_uuid`, `node_type`
- standardized numeric columns
- one-hot encoded categorical columns

These files are intended to be the direct feature source for graph construction.
