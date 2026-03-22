# CADETS Retained-Node Feature Cleaning Spec

This stage cleans and standardizes feature tables after node retention and before graph construction.

## Goal

The earlier feature extraction stage intentionally keeps rich, partially noisy feature tables.

Before graph construction, we now want a cleaner feature interface that:

- only keeps rows whose nodes survived node retention
- removes clearly unusable columns
- fills numeric missing values deterministically
- standardizes numeric features using train-window statistics only

## Pipeline Position

The order is now:

1. feature extraction
2. feature quality audit
3. file retention
4. full node retention
5. retained-node feature cleaning
6. filtered typed-edge construction
7. graph construction

## Scope

This stage processes all seven current feature groups:

- `process_view__process_node`
- `process_view__file_node`
- `process_view__network_node`
- `file_view__file_node`
- `file_view__process_node`
- `network_view__network_node`
- `network_view__process_node`

## Inputs

- `artifacts/features/feature_manifest.json`
- `artifacts/node_retention/retention_manifest.json`
- per-window feature TSV files
- per-window keep lists from `artifacts/node_retention/<window>/`

## Row Filtering Rule

For each feature file:

- keep only rows whose `node_uuid` appears in the retained list for that entity type
- use `process_keep_list.tsv` for `*_process_node.tsv`
- use `file_keep_list.tsv` for `*_file_node.tsv`
- use `network_keep_list.tsv` for `*_network_node.tsv`

## Train-Based Column Screening

Column screening is defined per feature group using retained `train` rows only.

### Text Columns

Drop a text column if it is fully missing on retained train rows.

This catches columns like:

- `permission_value`
- `ip_protocol`

when they are empty everywhere.

### Numeric Columns

For numeric columns:

1. treat missing values as `0`
2. compute train retained statistics
3. drop the column if it becomes constant after fill

This removes exact constant columns such as all-zero numeric features, while keeping rare but non-constant features.

## Standardization

For every kept numeric column:

- fit `mean` and `std` on retained `train` rows only
- apply the same transform to `train`, `val`, and both `test` windows

Transform:

- `x_clean = (x_filled - mean_train) / std_train`

No target leakage is allowed.

## Outputs

Per window:

- `artifacts/features_cleaned/<window>/*.tsv`
- `artifacts/features_cleaned/<window>/metadata.json`

Root:

- `artifacts/features_cleaned/feature_manifest.json`

## Notes

- Original feature files remain untouched.
- This stage is intentionally conservative: it removes only fully empty or constant columns, not merely sparse ones.
