# CADETS Current Stats

This note records the current retained-node, cleaned-feature, and filtered-edge scale after the latest preprocessing stages.

## Feature Dimensions

Numeric feature dimensions below are the current cleaned dimensions that are ready for graph construction.

| Feature Group | Numeric Dim | Dropped Columns |
| --- | ---: | --- |
| `process_view__process_node` | 21 | `-` |
| `process_view__file_node` | 15 | `permission_value`, `size_bytes` |
| `process_view__network_node` | 13 | `ip_protocol`, `bind_count` |
| `file_view__process_node` | 18 | `-` |
| `file_view__file_node` | 16 | `permission_value`, `size_bytes` |
| `network_view__process_node` | 14 | `bind_count` |
| `network_view__network_node` | 13 | `ip_protocol`, `bind_count` |

## Model-Ready Feature Dimensions

After low-cardinality categorical encoding, the effective GNN input dimensions are:

| Feature Group | Cleaned Numeric Dim | Encoded Categorical Dim | Final Model Dim |
| --- | ---: | ---: | ---: |
| `process_view__process_node` | 21 | 0 | 21 |
| `process_view__file_node` | 15 | 3 | 18 |
| `process_view__network_node` | 13 | 10 | 23 |
| `file_view__process_node` | 18 | 0 | 18 |
| `file_view__file_node` | 16 | 3 | 19 |
| `network_view__process_node` | 14 | 0 | 14 |
| `network_view__network_node` | 13 | 10 | 23 |

Categorical encoding currently includes:

- `file_type` for file-node feature groups
- `local_port_bucket`, `remote_port_bucket`, and `external_remote_ip_flag` for network-node feature groups

These model-ready outputs live under `artifacts/features_model_ready`.

## Current Window-Level Graph Scale

The counts below use:

- retained nodes from `artifacts/node_retention`
- cleaned features from `artifacts/features_cleaned`
- typed edges from `artifacts/edges_filtered`

| Window | Process Nodes | File Nodes | Network Nodes | Total Nodes | Retained GT Nodes | Typed Edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 164,022 | 304,276 | 72,436 | 540,734 | 8 | 7,309,391 |
| `val` | 20,405 | 78,730 | 6,211 | 105,346 | 9 | 918,147 |
| `test_2018-04-12` | 20,879 | 89,117 | 4,959 | 114,955 | 12,846 | 911,816 |
| `test_2018-04-13` | 18,718 | 75,434 | 5,933 | 100,085 | 16 | 829,676 |

## GT Breakdown

Retained GT node counts by type:

| Window | GT File | GT Process | GT Network | Total Retained GT |
| --- | ---: | ---: | ---: | ---: |
| `train` | 8 | 0 | 0 | 8 |
| `val` | 9 | 0 | 0 | 9 |
| `test_2018-04-12` | 12,806 | 15 | 25 | 12,846 |
| `test_2018-04-13` | 14 | 2 | 0 | 16 |

## Notes

- These numbers exclude the `unmatched` GT objects that are not represented as standard `process/file/network` nodes.
- The train graph currently remains the largest artifact in the pipeline and is the main reference point for deciding whether further graph compression is necessary.
