# CADETS Node Retention Spec

This note extends the earlier file-only retention stage into a conservative three-type node retention stage.

## Goal

The current graph is already smaller after file-node filtering and event-to-typed-edge aggregation, but the full-batch baseline still needs another pass that:

- continues to preserve GT nodes
- further compresses low-value nodes
- applies to `process`, `file`, and `network` together
- keeps the workflow auditable

This stage is intentionally conservative for `process` and `network`.

## Pipeline Position

The order is now:

1. feature extraction
2. file-node retention
3. full node retention
4. typed-edge filtering on existing aggregated edges
5. graph construction

Important:

- this stage does **not** revisit `events_raw`
- it reads already exported feature TSVs
- it inherits file decisions from the previous file-retention stage

## Scope

This stage runs per window:

- `train`
- `val`
- `test_2018-04-12`
- `test_2018-04-13`

It produces keep/drop decisions for:

- `process`
- `file`
- `network`

## Inputs

Per window:

- `artifacts/features/<window>/process_view__process_node.tsv`
- `artifacts/features/<window>/network_view__network_node.tsv`
- `artifacts/features/<window>/process_view__network_node.tsv`
- `artifacts/retention/<window>/file_keep_list.tsv`
- `artifacts/retention/<window>/summary.json`

Global:

- `artifacts/gt_entity_alignment/per_uuid.tsv`

## Retention Logic

### File Nodes

File-node decisions are inherited from the existing file-retention stage.

### Process Nodes

Hard keep if any of the following is true:

- GT process UUID
- `execute_count > 0`
- `fork_count > 0`
- `modify_process_count > 0`
- any network activity: `connect/send/recv/accept > 0`
- `write_count > 0`
- `create_object_count > 0`

Soft keep if not hard-kept and any of the following is true:

- `total_events > 10`
- `unique_file_count > 1`
- `unique_network_count > 0`
- `has_parent_flag = 1` and `total_events > 3`

Otherwise drop as `drop_low_value_process`.

This is meant to remove only obviously weak process leaves.

### Network Nodes

Hard keep if any of the following is true:

- GT network UUID
- any connection-like activity: `connect/accept/bind > 0`
- any I/O activity: `send/recv/message_send/message_recv > 0`

Soft keep if not hard-kept and any of the following is true:

- `unique_process_count >= 2`
- `file_active_process_count >= 1`
- `total_process_context_events > 3`

Otherwise drop as `drop_low_value_network`.

This is intentionally conservative because many network objects are externally facing and we do not want to remove useful sockets too aggressively.

## GT Audit Requirement

For every node type and every window, the stage must report:

- original candidate count
- retained count
- dropped count
- GT candidate count
- GT retained count
- GT dropped count
- exact dropped GT UUID list
- counts by decision reason

This is a hard requirement.

## Output Artifacts

Per window:

- `artifacts/node_retention/<window>/process_keep_decisions.tsv`
- `artifacts/node_retention/<window>/process_keep_list.tsv`
- `artifacts/node_retention/<window>/process_drop_list.tsv`
- `artifacts/node_retention/<window>/dropped_gt_process_list.tsv`
- `artifacts/node_retention/<window>/file_keep_list.tsv`
- `artifacts/node_retention/<window>/network_keep_decisions.tsv`
- `artifacts/node_retention/<window>/network_keep_list.tsv`
- `artifacts/node_retention/<window>/network_drop_list.tsv`
- `artifacts/node_retention/<window>/dropped_gt_network_list.tsv`
- `artifacts/node_retention/<window>/summary.json`

Root:

- `artifacts/node_retention/retention_manifest.json`

## Notes

- This stage is not the final compression stage.
- After node retention, edges must be filtered from the existing `typed_edges.tsv` files rather than recomputed from raw events.
