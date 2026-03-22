# CADETS File Node Retention Spec

This note locks the first executable file-node retention policy for the full-batch graph baseline.

## Goal

The current split-aware feature extraction keeps all active nodes, which is useful for auditing and ablation, but too heavy for direct full-batch graph training.

The immediate goal of this stage is:

- reduce graph size before graph construction
- preserve all process and network nodes
- aggressively compress low-value file leaf nodes
- guarantee that active GT file nodes are not dropped by the retention rule

This spec applies after feature extraction and before graph construction.

## Scope

This retention policy is applied per window:

- `train`
- `val`
- `test_2018-04-12`
- `test_2018-04-13`

It only decides whether a **file node** is kept in the graph for the current window.

For the first full-batch baseline:

- all `process` nodes are retained
- all `network` nodes are retained
- only `file` nodes are filtered

## Inputs

For each window, the retention script will read:

- `artifacts/features/<window>/file_view__file_node.tsv`
- `artifacts/features/<window>/process_view__file_node.tsv`
- `artifacts/features/<window>/metadata.json`

And the GT alignment outputs:

- `artifacts/gt_entity_alignment/per_uuid.tsv`
- `artifacts/gt_entity_alignment/summary.json`

## Required Columns

The retention rule depends on these columns from `file_view__file_node.tsv`:

- `node_uuid`
- `total_accesses`
- `unique_process_count`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `close_count`
- `create_object_count`
- `unlink_count`
- `rename_count`
- `modify_file_attr_count`

The retention rule may also use these columns from `process_view__file_node.tsv`:

- `node_uuid`
- `network_active_process_count`
- `avg_network_events_of_accessing_processes`
- `max_network_events_of_accessing_processes`
- `network_active_process_ratio`

## Node Universe

For a given window, the candidate file-node universe is:

- every row in `artifacts/features/<window>/file_view__file_node.tsv`

This means:

- we do **not** try to retain inactive file UUIDs that do not appear in the current window feature export
- a GT file node can only be retained if it is active in the current window feature table

## Retention Logic

The rule is deterministic and ordered.

### Step 1: Hard Keep

A file node is retained immediately if **any** of the following is true:

1. `node_uuid` is a GT file UUID
2. `execute_count > 0`
3. `write_count > 0`
4. `create_object_count > 0`
5. `unlink_count > 0`
6. `rename_count > 0`
7. `modify_file_attr_count > 0`

The hard-keep stage exists to preserve:

- labeled attack-relevant files
- behaviorally strong files
- files involved in mutation, execution, creation, or deletion semantics

### Step 2: Soft Keep

If a file node was not hard-kept, retain it if **any** of the following is true:

1. `total_accesses >= 3`
2. `unique_process_count >= 2`
3. `network_active_process_count >= 1`

The soft-keep stage preserves:

- shared files
- files that are not one-off leaves
- files touched by processes that also show network activity

### Step 3: Drop

Drop the file node if it satisfies **all** of the following:

1. it is not hard-kept
2. it is not soft-kept

This means the first full-batch baseline drops low-information file leaves such as:

- single-process files
- one-off read/open/close files
- files with weak local behavior and no process-network context

## Output Artifacts

For each window, the retention stage should write:

- `artifacts/retention/<window>/file_keep_decisions.tsv`
- `artifacts/retention/<window>/file_keep_list.tsv`
- `artifacts/retention/<window>/file_drop_list.tsv`
- `artifacts/retention/<window>/summary.json`

And one root-level manifest:

- `artifacts/retention/retention_manifest.json`

## `file_keep_decisions.tsv` Schema

Each row corresponds to one candidate file node in the current window.

Required columns:

- `node_uuid`
- `is_gt_file`
- `hard_keep`
- `soft_keep`
- `keep`
- `decision_reason`
- `total_accesses`
- `unique_process_count`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `create_object_count`
- `unlink_count`
- `rename_count`
- `modify_file_attr_count`
- `network_active_process_count`
- `network_active_process_ratio`

`decision_reason` must be one of:

- `gt_file`
- `execute`
- `write`
- `create_object`
- `unlink`
- `rename`
- `modify_file_attr`
- `total_accesses_ge_3`
- `shared_by_multiple_processes`
- `network_active_process`
- `drop_low_value_leaf`

## Graph Construction Contract

After retention, graph construction must use the following node policy:

- keep all process nodes present in the current window
- keep all network nodes present in the current window
- keep only file nodes where `keep = 1`

And the following edge policy:

- retain an edge only if both endpoint nodes survive node retention

No extra edge rescue logic is used in the first baseline beyond GT file hard-keep.

## Acceptance Checks

For each window, the retention stage must report:

- original file node count
- retained file node count
- dropped file node count
- file retention ratio
- GT file count in the candidate universe
- GT file retained count
- GT file retention ratio
- retained file count by reason

The first baseline must satisfy:

1. active GT file retention ratio = `1.0`
2. retained file node count is strictly less than original file node count
3. retained file node count is materially reduced, with a target reduction of at least `30%`

## Notes

- This spec filters only file nodes because file nodes are the main driver of graph blow-up.
- The `7` unmatched GT UUIDs are not force-mapped here; they remain outside this file-node rule.
- If this first policy is still too large for full-batch graph training, the next tightening step should modify only the soft-keep rules, not the hard-keep rules.
- The hard-keep rules are intentionally generous because avoiding GT loss is more important than maximum compression in the first full-batch baseline.
