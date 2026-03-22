# CADETS Feature Extraction

This note locks the first feature-extraction stage after:

- auditable `events_raw` rebuild
- local `process_entities` / `file_entities` / `network_entities` ingest
- GT alignment and day-level split selection

## Scope

The current implementation exports seven groups:

- `process_view__process_node`
- `process_view__file_node`
- `process_view__network_node`
- `file_view__process_node`
- `file_view__file_node`
- `network_view__process_node`
- `network_view__network_node`

The two weakest second-order groups are still postponed for now:

- `file_view__network_node`
- `network_view__file_node`

## Split Policy

Feature statistics are computed per time window, not on the full CADETS period.

Current windows come from `artifacts/day_split/split_manifest.json`:

- `train`: `2018-04-02` to `2018-04-10`
- `val`: `2018-04-11`
- `test_2018-04-12`: `2018-04-12`
- `test_2018-04-13`: `2018-04-13`

This avoids train/test leakage and matches the earlier design decision:

- training features aggregate over the full train window
- validation features use the validation day only
- test features are exported per test day

## Output Layout

Default output root:

- `artifacts/features/`

Per window, the extractor writes:

- `process_view__process_node.tsv`
- `process_view__file_node.tsv`
- `process_view__network_node.tsv`
- `file_view__process_node.tsv`
- `file_view__file_node.tsv`
- `network_view__process_node.tsv`
- `network_view__network_node.tsv`
- `metadata.json`

And at the root:

- `feature_manifest.json`

## Current Feature Sets

### `process_view__process_node`

Focus: process behavior as a subject in the provenance graph.

Current fields:

- `total_events`
- `event_type_diversity`
- `unique_file_count`
- `unique_network_count`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `connect_count`
- `send_count`
- `recv_count`
- `accept_count`
- `create_object_count`
- `fork_count`
- `mmap_count`
- `modify_process_count`
- `close_count`
- `file_interaction_ratio`
- `network_interaction_ratio`
- `fork_ratio`

### `process_view__file_node`

Focus: file-like objects viewed through the processes that touch them.

Current fields:

- `total_process_context_events`
- `unique_process_count`
- `event_type_diversity`
- `read_by_process_count`
- `write_by_process_count`
- `open_by_process_count`
- `exec_by_process_count`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `network_active_process_count`
- `avg_network_events_of_accessing_processes`
- `max_network_events_of_accessing_processes`
- `network_active_process_ratio`

Static columns from local `file_entities` are also kept:

- `host_id`
- `file_type`
- `permission_value`
- `size_bytes`

### `file_view__file_node`

Focus: file-like objects as event objects.

Current fields:

- `total_accesses`
- `unique_process_count`
- `event_type_diversity`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `close_count`
- `create_object_count`
- `unlink_count`
- `rename_count`
- `modify_file_attr_count`
- `read_ratio`
- `write_ratio`
- `execute_ratio`
- `open_ratio`

Static columns from local `file_entities` are also kept:

- `host_id`
- `file_type`
- `permission_value`
- `size_bytes`

### `process_view__network_node`

Focus: network objects viewed through the processes that use them.

Current fields:

- `total_process_context_events`
- `unique_process_count`
- `event_type_diversity`
- `connect_count`
- `accept_count`
- `bind_count`
- `send_count`
- `recv_count`
- `file_active_process_count`
- `avg_file_events_of_using_processes`
- `max_file_events_of_using_processes`
- `file_active_process_ratio`

Static columns from local `network_entities` are also kept:

- `local_address`
- `remote_address`
- `local_port`
- `remote_port`
- `local_port_bucket`
- `remote_port_bucket`
- `external_remote_ip_flag`
- `ip_protocol`

### `file_view__process_node`

Focus: process nodes viewed through their file access behavior.

Current fields:

- `total_file_events`
- `unique_file_count`
- `event_type_diversity`
- `read_count`
- `write_count`
- `open_count`
- `execute_count`
- `close_count`
- `create_object_count`
- `unlink_count`
- `rename_count`
- `modify_file_attr_count`
- `unique_read_file_count`
- `unique_write_file_count`
- `read_ratio`
- `write_ratio`
- `execute_ratio`

Static columns from local `process_entities` are also kept:

- `host_id`
- `subject_type`
- `has_parent_flag`

### `network_view__network_node`

Focus: netflow objects as event objects.

Current fields:

- `total_net_events`
- `unique_process_count`
- `event_type_diversity`
- `connect_count`
- `accept_count`
- `bind_count`
- `send_count`
- `recv_count`
- `send_recv_ratio`
- `message_send_count`
- `message_recv_count`
- `close_count`

Static columns from local `network_entities` are also kept:

- `local_address`
- `remote_address`
- `local_port`
- `remote_port`
- `local_port_bucket`
- `remote_port_bucket`
- `external_remote_ip_flag`
- `ip_protocol`

### `network_view__process_node`

Focus: process nodes viewed through their network interaction behavior.

Current fields:

- `total_network_events`
- `unique_network_count`
- `event_type_diversity`
- `unique_remote_ip_count`
- `unique_remote_port_count`
- `connect_count`
- `accept_count`
- `bind_count`
- `send_count`
- `recv_count`
- `close_count`
- `external_network_count`
- `external_network_ratio`
- `high_risk_port_contact_count`

Static columns from local `process_entities` are also kept:

- `host_id`
- `subject_type`
- `has_parent_flag`

## Why Seven Groups For Now

This is still a deliberate staging choice.

The project already discovered two real data-infrastructure issues:

- `event_uuid` is not globally unique in CADETS
- several GT UUIDs are pipe-style IPC objects rather than clean `Subject/FileObject/NetFlowObject` matches

Because of that, the extractor currently focuses on the seven most stable groups:

- split-aware aggregation
- stable node typing from local entity tables
- process/file/network self-view features
- process-centered cross-view features

The two remaining second-order groups:

- `file_view__network_node`
- `network_view__file_node`

are still postponed until this seven-group baseline is verified.

## Unmatched GT Note

After rebuilding local entity tables, `7` GT UUIDs still remain unmatched:

- `1` is fully unseen in `events_raw`
- `6` are visible on `2018-04-12` and appear in `aue_pipe` / pipe-style IPC events

These are not force-mapped into the three main entity tables in this stage.

## Usage

Summary only:

```bash
export CADETS_PG_PASSWORD=1234
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_feature_extract.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json \
  --summary-only
```

Full export:

```bash
export CADETS_PG_PASSWORD=1234
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_feature_extract.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```
