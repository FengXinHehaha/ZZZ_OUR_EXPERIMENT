# File Feature V2 And Screening Plan

This note records the current `file`-focused feature redesign direction after the
support-aware scorer, history-aware scorer, and file-specific reranker studies.

The goal is not to replace the current strongest line immediately. Instead, the
goal is to define a cleaner `file` feature table plus a high-recall file
screening stage that can shrink the normal-file candidate pool before later
reranking or retraining.

## Current Strong Baseline To Keep

Keep the current strongest end-to-end line as the control group:

- scorer: `top5_mean_log_support_floor128_file_history_file_only`
- calibration: `robust_zscore_by_type`
- post-rerank: `file_rerank_support`
- sparse policy: `top_count=300`
- moderate policy: `top_count=200`
- dense policy: `window_median_plus_mad(k=20)`

## File Feature V2

### Keep

These signals have already shown value directly or indirectly:

- `total_accesses`
- `unique_process_count`
- `event_type_diversity`
- `read_count`
- `write_count`
- `execute_count`
- `rename_count`
- `unlink_count`
- `modify_file_attr_count`
- `network_active_process_count`
- `max_network_events_of_accessing_processes`
- file type one-hot columns
- graph support features:
  - `total_degree`
  - `in_degree`
  - `out_degree`
  - `unique_process_neighbors`
  - `unique_network_neighbors`
  - `incident_group_file_read`
  - `incident_group_file_write`
  - `incident_group_file_meta`
- history features:
  - `prev_day_present`
  - `prev_day_percentile`
  - `decayed_history_2d`
  - `streak_days`

### Suspected Redundancy

These are worth ablating before carrying them into a smaller `file_feature_v2`
table:

- `close_count`
- `open_ratio`
- `read_ratio`
- `write_ratio`
- `execute_ratio`
- `avg_network_events_of_accessing_processes`
- repeated `read/write/open/execute` counters across `file_view__file_node` and
  `process_view__file_node`
- repeated `unique_process_count` across file views
- repeated `event_type_diversity` across file views

### High-Value Additions

The raw extraction code already defines path-oriented file features that are not
yet used in the current model-ready file table:

- `unique_known_path_count`
- `temp_path_count`
- `config_path_count`
- `system_bin_path_count`
- `system_lib_path_count`
- `log_path_count`
- `user_home_path_count`
- `hidden_path_count`
- `script_path_count`
- `missing_path_count`

These should be the first new candidates added to `file_feature_v2`.

## File Screening V1

The first screening policy is intentionally high recall. A file node is kept if
any one of the following conditions holds:

- current global rank is within top `10000`
- present in the previous window
- `unique_process_count >= 2`
- `total_accesses >= 3`
- `write_count + execute_count + rename_count + unlink_count + modify_file_attr_count >= 1`
- `network_active_process_count >= 1`
- path-risk count is non-zero when path columns are available:
  - `temp_path_count`
  - `hidden_path_count`
  - `script_path_count`
  - `system_bin_path_count`

This stage is not meant to be a final anomaly decision. It is only meant to
remove a large number of obviously ordinary file nodes while preserving as much
GT-file recall as possible.

The first executable comparison should evaluate these screening variants:

- `all_files`
- `score_top_rank`
- `screen_v1`
- `screen_v1_strict`
- `screen_v1_behavioral`
- `screen_v1_no_score`
- `screen_v1_no_history`

`screen_v1_strict` should keep the current score/history/behavior/path signals,
but replace the loose support union with a stronger support pair:

- keep if `(unique_process_count >= 2) AND (total_accesses >= 3)`

`screen_v1_behavioral` should only keep the current score/history/behavior/path
signals and ignore the broader support-only rules.

## Recommended Experimental Order

1. Measure file-screening retention and GT recall on existing evaluation output.
2. Keep the current best scorer and reranker fixed while applying file
   screening.
3. Only after screening is validated, build a real `file_feature_v2` table for
   training-time experiments.
