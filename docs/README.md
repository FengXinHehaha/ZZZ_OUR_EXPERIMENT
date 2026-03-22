# Docs

Use this directory for experiment design notes, data assumptions, evaluation rules, and paper-facing method summaries.

Implemented notes:

- `cadets_event_ingest.md` documents the first-stage CADETS event ingest workflow
- `cadets_feature_extraction.md` documents the first split-aware feature extraction stage
- `cadets_feature_cleaning.md` defines retained-node feature cleaning and train-based standardization
- `cadets_current_stats.md` records the current retained-node, cleaned-feature, and filtered-edge scale
- `cadets_file_node_retention.md` defines the first executable file-node filtering policy before graph construction
- `cadets_node_retention.md` defines the conservative full node-retention policy for process/file/network
- `cadets_edge_aggregation.md` defines the first window-level typed-edge compression stage
- `cadets_edge_filter.md` defines how to further filter typed edges without revisiting raw events
