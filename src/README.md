# Src

Use this directory for the new implementation of data extraction, graph construction, training, detection, and evaluation.

Current entrypoints:

- `cadets_event_ingest.py` builds the auditable two-layer CADETS event database
- `cadets_event_audit.py` prints file-level and day-level ingest summaries
- `cadets_event_repair_canonical.py` repairs `events_raw` when canonical dedup rules need to be rebuilt
- `cadets_entity_ingest.py` ingests raw Subject/FileObject/NetFlowObject records into local entity tables
- `cadets_gt_align.py` aligns `cadets_ground_truth.txt` against `events_raw`
- `cadets_gt_entity_align.py` refines GT alignment with entity-table types from a reference database
- `cadets_day_split.py` derives day-level split manifests and GT time windows
- `cadets_feature_extract.py` exports split-aware baseline node features from local entity tables
- `cadets_feature_clean.py` filters retained-node feature rows, drops empty/constant columns, and standardizes numeric features from train stats
- `cadets_feature_density.py` summarizes per-dimension non-missing and non-zero ratios for cleaned feature tables
- `cadets_feature_encode.py` encodes low-cardinality categorical columns into model-ready numeric features using train vocabularies
- `cadets_file_node_retention.py` filters file nodes with explicit GT-loss auditing
- `cadets_node_retention.py` extends retention to process/file/network while keeping GT-loss auditable
- `cadets_edge_aggregate.py` compresses raw event interactions into window-level typed edges
- `cadets_edge_filter.py` further filters typed edges using retained node lists only
