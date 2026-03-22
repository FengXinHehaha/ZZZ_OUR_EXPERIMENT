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
