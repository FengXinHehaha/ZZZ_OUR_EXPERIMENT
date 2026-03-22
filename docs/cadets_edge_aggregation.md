# CADETS Edge Aggregation

This note locks the first executable edge-compression stage for the full-batch graph baseline.

## Goal

The raw CADETS event stream is far too large to be used as a training graph directly.

This stage compresses event-level interactions into window-level typed edges so that:

- graph construction does not use raw event rows as edges
- edge count is reduced before model training
- relation semantics are still preserved through `event_type`

## Input

The edge aggregator consumes:

- `artifacts/features/feature_manifest.json`
- `artifacts/retention/retention_manifest.json`
- `events_raw` in `zzz_our_experiment_cadets`

Node policy:

- all process nodes from the current window are kept
- all network nodes from the current window are kept
- only retained file nodes from the current window are kept

## Aggregation Rule

For each window, edges are aggregated by:

- `src_uuid`
- `dst_uuid`
- `event_type`

This means one output edge corresponds to:

`(src_uuid, dst_uuid, event_type, window)`

## Output Columns

Per window, the exported edge file contains:

- `src_uuid`
- `src_type`
- `dst_uuid`
- `dst_type`
- `event_type`
- `event_count`
- `first_timestamp_ns`
- `last_timestamp_ns`

## Output Layout

Default output root:

- `artifacts/edges/`

Per window:

- `typed_edges.tsv`
- `metadata.json`

At the root:

- `edge_manifest.json`

## Notes

- Only edges whose two endpoints survive node retention are exported.
- `object_uuid` and `object2_uuid` interactions are both included.
- This stage keeps `event_type` explicit; no coarse relation collapsing is done yet.
- If the resulting typed-edge graph is still too large, the next tightening step should collapse fine-grained event types into broader relation families, not go back to raw event edges.
