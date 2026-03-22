# CADETS Typed-Edge Filter Spec

This note locks the second-stage edge compression rule after node retention.

## Goal

The original edge aggregation step already compressed raw event interactions into window-level typed edges.

This stage performs an additional pass:

- it reuses the existing `typed_edges.tsv` files
- it never revisits `events_raw`
- it keeps only edges whose endpoints survive the latest node-retention stage

## Motivation

The user explicitly requested that any further edge filtering must happen on the already aggregated edge set, not on the original event table.

That means the pipeline becomes:

- `41M+ raw events`
- first aggregation to window-level `typed_edges.tsv`
- further edge filtering only on those typed-edge files

## Edge Survival Rule

For a given window, keep a typed edge iff:

- `src_uuid` is present in the retained node list for `src_type`
- `dst_uuid` is present in the retained node list for `dst_type`

Otherwise the typed edge is dropped.

No extra event-type pruning is applied in this stage.

## Inputs

- `artifacts/edges/<window>/typed_edges.tsv`
- `artifacts/node_retention/<window>/process_keep_list.tsv`
- `artifacts/node_retention/<window>/file_keep_list.tsv`
- `artifacts/node_retention/<window>/network_keep_list.tsv`

Root manifests:

- `artifacts/edges/edge_manifest.json`
- `artifacts/node_retention/retention_manifest.json`

## Outputs

Per window:

- `artifacts/edges_filtered/<window>/typed_edges.tsv`
- `artifacts/edges_filtered/<window>/metadata.json`

Root:

- `artifacts/edges_filtered/edge_manifest.json`

## Required Metadata

Per window, metadata must include:

- original typed-edge count
- retained typed-edge count
- dropped typed-edge count
- typed-edge retention ratio
- retained event-weight sum
- dropped event-weight sum
- event-weight retention ratio
- node counts after final node retention
- total GT dropped count from the node-retention stage

## Notes

- This stage is intentionally simple and cheap.
- It is designed to support iterative node pruning without repeating the expensive raw-event aggregation step.
