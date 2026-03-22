# CADETS GT Alignment

This stage aligns the UUIDs in `cadets_ground_truth.txt` with the rebuilt `events_raw` table.

It answers three practical questions:

- Which GT UUIDs are visible in `events_raw` at all
- Whether each UUID appears as `subject_uuid`, `object_uuid`, or `object2_uuid`
- On which days each GT UUID is active in the current database

## Main Command

Set the database password first:

```bash
export CADETS_PG_PASSWORD=1234
```

Run the alignment:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_gt_align.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

If you want to point to a different GT file:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_gt_align.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json \
  --ground-truth-path "/path/to/cadets_ground_truth.txt"
```

## Outputs

The script writes these files to `artifacts/gt_alignment`:

- `summary.json`
- `role_summary.tsv`
- `daily_summary.tsv`
- `per_uuid.tsv`

## Current Scope

- This is an event-visibility alignment, not a final node-type ontology
- `subject_uuid` is treated as process-like evidence
- `object_uuid` and `object2_uuid` are treated as non-subject evidence
- This stage does not yet use entity tables such as process, file, or socket metadata
