# CADETS GT Entity Alignment

This stage refines GT alignment by adding entity-table type information.

It combines:

- event visibility from `zzz_our_experiment_cadets.events_raw`
- entity type membership from a reference database

## Main Command

```bash
export CADETS_PG_PASSWORD=1234

/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_gt_entity_align.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

## Outputs

The script writes:

- `artifacts/gt_entity_alignment/summary.json`
- `artifacts/gt_entity_alignment/per_uuid.tsv`

## Notes

- the script auto-detects which entity tables exist in the reference database
- if `process_entities/file_entities/network_entities` exist, it uses them
- otherwise it falls back to `processes/files/networks`
- `events_raw` in the new database remains the source of event visibility and active-day evidence
- if a GT UUID matches more than one entity table, the script marks it as `ambiguous`
