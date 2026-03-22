# CADETS Entity Ingest

This stage ingests raw `Subject`, `FileObject`, and `NetFlowObject` records into the new experiment database.

It creates and rebuilds:

- `process_entities`
- `file_entities`
- `network_entities`

and records file-level progress in:

- `entity_ingest_runs`
- `entity_ingest_file_audit`

## Main Command

```bash
export CADETS_PG_PASSWORD=1234

/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_entity_ingest.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

By default the script truncates `process_entities`, `file_entities`, and `network_entities` before rebuilding them.

If you want to keep existing rows and only append/update:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_entity_ingest.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json \
  --keep-existing
```

## What It Stores

`process_entities`
- subject type
- pid/cid
- parent subject UUID
- host and local principal
- process start timestamp
- command line and privilege level when available

`file_entities`
- file object type
- host and principal
- permission/epoch when available
- file descriptor and size when available

`network_entities`
- host
- local/remote addresses and ports
- IP protocol
- file descriptor when available

## Notes

- This stage no longer depends on `0310cadets` for the core entity dictionary
- duplicate object UUIDs are merged with `seen_count`
- the script uses nested file progress bars similar to event ingest
