# CADETS Event Ingest

This stage rebuilds the CADETS event database as an auditable two-layer ingest:

- `event_stage_raw` keeps every parsed `Event` row with `source_file`, `line_no`, and `raw_record`
- `events_raw` keeps one canonical row per exact event fingerprint

## What It Does

- Creates a fresh target database if needed
- Creates ingest audit tables and event tables
- Stages every `Event` record from `ta1-cadets-e3-official*.json*`
- Rebuilds a canonical `events_raw` table by earliest `(timestamp_ns, source_file, line_no)`
- Rebuilds a canonical `events_raw` table by exact-event fingerprint rather than `event_uuid`
- Produces file-level and day-level audit summaries
- Shows nested progress bars during ingest: outer file progress and inner per-file progress

## Main Commands

Set the database password first:

```bash
export CADETS_PG_PASSWORD=1234
```

Run the full ingest:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_event_ingest.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

If you want to drop and recreate the target database before rerunning:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_event_ingest.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json \
  --reset-db
```

Run a small smoke test on the first file only:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_event_ingest.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json \
  --file-limit 1 \
  --line-limit 50000
```

Print the latest audit summary:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_event_audit.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

Repair an already-ingested database when the canonical rules change:

```bash
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_event_repair_canonical.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

## Expected Validation

- 10 `ta1-cadets-e3-official*.json*` files show up in `ingest_file_audit`
- every file reaches `status=success`
- canonical coverage spans `2018-04-02` to `2018-04-13`
- `events_raw` row count is meaningfully larger than the old `0310cadets.events`

## Notes

- `events_raw` is rebuilt from the current run and stores one canonical row per exact event fingerprint
- duplicate UUID rows remain preserved in `event_stage_raw`
- this stage does not ingest entity tables or build features
- the inner progress bar uses bytes for full-file ingest and lines when `--line-limit` is set
- the stage table is `UNLOGGED` and batch inserts use PostgreSQL `COPY` for better ingest speed
- the ingest session turns off `synchronous_commit`, commits every configurable batch group, and reuses the original JSON line as `raw_record`
- `event_uuid` is indexed in `events_raw`, but it is not treated as globally unique in CADETS
- the repair script can rebuild `events_raw` directly from `raw_record` without rewriting `event_stage_raw`
