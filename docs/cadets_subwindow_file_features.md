# CADETS Subwindow File Features

This note documents the first CPU-side extractor for finer-grained file features derived from
subwindows inside each day-level evaluation window.

## Why

The hardest window (`test_2018-04-13`) still contains too many normal `file` nodes. Instead of
replacing the daily evaluation unit, this stage keeps the day-level labels intact and adds
within-day burst features that can later support:

- file screening
- learned file reranking
- day-over-day delta features

## Script

- extractor: `src/cadets_file_subwindow_feature_extract.py`

## Current Output

Each selected window exports:

- `file_subwindow__file_node.tsv`
- `metadata.json`

Current default output root:

- `artifacts/features_file_subwindow_6h`

## First-Pass Feature Columns

- `subwindow_hours`
- `total_subwindows`
- `active_subwindow_count`
- `first_active_subwindow`
- `last_active_subwindow`
- `active_subwindow_span`
- `max_subwindow_accesses`
- `mean_active_subwindow_accesses`
- `std_active_subwindow_accesses`
- `max_subwindow_ratio`
- `active_subwindow_ratio`
- `peak_subwindow_index`
- `peak_subwindow_accesses`
- `peak_subwindow_unique_process_count`
- `peak_subwindow_read_count`
- `peak_subwindow_write_count`
- `peak_subwindow_execute_count`
- `write_active_subwindow_count`
- `execute_active_subwindow_count`
- `risky_path_active_subwindow_count`
- `total_accesses_from_subwindows`

The current risky-path count uses the same lightweight path categories that were already useful in
path-aware reranking:

- `temp`
- `hidden`
- `script`
- `system_bin`

## Example

```bash
export CADETS_PG_PASSWORD=1234

/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  src/cadets_file_subwindow_feature_extract.py \
  --window val \
  --window test_2018-04-12 \
  --window test_2018-04-13 \
  --subwindow-hours 6 \
  --output-dir artifacts/features_file_subwindow_6h
```

## First Smoke Findings

Smoke extraction succeeded for:

- `val`
- `test_2018-04-13`

Observed row counts:

- `val`: `248,342`
- `test_2018-04-13`: `237,809`

At `6h` granularity, most files are still highly sparse:

- `active_subwindow_count` median is `1`
- `active_subwindow_count` p90 is `1`
- `active_subwindow_count` p99 is `1`

For more active files, the distribution opens up:

- `val`, files with `total_accesses >= 50`: `active_subwindow_count` p90 = `3`
- `val`, files with `total_accesses >= 100`: `active_subwindow_count` p90 = `4`
- `test_2018-04-13`, files with `total_accesses >= 50`: `active_subwindow_count` p90 = `4`
- `test_2018-04-13`, files with `total_accesses >= 100`: `active_subwindow_count` p90 = `4`

## Interpretation

This suggests:

- the subwindow idea is valid
- but `6h` is still fairly coarse for most normal files
- the next promising step is to re-run the same extractor at `2h` (and possibly `1h`) for
  stronger burstiness separation

## Recommended Next Step

Keep the day-level evaluation setup unchanged, and add one of these follow-ups:

1. `2h` subwindow extraction for `val`, `test_2018-04-12`, `test_2018-04-13`
2. subwindow-aware file screening using `active_subwindow_count`, `max_subwindow_ratio`, and
   `peak_subwindow_*`
3. day-over-day delta features using `test_2018-04-12` as the reference day for
   `test_2018-04-13`
