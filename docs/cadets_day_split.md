# CADETS Day Split

This stage turns GT alignment into a reusable day-level split manifest and GT window summary.

It produces:

- day-level event counts
- day-level GT activity and new-GT counts
- candidate attack days
- a recommended `train/val/test` split manifest
- GT time-window summaries for the candidate test days

## Main Command

```bash
export CADETS_PG_PASSWORD=1234

/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_day_split.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

## Outputs

The script writes:

- `artifacts/day_split/day_summary.tsv`
- `artifacts/day_split/gt_windows.tsv`
- `artifacts/day_split/split_manifest.json`

## Default Heuristic

- mark a day as a candidate attack day if:
  - `active_process_gt_count > 0`, or
  - `new_process_gt_count > 0`, or
  - `new_gt_count >= 100`
- use the day immediately before the first candidate attack day as `val`
- use all earlier days as `train`
- use all candidate attack days as `test`

This is a starting point for the paper pipeline, not a final immutable benchmark split.
