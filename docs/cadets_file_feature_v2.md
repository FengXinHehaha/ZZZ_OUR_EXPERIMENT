# CADETS File Feature V2

This note tracks the first dedicated `file feature v2` line.

## Why This Line Exists

The current strongest detection line is already file-centric:

- support-aware file scoring
- history-aware file scoring
- file-specific reranking

However, the current exported feature artifacts under `artifacts/features*` do not yet materialize
all of the file-path signals that are already defined in `src/cadets_feature_extract.py`.

In particular, the extractor query already computes path-category counts such as:

- `unique_known_path_count`
- `temp_path_count`
- `config_path_count`
- `system_bin_path_count`
- `system_lib_path_count`
- `log_path_count`
- `user_home_path_count`
- `hidden_path_count`
- `script_path_count`
- `missing_path_count`

But the current baseline feature artifacts in `artifacts/features/` were generated before those fields
were re-exported, so the downstream cleaned/model-ready directories do not contain them either.

## Goal

Produce an independent CPU-side feature pipeline that:

1. re-extracts features from PostgreSQL
2. cleans retained-node rows with the current retention manifest
3. encodes the cleaned outputs into graph-ready model features
4. writes everything into standalone v2 directories without touching the current baseline artifacts

## Output Directories

- extracted: `artifacts/features_file_v2`
- cleaned: `artifacts/features_cleaned_file_v2`
- model-ready: `artifacts/features_model_ready_file_v2`

## CPU Pipeline Command

```bash
export CADETS_PG_PASSWORD=1234
/home/fxh/anaconda3/envs/ZZZ_OUR_EXPERIMENT/bin/python \
  /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/src/cadets_feature_v2_pipeline.py \
  --config /home/fxh/DeepSeek/ZZZ_OUR_EXPERIMENT/configs/cadets_event_ingest.json
```

The script will print the baseline-vs-v2 column diff for:

- `file_view__file_node.tsv`
- `process_view__file_node.tsv`

## Expected First-Step Wins

The first expected improvement is not a new model yet. It is:

- making the already-defined path signals real in exported feature artifacts
- letting us test file-focused models on a stronger feature table
- keeping the baseline artifacts untouched for a clean A/B comparison
