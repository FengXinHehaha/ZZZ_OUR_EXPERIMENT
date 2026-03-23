# GPU Baseline Summary

This note records the current GPU-side baseline conclusions for the rebuilt CADETS experiment line.

The numbers below are taken from the recorded GPU terminal runs on March 23, 2026.

## Current Formal Baseline

Training configuration:

- model: multi-view full-batch GNN
- message passing: `rel_grouped`
- relation grouping: `coarse_v1`
- decoder: `dot`
- epochs: `30`
- checkpoint selection: `val_edge_loss`
- training caps: `--train-pos-edge-cap 3000000 --eval-pos-edge-cap 500000`

Evaluation configuration:

- node score aggregation: `top5_mean`
- node-type calibration: `robust_zscore_by_type`

Recommended GPU artifacts:

- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- evaluation directory: `artifacts/evaluations/rel_grouped_dot_30ep_eval`

## Formal Baseline Results

### Main metrics

| Window | ROC-AUC | AP |
| --- | ---: | ---: |
| `val` | `0.9160` | `0.000526` |
| `test_2018-04-12` | `0.8859` | `0.5264` |
| `test_2018-04-13` | `0.8699` | `0.001024` |

### Main ranking statistics

For `test_2018-04-12`:

- best GT rank: `79 / 114955`
- median GT rank: `17858.5 / 114955`
- top-100 hits: `1`
- top-1000 hits: `1`
- top-5000 hits: `3`
- top-10000 hits: `15`

For `test_2018-04-13`:

- best GT rank: `2587 / 100085`
- median GT rank: `6456.5 / 100085`
- top-5000 hits: `4`
- top-10000 hits: `11`

## What Was Tried

### Score aggregation and calibration

- Plain `mean` node scoring was not good enough for top-k retrieval.
- `top5_mean` improved front-of-list retrieval and became the default node score aggregation.
- `robust_zscore_by_type` was the strongest calibration method overall.
- Percentile and raw/robust hybrid calibrations were more aggressive in some top-k slices, but did not beat `robust_zscore_by_type` on the overall tradeoff.

### Decoder ablations

| Setup | `test_2018-04-12` ROC-AUC | `test_2018-04-12` AP | Conclusion |
| --- | ---: | ---: | --- |
| vanilla `dot` + `top5_mean` + `robust_zscore_by_type` | `0.8586` | `0.4718` | strong baseline before relation-aware message passing |
| `mlp` decoder | `0.5010` | `0.1249` | rejected |
| relation-aware `rel_mlp` decoder | `0.5011` | `0.1250` | rejected |

The main lesson was that putting relation information only in the decoder was not enough.

### Message-passing ablations

| Setup | `test_2018-04-12` ROC-AUC | `test_2018-04-12` AP | Conclusion |
| --- | ---: | ---: | --- |
| `rel_grouped` + `coarse_v1` + `dot` + `30ep` | `0.8859` | `0.5264` | current formal baseline |
| `rel_grouped` + `coarse_v1` + `dot` + `60ep` | `0.8860` | `0.5267` | not selected; nearly tied on main metric but worse GT ranks/top-k and weaker `test_2018-04-13` |
| `rel_grouped` + `coarse_v2` + `dot` + `30ep` | about `0.6357` | about `0.2556` during training-time eval | rejected; grouping was too fine and signal became weaker |

`coarse_v1` groups event types into:

- `file_read`
- `file_write`
- `file_meta`
- `process`
- `network`
- `flow_other`

## Why `rel_grouped_dot_30ep` Was Chosen

It beat the strongest vanilla baseline on the main test window:

- `ROC-AUC`: `0.8586 -> 0.8859`
- `AP`: `0.4718 -> 0.5264`

It also produced much better GT ranking than the strongest vanilla baseline:

- best GT rank: `1389 -> 79`
- median GT rank: `20667.5 -> 17858.5`

Compared with `rel_grouped_dot_60ep`, the `30ep` checkpoint was kept as the formal baseline because:

- the main metric gain at `60ep` was tiny
- `60ep` had worse top-k retrieval on `test_2018-04-12`
- `60ep` had weaker `test_2018-04-13` metrics

## Current Interpretation

The most useful relation signal is being captured in message passing, not only in the decoder.

In practice:

- relation-aware message passing helped
- decoder-only relation awareness failed
- over-splitting relation groups weakened the signal

## Recommended Next Directions

- Try a stronger relation-aware encoder family such as `RGCN` or another per-relation propagation variant.
- If relation grouping is revisited, keep the grouping compact and semantics-driven instead of splitting too finely.
- Keep the current evaluation defaults fixed while testing new encoders:
  `score_method=top5_mean`, `score_calibration=robust_zscore_by_type`
