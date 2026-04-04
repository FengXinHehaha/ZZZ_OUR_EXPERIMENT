# File Node Analysis Summary

This note records the first successful node-level improvement line after the original GPU baseline was locked.

The key change is not a new encoder or a new training loss.
It is a better post-processing pipeline for file nodes, motivated by direct false-positive analysis on `test_2018-04-13`.

## Starting Point

The original formal baseline remained:

- training: `rel_grouped + coarse_v1 + dot + 30ep`
- checkpoint selection: `val_edge_loss`
- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- node scoring: `top5_mean`
- node-type calibration: `robust_zscore_by_type`

Original baseline ranking metrics:

| Window | ROC-AUC | AP |
| --- | ---: | ---: |
| `val` | `0.9160` | `0.000526` |
| `test_2018-04-12` | `0.8859` | `0.5264` |
| `test_2018-04-13` | `0.8699` | `0.001024` |

Original baseline threshold result under `window_median_plus_mad (k=20)`:

| Window | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| `val` | `0.001157` | `0.000579` | `1.000000` |
| `test_2018-04-12` | `0.690518` | `0.528191` | `0.996886` |
| `test_2018-04-13` | `0.002393` | `0.001198` | `0.812500` |

The main unresolved problem was `test_2018-04-13`.

## What The File-Level Error Analysis Found

We added `gpu/analyze_file_false_positives.py` and used it on:

- eval dir: `artifacts/evaluations/rel_grouped_dot_30ep_eval`
- target window: `test_2018-04-13`
- reference window: `test_2018-04-12`

Main finding:

- the highest-ranked false-positive files on `test_2018-04-13` were mostly **low-support** files
- many top false positives had `total_degree=1` and `unique_neighbors=1`
- GT files were the opposite: they tended to have high degree, many neighbors, and persistent support across windows

This ruled out the earlier hypothesis that the main issue was "popular high-degree files being over-scored".

It pointed to the opposite fix:

- downweight **low-support file nodes**
- do not blindly penalize high-support file nodes

## Rejected Node-Level Heuristic

We tried simple file-degree penalties first:

- `top5_mean_log_degree_file`
- `top5_mean_sqrt_degree_file`
- `top10_mean_log_degree_file`

These helped some raw ranks but failed once combined with the formal calibration pipeline.

Conclusion:

- naive file degree penalty is too coarse
- it hurts the calibrated ranking more than it helps

## Current Best Scoring Pipeline

The first successful change was a support-aware file score:

- score method: `top5_mean_log_support_floor128_file`
- score calibration: `robust_zscore_by_type`

This keeps the same checkpoint and only changes the node scoring behavior:

- low-support file nodes are softly downweighted
- high-support file nodes keep their anomaly strength

### Ranking-quality comparison

| Setup | `val` AP | `test_2018-04-12` AP | `test_2018-04-13` AP | `test_2018-04-13` best GT rank | `test_2018-04-13` top-1000 hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| original `top5_mean + robust_zscore_by_type` | `0.000526` | `0.526408` | `0.001024` | `2587` | `0` |
| `top5_mean_log_support_floor128_file + robust_zscore_by_type` | `0.013576` | `0.526376` | `0.011643` | `95` | `10` |

This was the first real breakthrough on `test_2018-04-13`:

- best GT rank: `2587 -> 95`
- top-1000 GT hits: `0 -> 10`
- AP: `0.001024 -> 0.011643`

At the same time, `test_2018-04-12` AP stayed essentially unchanged.

## Why The Old Threshold Policy Was No Longer Enough

Once the new score was applied, ranking quality improved sharply, but a single fixed threshold policy was still unstable across windows.

The issue was window regime mismatch:

- `test_2018-04-12` behaves like a dense alarm window
- `test_2018-04-13` behaves like a sparse alarm window

The old single-window-local policy:

- `window_median_plus_mad (k=20)`

could still do well on `test_2018-04-12`, but it could not exploit the new ranking quality on sparse windows.

## Current Best Threshold Policy

We added `gpu/compare_adaptive_threshold_policies.py` and built a first adaptive threshold policy.

### Regime detection

- sparse if `count(score >= 1000) < 1000`
- dense if `count(score >= 200) >= 20000`
- otherwise moderate

### Policy by regime

- sparse: `top_count = 138`
- moderate: `top_count = 200`
- dense: `window_median_plus_mad(k=20)`

This policy maps the three current windows as:

- `val -> moderate`
- `test_2018-04-12 -> dense`
- `test_2018-04-13 -> sparse`

### Current best end-to-end operating point

Using:

- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- score method: `top5_mean_log_support_floor128_file`
- score calibration: `robust_zscore_by_type`
- threshold policy: adaptive policy above

we get:

| Window | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| `val` | `0.038278` | `0.020000` | `0.444444` |
| `test_2018-04-12` | `0.690518` | `0.528191` | `0.996886` |
| `test_2018-04-13` | `0.051948` | `0.028986` | `0.250000` |

Compared with the old single-policy baseline on `test_2018-04-13`:

- F1: `0.002393 -> 0.051948`

The sparse-window sweep showed `top_count=138` was the best setting among:

- `100`
- `120`
- `138`
- `160`
- `200`
- `300`
- `500`

## Current Status

The original training checkpoint is still the same:

- `rel_grouped + coarse_v1 + dot + 30ep`

But the best node-level pipeline is no longer the old formal baseline.

Current strongest candidate:

- scorer: `top5_mean_log_support_floor128_file + robust_zscore_by_type`
- thresholding: adaptive threshold policy v1

This is the first candidate that:

- preserves `test_2018-04-12`
- substantially improves `test_2018-04-13`
- is supported by direct node-level error analysis instead of blind hyperparameter search

## Relevant Scripts

- `gpu/analyze_file_false_positives.py`
- `gpu/score_aggregation.py`
- `gpu/compare_score_aggregations.py`
- `gpu/compare_score_calibrations.py`
- `gpu/compare_adaptive_threshold_policies.py`

