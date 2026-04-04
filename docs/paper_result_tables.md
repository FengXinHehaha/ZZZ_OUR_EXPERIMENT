# Paper Result Tables

This note reformats the current strongest results into compact, paper-facing tables.

The comparison below uses the same checkpoint throughout:

- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- training setup: `rel_grouped + coarse_v1 + dot + 30ep`

The difference is entirely in node scoring and threshold policy.

## Table 1. Main Comparison

| Method | Node Score | Threshold Policy | `val` AP | `val` F1 | `test_2018-04-12` AP | `test_2018-04-12` F1 | `test_2018-04-13` AP | `test_2018-04-13` F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | `top5_mean + robust_zscore_by_type` | single `window_median_plus_mad (k=20)` | `0.000526` | `0.001157` | `0.526408` | `0.690518` | `0.001024` | `0.002393` |
| Current best candidate | `top5_mean_log_support_floor128_file + robust_zscore_by_type` | adaptive policy v1 | `0.013576` | `0.038278` | `0.526376` | `0.690518` | `0.011643` | `0.051948` |

## Table 2. Ranking Improvement On `test_2018-04-13`

| Method | best GT rank | top-1000 GT hits | top-5000 GT hits | top-10000 GT hits | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original baseline | `2587` | `0` | `4` | `11` | `0.001024` |
| Current best candidate | `95` | `10` | `10` | `12` | `0.011643` |

## Table 3. Adaptive Threshold Policy V1

| Regime | Detector Rule | Threshold Action |
| --- | --- | --- |
| sparse | `count(score >= 1000) < 1000` | `top_count = 138` |
| moderate | otherwise, but not dense | `top_count = 200` |
| dense | `count(score >= 200) >= 20000` | `window_median_plus_mad (k=20)` |

## Table 4. Window-to-Regime Mapping

| Window | Regime | Applied Policy | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| `val` | `moderate` | `top_count = 200` | `0.020000` | `0.444444` | `0.038278` |
| `test_2018-04-12` | `dense` | `window_median_plus_mad (k=20)` | `0.528191` | `0.996886` | `0.690518` |
| `test_2018-04-13` | `sparse` | `top_count = 138` | `0.028986` | `0.250000` | `0.051948` |

## Table 5. Sparse-Window Sweep On `test_2018-04-13`

| sparse `top_count` | Precision | Recall | F1 | predicted positives |
| ---: | ---: | ---: | ---: | ---: |
| `100` | `0.020000` | `0.125000` | `0.034483` | `100` |
| `120` | `0.025000` | `0.187500` | `0.044118` | `120` |
| `138` | `0.028986` | `0.250000` | `0.051948` | `138` |
| `160` | `0.025000` | `0.250000` | `0.045455` | `160` |
| `200` | `0.020000` | `0.250000` | `0.037037` | `200` |
| `300` | `0.016667` | `0.312500` | `0.031646` | `300` |
| `500` | `0.016000` | `0.500000` | `0.031008` | `500` |

## Short Paper Claim

The current best candidate keeps the original training checkpoint fixed and only changes node-level post-processing.

Under this setting:

- `test_2018-04-12` remains essentially unchanged in AP/F1
- `test_2018-04-13` improves from `AP 0.001024 / F1 0.002393` to `AP 0.011643 / F1 0.051948`
- `test_2018-04-13` best GT rank improves from `2587` to `95`
- `test_2018-04-13` top-1000 GT hits improve from `0` to `10`

This makes the new pipeline the strongest current candidate for the paper-facing result section.

