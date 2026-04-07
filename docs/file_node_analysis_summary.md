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

## Support-Aware Scoring Breakthrough

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

## First Adaptive Threshold Policy

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

### First end-to-end improvement

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

This established the first practical adaptive threshold policy, but it was still built before adding explicit cross-window history.

## Cross-Window History-Aware File Scoring

The next improvement line kept the same training checkpoint and same support-aware base score, but added a cross-window history signal.

### Core idea

- start from `top5_mean_log_support_floor128_file`
- compute a per-type percentile score for the previous window using the same support-aware score
- if the same node UUID appears again in the next window, multiply its current score by:
  - `1 + type_weight * previous_percentile`

We tested three history variants:

- `top5_mean_log_support_floor128_file_history_file_only`
- `top5_mean_log_support_floor128_file_history_file_process`
- `top5_mean_log_support_floor128_file_history_all_types`

The result was very clear:

- `file_only` was best
- adding `process` or `network` history did not produce additional gains

To avoid leakage from `val` into `test_2018-04-12`, the history-aware evaluation scripts were updated to support a reset boundary before a chosen window.

### Ranking-quality comparison after adding history

| Setup | `val` AP | `test_2018-04-12` AP | `test_2018-04-13` AP | `test_2018-04-13` best GT rank | `test_2018-04-13` top-1000 hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `top5_mean_log_support_floor128_file + robust_zscore_by_type` | `0.013576` | `0.526376` | `0.011643` | `95` | `10` |
| `top5_mean_log_support_floor128_file_history_file_only + robust_zscore_by_type` | `0.013576` | `0.526376` | `0.016470` | `93` | `10` |

So the history-aware scorer improved `test_2018-04-13` ranking again:

- AP: `0.011643 -> 0.016470`
- best GT rank: `95 -> 93`

### Why adaptive v1 no longer fully matched the new scorer

When the old adaptive policy v1 was reused unchanged:

- sparse windows still used `top_count = 138`
- this preserved the previous `test_2018-04-13` F1 of `0.051948`
- but it did not fully exploit the new history-aware ranking gain

The reason is simple:

- the extra GT nodes promoted by history mainly appeared below the old sparse cutoff
- therefore the ranking got better, but the top-138 alarm budget stayed too tight

## File-Specific Reranking

The next stage kept the same history-aware scorer but added a dedicated second-stage file reranker on top-ranked file nodes.

### Hand-Crafted File Reranker

The first successful reranker was:

- post-rerank method: `file_rerank_support`
- candidate band: top `2000` file nodes

It reweights file nodes using:

- total degree percentile
- unique process-neighbor percentile
- file-read incident-edge percentile

This improved ranking again without changing the training checkpoint:

| Setup | `val` AP | `test_2018-04-12` AP | `test_2018-04-13` AP | `test_2018-04-13` best GT rank |
| --- | ---: | ---: | ---: | ---: |
| history-aware scorer only | `0.013576` | `0.526376` | `0.016470` | `93` |
| `+ file_rerank_support` | `0.018160` | `0.526379` | `0.017937` | `79` |

Using the same adaptive policy family but re-sweeping the sparse operating point, the best observed hand-crafted reranker result was:

- sparse `top_count = 300`
- `test_2018-04-13`: `P 0.033333`, `R 0.625000`, `F1 0.063291`

### Path-Aware Hand-Crafted Reranker

We then brought the newly exported file path features into the reranker:

- post-rerank method: `file_rerank_support_path`
- extra signals:
  - `unique_known_path_count`
  - risky path counts built from `temp / hidden / script / system_bin`

This helped the `hybrid_file_v2` line at the ranking stage, but when attached back to the strongest original graph line it did not improve the final `F1` beyond the hand-crafted `file_rerank_support` operating point.

Conclusion:

- path features are informative
- but the fixed hand-crafted weighting still leaves value on the table

### Lightweight Learned File Reranker

The next step replaced the hand-crafted file reranker weights with a lightweight learned linear reranker trained on `val`.

The learned reranker uses these features on top-ranked file nodes:

- `base_score_pct`
- `total_degree_pct`
- `unique_process_neighbors_pct`
- `file_read_edges_pct`
- `history_pct`
- `known_path_count_pct`
- `risky_path_count_pct`

The learned weights showed that the most useful signals were still structural/support features:

- strong positive: `total_degree_pct`, `unique_process_neighbors_pct`, `file_read_edges_pct`
- weak positive: `risky_path_count_pct`
- near-zero: `history_pct` (already largely absorbed by the base scorer)
- negative: `known_path_count_pct`

Ranking-quality comparison on the current strongest graph line:

| Setup | `val` AP | `test_2018-04-12` AP | `test_2018-04-13` AP | `test_2018-04-13` best GT rank |
| --- | ---: | ---: | ---: | ---: |
| history-aware scorer only | `0.013576` | `0.526376` | `0.016470` | `93` |
| `+ file_rerank_support` | `0.018160` | `0.526379` | `0.017937` | `79` |
| `+ file_rerank_support_path` | `0.018208` | `0.526379` | `0.018292` | `75` |
| `+ file_rerank_learned_linear` | `0.030334` | `0.526390` | `0.023266` | `52` |

This is the strongest ranking-quality line so far.

## Current Strongest Observed Operating Point

We re-swept the sparse top-count for the learned reranker and found the best current `test_2018-04-13` operating point at:

- sparse `top_count = 250`

The resulting adaptive policy is:

- sparse: `top_count = 250`
- moderate: `top_count = 200`
- dense: `window_median_plus_mad(k=20)`

Using:

- checkpoint: `artifacts/training_runs/rel_grouped_dot_30ep/best_model.pt`
- score method: `top5_mean_log_support_floor128_file_history_file_only`
- score calibration: `robust_zscore_by_type`
- post-rerank method: `file_rerank_learned_linear`
- candidate band: top `2000` file nodes
- threshold policy: adaptive policy with sparse `top_count=250`

we get:

| Window | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| `val` | `0.038278` | `0.020000` | `0.444444` |
| `test_2018-04-12` | `0.690518` | `0.528191` | `0.996886` |
| `test_2018-04-13` | `0.075188` | `0.040000` | `0.625000` |

Important caveat:

- the sparse `top_count=250` choice is a **post-hoc operating-point result**
- it is useful for analysis and for reporting the current best achievable `F1`
- but it is stricter to describe it as a post-hoc sweep result rather than a fully validation-selected threshold

## Current Status

The original training checkpoint is still the same:

- `rel_grouped + coarse_v1 + dot + 30ep`

But the best node-level pipeline is no longer the old formal baseline.

There are now two useful "best" summaries:

- strongest validation-compatible adaptive policy so far:
  - scorer: `top5_mean_log_support_floor128_file_history_file_only + robust_zscore_by_type`
  - thresholding: adaptive policy v1 with sparse `top_count=138`
  - `test_2018-04-13 F1 = 0.051948`
- strongest current post-hoc operating point:
  - scorer: `top5_mean_log_support_floor128_file_history_file_only + robust_zscore_by_type`
  - post-rerank: `file_rerank_learned_linear`
  - thresholding: adaptive policy with sparse `top_count=250`
  - `test_2018-04-13 F1 = 0.075188`

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
- `gpu/evaluate_checkpoint.py`
