# GPU

Use this directory for GPU-server-side training and smoke-test code.

Current entrypoints:

- `graph_loader.py` loads `graph.pt` and assembles block-concatenated per-view feature matrices
- `train_smoke.py` performs a lightweight device smoke test on a built graph artifact
- `train_gnn.py` trains a first full-batch multi-view GNN baseline on the built graph artifacts
  and saves the best checkpoint by a selectable metric (default: `val_edge_loss`); decoder options
  now include `dot` and `mlp`
- `evaluate_checkpoint.py` exports node-level anomaly scores and top-k summaries from a saved checkpoint
  and now defaults to `top5_mean + robust_zscore_by_type`
- `summarize_evaluation.py` prints a compact top-k hit summary from an evaluation output directory
- `analyze_gt_ranks.py` measures where GT nodes land in the ranked anomaly list, including top-ratio coverage
- `compare_score_aggregations.py` compares multiple node-score aggregations (`mean / max / top-k mean / q90`) on the same checkpoint
- `compare_score_calibrations.py` compares node-type calibration methods on top of a chosen base score
  (default: `top5_mean`), including raw+robust hybrid variants

Recommended workflow on the GPU server:

1. Pull this repository from GitHub.
2. Manually sync `artifacts/graphs/` from the CPU server.
3. Install the GPU-side Python environment and GPU-enabled `torch`.
4. Run `train_smoke.py` first to verify shapes, labels, and device transfer.
5. Start `train_gnn.py` after the smoke test passes.
6. Run `evaluate_checkpoint.py`, `summarize_evaluation.py`, `analyze_gt_ranks.py`, `compare_score_aggregations.py`, and `compare_score_calibrations.py` to inspect ranking quality.
