# GPU

Use this directory for GPU-server-side training and smoke-test code.

Current entrypoints:

- `graph_loader.py` loads `graph.pt` and assembles block-concatenated per-view feature matrices
- `train_smoke.py` performs a lightweight device smoke test on a built graph artifact
- `train_gnn.py` trains a first full-batch multi-view GNN baseline on the built graph artifacts
  and saves the best checkpoint by a selectable metric (default: `val_edge_loss`)
- `evaluate_checkpoint.py` exports node-level anomaly scores and top-k summaries from a saved checkpoint
- `summarize_evaluation.py` prints a compact top-k hit summary from an evaluation output directory
- `analyze_gt_ranks.py` measures where GT nodes land in the ranked anomaly list, including top-ratio coverage

Recommended workflow on the GPU server:

1. Pull this repository from GitHub.
2. Manually sync `artifacts/graphs/` from the CPU server.
3. Install the GPU-side Python environment and GPU-enabled `torch`.
4. Run `train_smoke.py` first to verify shapes, labels, and device transfer.
5. Start `train_gnn.py` after the smoke test passes.
6. Run `evaluate_checkpoint.py`, `summarize_evaluation.py`, and `analyze_gt_ranks.py` to inspect ranking quality.
