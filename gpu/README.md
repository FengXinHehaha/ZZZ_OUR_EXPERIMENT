# GPU

Use this directory for GPU-server-side training and smoke-test code.

Current entrypoints:

- `graph_loader.py` loads `graph.pt` and assembles block-concatenated per-view feature matrices
- `train_smoke.py` performs a lightweight device smoke test on a built graph artifact

Recommended workflow on the GPU server:

1. Pull this repository from GitHub.
2. Manually sync `artifacts/graphs/` from the CPU server.
3. Install the GPU-side Python environment and GPU-enabled `torch`.
4. Run `train_smoke.py` first to verify shapes, labels, and device transfer.
5. Start the formal training script after the smoke test passes.
