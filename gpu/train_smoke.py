import argparse
import json
from pathlib import Path

import torch

from graph_loader import build_all_view_matrices, load_graph, summarize_graph


DEFAULT_GRAPH = Path(__file__).resolve().parents[1] / "artifacts" / "graphs" / "train" / "graph.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight GPU/CPU smoke test on a built graph artifact."
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=str(DEFAULT_GRAPH),
        help=f"Path to graph.pt. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device. Default: cuda if available, else cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph).expanduser().resolve()
    device = torch.device(args.device)

    graph = load_graph(graph_path)
    summary = summarize_graph(graph)
    views = build_all_view_matrices(graph)

    print("[gpu-smoke] graph summary:")
    print(json.dumps(summary, indent=2))

    print("[gpu-smoke] block-concatenated view dims:")
    for view_name, payload in views.items():
        print(f"  {view_name}: {tuple(payload['x'].shape)} blocks={payload['block_ranges']}")

    edge_index = graph["edge_index"].to(device)
    edge_type = graph["edge_type"].to(device)
    y = graph["y"].to(device)

    moved = {
        "edge_index": tuple(edge_index.shape),
        "edge_type": tuple(edge_type.shape),
        "y": tuple(y.shape),
    }

    for view_name, payload in views.items():
        x = payload["x"].to(device)
        moved[view_name] = tuple(x.shape)

    print("[gpu-smoke] tensors moved successfully:")
    print(json.dumps(moved, indent=2))

    if device.type == "cuda":
        print(f"[gpu-smoke] cuda device: {torch.cuda.get_device_name(device)}")
        print(f"[gpu-smoke] allocated memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"[gpu-smoke] reserved memory: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    print("[gpu-smoke] success")


if __name__ == "__main__":
    main()
