from pathlib import Path
from typing import Dict, List, Tuple

import torch


VIEW_GROUP_ORDER = {
    "process_view": [
        "process_view__process_node",
        "process_view__file_node",
        "process_view__network_node",
    ],
    "file_view": [
        "file_view__process_node",
        "file_view__file_node",
    ],
    "network_view": [
        "network_view__process_node",
        "network_view__network_node",
    ],
}


def load_graph(graph_path: str | Path) -> Dict[str, object]:
    return torch.load(Path(graph_path), map_location="cpu")


def build_block_view_matrix(
    graph: Dict[str, object],
    view_name: str,
) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
    groups = VIEW_GROUP_ORDER[view_name]
    num_nodes = int(graph["num_nodes"])

    block_ranges: Dict[str, Tuple[int, int]] = {}
    total_dim = 0
    for group_name in groups:
        dim = int(graph["feature_groups"][group_name]["x"].shape[1])
        block_ranges[group_name] = (total_dim, total_dim + dim)
        total_dim += dim

    x_view = torch.zeros((num_nodes, total_dim), dtype=torch.float32)

    for group_name in groups:
        start, end = block_ranges[group_name]
        payload = graph["feature_groups"][group_name]
        node_ids = payload["node_ids"].long()
        x_group = payload["x"].float()
        x_view[node_ids, start:end] = x_group

    return x_view, block_ranges


def build_all_view_matrices(graph: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for view_name in VIEW_GROUP_ORDER:
        x_view, block_ranges = build_block_view_matrix(graph, view_name)
        result[view_name] = {
            "x": x_view,
            "block_ranges": block_ranges,
        }
    return result


def summarize_graph(graph: Dict[str, object]) -> Dict[str, object]:
    return {
        "window_name": graph["window_name"],
        "split": graph["split"],
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "gt_nodes": int(graph["y"].sum().item()),
        "event_type_count": len(graph["event_type_vocab"]),
        "feature_groups": {
            group_name: {
                "rows": int(payload["x"].shape[0]),
                "dim": int(payload["x"].shape[1]),
                "node_type": payload["node_type"],
            }
            for group_name, payload in sorted(graph["feature_groups"].items())
        },
    }
