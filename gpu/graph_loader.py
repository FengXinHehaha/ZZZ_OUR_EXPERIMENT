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

RELATION_GROUP_SCHEMES = ("coarse_v1", "coarse_v2")

RELATION_GROUPS_COARSE_V1 = (
    "file_read",
    "file_write",
    "file_meta",
    "process",
    "network",
    "flow_other",
)

RELATION_GROUPS_COARSE_V2 = (
    "file_read",
    "file_write",
    "file_meta",
    "process_lifecycle",
    "process_control",
    "network_connect",
    "network_io",
    "flow_other",
)


def load_graph(graph_path: str | Path) -> Dict[str, object]:
    return torch.load(Path(graph_path), map_location="cpu")


def event_name_to_relation_group(event_name: str, scheme: str = "coarse_v1") -> str:
    if event_name in {"EVENT_OPEN", "EVENT_READ", "EVENT_LSEEK", "EVENT_MMAP", "EVENT_CLOSE", "EVENT_FCNTL"}:
        return "file_read"
    if event_name in {"EVENT_WRITE", "EVENT_TRUNCATE"}:
        return "file_write"
    if event_name in {"EVENT_CREATE_OBJECT", "EVENT_LINK", "EVENT_MODIFY_FILE_ATTRIBUTES", "EVENT_RENAME", "EVENT_UNLINK"}:
        return "file_meta"

    if scheme == "coarse_v1":
        if event_name in {
            "EVENT_CHANGE_PRINCIPAL",
            "EVENT_EXECUTE",
            "EVENT_EXIT",
            "EVENT_FORK",
            "EVENT_MODIFY_PROCESS",
            "EVENT_SIGNAL",
        }:
            return "process"
        if event_name in {
            "EVENT_ACCEPT",
            "EVENT_BIND",
            "EVENT_CONNECT",
            "EVENT_RECVFROM",
            "EVENT_RECVMSG",
            "EVENT_SENDMSG",
            "EVENT_SENDTO",
        }:
            return "network"
        return "flow_other"

    if scheme == "coarse_v2":
        if event_name in {"EVENT_EXECUTE", "EVENT_EXIT", "EVENT_FORK"}:
            return "process_lifecycle"
        if event_name in {"EVENT_CHANGE_PRINCIPAL", "EVENT_MODIFY_PROCESS", "EVENT_SIGNAL"}:
            return "process_control"
        if event_name in {"EVENT_ACCEPT", "EVENT_BIND", "EVENT_CONNECT"}:
            return "network_connect"
        if event_name in {"EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_SENDMSG", "EVENT_SENDTO"}:
            return "network_io"
        return "flow_other"

    raise ValueError(f"Unsupported relation group scheme: {scheme}")


def build_relation_group_mapping(
    event_type_vocab: Dict[str, int],
    scheme: str = "coarse_v1",
) -> tuple[torch.Tensor, List[str]]:
    if scheme == "coarse_v1":
        group_names = list(RELATION_GROUPS_COARSE_V1)
    elif scheme == "coarse_v2":
        group_names = list(RELATION_GROUPS_COARSE_V2)
    else:
        raise ValueError(f"Unsupported relation group scheme: {scheme}")

    group_to_id = {name: idx for idx, name in enumerate(group_names)}
    mapping = torch.empty(len(event_type_vocab), dtype=torch.long)
    for event_name, event_id in event_type_vocab.items():
        mapping[event_id] = group_to_id[event_name_to_relation_group(event_name, scheme=scheme)]
    return mapping, group_names


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


def build_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    add_self_loops: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if device is None:
        device = edge_index.device
    device = torch.device(device)

    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)

    if edge_weight is None:
        weight = torch.ones(src.shape[0], device=device, dtype=torch.float32)
    else:
        weight = edge_weight.to(device=device, dtype=torch.float32)

    indices = torch.cat(
        [
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ],
        dim=1,
    )
    values = torch.cat([weight, weight], dim=0)

    if add_self_loops:
        loop_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
        loop_index = torch.stack([loop_nodes, loop_nodes], dim=0)
        indices = torch.cat([indices, loop_index], dim=1)
        values = torch.cat([values, torch.ones(num_nodes, device=device, dtype=torch.float32)], dim=0)

    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()

    row, col = adjacency.indices()
    val = adjacency.values()

    degree = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    degree.index_add_(0, row, val)
    degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    norm_values = degree_inv_sqrt[row] * val * degree_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        adjacency.indices(),
        norm_values,
        adjacency.shape,
        device=device,
        dtype=torch.float32,
    ).coalesce()


def build_relation_group_adjacencies(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    event_type_vocab: Dict[str, int],
    edge_weight: torch.Tensor | None = None,
    scheme: str = "coarse_v1",
    device: torch.device | str | None = None,
) -> tuple[List[str], List[torch.Tensor]]:
    if device is None:
        device = edge_index.device
    device = torch.device(device)

    relation_group_map, group_names = build_relation_group_mapping(event_type_vocab, scheme=scheme)
    relation_group_map = relation_group_map.to(device=device)
    edge_type = edge_type.to(device=device, dtype=torch.long)
    edge_group_ids = relation_group_map[edge_type]

    adjacencies: List[torch.Tensor] = []
    for group_id, _group_name in enumerate(group_names):
        mask = edge_group_ids == group_id
        group_edge_index = edge_index[:, mask]
        group_edge_weight = None if edge_weight is None else edge_weight[mask]
        adjacency = build_normalized_adjacency(
            edge_index=group_edge_index,
            num_nodes=num_nodes,
            edge_weight=group_edge_weight,
            add_self_loops=False,
            device=device,
        )
        adjacencies.append(adjacency)

    return group_names, adjacencies
