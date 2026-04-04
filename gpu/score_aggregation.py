import math
from typing import Dict, List

import torch


SCORE_METHODS = (
    "mean",
    "max",
    "top5_mean",
    "top10_mean",
    "q90",
    "top5_mean_log_degree_all",
    "top5_mean_log_degree_file",
    "top5_mean_sqrt_degree_file",
    "top10_mean_log_degree_file",
    "top5_mean_log_support_floor32_file",
    "top5_mean_log_support_floor128_file",
    "top10_mean_log_support_floor32_file",
    "top5_mean_log_support_floor128_file_history_file_only",
    "top5_mean_log_support_floor128_file_history_file_process",
    "top5_mean_log_support_floor128_file_history_all_types",
)

HISTORY_SOURCE_METHOD = "top5_mean_log_support_floor128_file"

HISTORY_METHOD_CONFIGS: Dict[str, Dict[str, object]] = {
    "top5_mean_log_support_floor128_file_history_file_only": {
        "base_method": HISTORY_SOURCE_METHOD,
        "type_weights": {
            "file": 1.0,
            "process": 0.0,
            "network": 0.0,
        },
    },
    "top5_mean_log_support_floor128_file_history_file_process": {
        "base_method": HISTORY_SOURCE_METHOD,
        "type_weights": {
            "file": 1.0,
            "process": 0.35,
            "network": 0.0,
        },
    },
    "top5_mean_log_support_floor128_file_history_all_types": {
        "base_method": HISTORY_SOURCE_METHOD,
        "type_weights": {
            "file": 1.0,
            "process": 0.35,
            "network": 0.20,
        },
    },
}


def compute_node_scores_mean(num_nodes: int, edge_index: torch.Tensor, edge_error: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].long()
    dst = edge_index[1].long()
    scores = torch.zeros(num_nodes, dtype=torch.float32)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    ones = torch.ones(edge_error.shape[0], dtype=torch.float32)

    scores.index_add_(0, src, edge_error)
    scores.index_add_(0, dst, edge_error)
    degrees.index_add_(0, src, ones)
    degrees.index_add_(0, dst, ones)
    return scores / degrees.clamp_min(1.0)


def compute_node_scores_max(num_nodes: int, edge_index: torch.Tensor, edge_error: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].long()
    dst = edge_index[1].long()
    scores = torch.full((num_nodes,), float("-inf"), dtype=torch.float32)
    scores.scatter_reduce_(0, src, edge_error, reduce="amax", include_self=True)
    scores.scatter_reduce_(0, dst, edge_error, reduce="amax", include_self=True)
    scores[~torch.isfinite(scores)] = 0.0
    return scores


def build_incident_segments(
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    incident_nodes = torch.cat([edge_index[0].long(), edge_index[1].long()], dim=0)
    incident_values = torch.cat([edge_error, edge_error], dim=0)
    order = torch.argsort(incident_nodes)
    incident_nodes = incident_nodes[order]
    incident_values = incident_values[order]

    counts = torch.bincount(incident_nodes, minlength=num_nodes)
    offsets = torch.cumsum(counts, dim=0)
    starts = offsets - counts
    return incident_values, starts, counts


def compute_segment_scores(
    num_nodes: int,
    incident_values: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    mode: str,
    param: float,
) -> torch.Tensor:
    scores = torch.zeros(num_nodes, dtype=torch.float32)
    nonzero_nodes = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()

    for node_id in nonzero_nodes:
        start = int(starts[node_id].item())
        count = int(counts[node_id].item())
        segment = incident_values[start : start + count]
        if mode == "topk_mean":
            topk = min(int(param), count)
            scores[node_id] = torch.topk(segment, k=topk).values.mean()
        elif mode == "quantile":
            scores[node_id] = torch.quantile(segment, float(param))
        else:
            raise ValueError(f"Unsupported segment mode: {mode}")

    return scores


def degree_penalty_from_counts(counts: torch.Tensor, mode: str) -> torch.Tensor:
    counts_float = counts.to(dtype=torch.float32)
    if mode == "log":
        return torch.log1p(counts_float).clamp_min(1.0)
    if mode == "sqrt":
        return torch.sqrt(counts_float).clamp_min(1.0)
    raise ValueError(f"Unsupported degree penalty mode: {mode}")


def apply_degree_penalty(
    scores: torch.Tensor,
    counts: torch.Tensor,
    node_types: List[str] | None,
    degree_mode: str,
    target_node_type: str | None,
) -> torch.Tensor:
    penalized = scores.clone()
    penalty = degree_penalty_from_counts(counts, degree_mode)
    mask = counts > 0
    if target_node_type is not None:
        if node_types is None:
            raise ValueError(f"node_types are required for target_node_type={target_node_type}")
        type_mask = torch.tensor(
            [node_type == target_node_type for node_type in node_types],
            dtype=torch.bool,
        )
        mask = mask & type_mask
    penalized[mask] = penalized[mask] / penalty[mask]
    return penalized


def apply_support_floor(
    scores: torch.Tensor,
    counts: torch.Tensor,
    node_types: List[str] | None,
    target_node_type: str,
    full_support_degree: int,
    min_scale: float = 0.25,
) -> torch.Tensor:
    if full_support_degree <= 1:
        raise ValueError("full_support_degree must be > 1")
    if node_types is None:
        raise ValueError(f"node_types are required for target_node_type={target_node_type}")

    supported = scores.clone()
    count_values = counts.to(dtype=torch.float32)
    log_counts = torch.log1p(count_values)
    full_support_log = math.log1p(float(full_support_degree))
    scales = (log_counts / full_support_log).clamp(min=min_scale, max=1.0)
    type_mask = torch.tensor(
        [node_type == target_node_type for node_type in node_types],
        dtype=torch.bool,
    )
    mask = (count_values > 0) & type_mask
    supported[mask] = supported[mask] * scales[mask]
    return supported


def compute_percentile_scores_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    percentiles = torch.zeros_like(scores)
    grouped: Dict[str, List[int]] = {}
    for node_id, node_type in enumerate(node_types):
        grouped.setdefault(node_type, []).append(node_id)

    for _, node_ids in grouped.items():
        idx = torch.tensor(node_ids, dtype=torch.long)
        group_scores = scores[idx]
        if group_scores.numel() <= 1:
            percentiles[idx] = 1.0
            continue
        order = torch.argsort(group_scores, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(group_scores.numel(), dtype=torch.float32)
        percentiles[idx] = 1.0 - ranks / float(group_scores.numel() - 1)

    return percentiles


def build_history_source_percentiles_by_uuid(
    scores: torch.Tensor,
    nodes_rows: List[Dict[str, str]],
    node_types: List[str],
) -> Dict[str, float]:
    percentiles = compute_percentile_scores_by_type(scores, node_types)
    return {
        row["node_uuid"]: float(percentiles[int(row["node_id"])].item())
        for row in nodes_rows
    }


def apply_history_boost(
    base_scores: torch.Tensor,
    nodes_rows: List[Dict[str, str]],
    previous_history_percentiles_by_uuid: Dict[str, float] | None,
    type_weights: Dict[str, float],
) -> torch.Tensor:
    if not previous_history_percentiles_by_uuid:
        return base_scores.clone()

    boosted = base_scores.clone()
    for row in nodes_rows:
        node_id = int(row["node_id"])
        node_uuid = row["node_uuid"]
        node_type = row["node_type"]
        history_percentile = previous_history_percentiles_by_uuid.get(node_uuid)
        if history_percentile is None:
            continue
        history_weight = float(type_weights.get(node_type, 0.0))
        if history_weight <= 0.0:
            continue
        boosted[node_id] = boosted[node_id] * (1.0 + history_weight * history_percentile)
    return boosted


def compute_all_score_methods(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    node_types: List[str] | None = None,
    nodes_rows: List[Dict[str, str]] | None = None,
    previous_history_percentiles_by_uuid: Dict[str, float] | None = None,
) -> Dict[str, torch.Tensor]:
    mean_scores = compute_node_scores_mean(num_nodes, edge_index, edge_error)
    max_scores = compute_node_scores_max(num_nodes, edge_index, edge_error)
    incident_values, starts, counts = build_incident_segments(edge_index, edge_error, num_nodes)
    top5_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="topk_mean", param=5)
    top10_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="topk_mean", param=10)
    q90_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="quantile", param=0.90)
    top5_log_degree_all_scores = apply_degree_penalty(
        top5_scores,
        counts,
        node_types,
        degree_mode="log",
        target_node_type=None,
    )
    top5_log_degree_file_scores = apply_degree_penalty(
        top5_scores,
        counts,
        node_types,
        degree_mode="log",
        target_node_type="file",
    )
    top5_sqrt_degree_file_scores = apply_degree_penalty(
        top5_scores,
        counts,
        node_types,
        degree_mode="sqrt",
        target_node_type="file",
    )
    top10_log_degree_file_scores = apply_degree_penalty(
        top10_scores,
        counts,
        node_types,
        degree_mode="log",
        target_node_type="file",
    )
    top5_log_support_floor32_file_scores = apply_support_floor(
        top5_scores,
        counts,
        node_types,
        target_node_type="file",
        full_support_degree=32,
        min_scale=0.25,
    )
    top5_log_support_floor128_file_scores = apply_support_floor(
        top5_scores,
        counts,
        node_types,
        target_node_type="file",
        full_support_degree=128,
        min_scale=0.25,
    )
    top10_log_support_floor32_file_scores = apply_support_floor(
        top10_scores,
        counts,
        node_types,
        target_node_type="file",
        full_support_degree=32,
        min_scale=0.25,
    )

    score_methods = {
        "mean": mean_scores,
        "max": max_scores,
        "top5_mean": top5_scores,
        "top10_mean": top10_scores,
        "q90": q90_scores,
        "top5_mean_log_degree_all": top5_log_degree_all_scores,
        "top5_mean_log_degree_file": top5_log_degree_file_scores,
        "top5_mean_sqrt_degree_file": top5_sqrt_degree_file_scores,
        "top10_mean_log_degree_file": top10_log_degree_file_scores,
        "top5_mean_log_support_floor32_file": top5_log_support_floor32_file_scores,
        "top5_mean_log_support_floor128_file": top5_log_support_floor128_file_scores,
        "top10_mean_log_support_floor32_file": top10_log_support_floor32_file_scores,
    }

    if nodes_rows is not None and node_types is not None:
        for method_name, config in HISTORY_METHOD_CONFIGS.items():
            score_methods[method_name] = apply_history_boost(
                base_scores=score_methods[str(config["base_method"])],
                nodes_rows=nodes_rows,
                previous_history_percentiles_by_uuid=previous_history_percentiles_by_uuid,
                type_weights=dict(config["type_weights"]),
            )

    return score_methods


def compute_node_scores_by_method(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    method: str,
    node_types: List[str] | None = None,
    nodes_rows: List[Dict[str, str]] | None = None,
    previous_history_percentiles_by_uuid: Dict[str, float] | None = None,
) -> torch.Tensor:
    score_methods = compute_all_score_methods(
        num_nodes,
        edge_index,
        edge_error,
        node_types=node_types,
        nodes_rows=nodes_rows,
        previous_history_percentiles_by_uuid=previous_history_percentiles_by_uuid,
    )
    if method not in score_methods:
        raise ValueError(f"Unsupported score method: {method}")
    return score_methods[method]
