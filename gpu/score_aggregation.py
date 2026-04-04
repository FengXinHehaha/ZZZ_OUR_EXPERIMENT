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
)


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


def compute_all_score_methods(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    node_types: List[str] | None = None,
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

    return {
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


def compute_node_scores_by_method(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    method: str,
    node_types: List[str] | None = None,
) -> torch.Tensor:
    score_methods = compute_all_score_methods(num_nodes, edge_index, edge_error, node_types=node_types)
    if method not in score_methods:
        raise ValueError(f"Unsupported score method: {method}")
    return score_methods[method]
