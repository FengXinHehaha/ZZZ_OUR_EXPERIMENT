from typing import Dict

import torch


SCORE_METHODS = ("mean", "max", "top5_mean", "top10_mean", "q90")


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


def compute_all_score_methods(num_nodes: int, edge_index: torch.Tensor, edge_error: torch.Tensor) -> Dict[str, torch.Tensor]:
    mean_scores = compute_node_scores_mean(num_nodes, edge_index, edge_error)
    max_scores = compute_node_scores_max(num_nodes, edge_index, edge_error)
    incident_values, starts, counts = build_incident_segments(edge_index, edge_error, num_nodes)
    top5_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="topk_mean", param=5)
    top10_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="topk_mean", param=10)
    q90_scores = compute_segment_scores(num_nodes, incident_values, starts, counts, mode="quantile", param=0.90)

    return {
        "mean": mean_scores,
        "max": max_scores,
        "top5_mean": top5_scores,
        "top10_mean": top10_scores,
        "q90": q90_scores,
    }


def compute_node_scores_by_method(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_error: torch.Tensor,
    method: str,
) -> torch.Tensor:
    score_methods = compute_all_score_methods(num_nodes, edge_index, edge_error)
    if method not in score_methods:
        raise ValueError(f"Unsupported score method: {method}")
    return score_methods[method]
