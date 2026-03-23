from typing import Dict, Iterable, List

import torch


CALIBRATION_METHODS = (
    "raw",
    "zscore_by_type",
    "robust_zscore_by_type",
    "percentile_by_type",
    "mix_raw25_robust75_by_type",
    "mix_raw50_robust50_by_type",
    "mix_raw75_robust25_by_type",
)


def _group_node_ids_by_type(node_types: Iterable[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for node_id, node_type in enumerate(node_types):
        groups.setdefault(node_type, []).append(node_id)
    return groups


def calibrate_raw(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    del node_types
    return scores.clone()


def calibrate_zscore_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    calibrated = torch.zeros_like(scores)
    for _, node_ids in _group_node_ids_by_type(node_types).items():
        idx = torch.tensor(node_ids, dtype=torch.long)
        group_scores = scores[idx]
        mean = group_scores.mean()
        std = group_scores.std(unbiased=False)
        if float(std.item()) < 1e-12:
            calibrated[idx] = 0.0
        else:
            calibrated[idx] = (group_scores - mean) / std
    return calibrated


def calibrate_robust_zscore_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    calibrated = torch.zeros_like(scores)
    for _, node_ids in _group_node_ids_by_type(node_types).items():
        idx = torch.tensor(node_ids, dtype=torch.long)
        group_scores = scores[idx]
        median = torch.median(group_scores)
        mad = torch.median(torch.abs(group_scores - median))
        scale = 1.4826 * mad
        if float(scale.item()) < 1e-12:
            calibrated[idx] = 0.0
        else:
            calibrated[idx] = (group_scores - median) / scale
    return calibrated


def calibrate_percentile_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    calibrated = torch.zeros_like(scores)
    for _, node_ids in _group_node_ids_by_type(node_types).items():
        idx = torch.tensor(node_ids, dtype=torch.long)
        group_scores = scores[idx]
        if group_scores.numel() == 1:
            calibrated[idx] = 1.0
            continue

        order = torch.argsort(group_scores, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(group_scores.numel(), dtype=torch.float32)
        calibrated[idx] = 1.0 - ranks / float(group_scores.numel() - 1)
    return calibrated


def _global_percentile(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return torch.ones_like(scores)

    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(scores.numel(), dtype=torch.float32)
    return 1.0 - ranks / float(scores.numel() - 1)


def _mix_raw_and_robust_by_type(scores: torch.Tensor, node_types: List[str], raw_weight: float) -> torch.Tensor:
    raw_percentile = _global_percentile(scores)
    robust_scores = calibrate_robust_zscore_by_type(scores, node_types)
    robust_percentile = _global_percentile(robust_scores)
    return raw_weight * raw_percentile + (1.0 - raw_weight) * robust_percentile


def calibrate_mix_raw25_robust75_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    return _mix_raw_and_robust_by_type(scores, node_types, raw_weight=0.25)


def calibrate_mix_raw50_robust50_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    return _mix_raw_and_robust_by_type(scores, node_types, raw_weight=0.50)


def calibrate_mix_raw75_robust25_by_type(scores: torch.Tensor, node_types: List[str]) -> torch.Tensor:
    return _mix_raw_and_robust_by_type(scores, node_types, raw_weight=0.75)


def calibrate_scores_by_method(scores: torch.Tensor, node_types: List[str], method: str) -> torch.Tensor:
    if method == "raw":
        return calibrate_raw(scores, node_types)
    if method == "zscore_by_type":
        return calibrate_zscore_by_type(scores, node_types)
    if method == "robust_zscore_by_type":
        return calibrate_robust_zscore_by_type(scores, node_types)
    if method == "percentile_by_type":
        return calibrate_percentile_by_type(scores, node_types)
    if method == "mix_raw25_robust75_by_type":
        return calibrate_mix_raw25_robust75_by_type(scores, node_types)
    if method == "mix_raw50_robust50_by_type":
        return calibrate_mix_raw50_robust50_by_type(scores, node_types)
    if method == "mix_raw75_robust25_by_type":
        return calibrate_mix_raw75_robust25_by_type(scores, node_types)
    raise ValueError(f"Unsupported calibration method: {method}")
