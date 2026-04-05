from typing import Dict, List

from analyze_file_false_positives import compute_selected_node_stats


POST_RERANK_METHODS = (
    "none",
    "file_rerank_support",
    "file_rerank_support_history",
)


def compute_percentile_lookup(values_by_key: Dict[str, float]) -> Dict[str, float]:
    if not values_by_key:
        return {}
    ordered = sorted(values_by_key.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) == 1:
        key, _ = ordered[0]
        return {key: 1.0}
    lookup: Dict[str, float] = {}
    denom = float(len(ordered) - 1)
    for rank, (key, _) in enumerate(ordered):
        lookup[key] = 1.0 - rank / denom
    return lookup


def build_previous_file_percentiles(rows: List[Dict[str, object]]) -> Dict[str, float]:
    file_scores = {
        str(row["node_uuid"]): float(row["score"])
        for row in rows
        if str(row["node_type"]) == "file"
    }
    return compute_percentile_lookup(file_scores)


def clone_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [dict(row) for row in rows]


def candidate_file_rows(rows: List[Dict[str, object]], candidate_rank_max: int) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["node_type"]) == "file" and int(row["rank"]) <= candidate_rank_max
    ]


def build_candidate_feature_tables(
    rows: List[Dict[str, object]],
    node_stats: Dict[int, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    total_degree = {
        str(row["node_uuid"]): float(node_stats.get(int(row["node_id"]), {}).get("total_degree", 0.0))
        for row in rows
    }
    unique_process_neighbors = {
        str(row["node_uuid"]): float(node_stats.get(int(row["node_id"]), {}).get("unique_process_neighbors", 0.0))
        for row in rows
    }
    file_read_edges = {
        str(row["node_uuid"]): float(node_stats.get(int(row["node_id"]), {}).get("incident_group_file_read", 0.0))
        for row in rows
    }
    return {
        "total_degree_pct": compute_percentile_lookup(total_degree),
        "unique_process_neighbors_pct": compute_percentile_lookup(unique_process_neighbors),
        "file_read_edges_pct": compute_percentile_lookup(file_read_edges),
    }


def score_support_bundle(
    base_score: float,
    uuid: str,
    feature_tables: Dict[str, Dict[str, float]],
) -> float:
    degree_pct = feature_tables["total_degree_pct"].get(uuid, 0.0)
    proc_pct = feature_tables["unique_process_neighbors_pct"].get(uuid, 0.0)
    read_pct = feature_tables["file_read_edges_pct"].get(uuid, 0.0)
    boost = 1.0 + 0.25 * degree_pct + 0.45 * proc_pct + 0.15 * read_pct
    return base_score * boost


def score_support_history_bundle(
    base_score: float,
    uuid: str,
    feature_tables: Dict[str, Dict[str, float]],
    previous_file_percentiles_by_uuid: Dict[str, float],
) -> float:
    degree_pct = feature_tables["total_degree_pct"].get(uuid, 0.0)
    proc_pct = feature_tables["unique_process_neighbors_pct"].get(uuid, 0.0)
    read_pct = feature_tables["file_read_edges_pct"].get(uuid, 0.0)
    history_pct = previous_file_percentiles_by_uuid.get(uuid, 0.0)
    boost = 1.0 + 0.45 * history_pct + 0.20 * proc_pct + 0.15 * degree_pct + 0.10 * read_pct
    return base_score * boost


def rerank_scored_rows(
    rows: List[Dict[str, object]],
    node_stats: Dict[int, Dict[str, object]],
    method_name: str,
    candidate_rank_max: int,
    previous_file_percentiles_by_uuid: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    if method_name in {"none", "base_score"}:
        return clone_rows(rows)

    reranked = clone_rows(rows)
    candidates = candidate_file_rows(reranked, candidate_rank_max)
    row_by_uuid = {str(row["node_uuid"]): row for row in reranked}
    feature_tables = build_candidate_feature_tables(candidates, node_stats)
    previous_lookup = previous_file_percentiles_by_uuid or {}

    for row in candidates:
        uuid = str(row["node_uuid"])
        base_score = float(row["score"])
        target_row = row_by_uuid[uuid]
        if method_name == "file_rerank_support":
            new_score = score_support_bundle(base_score, uuid, feature_tables)
        elif method_name == "file_rerank_support_history":
            new_score = score_support_history_bundle(
                base_score,
                uuid,
                feature_tables,
                previous_lookup,
            )
        else:
            raise ValueError(f"Unsupported rerank method: {method_name}")
        target_row["score"] = float(new_score)

    reranked.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank
    return reranked


def rerank_scored_rows_for_graph(
    rows: List[Dict[str, object]],
    graph_path,
    relation_group_scheme: str,
    method_name: str,
    candidate_rank_max: int,
    previous_file_percentiles_by_uuid: Dict[str, float] | None = None,
) -> List[Dict[str, object]]:
    candidates = candidate_file_rows(rows, candidate_rank_max)
    selected_node_ids = [int(row["node_id"]) for row in candidates]
    node_stats = compute_selected_node_stats(
        graph_path=graph_path,
        selected_node_ids=selected_node_ids,
        relation_group_scheme=relation_group_scheme,
    )
    return rerank_scored_rows(
        rows=rows,
        node_stats=node_stats,
        method_name=method_name,
        candidate_rank_max=candidate_rank_max,
        previous_file_percentiles_by_uuid=previous_file_percentiles_by_uuid,
    )
