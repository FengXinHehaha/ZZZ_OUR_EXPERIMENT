import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from analyze_file_false_positives import (
    load_json,
    load_scored_rows,
    percentile as linear_percentile,
    resolve_window_graph_paths,
)
from file_reranking import (
    POST_RERANK_METHODS,
    build_previous_file_percentiles,
    candidate_file_rows,
    rerank_scored_rows_for_graph,
)
from train_gnn import compute_binary_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare file-specific second-stage rerankers on top of an existing evaluation directory. "
            "The reranker only adjusts file nodes within a candidate rank band and keeps other nodes unchanged."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--graph-root",
        type=str,
        default=str(DEFAULT_GRAPH_ROOT),
        help=f"Fallback root containing <window>/graph.pt. Default: {DEFAULT_GRAPH_ROOT}",
    )
    parser.add_argument(
        "--candidate-rank-max",
        type=int,
        default=10000,
        help="Only rerank file nodes whose current rank is at most this cutoff. Default: 10000",
    )
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Optional K for Precision@K / Recall@K summaries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--history-reset-before-window",
        action="append",
        default=[],
        help="Optional window name(s) before which previous-window file history is cleared.",
    )
    parser.add_argument(
        "--relation-group-scheme",
        type=str,
        default="coarse_v1",
        help="Relation grouping used when deriving file-node graph stats. Default: coarse_v1",
    )
    parser.add_argument(
        "--post-rerank-feature-root",
        type=str,
        default="",
        help=(
            "Optional feature root containing <window>/file_view__file_node.tsv and "
            "process_view__file_node.tsv for path-aware rerankers. Default: auto-infer from graph path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "evaluations"),
        help="Directory for reranker comparison outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional reranker comparison run name.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "node_id", "node_uuid", "node_type", "is_gt", "score"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def precision_recall_at_k(is_gt_sorted: List[int], total_gt: int, k: int) -> Dict[str, float]:
    capped_k = min(k, len(is_gt_sorted))
    hits = sum(is_gt_sorted[:capped_k])
    precision = hits / capped_k if capped_k > 0 else 0.0
    recall = hits / total_gt if total_gt > 0 else 0.0
    return {
        "k": capped_k,
        "hits": int(hits),
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
    }


def build_topk_summary(is_gt_sorted: List[int], total_gt: int, ks: List[int]) -> List[Dict[str, float]]:
    unique_ks: List[int] = []
    seen = set()
    for k in ks:
        if k > 0 and k not in seen:
            unique_ks.append(k)
            seen.add(k)
    return [precision_recall_at_k(is_gt_sorted, total_gt, k) for k in unique_ks]


def summarize_gt_ranks(ranks: List[int], total_nodes: int) -> Dict[str, float]:
    ranks = sorted(ranks)
    if not ranks:
        return {
            "best_rank": 0,
            "median_rank": 0.0,
            "mean_rank": 0.0,
            "p10_rank": 0.0,
            "p25_rank": 0.0,
            "p75_rank": 0.0,
            "p90_rank": 0.0,
            "best_rank_ratio": 0.0,
            "median_rank_ratio": 0.0,
        }
    mean_rank = sum(ranks) / len(ranks)
    median_rank = linear_percentile(ranks, 0.50)
    best_rank = ranks[0]
    return {
        "best_rank": best_rank,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "p10_rank": linear_percentile(ranks, 0.10),
        "p25_rank": linear_percentile(ranks, 0.25),
        "p75_rank": linear_percentile(ranks, 0.75),
        "p90_rank": linear_percentile(ranks, 0.90),
        "best_rank_ratio": best_rank / total_nodes if total_nodes else 0.0,
        "median_rank_ratio": median_rank / total_nodes if total_nodes else 0.0,
    }


def clone_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cloned: List[Dict[str, object]] = []
    for row in rows:
        cloned.append(dict(row))
    return cloned


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


def rerank_rows_for_method(
    rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    node_stats: Dict[int, Dict[str, object]],
    previous_file_percentiles_by_uuid: Dict[str, float],
    method_name: str,
) -> List[Dict[str, object]]:
    reranked = clone_rows(rows)
    row_by_uuid = {str(row["node_uuid"]): row for row in reranked}
    feature_tables = build_candidate_feature_tables(candidate_rows, node_stats)

    for row in candidate_rows:
        uuid = str(row["node_uuid"])
        base_score = float(row["score"])
        target_row = row_by_uuid[uuid]
        if method_name == "base_score":
            new_score = base_score
        elif method_name == "file_rerank_support":
            new_score = score_support_bundle(base_score, uuid, feature_tables)
        elif method_name == "file_rerank_support_history":
            new_score = score_support_history_bundle(
                base_score,
                uuid,
                feature_tables,
                previous_file_percentiles_by_uuid,
            )
        else:
            raise ValueError(f"Unsupported rerank method: {method_name}")
        target_row["score"] = float(new_score)

    reranked.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank
    return reranked


def summarize_method(
    graph_summary: Dict[str, object],
    reranked_rows: List[Dict[str, object]],
    topk: List[int],
    node_scores_path: Path,
) -> Dict[str, object]:
    y = torch.tensor([int(row["is_gt"]) for row in reranked_rows], dtype=torch.float32)
    scores = torch.tensor([float(row["score"]) for row in reranked_rows], dtype=torch.float32)
    node_metrics = compute_binary_metrics(y, scores)
    gt_total = int(sum(int(row["is_gt"]) for row in reranked_rows))
    gt_ranks = [int(row["rank"]) for row in reranked_rows if int(row["is_gt"]) == 1]
    rank_summary = summarize_gt_ranks(gt_ranks, len(reranked_rows))
    topk_summary = build_topk_summary([int(row["is_gt"]) for row in reranked_rows], gt_total, topk)

    write_rows(node_scores_path, reranked_rows)
    return {
        "graph_name": graph_summary["name"],
        "num_nodes": len(reranked_rows),
        "num_edges": graph_summary["num_edges"],
        "gt_nodes": gt_total,
        "roc_auc": node_metrics["roc_auc"],
        "average_precision": node_metrics["average_precision"],
        "rank_summary": rank_summary,
        "topk": topk_summary,
        "node_scores_file": str(node_scores_path),
    }


def print_method_summary(window_name: str, method_name: str, summary: Dict[str, object]) -> None:
    rank_summary = summary["rank_summary"]
    topk_map = {entry["k"]: entry for entry in summary["topk"]}
    top10000_hits = topk_map.get(10000, {"hits": None})["hits"]
    print(
        f"  [method] {method_name}: "
        f"roc_auc={summary['roc_auc']} ap={summary['average_precision']} "
        f"best_gt_rank={rank_summary['best_rank']} "
        f"median_gt_rank={rank_summary['median_rank']:.1f} "
        f"top10000_hits={top10000_hits}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    graph_summaries = summary["graphs"]
    window_graph_paths = resolve_window_graph_paths(eval_dir, graph_root)

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    run_name = args.run_name or f"file_rerankers_{eval_dir.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    topk = args.topk or [100, 500, 1000, 5000, 10000]
    methods = ["base_score", *[method for method in POST_RERANK_METHODS if method != "none"]]
    history_reset_before_windows = set(args.history_reset_before_window)

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "candidate_rank_max": args.candidate_rank_max,
        "history_reset_before_windows": list(args.history_reset_before_window),
        "relation_group_scheme": args.relation_group_scheme,
        "post_rerank_feature_root": args.post_rerank_feature_root,
        "methods": methods,
        "topk": topk,
        "graphs": [],
    }

    print(f"[compare-file-rerankers] eval_dir={eval_dir}", flush=True)
    print(f"[compare-file-rerankers] candidate_rank_max={args.candidate_rank_max}", flush=True)
    print(f"[compare-file-rerankers] methods={methods}", flush=True)

    previous_file_percentiles_by_uuid: Dict[str, float] = {}
    for graph_summary in graph_summaries:
        window_name = str(graph_summary["name"])
        if window_name in history_reset_before_windows:
            previous_file_percentiles_by_uuid = {}

        base_rows = load_scored_rows(eval_dir, window_name)
        candidate_rows = candidate_file_rows(base_rows, args.candidate_rank_max)
        graph_path = window_graph_paths[window_name]

        print(
            f"[compare-file-rerankers] evaluating {window_name} "
            f"candidate_files={len(candidate_rows)}",
            flush=True,
        )

        method_summaries: Dict[str, object] = {}
        window_output_dir = run_dir / window_name
        ensure_dir(window_output_dir)
        for method_name in methods:
            reranked_rows = rerank_scored_rows_for_graph(
                rows=base_rows,
                graph_path=graph_path,
                relation_group_scheme=args.relation_group_scheme,
                method_name=method_name,
                candidate_rank_max=args.candidate_rank_max,
                previous_file_percentiles_by_uuid=previous_file_percentiles_by_uuid,
                feature_root=args.post_rerank_feature_root or None,
            )
            node_scores_path = window_output_dir / method_name / "node_scores.tsv"
            ensure_dir(node_scores_path.parent)
            method_summaries[method_name] = summarize_method(
                graph_summary=graph_summary,
                reranked_rows=reranked_rows,
                topk=topk,
                node_scores_path=node_scores_path,
            )
            print_method_summary(window_name, method_name, method_summaries[method_name])

        results["graphs"].append(
            {
                "name": window_name,
                "path": graph_summary["path"],
                "candidate_file_count": len(candidate_rows),
                "methods": method_summaries,
            }
        )
        previous_file_percentiles_by_uuid = build_previous_file_percentiles(base_rows)

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[compare-file-rerankers] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
