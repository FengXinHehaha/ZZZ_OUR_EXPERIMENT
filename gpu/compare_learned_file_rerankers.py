import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from analyze_file_false_positives import (
    compute_selected_node_stats,
    load_json,
    load_scored_rows,
    percentile as linear_percentile,
    resolve_window_graph_paths,
)
from file_reranking import (
    build_candidate_feature_tables,
    build_candidate_path_feature_tables,
    build_previous_file_percentiles,
    candidate_file_rows,
    clone_rows,
    compute_percentile_lookup,
    rerank_scored_rows,
)
from train_gnn import compute_binary_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features_model_ready_hybrid_file_v2"
LEARNED_METHOD_NAME = "file_rerank_learned_linear"
DEFAULT_FEATURE_NAMES = (
    "base_score_pct",
    "total_degree_pct",
    "unique_process_neighbors_pct",
    "file_read_edges_pct",
    "history_pct",
    "known_path_count_pct",
    "risky_path_count_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight learned file reranker on one window and compare it against "
            "hand-crafted rerankers on the remaining windows."
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
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=(
            "Feature root containing <window>/file_view__file_node.tsv and "
            f"process_view__file_node.tsv. Default: {DEFAULT_FEATURE_ROOT}"
        ),
    )
    parser.add_argument(
        "--select-window",
        type=str,
        default="val",
        help="Window used to fit the lightweight learned reranker. Default: val",
    )
    parser.add_argument(
        "--candidate-rank-max",
        type=int,
        default=2000,
        help="Only rerank file nodes whose current rank is at most this cutoff. Default: 2000",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Number of optimization steps for the learned reranker. Default: 1000",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for the learned reranker. Default: 0.05",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the learned reranker. Default: 1e-4",
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
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Optional K for Precision@K / Recall@K summaries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "evaluations"),
        help="Directory for learned-reranker outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output directory name.",
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


def print_method_summary(method_name: str, summary: Dict[str, object]) -> None:
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


def build_learned_feature_rows(
    candidate_rows: List[Dict[str, object]],
    node_stats: Dict[int, Dict[str, object]],
    previous_file_percentiles_by_uuid: Dict[str, float],
    graph_path: Path,
    feature_root: Path,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    support_tables = build_candidate_feature_tables(candidate_rows, node_stats)
    path_tables = build_candidate_path_feature_tables(candidate_rows, graph_path, feature_root)
    base_score_lookup = {str(row["node_uuid"]): float(row["score"]) for row in candidate_rows}
    base_score_pct = compute_percentile_lookup(base_score_lookup)

    feature_rows: Dict[str, Dict[str, float]] = {}
    for row in candidate_rows:
        uuid = str(row["node_uuid"])
        feature_rows[uuid] = {
            "base_score_pct": base_score_pct.get(uuid, 0.0),
            "total_degree_pct": support_tables["total_degree_pct"].get(uuid, 0.0),
            "unique_process_neighbors_pct": support_tables["unique_process_neighbors_pct"].get(uuid, 0.0),
            "file_read_edges_pct": support_tables["file_read_edges_pct"].get(uuid, 0.0),
            "history_pct": previous_file_percentiles_by_uuid.get(uuid, 0.0),
            "known_path_count_pct": path_tables["known_path_count_pct"].get(uuid, 0.0),
            "risky_path_count_pct": path_tables["risky_path_count_pct"].get(uuid, 0.0),
        }
    return feature_rows, support_tables, path_tables


def tensorize_training_rows(
    candidate_rows: List[Dict[str, object]],
    feature_rows: Dict[str, Dict[str, float]],
    feature_names: Tuple[str, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[float]] = []
    ys: List[float] = []
    for row in candidate_rows:
        uuid = str(row["node_uuid"])
        xs.append([float(feature_rows[uuid].get(name, 0.0)) for name in feature_names])
        ys.append(float(int(row["is_gt"])))
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def train_learned_linear_reranker(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_names: Tuple[str, ...],
    train_steps: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, object]:
    weights = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.AdamW([weights, bias], lr=lr, weight_decay=weight_decay)

    positives = float(y.sum().item())
    negatives = float(y.shape[0] - positives)
    pos_weight_value = max(1.0, negatives / max(positives, 1.0))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32))

    best_state = None
    best_loss = math.inf
    losses: List[float] = []
    for _ in range(train_steps):
        logits = x @ weights + bias
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "weights": weights.detach().clone(),
                "bias": bias.detach().clone(),
            }

    assert best_state is not None
    learned_weights = {
        name: float(best_state["weights"][index].item())
        for index, name in enumerate(feature_names)
    }
    return {
        "feature_names": list(feature_names),
        "weights": learned_weights,
        "bias": float(best_state["bias"].item()),
        "best_loss": best_loss,
        "final_loss": losses[-1] if losses else None,
        "train_steps": train_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "positives": int(positives),
        "negatives": int(negatives),
        "pos_weight": float(pos_weight_value),
    }


def rerank_rows_with_learned_model(
    rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    feature_rows: Dict[str, Dict[str, float]],
    learned_state: Dict[str, object],
) -> List[Dict[str, object]]:
    reranked = clone_rows(rows)
    row_by_uuid = {str(row["node_uuid"]): row for row in reranked}
    feature_names: List[str] = list(learned_state["feature_names"])
    weights = learned_state["weights"]
    bias = float(learned_state["bias"])

    for row in candidate_rows:
        uuid = str(row["node_uuid"])
        target_row = row_by_uuid[uuid]
        base_score = float(target_row["score"])
        logit = bias + sum(float(weights[name]) * float(feature_rows[uuid].get(name, 0.0)) for name in feature_names)
        boost = 1.0 + 1.0 / (1.0 + math.exp(-logit))
        target_row["score"] = float(base_score * boost)

    reranked.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank
    return reranked


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()

    summary = load_json(eval_dir / "evaluation_summary.json")
    graph_summaries = list(summary["graphs"])
    window_graph_paths = resolve_window_graph_paths(eval_dir, graph_root)

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    run_name = args.run_name or f"file_learned_rerankers_{eval_dir.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    topk = args.topk or [100, 500, 1000, 5000, 10000]
    history_reset_before_windows = set(args.history_reset_before_window)
    methods = ["base_score", "file_rerank_support", "file_rerank_support_path", LEARNED_METHOD_NAME]

    window_contexts: List[Dict[str, object]] = []
    previous_file_percentiles_by_uuid: Dict[str, float] = {}
    for graph_summary in graph_summaries:
        window_name = str(graph_summary["name"])
        if window_name in history_reset_before_windows:
            previous_file_percentiles_by_uuid = {}

        base_rows = load_scored_rows(eval_dir, window_name)
        candidate_rows = candidate_file_rows(base_rows, args.candidate_rank_max)
        graph_path = window_graph_paths[window_name]
        node_stats = compute_selected_node_stats(
            graph_path=graph_path,
            selected_node_ids=[int(row["node_id"]) for row in candidate_rows],
            relation_group_scheme=args.relation_group_scheme,
        )
        feature_rows, support_tables, path_tables = build_learned_feature_rows(
            candidate_rows=candidate_rows,
            node_stats=node_stats,
            previous_file_percentiles_by_uuid=previous_file_percentiles_by_uuid,
            graph_path=graph_path,
            feature_root=feature_root,
        )
        window_contexts.append(
            {
                "graph_summary": graph_summary,
                "window_name": window_name,
                "base_rows": base_rows,
                "candidate_rows": candidate_rows,
                "graph_path": graph_path,
                "node_stats": node_stats,
                "support_tables": support_tables,
                "path_tables": path_tables,
                "feature_rows": feature_rows,
                "previous_file_percentiles_by_uuid": dict(previous_file_percentiles_by_uuid),
            }
        )
        previous_file_percentiles_by_uuid = build_previous_file_percentiles(base_rows)

    select_context = next((ctx for ctx in window_contexts if ctx["window_name"] == args.select_window), None)
    if select_context is None:
        raise ValueError(f"Select window not found in eval dir: {args.select_window}")

    x_train, y_train = tensorize_training_rows(
        select_context["candidate_rows"],
        select_context["feature_rows"],
        DEFAULT_FEATURE_NAMES,
    )
    if int(y_train.sum().item()) <= 0:
        raise ValueError(
            "No positive file candidates found in the select window within the current "
            "--candidate-rank-max. Increase the cutoff or choose a different select window."
        )
    learned_state = train_learned_linear_reranker(
        x=x_train,
        y=y_train,
        feature_names=DEFAULT_FEATURE_NAMES,
        train_steps=args.train_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "base_post_rerank_method": summary.get("post_rerank_method"),
        "candidate_rank_max": args.candidate_rank_max,
        "select_window": args.select_window,
        "history_reset_before_windows": list(args.history_reset_before_window),
        "relation_group_scheme": args.relation_group_scheme,
        "feature_root": str(feature_root),
        "methods": methods,
        "topk": topk,
        "learned_state": learned_state,
        "graphs": [],
    }

    print(f"[compare-learned-file-rerankers] eval_dir={eval_dir}", flush=True)
    print(f"[compare-learned-file-rerankers] feature_root={feature_root}", flush=True)
    print(f"[compare-learned-file-rerankers] select_window={args.select_window}", flush=True)
    print(f"[compare-learned-file-rerankers] candidate_rank_max={args.candidate_rank_max}", flush=True)
    print(f"[compare-learned-file-rerankers] learned_state={learned_state}", flush=True)

    for context in window_contexts:
        graph_summary = context["graph_summary"]
        window_name = context["window_name"]
        base_rows = context["base_rows"]
        candidate_rows = context["candidate_rows"]
        node_stats = context["node_stats"]
        support_tables = context["support_tables"]
        path_tables = context["path_tables"]
        feature_rows = context["feature_rows"]

        print(
            f"[compare-learned-file-rerankers] evaluating {window_name} "
            f"candidate_files={len(candidate_rows)}",
            flush=True,
        )

        method_rows = {
            "base_score": clone_rows(base_rows),
            "file_rerank_support": rerank_scored_rows(
                rows=base_rows,
                node_stats=node_stats,
                method_name="file_rerank_support",
                candidate_rank_max=args.candidate_rank_max,
            ),
            "file_rerank_support_path": rerank_scored_rows(
                rows=base_rows,
                node_stats=node_stats,
                method_name="file_rerank_support_path",
                candidate_rank_max=args.candidate_rank_max,
                path_feature_tables=path_tables,
            ),
            LEARNED_METHOD_NAME: rerank_rows_with_learned_model(
                rows=base_rows,
                candidate_rows=candidate_rows,
                feature_rows=feature_rows,
                learned_state=learned_state,
            ),
        }

        method_summaries: Dict[str, object] = {}
        window_output_dir = run_dir / window_name
        ensure_dir(window_output_dir)
        for method_name in methods:
            node_scores_path = window_output_dir / method_name / "node_scores.tsv"
            ensure_dir(node_scores_path.parent)
            method_summaries[method_name] = summarize_method(
                graph_summary=graph_summary,
                reranked_rows=method_rows[method_name],
                topk=topk,
                node_scores_path=node_scores_path,
            )
            print_method_summary(method_name, method_summaries[method_name])

        results["graphs"].append(
            {
                "name": window_name,
                "path": graph_summary["path"],
                "candidate_file_count": len(candidate_rows),
                "methods": method_summaries,
            }
        )

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[compare-learned-file-rerankers] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
