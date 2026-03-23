import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from score_aggregation import SCORE_METHODS, compute_node_scores_by_method
from score_calibration import CALIBRATION_METHODS, calibrate_scores_by_method
from train_gnn import (
    DEFAULT_EVAL_GRAPHS,
    MultiViewFullBatchGAE,
    compute_binary_metrics,
    prepare_graph_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "training_runs" / "gpu_trial_60ep" / "best_model.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare node-score calibration methods on the same checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to best_model.pt or last_model.pt. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--graph",
        action="append",
        default=[],
        help="Optional graph.pt path(s). Defaults to checkpoint config eval_graphs or val+test windows.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for evaluation.",
    )
    parser.add_argument(
        "--base-score-method",
        type=str,
        default="top5_mean",
        choices=sorted(SCORE_METHODS),
        help="Base node-score aggregation before calibration. Default: top5_mean.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Optional K for Precision@K / Recall@K. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for comparison outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional comparison run name.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_graphs(args_graphs: List[str], checkpoint_config: Dict[str, object]) -> List[Path]:
    if args_graphs:
        return [Path(item).expanduser().resolve() for item in args_graphs]

    config_graphs = checkpoint_config.get("eval_graphs", [])
    if config_graphs:
        return [Path(item).expanduser().resolve() for item in config_graphs]

    return [path.resolve() for path in DEFAULT_EVAL_GRAPHS]


def read_nodes_table(nodes_path: Path) -> List[Dict[str, str]]:
    with nodes_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def percentile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


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
    median_rank = percentile(ranks, 0.50)
    best_rank = ranks[0]
    return {
        "best_rank": best_rank,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "p10_rank": percentile(ranks, 0.10),
        "p25_rank": percentile(ranks, 0.25),
        "p75_rank": percentile(ranks, 0.75),
        "p90_rank": percentile(ranks, 0.90),
        "best_rank_ratio": best_rank / total_nodes if total_nodes else 0.0,
        "median_rank_ratio": median_rank / total_nodes if total_nodes else 0.0,
    }


def score_rows(
    nodes_rows: List[Dict[str, str]],
    node_scores: torch.Tensor,
) -> List[Dict[str, object]]:
    scored_rows: List[Dict[str, object]] = []
    for row in nodes_rows:
        node_id = int(row["node_id"])
        scored_rows.append(
            {
                "node_id": node_id,
                "node_uuid": row["node_uuid"],
                "node_type": row["node_type"],
                "is_gt": int(row["is_gt"]),
                "score": float(node_scores[node_id].item()),
            }
        )
    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
    return scored_rows


def summarize_method(
    graph_name: str,
    num_nodes: int,
    num_edges: int,
    gt_total: int,
    y: torch.Tensor,
    node_scores: torch.Tensor,
    nodes_rows: List[Dict[str, str]],
    topk: List[int],
) -> Dict[str, object]:
    scored_rows = score_rows(nodes_rows, node_scores)
    node_metrics = compute_binary_metrics(y, node_scores)
    topk_summary = build_topk_summary([row["is_gt"] for row in scored_rows], gt_total, topk)
    gt_ranks = [int(row["rank"]) for row in scored_rows if int(row["is_gt"]) == 1]
    rank_summary = summarize_gt_ranks(gt_ranks, num_nodes)

    return {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "gt_nodes": gt_total,
        "roc_auc": node_metrics["roc_auc"],
        "average_precision": node_metrics["average_precision"],
        "rank_summary": rank_summary,
        "topk": topk_summary,
    }


def evaluate_single_graph(
    model: MultiViewFullBatchGAE,
    graph_path: Path,
    device: torch.device,
    base_score_method: str,
    topk: List[int],
    window_output_dir: Path,
) -> Dict[str, object]:
    payload = prepare_graph_payload(graph_path, device=device, restrict_to_normal_edges=False)

    model.eval()
    with torch.no_grad():
        encoded = model.encode(payload["x_views"], payload["adjacency"])
        z_fused = encoded["z_fused"]
        edge_error = 1.0 - torch.sigmoid(
            model.decode_edges(z_fused, payload["edge_index"], payload["edge_type"])
        )

    nodes_rows = read_nodes_table(graph_path.parent / "nodes.tsv")
    node_types = [row["node_type"] for row in nodes_rows]
    base_scores = compute_node_scores_by_method(
        payload["num_nodes"],
        payload["edge_index"].detach().cpu(),
        edge_error.detach().cpu().to(dtype=torch.float32),
        base_score_method,
    )
    y_cpu = payload["y"].detach().cpu().to(dtype=torch.float32)

    method_summaries: Dict[str, object] = {}
    for method_name in CALIBRATION_METHODS:
        calibrated_scores = calibrate_scores_by_method(base_scores, node_types, method_name)
        method_summaries[method_name] = summarize_method(
            graph_name=payload["name"],
            num_nodes=payload["num_nodes"],
            num_edges=int(payload["edge_index"].shape[1]),
            gt_total=int(payload["y"].sum().item()),
            y=y_cpu,
            node_scores=calibrated_scores,
            nodes_rows=nodes_rows,
            topk=topk,
        )

    summary = {
        "name": payload["name"],
        "path": str(graph_path),
        "base_score_method": base_score_method,
        "methods": method_summaries,
    }

    ensure_dir(window_output_dir)
    with (window_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    graph_paths = resolve_graphs(args.graph, config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    run_name = args.run_name or f"score_calibs_{checkpoint_path.parent.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    model = MultiViewFullBatchGAE(
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
        decoder_type=str(config.get("decoder_type", "dot")),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", config["latent_dim"] * 2)),
        num_relations=int(config.get("num_relations", 0)),
        relation_embedding_dim=int(config.get("relation_embedding_dim", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    topk = args.topk or [100, 500, 1000, 5000, 10000]

    print(f"[compare-score-calibs] checkpoint={checkpoint_path}", flush=True)
    print(f"[compare-score-calibs] device={device}", flush=True)
    print(f"[compare-score-calibs] base_score_method={args.base_score_method}", flush=True)
    print(f"[compare-score-calibs] graphs={[str(path) for path in graph_paths]}", flush=True)

    aggregate = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_record": checkpoint.get("best_record"),
        "selection_metric": config.get("selection_metric"),
        "base_score_method": args.base_score_method,
        "topk": topk,
        "graphs": [],
    }

    for graph_path in graph_paths:
        graph_path = graph_path.resolve()
        window_name = graph_path.parent.name
        print(f"[compare-score-calibs] evaluating {window_name}", flush=True)
        summary = evaluate_single_graph(
            model=model,
            graph_path=graph_path,
            device=device,
            base_score_method=args.base_score_method,
            topk=topk,
            window_output_dir=run_dir / window_name,
        )
        aggregate["graphs"].append(summary)
        for method_name, method_summary in summary["methods"].items():
            print_method_summary(method_name, method_summary)

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"[compare-score-calibs] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
