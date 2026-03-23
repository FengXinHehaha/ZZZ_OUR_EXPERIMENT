import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from train_gnn import (
    DEFAULT_EVAL_GRAPHS,
    MultiViewFullBatchGAE,
    compute_binary_metrics,
    compute_node_scores,
    maybe_cap_edge_index,
    prepare_graph_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "training_runs" / "gpu_trial_60ep" / "best_model.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint and export node-level anomaly scores."
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
        "--edge-cap",
        type=int,
        default=0,
        help="Optional cap on positive edges for edge-loss reporting. 0 means use all.",
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
        help=f"Directory for evaluation outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional evaluation run name.",
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
    unique_ks = []
    seen = set()
    for k in ks:
        if k > 0 and k not in seen:
            unique_ks.append(k)
            seen.add(k)
    return [precision_recall_at_k(is_gt_sorted, total_gt, k) for k in unique_ks]


def evaluate_single_graph(
    model: MultiViewFullBatchGAE,
    graph_path: Path,
    device: torch.device,
    edge_cap: int,
    topk: List[int],
    window_output_dir: Path,
) -> Dict[str, object]:
    payload = prepare_graph_payload(graph_path, device=device, restrict_to_normal_edges=False)

    model.eval()
    with torch.no_grad():
        encoded = model.encode(payload["x_views"], payload["adjacency"])
        z_fused = encoded["z_fused"]

        positive_edges = maybe_cap_edge_index(payload["edge_index"], edge_cap)
        negative_edges = maybe_cap_edge_index(payload["edge_index"], edge_cap)
        if negative_edges.shape[1] > 0:
            negative_edges = torch.stack(
                [
                    torch.randint(0, payload["num_nodes"], (negative_edges.shape[1],), device=device),
                    torch.randint(0, payload["num_nodes"], (negative_edges.shape[1],), device=device),
                ],
                dim=0,
            )

        pos_logits = model.decode_edges(z_fused, positive_edges)
        neg_logits = model.decode_edges(z_fused, negative_edges)

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        edge_loss = float((pos_loss + neg_loss).item())

        edge_error = 1.0 - torch.sigmoid(model.decode_edges(z_fused, payload["edge_index"]))
        node_scores = compute_node_scores(payload["num_nodes"], payload["edge_index"], edge_error)
        node_metrics = compute_binary_metrics(payload["y"], node_scores)

        gate_means = encoded["gate_alpha"].mean(dim=0).detach().cpu().tolist()

    nodes_rows = read_nodes_table(graph_path.parent / "nodes.tsv")
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

    ensure_dir(window_output_dir)
    node_scores_path = window_output_dir / "node_scores.tsv"
    with node_scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "node_id", "node_uuid", "node_type", "is_gt", "score"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(scored_rows)

    total_gt = int(payload["y"].sum().item())
    topk_summary = build_topk_summary([row["is_gt"] for row in scored_rows], total_gt, topk)

    summary = {
        "name": payload["name"],
        "path": str(graph_path),
        "num_nodes": payload["num_nodes"],
        "num_edges": int(payload["edge_index"].shape[1]),
        "gt_nodes": total_gt,
        "edge_loss": edge_loss,
        "roc_auc": node_metrics["roc_auc"],
        "average_precision": node_metrics["average_precision"],
        "mean_gate_alpha": {
            "process_view": float(gate_means[0]),
            "file_view": float(gate_means[1]),
            "network_view": float(gate_means[2]),
        },
        "topk": topk_summary,
        "node_scores_file": str(node_scores_path),
    }

    with (window_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    graph_paths = resolve_graphs(args.graph, config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    run_name = args.run_name or f"eval_{checkpoint_path.parent.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    model = MultiViewFullBatchGAE(
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    topk = args.topk or [100, 1000, 5000, 10000]

    print(f"[eval-checkpoint] checkpoint={checkpoint_path}", flush=True)
    print(f"[eval-checkpoint] device={device}", flush=True)
    print(f"[eval-checkpoint] graphs={[str(path) for path in graph_paths]}", flush=True)

    aggregate = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_record": checkpoint.get("best_record"),
        "selection_metric": config.get("selection_metric"),
        "graphs": [],
    }

    for graph_path in graph_paths:
        graph_path = graph_path.resolve()
        window_name = graph_path.parent.name
        print(f"[eval-checkpoint] evaluating {window_name}", flush=True)
        summary = evaluate_single_graph(
            model=model,
            graph_path=graph_path,
            device=device,
            edge_cap=args.edge_cap,
            topk=topk,
            window_output_dir=run_dir / window_name,
        )
        aggregate["graphs"].append(summary)
        print(
            f"[eval-checkpoint] {window_name}: "
            f"roc_auc={summary['roc_auc']} ap={summary['average_precision']} "
            f"edge_loss={summary['edge_loss']:.6f}",
            flush=True,
        )

    with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"[eval-checkpoint] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
