import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Set

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exact-hit and hop-based coverage for ranked anomaly outputs."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Top-k cutoffs to analyze. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hop-radius",
        type=int,
        action="append",
        default=[],
        help="Hop radius for neighborhood-aware coverage, e.g. 1 or 2. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_node_scores(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_undirected_neighbors(edge_index: torch.Tensor, num_nodes: int) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(num_nodes)]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    for src, dst in zip(src_list, dst_list):
        if src == dst:
            neighbors[src].add(dst)
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def multi_source_within_radius(
    seeds: Iterable[int],
    neighbors: List[Set[int]],
    radius: int,
) -> Set[int]:
    visited: Set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    for seed in seeds:
        if seed in visited:
            continue
        visited.add(seed)
        queue.append((seed, 0))

    while queue:
        node_id, depth = queue.popleft()
        if depth >= radius:
            continue
        for neighbor_id in neighbors[node_id]:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, depth + 1))
    return visited


def summarize_exact_topk(rows: List[Dict[str, str]], gt_total: int, k: int) -> Dict[str, float]:
    capped = rows[: min(k, len(rows))]
    hits = sum(int(row["is_gt"]) for row in capped)
    return {
        "k": len(capped),
        "hits": hits,
        "precision_at_k": hits / len(capped) if capped else 0.0,
        "recall_at_k": hits / gt_total if gt_total else 0.0,
    }


def summarize_hop_topk(
    rows: List[Dict[str, str]],
    gt_node_ids: Set[int],
    gt_total: int,
    neighbors: List[Set[int]],
    k: int,
    radius: int,
) -> Dict[str, float]:
    capped = rows[: min(k, len(rows))]
    topk_node_ids = [int(row["node_id"]) for row in capped]

    near_gt_nodes = multi_source_within_radius(gt_node_ids, neighbors, radius)
    near_topk_nodes = multi_source_within_radius(topk_node_ids, neighbors, radius)

    alert_hits = sum(1 for node_id in topk_node_ids if node_id in near_gt_nodes)
    gt_hits = sum(1 for node_id in gt_node_ids if node_id in near_topk_nodes)

    return {
        "radius": radius,
        "k": len(capped),
        "alert_hits": alert_hits,
        "alert_precision_at_k": alert_hits / len(capped) if capped else 0.0,
        "gt_hits": gt_hits,
        "gt_recall_at_k": gt_hits / gt_total if gt_total else 0.0,
    }


def summarize_window(
    graph_summary: Dict[str, object],
    rows: List[Dict[str, str]],
    topk_values: List[int],
    hop_radii: List[int],
) -> Dict[str, object]:
    graph_path = Path(graph_summary["path"]).expanduser().resolve()
    graph = torch.load(graph_path, map_location="cpu")
    num_nodes = int(graph["num_nodes"])
    neighbors = build_undirected_neighbors(graph["edge_index"], num_nodes)

    gt_rows = [row for row in rows if int(row["is_gt"]) == 1]
    gt_node_ids = {int(row["node_id"]) for row in gt_rows}
    gt_total = len(gt_node_ids)

    hop_topk = {
        str(radius): [
            summarize_hop_topk(rows, gt_node_ids, gt_total, neighbors, k, radius)
            for k in topk_values
        ]
        for radius in hop_radii
    }

    return {
        "name": graph_summary["name"],
        "graph_path": str(graph_path),
        "num_nodes": len(rows),
        "gt_nodes": gt_total,
        "roc_auc": graph_summary.get("roc_auc"),
        "average_precision": graph_summary.get("average_precision"),
        "edge_loss": graph_summary.get("edge_loss"),
        "exact_topk": [summarize_exact_topk(rows, gt_total, k) for k in topk_values],
        "hop_topk": hop_topk,
    }


def print_window_summary(window: Dict[str, object]) -> None:
    print("")
    print(
        f"[window] {window['name']} "
        f"gt_nodes={window['gt_nodes']} total_nodes={window['num_nodes']} "
        f"roc_auc={window['roc_auc']} ap={window['average_precision']} edge_loss={window['edge_loss']:.6f}"
    )
    for entry in window["exact_topk"]:
        print(
            "  [exact-topk] "
            f"k={entry['k']} "
            f"hits={entry['hits']} "
            f"precision={entry['precision_at_k']:.6f} "
            f"recall={entry['recall_at_k']:.6f}"
        )
    for radius, entries in window["hop_topk"].items():
        for entry in entries:
            print(
                "  [hop-topk] "
                f"hop={radius} "
                f"k={entry['k']} "
                f"alert_hits={entry['alert_hits']} "
                f"alert_precision={entry['alert_precision_at_k']:.6f} "
                f"gt_hits={entry['gt_hits']} "
                f"gt_recall={entry['gt_recall_at_k']:.6f}"
            )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / "hop_hit_analysis.json"

    topk_values = sorted(set(args.topk or [100, 500, 1000, 5000, 10000]))
    hop_radii = sorted(set(args.hop_radius or [1, 2]))

    summary = load_json(summary_path)
    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "topk_values": topk_values,
        "hop_radii": hop_radii,
        "windows": [],
    }

    print(f"[hop-hit-analysis] eval_dir={eval_dir}")
    print(f"[hop-hit-analysis] checkpoint={summary['checkpoint']}")
    print(f"[hop-hit-analysis] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("score_method") is not None:
        print(f"[hop-hit-analysis] score_method={summary['score_method']}")
    if summary.get("score_calibration") is not None:
        print(f"[hop-hit-analysis] score_calibration={summary['score_calibration']}")

    for graph_summary in summary["graphs"]:
        rows = read_node_scores(Path(graph_summary["node_scores_file"]))
        window_result = summarize_window(graph_summary, rows, topk_values, hop_radii)
        results["windows"].append(window_result)
        print_window_summary(window_result)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("")
    print(f"[hop-hit-analysis] wrote {output_path}")


if __name__ == "__main__":
    main()
