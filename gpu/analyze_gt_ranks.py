import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze where GT nodes land in the ranked anomaly list."
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
        "--top-ratio",
        type=float,
        action="append",
        default=[],
        help="Top-ratio cutoffs to analyze, e.g. 0.01 for top 1%%. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_node_scores(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def topk_summary(ranks: Iterable[int], gt_total: int, k: int) -> Dict[str, float]:
    hits = sum(1 for rank in ranks if rank <= k)
    return {
        "k": k,
        "hits": hits,
        "precision_at_k": hits / k if k > 0 else 0.0,
        "recall_at_k": hits / gt_total if gt_total else 0.0,
    }


def top_ratio_summary(ranks: Iterable[int], gt_total: int, total_nodes: int, ratio: float) -> Dict[str, float]:
    cutoff = max(1, int(total_nodes * ratio))
    hits = sum(1 for rank in ranks if rank <= cutoff)
    return {
        "ratio": ratio,
        "cutoff_rank": cutoff,
        "hits": hits,
        "precision_at_cutoff": hits / cutoff if cutoff > 0 else 0.0,
        "recall_at_cutoff": hits / gt_total if gt_total else 0.0,
    }


def summarize_ranks(ranks: List[int], total_nodes: int) -> Dict[str, float]:
    ranks = sorted(ranks)
    if not ranks:
        return {
            "best_rank": 0,
            "worst_rank": 0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "best_rank_ratio": 0.0,
            "median_rank_ratio": 0.0,
            "p10_rank": 0.0,
            "p25_rank": 0.0,
            "p50_rank": 0.0,
            "p75_rank": 0.0,
            "p90_rank": 0.0,
        }

    mean_rank = sum(ranks) / len(ranks)
    median_rank = percentile(ranks, 0.50)
    best_rank = ranks[0]
    return {
        "best_rank": best_rank,
        "worst_rank": ranks[-1],
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "best_rank_ratio": best_rank / total_nodes if total_nodes else 0.0,
        "median_rank_ratio": median_rank / total_nodes if total_nodes else 0.0,
        "p10_rank": percentile(ranks, 0.10),
        "p25_rank": percentile(ranks, 0.25),
        "p50_rank": median_rank,
        "p75_rank": percentile(ranks, 0.75),
        "p90_rank": percentile(ranks, 0.90),
    }


def summarize_window(
    graph_summary: Dict[str, object],
    rows: List[Dict[str, str]],
    topk_values: List[int],
    top_ratio_values: List[float],
) -> Dict[str, object]:
    gt_rows = [row for row in rows if int(row["is_gt"]) == 1]
    total_nodes = len(rows)
    gt_total = len(gt_rows)
    gt_ranks = sorted(int(row["rank"]) for row in gt_rows)

    by_type: Dict[str, List[int]] = {}
    for row in gt_rows:
        by_type.setdefault(row["node_type"], []).append(int(row["rank"]))

    type_summaries: Dict[str, object] = {}
    for node_type, ranks in sorted(by_type.items()):
        ranks = sorted(ranks)
        type_summaries[node_type] = {
            "count": len(ranks),
            **summarize_ranks(ranks, total_nodes),
            "topk": [topk_summary(ranks, len(ranks), k) for k in topk_values],
            "top_ratio": [
                top_ratio_summary(ranks, len(ranks), total_nodes, ratio)
                for ratio in top_ratio_values
            ],
        }

    return {
        "name": graph_summary["name"],
        "num_nodes": total_nodes,
        "gt_nodes": gt_total,
        "roc_auc": graph_summary.get("roc_auc"),
        "average_precision": graph_summary.get("average_precision"),
        "edge_loss": graph_summary.get("edge_loss"),
        "rank_summary": summarize_ranks(gt_ranks, total_nodes),
        "topk": [topk_summary(gt_ranks, gt_total, k) for k in topk_values],
        "top_ratio": [
            top_ratio_summary(gt_ranks, gt_total, total_nodes, ratio)
            for ratio in top_ratio_values
        ],
        "by_type": type_summaries,
    }


def print_window_summary(window: Dict[str, object]) -> None:
    rank_summary = window["rank_summary"]
    print("")
    print(
        f"[window] {window['name']} "
        f"gt_nodes={window['gt_nodes']} total_nodes={window['num_nodes']} "
        f"roc_auc={window['roc_auc']} ap={window['average_precision']} edge_loss={window['edge_loss']:.6f}"
    )
    print(
        "  [gt-ranks] "
        f"best={rank_summary['best_rank']} "
        f"median={rank_summary['median_rank']:.1f} "
        f"mean={rank_summary['mean_rank']:.1f} "
        f"p10={rank_summary['p10_rank']:.1f} "
        f"p25={rank_summary['p25_rank']:.1f} "
        f"p75={rank_summary['p75_rank']:.1f} "
        f"p90={rank_summary['p90_rank']:.1f} "
        f"best_ratio={rank_summary['best_rank_ratio']:.6f} "
        f"median_ratio={rank_summary['median_rank_ratio']:.6f}"
    )
    for entry in window["top_ratio"]:
        print(
            "  [top-ratio] "
            f"ratio={entry['ratio']:.4f} "
            f"cutoff_rank={entry['cutoff_rank']} "
            f"hits={entry['hits']} "
            f"precision={entry['precision_at_cutoff']:.6f} "
            f"recall={entry['recall_at_cutoff']:.6f}"
        )
    for entry in window["topk"]:
        print(
            "  [topk] "
            f"k={entry['k']} "
            f"hits={entry['hits']} "
            f"precision={entry['precision_at_k']:.6f} "
            f"recall={entry['recall_at_k']:.6f}"
        )
    for node_type, summary in window["by_type"].items():
        print(
            "  [gt-type] "
            f"type={node_type} "
            f"count={summary['count']} "
            f"best={summary['best_rank']} "
            f"median={summary['median_rank']:.1f} "
            f"best_ratio={summary['best_rank_ratio']:.6f}"
        )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / "gt_rank_analysis.json"

    topk_values = sorted(set(args.topk or [100, 500, 1000, 5000, 10000]))
    top_ratio_values = sorted(set(args.top_ratio or [0.01, 0.05, 0.10]))

    summary = load_json(summary_path)
    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "topk_values": topk_values,
        "top_ratio_values": top_ratio_values,
        "windows": [],
    }

    print(f"[gt-rank-analysis] eval_dir={eval_dir}")
    print(f"[gt-rank-analysis] checkpoint={summary['checkpoint']}")
    print(f"[gt-rank-analysis] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("score_method") is not None:
        print(f"[gt-rank-analysis] score_method={summary['score_method']}")
    if summary.get("score_calibration") is not None:
        print(f"[gt-rank-analysis] score_calibration={summary['score_calibration']}")

    for graph_summary in summary["graphs"]:
        rows = read_node_scores(Path(graph_summary["node_scores_file"]))
        window_result = summarize_window(graph_summary, rows, topk_values, top_ratio_values)
        results["windows"].append(window_result)
        print_window_summary(window_result)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("")
    print(f"[gt-rank-analysis] wrote {output_path}")


if __name__ == "__main__":
    main()
