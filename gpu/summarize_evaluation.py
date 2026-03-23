import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact summary of evaluation outputs, including top-k hit counts."
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
        help="Optional top-k values to inspect from node_scores.tsv. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_node_scores(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def summarize_topk(rows: List[Dict[str, str]], k: int, gt_total: int) -> Dict[str, object]:
    capped = rows[: min(k, len(rows))]
    hits = sum(int(row["is_gt"]) for row in capped)

    by_type: Dict[str, int] = {}
    gt_by_type: Dict[str, int] = {}
    for row in capped:
        node_type = row["node_type"]
        by_type[node_type] = by_type.get(node_type, 0) + 1
        if int(row["is_gt"]) == 1:
            gt_by_type[node_type] = gt_by_type.get(node_type, 0) + 1

    return {
        "k": len(capped),
        "gt_hits": hits,
        "precision_at_k": hits / len(capped) if capped else 0.0,
        "recall_at_k": hits / gt_total if gt_total else 0.0,
        "node_type_counts": by_type,
        "gt_type_hits": gt_by_type,
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"

    summary = load_json(summary_path)
    graphs = summary["graphs"]
    topk_values = args.topk or [100, 500, 1000, 5000]

    print(f"[evaluation-summary] eval_dir={eval_dir}")
    print(f"[evaluation-summary] checkpoint={summary['checkpoint']}")
    print(f"[evaluation-summary] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("best_record") is not None:
        best_record = summary["best_record"]
        print(
            "[evaluation-summary] best_record="
            f"epoch={best_record['epoch']} "
            f"{best_record['selection_metric']}={best_record['selection_score']}"
        )

    for graph in graphs:
        print("")
        print(
            f"[window] {graph['name']} "
            f"roc_auc={graph['roc_auc']} "
            f"ap={graph['average_precision']} "
            f"edge_loss={graph['edge_loss']:.6f} "
            f"gt_nodes={graph['gt_nodes']}"
        )

        for topk_entry in graph.get("topk", []):
            print(
                f"  [saved-topk] k={topk_entry['k']} "
                f"hits={topk_entry['hits']} "
                f"precision={topk_entry['precision_at_k']:.6f} "
                f"recall={topk_entry['recall_at_k']:.6f}"
            )

        node_scores_path = Path(graph["node_scores_file"])
        rows = read_node_scores(node_scores_path)
        for k in topk_values:
            topk_summary = summarize_topk(rows, k, int(graph["gt_nodes"]))
            print(
                f"  [topk] k={topk_summary['k']} "
                f"hits={topk_summary['gt_hits']} "
                f"precision={topk_summary['precision_at_k']:.6f} "
                f"recall={topk_summary['recall_at_k']:.6f} "
                f"node_types={topk_summary['node_type_counts']} "
                f"gt_types={topk_summary['gt_type_hits']}"
            )


if __name__ == "__main__":
    main()
