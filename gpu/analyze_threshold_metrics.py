import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a node-score threshold on one window and report exact node-level P/R/F1 on all windows."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--select-window",
        type=str,
        default="val",
        help="Window used to select the threshold. Default: val",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric optimized on the selection window. Default: f1",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_node_scores(path: Path) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "node_id": int(row["node_id"]),
                    "node_uuid": row["node_uuid"],
                    "node_type": row["node_type"],
                    "is_gt": int(row["is_gt"]),
                    "score": float(row["score"]),
                }
            )
    return rows


def compute_confusion(rows: List[Dict[str, float | int | str]], threshold: float) -> Dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred_positive = float(row["score"]) >= threshold
        is_positive = int(row["is_gt"]) == 1
        if pred_positive and is_positive:
            tp += 1
        elif pred_positive and not is_positive:
            fp += 1
        elif (not pred_positive) and is_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_positive": tp + fp,
        "predicted_negative": tn + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def candidate_metric(metrics: Dict[str, float | int], optimize: str) -> float:
    return float(metrics[optimize])


def choose_better(
    candidate: Dict[str, float | int],
    best: Dict[str, float | int] | None,
    optimize: str,
) -> bool:
    if best is None:
        return True

    candidate_primary = candidate_metric(candidate, optimize)
    best_primary = candidate_metric(best, optimize)
    if candidate_primary != best_primary:
        return candidate_primary > best_primary

    if float(candidate["f1"]) != float(best["f1"]):
        return float(candidate["f1"]) > float(best["f1"])
    if float(candidate["precision"]) != float(best["precision"]):
        return float(candidate["precision"]) > float(best["precision"])
    if float(candidate["recall"]) != float(best["recall"]):
        return float(candidate["recall"]) > float(best["recall"])
    if int(candidate["predicted_positive"]) != int(best["predicted_positive"]):
        return int(candidate["predicted_positive"]) < int(best["predicted_positive"])
    return float(candidate["threshold"]) > float(best["threshold"])


def find_best_threshold(rows: List[Dict[str, float | int | str]], optimize: str) -> Dict[str, float | int]:
    sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    if not sorted_rows:
        raise ValueError("No node scores found.")

    total_positive = sum(int(row["is_gt"]) for row in sorted_rows)
    total_negative = len(sorted_rows) - total_positive
    tp = 0
    fp = 0
    best: Dict[str, float | int] | None = None
    index = 0
    while index < len(sorted_rows):
        score = float(sorted_rows[index]["score"])
        next_index = index
        group_tp = 0
        group_fp = 0
        while next_index < len(sorted_rows) and float(sorted_rows[next_index]["score"]) == score:
            if int(sorted_rows[next_index]["is_gt"]) == 1:
                group_tp += 1
            else:
                group_fp += 1
            next_index += 1
        tp += group_tp
        fp += group_fp
        fn = total_positive - tp
        tn = total_negative - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        candidate = {
            "threshold": score,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "predicted_positive": tp + fp,
            "predicted_negative": tn + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if choose_better(candidate, best, optimize):
            best = candidate
        index = next_index

    assert best is not None
    return best


def summarize_window(
    graph_summary: Dict[str, object],
    rows: List[Dict[str, float | int | str]],
    threshold: float,
) -> Dict[str, object]:
    confusion = compute_confusion(rows, threshold)
    return {
        "name": graph_summary["name"],
        "threshold": threshold,
        "num_nodes": len(rows),
        "gt_nodes": int(sum(int(row["is_gt"]) for row in rows)),
        "roc_auc": graph_summary.get("roc_auc"),
        "average_precision": graph_summary.get("average_precision"),
        "edge_loss": graph_summary.get("edge_loss"),
        **confusion,
    }


def print_window_summary(window: Dict[str, object]) -> None:
    print(
        f"[window] {window['name']} "
        f"threshold={window['threshold']:.12f} "
        f"roc_auc={window['roc_auc']} ap={window['average_precision']} "
        f"precision={window['precision']:.6f} "
        f"recall={window['recall']:.6f} "
        f"f1={window['f1']:.6f} "
        f"tp={window['tp']} fp={window['fp']} fn={window['fn']} tn={window['tn']} "
        f"pred_pos={window['predicted_positive']}"
    )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / "threshold_metrics.json"

    summary = load_json(summary_path)
    graphs = summary["graphs"]

    select_graph = next((graph for graph in graphs if graph["name"] == args.select_window), None)
    if select_graph is None:
        available = [graph["name"] for graph in graphs]
        raise ValueError(f"Selection window '{args.select_window}' not found. Available: {available}")

    select_rows = read_node_scores(Path(select_graph["node_scores_file"]))
    selected_threshold = find_best_threshold(select_rows, args.optimize)
    threshold = float(selected_threshold["threshold"])

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "selection_window": args.select_window,
        "selection_optimize": args.optimize,
        "selection_result": selected_threshold,
        "windows": [],
    }

    print(f"[threshold-metrics] eval_dir={eval_dir}")
    print(f"[threshold-metrics] checkpoint={summary['checkpoint']}")
    print(f"[threshold-metrics] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("score_method") is not None:
        print(f"[threshold-metrics] score_method={summary['score_method']}")
    if summary.get("score_calibration") is not None:
        print(f"[threshold-metrics] score_calibration={summary['score_calibration']}")
    print(
        f"[threshold-metrics] selected threshold on {args.select_window}: "
        f"threshold={threshold:.12f} "
        f"precision={selected_threshold['precision']:.6f} "
        f"recall={selected_threshold['recall']:.6f} "
        f"f1={selected_threshold['f1']:.6f} "
        f"tp={selected_threshold['tp']} fp={selected_threshold['fp']} fn={selected_threshold['fn']} "
        f"pred_pos={selected_threshold['predicted_positive']}"
    )

    for graph in graphs:
        rows = read_node_scores(Path(graph["node_scores_file"]))
        window_result = summarize_window(graph, rows, threshold)
        results["windows"].append(window_result)
        print_window_summary(window_result)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[threshold-metrics] wrote {output_path}")


if __name__ == "__main__":
    main()
