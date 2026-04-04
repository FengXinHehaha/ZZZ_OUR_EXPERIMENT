import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare absolute and window-local threshold strategies for exact node-level P/R/F1."
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
        help="Window used to select the operating point. Default: val",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric optimized on the selection window. Default: f1",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        action="append",
        default=[],
        help="Candidate local top-ratio thresholds, e.g. 0.001 for top 0.1%%. Can be passed multiple times.",
    )
    parser.add_argument(
        "--zscore-k",
        type=float,
        action="append",
        default=[],
        help="Candidate local mean+K*std thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--robust-k",
        type=float,
        action="append",
        default=[],
        help="Candidate local median+K*1.4826*MAD thresholds. Can be passed multiple times.",
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


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
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
    return float(candidate.get("threshold", 0.0)) > float(best.get("threshold", 0.0))


def compute_confusion_threshold(rows: List[Dict[str, float | int | str]], threshold: float) -> Dict[str, float | int]:
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
    return {
        "threshold": threshold,
        **metrics_from_counts(tp, fp, tn, fn),
    }


def find_best_absolute_threshold(rows: List[Dict[str, float | int | str]], optimize: str) -> Dict[str, float | int]:
    sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
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
        candidate = {
            "threshold": score,
            **metrics_from_counts(tp, fp, tn, fn),
        }
        if choose_better(candidate, best, optimize):
            best = candidate
        index = next_index
    assert best is not None
    return best


def compute_confusion_top_ratio(rows: List[Dict[str, float | int | str]], ratio: float) -> Dict[str, float | int]:
    sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    cutoff = min(len(sorted_rows), max(1, math.ceil(len(sorted_rows) * ratio)))
    tp = sum(int(row["is_gt"]) for row in sorted_rows[:cutoff])
    fp = cutoff - tp
    fn = sum(int(row["is_gt"]) for row in sorted_rows[cutoff:])
    tn = len(sorted_rows) - cutoff - fn
    return {
        "ratio": ratio,
        "threshold": float(sorted_rows[cutoff - 1]["score"]) if cutoff > 0 else math.inf,
        "cutoff": cutoff,
        **metrics_from_counts(tp, fp, tn, fn),
    }


def mean_std_threshold(rows: List[Dict[str, float | int | str]], k: float) -> float:
    scores = [float(row["score"]) for row in rows]
    if not scores:
        return math.inf
    mean_value = sum(scores) / len(scores)
    variance = sum((score - mean_value) ** 2 for score in scores) / len(scores)
    std_value = variance ** 0.5
    return mean_value + k * std_value


def robust_threshold(rows: List[Dict[str, float | int | str]], k: float) -> float:
    scores = [float(row["score"]) for row in rows]
    if not scores:
        return math.inf
    med = median(scores)
    abs_deviation = [abs(score - med) for score in scores]
    mad = median(abs_deviation)
    robust_scale = 1.4826 * mad
    return med + k * robust_scale


def compute_confusion_mean_std(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, float | int]:
    threshold = mean_std_threshold(rows, k)
    return {
        "k": k,
        **compute_confusion_threshold(rows, threshold),
    }


def compute_confusion_robust(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, float | int]:
    threshold = robust_threshold(rows, k)
    return {
        "k": k,
        **compute_confusion_threshold(rows, threshold),
    }


def summarize_window(
    graph_summary: Dict[str, object],
    strategy_name: str,
    rows: List[Dict[str, float | int | str]],
    metrics: Dict[str, float | int],
) -> Dict[str, object]:
    return {
        "name": graph_summary["name"],
        "strategy": strategy_name,
        "num_nodes": len(rows),
        "gt_nodes": int(sum(int(row["is_gt"]) for row in rows)),
        "roc_auc": graph_summary.get("roc_auc"),
        "average_precision": graph_summary.get("average_precision"),
        "edge_loss": graph_summary.get("edge_loss"),
        **metrics,
    }


def print_strategy_summary(strategy_name: str, selected: Dict[str, float | int]) -> None:
    extras = []
    if "ratio" in selected:
        extras.append(f"ratio={selected['ratio']:.6f}")
    if "k" in selected:
        extras.append(f"k={selected['k']:.6f}")
    extras.append(f"threshold={float(selected['threshold']):.12f}")
    print(
        f"[threshold-strategies] {strategy_name}: "
        + " ".join(extras)
        + f" precision={selected['precision']:.6f}"
        + f" recall={selected['recall']:.6f}"
        + f" f1={selected['f1']:.6f}"
        + f" tp={selected['tp']} fp={selected['fp']} fn={selected['fn']}"
        + f" pred_pos={selected['predicted_positive']}"
    )


def print_window_summary(window: Dict[str, object]) -> None:
    extras = []
    if "ratio" in window:
        extras.append(f"ratio={window['ratio']:.6f}")
    if "k" in window:
        extras.append(f"k={window['k']:.6f}")
    extras.append(f"threshold={float(window['threshold']):.12f}")
    print(
        f"  [window] {window['name']} "
        + " ".join(extras)
        + f" roc_auc={window['roc_auc']} ap={window['average_precision']}"
        + f" precision={window['precision']:.6f}"
        + f" recall={window['recall']:.6f}"
        + f" f1={window['f1']:.6f}"
        + f" tp={window['tp']} fp={window['fp']} fn={window['fn']} tn={window['tn']}"
        + f" pred_pos={window['predicted_positive']}"
    )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / "threshold_strategy_comparison.json"

    top_ratios = sorted(set(args.top_ratio or [0.0005, 0.001, 0.005, 0.01]))
    zscore_ks = sorted(set(args.zscore_k or [1.5, 2.0, 2.5, 3.0]))
    robust_ks = sorted(set(args.robust_k or [1.5, 2.0, 2.5, 3.0]))

    summary = load_json(summary_path)
    graphs = summary["graphs"]
    graph_rows = {
        graph["name"]: read_node_scores(Path(graph["node_scores_file"]))
        for graph in graphs
    }

    select_graph = next((graph for graph in graphs if graph["name"] == args.select_window), None)
    if select_graph is None:
        available = [graph["name"] for graph in graphs]
        raise ValueError(f"Selection window '{args.select_window}' not found. Available: {available}")
    select_rows = graph_rows[args.select_window]

    absolute_selected = find_best_absolute_threshold(select_rows, args.optimize)

    top_ratio_selected = None
    for ratio in top_ratios:
        candidate = compute_confusion_top_ratio(select_rows, ratio)
        if choose_better(candidate, top_ratio_selected, args.optimize):
            top_ratio_selected = candidate
    assert top_ratio_selected is not None

    zscore_selected = None
    for k in zscore_ks:
        candidate = compute_confusion_mean_std(select_rows, k)
        if choose_better(candidate, zscore_selected, args.optimize):
            zscore_selected = candidate
    assert zscore_selected is not None

    robust_selected = None
    for k in robust_ks:
        candidate = compute_confusion_robust(select_rows, k)
        if choose_better(candidate, robust_selected, args.optimize):
            robust_selected = candidate
    assert robust_selected is not None

    strategies = {
        "val_best_absolute_threshold": (
            absolute_selected,
            lambda rows: compute_confusion_threshold(rows, float(absolute_selected["threshold"])),
        ),
        "window_top_ratio": (
            top_ratio_selected,
            lambda rows: compute_confusion_top_ratio(rows, float(top_ratio_selected["ratio"])),
        ),
        "window_mean_plus_std": (
            zscore_selected,
            lambda rows: compute_confusion_mean_std(rows, float(zscore_selected["k"])),
        ),
        "window_median_plus_mad": (
            robust_selected,
            lambda rows: compute_confusion_robust(rows, float(robust_selected["k"])),
        ),
    }

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "selection_window": args.select_window,
        "selection_optimize": args.optimize,
        "candidates": {
            "top_ratios": top_ratios,
            "zscore_ks": zscore_ks,
            "robust_ks": robust_ks,
        },
        "strategies": {},
    }

    print(f"[threshold-strategies] eval_dir={eval_dir}")
    print(f"[threshold-strategies] checkpoint={summary['checkpoint']}")
    print(f"[threshold-strategies] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("score_method") is not None:
        print(f"[threshold-strategies] score_method={summary['score_method']}")
    if summary.get("score_calibration") is not None:
        print(f"[threshold-strategies] score_calibration={summary['score_calibration']}")
    print(f"[threshold-strategies] selection_window={args.select_window} optimize={args.optimize}")

    for strategy_name, (selected, evaluator) in strategies.items():
        print_strategy_summary(strategy_name, selected)
        strategy_result = {
            "selected": selected,
            "windows": [],
        }
        for graph in graphs:
            rows = graph_rows[graph["name"]]
            window_metrics = evaluator(rows)
            window_result = summarize_window(graph, strategy_name, rows, window_metrics)
            strategy_result["windows"].append(window_result)
            print_window_summary(window_result)
        results["strategies"][strategy_name] = strategy_result
        print("")

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[threshold-strategies] wrote {output_path}")


if __name__ == "__main__":
    main()
