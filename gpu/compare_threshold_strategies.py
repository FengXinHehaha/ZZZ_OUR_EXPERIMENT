import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Tuple

from file_screening import eligible_rows, eligible_scores, is_screen_kept


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "baseline_eval_60ep"

DEFAULT_TOP_RATIOS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
DEFAULT_TOP_COUNTS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_ZSCORE_KS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEFAULT_ROBUST_KS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare absolute, window-local, and type-wise local threshold strategies for exact node-level P/R/F1."
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
        "--top-ratio-range",
        type=str,
        action="append",
        default=[],
        help="Dense top-ratio grid as start:stop:step, e.g. 0.0001:0.01:0.0001. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-count",
        type=int,
        action="append",
        default=[],
        help="Candidate local top-count thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--zscore-k",
        type=float,
        action="append",
        default=[],
        help="Candidate local mean+K*std thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--zscore-k-range",
        type=str,
        action="append",
        default=[],
        help="Dense local mean+K*std grid as start:stop:step. Can be passed multiple times.",
    )
    parser.add_argument(
        "--robust-k",
        type=float,
        action="append",
        default=[],
        help="Candidate local median+K*1.4826*MAD thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--robust-k-range",
        type=str,
        action="append",
        default=[],
        help="Dense local median+K*1.4826*MAD grid as start:stop:step. Can be passed multiple times.",
    )
    parser.add_argument(
        "--typewise-zscore-k",
        type=float,
        action="append",
        default=[],
        help="Candidate node-type-specific mean+K*std thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--typewise-zscore-k-range",
        type=str,
        action="append",
        default=[],
        help="Dense node-type-specific mean+K*std grid as start:stop:step. Can be passed multiple times.",
    )
    parser.add_argument(
        "--typewise-robust-k",
        type=float,
        action="append",
        default=[],
        help="Candidate node-type-specific median+K*1.4826*MAD thresholds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--typewise-robust-k-range",
        type=str,
        action="append",
        default=[],
        help="Dense node-type-specific median+K*1.4826*MAD grid as start:stop:step. Can be passed multiple times.",
    )
    parser.add_argument(
        "--print-top-candidates",
        type=int,
        default=5,
        help="How many top candidates to print per strategy on the selection window. Default: 5",
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
                    "screen_keep": int(row.get("screen_keep", 1) or 1),
                }
            )
    return rows


def parse_range_spec(spec: str) -> List[float]:
    parts = [item.strip() for item in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid range '{spec}'. Expected start:stop:step.")

    start = float(parts[0])
    stop = float(parts[1])
    step = float(parts[2])
    if step == 0.0:
        raise ValueError(f"Invalid range '{spec}'. Step must be non-zero.")
    if start < stop and step < 0.0:
        raise ValueError(f"Invalid range '{spec}'. Step must be positive for ascending ranges.")
    if start > stop and step > 0.0:
        raise ValueError(f"Invalid range '{spec}'. Step must be negative for descending ranges.")

    values: List[float] = []
    epsilon = abs(step) * 1e-9
    current = start
    if step > 0.0:
        while current <= stop + epsilon:
            values.append(round(current, 12))
            current += step
    else:
        while current >= stop - epsilon:
            values.append(round(current, 12))
            current += step
    return values


def resolve_float_candidates(
    explicit_values: List[float],
    range_specs: List[str],
    default_values: List[float],
) -> List[float]:
    values = list(explicit_values)
    for spec in range_specs:
        values.extend(parse_range_spec(spec))
    source = values if values else list(default_values)
    return sorted({round(value, 12) for value in source})


def resolve_int_candidates(explicit_values: List[int], default_values: List[int]) -> List[int]:
    source = explicit_values if explicit_values else list(default_values)
    return sorted({value for value in source if value > 0})


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


def candidate_threshold_anchor(candidate: Dict[str, object]) -> float:
    threshold_summary = candidate.get("threshold_summary")
    if isinstance(threshold_summary, dict):
        return float(threshold_summary.get("median", 0.0))
    threshold = candidate.get("threshold")
    if threshold is None:
        return 0.0
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return value


def choose_better(
    candidate: Dict[str, object],
    best: Dict[str, object] | None,
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
    return candidate_threshold_anchor(candidate) > candidate_threshold_anchor(best)


def sort_candidates(candidates: List[Dict[str, object]], optimize: str) -> List[Dict[str, object]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            float(candidate[optimize]),
            float(candidate["f1"]),
            float(candidate["precision"]),
            float(candidate["recall"]),
            -int(candidate["predicted_positive"]),
            candidate_threshold_anchor(candidate),
        ),
        reverse=True,
    )


def select_best_candidate(candidates: List[Dict[str, object]], optimize: str) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    for candidate in candidates:
        if choose_better(candidate, best, optimize):
            best = candidate
    assert best is not None
    return best


def compute_confusion_threshold(rows: List[Dict[str, float | int | str]], threshold: float) -> Dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred_positive = is_screen_kept(row) and float(row["score"]) >= threshold
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


def scan_absolute_thresholds(
    rows: List[Dict[str, float | int | str]],
    optimize: str,
    top_n: int,
) -> Tuple[Dict[str, object], int, List[Dict[str, object]]]:
    sorted_rows = sorted(eligible_rows(rows), key=lambda row: float(row["score"]), reverse=True)
    total_positive = sum(int(row["is_gt"]) for row in sorted_rows)
    total_positive = sum(int(row["is_gt"]) for row in rows)
    total_negative = len(rows) - total_positive
    tp = 0
    fp = 0
    best: Dict[str, object] | None = None
    best_candidates: List[Dict[str, object]] = []
    unique_thresholds = 0
    if not sorted_rows:
        candidate = {
            "threshold": math.inf,
            **metrics_from_counts(0, 0, total_negative, total_positive),
        }
        return candidate, 0, [candidate]
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
        unique_thresholds += 1
        best_candidates.append(candidate)
        best_candidates = sort_candidates(best_candidates, optimize)[:top_n]
        index = next_index
    assert best is not None
    return best, unique_thresholds, best_candidates


def compute_confusion_top_ratio(rows: List[Dict[str, float | int | str]], ratio: float) -> Dict[str, object]:
    sorted_rows = sorted(eligible_rows(rows), key=lambda row: float(row["score"]), reverse=True)
    if not sorted_rows:
        total_positive = sum(int(row["is_gt"]) for row in rows)
        total_negative = len(rows) - total_positive
        return {
            "ratio": 0.0,
            "cutoff": 0,
            "threshold": math.inf,
            **metrics_from_counts(0, 0, total_negative, total_positive),
        }
    cutoff = min(len(sorted_rows), max(1, math.ceil(len(sorted_rows) * ratio)))
    selected_ids = {int(row["node_id"]) for row in sorted_rows[:cutoff]}
    tp = fp = tn = fn = 0
    for row in rows:
        pred_positive = int(row["node_id"]) in selected_ids
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
        "ratio": ratio,
        "cutoff": cutoff,
        "threshold": float(sorted_rows[cutoff - 1]["score"]) if cutoff > 0 else math.inf,
        **metrics_from_counts(tp, fp, tn, fn),
    }


def compute_confusion_top_count(rows: List[Dict[str, float | int | str]], count: int) -> Dict[str, object]:
    sorted_rows = sorted(eligible_rows(rows), key=lambda row: float(row["score"]), reverse=True)
    if not sorted_rows:
        total_positive = sum(int(row["is_gt"]) for row in rows)
        total_negative = len(rows) - total_positive
        return {
            "count": 0,
            "ratio": 0.0,
            "threshold": math.inf,
            **metrics_from_counts(0, 0, total_negative, total_positive),
        }
    cutoff = min(len(sorted_rows), max(1, count))
    selected_ids = {int(row["node_id"]) for row in sorted_rows[:cutoff]}
    tp = fp = tn = fn = 0
    for row in rows:
        pred_positive = int(row["node_id"]) in selected_ids
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
        "count": cutoff,
        "ratio": cutoff / len(sorted_rows) if sorted_rows else 0.0,
        "threshold": float(sorted_rows[cutoff - 1]["score"]) if cutoff > 0 else math.inf,
        **metrics_from_counts(tp, fp, tn, fn),
    }


def mean_std_threshold_from_scores(scores: List[float], k: float) -> float:
    if not scores:
        return math.inf
    mean_value = sum(scores) / len(scores)
    variance = sum((score - mean_value) ** 2 for score in scores) / len(scores)
    std_value = variance ** 0.5
    return mean_value + k * std_value


def robust_threshold_from_scores(scores: List[float], k: float) -> float:
    if not scores:
        return math.inf
    med = median(scores)
    abs_deviation = [abs(score - med) for score in scores]
    mad = median(abs_deviation)
    robust_scale = 1.4826 * mad
    return med + k * robust_scale


def mean_std_threshold(rows: List[Dict[str, float | int | str]], k: float) -> float:
    return mean_std_threshold_from_scores(eligible_scores(rows), k)


def robust_threshold(rows: List[Dict[str, float | int | str]], k: float) -> float:
    return robust_threshold_from_scores(eligible_scores(rows), k)


def summarize_typewise_thresholds(thresholds: Dict[str, float]) -> Dict[str, object]:
    ordered_items = sorted(thresholds.items())
    values = [value for _, value in ordered_items]
    summary = {
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "num_types": len(values),
    }
    return {
        "thresholds_by_type": {node_type: threshold for node_type, threshold in ordered_items},
        "threshold_summary": summary,
    }


def typewise_thresholds(
    rows: List[Dict[str, float | int | str]],
    threshold_builder: Callable[[List[float]], float],
) -> Dict[str, float]:
    scores_by_type: Dict[str, List[float]] = {}
    for row in rows:
        if not is_screen_kept(row):
            continue
        node_type = str(row["node_type"])
        scores_by_type.setdefault(node_type, []).append(float(row["score"]))
    return {
        node_type: threshold_builder(scores)
        for node_type, scores in sorted(scores_by_type.items())
    }


def compute_confusion_typewise_thresholds(
    rows: List[Dict[str, float | int | str]],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    tp = fp = tn = fn = 0
    for row in rows:
        node_type = str(row["node_type"])
        threshold = thresholds.get(node_type, math.inf)
        pred_positive = is_screen_kept(row) and float(row["score"]) >= threshold
        is_positive = int(row["is_gt"]) == 1
        if pred_positive and is_positive:
            tp += 1
        elif pred_positive and not is_positive:
            fp += 1
        elif (not pred_positive) and is_positive:
            fn += 1
        else:
            tn += 1
    threshold_details = summarize_typewise_thresholds(thresholds)
    return {
        "threshold": float(threshold_details["threshold_summary"]["median"]),
        **threshold_details,
        **metrics_from_counts(tp, fp, tn, fn),
    }


def compute_confusion_mean_std(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, object]:
    threshold = mean_std_threshold(rows, k)
    return {
        "k": k,
        **compute_confusion_threshold(rows, threshold),
    }


def compute_confusion_robust(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, object]:
    threshold = robust_threshold(rows, k)
    return {
        "k": k,
        **compute_confusion_threshold(rows, threshold),
    }


def compute_confusion_typewise_mean_std(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, object]:
    thresholds = typewise_thresholds(rows, lambda scores: mean_std_threshold_from_scores(scores, k))
    return {
        "k": k,
        **compute_confusion_typewise_thresholds(rows, thresholds),
    }


def compute_confusion_typewise_robust(rows: List[Dict[str, float | int | str]], k: float) -> Dict[str, object]:
    thresholds = typewise_thresholds(rows, lambda scores: robust_threshold_from_scores(scores, k))
    return {
        "k": k,
        **compute_confusion_typewise_thresholds(rows, thresholds),
    }


def summarize_window(
    graph_summary: Dict[str, object],
    strategy_name: str,
    rows: List[Dict[str, float | int | str]],
    metrics: Dict[str, object],
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


def format_threshold_bits(metrics: Dict[str, object]) -> str:
    extras: List[str] = []
    if "count" in metrics:
        extras.append(f"count={int(metrics['count'])}")
    if "ratio" in metrics:
        extras.append(f"ratio={float(metrics['ratio']):.6f}")
    if "k" in metrics:
        extras.append(f"k={float(metrics['k']):.6f}")
    threshold_summary = metrics.get("threshold_summary")
    if isinstance(threshold_summary, dict):
        extras.append(f"threshold_median={float(threshold_summary['median']):.12f}")
        extras.append(f"threshold_min={float(threshold_summary['min']):.12f}")
        extras.append(f"threshold_max={float(threshold_summary['max']):.12f}")
    elif "threshold" in metrics:
        extras.append(f"threshold={float(metrics['threshold']):.12f}")
    return " ".join(extras)


def print_strategy_summary(strategy_name: str, selected: Dict[str, object]) -> None:
    print(
        f"[threshold-strategies] {strategy_name}: "
        + format_threshold_bits(selected)
        + f" precision={float(selected['precision']):.6f}"
        + f" recall={float(selected['recall']):.6f}"
        + f" f1={float(selected['f1']):.6f}"
        + f" tp={int(selected['tp'])} fp={int(selected['fp'])} fn={int(selected['fn'])}"
        + f" pred_pos={int(selected['predicted_positive'])}"
    )


def print_selection_candidates(
    strategy_name: str,
    candidates: List[Dict[str, object]],
    optimize: str,
    limit: int,
) -> None:
    if limit <= 0 or not candidates:
        return
    print(f"  [selection-top-candidates] {strategy_name} optimize={optimize}")
    for index, candidate in enumerate(sort_candidates(candidates, optimize)[:limit], start=1):
        print(
            f"    {index}. "
            + format_threshold_bits(candidate)
            + f" precision={float(candidate['precision']):.6f}"
            + f" recall={float(candidate['recall']):.6f}"
            + f" f1={float(candidate['f1']):.6f}"
            + f" tp={int(candidate['tp'])} fp={int(candidate['fp'])} fn={int(candidate['fn'])}"
            + f" pred_pos={int(candidate['predicted_positive'])}"
        )


def print_window_summary(window: Dict[str, object]) -> None:
    print(
        f"  [window] {window['name']} "
        + format_threshold_bits(window)
        + f" roc_auc={window['roc_auc']} ap={window['average_precision']}"
        + f" precision={float(window['precision']):.6f}"
        + f" recall={float(window['recall']):.6f}"
        + f" f1={float(window['f1']):.6f}"
        + f" tp={int(window['tp'])} fp={int(window['fp'])} fn={int(window['fn'])} tn={int(window['tn'])}"
        + f" pred_pos={int(window['predicted_positive'])}"
    )


def build_window_leaderboards(
    strategies: Dict[str, Dict[str, object]],
    optimize: str,
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for strategy_name, strategy_result in strategies.items():
        selected = strategy_result["selected"]
        for window in strategy_result["windows"]:
            grouped.setdefault(window["name"], []).append(
                {
                    "strategy": strategy_name,
                    "selection_params": {
                        key: selected[key]
                        for key in (
                            "threshold",
                            "count",
                            "ratio",
                            "k",
                            "threshold_summary",
                            "thresholds_by_type",
                        )
                        if key in selected
                    },
                    "window_params": {
                        key: window[key]
                        for key in (
                            "threshold",
                            "count",
                            "ratio",
                            "k",
                            "threshold_summary",
                            "thresholds_by_type",
                        )
                        if key in window
                    },
                    "precision": window["precision"],
                    "recall": window["recall"],
                    "f1": window["f1"],
                    "tp": window["tp"],
                    "fp": window["fp"],
                    "fn": window["fn"],
                    "predicted_positive": window["predicted_positive"],
                    "roc_auc": window["roc_auc"],
                    "average_precision": window["average_precision"],
                }
            )
    return {
        window_name: sort_candidates(entries, optimize)
        for window_name, entries in grouped.items()
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / "threshold_strategy_comparison.json"

    top_ratios = resolve_float_candidates(args.top_ratio, args.top_ratio_range, DEFAULT_TOP_RATIOS)
    top_counts = resolve_int_candidates(args.top_count, DEFAULT_TOP_COUNTS)
    zscore_ks = resolve_float_candidates(args.zscore_k, args.zscore_k_range, DEFAULT_ZSCORE_KS)
    robust_ks = resolve_float_candidates(args.robust_k, args.robust_k_range, DEFAULT_ROBUST_KS)
    typewise_zscore_ks = resolve_float_candidates(
        args.typewise_zscore_k,
        args.typewise_zscore_k_range,
        zscore_ks,
    )
    typewise_robust_ks = resolve_float_candidates(
        args.typewise_robust_k,
        args.typewise_robust_k_range,
        robust_ks,
    )

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

    absolute_selected, absolute_candidate_count, absolute_top_candidates = scan_absolute_thresholds(
        select_rows,
        args.optimize,
        max(args.print_top_candidates, 1),
    )

    top_count_candidates = [compute_confusion_top_count(select_rows, count) for count in top_counts]
    top_ratio_candidates = [compute_confusion_top_ratio(select_rows, ratio) for ratio in top_ratios]
    zscore_candidates = [compute_confusion_mean_std(select_rows, k) for k in zscore_ks]
    robust_candidates = [compute_confusion_robust(select_rows, k) for k in robust_ks]
    typewise_zscore_candidates = [compute_confusion_typewise_mean_std(select_rows, k) for k in typewise_zscore_ks]
    typewise_robust_candidates = [compute_confusion_typewise_robust(select_rows, k) for k in typewise_robust_ks]

    strategies: Dict[str, Dict[str, object]] = {
        "val_best_absolute_threshold": {
            "selected": absolute_selected,
            "selection_candidates": absolute_top_candidates,
            "selection_candidate_count": absolute_candidate_count,
            "evaluator": lambda rows: compute_confusion_threshold(rows, float(absolute_selected["threshold"])),
        },
        "window_top_count": {
            "selected": select_best_candidate(top_count_candidates, args.optimize),
            "selection_candidates": sort_candidates(top_count_candidates, args.optimize),
            "selection_candidate_count": len(top_count_candidates),
            "evaluator": None,
        },
        "window_top_ratio": {
            "selected": select_best_candidate(top_ratio_candidates, args.optimize),
            "selection_candidates": sort_candidates(top_ratio_candidates, args.optimize),
            "selection_candidate_count": len(top_ratio_candidates),
            "evaluator": None,
        },
        "window_mean_plus_std": {
            "selected": select_best_candidate(zscore_candidates, args.optimize),
            "selection_candidates": sort_candidates(zscore_candidates, args.optimize),
            "selection_candidate_count": len(zscore_candidates),
            "evaluator": None,
        },
        "window_median_plus_mad": {
            "selected": select_best_candidate(robust_candidates, args.optimize),
            "selection_candidates": sort_candidates(robust_candidates, args.optimize),
            "selection_candidate_count": len(robust_candidates),
            "evaluator": None,
        },
        "window_typewise_mean_plus_std": {
            "selected": select_best_candidate(typewise_zscore_candidates, args.optimize),
            "selection_candidates": sort_candidates(typewise_zscore_candidates, args.optimize),
            "selection_candidate_count": len(typewise_zscore_candidates),
            "evaluator": None,
        },
        "window_typewise_median_plus_mad": {
            "selected": select_best_candidate(typewise_robust_candidates, args.optimize),
            "selection_candidates": sort_candidates(typewise_robust_candidates, args.optimize),
            "selection_candidate_count": len(typewise_robust_candidates),
            "evaluator": None,
        },
    }

    strategies["window_top_count"]["evaluator"] = lambda rows: compute_confusion_top_count(
        rows,
        int(strategies["window_top_count"]["selected"]["count"]),
    )
    strategies["window_top_ratio"]["evaluator"] = lambda rows: compute_confusion_top_ratio(
        rows,
        float(strategies["window_top_ratio"]["selected"]["ratio"]),
    )
    strategies["window_mean_plus_std"]["evaluator"] = lambda rows: compute_confusion_mean_std(
        rows,
        float(strategies["window_mean_plus_std"]["selected"]["k"]),
    )
    strategies["window_median_plus_mad"]["evaluator"] = lambda rows: compute_confusion_robust(
        rows,
        float(strategies["window_median_plus_mad"]["selected"]["k"]),
    )
    strategies["window_typewise_mean_plus_std"]["evaluator"] = lambda rows: compute_confusion_typewise_mean_std(
        rows,
        float(strategies["window_typewise_mean_plus_std"]["selected"]["k"]),
    )
    strategies["window_typewise_median_plus_mad"]["evaluator"] = lambda rows: compute_confusion_typewise_robust(
        rows,
        float(strategies["window_typewise_median_plus_mad"]["selected"]["k"]),
    )

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "file_screen_policy": summary.get("file_screen_policy"),
        "post_rerank_method": summary.get("post_rerank_method"),
        "post_rerank_candidate_rank_max": summary.get("post_rerank_candidate_rank_max"),
        "selection_window": args.select_window,
        "selection_optimize": args.optimize,
        "candidates": {
            "top_counts": top_counts,
            "top_ratios": top_ratios,
            "zscore_ks": zscore_ks,
            "robust_ks": robust_ks,
            "typewise_zscore_ks": typewise_zscore_ks,
            "typewise_robust_ks": typewise_robust_ks,
        },
        "strategies": {},
        "window_leaderboards": {},
    }

    print(f"[threshold-strategies] eval_dir={eval_dir}")
    print(f"[threshold-strategies] checkpoint={summary['checkpoint']}")
    print(f"[threshold-strategies] checkpoint_epoch={summary['checkpoint_epoch']}")
    if summary.get("score_method") is not None:
        print(f"[threshold-strategies] score_method={summary['score_method']}")
    if summary.get("score_calibration") is not None:
        print(f"[threshold-strategies] score_calibration={summary['score_calibration']}")
    if summary.get("file_screen_policy") is not None:
        print(f"[threshold-strategies] file_screen_policy={summary['file_screen_policy']}")
    if summary.get("post_rerank_method") is not None:
        print(
            f"[threshold-strategies] post_rerank_method={summary['post_rerank_method']} "
            f"candidate_rank_max={summary.get('post_rerank_candidate_rank_max')}"
        )
    print(f"[threshold-strategies] selection_window={args.select_window} optimize={args.optimize}")
    print(
        "[threshold-strategies] candidate-grid "
        f"top_counts={top_counts} "
        f"top_ratios={top_ratios} "
        f"zscore_ks={zscore_ks} "
        f"robust_ks={robust_ks} "
        f"typewise_zscore_ks={typewise_zscore_ks} "
        f"typewise_robust_ks={typewise_robust_ks}"
    )

    for strategy_name, strategy_state in strategies.items():
        selected = strategy_state["selected"]
        evaluator = strategy_state["evaluator"]
        assert callable(evaluator)

        print_strategy_summary(strategy_name, selected)
        print_selection_candidates(
            strategy_name,
            strategy_state["selection_candidates"],
            args.optimize,
            args.print_top_candidates,
        )

        strategy_result = {
            "selected": selected,
            "selection_candidates": strategy_state["selection_candidates"],
            "selection_candidate_count": strategy_state["selection_candidate_count"],
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

    results["window_leaderboards"] = build_window_leaderboards(results["strategies"], args.optimize)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[threshold-strategies] wrote {output_path}")


if __name__ == "__main__":
    main()
