import argparse
import json
from pathlib import Path
from typing import Dict, List

from compare_adaptive_threshold_policies import classify_regime
from compare_threshold_strategies import (
    compute_confusion_top_count,
    compute_confusion_robust,
    load_json,
    print_window_summary,
    read_node_scores,
    summarize_window,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sparse-window threshold policies that reserve a larger part of the alarm budget for file nodes, "
            "while leaving moderate and dense windows unchanged."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--sparse-score-threshold",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--sparse-max-count",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--dense-score-threshold",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--dense-min-count",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--moderate-top-count",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--dense-robust-k",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--sparse-total-count",
        type=int,
        default=300,
        help="Total sparse alarm budget before splitting into file/non-file. Default: 300",
    )
    parser.add_argument(
        "--sparse-file-fraction",
        type=float,
        action="append",
        default=[],
        help="Fractions of sparse budget allocated to file nodes. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="file_budget_policy_comparison",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_top_count_by_type(
    rows: List[Dict[str, float | int | str]],
    file_count: int,
    other_count: int,
) -> Dict[str, object]:
    file_rows = [row for row in rows if str(row["node_type"]) == "file"]
    other_rows = [row for row in rows if str(row["node_type"]) != "file"]
    selected = file_rows[:file_count] + other_rows[:other_count]
    selected_ids = {int(row["node_id"]) for row in selected}

    tp = fp = tn = fn = 0
    for row in rows:
        is_pred = int(row["node_id"]) in selected_ids
        is_gt = int(row["is_gt"]) == 1
        if is_pred and is_gt:
            tp += 1
        elif is_pred and not is_gt:
            fp += 1
        elif not is_pred and is_gt:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    threshold = 0.0
    if selected:
        threshold = min(float(row["score"]) for row in selected)
    return {
        "file_count": file_count,
        "other_count": other_count,
        "count": file_count + other_count,
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


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    graphs = summary["graphs"]
    graph_rows = {
        graph["name"]: read_node_scores(Path(graph["node_scores_file"]))
        for graph in graphs
    }
    file_fractions = args.sparse_file_fraction or [0.5, 0.7, 0.8, 0.9]

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "policy_config": {
            "sparse_score_threshold": args.sparse_score_threshold,
            "sparse_max_count": args.sparse_max_count,
            "dense_score_threshold": args.dense_score_threshold,
            "dense_min_count": args.dense_min_count,
            "moderate_top_count": args.moderate_top_count,
            "dense_robust_k": args.dense_robust_k,
            "sparse_total_count": args.sparse_total_count,
            "sparse_file_fractions": file_fractions,
        },
        "policies": [],
    }

    print(f"[compare-file-budget] eval_dir={eval_dir}", flush=True)
    print(f"[compare-file-budget] sparse_total_count={args.sparse_total_count}", flush=True)
    print(f"[compare-file-budget] sparse_file_fractions={file_fractions}", flush=True)

    for fraction in file_fractions:
        file_count = int(round(args.sparse_total_count * fraction))
        file_count = max(0, min(file_count, args.sparse_total_count))
        other_count = max(0, args.sparse_total_count - file_count)
        policy_name = f"sparse_file_fraction_{fraction:.2f}"
        policy_result = {
            "policy_name": policy_name,
            "sparse_file_fraction": fraction,
            "sparse_file_count": file_count,
            "sparse_other_count": other_count,
            "windows": [],
        }

        print(
            f"[compare-file-budget] policy={policy_name} "
            f"file_count={file_count} other_count={other_count}",
            flush=True,
        )

        for graph in graphs:
            rows = graph_rows[graph["name"]]
            regime_info = classify_regime(
                rows=rows,
                sparse_score_threshold=args.sparse_score_threshold,
                sparse_max_count=args.sparse_max_count,
                dense_score_threshold=args.dense_score_threshold,
                dense_min_count=args.dense_min_count,
            )
            regime = str(regime_info["regime"])

            if regime == "sparse":
                metrics = split_top_count_by_type(rows, file_count=file_count, other_count=other_count)
                metrics["policy"] = "file_priority_top_count"
            elif regime == "moderate":
                metrics = compute_confusion_top_count(rows, args.moderate_top_count)
                metrics["policy"] = "top_count"
            else:
                metrics = compute_confusion_robust(rows, args.dense_robust_k)
                metrics["policy"] = "window_median_plus_mad"
                metrics["k"] = args.dense_robust_k

            window_result = summarize_window(graph, policy_name, rows, {**metrics, **regime_info})
            policy_result["windows"].append(window_result)
            print_window_summary(window_result)

        results["policies"].append(policy_result)

    output_path = eval_dir / f"{args.output_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[compare-file-budget] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
