import argparse
import json
from pathlib import Path
from typing import Dict, List

from compare_threshold_strategies import (
    compute_confusion_robust,
    compute_confusion_threshold,
    compute_confusion_top_count,
    load_json,
    print_window_summary,
    read_node_scores,
    summarize_window,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_support128_file_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate simple adaptive threshold policies that first classify each window into "
            "sparse / moderate / dense regimes based on score-tail mass, then apply a regime-specific threshold."
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
        help="Sparse-regime detector: score cutoff used to count high-score nodes. Default: 1000",
    )
    parser.add_argument(
        "--sparse-max-count",
        type=int,
        default=1000,
        help="Sparse-regime detector: if count(score >= sparse-score-threshold) is below this, use sparse policy. Default: 1000",
    )
    parser.add_argument(
        "--dense-score-threshold",
        type=float,
        default=200.0,
        help="Dense-regime detector: score cutoff used to count broad high-score mass. Default: 200",
    )
    parser.add_argument(
        "--dense-min-count",
        type=int,
        default=20000,
        help="Dense-regime detector: if count(score >= dense-score-threshold) is at least this, use dense policy. Default: 20000",
    )
    parser.add_argument(
        "--sparse-top-count",
        type=int,
        default=138,
        help="Top-count threshold used for sparse windows. Default: 138",
    )
    parser.add_argument(
        "--moderate-top-count",
        type=int,
        default=200,
        help="Top-count threshold used for moderate windows. Default: 200",
    )
    parser.add_argument(
        "--dense-robust-k",
        type=float,
        default=20.0,
        help="Median+MAD k used for dense windows. Default: 20.0",
    )
    parser.add_argument(
        "--dense-absolute-threshold",
        type=float,
        default=0.0,
        help="Optional override: if > 0, dense windows use this absolute threshold instead of robust-k. Default: 0",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="adaptive_threshold_policy_v1",
        help="Output JSON filename stem under eval-dir. Default: adaptive_threshold_policy_v1",
    )
    return parser.parse_args()


def count_scores_at_or_above(rows: List[Dict[str, float | int | str]], threshold: float) -> int:
    return sum(1 for row in rows if float(row["score"]) >= threshold)


def classify_regime(
    rows: List[Dict[str, float | int | str]],
    sparse_score_threshold: float,
    sparse_max_count: int,
    dense_score_threshold: float,
    dense_min_count: int,
) -> Dict[str, float | int | str]:
    sparse_tail_count = count_scores_at_or_above(rows, sparse_score_threshold)
    dense_mass_count = count_scores_at_or_above(rows, dense_score_threshold)
    if sparse_tail_count < sparse_max_count:
        regime = "sparse"
    elif dense_mass_count >= dense_min_count:
        regime = "dense"
    else:
        regime = "moderate"

    return {
        "regime": regime,
        "sparse_score_threshold": sparse_score_threshold,
        "sparse_tail_count": sparse_tail_count,
        "sparse_max_count": sparse_max_count,
        "dense_score_threshold": dense_score_threshold,
        "dense_mass_count": dense_mass_count,
        "dense_min_count": dense_min_count,
    }


def evaluate_policy_on_window(
    rows: List[Dict[str, float | int | str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    regime_info = classify_regime(
        rows=rows,
        sparse_score_threshold=args.sparse_score_threshold,
        sparse_max_count=args.sparse_max_count,
        dense_score_threshold=args.dense_score_threshold,
        dense_min_count=args.dense_min_count,
    )
    regime = regime_info["regime"]

    if regime == "sparse":
        metrics = compute_confusion_top_count(rows, args.sparse_top_count)
        policy_details = {
            "policy": "top_count",
            "count": args.sparse_top_count,
        }
    elif regime == "moderate":
        metrics = compute_confusion_top_count(rows, args.moderate_top_count)
        policy_details = {
            "policy": "top_count",
            "count": args.moderate_top_count,
        }
    else:
        if args.dense_absolute_threshold > 0:
            metrics = compute_confusion_threshold(rows, args.dense_absolute_threshold)
            policy_details = {
                "policy": "absolute_threshold",
                "threshold": args.dense_absolute_threshold,
            }
        else:
            metrics = compute_confusion_robust(rows, args.dense_robust_k)
            policy_details = {
                "policy": "window_median_plus_mad",
                "k": args.dense_robust_k,
            }

    return {
        **metrics,
        **policy_details,
        **regime_info,
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary_path = eval_dir / "evaluation_summary.json"
    output_path = eval_dir / f"{args.output_name}.json"

    summary = load_json(summary_path)
    graphs = summary["graphs"]
    graph_rows = {
        graph["name"]: read_node_scores(Path(graph["node_scores_file"]))
        for graph in graphs
    }

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
            "sparse_top_count": args.sparse_top_count,
            "moderate_top_count": args.moderate_top_count,
            "dense_robust_k": args.dense_robust_k,
            "dense_absolute_threshold": args.dense_absolute_threshold,
        },
        "windows": [],
    }

    print(f"[adaptive-threshold] eval_dir={eval_dir}", flush=True)
    print(f"[adaptive-threshold] checkpoint={summary['checkpoint']}", flush=True)
    print(f"[adaptive-threshold] checkpoint_epoch={summary['checkpoint_epoch']}", flush=True)
    if summary.get("score_method") is not None:
        print(f"[adaptive-threshold] score_method={summary['score_method']}", flush=True)
    if summary.get("score_calibration") is not None:
        print(f"[adaptive-threshold] score_calibration={summary['score_calibration']}", flush=True)
    print(f"[adaptive-threshold] policy_config={results['policy_config']}", flush=True)

    for graph in graphs:
        rows = graph_rows[graph["name"]]
        metrics = evaluate_policy_on_window(rows, args)
        window_result = summarize_window(graph, "adaptive_threshold_policy_v1", rows, metrics)
        results["windows"].append(window_result)
        print(
            f"[adaptive-threshold] window={graph['name']} "
            f"regime={metrics['regime']} "
            f"policy={metrics['policy']} "
            f"sparse_tail_count={metrics['sparse_tail_count']} "
            f"dense_mass_count={metrics['dense_mass_count']}",
            flush=True,
        )
        print_window_summary(window_result)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[adaptive-threshold] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
