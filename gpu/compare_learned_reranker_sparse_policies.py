import argparse
import json
from pathlib import Path
from typing import Dict, List

from compare_adaptive_threshold_policies import evaluate_policy_on_window
from compare_threshold_strategies import load_json, read_node_scores


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_learned_rerank_eval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep sparse_top_count values for a learned-reranker evaluation directory and "
            "report val / test_2018-04-12 / test_2018-04-13 precision, recall, and F1."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--sparse-top-count",
        type=int,
        action="append",
        default=[],
        help="Sparse top-count to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sparse-score-threshold",
        type=float,
        default=1000.0,
        help="Sparse-regime detector score cutoff. Default: 1000",
    )
    parser.add_argument(
        "--sparse-max-count",
        type=int,
        default=1000,
        help="Sparse-regime detector max high-score count. Default: 1000",
    )
    parser.add_argument(
        "--dense-score-threshold",
        type=float,
        default=200.0,
        help="Dense-regime detector score cutoff. Default: 200",
    )
    parser.add_argument(
        "--dense-min-count",
        type=int,
        default=20000,
        help="Dense-regime detector min broad high-score mass. Default: 20000",
    )
    parser.add_argument(
        "--moderate-top-count",
        type=int,
        default=200,
        help="Moderate top-count budget. Default: 200",
    )
    parser.add_argument(
        "--dense-robust-k",
        type=float,
        default=20.0,
        help="Dense median+MAD k. Default: 20.0",
    )
    parser.add_argument(
        "--dense-absolute-threshold",
        type=float,
        default=0.0,
        help="Optional dense absolute threshold override. Default: 0",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="learned_reranker_sparse_sweep",
        help="Output JSON filename stem under eval-dir. Default: learned_reranker_sparse_sweep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    graphs = list(summary["graphs"])
    graph_rows: Dict[str, List[Dict[str, object]]] = {
        graph["name"]: read_node_scores(Path(graph["node_scores_file"]))
        for graph in graphs
    }
    sparse_top_counts = args.sparse_top_count or [150, 200, 250, 300, 350, 400, 500]

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "post_rerank_method": summary.get("post_rerank_method"),
        "post_rerank_candidate_rank_max": summary.get("post_rerank_candidate_rank_max"),
        "learned_state": summary.get("learned_state"),
        "sparse_top_counts": sparse_top_counts,
        "settings": [],
    }

    print(f"[learned-reranker-sparse-sweep] eval_dir={eval_dir}", flush=True)
    print(f"[learned-reranker-sparse-sweep] sparse_top_counts={sparse_top_counts}", flush=True)

    for sparse_top_count in sparse_top_counts:
        setting = {
            "sparse_top_count": sparse_top_count,
            "windows": [],
        }
        local_args = argparse.Namespace(
            sparse_score_threshold=args.sparse_score_threshold,
            sparse_max_count=args.sparse_max_count,
            dense_score_threshold=args.dense_score_threshold,
            dense_min_count=args.dense_min_count,
            sparse_top_count=sparse_top_count,
            moderate_top_count=args.moderate_top_count,
            dense_robust_k=args.dense_robust_k,
            dense_absolute_threshold=args.dense_absolute_threshold,
        )
        for graph in graphs:
            window_name = graph["name"]
            metrics = evaluate_policy_on_window(graph_rows[window_name], local_args)
            window_result = {
                "name": window_name,
                "regime": metrics["regime"],
                "policy": metrics["policy"],
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "predicted_positive": int(metrics["predicted_positive"]),
            }
            setting["windows"].append(window_result)
        results["settings"].append(setting)
        window_map = {w["name"]: w for w in setting["windows"]}
        print(
            f"sparse={sparse_top_count}",
            "val_f1=", round(window_map["val"]["f1"], 6),
            "0412_f1=", round(window_map["test_2018-04-12"]["f1"], 6),
            "0413_f1=", round(window_map["test_2018-04-13"]["f1"], 6),
            "0413_precision=", round(window_map["test_2018-04-13"]["precision"], 6),
            "0413_recall=", round(window_map["test_2018-04-13"]["recall"], 6),
            "0413_pred_pos=", window_map["test_2018-04-13"]["predicted_positive"],
            flush=True,
        )

    output_path = eval_dir / f"{args.output_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[learned-reranker-sparse-sweep] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
