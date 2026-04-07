import argparse
import json
from pathlib import Path
from typing import Dict, List

from analyze_file_false_positives import (
    compute_selected_node_stats,
    load_json,
    load_scored_rows,
    resolve_window_graph_paths,
)
from compare_adaptive_threshold_policies import evaluate_policy_on_window
from compare_learned_file_rerankers import (
    DEFAULT_FEATURE_NAMES,
    build_learned_feature_rows,
    rerank_rows_with_learned_model,
    tensorize_training_rows,
    train_learned_linear_reranker,
)
from file_reranking import build_previous_file_percentiles, candidate_file_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
)
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features_model_ready_hybrid_file_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep candidate_rank_max values for the lightweight learned file reranker and "
            "report val / test_2018-04-12 / test_2018-04-13 precision, recall, and F1."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Base evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--graph-root",
        type=str,
        default=str(DEFAULT_GRAPH_ROOT),
        help=f"Fallback root containing <window>/graph.pt. Default: {DEFAULT_GRAPH_ROOT}",
    )
    parser.add_argument(
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=f"Feature root for learned reranker features. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--select-window",
        type=str,
        default="val",
        help="Window used to fit the lightweight learned reranker. Default: val",
    )
    parser.add_argument(
        "--candidate-rank-max",
        type=int,
        action="append",
        default=[],
        help="Candidate-rank-max to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Number of optimization steps for the learned reranker. Default: 1000",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for the learned reranker. Default: 0.05",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the learned reranker. Default: 1e-4",
    )
    parser.add_argument(
        "--history-reset-before-window",
        action="append",
        default=[],
        help="Optional window name(s) before which previous-window file history is cleared.",
    )
    parser.add_argument(
        "--relation-group-scheme",
        type=str,
        default="coarse_v1",
        help="Relation grouping used when deriving file-node graph stats. Default: coarse_v1",
    )
    parser.add_argument(
        "--sparse-top-count",
        type=int,
        default=250,
        help="Sparse top-count budget. Default: 250",
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
        default="learned_reranker_candidate_max_sweep",
        help="Output JSON filename stem under eval-dir. Default: learned_reranker_candidate_max_sweep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()

    summary = load_json(eval_dir / "evaluation_summary.json")
    graph_summaries = list(summary["graphs"])
    graph_paths = resolve_window_graph_paths(eval_dir, graph_root)

    candidate_rank_maxs = args.candidate_rank_max or [1000, 2000, 5000, 10000]
    history_reset_before_windows = set(args.history_reset_before_window)

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "candidate_rank_maxs": candidate_rank_maxs,
        "select_window": args.select_window,
        "feature_root": str(feature_root),
        "sparse_top_count": args.sparse_top_count,
        "settings": [],
    }

    print(f"[learned-reranker-candidate-sweep] eval_dir={eval_dir}", flush=True)
    print(f"[learned-reranker-candidate-sweep] candidate_rank_maxs={candidate_rank_maxs}", flush=True)
    print(f"[learned-reranker-candidate-sweep] sparse_top_count={args.sparse_top_count}", flush=True)

    for candidate_rank_max in candidate_rank_maxs:
        window_contexts: List[Dict[str, object]] = []
        previous_file_percentiles_by_uuid: Dict[str, float] = {}

        for graph_summary in graph_summaries:
            window_name = str(graph_summary["name"])
            if window_name in history_reset_before_windows:
                previous_file_percentiles_by_uuid = {}

            base_rows = load_scored_rows(eval_dir, window_name)
            candidate_rows = candidate_file_rows(base_rows, candidate_rank_max)
            graph_path = graph_paths[window_name]
            node_stats = compute_selected_node_stats(
                graph_path=graph_path,
                selected_node_ids=[int(row["node_id"]) for row in candidate_rows],
                relation_group_scheme=args.relation_group_scheme,
            )
            feature_rows, _, _ = build_learned_feature_rows(
                candidate_rows=candidate_rows,
                node_stats=node_stats,
                previous_file_percentiles_by_uuid=previous_file_percentiles_by_uuid,
                graph_path=graph_path,
                feature_root=feature_root,
            )
            window_contexts.append(
                {
                    "window_name": window_name,
                    "base_rows": base_rows,
                    "candidate_rows": candidate_rows,
                    "feature_rows": feature_rows,
                }
            )
            previous_file_percentiles_by_uuid = build_previous_file_percentiles(base_rows)

        select_context = next((ctx for ctx in window_contexts if ctx["window_name"] == args.select_window), None)
        if select_context is None:
            raise ValueError(f"Select window not found in eval dir: {args.select_window}")

        x_train, y_train = tensorize_training_rows(
            select_context["candidate_rows"],
            select_context["feature_rows"],
            DEFAULT_FEATURE_NAMES,
        )
        if int(y_train.sum().item()) <= 0:
            raise ValueError(
                f"No positive file candidates found for select window {args.select_window} "
                f"with candidate_rank_max={candidate_rank_max}"
            )
        learned_state = train_learned_linear_reranker(
            x=x_train,
            y=y_train,
            feature_names=DEFAULT_FEATURE_NAMES,
            train_steps=args.train_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        local_args = argparse.Namespace(
            sparse_score_threshold=args.sparse_score_threshold,
            sparse_max_count=args.sparse_max_count,
            dense_score_threshold=args.dense_score_threshold,
            dense_min_count=args.dense_min_count,
            sparse_top_count=args.sparse_top_count,
            moderate_top_count=args.moderate_top_count,
            dense_robust_k=args.dense_robust_k,
            dense_absolute_threshold=args.dense_absolute_threshold,
        )
        setting = {
            "candidate_rank_max": candidate_rank_max,
            "learned_state": learned_state,
            "windows": [],
        }
        for context in window_contexts:
            reranked_rows = rerank_rows_with_learned_model(
                rows=context["base_rows"],
                candidate_rows=context["candidate_rows"],
                feature_rows=context["feature_rows"],
                learned_state=learned_state,
            )
            metrics = evaluate_policy_on_window(reranked_rows, local_args)
            setting["windows"].append(
                {
                    "name": context["window_name"],
                    "regime": metrics["regime"],
                    "policy": metrics["policy"],
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "predicted_positive": int(metrics["predicted_positive"]),
                }
            )

        results["settings"].append(setting)
        window_map = {w["name"]: w for w in setting["windows"]}
        print(
            "candidate_rank_max=", candidate_rank_max,
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
    print(f"[learned-reranker-candidate-sweep] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
