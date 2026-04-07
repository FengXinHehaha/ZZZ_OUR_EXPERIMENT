import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from analyze_file_false_positives import (
    compute_selected_node_stats,
    load_json,
    load_scored_rows,
    resolve_window_graph_paths,
)
from compare_learned_file_rerankers import (
    DEFAULT_FEATURE_NAMES,
    LEARNED_METHOD_NAME,
    build_learned_feature_rows,
    ensure_dir,
    rerank_rows_with_learned_model,
    summarize_method,
    tensorize_training_rows,
    train_learned_linear_reranker,
)
from file_reranking import build_previous_file_percentiles, candidate_file_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features_model_ready_hybrid_file_v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a lightweight learned file reranker on one evaluation window and export a new "
            "evaluation directory whose node_scores.tsv files already contain the learned reranked scores."
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
        default=2000,
        help="Only rerank file nodes whose current rank is at most this cutoff. Default: 2000",
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
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Optional K for Precision@K / Recall@K summaries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for exported evaluation outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional exported evaluation directory name.",
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

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    run_name = args.run_name or (
        f"learned_eval_{eval_dir.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    topk = args.topk or [100, 1000, 5000, 10000]
    history_reset_before_windows = set(args.history_reset_before_window)

    window_contexts: List[Dict[str, object]] = []
    previous_file_percentiles_by_uuid: Dict[str, float] = {}
    for graph_summary in graph_summaries:
        window_name = str(graph_summary["name"])
        if window_name in history_reset_before_windows:
            previous_file_percentiles_by_uuid = {}

        base_rows = load_scored_rows(eval_dir, window_name)
        candidate_rows = candidate_file_rows(base_rows, args.candidate_rank_max)
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
                "graph_summary": graph_summary,
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
            "No positive file candidates found in the select window within the current "
            "--candidate-rank-max. Increase the cutoff or choose a different select window."
        )
    learned_state = train_learned_linear_reranker(
        x=x_train,
        y=y_train,
        feature_names=DEFAULT_FEATURE_NAMES,
        train_steps=args.train_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    aggregate = {
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "best_record": summary.get("best_record"),
        "selection_metric": summary.get("selection_metric"),
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "post_rerank_method": LEARNED_METHOD_NAME,
        "post_rerank_candidate_rank_max": args.candidate_rank_max,
        "post_rerank_feature_root": str(feature_root),
        "history_source_method": summary.get("history_source_method"),
        "history_reset_before_windows": list(args.history_reset_before_window),
        "select_window": args.select_window,
        "learned_state": learned_state,
        "graphs": [],
    }

    print(f"[export-learned-file-reranker] base_eval_dir={eval_dir}", flush=True)
    print(f"[export-learned-file-reranker] feature_root={feature_root}", flush=True)
    print(f"[export-learned-file-reranker] select_window={args.select_window}", flush=True)
    print(f"[export-learned-file-reranker] candidate_rank_max={args.candidate_rank_max}", flush=True)
    print(f"[export-learned-file-reranker] learned_state={learned_state}", flush=True)

    for context in window_contexts:
        graph_summary = context["graph_summary"]
        window_name = context["window_name"]
        reranked_rows = rerank_rows_with_learned_model(
            rows=context["base_rows"],
            candidate_rows=context["candidate_rows"],
            feature_rows=context["feature_rows"],
            learned_state=learned_state,
        )
        window_output_dir = run_dir / window_name
        ensure_dir(window_output_dir)
        summary_row = summarize_method(
            graph_summary=graph_summary,
            reranked_rows=reranked_rows,
            topk=topk,
            node_scores_path=window_output_dir / "node_scores.tsv",
        )
        summary_row.update(
            {
                "path": graph_summary["path"],
                "edge_loss": graph_summary.get("edge_loss"),
                "score_method": graph_summary.get("score_method", summary.get("score_method")),
                "score_calibration": graph_summary.get("score_calibration", summary.get("score_calibration")),
                "post_rerank_method": LEARNED_METHOD_NAME,
                "post_rerank_candidate_rank_max": args.candidate_rank_max,
                "post_rerank_feature_root": str(feature_root),
            }
        )
        with (window_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_row, handle, indent=2)
        aggregate["graphs"].append(summary_row)
        print(
            f"[export-learned-file-reranker] {window_name}: "
            f"roc_auc={summary_row['roc_auc']} ap={summary_row['average_precision']} "
            f"post_rerank_method={summary_row['post_rerank_method']}",
            flush=True,
        )

    with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"[export-learned-file-reranker] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
