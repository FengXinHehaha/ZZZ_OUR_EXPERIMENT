import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from analyze_file_false_positives import load_json, load_scored_rows, resolve_window_graph_paths
from compare_adaptive_threshold_policies import evaluate_policy_on_window
from compare_threshold_strategies import print_window_summary, summarize_window
from file_reranking import build_previous_file_percentiles, rerank_scored_rows_for_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep post-rerank candidate-rank-max values on top of an existing evaluation directory, "
            "and score each setting with the same adaptive threshold policy."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--graph-root",
        type=str,
        default=str(DEFAULT_GRAPH_ROOT),
        help=f"Fallback root containing <window>/graph.pt. Default: {DEFAULT_GRAPH_ROOT}",
    )
    parser.add_argument(
        "--candidate-rank-max",
        type=int,
        action="append",
        default=[],
        help="Candidate-rank-max to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--post-rerank-method",
        type=str,
        default="file_rerank_support",
        help="Post-rerank method to apply. Default: file_rerank_support",
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
        "--sparse-top-count",
        type=int,
        default=300,
        help="Sparse top-count budget. Default: 300",
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
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "evaluations"),
        help="Directory for sweep outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output directory name.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summarize_candidate_setting(
    eval_summary: Dict[str, object],
    graph_summaries: List[Dict[str, object]],
    graph_rows: Dict[str, List[Dict[str, object]]],
    graph_paths: Dict[str, Path],
    args: argparse.Namespace,
    candidate_rank_max: int,
) -> Dict[str, object]:
    history_reset_before_windows = set(args.history_reset_before_window)
    previous_file_percentiles_by_uuid: Dict[str, float] = {}
    windows: List[Dict[str, object]] = []

    for graph in graph_summaries:
        window_name = str(graph["name"])
        if window_name in history_reset_before_windows:
            previous_file_percentiles_by_uuid = {}

        base_rows = graph_rows[window_name]
        reranked_rows = rerank_scored_rows_for_graph(
            rows=base_rows,
            graph_path=graph_paths[window_name],
            relation_group_scheme=args.relation_group_scheme,
            method_name=args.post_rerank_method,
            candidate_rank_max=candidate_rank_max,
            previous_file_percentiles_by_uuid=previous_file_percentiles_by_uuid,
        )
        metrics = evaluate_policy_on_window(reranked_rows, args)
        window_result = summarize_window(
            graph,
            f"{args.post_rerank_method}_adaptive_policy",
            reranked_rows,
            metrics,
        )
        windows.append(window_result)
        print(
            f"[rerank-candidate-sweep] candidate_rank_max={candidate_rank_max} "
            f"window={window_name} regime={metrics['regime']} policy={metrics['policy']}",
            flush=True,
        )
        print_window_summary(window_result)
        previous_file_percentiles_by_uuid = build_previous_file_percentiles(base_rows)

    return {
        "candidate_rank_max": candidate_rank_max,
        "post_rerank_method": args.post_rerank_method,
        "windows": windows,
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    graph_summaries = list(summary["graphs"])
    graph_rows = {graph["name"]: load_scored_rows(eval_dir, graph["name"]) for graph in graph_summaries}
    graph_paths = resolve_window_graph_paths(eval_dir, graph_root)

    candidate_rank_maxs = args.candidate_rank_max or [2000, 5000, 10000, 20000]
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    run_name = args.run_name or (
        f"rerank_candidate_sweep_{eval_dir.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    results = {
        "eval_dir": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "base_post_rerank_method": summary.get("post_rerank_method"),
        "sweep_post_rerank_method": args.post_rerank_method,
        "history_reset_before_windows": list(args.history_reset_before_window),
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
        "candidate_rank_maxs": candidate_rank_maxs,
        "settings": [],
    }

    print(f"[rerank-candidate-sweep] eval_dir={eval_dir}", flush=True)
    print(f"[rerank-candidate-sweep] sweep_post_rerank_method={args.post_rerank_method}", flush=True)
    print(f"[rerank-candidate-sweep] candidate_rank_maxs={candidate_rank_maxs}", flush=True)
    print(f"[rerank-candidate-sweep] policy_config={results['policy_config']}", flush=True)

    for candidate_rank_max in candidate_rank_maxs:
        setting = summarize_candidate_setting(
            eval_summary=summary,
            graph_summaries=graph_summaries,
            graph_rows=graph_rows,
            graph_paths=graph_paths,
            args=args,
            candidate_rank_max=candidate_rank_max,
        )
        results["settings"].append(setting)

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[rerank-candidate-sweep] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
