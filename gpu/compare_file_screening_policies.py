import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_history_file_only_eval"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features"
DEFAULT_FALLBACK_FEATURE_ROOTS = (
    REPO_ROOT / "artifacts" / "features",
    REPO_ROOT / "artifacts" / "features_cleaned",
    REPO_ROOT / "artifacts" / "features_model_ready",
)
DEFAULT_HISTORY_RESET_BEFORE_WINDOWS = ("test_2018-04-12",)
DEFAULT_TOPK = (300, 1000, 5000, 10000)
PATH_RISK_COLUMNS = (
    "temp_path_count",
    "hidden_path_count",
    "script_path_count",
    "system_bin_path_count",
)
BEHAVIOR_COLUMNS = (
    "write_count",
    "execute_count",
    "rename_count",
    "unlink_count",
    "modify_file_attr_count",
)
POLICY_NAMES = (
    "all_files",
    "score_top_rank",
    "screen_v1",
    "screen_v1_strict",
    "screen_v1_behavioral",
    "screen_v1_no_score",
    "screen_v1_no_history",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare high-recall file-node screening policies on top of an existing evaluation directory. "
            "Non-file nodes are always retained; policies only decide which file nodes remain in the candidate set."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=f"Feature root containing <window>/file_view__file_node.tsv. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--score-top-rank-max",
        type=int,
        default=10000,
        help="Keep file nodes already ranked within this global top-k band. Default: 10000",
    )
    parser.add_argument(
        "--min-unique-process-count",
        type=float,
        default=2.0,
        help="Keep file nodes touched by at least this many unique processes. Default: 2",
    )
    parser.add_argument(
        "--min-total-accesses",
        type=float,
        default=3.0,
        help="Keep file nodes with at least this many accesses. Default: 3",
    )
    parser.add_argument(
        "--min-behavior-count",
        type=float,
        default=1.0,
        help="Keep file nodes whose risky behavior counter reaches this threshold. Default: 1",
    )
    parser.add_argument(
        "--min-network-active-process-count",
        type=float,
        default=1.0,
        help="Keep file nodes backed by at least this many network-active processes. Default: 1",
    )
    parser.add_argument(
        "--min-path-risk-count",
        type=float,
        default=1.0,
        help="Keep file nodes whose path-risk counter reaches this threshold when path columns exist. Default: 1",
    )
    parser.add_argument(
        "--topk",
        type=int,
        action="append",
        default=[],
        help="Optional rank cutoffs for kept-file coverage summaries. Can be passed multiple times.",
    )
    parser.add_argument(
        "--history-reset-before-window",
        action="append",
        default=list(DEFAULT_HISTORY_RESET_BEFORE_WINDOWS),
        help=(
            "Window name(s) before which previous-window file presence is cleared. "
            "Default resets before test_2018-04-12 so test windows do not borrow val presence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "evaluations"),
        help="Directory for screening comparison outputs.",
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


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def to_int(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def read_tsv_rows_by_uuid(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {str(row["node_uuid"]): row for row in reader}


def load_scored_rows(eval_dir: Path, window_name: str) -> List[Dict[str, object]]:
    scores_path = eval_dir / window_name / "node_scores.tsv"
    with scores_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "rank": to_int(row["rank"]),
                    "node_id": to_int(row["node_id"]),
                    "node_uuid": row["node_uuid"],
                    "node_type": row["node_type"],
                    "is_gt": to_int(row["is_gt"]),
                    "score": to_float(row["score"]),
                }
            )
    rows.sort(key=lambda item: int(item["rank"]))
    return rows


def load_window_file_features(
    feature_root: Path,
    window_name: str,
    fallback_feature_roots: Tuple[Path, ...] = DEFAULT_FALLBACK_FEATURE_ROOTS,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Path]:
    candidate_roots: List[Path] = []
    seen: set[Path] = set()
    for root in (feature_root, *fallback_feature_roots):
        resolved = Path(root).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidate_roots.append(resolved)

    for root in candidate_roots:
        window_dir = root / window_name
        file_view_path = window_dir / "file_view__file_node.tsv"
        process_view_path = window_dir / "process_view__file_node.tsv"
        if file_view_path.exists() and process_view_path.exists():
            return (
                read_tsv_rows_by_uuid(file_view_path),
                read_tsv_rows_by_uuid(process_view_path),
                root,
            )

    searched = [str(root / window_name) for root in candidate_roots]
    raise FileNotFoundError(
        "Missing file/process feature tables for window "
        f"{window_name}. Searched: {searched}"
    )


def active_path_columns(file_view_rows: Dict[str, Dict[str, str]]) -> List[str]:
    if not file_view_rows:
        return []
    sample_row = next(iter(file_view_rows.values()))
    return [column for column in PATH_RISK_COLUMNS if column in sample_row]


def compute_file_feature_bundle(
    node_uuid: str,
    file_view_rows: Dict[str, Dict[str, str]],
    process_view_rows: Dict[str, Dict[str, str]],
    previous_file_uuids: set[str],
    available_path_columns: List[str],
) -> Dict[str, float | int]:
    file_row = file_view_rows.get(node_uuid, {})
    process_row = process_view_rows.get(node_uuid, {})
    unique_process_count = max(
        to_float(file_row.get("unique_process_count")),
        to_float(process_row.get("unique_process_count")),
    )
    total_accesses = to_float(file_row.get("total_accesses"))
    behavior_count = sum(to_float(file_row.get(column)) for column in BEHAVIOR_COLUMNS)
    network_active_process_count = to_float(process_row.get("network_active_process_count"))
    path_risk_count = sum(to_float(file_row.get(column)) for column in available_path_columns)
    return {
        "unique_process_count": unique_process_count,
        "total_accesses": total_accesses,
        "behavior_count": behavior_count,
        "network_active_process_count": network_active_process_count,
        "path_risk_count": path_risk_count,
        "prev_day_present": 1 if node_uuid in previous_file_uuids else 0,
    }


def compute_rule_flags(
    row: Dict[str, object],
    bundle: Dict[str, float | int],
    args: argparse.Namespace,
    has_path_columns: bool,
) -> Dict[str, bool]:
    return {
        "rank_top": int(row["rank"]) <= args.score_top_rank_max,
        "prev_day_present": int(bundle["prev_day_present"]) == 1,
        "multi_process": float(bundle["unique_process_count"]) >= args.min_unique_process_count,
        "multi_access": float(bundle["total_accesses"]) >= args.min_total_accesses,
        "behavioral_ops": float(bundle["behavior_count"]) >= args.min_behavior_count,
        "network_backed": float(bundle["network_active_process_count"]) >= args.min_network_active_process_count,
        "risky_path": has_path_columns and float(bundle["path_risk_count"]) >= args.min_path_risk_count,
    }


def keep_file_by_policy(policy_name: str, flags: Dict[str, bool]) -> bool:
    if policy_name == "all_files":
        return True
    if policy_name == "score_top_rank":
        return flags["rank_top"]
    if policy_name == "screen_v1":
        return any(flags.values())
    if policy_name == "screen_v1_strict":
        return (
            flags["rank_top"]
            or flags["prev_day_present"]
            or flags["behavioral_ops"]
            or flags["network_backed"]
            or flags["risky_path"]
            or (flags["multi_process"] and flags["multi_access"])
        )
    if policy_name == "screen_v1_behavioral":
        return (
            flags["rank_top"]
            or flags["prev_day_present"]
            or flags["behavioral_ops"]
            or flags["network_backed"]
            or flags["risky_path"]
        )
    if policy_name == "screen_v1_no_score":
        return any(value for key, value in flags.items() if key != "rank_top")
    if policy_name == "screen_v1_no_history":
        return any(value for key, value in flags.items() if key != "prev_day_present")
    raise ValueError(f"Unsupported policy: {policy_name}")


def summarize_counter(counter: Counter, limit: int = 12) -> List[Dict[str, object]]:
    return [{"value": key, "count": int(value)} for key, value in counter.most_common(limit)]


def kept_rank_coverage(
    file_rows: List[Dict[str, object]],
    kept_file_uuids: set[str],
    topk_values: List[int],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for k in topk_values:
        selected = [row for row in file_rows if int(row["rank"]) <= k]
        kept = sum(1 for row in selected if str(row["node_uuid"]) in kept_file_uuids)
        summaries.append(
            {
                "k": k,
                "file_rows": len(selected),
                "kept_file_rows": kept,
                "coverage": (kept / len(selected)) if selected else 0.0,
            }
        )
    return summaries


def summarize_window_policy(
    rows: List[Dict[str, object]],
    policy_name: str,
    flags_by_uuid: Dict[str, Dict[str, bool]],
    feature_bundles_by_uuid: Dict[str, Dict[str, float | int]],
    kept_file_uuids: set[str],
    available_path_columns: List[str],
    previous_window_name: str | None,
    topk_values: List[int],
) -> Dict[str, object]:
    total_nodes = len(rows)
    file_rows = [row for row in rows if str(row["node_type"]) == "file"]
    non_file_rows = [row for row in rows if str(row["node_type"]) != "file"]
    gt_rows = [row for row in rows if int(row["is_gt"]) == 1]
    gt_file_rows = [row for row in file_rows if int(row["is_gt"]) == 1]
    kept_file_rows = [row for row in file_rows if str(row["node_uuid"]) in kept_file_uuids]
    kept_gt_file_rows = [row for row in gt_file_rows if str(row["node_uuid"]) in kept_file_uuids]

    rule_counter: Counter = Counter()
    reason_counter: Counter = Counter()
    for row in kept_file_rows:
        uuid = str(row["node_uuid"])
        active_rules = sorted(key for key, value in flags_by_uuid[uuid].items() if value)
        rule_counter.update(active_rules)
        reason_counter["+".join(active_rules) if active_rules else "no_rule"] += 1

    lost_gt_preview: List[Dict[str, object]] = []
    for row in gt_file_rows:
        if str(row["node_uuid"]) in kept_file_uuids:
            continue
        uuid = str(row["node_uuid"])
        bundle = feature_bundles_by_uuid[uuid]
        lost_gt_preview.append(
            {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "node_uuid": uuid,
                "active_rules": [key for key, value in flags_by_uuid[uuid].items() if value],
                "unique_process_count": float(bundle["unique_process_count"]),
                "total_accesses": float(bundle["total_accesses"]),
                "behavior_count": float(bundle["behavior_count"]),
                "network_active_process_count": float(bundle["network_active_process_count"]),
                "path_risk_count": float(bundle["path_risk_count"]),
                "prev_day_present": int(bundle["prev_day_present"]),
            }
        )
    lost_gt_preview.sort(key=lambda item: int(item["rank"]))

    kept_total_nodes = len(non_file_rows) + len(kept_file_rows)
    gt_kept_total = len([row for row in gt_rows if str(row["node_type"]) != "file"]) + len(kept_gt_file_rows)
    return {
        "policy_name": policy_name,
        "previous_window_name": previous_window_name,
        "path_rule_columns": available_path_columns,
        "total_nodes": total_nodes,
        "total_file_nodes": len(file_rows),
        "kept_file_nodes": len(kept_file_rows),
        "screened_out_file_nodes": len(file_rows) - len(kept_file_rows),
        "kept_total_nodes": kept_total_nodes,
        "file_retention_ratio": (len(kept_file_rows) / len(file_rows)) if file_rows else 0.0,
        "total_retention_ratio": (kept_total_nodes / total_nodes) if total_nodes else 0.0,
        "gt_total": len(gt_rows),
        "gt_kept_total": gt_kept_total,
        "gt_total_recall": (gt_kept_total / len(gt_rows)) if gt_rows else 0.0,
        "gt_file_total": len(gt_file_rows),
        "gt_file_kept": len(kept_gt_file_rows),
        "gt_file_recall": (len(kept_gt_file_rows) / len(gt_file_rows)) if gt_file_rows else 0.0,
        "kept_rank_coverage": kept_rank_coverage(file_rows, kept_file_uuids, topk_values),
        "retention_rule_hits": summarize_counter(rule_counter),
        "retention_reason_patterns": summarize_counter(reason_counter),
        "lost_gt_file_preview": lost_gt_preview[:10],
    }


def print_window_summary(window_name: str, summary: Dict[str, object]) -> None:
    print(
        f"  [window] {window_name} "
        f"file_retention={float(summary['file_retention_ratio']):.6f} "
        f"total_retention={float(summary['total_retention_ratio']):.6f} "
        f"gt_file_recall={float(summary['gt_file_recall']):.6f} "
        f"kept_files={summary['kept_file_nodes']} "
        f"lost_gt_files={int(summary['gt_file_total']) - int(summary['gt_file_kept'])}",
        flush=True,
    )


def run_policy_on_window(
    rows: List[Dict[str, object]],
    policy_name: str,
    file_view_rows: Dict[str, Dict[str, str]],
    process_view_rows: Dict[str, Dict[str, str]],
    previous_file_uuids: set[str],
    available_path_columns: List[str],
    args: argparse.Namespace,
    previous_window_name: str | None,
    topk_values: List[int],
) -> Dict[str, object]:
    flags_by_uuid: Dict[str, Dict[str, bool]] = {}
    feature_bundles_by_uuid: Dict[str, Dict[str, float | int]] = {}
    kept_file_uuids: set[str] = set()

    for row in rows:
        if str(row["node_type"]) != "file":
            continue
        uuid = str(row["node_uuid"])
        bundle = compute_file_feature_bundle(
            node_uuid=uuid,
            file_view_rows=file_view_rows,
            process_view_rows=process_view_rows,
            previous_file_uuids=previous_file_uuids,
            available_path_columns=available_path_columns,
        )
        feature_bundles_by_uuid[uuid] = bundle
        flags = compute_rule_flags(
            row=row,
            bundle=bundle,
            args=args,
            has_path_columns=bool(available_path_columns),
        )
        flags_by_uuid[uuid] = flags
        if keep_file_by_policy(policy_name, flags):
            kept_file_uuids.add(uuid)

    return summarize_window_policy(
        rows=rows,
        policy_name=policy_name,
        flags_by_uuid=flags_by_uuid,
        feature_bundles_by_uuid=feature_bundles_by_uuid,
        kept_file_uuids=kept_file_uuids,
        available_path_columns=available_path_columns,
        previous_window_name=previous_window_name,
        topk_values=topk_values,
    )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    summary = load_json(eval_dir / "evaluation_summary.json")
    graph_summaries = list(summary["graphs"])
    topk_values = sorted({k for k in (args.topk or list(DEFAULT_TOPK)) if k > 0})

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    run_name = args.run_name or f"file_screening_policies_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    results = {
        "eval_dir": str(eval_dir),
        "feature_root": str(feature_root),
        "fallback_feature_roots": [str(Path(root).expanduser().resolve()) for root in DEFAULT_FALLBACK_FEATURE_ROOTS],
        "checkpoint": summary["checkpoint"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "score_method": summary.get("score_method"),
        "score_calibration": summary.get("score_calibration"),
        "history_reset_before_windows": list(args.history_reset_before_window),
        "policy_config": {
            "score_top_rank_max": args.score_top_rank_max,
            "min_unique_process_count": args.min_unique_process_count,
            "min_total_accesses": args.min_total_accesses,
            "min_behavior_count": args.min_behavior_count,
            "min_network_active_process_count": args.min_network_active_process_count,
            "min_path_risk_count": args.min_path_risk_count,
        },
        "topk_values": topk_values,
        "policies": [],
    }

    history_reset_before_windows = set(args.history_reset_before_window)
    previous_file_uuids: set[str] = set()
    previous_window_name: str | None = None

    print(f"[compare-file-screen] eval_dir={eval_dir}", flush=True)
    print(f"[compare-file-screen] feature_root={feature_root}", flush=True)
    print(f"[compare-file-screen] policy_config={results['policy_config']}", flush=True)
    print(f"[compare-file-screen] history_reset_before_windows={sorted(history_reset_before_windows)}", flush=True)

    per_policy_windows = {policy_name: [] for policy_name in POLICY_NAMES}
    for graph in graph_summaries:
        window_name = str(graph["name"])
        if window_name in history_reset_before_windows:
            previous_file_uuids = set()
            previous_window_name = None

        rows = load_scored_rows(eval_dir, window_name)
        file_view_rows, process_view_rows, resolved_feature_root = load_window_file_features(feature_root, window_name)
        available_path_columns = active_path_columns(file_view_rows)

        print(
            f"[compare-file-screen] window={window_name} "
            f"file_rows={sum(1 for row in rows if str(row['node_type']) == 'file')} "
            f"resolved_feature_root={resolved_feature_root} "
            f"path_rule_columns={available_path_columns}",
            flush=True,
        )

        for policy_name in POLICY_NAMES:
            window_summary = run_policy_on_window(
                rows=rows,
                policy_name=policy_name,
                file_view_rows=file_view_rows,
                process_view_rows=process_view_rows,
                previous_file_uuids=previous_file_uuids,
                available_path_columns=available_path_columns,
                args=args,
                previous_window_name=previous_window_name,
                topk_values=topk_values,
            )
            window_summary["name"] = window_name
            window_summary["resolved_feature_root"] = str(resolved_feature_root)
            per_policy_windows[policy_name].append(window_summary)
            print(f"[compare-file-screen] policy={policy_name}", flush=True)
            print_window_summary(window_name, window_summary)

        previous_file_uuids = {uuid for uuid, row in file_view_rows.items() if row.get("node_type") == "file"}
        previous_window_name = window_name

    for policy_name in POLICY_NAMES:
        results["policies"].append(
            {
                "policy_name": policy_name,
                "windows": per_policy_windows[policy_name],
            }
        )

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[compare-file-screen] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
