import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE_SCREEN_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features"
DEFAULT_FILE_SCREEN_FALLBACK_FEATURE_ROOTS = (
    REPO_ROOT / "artifacts" / "features",
    REPO_ROOT / "artifacts" / "features_cleaned",
    REPO_ROOT / "artifacts" / "features_model_ready",
)
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
FILE_SCREEN_POLICIES = (
    "none",
    "all_files",
    "score_top_rank",
    "screen_v1",
    "screen_v1_strict",
    "screen_v1_behavioral",
    "screen_v1_no_score",
    "screen_v1_no_history",
)


def to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def read_tsv_rows_by_uuid(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {str(row["node_uuid"]): row for row in reader}


def load_window_file_features(
    feature_root: Path,
    window_name: str,
    fallback_feature_roots: Tuple[Path, ...] = DEFAULT_FILE_SCREEN_FALLBACK_FEATURE_ROOTS,
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
    score_top_rank_max: int,
    min_unique_process_count: float,
    min_total_accesses: float,
    min_behavior_count: float,
    min_network_active_process_count: float,
    min_path_risk_count: float,
    has_path_columns: bool,
) -> Dict[str, bool]:
    return {
        "rank_top": int(row["rank"]) <= score_top_rank_max,
        "prev_day_present": int(bundle["prev_day_present"]) == 1,
        "multi_process": float(bundle["unique_process_count"]) >= min_unique_process_count,
        "multi_access": float(bundle["total_accesses"]) >= min_total_accesses,
        "behavioral_ops": float(bundle["behavior_count"]) >= min_behavior_count,
        "network_backed": float(bundle["network_active_process_count"]) >= min_network_active_process_count,
        "risky_path": has_path_columns and float(bundle["path_risk_count"]) >= min_path_risk_count,
    }


def keep_file_by_policy(policy_name: str, flags: Dict[str, bool]) -> bool:
    if policy_name in {"none", "all_files"}:
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
    raise ValueError(f"Unsupported file screen policy: {policy_name}")


def apply_file_screen_policy(
    rows: List[Dict[str, object]],
    feature_root: Path,
    window_name: str,
    policy_name: str,
    previous_file_uuids: set[str] | None = None,
    score_top_rank_max: int = 10000,
    min_unique_process_count: float = 2.0,
    min_total_accesses: float = 3.0,
    min_behavior_count: float = 1.0,
    min_network_active_process_count: float = 1.0,
    min_path_risk_count: float = 1.0,
) -> Tuple[List[Dict[str, object]], Dict[str, object], set[str]]:
    previous_uuids = previous_file_uuids or set()
    if policy_name == "none":
        passthrough = [dict(row, screen_keep=1) for row in rows]
        return (
            passthrough,
            {
                "file_screen_policy": "none",
                "resolved_feature_root": None,
                "path_rule_columns": [],
                "total_file_nodes": sum(1 for row in rows if str(row["node_type"]) == "file"),
                "kept_file_nodes": sum(1 for row in rows if str(row["node_type"]) == "file"),
                "screened_out_file_nodes": 0,
                "file_retention_ratio": 1.0,
                "gt_file_total": sum(1 for row in rows if str(row["node_type"]) == "file" and int(row["is_gt"]) == 1),
                "gt_file_kept": sum(1 for row in rows if str(row["node_type"]) == "file" and int(row["is_gt"]) == 1),
                "gt_file_recall": 1.0,
            },
            set(),
        )

    file_view_rows, process_view_rows, resolved_feature_root = load_window_file_features(feature_root, window_name)
    available_path_columns = active_path_columns(file_view_rows)
    file_rows = [row for row in rows if str(row["node_type"]) == "file"]
    total_file_nodes = len(file_rows)
    gt_file_total = sum(1 for row in file_rows if int(row["is_gt"]) == 1)

    keep_uuids: set[str] = set()
    for row in file_rows:
        uuid = str(row["node_uuid"])
        bundle = compute_file_feature_bundle(
            node_uuid=uuid,
            file_view_rows=file_view_rows,
            process_view_rows=process_view_rows,
            previous_file_uuids=previous_uuids,
            available_path_columns=available_path_columns,
        )
        flags = compute_rule_flags(
            row=row,
            bundle=bundle,
            score_top_rank_max=score_top_rank_max,
            min_unique_process_count=min_unique_process_count,
            min_total_accesses=min_total_accesses,
            min_behavior_count=min_behavior_count,
            min_network_active_process_count=min_network_active_process_count,
            min_path_risk_count=min_path_risk_count,
            has_path_columns=bool(available_path_columns),
        )
        if keep_file_by_policy(policy_name, flags):
            keep_uuids.add(uuid)

    scores = [float(row["score"]) for row in rows]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    score_floor = min_score - max(1.0, (max_score - min_score) + 1.0)

    screened_rows: List[Dict[str, object]] = []
    screened_out_index = 0
    gt_file_kept = 0
    kept_file_nodes = 0
    for row in rows:
        updated = dict(row)
        if str(row["node_type"]) != "file":
            updated["screen_keep"] = 1
        else:
            keep = str(row["node_uuid"]) in keep_uuids
            updated["screen_keep"] = 1 if keep else 0
            if keep:
                kept_file_nodes += 1
                if int(row["is_gt"]) == 1:
                    gt_file_kept += 1
            else:
                screened_out_index += 1
                updated["score"] = float(score_floor - screened_out_index * 1e-6)
        screened_rows.append(updated)

    screened_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, row in enumerate(screened_rows, start=1):
        row["rank"] = rank

    summary = {
        "file_screen_policy": policy_name,
        "resolved_feature_root": str(resolved_feature_root),
        "path_rule_columns": list(available_path_columns),
        "total_file_nodes": total_file_nodes,
        "kept_file_nodes": kept_file_nodes,
        "screened_out_file_nodes": total_file_nodes - kept_file_nodes,
        "file_retention_ratio": (kept_file_nodes / total_file_nodes) if total_file_nodes else 0.0,
        "gt_file_total": gt_file_total,
        "gt_file_kept": gt_file_kept,
        "gt_file_recall": (gt_file_kept / gt_file_total) if gt_file_total else 0.0,
    }
    next_previous_file_uuids = set(file_view_rows.keys())
    return screened_rows, summary, next_previous_file_uuids


def is_screen_kept(row: Dict[str, float | int | str]) -> bool:
    return int(row.get("screen_keep", 1)) == 1


def eligible_rows(rows: List[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    return [row for row in rows if is_screen_kept(row)]


def eligible_scores(rows: List[Dict[str, float | int | str]]) -> List[float]:
    return [float(row["score"]) for row in rows if is_screen_kept(row)]


def score_floor_for_screened_rows(rows: List[Dict[str, object]]) -> float:
    scores = [float(row["score"]) for row in rows]
    if not scores:
        return -1.0
    return min(scores) - max(1.0, max(scores) - min(scores) + 1.0)


def threshold_or_inf_for_empty(threshold: float, rows: List[Dict[str, float | int | str]]) -> float:
    return threshold if eligible_rows(rows) else math.inf
