import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_FEATURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "features"
DEFAULT_GT_ALIGNMENT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gt_entity_alignment" / "per_uuid.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "retention"

FILE_VIEW_FILE = "file_view__file_node.tsv"
PROCESS_VIEW_FILE = "process_view__file_node.tsv"

FILE_NUMERIC_COLUMNS = [
    "size_bytes",
    "total_accesses",
    "unique_process_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "close_count",
    "create_object_count",
    "unlink_count",
    "rename_count",
    "modify_file_attr_count",
    "read_ratio",
    "write_ratio",
    "execute_ratio",
    "open_ratio",
]

PROCESS_CONTEXT_NUMERIC_COLUMNS = [
    "total_process_context_events",
    "unique_process_count",
    "event_type_diversity",
    "read_by_process_count",
    "write_by_process_count",
    "open_by_process_count",
    "exec_by_process_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "network_active_process_count",
    "avg_network_events_of_accessing_processes",
    "max_network_events_of_accessing_processes",
    "network_active_process_ratio",
]

DECISION_COLUMNS = [
    "node_uuid",
    "file_type",
    "is_gt_file",
    "hard_keep",
    "soft_keep",
    "keep",
    "decision_reason",
    "dropped_at_step",
    "total_accesses",
    "unique_process_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "create_object_count",
    "unlink_count",
    "rename_count",
    "modify_file_attr_count",
    "network_active_process_count",
    "network_active_process_ratio",
]

DROPPED_GT_COLUMNS = [
    "node_uuid",
    "window_name",
    "file_type",
    "decision_reason",
    "dropped_at_step",
    "total_accesses",
    "unique_process_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "create_object_count",
    "unlink_count",
    "rename_count",
    "modify_file_attr_count",
    "network_active_process_count",
    "network_active_process_ratio",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply file-node retention rules with GT-loss audit.")
    parser.add_argument(
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=f"Root directory containing extracted feature TSVs. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--gt-alignment",
        type=str,
        default=str(DEFAULT_GT_ALIGNMENT_PATH),
        help=f"Path to GT entity alignment TSV. Default: {DEFAULT_GT_ALIGNMENT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for retention outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def to_number(value: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def to_int(value: str) -> int:
    return int(round(to_number(value)))


def format_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_gt_file_set(path: Path) -> Set[str]:
    gt_files: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["entity_type"] == "file" and row["visible_in_events"] == "yes":
                gt_files.add(row["gt_uuid"])
    return gt_files


def load_tsv_by_uuid(path: Path, numeric_columns: Iterable[str]) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    numeric_columns = list(numeric_columns)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cleaned: Dict[str, object] = dict(row)
            for col in numeric_columns:
                cleaned[col] = to_number(row.get(col, ""))
            rows[row["node_uuid"]] = cleaned
    return rows


def decision_for_row(file_row: Dict[str, object], process_row: Dict[str, object], gt_files: Set[str]) -> Dict[str, object]:
    node_uuid = str(file_row["node_uuid"])
    file_type = str(file_row.get("file_type", ""))
    is_gt = node_uuid in gt_files

    total_accesses = to_int(str(file_row.get("total_accesses", 0)))
    unique_process_count = to_int(str(file_row.get("unique_process_count", 0)))
    read_count = to_int(str(file_row.get("read_count", 0)))
    write_count = to_int(str(file_row.get("write_count", 0)))
    open_count = to_int(str(file_row.get("open_count", 0)))
    execute_count = to_int(str(file_row.get("execute_count", 0)))
    create_object_count = to_int(str(file_row.get("create_object_count", 0)))
    unlink_count = to_int(str(file_row.get("unlink_count", 0)))
    rename_count = to_int(str(file_row.get("rename_count", 0)))
    modify_file_attr_count = to_int(str(file_row.get("modify_file_attr_count", 0)))
    network_active_process_count = to_int(str(process_row.get("network_active_process_count", 0)))
    network_active_process_ratio = to_number(str(process_row.get("network_active_process_ratio", 0)))

    hard_keep = 0
    soft_keep = 0
    keep = 0
    decision_reason = "drop_low_value_leaf"
    dropped_at_step = "drop_stage"

    if is_gt:
        hard_keep = 1
        keep = 1
        decision_reason = "gt_file"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_FILE" and execute_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "execute"
        dropped_at_step = "not_dropped"
    elif write_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "write"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_FILE" and create_object_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "create_object"
        dropped_at_step = "not_dropped"
    elif unlink_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "unlink"
        dropped_at_step = "not_dropped"
    elif rename_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "rename"
        dropped_at_step = "not_dropped"
    elif modify_file_attr_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "modify_file_attr"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_FILE" and total_accesses >= 3:
        soft_keep = 1
        keep = 1
        decision_reason = "total_accesses_ge_3_file"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_FILE" and unique_process_count >= 2:
        soft_keep = 1
        keep = 1
        decision_reason = "shared_by_multiple_processes_file"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_FILE" and network_active_process_count >= 1:
        soft_keep = 1
        keep = 1
        decision_reason = "network_active_process_file"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_DIR" and total_accesses >= 3:
        soft_keep = 1
        keep = 1
        decision_reason = "total_accesses_ge_3_dir"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_DIR" and unique_process_count >= 2:
        soft_keep = 1
        keep = 1
        decision_reason = "shared_by_multiple_processes_dir"
        dropped_at_step = "not_dropped"
    elif file_type == "FILE_OBJECT_UNIX_SOCKET" and total_accesses >= 5:
        soft_keep = 1
        keep = 1
        decision_reason = "total_accesses_ge_5_unix_socket"
        dropped_at_step = "not_dropped"

    return {
        "node_uuid": node_uuid,
        "file_type": file_type,
        "is_gt_file": int(is_gt),
        "hard_keep": hard_keep,
        "soft_keep": soft_keep,
        "keep": keep,
        "decision_reason": decision_reason,
        "dropped_at_step": dropped_at_step,
        "total_accesses": total_accesses,
        "unique_process_count": unique_process_count,
        "read_count": read_count,
        "write_count": write_count,
        "open_count": open_count,
        "execute_count": execute_count,
        "create_object_count": create_object_count,
        "unlink_count": unlink_count,
        "rename_count": rename_count,
        "modify_file_attr_count": modify_file_attr_count,
        "network_active_process_count": network_active_process_count,
        "network_active_process_ratio": f"{network_active_process_ratio:.6f}",
    }


def write_tsv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_window(window_name: str, decisions: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    kept = [row for row in decisions if row["keep"] == 1]
    dropped = [row for row in decisions if row["keep"] == 0]
    gt_candidates = [row for row in decisions if row["is_gt_file"] == 1]
    gt_kept = [row for row in kept if row["is_gt_file"] == 1]
    gt_dropped = [row for row in dropped if row["is_gt_file"] == 1]

    retained_by_reason = Counter(row["decision_reason"] for row in kept)
    dropped_by_reason = Counter(row["decision_reason"] for row in dropped)
    gt_drops_by_reason = Counter(row["decision_reason"] for row in gt_dropped)
    gt_drops_by_step = Counter(row["dropped_at_step"] for row in gt_dropped)
    dropped_by_step = Counter(row["dropped_at_step"] for row in dropped)

    dropped_gt_path = output_dir / "dropped_gt_file_list.tsv"
    write_tsv(
        dropped_gt_path,
        DROPPED_GT_COLUMNS,
        [
            {
                "window_name": window_name,
                **{key: row[key] for key in DROPPED_GT_COLUMNS if key not in {"window_name"}},
            }
            for row in gt_dropped
        ],
    )

    summary = {
        "window_name": window_name,
        "original_file_node_count": len(decisions),
        "retained_file_node_count": len(kept),
        "dropped_file_node_count": len(dropped),
        "file_retention_ratio": format_ratio(len(kept), len(decisions)),
        "gt_file_count_in_candidate_universe": len(gt_candidates),
        "gt_file_retained_count": len(gt_kept),
        "gt_file_dropped_count": len(gt_dropped),
        "gt_file_retention_ratio": format_ratio(len(gt_kept), len(gt_candidates)),
        "gt_file_drop_ratio": format_ratio(len(gt_dropped), len(gt_candidates)),
        "retained_file_count_by_reason": dict(retained_by_reason),
        "dropped_file_count_by_reason": dict(dropped_by_reason),
        "dropped_file_count_by_step": dict(dropped_by_step),
        "gt_drops_by_reason": dict(gt_drops_by_reason),
        "gt_drops_by_step": dict(gt_drops_by_step),
        "dropped_gt_uuid_list_path": str(dropped_gt_path),
        "dropped_gt_uuid_list": [row["node_uuid"] for row in gt_dropped],
    }
    return summary


def process_window(window_dir: Path, gt_files: Set[str], output_root: Path) -> Dict[str, object]:
    window_name = window_dir.name
    output_dir = output_root / window_name
    ensure_dir(output_dir)

    file_rows = load_tsv_by_uuid(window_dir / FILE_VIEW_FILE, FILE_NUMERIC_COLUMNS)
    process_rows = load_tsv_by_uuid(window_dir / PROCESS_VIEW_FILE, PROCESS_CONTEXT_NUMERIC_COLUMNS)

    decisions: List[Dict[str, object]] = []
    for node_uuid, file_row in file_rows.items():
        process_row = process_rows.get(node_uuid, {})
        decisions.append(decision_for_row(file_row, process_row, gt_files))

    decisions.sort(key=lambda row: row["node_uuid"])

    write_tsv(output_dir / "file_keep_decisions.tsv", DECISION_COLUMNS, decisions)
    write_tsv(output_dir / "file_keep_list.tsv", DECISION_COLUMNS, [row for row in decisions if row["keep"] == 1])
    write_tsv(output_dir / "file_drop_list.tsv", DECISION_COLUMNS, [row for row in decisions if row["keep"] == 0])

    summary = summarize_window(window_name, decisions, output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"[retention] {window_name}: original={summary['original_file_node_count']:,} "
        f"retained={summary['retained_file_node_count']:,} dropped={summary['dropped_file_node_count']:,} "
        f"gt_dropped={summary['gt_file_dropped_count']:,}"
    )
    return summary


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    gt_alignment_path = Path(args.gt_alignment).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    gt_files = load_gt_file_set(gt_alignment_path)
    window_dirs = sorted(path for path in feature_root.iterdir() if path.is_dir())

    summaries = [process_window(window_dir, gt_files, output_dir) for window_dir in window_dirs]

    manifest = {
        "feature_root": str(feature_root),
        "gt_alignment_path": str(gt_alignment_path),
        "windows": summaries,
        "notes": [
            "All process and network nodes are retained in this stage; only file nodes are filtered.",
            "Each filtering run reports GT loss explicitly.",
        ],
    }

    with (output_dir / "retention_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"[retention] wrote {output_dir / 'retention_manifest.json'}")


if __name__ == "__main__":
    main()
