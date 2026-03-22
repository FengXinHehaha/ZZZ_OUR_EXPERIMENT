import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


DEFAULT_FEATURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "features"
DEFAULT_FILE_RETENTION_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "retention"
DEFAULT_GT_ALIGNMENT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gt_entity_alignment" / "per_uuid.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "node_retention"

PROCESS_VIEW_PROCESS_FILE = "process_view__process_node.tsv"
NETWORK_VIEW_NETWORK_FILE = "network_view__network_node.tsv"
PROCESS_VIEW_NETWORK_FILE = "process_view__network_node.tsv"

PROCESS_NUMERIC_COLUMNS = [
    "has_parent_flag",
    "total_events",
    "event_type_diversity",
    "unique_file_count",
    "unique_network_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "connect_count",
    "send_count",
    "recv_count",
    "accept_count",
    "create_object_count",
    "fork_count",
    "mmap_count",
    "modify_process_count",
    "close_count",
    "file_interaction_ratio",
    "network_interaction_ratio",
    "fork_ratio",
]

NETWORK_NUMERIC_COLUMNS = [
    "total_net_events",
    "unique_process_count",
    "event_type_diversity",
    "connect_count",
    "accept_count",
    "bind_count",
    "send_count",
    "recv_count",
    "message_send_count",
    "message_recv_count",
    "close_count",
]

NETWORK_CONTEXT_NUMERIC_COLUMNS = [
    "total_process_context_events",
    "unique_process_count",
    "event_type_diversity",
    "connect_count",
    "accept_count",
    "bind_count",
    "send_count",
    "recv_count",
    "file_active_process_count",
    "avg_file_events_of_using_processes",
    "max_file_events_of_using_processes",
    "file_active_process_ratio",
]

PROCESS_DECISION_COLUMNS = [
    "node_uuid",
    "node_type",
    "is_gt_process",
    "hard_keep",
    "soft_keep",
    "keep",
    "decision_reason",
    "dropped_at_step",
    "has_parent_flag",
    "total_events",
    "unique_file_count",
    "unique_network_count",
    "read_count",
    "write_count",
    "open_count",
    "execute_count",
    "connect_count",
    "send_count",
    "recv_count",
    "accept_count",
    "create_object_count",
    "fork_count",
    "modify_process_count",
]

NETWORK_DECISION_COLUMNS = [
    "node_uuid",
    "node_type",
    "is_gt_network",
    "hard_keep",
    "soft_keep",
    "keep",
    "decision_reason",
    "dropped_at_step",
    "external_remote_ip_flag",
    "total_net_events",
    "total_process_context_events",
    "unique_process_count",
    "event_type_diversity",
    "connect_count",
    "accept_count",
    "bind_count",
    "send_count",
    "recv_count",
    "message_send_count",
    "message_recv_count",
    "close_count",
    "file_active_process_count",
]

GT_DROP_COLUMNS = [
    "node_uuid",
    "window_name",
    "entity_type",
    "decision_reason",
    "dropped_at_step",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply process/file/network node retention with GT-loss audit.")
    parser.add_argument(
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=f"Root directory containing extracted feature TSVs. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--file-retention-root",
        type=str,
        default=str(DEFAULT_FILE_RETENTION_ROOT),
        help=f"Directory containing prior file-retention outputs. Default: {DEFAULT_FILE_RETENTION_ROOT}",
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
        help=f"Directory for node-retention outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def to_number(value: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def to_int(value: object) -> int:
    return int(round(to_number(str(value))))


def format_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_gt_sets(path: Path) -> Dict[str, Set[str]]:
    gt_sets = {"process": set(), "file": set(), "network": set()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_type = row["entity_type"]
            if entity_type not in gt_sets:
                continue
            if row["visible_in_events"] == "yes":
                gt_sets[entity_type].add(row["gt_uuid"])
    return gt_sets


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


def write_tsv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def decision_for_process(row: Dict[str, object], gt_processes: Set[str]) -> Dict[str, object]:
    node_uuid = str(row["node_uuid"])
    is_gt = node_uuid in gt_processes

    total_events = to_int(row.get("total_events", 0))
    unique_file_count = to_int(row.get("unique_file_count", 0))
    unique_network_count = to_int(row.get("unique_network_count", 0))
    read_count = to_int(row.get("read_count", 0))
    write_count = to_int(row.get("write_count", 0))
    open_count = to_int(row.get("open_count", 0))
    execute_count = to_int(row.get("execute_count", 0))
    connect_count = to_int(row.get("connect_count", 0))
    send_count = to_int(row.get("send_count", 0))
    recv_count = to_int(row.get("recv_count", 0))
    accept_count = to_int(row.get("accept_count", 0))
    create_object_count = to_int(row.get("create_object_count", 0))
    fork_count = to_int(row.get("fork_count", 0))
    modify_process_count = to_int(row.get("modify_process_count", 0))
    has_parent_flag = to_int(row.get("has_parent_flag", 0))

    network_activity = connect_count + send_count + recv_count + accept_count

    hard_keep = 0
    soft_keep = 0
    keep = 0
    decision_reason = "drop_low_value_process"
    dropped_at_step = "drop_stage"

    if is_gt:
        hard_keep = 1
        keep = 1
        decision_reason = "gt_process"
        dropped_at_step = "not_dropped"
    elif execute_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "execute"
        dropped_at_step = "not_dropped"
    elif fork_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "fork"
        dropped_at_step = "not_dropped"
    elif modify_process_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "modify_process"
        dropped_at_step = "not_dropped"
    elif network_activity > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "network_activity"
        dropped_at_step = "not_dropped"
    elif write_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "write"
        dropped_at_step = "not_dropped"
    elif create_object_count > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "create_object"
        dropped_at_step = "not_dropped"
    elif total_events > 10:
        soft_keep = 1
        keep = 1
        decision_reason = "total_events_gt_10"
        dropped_at_step = "not_dropped"
    elif unique_file_count > 1:
        soft_keep = 1
        keep = 1
        decision_reason = "shared_file_context"
        dropped_at_step = "not_dropped"
    elif unique_network_count > 0:
        soft_keep = 1
        keep = 1
        decision_reason = "network_context"
        dropped_at_step = "not_dropped"
    elif has_parent_flag == 1 and total_events > 3:
        soft_keep = 1
        keep = 1
        decision_reason = "child_process_activity"
        dropped_at_step = "not_dropped"

    return {
        "node_uuid": node_uuid,
        "node_type": str(row.get("node_type", "process")),
        "is_gt_process": int(is_gt),
        "hard_keep": hard_keep,
        "soft_keep": soft_keep,
        "keep": keep,
        "decision_reason": decision_reason,
        "dropped_at_step": dropped_at_step,
        "has_parent_flag": has_parent_flag,
        "total_events": total_events,
        "unique_file_count": unique_file_count,
        "unique_network_count": unique_network_count,
        "read_count": read_count,
        "write_count": write_count,
        "open_count": open_count,
        "execute_count": execute_count,
        "connect_count": connect_count,
        "send_count": send_count,
        "recv_count": recv_count,
        "accept_count": accept_count,
        "create_object_count": create_object_count,
        "fork_count": fork_count,
        "modify_process_count": modify_process_count,
    }


def decision_for_network(
    network_row: Dict[str, object],
    process_context_row: Dict[str, object],
    gt_networks: Set[str],
) -> Dict[str, object]:
    node_uuid = str(network_row["node_uuid"])
    is_gt = node_uuid in gt_networks

    total_net_events = to_int(network_row.get("total_net_events", 0))
    total_process_context_events = to_int(process_context_row.get("total_process_context_events", total_net_events))
    unique_process_count = to_int(process_context_row.get("unique_process_count", network_row.get("unique_process_count", 0)))
    event_type_diversity = to_int(process_context_row.get("event_type_diversity", network_row.get("event_type_diversity", 0)))
    connect_count = to_int(process_context_row.get("connect_count", network_row.get("connect_count", 0)))
    accept_count = to_int(process_context_row.get("accept_count", network_row.get("accept_count", 0)))
    bind_count = to_int(process_context_row.get("bind_count", network_row.get("bind_count", 0)))
    send_count = to_int(process_context_row.get("send_count", network_row.get("send_count", 0)))
    recv_count = to_int(process_context_row.get("recv_count", network_row.get("recv_count", 0)))
    message_send_count = to_int(network_row.get("message_send_count", 0))
    message_recv_count = to_int(network_row.get("message_recv_count", 0))
    close_count = to_int(network_row.get("close_count", 0))
    file_active_process_count = to_int(process_context_row.get("file_active_process_count", 0))
    external_remote_ip_flag = str(network_row.get("external_remote_ip_flag", ""))

    conn_activity = connect_count + accept_count + bind_count
    io_activity = send_count + recv_count + message_send_count + message_recv_count

    hard_keep = 0
    soft_keep = 0
    keep = 0
    decision_reason = "drop_low_value_network"
    dropped_at_step = "drop_stage"

    if is_gt:
        hard_keep = 1
        keep = 1
        decision_reason = "gt_network"
        dropped_at_step = "not_dropped"
    elif conn_activity > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "conn_accept_bind"
        dropped_at_step = "not_dropped"
    elif io_activity > 0:
        hard_keep = 1
        keep = 1
        decision_reason = "io_activity"
        dropped_at_step = "not_dropped"
    elif unique_process_count >= 2:
        soft_keep = 1
        keep = 1
        decision_reason = "multi_process_context"
        dropped_at_step = "not_dropped"
    elif file_active_process_count >= 1:
        soft_keep = 1
        keep = 1
        decision_reason = "file_active_process_context"
        dropped_at_step = "not_dropped"
    elif total_process_context_events > 3:
        soft_keep = 1
        keep = 1
        decision_reason = "context_events_gt_3"
        dropped_at_step = "not_dropped"

    return {
        "node_uuid": node_uuid,
        "node_type": str(network_row.get("node_type", "network")),
        "is_gt_network": int(is_gt),
        "hard_keep": hard_keep,
        "soft_keep": soft_keep,
        "keep": keep,
        "decision_reason": decision_reason,
        "dropped_at_step": dropped_at_step,
        "external_remote_ip_flag": external_remote_ip_flag,
        "total_net_events": total_net_events,
        "total_process_context_events": total_process_context_events,
        "unique_process_count": unique_process_count,
        "event_type_diversity": event_type_diversity,
        "connect_count": connect_count,
        "accept_count": accept_count,
        "bind_count": bind_count,
        "send_count": send_count,
        "recv_count": recv_count,
        "message_send_count": message_send_count,
        "message_recv_count": message_recv_count,
        "close_count": close_count,
        "file_active_process_count": file_active_process_count,
    }


def summarize_type(
    decisions: List[Dict[str, object]],
    entity_label: str,
    gt_flag_column: str,
    output_dir: Path,
    window_name: str,
) -> Dict[str, object]:
    kept = [row for row in decisions if row["keep"] == 1]
    dropped = [row for row in decisions if row["keep"] == 0]
    gt_candidates = [row for row in decisions if row[gt_flag_column] == 1]
    gt_kept = [row for row in kept if row[gt_flag_column] == 1]
    gt_dropped = [row for row in dropped if row[gt_flag_column] == 1]

    retained_by_reason = Counter(row["decision_reason"] for row in kept)
    dropped_by_reason = Counter(row["decision_reason"] for row in dropped)
    gt_drops_by_reason = Counter(row["decision_reason"] for row in gt_dropped)
    gt_drops_by_step = Counter(row["dropped_at_step"] for row in gt_dropped)
    dropped_by_step = Counter(row["dropped_at_step"] for row in dropped)

    dropped_gt_path = output_dir / f"dropped_gt_{entity_label}_list.tsv"
    write_tsv(
        dropped_gt_path,
        GT_DROP_COLUMNS,
        [
            {
                "node_uuid": row["node_uuid"],
                "window_name": window_name,
                "entity_type": entity_label,
                "decision_reason": row["decision_reason"],
                "dropped_at_step": row["dropped_at_step"],
            }
            for row in gt_dropped
        ],
    )

    prefix = f"{entity_label}_"
    return {
        f"original_{entity_label}_node_count": len(decisions),
        f"retained_{entity_label}_node_count": len(kept),
        f"dropped_{entity_label}_node_count": len(dropped),
        f"{entity_label}_retention_ratio": format_ratio(len(kept), len(decisions)),
        f"gt_{entity_label}_count_in_candidate_universe": len(gt_candidates),
        f"gt_{entity_label}_retained_count": len(gt_kept),
        f"gt_{entity_label}_dropped_count": len(gt_dropped),
        f"gt_{entity_label}_retention_ratio": format_ratio(len(gt_kept), len(gt_candidates)),
        f"gt_{entity_label}_drop_ratio": format_ratio(len(gt_dropped), len(gt_candidates)),
        f"retained_{entity_label}_count_by_reason": dict(retained_by_reason),
        f"dropped_{entity_label}_count_by_reason": dict(dropped_by_reason),
        f"dropped_{entity_label}_count_by_step": dict(dropped_by_step),
        f"gt_{entity_label}_drops_by_reason": dict(gt_drops_by_reason),
        f"gt_{entity_label}_drops_by_step": dict(gt_drops_by_step),
        f"dropped_gt_{entity_label}_uuid_list_path": str(dropped_gt_path),
        f"dropped_gt_{entity_label}_uuid_list": [row["node_uuid"] for row in gt_dropped],
    }


def copy_file_retention_outputs(file_retention_root: Path, output_root: Path, window_name: str) -> Dict[str, object]:
    source_dir = file_retention_root / window_name
    target_dir = output_root / window_name
    ensure_dir(target_dir)

    for filename in [
        "file_keep_decisions.tsv",
        "file_keep_list.tsv",
        "file_drop_list.tsv",
        "dropped_gt_file_list.tsv",
        "summary.json",
    ]:
        shutil.copy2(source_dir / filename, target_dir / filename)

    with (source_dir / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_window(
    window_dir: Path,
    file_retention_root: Path,
    gt_sets: Dict[str, Set[str]],
    output_root: Path,
) -> Dict[str, object]:
    window_name = window_dir.name
    output_dir = output_root / window_name
    ensure_dir(output_dir)

    file_summary = copy_file_retention_outputs(file_retention_root, output_root, window_name)

    process_rows = load_tsv_by_uuid(window_dir / PROCESS_VIEW_PROCESS_FILE, PROCESS_NUMERIC_COLUMNS)
    network_rows = load_tsv_by_uuid(window_dir / NETWORK_VIEW_NETWORK_FILE, NETWORK_NUMERIC_COLUMNS)
    network_context_rows = load_tsv_by_uuid(window_dir / PROCESS_VIEW_NETWORK_FILE, NETWORK_CONTEXT_NUMERIC_COLUMNS)

    process_decisions = [decision_for_process(row, gt_sets["process"]) for row in process_rows.values()]
    process_decisions.sort(key=lambda row: row["node_uuid"])
    write_tsv(output_dir / "process_keep_decisions.tsv", PROCESS_DECISION_COLUMNS, process_decisions)
    write_tsv(
        output_dir / "process_keep_list.tsv",
        PROCESS_DECISION_COLUMNS,
        [row for row in process_decisions if row["keep"] == 1],
    )
    write_tsv(
        output_dir / "process_drop_list.tsv",
        PROCESS_DECISION_COLUMNS,
        [row for row in process_decisions if row["keep"] == 0],
    )
    process_summary = summarize_type(process_decisions, "process", "is_gt_process", output_dir, window_name)

    network_decisions = [
        decision_for_network(row, network_context_rows.get(node_uuid, {}), gt_sets["network"])
        for node_uuid, row in network_rows.items()
    ]
    network_decisions.sort(key=lambda row: row["node_uuid"])
    write_tsv(output_dir / "network_keep_decisions.tsv", NETWORK_DECISION_COLUMNS, network_decisions)
    write_tsv(
        output_dir / "network_keep_list.tsv",
        NETWORK_DECISION_COLUMNS,
        [row for row in network_decisions if row["keep"] == 1],
    )
    write_tsv(
        output_dir / "network_drop_list.tsv",
        NETWORK_DECISION_COLUMNS,
        [row for row in network_decisions if row["keep"] == 0],
    )
    network_summary = summarize_type(network_decisions, "network", "is_gt_network", output_dir, window_name)

    summary = {
        "window_name": window_name,
        **file_summary,
        **process_summary,
        **network_summary,
        "original_total_node_count": (
            file_summary["original_file_node_count"]
            + process_summary["original_process_node_count"]
            + network_summary["original_network_node_count"]
        ),
        "retained_total_node_count": (
            file_summary["retained_file_node_count"]
            + process_summary["retained_process_node_count"]
            + network_summary["retained_network_node_count"]
        ),
        "dropped_total_node_count": (
            file_summary["dropped_file_node_count"]
            + process_summary["dropped_process_node_count"]
            + network_summary["dropped_network_node_count"]
        ),
        "total_node_retention_ratio": format_ratio(
            (
                file_summary["retained_file_node_count"]
                + process_summary["retained_process_node_count"]
                + network_summary["retained_network_node_count"]
            ),
            (
                file_summary["original_file_node_count"]
                + process_summary["original_process_node_count"]
                + network_summary["original_network_node_count"]
            ),
        ),
        "total_gt_dropped_count": (
            file_summary["gt_file_dropped_count"]
            + process_summary["gt_process_dropped_count"]
            + network_summary["gt_network_dropped_count"]
        ),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"[node-retention] {window_name}: total={summary['original_total_node_count']:,} "
        f"retained={summary['retained_total_node_count']:,} dropped={summary['dropped_total_node_count']:,} "
        f"gt_dropped={summary['total_gt_dropped_count']:,}"
    )
    return summary


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    file_retention_root = Path(args.file_retention_root).expanduser().resolve()
    gt_alignment_path = Path(args.gt_alignment).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    gt_sets = load_gt_sets(gt_alignment_path)
    window_dirs = sorted(path for path in feature_root.iterdir() if path.is_dir())
    summaries = [process_window(window_dir, file_retention_root, gt_sets, output_dir) for window_dir in window_dirs]

    manifest = {
        "feature_root": str(feature_root),
        "file_retention_root": str(file_retention_root),
        "gt_alignment_path": str(gt_alignment_path),
        "windows": summaries,
        "notes": [
            "File-node decisions are inherited from the prior file-retention stage.",
            "Process and network nodes are filtered conservatively with GT-loss audit enabled.",
            "Every filtering stage reports dropped GT UUIDs explicitly.",
        ],
    }

    with (output_dir / "retention_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"[node-retention] wrote {output_dir / 'retention_manifest.json'}")


if __name__ == "__main__":
    main()
