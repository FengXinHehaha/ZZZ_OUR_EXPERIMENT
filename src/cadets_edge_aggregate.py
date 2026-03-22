import argparse
import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from cadets_event_ingest_common import build_common_parser, connect_target_db, load_ingest_config


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features" / "feature_manifest.json"
DEFAULT_RETENTION_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "retention" / "retention_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "edges"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ns_to_text(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Aggregate CADETS event-level edges into window-level typed edges.")
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=str(DEFAULT_FEATURE_MANIFEST),
        help=f"Path to feature_manifest.json. Default: {DEFAULT_FEATURE_MANIFEST}",
    )
    parser.add_argument(
        "--retention-manifest",
        type=str,
        default=str(DEFAULT_RETENTION_MANIFEST),
        help=f"Path to retention_manifest.json. Default: {DEFAULT_RETENTION_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for aggregated edges. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def tune_session(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SET work_mem TO '1GB'")
        cursor.execute("SET maintenance_work_mem TO '1GB'")
        cursor.execute("SET max_parallel_workers_per_gather TO 4")
        cursor.execute("SET jit TO OFF")
    conn.commit()


def read_uuid_list(path: Path) -> List[str]:
    values: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            values.append(row["node_uuid"])
    return values


def load_retained_nodes(feature_root: Path, retention_root: Path, window_name: str) -> Dict[str, List[str]]:
    return {
        "process": read_uuid_list(feature_root / window_name / "process_view__process_node.tsv"),
        "file": read_uuid_list(retention_root / window_name / "file_keep_list.tsv"),
        "network": read_uuid_list(feature_root / window_name / "network_view__network_node.tsv"),
    }


def load_retention_map(retention_manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {item["window_name"]: item for item in retention_manifest["windows"]}


def write_temp_retained_nodes(conn, node_lists: Dict[str, List[str]]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS tmp_retained_nodes")
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_retained_nodes (
                uuid VARCHAR(255) PRIMARY KEY,
                node_type TEXT NOT NULL
            )
            """
        )
        buffer = StringIO()
        for node_type, values in node_lists.items():
            for uuid in values:
                buffer.write(f"{uuid}\t{node_type}\n")
        buffer.seek(0)
        cursor.copy_from(buffer, "tmp_retained_nodes", columns=("uuid", "node_type"))
    conn.commit()


EDGE_AGG_QUERY = """
WITH edge_events AS (
    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object_uuid IS NOT NULL
      AND e.object_uuid <> ''

    UNION ALL

    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object2_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object2_uuid IS NOT NULL
      AND e.object2_uuid <> ''
)
SELECT
    src_uuid,
    src_type,
    dst_uuid,
    dst_type,
    event_type,
    COUNT(*)::bigint AS event_count,
    MIN(timestamp_ns)::bigint AS first_timestamp_ns,
    MAX(timestamp_ns)::bigint AS last_timestamp_ns
FROM edge_events
GROUP BY
    src_uuid,
    src_type,
    dst_uuid,
    dst_type,
    event_type
"""


EDGE_SUMMARY_QUERY = """
WITH edge_events AS (
    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object_uuid IS NOT NULL
      AND e.object_uuid <> ''

    UNION ALL

    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object2_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object2_uuid IS NOT NULL
      AND e.object2_uuid <> ''
)
SELECT
    COUNT(*)::bigint AS raw_edge_events,
    COUNT(DISTINCT (src_uuid, dst_uuid))::bigint AS unique_untyped_edges,
    COUNT(DISTINCT (src_uuid, dst_uuid, event_type))::bigint AS unique_typed_edges
FROM edge_events
"""


EDGE_BY_DST_TYPE_QUERY = """
WITH edge_events AS (
    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object_uuid IS NOT NULL
      AND e.object_uuid <> ''

    UNION ALL

    SELECT
        e.subject_uuid AS src_uuid,
        s.node_type AS src_type,
        e.object2_uuid AS dst_uuid,
        t.node_type AS dst_type,
        e.event_type,
        e.timestamp_ns
    FROM events_raw AS e
    JOIN tmp_retained_nodes AS s
      ON s.uuid = e.subject_uuid
    JOIN tmp_retained_nodes AS t
      ON t.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
      AND e.object2_uuid IS NOT NULL
      AND e.object2_uuid <> ''
)
SELECT
    dst_type,
    COUNT(*)::bigint AS raw_edge_events,
    COUNT(DISTINCT (src_uuid, dst_uuid, event_type))::bigint AS unique_typed_edges
FROM edge_events
GROUP BY dst_type
ORDER BY dst_type
"""


def count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def export_window_edges(conn, output_path: Path, start_ns: int, end_ns: int) -> int:
    with conn.cursor() as cursor:
        rendered = cursor.mogrify(EDGE_AGG_QUERY, (start_ns, end_ns, start_ns, end_ns)).decode("utf-8")
        copy_sql = "COPY (" + rendered + ") TO STDOUT WITH (FORMAT CSV, HEADER TRUE, DELIMITER E'\\t')"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            cursor.copy_expert(copy_sql, handle)
            handle.flush()
    return count_tsv_rows(output_path)


def fetch_summary(conn, start_ns: int, end_ns: int) -> Dict[str, object]:
    with conn.cursor() as cursor:
        cursor.execute(EDGE_SUMMARY_QUERY, (start_ns, end_ns, start_ns, end_ns))
        raw_edge_events, unique_untyped_edges, unique_typed_edges = cursor.fetchone()

        cursor.execute(EDGE_BY_DST_TYPE_QUERY, (start_ns, end_ns, start_ns, end_ns))
        by_dst = {
            row[0]: {
                "raw_edge_events": int(row[1]),
                "unique_typed_edges": int(row[2]),
            }
            for row in cursor.fetchall()
        }

    return {
        "raw_edge_events": int(raw_edge_events),
        "unique_untyped_edges": int(unique_untyped_edges),
        "unique_typed_edges": int(unique_typed_edges),
        "by_dst_type": by_dst,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_ingest_config(args)

    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    retention_manifest_path = Path(args.retention_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    feature_manifest = load_json(feature_manifest_path)
    retention_manifest = load_json(retention_manifest_path)
    retention_map = load_retention_map(retention_manifest)

    feature_root = feature_manifest_path.parent
    retention_root = retention_manifest_path.parent

    conn = connect_target_db(config.database)
    try:
        tune_session(conn)
        window_summaries = []
        for window in tqdm(feature_manifest["windows"], desc="edge windows", unit="window", dynamic_ncols=True):
            name = window["window_name"]
            start_ns = int(window["start_ns"])
            end_ns = int(window["end_ns"])

            node_lists = load_retained_nodes(feature_root, retention_root, name)
            write_temp_retained_nodes(conn, node_lists)

            window_dir = output_dir / name
            ensure_dir(window_dir)
            edge_path = window_dir / "typed_edges.tsv"

            print(f"[edge-aggregate] {name}: exporting typed edges", flush=True)
            typed_edge_rows = export_window_edges(conn, edge_path, start_ns, end_ns)
            summary = fetch_summary(conn, start_ns, end_ns)

            metadata = {
                "window_name": name,
                "days": list(window["days"]),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "start_time_utc": ns_to_text(start_ns),
                "end_time_utc_exclusive": ns_to_text(end_ns),
                "process_node_count": len(node_lists["process"]),
                "retained_file_node_count": len(node_lists["file"]),
                "network_node_count": len(node_lists["network"]),
                "total_node_count": len(node_lists["process"]) + len(node_lists["file"]) + len(node_lists["network"]),
                "typed_edge_rows": typed_edge_rows,
                "raw_edge_events": summary["raw_edge_events"],
                "unique_untyped_edges": summary["unique_untyped_edges"],
                "unique_typed_edges": summary["unique_typed_edges"],
                "retention_file_ratio": retention_map[name]["file_retention_ratio"],
                "by_dst_type": summary["by_dst_type"],
                "edge_file": "typed_edges.tsv",
            }

            with (window_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
                handle.write("\n")

            print(
                f"[edge-aggregate] {name}: nodes={metadata['total_node_count']:,} "
                f"raw_events={metadata['raw_edge_events']:,} "
                f"typed_edges={metadata['unique_typed_edges']:,}",
                flush=True,
            )
            window_summaries.append(metadata)
            conn.commit()

        manifest = {
            "feature_manifest_path": str(feature_manifest_path),
            "retention_manifest_path": str(retention_manifest_path),
            "windows": window_summaries,
            "notes": [
                "Edges are aggregated by (src_uuid, dst_uuid, event_type, window).",
                "Each edge keeps event_count, first_timestamp_ns, and last_timestamp_ns.",
                "Only edges whose endpoints survive node retention are exported.",
            ],
        }
        with (output_dir / "edge_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"[edge-aggregate] outputs written to: {output_dir}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
