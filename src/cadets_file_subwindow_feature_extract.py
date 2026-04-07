import argparse
from pathlib import Path
from typing import Dict, List

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.total = total
            self.desc = desc

        def __enter__(self):
            if self.desc:
                print(f"[{self.desc}] progress start", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            return None

from cadets_event_ingest_common import build_common_parser, connect_target_db, load_ingest_config
from cadets_feature_extract import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    copy_query_to_tsv,
    ensure_output_dir,
    fetch_event_count,
    file_path_condition,
    load_split_manifest,
    log,
    ns_to_text,
    print_summary,
    tune_feature_session,
    window_bounds,
    write_json,
    build_feature_windows,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features_file_subwindow_6h"
FILE_SUBWINDOW_FILE = "file_subwindow__file_node.tsv"


RISKY_PATH_CONDITION = " OR ".join(
    [
        f"({file_path_condition('h.path_text', 'temp')})",
        f"({file_path_condition('h.path_text', 'hidden')})",
        f"({file_path_condition('h.path_text', 'script')})",
        f"({file_path_condition('h.path_text', 'system_bin')})",
    ]
)


FILE_SUBWINDOW_QUERY = f"""
WITH file_hits AS (
    SELECT
        e.subject_uuid AS process_uuid,
        e.object_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text,
        FLOOR((e.timestamp_ns - %s)::numeric / %s)::bigint AS subwindow_idx
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
    UNION ALL
    SELECT
        e.subject_uuid AS process_uuid,
        e.object2_uuid AS node_uuid,
        e.event_type,
        COALESCE(
            NULLIF(BTRIM(COALESCE(e.object2_path, '')), ''),
            NULLIF(BTRIM(COALESCE(e.file_descriptor, '')), ''),
            ''
        ) AS path_text,
        FLOOR((e.timestamp_ns - %s)::numeric / %s)::bigint AS subwindow_idx
    FROM events_raw AS e
    JOIN file_entities AS f
      ON f.uuid = e.object2_uuid
    WHERE e.timestamp_ns >= %s
      AND e.timestamp_ns < %s
),
bucket_stats AS (
    SELECT
        h.node_uuid,
        h.subwindow_idx,
        COUNT(*)::bigint AS accesses,
        COUNT(DISTINCT h.process_uuid)::bigint AS unique_process_count,
        COUNT(*) FILTER (WHERE h.event_type = 'EVENT_READ')::bigint AS read_count,
        COUNT(*) FILTER (WHERE h.event_type = 'EVENT_WRITE')::bigint AS write_count,
        COUNT(*) FILTER (WHERE h.event_type = 'EVENT_EXECUTE')::bigint AS execute_count,
        COUNT(*) FILTER (WHERE {RISKY_PATH_CONDITION})::bigint AS risky_path_access_count
    FROM file_hits AS h
    GROUP BY h.node_uuid, h.subwindow_idx
),
ranked_buckets AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.node_uuid
            ORDER BY b.accesses DESC, b.unique_process_count DESC, b.subwindow_idx ASC
        ) AS bucket_rank
    FROM bucket_stats AS b
),
aggregate_stats AS (
    SELECT
        b.node_uuid,
        COUNT(*)::bigint AS active_subwindow_count,
        MIN(b.subwindow_idx)::bigint AS first_active_subwindow,
        MAX(b.subwindow_idx)::bigint AS last_active_subwindow,
        (MAX(b.subwindow_idx) - MIN(b.subwindow_idx) + 1)::bigint AS active_subwindow_span,
        MAX(b.accesses)::bigint AS max_subwindow_accesses,
        ROUND(AVG(b.accesses)::numeric, 6) AS mean_active_subwindow_accesses,
        ROUND(COALESCE(STDDEV_SAMP(b.accesses), 0)::numeric, 6) AS std_active_subwindow_accesses,
        ROUND(MAX(b.accesses)::numeric / NULLIF(SUM(b.accesses), 0), 6) AS max_subwindow_ratio,
        ROUND(COUNT(*)::numeric / %s, 6) AS active_subwindow_ratio,
        COUNT(*) FILTER (WHERE b.write_count > 0)::bigint AS write_active_subwindow_count,
        COUNT(*) FILTER (WHERE b.execute_count > 0)::bigint AS execute_active_subwindow_count,
        COUNT(*) FILTER (WHERE b.risky_path_access_count > 0)::bigint AS risky_path_active_subwindow_count,
        SUM(b.accesses)::bigint AS total_accesses_from_subwindows
    FROM bucket_stats AS b
    GROUP BY b.node_uuid
)
SELECT
    a.node_uuid,
    'file'::text AS node_type,
    COALESCE(f.host_id, '') AS host_id,
    COALESCE(f.file_type, '') AS file_type,
    COALESCE(f.size_bytes, 0)::bigint AS size_bytes,
    %s::bigint AS subwindow_hours,
    %s::bigint AS total_subwindows,
    a.active_subwindow_count,
    a.first_active_subwindow,
    a.last_active_subwindow,
    a.active_subwindow_span,
    a.max_subwindow_accesses,
    a.mean_active_subwindow_accesses,
    a.std_active_subwindow_accesses,
    a.max_subwindow_ratio,
    a.active_subwindow_ratio,
    rb.subwindow_idx::bigint AS peak_subwindow_index,
    rb.accesses::bigint AS peak_subwindow_accesses,
    rb.unique_process_count::bigint AS peak_subwindow_unique_process_count,
    rb.read_count::bigint AS peak_subwindow_read_count,
    rb.write_count::bigint AS peak_subwindow_write_count,
    rb.execute_count::bigint AS peak_subwindow_execute_count,
    a.write_active_subwindow_count,
    a.execute_active_subwindow_count,
    a.risky_path_active_subwindow_count,
    a.total_accesses_from_subwindows
FROM aggregate_stats AS a
JOIN ranked_buckets AS rb
  ON rb.node_uuid = a.node_uuid
 AND rb.bucket_rank = 1
JOIN file_entities AS f
  ON f.uuid = a.node_uuid
ORDER BY a.node_uuid
"""


def filter_windows(windows: List[Dict[str, object]], selected_names: List[str]) -> List[Dict[str, object]]:
    if not selected_names:
        return windows
    name_set = set(selected_names)
    filtered = [window for window in windows if str(window["name"]) in name_set]
    missing = sorted(name_set - {str(window["name"]) for window in filtered})
    if missing:
        raise ValueError(f"Unknown window names: {missing}")
    return filtered


def export_window_subwindow_features(
    conn,
    window: Dict[str, object],
    output_dir: Path,
    subwindow_hours: int,
) -> Dict[str, object]:
    days = list(window["days"])
    start_ns, end_ns = window_bounds(days)
    window_dir = output_dir / str(window["name"])
    ensure_output_dir(window_dir)

    subwindow_ns = subwindow_hours * 60 * 60 * 1_000_000_000
    total_subwindows = max(1, (end_ns - start_ns) // subwindow_ns)

    log(
        f"[file-subwindow-extract] {window['name']}: "
        f"preparing {days[0]} -> {days[-1]} subwindow_hours={subwindow_hours}"
    )
    event_count = fetch_event_count(conn, start_ns, end_ns)
    log(f"[file-subwindow-extract] {window['name']}: event_count={event_count:,}")

    row_count = copy_query_to_tsv(
        conn,
        FILE_SUBWINDOW_QUERY,
        (
            start_ns,
            subwindow_ns,
            start_ns,
            end_ns,
            start_ns,
            subwindow_ns,
            start_ns,
            end_ns,
            total_subwindows,
            subwindow_hours,
            total_subwindows,
        ),
        window_dir / FILE_SUBWINDOW_FILE,
        log_label=f"{window['name']} file_subwindow__file_node",
    )

    metadata = {
        "window_name": window["name"],
        "split": window["split"],
        "days": days,
        "subwindow_hours": subwindow_hours,
        "total_subwindows": total_subwindows,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "start_time_utc": ns_to_text(start_ns),
        "end_time_utc_exclusive": ns_to_text(end_ns),
        "event_count": event_count,
        "file_subwindow_rows": row_count,
        "feature_files": {
            "file_subwindow__file_node": FILE_SUBWINDOW_FILE,
        },
    }
    write_json(window_dir / "metadata.json", metadata)
    log(f"[file-subwindow-extract] {window['name']}: done rows={row_count:,}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Extract CPU-side 6h subwindow file burst features from CADETS.")
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=str(DEFAULT_SPLIT_MANIFEST_PATH),
        help=f"Path to split_manifest.json. Default: {DEFAULT_SPLIT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for subwindow file features. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--subwindow-hours",
        type=int,
        default=6,
        help="Subwindow width in hours. Default: 6",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Optional window name to export. Can be repeated, e.g. --window val --window test_2018-04-13",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print selected windows without exporting TSV feature files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.subwindow_hours <= 0 or 24 % args.subwindow_hours != 0:
        raise ValueError("--subwindow-hours must be a positive divisor of 24")

    split_manifest_path = Path(args.split_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    split_manifest = load_split_manifest(split_manifest_path)
    windows = filter_windows(build_feature_windows(split_manifest), list(args.window))

    if args.summary_only:
        window_metadata = [
            {
                "window_name": window["name"],
                "split": window["split"],
                "days": list(window["days"]),
                "subwindow_hours": args.subwindow_hours,
            }
            for window in windows
        ]
        print_summary(window_metadata)
        return

    config = load_ingest_config(args)
    conn = connect_target_db(config.database)
    try:
        tune_feature_session(conn)
        window_metadata = []
        with tqdm(total=len(windows), desc="file subwindow windows", unit="window", dynamic_ncols=True) as bar:
            for window in windows:
                metadata = export_window_subwindow_features(
                    conn=conn,
                    window=window,
                    output_dir=output_dir,
                    subwindow_hours=args.subwindow_hours,
                )
                window_metadata.append(metadata)
                bar.update(1)

        print_summary(window_metadata)
        write_json(
            output_dir / "feature_manifest.json",
            {
                "split_manifest_path": str(split_manifest_path),
                "subwindow_hours": args.subwindow_hours,
                "windows": window_metadata,
                "notes": [
                    "This stage exports a file_subwindow__file_node table with burstiness-oriented within-day features.",
                    "It is intended for CPU-side file screening and learned reranking experiments.",
                ],
            },
        )
        print(f"[file-subwindow-extract] outputs written to: {output_dir}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
