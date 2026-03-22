import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

from cadets_event_ingest_common import (
    connect_target_db,
    load_ingest_config,
    timestamp_to_text,
)


DEFAULT_GT_PATH = Path("/home/fxh/DeepSeek/26.03.17 version/cadets_ground_truth.txt")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "gt_alignment"


def load_ground_truth(path: Path) -> Tuple[List[str], int]:
    uuids: List[str] = []
    seen = set()
    duplicate_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            gt_uuid = raw_line.strip().upper()
            if not gt_uuid:
                continue
            if gt_uuid in seen:
                duplicate_lines += 1
                continue
            seen.add(gt_uuid)
            uuids.append(gt_uuid)
    return uuids, duplicate_lines


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_gt_into_temp_table(conn, gt_uuids: List[str]) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS gt_input")
        cursor.execute(
            """
            CREATE TEMP TABLE gt_input (
                gt_uuid VARCHAR(255) PRIMARY KEY
            )
            """
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for gt_uuid in gt_uuids:
        writer.writerow([gt_uuid])
    buffer.seek(0)

    with conn.cursor() as cursor:
        cursor.copy_expert(
            """
            COPY gt_input (gt_uuid)
            FROM STDIN WITH (FORMAT CSV)
            """,
            buffer,
        )
        cursor.execute("ANALYZE gt_input")
        cursor.execute("SET work_mem TO '512MB'")
    conn.commit()


def build_alignment_temp_tables(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS gt_role_hits")
        cursor.execute(
            """
            CREATE TEMP TABLE gt_role_hits AS
            SELECT
                g.gt_uuid,
                'subject'::text AS role,
                COUNT(*)::bigint AS event_hits,
                MIN(e.timestamp_ns) AS first_seen_ns,
                MAX(e.timestamp_ns) AS last_seen_ns,
                COUNT(*) FILTER (WHERE e.exec_name IS NOT NULL) AS named_event_hits,
                COUNT(*) FILTER (WHERE e.object_path IS NOT NULL) AS path_event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.subject_uuid = g.gt_uuid
            GROUP BY g.gt_uuid

            UNION ALL

            SELECT
                g.gt_uuid,
                'object'::text AS role,
                COUNT(*)::bigint AS event_hits,
                MIN(e.timestamp_ns) AS first_seen_ns,
                MAX(e.timestamp_ns) AS last_seen_ns,
                COUNT(*) FILTER (WHERE e.exec_name IS NOT NULL) AS named_event_hits,
                COUNT(*) FILTER (WHERE e.object_path IS NOT NULL) AS path_event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.object_uuid = g.gt_uuid
            GROUP BY g.gt_uuid

            UNION ALL

            SELECT
                g.gt_uuid,
                'object2'::text AS role,
                COUNT(*)::bigint AS event_hits,
                MIN(e.timestamp_ns) AS first_seen_ns,
                MAX(e.timestamp_ns) AS last_seen_ns,
                COUNT(*) FILTER (WHERE e.exec_name IS NOT NULL) AS named_event_hits,
                COUNT(*) FILTER (WHERE e.object2_path IS NOT NULL) AS path_event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.object2_uuid = g.gt_uuid
            GROUP BY g.gt_uuid
            """
        )
        cursor.execute("CREATE INDEX gt_role_hits_uuid_idx ON gt_role_hits (gt_uuid)")

        cursor.execute("DROP TABLE IF EXISTS gt_daily_hits")
        cursor.execute(
            """
            CREATE TEMP TABLE gt_daily_hits AS
            SELECT
                g.gt_uuid,
                'subject'::text AS role,
                to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                COUNT(*)::bigint AS event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.subject_uuid = g.gt_uuid
            GROUP BY g.gt_uuid, day

            UNION ALL

            SELECT
                g.gt_uuid,
                'object'::text AS role,
                to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                COUNT(*)::bigint AS event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.object_uuid = g.gt_uuid
            GROUP BY g.gt_uuid, day

            UNION ALL

            SELECT
                g.gt_uuid,
                'object2'::text AS role,
                to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                COUNT(*)::bigint AS event_hits
            FROM gt_input AS g
            JOIN events_raw AS e
              ON e.object2_uuid = g.gt_uuid
            GROUP BY g.gt_uuid, day
            """
        )
        cursor.execute("CREATE INDEX gt_daily_hits_uuid_idx ON gt_daily_hits (gt_uuid)")
    conn.commit()


def classify_role_pattern(subject_hits: int, object_hits: int, object2_hits: int) -> str:
    roles = []
    if subject_hits > 0:
        roles.append("subject")
    if object_hits > 0:
        roles.append("object")
    if object2_hits > 0:
        roles.append("object2")
    if not roles:
        return "unseen"
    return "+".join(roles)


def fetch_alignment_rows(conn) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM gt_input")
        total_gt_uuids = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT
                role,
                COUNT(*)::bigint AS hit_uuid_count,
                SUM(event_hits)::bigint AS event_hits,
                MIN(first_seen_ns),
                MAX(last_seen_ns),
                SUM(named_event_hits)::bigint,
                SUM(path_event_hits)::bigint
            FROM gt_role_hits
            GROUP BY role
            ORDER BY role
            """
        )
        role_summary = []
        for row in cursor.fetchall():
            role_summary.append(
                {
                    "role": row[0],
                    "hit_uuid_count": row[1],
                    "event_hits": row[2],
                    "first_seen": timestamp_to_text(row[3]),
                    "last_seen": timestamp_to_text(row[4]),
                    "named_event_hits": row[5],
                    "path_event_hits": row[6],
                }
            )

        cursor.execute(
            """
            SELECT
                day,
                COUNT(DISTINCT gt_uuid)::bigint AS gt_uuid_count,
                SUM(event_hits)::bigint AS total_event_hits,
                COUNT(DISTINCT gt_uuid) FILTER (WHERE role = 'subject')::bigint AS subject_gt_count,
                COUNT(DISTINCT gt_uuid) FILTER (WHERE role = 'object')::bigint AS object_gt_count,
                COUNT(DISTINCT gt_uuid) FILTER (WHERE role = 'object2')::bigint AS object2_gt_count
            FROM gt_daily_hits
            GROUP BY day
            ORDER BY day
            """
        )
        daily_summary = []
        for row in cursor.fetchall():
            daily_summary.append(
                {
                    "day": row[0],
                    "gt_uuid_count": row[1],
                    "total_event_hits": row[2],
                    "subject_gt_count": row[3],
                    "object_gt_count": row[4],
                    "object2_gt_count": row[5],
                }
            )

        cursor.execute(
            """
            WITH distinct_days AS (
                SELECT DISTINCT gt_uuid, day
                FROM gt_daily_hits
            ),
            day_strings AS (
                SELECT gt_uuid, string_agg(day, ',' ORDER BY day) AS active_days
                FROM distinct_days
                GROUP BY gt_uuid
            )
            SELECT
                g.gt_uuid,
                COALESCE(SUM(r.event_hits), 0)::bigint AS total_event_hits,
                COALESCE(SUM(CASE WHEN r.role = 'subject' THEN r.event_hits ELSE 0 END), 0)::bigint AS subject_hits,
                COALESCE(SUM(CASE WHEN r.role = 'object' THEN r.event_hits ELSE 0 END), 0)::bigint AS object_hits,
                COALESCE(SUM(CASE WHEN r.role = 'object2' THEN r.event_hits ELSE 0 END), 0)::bigint AS object2_hits,
                COALESCE(SUM(CASE WHEN r.role = 'subject' THEN r.named_event_hits ELSE 0 END), 0)::bigint AS subject_named_hits,
                COALESCE(SUM(CASE WHEN r.role = 'object' THEN r.path_event_hits ELSE 0 END), 0)::bigint AS object_path_hits,
                COALESCE(SUM(CASE WHEN r.role = 'object2' THEN r.path_event_hits ELSE 0 END), 0)::bigint AS object2_path_hits,
                MIN(r.first_seen_ns) AS first_seen_ns,
                MAX(r.last_seen_ns) AS last_seen_ns,
                COALESCE(day_strings.active_days, '') AS active_days
            FROM gt_input AS g
            LEFT JOIN gt_role_hits AS r
              ON g.gt_uuid = r.gt_uuid
            LEFT JOIN day_strings
              ON g.gt_uuid = day_strings.gt_uuid
            GROUP BY g.gt_uuid, day_strings.active_days
            ORDER BY total_event_hits DESC, g.gt_uuid
            """
        )
        per_uuid_rows = []
        visible_gt_count = 0
        subject_visible_count = 0
        object_visible_count = 0
        object2_visible_count = 0
        pattern_counts: Dict[str, int] = {}
        for row in cursor.fetchall():
            total_event_hits = int(row[1])
            subject_hits = int(row[2])
            object_hits = int(row[3])
            object2_hits = int(row[4])
            role_pattern = classify_role_pattern(subject_hits, object_hits, object2_hits)
            if total_event_hits > 0:
                visible_gt_count += 1
            if subject_hits > 0:
                subject_visible_count += 1
            if object_hits > 0:
                object_visible_count += 1
            if object2_hits > 0:
                object2_visible_count += 1
            pattern_counts[role_pattern] = pattern_counts.get(role_pattern, 0) + 1
            per_uuid_rows.append(
                {
                    "gt_uuid": row[0],
                    "total_event_hits": total_event_hits,
                    "subject_hits": subject_hits,
                    "object_hits": object_hits,
                    "object2_hits": object2_hits,
                    "subject_named_hits": int(row[5]),
                    "object_path_hits": int(row[6]),
                    "object2_path_hits": int(row[7]),
                    "first_seen": timestamp_to_text(row[8]),
                    "last_seen": timestamp_to_text(row[9]),
                    "active_days": row[10],
                    "role_pattern": role_pattern,
                    "visible_in_events": "yes" if total_event_hits > 0 else "no",
                    "process_like": "yes" if subject_hits > 0 else "no",
                }
            )

    summary = {
        "total_gt_uuids": total_gt_uuids,
        "visible_gt_uuids": visible_gt_count,
        "invisible_gt_uuids": total_gt_uuids - visible_gt_count,
        "subject_visible_gt_uuids": subject_visible_count,
        "object_visible_gt_uuids": object_visible_count,
        "object2_visible_gt_uuids": object2_visible_count,
        "role_pattern_counts": pattern_counts,
    }
    return summary, role_summary, daily_summary, per_uuid_rows


def print_summary(summary: Dict[str, object], role_summary: List[Dict[str, object]], daily_summary: List[Dict[str, object]]) -> None:
    print("[gt-align] summary")
    print(f"  total_gt_uuids: {summary['total_gt_uuids']:,}")
    print(f"  visible_gt_uuids: {summary['visible_gt_uuids']:,}")
    print(f"  invisible_gt_uuids: {summary['invisible_gt_uuids']:,}")
    print(f"  subject_visible_gt_uuids: {summary['subject_visible_gt_uuids']:,}")
    print(f"  object_visible_gt_uuids: {summary['object_visible_gt_uuids']:,}")
    print(f"  object2_visible_gt_uuids: {summary['object2_visible_gt_uuids']:,}")
    print("  role_pattern_counts:")
    for pattern, count in sorted(summary["role_pattern_counts"].items()):
        print(f"    {pattern}: {count:,}")

    print("[gt-align] role summary")
    for row in role_summary:
        print(
            f"  {row['role']}: hit_uuids={row['hit_uuid_count']:,} "
            f"event_hits={row['event_hits']:,} "
            f"window={row['first_seen']} -> {row['last_seen']}"
        )

    print("[gt-align] daily coverage")
    for row in daily_summary:
        print(
            f"  {row['day']}: gt_uuids={row['gt_uuid_count']:,} "
            f"event_hits={row['total_event_hits']:,} "
            f"subject={row['subject_gt_count']:,} "
            f"object={row['object_gt_count']:,} "
            f"object2={row['object2_gt_count']:,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Align CADETS ground-truth UUIDs against events_raw.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument(
        "--ground-truth-path",
        type=str,
        default=str(DEFAULT_GT_PATH),
        help=f"Ground-truth UUID list. Default: {DEFAULT_GT_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for GT alignment outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    config = load_ingest_config(args)
    gt_path = Path(args.ground_truth_path).expanduser().resolve()
    if not gt_path.exists():
        raise SystemExit(f"Ground-truth file does not exist: {gt_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    gt_uuids, duplicate_lines = load_ground_truth(gt_path)
    conn = connect_target_db(config.database)
    try:
        load_gt_into_temp_table(conn, gt_uuids)
        build_alignment_temp_tables(conn)
        summary, role_summary, daily_summary, per_uuid_rows = fetch_alignment_rows(conn)
    finally:
        conn.close()

    summary["ground_truth_path"] = str(gt_path)
    summary["output_dir"] = str(output_dir)
    summary["duplicate_lines_ignored"] = duplicate_lines

    write_tsv(
        output_dir / "role_summary.tsv",
        ["role", "hit_uuid_count", "event_hits", "first_seen", "last_seen", "named_event_hits", "path_event_hits"],
        role_summary,
    )
    write_tsv(
        output_dir / "daily_summary.tsv",
        ["day", "gt_uuid_count", "total_event_hits", "subject_gt_count", "object_gt_count", "object2_gt_count"],
        daily_summary,
    )
    write_tsv(
        output_dir / "per_uuid.tsv",
        [
            "gt_uuid",
            "total_event_hits",
            "subject_hits",
            "object_hits",
            "object2_hits",
            "subject_named_hits",
            "object_path_hits",
            "object2_path_hits",
            "first_seen",
            "last_seen",
            "active_days",
            "role_pattern",
            "visible_in_events",
            "process_like",
        ],
        per_uuid_rows,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_summary(summary, role_summary, daily_summary)
    print(f"[gt-align] outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
