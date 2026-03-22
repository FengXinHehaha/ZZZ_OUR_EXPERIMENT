import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

from cadets_event_ingest_common import (
    DatabaseConfig,
    connect_target_db,
    load_ingest_config,
    make_connection,
    timestamp_to_text,
)


DEFAULT_GT_PATH = Path("/home/fxh/DeepSeek/26.03.17 version/cadets_ground_truth.txt")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "gt_entity_alignment"
DEFAULT_REFERENCE_DB = "0310cadets"


def load_ground_truth(path: Path) -> List[str]:
    gt_uuids: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            gt_uuid = raw_line.strip().upper()
            if not gt_uuid or gt_uuid in seen:
                continue
            seen.add(gt_uuid)
            gt_uuids.append(gt_uuid)
    return gt_uuids


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
        cursor.execute("CREATE TEMP TABLE gt_input (gt_uuid VARCHAR(255) PRIMARY KEY)")

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


def detect_reference_tables(conn) -> Dict[str, str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('process_entities', 'file_entities', 'network_entities', 'processes', 'files', 'networks')
            """
        )
        existing = {row[0] for row in cursor.fetchall()}

    if {"process_entities", "file_entities", "network_entities"}.issubset(existing):
        return {
            "process": "process_entities",
            "file": "file_entities",
            "network": "network_entities",
        }
    if {"processes", "files", "networks"}.issubset(existing):
        return {
            "process": "processes",
            "file": "files",
            "network": "networks",
        }
    raise RuntimeError(
        "Could not find a full reference entity table set. "
        "Expected either process_entities/file_entities/network_entities or processes/files/networks."
    )


def fetch_event_alignment_loaded(conn) -> Dict[str, Dict[str, object]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            WITH role_hits AS (
                SELECT
                    g.gt_uuid,
                    'subject'::text AS role,
                    COUNT(*)::bigint AS event_hits,
                    MIN(e.timestamp_ns) AS first_seen_ns,
                    MAX(e.timestamp_ns) AS last_seen_ns
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
                    MAX(e.timestamp_ns) AS last_seen_ns
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
                    MAX(e.timestamp_ns) AS last_seen_ns
                FROM gt_input AS g
                JOIN events_raw AS e
                  ON e.object2_uuid = g.gt_uuid
                GROUP BY g.gt_uuid
            ),
            distinct_days AS (
                SELECT DISTINCT
                    g.gt_uuid,
                    to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day
                FROM gt_input AS g
                JOIN events_raw AS e
                  ON e.subject_uuid = g.gt_uuid
                    OR e.object_uuid = g.gt_uuid
                    OR e.object2_uuid = g.gt_uuid
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
                MIN(r.first_seen_ns) AS first_seen_ns,
                MAX(r.last_seen_ns) AS last_seen_ns,
                COALESCE(day_strings.active_days, '') AS active_days
            FROM gt_input AS g
            LEFT JOIN role_hits AS r
              ON g.gt_uuid = r.gt_uuid
            LEFT JOIN day_strings
              ON g.gt_uuid = day_strings.gt_uuid
            GROUP BY g.gt_uuid, day_strings.active_days
            """
        )
        rows = cursor.fetchall()

    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        subject_hits = int(row[2])
        object_hits = int(row[3])
        object2_hits = int(row[4])
        roles = []
        if subject_hits > 0:
            roles.append("subject")
        if object_hits > 0:
            roles.append("object")
        if object2_hits > 0:
            roles.append("object2")
        result[row[0]] = {
            "total_event_hits": int(row[1]),
            "subject_hits": subject_hits,
            "object_hits": object_hits,
            "object2_hits": object2_hits,
            "first_seen": timestamp_to_text(row[5]),
            "last_seen": timestamp_to_text(row[6]),
            "active_days": row[7],
            "role_pattern": "+".join(roles) if roles else "unseen",
            "visible_in_events": int(row[1]) > 0,
        }
    return result


def fetch_reference_alignment_loaded(conn) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    table_map = detect_reference_tables(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            WITH process_hits AS (
                SELECT gt.gt_uuid, 'process'::text AS entity_type
                FROM gt_input AS gt
                JOIN {table_map['process']} AS p
                  ON UPPER(p.uuid) = gt.gt_uuid
            ),
            file_hits AS (
                SELECT gt.gt_uuid, 'file'::text AS entity_type
                FROM gt_input AS gt
                JOIN {table_map['file']} AS f
                  ON UPPER(f.uuid) = gt.gt_uuid
            ),
            network_hits AS (
                SELECT gt.gt_uuid, 'network'::text AS entity_type
                FROM gt_input AS gt
                JOIN {table_map['network']} AS n
                  ON UPPER(n.uuid) = gt.gt_uuid
            ),
            unioned AS (
                SELECT * FROM process_hits
                UNION ALL
                SELECT * FROM file_hits
                UNION ALL
                SELECT * FROM network_hits
            )
            SELECT
                gt.gt_uuid,
                COALESCE(string_agg(u.entity_type, '+' ORDER BY u.entity_type), '') AS matched_entity_types,
                COUNT(u.entity_type)::int AS matched_table_count
            FROM gt_input AS gt
            LEFT JOIN unioned AS u
              ON gt.gt_uuid = u.gt_uuid
            GROUP BY gt.gt_uuid
            """
        )
        rows = cursor.fetchall()

    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        matched_types = row[1]
        matched_count = int(row[2])
        if matched_count == 0:
            entity_type = "unmatched"
        elif matched_count == 1:
            entity_type = matched_types
        else:
            entity_type = "ambiguous"
        result[row[0]] = {
            "matched_entity_types": matched_types,
            "matched_table_count": matched_count,
            "entity_type": entity_type,
        }
    return result, table_map


def build_reference_database_config(base: DatabaseConfig, reference_db: str) -> DatabaseConfig:
    return DatabaseConfig(
        host=base.host,
        port=base.port,
        user=base.user,
        password=base.password,
        admin_db=base.admin_db,
        target_db=reference_db,
    )


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    entity_type_counts: Dict[str, int] = {}
    visible_entity_counts: Dict[str, int] = {}
    role_pattern_counts: Dict[str, int] = {}
    visible_count = 0
    for row in rows:
        entity_type = str(row["entity_type"])
        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        role_pattern = str(row["role_pattern"])
        role_pattern_counts[role_pattern] = role_pattern_counts.get(role_pattern, 0) + 1
        if row["visible_in_events"] == "yes":
            visible_count += 1
            visible_entity_counts[entity_type] = visible_entity_counts.get(entity_type, 0) + 1
    return {
        "total_gt_uuids": len(rows),
        "visible_gt_uuids": visible_count,
        "invisible_gt_uuids": len(rows) - visible_count,
        "entity_type_counts": entity_type_counts,
        "visible_entity_type_counts": visible_entity_counts,
        "role_pattern_counts": role_pattern_counts,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print("[gt-entity-align] summary")
    print(f"  total_gt_uuids: {summary['total_gt_uuids']:,}")
    print(f"  visible_gt_uuids: {summary['visible_gt_uuids']:,}")
    print(f"  invisible_gt_uuids: {summary['invisible_gt_uuids']:,}")
    print("  entity_type_counts:")
    for key, value in sorted(summary["entity_type_counts"].items()):
        print(f"    {key}: {value:,}")
    print("  visible_entity_type_counts:")
    for key, value in sorted(summary["visible_entity_type_counts"].items()):
        print(f"    {key}: {value:,}")
    print("  role_pattern_counts:")
    for key, value in sorted(summary["role_pattern_counts"].items()):
        print(f"    {key}: {value:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align CADETS GT UUIDs to entity tables and event visibility.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument("--reference-db", type=str, default=DEFAULT_REFERENCE_DB)
    parser.add_argument("--ground-truth-path", type=str, default=str(DEFAULT_GT_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config = load_ingest_config(args)
    gt_path = Path(args.ground_truth_path).expanduser().resolve()
    if not gt_path.exists():
        raise SystemExit(f"Ground-truth file does not exist: {gt_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    gt_uuids = load_ground_truth(gt_path)

    target_conn = connect_target_db(config.database)
    try:
        load_gt_into_temp_table(target_conn, gt_uuids)
        event_alignment = fetch_event_alignment_loaded(target_conn)
    finally:
        target_conn.close()

    reference_db = build_reference_database_config(config.database, args.reference_db)
    reference_conn = make_connection(reference_db, reference_db.target_db, autocommit=False)
    try:
        load_gt_into_temp_table(reference_conn, gt_uuids)
        reference_alignment, table_map = fetch_reference_alignment_loaded(reference_conn)
    finally:
        reference_conn.close()

    rows: List[Dict[str, object]] = []
    for gt_uuid in gt_uuids:
        event_row = event_alignment[gt_uuid]
        ref_row = reference_alignment[gt_uuid]
        rows.append(
            {
                "gt_uuid": gt_uuid,
                "entity_type": ref_row["entity_type"],
                "matched_entity_types": ref_row["matched_entity_types"],
                "matched_table_count": ref_row["matched_table_count"],
                "visible_in_events": "yes" if event_row["visible_in_events"] else "no",
                "total_event_hits": event_row["total_event_hits"],
                "subject_hits": event_row["subject_hits"],
                "object_hits": event_row["object_hits"],
                "object2_hits": event_row["object2_hits"],
                "role_pattern": event_row["role_pattern"],
                "first_seen": event_row["first_seen"],
                "last_seen": event_row["last_seen"],
                "active_days": event_row["active_days"],
            }
        )

    summary = summarize_rows(rows)
    summary["ground_truth_path"] = str(gt_path)
    summary["reference_db"] = args.reference_db
    summary["reference_tables"] = table_map
    summary["output_dir"] = str(output_dir)

    write_tsv(
        output_dir / "per_uuid.tsv",
        [
            "gt_uuid",
            "entity_type",
            "matched_entity_types",
            "matched_table_count",
            "visible_in_events",
            "total_event_hits",
            "subject_hits",
            "object_hits",
            "object2_hits",
            "role_pattern",
            "first_seen",
            "last_seen",
            "active_days",
        ],
        rows,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"[gt-entity-align] outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
