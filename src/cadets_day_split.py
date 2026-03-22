import argparse
import csv
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cadets_event_ingest_common import connect_target_db, load_ingest_config


DEFAULT_GT_ENTITY_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "gt_entity_alignment" / "per_uuid.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "day_split"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S UTC"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_utc_timestamp(text: str) -> Optional[datetime]:
    text = text.strip()
    if not text or text == "N/A":
        return None
    return datetime.strptime(text, TIMESTAMP_FMT)


def load_gt_entity_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fetch_events_daily_summary(conn) -> List[Dict[str, object]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                to_char(timezone('UTC', to_timestamp(timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                COUNT(*)::bigint AS total_events
            FROM events_raw
            GROUP BY day
            ORDER BY day
            """
        )
        return [{"day": row[0], "total_events": int(row[1])} for row in cursor.fetchall()]


def load_temp_uuid_table(conn, table_name: str, uuids: List[str]) -> None:
    with conn.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TEMP TABLE {table_name} (gt_uuid VARCHAR(255) PRIMARY KEY)")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for gt_uuid in uuids:
        writer.writerow([gt_uuid])
    buffer.seek(0)

    with conn.cursor() as cursor:
        cursor.copy_expert(
            f"""
            COPY {table_name} (gt_uuid)
            FROM STDIN WITH (FORMAT CSV)
            """,
            buffer,
        )
        cursor.execute(f"ANALYZE {table_name}")
        cursor.execute("SET work_mem TO '512MB'")
    conn.commit()


def fetch_gt_event_windows(conn, all_gt_uuids: List[str], process_gt_uuids: List[str]) -> Dict[str, Dict[str, str]]:
    load_temp_uuid_table(conn, "gt_all_input", all_gt_uuids)
    load_temp_uuid_table(conn, "gt_process_input", process_gt_uuids)

    with conn.cursor() as cursor:
        cursor.execute(
            """
            WITH all_gt_hits AS (
                SELECT
                    to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                    MIN(e.timestamp_ns) AS all_gt_first_ns,
                    MAX(e.timestamp_ns) AS all_gt_last_ns
                FROM events_raw AS e
                WHERE EXISTS (SELECT 1 FROM gt_all_input g WHERE g.gt_uuid = e.subject_uuid)
                   OR EXISTS (SELECT 1 FROM gt_all_input g WHERE g.gt_uuid = e.object_uuid)
                   OR EXISTS (SELECT 1 FROM gt_all_input g WHERE g.gt_uuid = e.object2_uuid)
                GROUP BY day
            ),
            process_subject_hits AS (
                SELECT
                    to_char(timezone('UTC', to_timestamp(e.timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                    MIN(e.timestamp_ns) AS process_subject_first_ns,
                    MAX(e.timestamp_ns) AS process_subject_last_ns
                FROM events_raw AS e
                JOIN gt_process_input AS g
                  ON g.gt_uuid = e.subject_uuid
                GROUP BY day
            )
            SELECT
                COALESCE(a.day, p.day) AS day,
                a.all_gt_first_ns,
                a.all_gt_last_ns,
                p.process_subject_first_ns,
                p.process_subject_last_ns
            FROM all_gt_hits AS a
            FULL OUTER JOIN process_subject_hits AS p
              ON a.day = p.day
            ORDER BY day
            """
        )
        rows = cursor.fetchall()

    windows: Dict[str, Dict[str, str]] = {}
    for row in rows:
        windows[row[0]] = {
            "all_gt_first_seen": ns_to_text(row[1]),
            "all_gt_last_seen": ns_to_text(row[2]),
            "process_subject_first_seen": ns_to_text(row[3]),
            "process_subject_last_seen": ns_to_text(row[4]),
        }
    return windows


def ns_to_text(timestamp_ns: Optional[int]) -> str:
    if timestamp_ns is None:
        return "N/A"
    return datetime.utcfromtimestamp(timestamp_ns / 1_000_000_000).strftime(TIMESTAMP_FMT)


def build_day_stats(gt_rows: List[Dict[str, str]], day_event_summary: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    stats: Dict[str, Dict[str, object]] = {}
    ordered_days = [row["day"] for row in day_event_summary]
    for row in day_event_summary:
        day = str(row["day"])
        stats[day] = {
            "day": day,
            "total_events": int(row["total_events"]),
            "active_gt_count": 0,
            "new_gt_count": 0,
            "active_file_gt_count": 0,
            "active_process_gt_count": 0,
            "active_network_gt_count": 0,
            "active_unmatched_gt_count": 0,
            "new_file_gt_count": 0,
            "new_process_gt_count": 0,
            "new_network_gt_count": 0,
            "new_unmatched_gt_count": 0,
            "subject_active_gt_count": 0,
            "object_active_gt_count": 0,
            "object2_active_gt_count": 0,
            "candidate_attack_day": "no",
        }

    for row in gt_rows:
        entity_type = row["entity_type"]
        role_pattern = row["role_pattern"]
        active_days = [day for day in row["active_days"].split(",") if day]
        first_seen_ts = parse_utc_timestamp(row["first_seen"])
        first_seen_day = first_seen_ts.strftime("%Y-%m-%d") if first_seen_ts else None

        for day in active_days:
            day_stats = stats.setdefault(
                day,
                {
                    "day": day,
                    "total_events": 0,
                    "active_gt_count": 0,
                    "new_gt_count": 0,
                    "active_file_gt_count": 0,
                    "active_process_gt_count": 0,
                    "active_network_gt_count": 0,
                    "active_unmatched_gt_count": 0,
                    "new_file_gt_count": 0,
                    "new_process_gt_count": 0,
                    "new_network_gt_count": 0,
                    "new_unmatched_gt_count": 0,
                    "subject_active_gt_count": 0,
                    "object_active_gt_count": 0,
                    "object2_active_gt_count": 0,
                    "candidate_attack_day": "no",
                },
            )
            day_stats["active_gt_count"] += 1
            day_stats[f"active_{entity_type}_gt_count"] += 1
            if "subject" in role_pattern:
                day_stats["subject_active_gt_count"] += 1
            if "object" in role_pattern:
                day_stats["object_active_gt_count"] += 1
            if "object2" in role_pattern:
                day_stats["object2_active_gt_count"] += 1

        if first_seen_day:
            day_stats = stats[first_seen_day]
            day_stats["new_gt_count"] += 1
            day_stats[f"new_{entity_type}_gt_count"] += 1

    sorted_days = sorted(stats)
    for day in sorted_days:
        row = stats[day]
        if row["active_process_gt_count"] > 0 or row["new_process_gt_count"] > 0:
            row["candidate_attack_day"] = "yes"
        elif row["new_gt_count"] >= 100:
            row["candidate_attack_day"] = "yes"

    return [stats[day] for day in sorted_days], sorted_days


def build_windows(gt_rows: List[Dict[str, str]], gt_event_windows: Dict[str, Dict[str, str]], split_days: List[str]) -> List[Dict[str, object]]:
    new_first_seen_by_day: Dict[str, List[datetime]] = defaultdict(list)
    new_first_seen_by_day_process: Dict[str, List[datetime]] = defaultdict(list)

    for row in gt_rows:
        first_seen_ts = parse_utc_timestamp(row["first_seen"])
        if first_seen_ts is None:
            continue
        day = first_seen_ts.strftime("%Y-%m-%d")
        new_first_seen_by_day[day].append(first_seen_ts)
        if row["entity_type"] == "process":
            new_first_seen_by_day_process[day].append(first_seen_ts)

    windows = []
    for day in split_days:
        event_window = gt_event_windows.get(day, {})
        all_firsts = sorted(new_first_seen_by_day.get(day, []))
        process_firsts = sorted(new_first_seen_by_day_process.get(day, []))
        windows.append(
            {
                "day": day,
                "all_gt_first_seen": event_window.get("all_gt_first_seen", "N/A"),
                "all_gt_last_seen": event_window.get("all_gt_last_seen", "N/A"),
                "process_subject_first_seen": event_window.get("process_subject_first_seen", "N/A"),
                "process_subject_last_seen": event_window.get("process_subject_last_seen", "N/A"),
                "new_gt_first_seen_start": all_firsts[0].strftime(TIMESTAMP_FMT) if all_firsts else "N/A",
                "new_gt_first_seen_end": all_firsts[-1].strftime(TIMESTAMP_FMT) if all_firsts else "N/A",
                "new_process_gt_first_seen_start": process_firsts[0].strftime(TIMESTAMP_FMT) if process_firsts else "N/A",
                "new_process_gt_first_seen_end": process_firsts[-1].strftime(TIMESTAMP_FMT) if process_firsts else "N/A",
            }
        )
    return windows


def build_split_manifest(day_rows: List[Dict[str, object]]) -> Dict[str, object]:
    ordered_days = [row["day"] for row in day_rows]
    attack_days = [row["day"] for row in day_rows if row["candidate_attack_day"] == "yes"]
    if not attack_days:
        raise ValueError("No candidate attack days were detected from GT alignment.")

    first_attack_day = attack_days[0]
    first_attack_index = ordered_days.index(first_attack_day)

    val_days: List[str] = []
    train_days: List[str] = []
    if first_attack_index > 0:
        val_days = [ordered_days[first_attack_index - 1]]
        train_days = ordered_days[: first_attack_index - 1]
    else:
        train_days = []

    manifest = {
        "recommended_split": {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": attack_days,
        },
        "heuristic": {
            "attack_day_rule": "candidate_attack_day = yes if active_process_gt_count > 0 or new_process_gt_count > 0 or new_gt_count >= 100",
            "val_day_rule": "use the last day immediately before the first candidate attack day",
            "train_day_rule": "use all earlier days before val",
        },
    }
    return manifest


def print_summary(day_rows: List[Dict[str, object]], manifest: Dict[str, object]) -> None:
    print("[day-split] daily summary")
    for row in day_rows:
        print(
            f"  {row['day']}: events={row['total_events']:,} "
            f"active_gt={row['active_gt_count']:,} new_gt={row['new_gt_count']:,} "
            f"process_active={row['active_process_gt_count']:,} "
            f"candidate_attack_day={row['candidate_attack_day']}"
        )

    split = manifest["recommended_split"]
    print("[day-split] recommended split")
    print(f"  train_days: {', '.join(split['train_days']) if split['train_days'] else '[]'}")
    print(f"  val_days: {', '.join(split['val_days']) if split['val_days'] else '[]'}")
    print(f"  test_days: {', '.join(split['test_days']) if split['test_days'] else '[]'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build day-level splits and GT time-window summaries for CADETS.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument("--gt-entity-path", type=str, default=str(DEFAULT_GT_ENTITY_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config = load_ingest_config(args)
    gt_entity_path = Path(args.gt_entity_path).expanduser().resolve()
    if not gt_entity_path.exists():
        raise SystemExit(f"GT entity alignment file does not exist: {gt_entity_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    gt_rows = load_gt_entity_rows(gt_entity_path)
    conn = connect_target_db(config.database)
    try:
        event_day_rows = fetch_events_daily_summary(conn)
        all_gt_uuids = [row["gt_uuid"] for row in gt_rows]
        process_gt_uuids = [row["gt_uuid"] for row in gt_rows if row["entity_type"] == "process"]
        gt_event_windows = fetch_gt_event_windows(conn, all_gt_uuids, process_gt_uuids)
    finally:
        conn.close()

    day_rows, ordered_days = build_day_stats(gt_rows, event_day_rows)
    manifest = build_split_manifest(day_rows)
    test_days = manifest["recommended_split"]["test_days"]
    window_rows = build_windows(gt_rows, gt_event_windows, test_days)

    write_tsv(
        output_dir / "day_summary.tsv",
        [
            "day",
            "total_events",
            "active_gt_count",
            "new_gt_count",
            "active_file_gt_count",
            "active_process_gt_count",
            "active_network_gt_count",
            "active_unmatched_gt_count",
            "new_file_gt_count",
            "new_process_gt_count",
            "new_network_gt_count",
            "new_unmatched_gt_count",
            "subject_active_gt_count",
            "object_active_gt_count",
            "object2_active_gt_count",
            "candidate_attack_day",
        ],
        day_rows,
    )
    write_tsv(
        output_dir / "gt_windows.tsv",
        [
            "day",
            "all_gt_first_seen",
            "all_gt_last_seen",
            "process_subject_first_seen",
            "process_subject_last_seen",
            "new_gt_first_seen_start",
            "new_gt_first_seen_end",
            "new_process_gt_first_seen_start",
            "new_process_gt_first_seen_end",
        ],
        window_rows,
    )
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print_summary(day_rows, manifest)
    print(f"[day-split] outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
