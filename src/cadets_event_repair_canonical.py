import argparse

from cadets_event_ingest_common import (
    connect_target_db,
    create_events_raw_indexes,
    get_latest_run_id,
    load_ingest_config,
    print_run_summary,
    rebuild_events_raw,
    tune_ingest_session,
    update_file_audit_canonical_stats,
    fetch_run_summary,
)


def has_stage_fingerprint_column(conn) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'event_stage_raw'
                  AND column_name = 'event_fingerprint'
            )
            """
        )
        return bool(cursor.fetchone()[0])


def recreate_events_raw_table(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS events_raw")
        cursor.execute(
            """
            CREATE TABLE events_raw (
                event_fingerprint CHAR(32) PRIMARY KEY,
                run_id BIGINT NOT NULL,
                source_file TEXT NOT NULL,
                source_line_no BIGINT NOT NULL,
                duplicate_count INTEGER NOT NULL DEFAULT 1,
                event_uuid VARCHAR(255) NOT NULL,
                event_type VARCHAR(100),
                timestamp_ns BIGINT,
                sequence_num BIGINT,
                thread_id INTEGER,
                host_id VARCHAR(255),
                subject_uuid VARCHAR(255),
                object_uuid VARCHAR(255),
                object2_uuid VARCHAR(255),
                object_path TEXT,
                object2_path TEXT,
                event_name VARCHAR(255),
                size_bytes BIGINT,
                exec_name VARCHAR(255),
                parent_pid VARCHAR(50),
                file_descriptor VARCHAR(50),
                return_value VARCHAR(50),
                cdm_version VARCHAR(32),
                source_tag VARCHAR(255),
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair the canonical events_raw table using exact-event fingerprint deduplication.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument("--run-id", type=int, default=None, help="Run to repair. Defaults to the latest run.")
    args = parser.parse_args()

    config = load_ingest_config(args)
    conn = connect_target_db(config.database)
    try:
        tune_ingest_session(conn)
        run_id = args.run_id if args.run_id is not None else get_latest_run_id(conn)
        if run_id is None:
            raise SystemExit("No ingest run found in the target database.")

        use_stage_fingerprint = has_stage_fingerprint_column(conn)
        print(f"[repair] target database: {config.database.target_db}")
        print(f"[repair] run_id: {run_id}")
        print(
            "[repair] fingerprint source:",
            "materialized stage column" if use_stage_fingerprint else "on-the-fly md5(raw_record::text)",
        )
        print("[repair] recreating events_raw with fingerprint primary key")
        recreate_events_raw_table(conn)
        print("[repair] rebuilding canonical rows from event_stage_raw")
        rebuild_events_raw(conn, run_id, use_stage_fingerprint=use_stage_fingerprint)
        print("[repair] creating events_raw indexes")
        create_events_raw_indexes(conn)
        print("[repair] updating ingest_file_audit canonical stats")
        update_file_audit_canonical_stats(conn, run_id, use_stage_fingerprint=use_stage_fingerprint)
        print("[repair] done")
        summary = fetch_run_summary(conn, run_id)
        print_run_summary(summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
