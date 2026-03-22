import argparse

from cadets_event_ingest_common import (
    connect_target_db,
    fetch_run_summary,
    get_latest_run_id,
    load_ingest_config,
    print_run_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit report for the CADETS event ingest database.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument("--run-id", type=int, default=None, help="Run to report. Defaults to the latest run.")
    args = parser.parse_args()

    config = load_ingest_config(args)
    conn = connect_target_db(config.database)
    try:
        run_id = args.run_id if args.run_id is not None else get_latest_run_id(conn)
        if run_id is None:
            raise SystemExit("No ingest run found in the target database.")
        summary = fetch_run_summary(conn, run_id)
        print_run_summary(summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
