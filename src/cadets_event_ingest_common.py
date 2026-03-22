import argparse
import csv
import glob
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import sql

try:
    import orjson
except ImportError:
    orjson = None


EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cadets_event_ingest.json"
DEFAULT_SOURCE_GLOB = "/home/fxh/DeepSeek/cadets_total_datasets/cadets/ta1-cadets-e3-official*.json*"


def utc_day_from_ns(timestamp_ns: Optional[int]) -> Optional[str]:
    if timestamp_ns is None:
        return None
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d")


def timestamp_to_text(timestamp_ns: Optional[int]) -> str:
    if timestamp_ns is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    admin_db: str
    target_db: str


@dataclass
class IngestConfig:
    database: DatabaseConfig
    source_glob: str
    batch_size: int
    page_size: int
    commit_every_batches: int
    progress_every_lines: int
    file_limit: Optional[int]
    line_limit: Optional[int]
    reset_db: bool


def resolve_password(config: Dict[str, Any], cli_password: Optional[str]) -> str:
    if cli_password:
        return cli_password
    database_cfg = config.get("database", {})
    if database_cfg.get("password"):
        return str(database_cfg["password"])
    password_env = database_cfg.get("password_env")
    if password_env and os.environ.get(password_env):
        return os.environ[password_env]
    if os.environ.get("CADETS_PG_PASSWORD"):
        return os.environ["CADETS_PG_PASSWORD"]
    raise ValueError(
        "Database password is missing. Pass --password, set database.password in the config, "
        "or export the environment variable named by database.password_env."
    )


def load_ingest_config(args: argparse.Namespace) -> IngestConfig:
    defaults: Dict[str, Any] = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "admin_db": "postgres",
            "target_db": "zzz_our_experiment_cadets",
        },
        "source_glob": DEFAULT_SOURCE_GLOB,
        "batch_size": 5000,
        "page_size": 1000,
        "commit_every_batches": 20,
        "progress_every_lines": 200000,
        "file_limit": None,
        "line_limit": None,
    }
    file_config = load_json_config(getattr(args, "config", None))
    merged = deep_merge(defaults, file_config)
    if getattr(args, "source_glob", None):
        merged["source_glob"] = args.source_glob
    if getattr(args, "host", None):
        merged["database"]["host"] = args.host
    if getattr(args, "port", None):
        merged["database"]["port"] = args.port
    if getattr(args, "user", None):
        merged["database"]["user"] = args.user
    if getattr(args, "admin_db", None):
        merged["database"]["admin_db"] = args.admin_db
    if getattr(args, "target_db", None):
        merged["database"]["target_db"] = args.target_db
    if getattr(args, "batch_size", None):
        merged["batch_size"] = args.batch_size
    if getattr(args, "page_size", None):
        merged["page_size"] = args.page_size
    if getattr(args, "commit_every_batches", None):
        merged["commit_every_batches"] = args.commit_every_batches
    if getattr(args, "progress_every_lines", None):
        merged["progress_every_lines"] = args.progress_every_lines
    if getattr(args, "file_limit", None) is not None:
        merged["file_limit"] = args.file_limit
    if getattr(args, "line_limit", None) is not None:
        merged["line_limit"] = args.line_limit

    password = resolve_password(merged, getattr(args, "password", None))
    database_cfg = DatabaseConfig(
        host=str(merged["database"]["host"]),
        port=int(merged["database"]["port"]),
        user=str(merged["database"]["user"]),
        password=password,
        admin_db=str(merged["database"]["admin_db"]),
        target_db=str(merged["database"]["target_db"]),
    )
    return IngestConfig(
        database=database_cfg,
        source_glob=str(merged["source_glob"]),
        batch_size=int(merged["batch_size"]),
        page_size=int(merged["page_size"]),
        commit_every_batches=int(merged["commit_every_batches"]),
        progress_every_lines=int(merged["progress_every_lines"]),
        file_limit=int(merged["file_limit"]) if merged["file_limit"] is not None else None,
        line_limit=int(merged["line_limit"]) if merged["line_limit"] is not None else None,
        reset_db=bool(getattr(args, "reset_db", False)),
    )


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help=f"JSON config path. Default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--admin-db", type=str, default=None)
    parser.add_argument("--target-db", type=str, default=None)
    parser.add_argument("--source-glob", type=str, default=None)
    parser.add_argument("--file-limit", type=int, default=None)
    parser.add_argument("--line-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--commit-every-batches", type=int, default=None)
    parser.add_argument("--progress-every-lines", type=int, default=None)
    parser.add_argument("--reset-db", action="store_true", help="Drop and recreate the target database before ingest.")
    return parser


def discover_source_files(source_glob: str, file_limit: Optional[int]) -> List[str]:
    files = sorted(glob.glob(source_glob))
    if not files:
        raise FileNotFoundError(f"No source files match: {source_glob}")
    if file_limit is not None:
        return files[:file_limit]
    return files


def make_connection(database: DatabaseConfig, dbname: str, autocommit: bool = False):
    conn = psycopg2.connect(
        host=database.host,
        port=database.port,
        user=database.user,
        password=database.password,
        dbname=dbname,
    )
    conn.autocommit = autocommit
    return conn


def ensure_database_exists(database: DatabaseConfig) -> None:
    conn = make_connection(database, database.admin_db, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database.target_db,))
            if cursor.fetchone():
                print(f"[db] Target database already exists: {database.target_db}")
                return
            print(f"[db] Creating target database: {database.target_db}")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database.target_db)))
    finally:
        conn.close()


def drop_database_if_exists(database: DatabaseConfig) -> None:
    conn = make_connection(database, database.admin_db, autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database.target_db,))
            if not cursor.fetchone():
                print(f"[db] Target database does not exist, nothing to drop: {database.target_db}")
                return
            print(f"[db] Dropping target database: {database.target_db}")
            cursor.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database.target_db)))
    finally:
        conn.close()


def connect_target_db(database: DatabaseConfig):
    return make_connection(database, database.target_db, autocommit=False)


def tune_ingest_session(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SET synchronous_commit TO OFF")
        cursor.execute("SET work_mem TO '256MB'")
        cursor.execute("SET maintenance_work_mem TO '1GB'")
    conn.commit()


def ensure_schema(conn) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id BIGSERIAL PRIMARY KEY,
            started_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP WITHOUT TIME ZONE,
            source_glob TEXT NOT NULL,
            status VARCHAR(32) NOT NULL,
            source_file_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ingest_file_audit (
            audit_id BIGSERIAL PRIMARY KEY,
            run_id BIGINT NOT NULL REFERENCES ingest_runs(run_id) ON DELETE CASCADE,
            source_file TEXT NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            total_lines BIGINT NOT NULL DEFAULT 0,
            event_lines BIGINT NOT NULL DEFAULT 0,
            parse_errors BIGINT NOT NULL DEFAULT 0,
            stage_rows BIGINT NOT NULL DEFAULT 0,
            canonical_rows BIGINT NOT NULL DEFAULT 0,
            duplicate_uuid_rows BIGINT NOT NULL DEFAULT 0,
            min_timestamp_ns BIGINT,
            max_timestamp_ns BIGINT,
            min_day DATE,
            max_day DATE,
            status VARCHAR(32) NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (run_id, source_file)
        )
        """,
        """
        CREATE UNLOGGED TABLE IF NOT EXISTS event_stage_raw (
            run_id BIGINT NOT NULL,
            source_file TEXT NOT NULL,
            line_no BIGINT NOT NULL,
            event_fingerprint CHAR(32),
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
            raw_record JSONB NOT NULL,
            ingested_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events_raw (
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
        """,
    ]
    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    conn.commit()


def create_run(conn, source_glob: str, source_file_count: int) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ingest_runs (source_glob, status, source_file_count)
            VALUES (%s, %s, %s)
            RETURNING run_id
            """,
            (source_glob, "running", source_file_count),
        )
        run_id = cursor.fetchone()[0]
    conn.commit()
    return run_id


def finalize_run(conn, run_id: int, status: str, notes: Optional[str] = None) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE ingest_runs
            SET status = %s,
                notes = %s,
                finished_at = CURRENT_TIMESTAMP
            WHERE run_id = %s
            """,
            (status, notes, run_id),
        )
    conn.commit()


def create_file_audit(conn, run_id: int, source_file: str, file_size_bytes: int) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ingest_file_audit (run_id, source_file, file_size_bytes, status)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, source_file, file_size_bytes, "running"),
        )
    conn.commit()


def update_file_audit_stage(
    conn,
    run_id: int,
    source_file: str,
    total_lines: int,
    event_lines: int,
    parse_errors: int,
    stage_rows: int,
    min_timestamp_ns: Optional[int],
    max_timestamp_ns: Optional[int],
    status: str,
    error_message: Optional[str] = None,
) -> None:
    min_day = utc_day_from_ns(min_timestamp_ns)
    max_day = utc_day_from_ns(max_timestamp_ns)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE ingest_file_audit
            SET total_lines = %s,
                event_lines = %s,
                parse_errors = %s,
                stage_rows = %s,
                min_timestamp_ns = %s,
                max_timestamp_ns = %s,
                min_day = %s,
                max_day = %s,
                status = %s,
                error_message = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = %s AND source_file = %s
            """,
            (
                total_lines,
                event_lines,
                parse_errors,
                stage_rows,
                min_timestamp_ns,
                max_timestamp_ns,
                min_day,
                max_day,
                status,
                error_message,
                run_id,
                source_file,
            ),
        )
    conn.commit()


def insert_stage_batch(conn, batch: List[Tuple[Any, ...]], page_size: int) -> None:
    if not batch:
        return
    del page_size
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in batch:
        serialized = []
        for index, value in enumerate(row):
            if value is None:
                serialized.append(r"\N")
            else:
                serialized.append(value)
        writer.writerow(serialized)
    buffer.seek(0)

    copy_sql = """
        COPY event_stage_raw (
            run_id, source_file, line_no, event_fingerprint, event_uuid, event_type, timestamp_ns, sequence_num,
            thread_id, host_id, subject_uuid, object_uuid, object2_uuid, object_path, object2_path,
            event_name, size_bytes, exec_name, parent_pid, file_descriptor, return_value,
            cdm_version, source_tag, raw_record
        )
        FROM STDIN WITH (FORMAT CSV, NULL '\\N')
    """
    with conn.cursor() as cursor:
        cursor.copy_expert(copy_sql, buffer)


def extract_uuid(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for nested_value in value.values():
            if isinstance(nested_value, str):
                return nested_value
    return None


def extract_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for nested_value in value.values():
            if isinstance(nested_value, str):
                return nested_value
    return None


def extract_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for nested_value in value.values():
            if isinstance(nested_value, int):
                return nested_value
    return None


def parse_json_line(raw_line: bytes) -> Dict[str, Any]:
    if orjson is not None:
        return orjson.loads(raw_line)
    return json.loads(raw_line.decode("utf-8", errors="ignore"))


def parse_event_record(record: Dict[str, Any], raw_record_text: str) -> Optional[Dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or EVENT_KEY not in datum:
        return None
    event = datum[EVENT_KEY]
    if not isinstance(event, dict):
        return None
    props = event.get("properties", {})
    props_map = props.get("map", {}) if isinstance(props, dict) else {}
    event_uuid = event.get("uuid")
    if not isinstance(event_uuid, str) or not event_uuid:
        return None

    parsed = {
        "event_fingerprint": hashlib.md5(raw_record_text.encode("utf-8")).hexdigest(),
        "event_uuid": event_uuid,
        "event_type": event.get("type"),
        "timestamp_ns": event.get("timestampNanos"),
        "sequence_num": extract_int(event.get("sequence")),
        "thread_id": extract_int(event.get("threadId")),
        "host_id": event.get("hostId"),
        "subject_uuid": extract_uuid(event.get("subject")),
        "object_uuid": extract_uuid(event.get("predicateObject")),
        "object2_uuid": extract_uuid(event.get("predicateObject2")),
        "object_path": extract_string(event.get("predicateObjectPath")),
        "object2_path": extract_string(event.get("predicateObject2Path")),
        "event_name": extract_string(event.get("name")),
        "size_bytes": extract_int(event.get("size")),
        "exec_name": props_map.get("exec") if isinstance(props_map, dict) else None,
        "parent_pid": props_map.get("ppid") if isinstance(props_map, dict) else None,
        "file_descriptor": props_map.get("fd") if isinstance(props_map, dict) else None,
        "return_value": props_map.get("return_value") if isinstance(props_map, dict) else None,
        "cdm_version": record.get("CDMVersion"),
        "source_tag": record.get("source"),
        "raw_record_text": raw_record_text,
    }
    return parsed


def batch_row_from_event(run_id: int, source_file: str, line_no: int, parsed: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        run_id,
        source_file,
        line_no,
        parsed["event_fingerprint"],
        parsed["event_uuid"],
        parsed["event_type"],
        parsed["timestamp_ns"],
        parsed["sequence_num"],
        parsed["thread_id"],
        parsed["host_id"],
        parsed["subject_uuid"],
        parsed["object_uuid"],
        parsed["object2_uuid"],
        parsed["object_path"],
        parsed["object2_path"],
        parsed["event_name"],
        parsed["size_bytes"],
        parsed["exec_name"],
        parsed["parent_pid"],
        parsed["file_descriptor"],
        parsed["return_value"],
        parsed["cdm_version"],
        parsed["source_tag"],
        parsed["raw_record_text"],
    )


def print_progress(source_file: str, total_lines: int, event_lines: int, stage_rows: int) -> None:
    print(
        f"[ingest] {os.path.basename(source_file)} "
        f"lines={total_lines:,} events={event_lines:,} staged={stage_rows:,}"
    )


def create_stage_indexes(conn) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_event_stage_raw_run_source_line ON event_stage_raw (run_id, source_file, line_no)",
        "CREATE INDEX IF NOT EXISTS idx_event_stage_raw_run_fingerprint ON event_stage_raw (run_id, event_fingerprint)",
    ]
    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    conn.commit()


def rebuild_events_raw(conn, run_id: int, use_stage_fingerprint: bool = True) -> None:
    fingerprint_expr = "event_fingerprint" if use_stage_fingerprint else "md5(raw_record::text)"
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM events_raw")
        cursor.execute(
            f"""
            WITH ranked AS (
                SELECT
                    run_id,
                    source_file,
                    line_no,
                    {fingerprint_expr} AS event_fingerprint,
                    event_uuid,
                    event_type,
                    timestamp_ns,
                    sequence_num,
                    thread_id,
                    host_id,
                    subject_uuid,
                    object_uuid,
                    object2_uuid,
                    object_path,
                    object2_path,
                    event_name,
                    size_bytes,
                    exec_name,
                    parent_pid,
                    file_descriptor,
                    return_value,
                    cdm_version,
                    source_tag,
                    ROW_NUMBER() OVER (
                        PARTITION BY {fingerprint_expr}
                        ORDER BY source_file, line_no
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY {fingerprint_expr}) AS duplicate_count
                FROM event_stage_raw
                WHERE run_id = %s
            )
            INSERT INTO events_raw (
                event_fingerprint, run_id, source_file, source_line_no, duplicate_count, event_uuid, event_type,
                timestamp_ns, sequence_num, thread_id, host_id, subject_uuid, object_uuid,
                object2_uuid, object_path, object2_path, event_name, size_bytes, exec_name,
                parent_pid, file_descriptor, return_value, cdm_version, source_tag
            )
            SELECT
                event_fingerprint, run_id, source_file, line_no, duplicate_count, event_uuid, event_type,
                timestamp_ns, sequence_num, thread_id, host_id, subject_uuid, object_uuid,
                object2_uuid, object_path, object2_path, event_name, size_bytes, exec_name,
                parent_pid, file_descriptor, return_value, cdm_version, source_tag
            FROM ranked
            WHERE rn = 1
            """,
            (run_id,),
        )
    conn.commit()


def create_events_raw_indexes(conn) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_events_raw_event_uuid ON events_raw (event_uuid)",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_timestamp ON events_raw (timestamp_ns)",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_subject ON events_raw (subject_uuid) WHERE subject_uuid IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_object ON events_raw (object_uuid) WHERE object_uuid IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_object2 ON events_raw (object2_uuid) WHERE object2_uuid IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_type ON events_raw (event_type)",
        "CREATE INDEX IF NOT EXISTS idx_events_raw_exec ON events_raw (exec_name) WHERE exec_name IS NOT NULL",
    ]
    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    conn.commit()


def update_file_audit_canonical_stats(conn, run_id: int, use_stage_fingerprint: bool = True) -> None:
    fingerprint_expr = "event_fingerprint" if use_stage_fingerprint else "md5(raw_record::text)"
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            WITH ranked AS (
                SELECT
                    source_file,
                    ROW_NUMBER() OVER (
                        PARTITION BY {fingerprint_expr}
                        ORDER BY source_file, line_no
                    ) AS rn
                FROM event_stage_raw
                WHERE run_id = %s
            ),
            aggregated AS (
                SELECT
                    source_file,
                    COUNT(*) FILTER (WHERE rn = 1) AS canonical_rows,
                    COUNT(*) FILTER (WHERE rn > 1) AS duplicate_uuid_rows
                FROM ranked
                GROUP BY source_file
            )
            UPDATE ingest_file_audit AS audit
            SET canonical_rows = aggregated.canonical_rows,
                duplicate_uuid_rows = aggregated.duplicate_uuid_rows,
                status = 'success',
                updated_at = CURRENT_TIMESTAMP
            FROM aggregated
            WHERE audit.run_id = %s
              AND audit.source_file = aggregated.source_file
            """,
            (run_id, run_id),
        )
    conn.commit()


def mark_file_audit_failed(conn, run_id: int, source_file: str, error_message: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE ingest_file_audit
            SET status = 'failed',
                error_message = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = %s AND source_file = %s
            """,
            (error_message[:5000], run_id, source_file),
        )
    conn.commit()


def get_latest_run_id(conn) -> Optional[int]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT run_id FROM ingest_runs ORDER BY run_id DESC LIMIT 1")
        row = cursor.fetchone()
    return row[0] if row else None


def fetch_run_summary(conn, run_id: int) -> Dict[str, Any]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id, started_at, finished_at, source_glob, status, source_file_count, notes
            FROM ingest_runs
            WHERE run_id = %s
            """,
            (run_id,),
        )
        run_row = cursor.fetchone()
        if not run_row:
            raise ValueError(f"Run {run_id} does not exist.")

        cursor.execute(
            """
            SELECT
                source_file,
                total_lines,
                event_lines,
                parse_errors,
                stage_rows,
                canonical_rows,
                duplicate_uuid_rows,
                min_day,
                max_day,
                status
            FROM ingest_file_audit
            WHERE run_id = %s
            ORDER BY source_file
            """,
            (run_id,),
        )
        file_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                COUNT(*) AS canonical_events,
                MIN(timestamp_ns) AS min_timestamp_ns,
                MAX(timestamp_ns) AS max_timestamp_ns,
                SUM(CASE WHEN duplicate_count > 1 THEN duplicate_count - 1 ELSE 0 END) AS duplicate_rows
            FROM events_raw
            WHERE run_id = %s
            """,
            (run_id,),
        )
        overall_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT
                to_char(timezone('UTC', to_timestamp(timestamp_ns / 1000000000.0)), 'YYYY-MM-DD') AS day,
                COUNT(*) AS row_count
            FROM events_raw
            WHERE run_id = %s
            GROUP BY 1
            ORDER BY 1
            """,
            (run_id,),
        )
        day_rows = cursor.fetchall()

    return {
        "run": run_row,
        "files": file_rows,
        "overall": overall_row,
        "days": day_rows,
    }


def print_run_summary(summary: Dict[str, Any]) -> None:
    run_id, started_at, finished_at, source_glob, status, source_file_count, notes = summary["run"]
    canonical_events, min_timestamp_ns, max_timestamp_ns, duplicate_rows = summary["overall"]

    print("=" * 88)
    print(f"Run ID: {run_id}")
    print(f"Status: {status}")
    print(f"Source glob: {source_glob}")
    print(f"Source files: {source_file_count}")
    print(f"Started at: {started_at}")
    print(f"Finished at: {finished_at}")
    if notes:
        print(f"Notes: {notes}")
    print(f"Canonical events: {canonical_events:,}" if canonical_events is not None else "Canonical events: 0")
    print(f"Global min timestamp: {timestamp_to_text(min_timestamp_ns)}")
    print(f"Global max timestamp: {timestamp_to_text(max_timestamp_ns)}")
    print(f"Duplicate stage rows: {duplicate_rows or 0:,}")
    print("=" * 88)
    print("Per-file audit")
    for row in summary["files"]:
        (
            source_file,
            total_lines,
            event_lines,
            parse_errors,
            stage_rows,
            canonical_rows,
            duplicate_uuid_rows,
            min_day,
            max_day,
            status,
        ) = row
        print(
            f"- {os.path.basename(source_file)} | status={status} | "
            f"lines={total_lines:,} | events={event_lines:,} | stage={stage_rows:,} | "
            f"canonical={canonical_rows:,} | duplicates={duplicate_uuid_rows:,} | "
            f"parse_errors={parse_errors:,} | days={min_day} -> {max_day}"
        )
    print("=" * 88)
    print("Daily counts")
    for day, row_count in summary["days"]:
        print(f"- {day}: {row_count:,}")
    print("=" * 88)


def fail_and_exit(message: str, exit_code: int = 1) -> None:
    print(f"[error] {message}", file=sys.stderr)
    raise SystemExit(exit_code)
