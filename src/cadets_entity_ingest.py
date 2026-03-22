import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from psycopg2.extras import execute_values
from tqdm import tqdm

from cadets_event_ingest_common import (
    build_common_parser,
    connect_target_db,
    discover_source_files,
    extract_int,
    extract_string,
    extract_uuid,
    load_ingest_config,
    parse_json_line,
    tune_ingest_session,
)


SUBJECT_KEY = "com.bbn.tc.schema.avro.cdm18.Subject"
FILE_OBJECT_KEY = "com.bbn.tc.schema.avro.cdm18.FileObject"
NETFLOW_KEY = "com.bbn.tc.schema.avro.cdm18.NetFlowObject"


def ensure_entity_schema(conn) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS entity_ingest_runs (
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
        CREATE TABLE IF NOT EXISTS entity_ingest_file_audit (
            audit_id BIGSERIAL PRIMARY KEY,
            run_id BIGINT NOT NULL REFERENCES entity_ingest_runs(run_id) ON DELETE CASCADE,
            source_file TEXT NOT NULL,
            file_size_bytes BIGINT NOT NULL,
            total_lines BIGINT NOT NULL DEFAULT 0,
            subject_rows BIGINT NOT NULL DEFAULT 0,
            file_rows BIGINT NOT NULL DEFAULT 0,
            network_rows BIGINT NOT NULL DEFAULT 0,
            parse_errors BIGINT NOT NULL DEFAULT 0,
            status VARCHAR(32) NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (run_id, source_file)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS process_entities (
            uuid VARCHAR(255) PRIMARY KEY,
            subject_type VARCHAR(100),
            cid BIGINT,
            parent_subject_uuid VARCHAR(255),
            host_id VARCHAR(255),
            local_principal_uuid VARCHAR(255),
            start_timestamp_ns BIGINT,
            cmd_line TEXT,
            privilege_level VARCHAR(255),
            cdm_version VARCHAR(32),
            source_tag VARCHAR(255),
            seen_count BIGINT NOT NULL DEFAULT 1,
            first_source_file TEXT NOT NULL,
            first_line_no BIGINT NOT NULL,
            last_source_file TEXT NOT NULL,
            last_line_no BIGINT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS file_entities (
            uuid VARCHAR(255) PRIMARY KEY,
            file_type VARCHAR(100),
            host_id VARCHAR(255),
            permission_value TEXT,
            epoch_value BIGINT,
            local_principal_uuid VARCHAR(255),
            file_descriptor VARCHAR(255),
            size_bytes BIGINT,
            cdm_version VARCHAR(32),
            source_tag VARCHAR(255),
            seen_count BIGINT NOT NULL DEFAULT 1,
            first_source_file TEXT NOT NULL,
            first_line_no BIGINT NOT NULL,
            last_source_file TEXT NOT NULL,
            last_line_no BIGINT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS network_entities (
            uuid VARCHAR(255) PRIMARY KEY,
            host_id VARCHAR(255),
            local_address VARCHAR(255),
            local_port INTEGER,
            remote_address VARCHAR(255),
            remote_port INTEGER,
            ip_protocol VARCHAR(255),
            file_descriptor VARCHAR(255),
            cdm_version VARCHAR(32),
            source_tag VARCHAR(255),
            seen_count BIGINT NOT NULL DEFAULT 1,
            first_source_file TEXT NOT NULL,
            first_line_no BIGINT NOT NULL,
            last_source_file TEXT NOT NULL,
            last_line_no BIGINT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    conn.commit()


def truncate_entity_tables(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            TRUNCATE TABLE
                process_entities,
                file_entities,
                network_entities
            """
        )
    conn.commit()


def create_run(conn, source_glob: str, source_file_count: int) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO entity_ingest_runs (source_glob, status, source_file_count)
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
            UPDATE entity_ingest_runs
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
            INSERT INTO entity_ingest_file_audit (run_id, source_file, file_size_bytes, status)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, source_file, file_size_bytes, "running"),
        )
    conn.commit()


def update_file_audit(
    conn,
    run_id: int,
    source_file: str,
    total_lines: int,
    subject_rows: int,
    file_rows: int,
    network_rows: int,
    parse_errors: int,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE entity_ingest_file_audit
            SET total_lines = %s,
                subject_rows = %s,
                file_rows = %s,
                network_rows = %s,
                parse_errors = %s,
                status = %s,
                error_message = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = %s AND source_file = %s
            """,
            (
                total_lines,
                subject_rows,
                file_rows,
                network_rows,
                parse_errors,
                status,
                error_message,
                run_id,
                source_file,
            ),
        )
    conn.commit()


def merge_value(existing: Any, new_value: Any) -> Any:
    if existing not in (None, ""):
        return existing
    return new_value


def merge_entity_record(existing: Dict[str, Any], new_record: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in new_record.items():
        if key in {"uuid", "first_source_file", "first_line_no"}:
            continue
        if key == "seen_count":
            merged[key] = int(merged.get(key, 0)) + int(value)
            continue
        if key in {"last_source_file", "last_line_no"}:
            merged[key] = value
            continue
        merged[key] = merge_value(merged.get(key), value)
    return merged


def parse_subject_record(record: Dict[str, Any], source_file: str, line_no: int) -> Optional[Dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or SUBJECT_KEY not in datum:
        return None
    subject = datum[SUBJECT_KEY]
    if not isinstance(subject, dict):
        return None
    subject_uuid = subject.get("uuid")
    if not isinstance(subject_uuid, str) or not subject_uuid:
        return None
    return {
        "uuid": subject_uuid.upper(),
        "subject_type": extract_string(subject.get("type")),
        "cid": extract_int(subject.get("cid")),
        "parent_subject_uuid": extract_uuid(subject.get("parentSubject")),
        "host_id": extract_string(subject.get("hostId")),
        "local_principal_uuid": extract_uuid(subject.get("localPrincipal")),
        "start_timestamp_ns": extract_int(subject.get("startTimestampNanos")),
        "cmd_line": extract_string(subject.get("cmdLine")),
        "privilege_level": extract_string(subject.get("privilegeLevel")),
        "cdm_version": extract_string(record.get("CDMVersion")),
        "source_tag": extract_string(record.get("source")),
        "seen_count": 1,
        "first_source_file": source_file,
        "first_line_no": line_no,
        "last_source_file": source_file,
        "last_line_no": line_no,
    }


def parse_file_record(record: Dict[str, Any], source_file: str, line_no: int) -> Optional[Dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or FILE_OBJECT_KEY not in datum:
        return None
    file_object = datum[FILE_OBJECT_KEY]
    if not isinstance(file_object, dict):
        return None
    file_uuid = file_object.get("uuid")
    if not isinstance(file_uuid, str) or not file_uuid:
        return None
    base_object = file_object.get("baseObject") if isinstance(file_object.get("baseObject"), dict) else {}
    return {
        "uuid": file_uuid.upper(),
        "file_type": extract_string(file_object.get("type")),
        "host_id": extract_string(base_object.get("hostId")),
        "permission_value": extract_string(base_object.get("permission")),
        "epoch_value": extract_int(base_object.get("epoch")),
        "local_principal_uuid": extract_uuid(file_object.get("localPrincipal")),
        "file_descriptor": extract_string(file_object.get("fileDescriptor")),
        "size_bytes": extract_int(file_object.get("size")),
        "cdm_version": extract_string(record.get("CDMVersion")),
        "source_tag": extract_string(record.get("source")),
        "seen_count": 1,
        "first_source_file": source_file,
        "first_line_no": line_no,
        "last_source_file": source_file,
        "last_line_no": line_no,
    }


def parse_netflow_record(record: Dict[str, Any], source_file: str, line_no: int) -> Optional[Dict[str, Any]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or NETFLOW_KEY not in datum:
        return None
    netflow = datum[NETFLOW_KEY]
    if not isinstance(netflow, dict):
        return None
    net_uuid = netflow.get("uuid")
    if not isinstance(net_uuid, str) or not net_uuid:
        return None
    base_object = netflow.get("baseObject") if isinstance(netflow.get("baseObject"), dict) else {}
    return {
        "uuid": net_uuid.upper(),
        "host_id": extract_string(base_object.get("hostId")),
        "local_address": extract_string(netflow.get("localAddress")),
        "local_port": extract_int(netflow.get("localPort")),
        "remote_address": extract_string(netflow.get("remoteAddress")),
        "remote_port": extract_int(netflow.get("remotePort")),
        "ip_protocol": extract_string(netflow.get("ipProtocol")),
        "file_descriptor": extract_string(netflow.get("fileDescriptor")),
        "cdm_version": extract_string(record.get("CDMVersion")),
        "source_tag": extract_string(record.get("source")),
        "seen_count": 1,
        "first_source_file": source_file,
        "first_line_no": line_no,
        "last_source_file": source_file,
        "last_line_no": line_no,
    }


def flush_process_batch(conn, batch: Dict[str, Dict[str, Any]]) -> None:
    if not batch:
        return
    sql = """
        INSERT INTO process_entities (
            uuid, subject_type, cid, parent_subject_uuid, host_id, local_principal_uuid,
            start_timestamp_ns, cmd_line, privilege_level, cdm_version, source_tag, seen_count,
            first_source_file, first_line_no, last_source_file, last_line_no
        ) VALUES %s
        ON CONFLICT (uuid) DO UPDATE SET
            subject_type = COALESCE(process_entities.subject_type, EXCLUDED.subject_type),
            cid = COALESCE(process_entities.cid, EXCLUDED.cid),
            parent_subject_uuid = COALESCE(process_entities.parent_subject_uuid, EXCLUDED.parent_subject_uuid),
            host_id = COALESCE(process_entities.host_id, EXCLUDED.host_id),
            local_principal_uuid = COALESCE(process_entities.local_principal_uuid, EXCLUDED.local_principal_uuid),
            start_timestamp_ns = COALESCE(process_entities.start_timestamp_ns, EXCLUDED.start_timestamp_ns),
            cmd_line = COALESCE(process_entities.cmd_line, EXCLUDED.cmd_line),
            privilege_level = COALESCE(process_entities.privilege_level, EXCLUDED.privilege_level),
            cdm_version = COALESCE(process_entities.cdm_version, EXCLUDED.cdm_version),
            source_tag = COALESCE(process_entities.source_tag, EXCLUDED.source_tag),
            seen_count = process_entities.seen_count + EXCLUDED.seen_count,
            last_source_file = EXCLUDED.last_source_file,
            last_line_no = EXCLUDED.last_line_no,
            updated_at = CURRENT_TIMESTAMP
    """
    values = [
        (
            row["uuid"],
            row["subject_type"],
            row["cid"],
            row["parent_subject_uuid"],
            row["host_id"],
            row["local_principal_uuid"],
            row["start_timestamp_ns"],
            row["cmd_line"],
            row["privilege_level"],
            row["cdm_version"],
            row["source_tag"],
            row["seen_count"],
            row["first_source_file"],
            row["first_line_no"],
            row["last_source_file"],
            row["last_line_no"],
        )
        for row in batch.values()
    ]
    with conn.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=1000)


def flush_file_batch(conn, batch: Dict[str, Dict[str, Any]]) -> None:
    if not batch:
        return
    sql = """
        INSERT INTO file_entities (
            uuid, file_type, host_id, permission_value, epoch_value, local_principal_uuid,
            file_descriptor, size_bytes, cdm_version, source_tag, seen_count,
            first_source_file, first_line_no, last_source_file, last_line_no
        ) VALUES %s
        ON CONFLICT (uuid) DO UPDATE SET
            file_type = COALESCE(file_entities.file_type, EXCLUDED.file_type),
            host_id = COALESCE(file_entities.host_id, EXCLUDED.host_id),
            permission_value = COALESCE(file_entities.permission_value, EXCLUDED.permission_value),
            epoch_value = COALESCE(file_entities.epoch_value, EXCLUDED.epoch_value),
            local_principal_uuid = COALESCE(file_entities.local_principal_uuid, EXCLUDED.local_principal_uuid),
            file_descriptor = COALESCE(file_entities.file_descriptor, EXCLUDED.file_descriptor),
            size_bytes = COALESCE(file_entities.size_bytes, EXCLUDED.size_bytes),
            cdm_version = COALESCE(file_entities.cdm_version, EXCLUDED.cdm_version),
            source_tag = COALESCE(file_entities.source_tag, EXCLUDED.source_tag),
            seen_count = file_entities.seen_count + EXCLUDED.seen_count,
            last_source_file = EXCLUDED.last_source_file,
            last_line_no = EXCLUDED.last_line_no,
            updated_at = CURRENT_TIMESTAMP
    """
    values = [
        (
            row["uuid"],
            row["file_type"],
            row["host_id"],
            row["permission_value"],
            row["epoch_value"],
            row["local_principal_uuid"],
            row["file_descriptor"],
            row["size_bytes"],
            row["cdm_version"],
            row["source_tag"],
            row["seen_count"],
            row["first_source_file"],
            row["first_line_no"],
            row["last_source_file"],
            row["last_line_no"],
        )
        for row in batch.values()
    ]
    with conn.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=1000)


def flush_network_batch(conn, batch: Dict[str, Dict[str, Any]]) -> None:
    if not batch:
        return
    sql = """
        INSERT INTO network_entities (
            uuid, host_id, local_address, local_port, remote_address, remote_port,
            ip_protocol, file_descriptor, cdm_version, source_tag, seen_count,
            first_source_file, first_line_no, last_source_file, last_line_no
        ) VALUES %s
        ON CONFLICT (uuid) DO UPDATE SET
            host_id = COALESCE(network_entities.host_id, EXCLUDED.host_id),
            local_address = COALESCE(network_entities.local_address, EXCLUDED.local_address),
            local_port = COALESCE(network_entities.local_port, EXCLUDED.local_port),
            remote_address = COALESCE(network_entities.remote_address, EXCLUDED.remote_address),
            remote_port = COALESCE(network_entities.remote_port, EXCLUDED.remote_port),
            ip_protocol = COALESCE(network_entities.ip_protocol, EXCLUDED.ip_protocol),
            file_descriptor = COALESCE(network_entities.file_descriptor, EXCLUDED.file_descriptor),
            cdm_version = COALESCE(network_entities.cdm_version, EXCLUDED.cdm_version),
            source_tag = COALESCE(network_entities.source_tag, EXCLUDED.source_tag),
            seen_count = network_entities.seen_count + EXCLUDED.seen_count,
            last_source_file = EXCLUDED.last_source_file,
            last_line_no = EXCLUDED.last_line_no,
            updated_at = CURRENT_TIMESTAMP
    """
    values = [
        (
            row["uuid"],
            row["host_id"],
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
            row["ip_protocol"],
            row["file_descriptor"],
            row["cdm_version"],
            row["source_tag"],
            row["seen_count"],
            row["first_source_file"],
            row["first_line_no"],
            row["last_source_file"],
            row["last_line_no"],
        )
        for row in batch.values()
    ]
    with conn.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=1000)


def flush_all_batches(
    conn,
    process_batch: Dict[str, Dict[str, Any]],
    file_batch: Dict[str, Dict[str, Any]],
    network_batch: Dict[str, Dict[str, Any]],
) -> None:
    flush_process_batch(conn, process_batch)
    flush_file_batch(conn, file_batch)
    flush_network_batch(conn, network_batch)
    process_batch.clear()
    file_batch.clear()
    network_batch.clear()


def count_entities(conn) -> Dict[str, int]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM process_entities),
                (SELECT COUNT(*) FROM file_entities),
                (SELECT COUNT(*) FROM network_entities)
            """
        )
        row = cursor.fetchone()
    return {"process": int(row[0]), "file": int(row[1]), "network": int(row[2])}


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Ingest Subject/FileObject/NetFlowObject into local entity tables.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not truncate process_entities/file_entities/network_entities before rebuilding.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_ingest_config(args)
    source_files = discover_source_files(config.source_glob, config.file_limit)

    conn = connect_target_db(config.database)
    try:
        tune_ingest_session(conn)
        ensure_entity_schema(conn)
        if not args.keep_existing:
            print("[entity] truncating process_entities/file_entities/network_entities")
            truncate_entity_tables(conn)

        run_id = create_run(conn, config.source_glob, len(source_files))
        print(f"[entity] created ingest run {run_id}")

        process_batch: Dict[str, Dict[str, Any]] = {}
        file_batch: Dict[str, Dict[str, Any]] = {}
        network_batch: Dict[str, Dict[str, Any]] = {}
        pending_flushes = 0

        outer_bar = tqdm(source_files, desc="entity files", unit="file")
        try:
            for source_file in outer_bar:
                outer_bar.set_postfix(current=os.path.basename(source_file))
                file_size = os.path.getsize(source_file)
                create_file_audit(conn, run_id, source_file, file_size)

                total_lines = 0
                subject_rows = 0
                file_rows = 0
                network_rows = 0
                parse_errors = 0

                try:
                    with open(source_file, "rb") as handle, tqdm(
                        total=file_size,
                        desc=os.path.basename(source_file),
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    ) as file_bar:
                        for line_no, raw_line in enumerate(handle, start=1):
                            total_lines += 1
                            file_bar.update(len(raw_line))
                            if config.line_limit is not None and total_lines > config.line_limit:
                                break

                            try:
                                record = parse_json_line(raw_line)
                            except Exception:
                                parse_errors += 1
                                continue

                            subject_row = parse_subject_record(record, source_file, line_no)
                            if subject_row is not None:
                                subject_rows += 1
                                existing = process_batch.get(subject_row["uuid"])
                                process_batch[subject_row["uuid"]] = (
                                    subject_row if existing is None else merge_entity_record(existing, subject_row)
                                )

                            file_row = parse_file_record(record, source_file, line_no)
                            if file_row is not None:
                                file_rows += 1
                                existing = file_batch.get(file_row["uuid"])
                                file_batch[file_row["uuid"]] = (
                                    file_row if existing is None else merge_entity_record(existing, file_row)
                                )

                            network_row = parse_netflow_record(record, source_file, line_no)
                            if network_row is not None:
                                network_rows += 1
                                existing = network_batch.get(network_row["uuid"])
                                network_batch[network_row["uuid"]] = (
                                    network_row if existing is None else merge_entity_record(existing, network_row)
                                )

                            total_pending = len(process_batch) + len(file_batch) + len(network_batch)
                            if total_pending >= config.batch_size:
                                flush_all_batches(conn, process_batch, file_batch, network_batch)
                                pending_flushes += 1
                                if pending_flushes >= config.commit_every_batches:
                                    conn.commit()
                                    pending_flushes = 0

                    flush_all_batches(conn, process_batch, file_batch, network_batch)
                    conn.commit()
                    pending_flushes = 0
                    update_file_audit(
                        conn,
                        run_id,
                        source_file,
                        total_lines,
                        subject_rows,
                        file_rows,
                        network_rows,
                        parse_errors,
                        "success",
                    )
                    print(
                        f"[entity] {os.path.basename(source_file)} "
                        f"lines={total_lines:,} subject={subject_rows:,} file={file_rows:,} network={network_rows:,}"
                    )
                except Exception as exc:
                    conn.rollback()
                    update_file_audit(
                        conn,
                        run_id,
                        source_file,
                        total_lines,
                        subject_rows,
                        file_rows,
                        network_rows,
                        parse_errors,
                        "failed",
                        str(exc),
                    )
                    raise
        finally:
            outer_bar.close()

        totals = count_entities(conn)
        finalize_run(
            conn,
            run_id,
            "success",
            notes=(
                f"process_entities={totals['process']:,}, "
                f"file_entities={totals['file']:,}, "
                f"network_entities={totals['network']:,}"
            ),
        )
        print("[entity] ingest finished successfully")
        print(f"[entity] process_entities: {totals['process']:,}")
        print(f"[entity] file_entities: {totals['file']:,}")
        print(f"[entity] network_entities: {totals['network']:,}")
    except Exception as exc:
        conn.rollback()
        latest_run_id = None
        with conn.cursor() as cursor:
            cursor.execute("SELECT run_id FROM entity_ingest_runs ORDER BY run_id DESC LIMIT 1")
            row = cursor.fetchone()
            latest_run_id = row[0] if row else None
        if latest_run_id is not None:
            finalize_run(conn, latest_run_id, "failed", notes=str(exc))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
