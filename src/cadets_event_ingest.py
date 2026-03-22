import json
import os
import traceback

from tqdm import tqdm

from cadets_event_ingest_common import (
    batch_row_from_event,
    build_common_parser,
    connect_target_db,
    create_events_raw_indexes,
    create_file_audit,
    create_run,
    create_stage_indexes,
    discover_source_files,
    drop_database_if_exists,
    ensure_database_exists,
    ensure_schema,
    fail_and_exit,
    finalize_run,
    insert_stage_batch,
    load_ingest_config,
    mark_file_audit_failed,
    parse_json_line,
    parse_event_record,
    print_run_summary,
    rebuild_events_raw,
    tune_ingest_session,
    update_file_audit_canonical_stats,
    update_file_audit_stage,
    fetch_run_summary,
)


def ingest_file(conn, config, run_id: int, source_file: str) -> None:
    file_size = os.path.getsize(source_file)
    create_file_audit(conn, run_id, source_file, file_size)

    total_lines = 0
    event_lines = 0
    parse_errors = 0
    stage_rows = 0
    min_timestamp_ns = None
    max_timestamp_ns = None
    batch = []
    pending_batches = 0
    line_limited = config.line_limit is not None
    progress_total = config.line_limit if line_limited else file_size
    progress_unit = "line" if line_limited else "B"
    progress_unit_scale = not line_limited

    try:
        with open(source_file, "rb") as handle, tqdm(
            total=progress_total,
            desc=os.path.basename(source_file),
            unit=progress_unit,
            unit_scale=progress_unit_scale,
            dynamic_ncols=True,
            leave=False,
        ) as file_pbar:
            for line_no, raw_line in enumerate(handle, start=1):
                if config.line_limit is not None and line_no > config.line_limit:
                    break
                total_lines += 1
                file_pbar.update(1 if line_limited else len(raw_line))
                stripped_bytes = raw_line.strip()
                if not stripped_bytes:
                    continue
                try:
                    record = parse_json_line(stripped_bytes)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                except ValueError:
                    parse_errors += 1
                    continue

                raw_record_text = stripped_bytes.decode("utf-8", errors="ignore")
                parsed = parse_event_record(record, raw_record_text)
                if parsed is None:
                    continue

                event_lines += 1
                timestamp_ns = parsed["timestamp_ns"]
                if timestamp_ns is not None:
                    min_timestamp_ns = timestamp_ns if min_timestamp_ns is None else min(min_timestamp_ns, timestamp_ns)
                    max_timestamp_ns = timestamp_ns if max_timestamp_ns is None else max(max_timestamp_ns, timestamp_ns)

                batch.append(batch_row_from_event(run_id, source_file, line_no, parsed))
                if len(batch) >= config.batch_size:
                    insert_stage_batch(conn, batch, config.page_size)
                    stage_rows += len(batch)
                    pending_batches += 1
                    batch.clear()
                    if pending_batches >= config.commit_every_batches:
                        conn.commit()
                        pending_batches = 0

                if total_lines % config.progress_every_lines == 0:
                    file_pbar.set_postfix(
                        lines=f"{total_lines:,}",
                        events=f"{event_lines:,}",
                        staged=f"{stage_rows + len(batch):,}",
                    )

        if batch:
            insert_stage_batch(conn, batch, config.page_size)
            stage_rows += len(batch)
            pending_batches += 1
            batch.clear()
        if pending_batches > 0:
            conn.commit()
            pending_batches = 0

        update_file_audit_stage(
            conn=conn,
            run_id=run_id,
            source_file=source_file,
            total_lines=total_lines,
            event_lines=event_lines,
            parse_errors=parse_errors,
            stage_rows=stage_rows,
            min_timestamp_ns=min_timestamp_ns,
            max_timestamp_ns=max_timestamp_ns,
            status="staged",
        )
        tqdm.write(
            f"[ingest] {os.path.basename(source_file)} done | "
            f"lines={total_lines:,} events={event_lines:,} staged={stage_rows:,} parse_errors={parse_errors:,}"
        )
    except Exception as exc:
        error_message = f"{exc.__class__.__name__}: {exc}"
        conn.rollback()
        update_file_audit_stage(
            conn=conn,
            run_id=run_id,
            source_file=source_file,
            total_lines=total_lines,
            event_lines=event_lines,
            parse_errors=parse_errors,
            stage_rows=stage_rows,
            min_timestamp_ns=min_timestamp_ns,
            max_timestamp_ns=max_timestamp_ns,
            status="failed",
            error_message=error_message,
        )
        mark_file_audit_failed(conn, run_id, source_file, error_message)
        raise


def main() -> None:
    parser = build_common_parser("Ingest CADETS Event records into the auditable two-layer database.")
    args = parser.parse_args()
    config = load_ingest_config(args)

    try:
        source_files = discover_source_files(config.source_glob, config.file_limit)
    except FileNotFoundError as exc:
        fail_and_exit(str(exc))

    print("[config] Target database:", config.database.target_db)
    print("[config] Source glob:", config.source_glob)
    print(f"[config] Source files discovered: {len(source_files)}")
    if config.reset_db:
        print("[config] Reset target database before ingest: enabled")
    if config.line_limit is not None:
        print(f"[config] Line limit per file: {config.line_limit}")
    if config.file_limit is not None:
        print(f"[config] File limit: {config.file_limit}")

    if config.reset_db:
        drop_database_if_exists(config.database)
    ensure_database_exists(config.database)
    conn = connect_target_db(config.database)
    run_id = None
    try:
        tune_ingest_session(conn)
        ensure_schema(conn)
        run_id = create_run(conn, config.source_glob, len(source_files))
        print(f"[run] Created ingest run {run_id}")

        with tqdm(total=len(source_files), desc="source files", unit="file", dynamic_ncols=True) as file_progress:
            for index, source_file in enumerate(source_files, start=1):
                file_progress.set_postfix(current=os.path.basename(source_file))
                tqdm.write(f"[run] [{index}/{len(source_files)}] staging {source_file}")
                ingest_file(conn, config, run_id, source_file)
                file_progress.update(1)

        print("[run] Creating stage indexes")
        create_stage_indexes(conn)
        print("[run] Rebuilding canonical events_raw")
        rebuild_events_raw(conn, run_id)
        print("[run] Creating canonical indexes")
        create_events_raw_indexes(conn)
        print("[run] Updating file audit canonical stats")
        update_file_audit_canonical_stats(conn, run_id)
        finalize_run(conn, run_id, "success")

        print("[run] Ingest completed successfully")
        summary = fetch_run_summary(conn, run_id)
        print_run_summary(summary)
    except Exception as exc:
        notes = f"{exc.__class__.__name__}: {exc}"
        conn.rollback()
        if run_id is not None:
            finalize_run(conn, run_id, "failed", notes=notes)
        traceback.print_exc()
        fail_and_exit(notes)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
