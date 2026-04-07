import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from cadets_event_ingest_common import build_common_parser, connect_target_db, load_ingest_config
from cadets_feature_extract import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    FILE_VIEW_FILE,
    FILE_VIEW_QUERY,
    PROCESS_VIEW_FILE_NODE_FILE,
    PROCESS_VIEW_FILE_NODE_QUERY,
    build_feature_windows,
    copy_query_to_tsv,
    ensure_output_dir,
    fetch_event_count,
    load_split_manifest,
    log,
    ns_to_text,
    print_summary,
    tune_feature_session,
    window_bounds,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features_file_only_v2"


def export_window_file_features(conn, window: Dict[str, object], output_dir: Path) -> Dict[str, object]:
    days = list(window["days"])
    start_ns, end_ns = window_bounds(days)
    window_dir = output_dir / str(window["name"])
    ensure_output_dir(window_dir)

    log(f"[file-feature-extract-v2] {window['name']}: preparing window {days[0]} -> {days[-1]}")
    event_count = fetch_event_count(conn, start_ns, end_ns)
    log(f"[file-feature-extract-v2] {window['name']}: event_count={event_count:,}")

    process_view_file_rows = copy_query_to_tsv(
        conn,
        PROCESS_VIEW_FILE_NODE_QUERY,
        (start_ns, end_ns, start_ns, end_ns, start_ns, end_ns),
        window_dir / PROCESS_VIEW_FILE_NODE_FILE,
        log_label=f"{window['name']} process_view__file_node",
    )

    file_rows = copy_query_to_tsv(
        conn,
        FILE_VIEW_QUERY,
        (start_ns, end_ns, start_ns, end_ns),
        window_dir / FILE_VIEW_FILE,
        log_label=f"{window['name']} file_view__file_node",
    )

    metadata = {
        "window_name": window["name"],
        "split": window["split"],
        "days": days,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "start_time_utc": ns_to_text(start_ns),
        "end_time_utc_exclusive": ns_to_text(end_ns),
        "event_count": event_count,
        "file_node_count": file_rows,
        "process_view__file_node_rows": process_view_file_rows,
        "file_view_rows": file_rows,
        "feature_files": {
            "process_view__file_node": PROCESS_VIEW_FILE_NODE_FILE,
            "file_view__file_node": FILE_VIEW_FILE,
        },
    }
    write_json(window_dir / "metadata.json", metadata)
    log(
        f"[file-feature-extract-v2] {window['name']}: done "
        f"process_view_file_rows={process_view_file_rows:,} "
        f"file_rows={file_rows:,}"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser("Extract file-focused CADETS node feature v2 tables.")
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
        help=f"Output directory for file-only v2 features. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print split/window counts without exporting TSV feature files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_ingest_config(args)
    split_manifest_path = Path(args.split_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    split_manifest = load_split_manifest(split_manifest_path)
    windows = build_feature_windows(split_manifest)

    if args.summary_only:
        window_metadata: List[Dict[str, object]] = []
        for window in windows:
            window_metadata.append(
                {
                    "window_name": window["name"],
                    "split": window["split"],
                    "days": list(window["days"]),
                }
            )
        print_summary(window_metadata)
        return

    conn = connect_target_db(config.database)
    try:
        tune_feature_session(conn)
        window_metadata = []
        with tqdm(total=len(windows), desc="file feature windows", unit="window", dynamic_ncols=True) as window_bar:
            for window in windows:
                metadata = export_window_file_features(conn, window, output_dir)
                window_metadata.append(metadata)
                window_bar.update(1)

        print_summary(window_metadata)
        write_json(
            output_dir / "feature_manifest.json",
            {
                "split_manifest_path": str(split_manifest_path),
                "windows": window_metadata,
                "notes": [
                    "This file-focused v2 stage exports only process_view__file_node and file_view__file_node.",
                    "It is intended for faster CPU-side file feature iteration before full all-group re-export.",
                ],
            },
        )
        print(f"[file-feature-extract-v2] outputs written to: {output_dir}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
