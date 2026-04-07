import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "cadets_event_ingest.json"
DEFAULT_SPLIT_MANIFEST_PATH = REPO_ROOT / "artifacts" / "day_split" / "split_manifest.json"
DEFAULT_NODE_RETENTION_MANIFEST = REPO_ROOT / "artifacts" / "node_retention" / "retention_manifest.json"
DEFAULT_EXTRACT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_file_v2"
DEFAULT_CLEAN_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_cleaned_file_v2"
DEFAULT_ENCODE_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_model_ready_file_v2"
DEFAULT_BASELINE_FEATURE_ROOT = REPO_ROOT / "artifacts" / "features"
DEFAULT_FILE_ONLY_EXTRACT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_file_only_v2"
DEFAULT_FILE_ONLY_CLEAN_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_cleaned_file_only_v2"
DEFAULT_FILE_ONLY_ENCODE_OUTPUT_DIR = REPO_ROOT / "artifacts" / "features_model_ready_file_only_v2"
FILE_GROUP_FILES = [
    "file_view__file_node.tsv",
    "process_view__file_node.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CPU-side file feature v2 pipeline that re-materializes extracted, cleaned, and "
            "model-ready features into independent v2 directories."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the ingest config JSON. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=str(DEFAULT_SPLIT_MANIFEST_PATH),
        help=f"Path to split_manifest.json. Default: {DEFAULT_SPLIT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--node-retention-manifest",
        type=str,
        default=str(DEFAULT_NODE_RETENTION_MANIFEST),
        help=f"Path to node_retention/retention_manifest.json. Default: {DEFAULT_NODE_RETENTION_MANIFEST}",
    )
    parser.add_argument(
        "--extract-output-dir",
        type=str,
        default=str(DEFAULT_EXTRACT_OUTPUT_DIR),
        help=f"Directory for v2 extracted features. Default: {DEFAULT_EXTRACT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--clean-output-dir",
        type=str,
        default=str(DEFAULT_CLEAN_OUTPUT_DIR),
        help=f"Directory for v2 cleaned features. Default: {DEFAULT_CLEAN_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--encode-output-dir",
        type=str,
        default=str(DEFAULT_ENCODE_OUTPUT_DIR),
        help=f"Directory for v2 model-ready features. Default: {DEFAULT_ENCODE_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--baseline-feature-root",
        type=str,
        default=str(DEFAULT_BASELINE_FEATURE_ROOT),
        help=f"Directory of the current extracted baseline features for column diffing. Default: {DEFAULT_BASELINE_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--file-only",
        action="store_true",
        help=(
            "Run the faster file-only v2 extractor/clean/encode chain. "
            "This only materializes process_view__file_node and file_view__file_node."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip the raw feature extraction stage and reuse an existing extracted v2 directory.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip the cleaned-feature stage and reuse an existing cleaned v2 directory.",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Skip the model-ready encoding stage and reuse an existing encoded v2 directory.",
    )
    return parser.parse_args()


def run_command(command: List[str]) -> None:
    print(f"[feature-v2-pipeline] running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def summarize_group_columns(label: str, feature_root: Path) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    train_dir = feature_root / "train"
    for filename in FILE_GROUP_FILES:
        path = train_dir / filename
        if not path.exists():
            summary[filename] = {
                "path": str(path),
                "exists": False,
                "columns": [],
            }
            continue
        header = read_header(path)
        summary[filename] = {
            "path": str(path),
            "exists": True,
            "columns": header,
        }
        print(
            f"[feature-v2-pipeline] {label} {filename}: columns={len(header)}",
            flush=True,
        )
    return summary


def print_column_diff(
    baseline_summary: Dict[str, Dict[str, object]],
    v2_summary: Dict[str, Dict[str, object]],
) -> None:
    for filename in FILE_GROUP_FILES:
        baseline_cols = set(baseline_summary.get(filename, {}).get("columns", []))
        v2_cols = set(v2_summary.get(filename, {}).get("columns", []))
        added = sorted(v2_cols - baseline_cols)
        removed = sorted(baseline_cols - v2_cols)
        print(f"[feature-v2-pipeline] diff {filename}", flush=True)
        print(f"  added={added}", flush=True)
        print(f"  removed={removed}", flush=True)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    split_manifest_path = Path(args.split_manifest).expanduser().resolve()
    node_retention_manifest = Path(args.node_retention_manifest).expanduser().resolve()
    extract_output_dir = Path(args.extract_output_dir).expanduser().resolve()
    clean_output_dir = Path(args.clean_output_dir).expanduser().resolve()
    encode_output_dir = Path(args.encode_output_dir).expanduser().resolve()
    baseline_feature_root = Path(args.baseline_feature_root).expanduser().resolve()

    extractor_script = REPO_ROOT / "src" / "cadets_feature_extract.py"
    if args.file_only:
        extractor_script = REPO_ROOT / "src" / "cadets_file_feature_extract_v2.py"
        if str(args.extract_output_dir) == str(DEFAULT_EXTRACT_OUTPUT_DIR):
            extract_output_dir = DEFAULT_FILE_ONLY_EXTRACT_OUTPUT_DIR.resolve()
        if str(args.clean_output_dir) == str(DEFAULT_CLEAN_OUTPUT_DIR):
            clean_output_dir = DEFAULT_FILE_ONLY_CLEAN_OUTPUT_DIR.resolve()
        if str(args.encode_output_dir) == str(DEFAULT_ENCODE_OUTPUT_DIR):
            encode_output_dir = DEFAULT_FILE_ONLY_ENCODE_OUTPUT_DIR.resolve()

    if not args.skip_extract:
        run_command(
            [
                sys.executable,
                str(extractor_script),
                "--config",
                str(config_path),
                "--split-manifest",
                str(split_manifest_path),
                "--output-dir",
                str(extract_output_dir),
            ]
        )

    if not args.skip_clean:
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "src" / "cadets_feature_clean.py"),
                "--feature-manifest",
                str(extract_output_dir / "feature_manifest.json"),
                "--node-retention-manifest",
                str(node_retention_manifest),
                "--output-dir",
                str(clean_output_dir),
            ]
        )

    if not args.skip_encode:
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "src" / "cadets_feature_encode.py"),
                "--feature-manifest",
                str(clean_output_dir / "feature_manifest.json"),
                "--output-dir",
                str(encode_output_dir),
            ]
        )

    baseline_summary = summarize_group_columns("baseline", baseline_feature_root)
    extracted_v2_summary = summarize_group_columns("v2-extracted", extract_output_dir)
    print_column_diff(baseline_summary, extracted_v2_summary)


if __name__ == "__main__":
    main()
