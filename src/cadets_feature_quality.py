import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_FEATURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "features"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "feature_quality"

TEXT_COLUMNS = {
    "node_uuid",
    "node_type",
    "host_id",
    "subject_type",
    "cmd_line",
    "file_type",
    "file_descriptor",
    "permission_value",
    "local_address",
    "remote_address",
    "local_port_bucket",
    "remote_port_bucket",
    "external_remote_ip_flag",
    "ip_protocol",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check empty-value and zero-vector quality of extracted feature TSV files.")
    parser.add_argument(
        "--feature-root",
        type=str,
        default=str(DEFAULT_FEATURE_ROOT),
        help=f"Directory containing per-window feature TSV files. Default: {DEFAULT_FEATURE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for summary outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def feature_files(feature_root: Path) -> List[Path]:
    return sorted(feature_root.glob("*/*.tsv"))


def split_and_group_name(path: Path, feature_root: Path) -> str:
    relative = path.relative_to(feature_root)
    return f"{relative.parent.name}/{relative.name}"


def is_missing_text(value: str) -> bool:
    return value is None or value == ""


def is_missing_numeric(value: str) -> bool:
    return value is None or value == ""


def is_zero_numeric(value: str) -> bool:
    if value is None or value == "":
        return True
    try:
        return float(value) == 0.0
    except ValueError:
        return False


def analyze_file(path: Path) -> Dict[str, object]:
    row_count = 0
    any_missing_rows = 0
    all_zero_rows = 0
    column_missing_counts: Dict[str, int] = {}
    column_zero_counts: Dict[str, int] = {}
    columns: List[str] = []
    numeric_columns: List[str] = []
    text_columns: List[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = list(reader.fieldnames or [])
        text_columns = [col for col in columns if col in TEXT_COLUMNS]
        numeric_columns = [col for col in columns if col not in TEXT_COLUMNS]
        column_missing_counts = {col: 0 for col in columns}
        column_zero_counts = {col: 0 for col in numeric_columns}

        for row in reader:
            row_count += 1
            row_has_missing = False
            row_all_zero = True if numeric_columns else False

            for col in columns:
                value = row.get(col, "")
                if col in TEXT_COLUMNS:
                    missing = is_missing_text(value)
                else:
                    missing = is_missing_numeric(value)

                if missing:
                    column_missing_counts[col] += 1
                    row_has_missing = True

                if col in numeric_columns:
                    zero = is_zero_numeric(value)
                    if zero:
                        column_zero_counts[col] += 1
                    else:
                        row_all_zero = False

            if row_has_missing:
                any_missing_rows += 1
            if row_all_zero:
                all_zero_rows += 1

    missing_ratios = {col: (count / row_count if row_count else 0.0) for col, count in column_missing_counts.items()}
    zero_ratios = {col: (count / row_count if row_count else 0.0) for col, count in column_zero_counts.items()}

    top_missing = sorted(missing_ratios.items(), key=lambda item: item[1], reverse=True)[:8]
    top_zero = sorted(zero_ratios.items(), key=lambda item: item[1], reverse=True)[:8]

    return {
        "rows": row_count,
        "columns": columns,
        "numeric_feature_columns": numeric_columns,
        "text_columns": text_columns,
        "all_zero_numeric_rows": all_zero_rows,
        "all_zero_numeric_ratio": (all_zero_rows / row_count if row_count else 0.0),
        "rows_with_any_missing": any_missing_rows,
        "rows_with_any_missing_ratio": (any_missing_rows / row_count if row_count else 0.0),
        "column_missing_ratio": missing_ratios,
        "column_zero_ratio": zero_ratios,
        "top_missing_columns": top_missing,
        "top_zero_columns": top_zero,
    }


def main() -> None:
    args = build_parser().parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    manifest = {"feature_root": str(feature_root), "files": {}}
    rows = []

    for path in feature_files(feature_root):
        key = split_and_group_name(path, feature_root)
        stats = analyze_file(path)
        manifest["files"][key] = stats
        rows.append(
            {
                "file": key,
                "rows": stats["rows"],
                "all_zero_numeric_ratio": stats["all_zero_numeric_ratio"],
                "rows_with_any_missing_ratio": stats["rows_with_any_missing_ratio"],
                "top_missing_columns": stats["top_missing_columns"],
                "top_zero_columns": stats["top_zero_columns"],
            }
        )

    with (output_dir / "feature_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    with (output_dir / "feature_quality_brief.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "file\trows\tall_zero_numeric_ratio\trows_with_any_missing_ratio\ttop_missing_columns\ttop_zero_columns\n"
        )
        for row in rows:
            top_missing = ";".join(f"{name}:{ratio:.6f}" for name, ratio in row["top_missing_columns"])
            top_zero = ";".join(f"{name}:{ratio:.6f}" for name, ratio in row["top_zero_columns"])
            handle.write(
                f"{row['file']}\t{row['rows']}\t{row['all_zero_numeric_ratio']:.6f}\t"
                f"{row['rows_with_any_missing_ratio']:.6f}\t{top_missing}\t{top_zero}\n"
            )

    print(f"[feature-quality] wrote {output_dir / 'feature_quality_summary.json'}")
    print(f"[feature-quality] wrote {output_dir / 'feature_quality_brief.tsv'}")


if __name__ == "__main__":
    main()
