import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features_cleaned" / "feature_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "feature_quality"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-dimension non-missing and non-zero ratios for cleaned feature TSV files."
    )
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=str(DEFAULT_FEATURE_MANIFEST),
        help=f"Path to cleaned feature_manifest.json. Default: {DEFAULT_FEATURE_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for density outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def analyze_file(path: Path, numeric_columns: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {
        column: {
            "non_missing_count": 0,
            "non_zero_count": 0,
        }
        for column in numeric_columns
    }
    row_count = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row_count += 1
            for column in numeric_columns:
                value = row.get(column, "")
                if value != "":
                    stats[column]["non_missing_count"] += 1
                if to_float(value) != 0.0:
                    stats[column]["non_zero_count"] += 1

    return {
        "rows": row_count,
        "columns": {
            column: {
                "non_missing_count": values["non_missing_count"],
                "non_missing_ratio": (values["non_missing_count"] / row_count if row_count else 0.0),
                "non_zero_count": values["non_zero_count"],
                "non_zero_ratio": (values["non_zero_count"] / row_count if row_count else 0.0),
            }
            for column, values in stats.items()
        },
    }


def main() -> None:
    args = parse_args()
    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    manifest = load_json(feature_manifest_path)
    feature_root = feature_manifest_path.parent

    summary: Dict[str, object] = {
        "feature_manifest_path": str(feature_manifest_path),
        "windows": {},
    }
    rows = []

    for window in manifest["windows"]:
        window_name = window["window_name"]
        summary["windows"][window_name] = {}
        for group_name, group_meta in sorted(window["groups"].items()):
            numeric_columns = list(group_meta["kept_numeric_columns"])
            file_path = feature_root / window_name / f"{group_name}.tsv"
            stats = analyze_file(file_path, numeric_columns)
            summary["windows"][window_name][group_name] = stats

            for feature_name, values in stats["columns"].items():
                rows.append(
                    {
                        "window_name": window_name,
                        "feature_group": group_name,
                        "feature_name": feature_name,
                        "rows": stats["rows"],
                        "non_missing_count": values["non_missing_count"],
                        "non_missing_ratio": values["non_missing_ratio"],
                        "non_zero_count": values["non_zero_count"],
                        "non_zero_ratio": values["non_zero_ratio"],
                    }
                )

    with (output_dir / "cleaned_feature_density.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    with (output_dir / "cleaned_feature_density.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window_name",
                "feature_group",
                "feature_name",
                "rows",
                "non_missing_count",
                "non_missing_ratio",
                "non_zero_count",
                "non_zero_ratio",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "non_missing_ratio": f"{row['non_missing_ratio']:.6f}",
                    "non_zero_ratio": f"{row['non_zero_ratio']:.6f}",
                }
            )

    print(f"[feature-density] wrote {output_dir / 'cleaned_feature_density.json'}")
    print(f"[feature-density] wrote {output_dir / 'cleaned_feature_density.tsv'}")


if __name__ == "__main__":
    main()
