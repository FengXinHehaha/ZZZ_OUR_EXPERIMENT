import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from tqdm import tqdm


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features" / "feature_manifest.json"
DEFAULT_NODE_RETENTION_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "node_retention" / "retention_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features_cleaned"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean retained-node feature TSVs and standardize numeric columns using train-window statistics."
    )
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=str(DEFAULT_FEATURE_MANIFEST),
        help=f"Path to feature_manifest.json. Default: {DEFAULT_FEATURE_MANIFEST}",
    )
    parser.add_argument(
        "--node-retention-manifest",
        type=str,
        default=str(DEFAULT_NODE_RETENTION_MANIFEST),
        help=f"Path to node_retention retention_manifest.json. Default: {DEFAULT_NODE_RETENTION_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for cleaned feature outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_uuid_set(path: Path) -> Set[str]:
    values: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        progress = tqdm(
            reader,
            desc=f"keep-set {path.parent.name}/{path.name}",
            unit="row",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=False,
        )
        for row in progress:
            values.add(row["node_uuid"])
    return values


def entity_type_for_feature_file(filename: str) -> str:
    if filename.endswith("__process_node.tsv"):
        return "process"
    if filename.endswith("__file_node.tsv"):
        return "file"
    if filename.endswith("__network_node.tsv"):
        return "network"
    raise ValueError(f"Unsupported feature filename: {filename}")


def load_keep_sets(retention_root: Path, window_name: str) -> Dict[str, Set[str]]:
    window_dir = retention_root / window_name
    return {
        "process": read_uuid_set(window_dir / "process_keep_list.tsv"),
        "file": read_uuid_set(window_dir / "file_keep_list.tsv"),
        "network": read_uuid_set(window_dir / "network_keep_list.tsv"),
    }


def to_numeric(value: str) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def init_numeric_stats(columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    return {
        column: {
            "count": 0.0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": math.inf,
            "max": -math.inf,
            "missing": 0.0,
        }
        for column in columns
    }


def build_group_spec(train_path: Path, keep_set: Set[str]) -> Dict[str, object]:
    with train_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = list(reader.fieldnames or [])
        text_columns = [column for column in columns if column in TEXT_COLUMNS]
        numeric_columns = [column for column in columns if column not in TEXT_COLUMNS]

        text_missing = {column: 0 for column in text_columns}
        numeric_stats = init_numeric_stats(numeric_columns)
        retained_rows = 0

        progress = tqdm(
            reader,
            desc=f"clean-spec {train_path.parent.name}/{train_path.name}",
            unit="row",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=False,
        )
        for row in progress:
            if row["node_uuid"] not in keep_set:
                continue
            retained_rows += 1
            for column in text_columns:
                if row.get(column, "") == "":
                    text_missing[column] += 1
            for column in numeric_columns:
                raw = row.get(column, "")
                if raw == "":
                    numeric_stats[column]["missing"] += 1
                value = to_numeric(raw)
                stats = numeric_stats[column]
                stats["count"] += 1
                stats["sum"] += value
                stats["sumsq"] += value * value
                if value < stats["min"]:
                    stats["min"] = value
                if value > stats["max"]:
                    stats["max"] = value

    dropped_text_columns = [column for column in text_columns if retained_rows and text_missing[column] == retained_rows]
    kept_text_columns = [column for column in text_columns if column not in dropped_text_columns]

    dropped_numeric_columns: List[str] = []
    kept_numeric_columns: List[str] = []
    numeric_scaler: Dict[str, Dict[str, float]] = {}

    for column in numeric_columns:
        stats = numeric_stats[column]
        if stats["count"] == 0:
            dropped_numeric_columns.append(column)
            continue
        if stats["min"] == stats["max"]:
            dropped_numeric_columns.append(column)
            continue

        kept_numeric_columns.append(column)
        mean = stats["sum"] / stats["count"]
        variance = max(0.0, (stats["sumsq"] / stats["count"]) - (mean * mean))
        std = math.sqrt(variance)
        numeric_scaler[column] = {
            "mean": mean,
            "std": std,
            "missing_ratio_before_fill": (stats["missing"] / stats["count"] if stats["count"] else 0.0),
            "min_after_fill": stats["min"],
            "max_after_fill": stats["max"],
        }

    return {
        "train_reference_path": str(train_path),
        "train_retained_rows": retained_rows,
        "original_columns": columns,
        "original_text_columns": text_columns,
        "original_numeric_columns": numeric_columns,
        "kept_text_columns": kept_text_columns,
        "kept_numeric_columns": kept_numeric_columns,
        "dropped_text_columns": dropped_text_columns,
        "dropped_numeric_columns": dropped_numeric_columns,
        "numeric_scaler": numeric_scaler,
    }


def transform_numeric(value: str, mean: float, std: float) -> str:
    numeric = to_numeric(value)
    transformed = (numeric - mean) / std
    return f"{transformed:.6f}"


def write_cleaned_file(
    input_path: Path,
    output_path: Path,
    keep_set: Set[str],
    group_spec: Dict[str, object],
) -> Dict[str, object]:
    kept_text_columns = list(group_spec["kept_text_columns"])
    kept_numeric_columns = list(group_spec["kept_numeric_columns"])
    scaler = group_spec["numeric_scaler"]
    fieldnames = kept_text_columns + kept_numeric_columns

    input_rows = 0
    output_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        progress = tqdm(
            reader,
            desc=f"clean-write {input_path.parent.name}/{input_path.name}",
            unit="row",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=False,
        )
        for row in progress:
            input_rows += 1
            if row["node_uuid"] not in keep_set:
                continue

            cleaned_row = {column: row.get(column, "") for column in kept_text_columns}
            for column in kept_numeric_columns:
                cleaned_row[column] = transform_numeric(
                    row.get(column, ""),
                    scaler[column]["mean"],
                    scaler[column]["std"],
                )

            writer.writerow(cleaned_row)
            output_rows += 1

    return {
        "input_rows": input_rows,
        "retained_rows": output_rows,
        "dropped_rows": input_rows - output_rows,
        "cleaned_columns": fieldnames,
        "kept_text_columns": kept_text_columns,
        "kept_numeric_columns": kept_numeric_columns,
        "dropped_text_columns": list(group_spec["dropped_text_columns"]),
        "dropped_numeric_columns": list(group_spec["dropped_numeric_columns"]),
    }


def main() -> None:
    args = parse_args()
    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    node_retention_manifest_path = Path(args.node_retention_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    feature_manifest = load_json(feature_manifest_path)
    node_retention_manifest = load_json(node_retention_manifest_path)

    feature_root = feature_manifest_path.parent
    retention_root = node_retention_manifest_path.parent

    train_window = next(window for window in feature_manifest["windows"] if window["window_name"] == "train")
    feature_groups = sorted(train_window["feature_files"].keys())

    keep_sets_by_window = {
        window["window_name"]: load_keep_sets(retention_root, window["window_name"])
        for window in node_retention_manifest["windows"]
    }

    group_specs: Dict[str, Dict[str, object]] = {}
    for group_name in feature_groups:
        filename = train_window["feature_files"][group_name]
        entity_type = entity_type_for_feature_file(filename)
        train_path = feature_root / "train" / filename
        group_specs[group_name] = build_group_spec(train_path, keep_sets_by_window["train"][entity_type])
        print(
            f"[feature-clean] {group_name}: keep_text={len(group_specs[group_name]['kept_text_columns'])} "
            f"keep_numeric={len(group_specs[group_name]['kept_numeric_columns'])} "
            f"drop_numeric={len(group_specs[group_name]['dropped_numeric_columns'])}",
            flush=True,
        )

    window_outputs = []
    for window in feature_manifest["windows"]:
        window_name = window["window_name"]
        window_dir = output_dir / window_name
        ensure_dir(window_dir)
        keep_sets = keep_sets_by_window[window_name]
        group_outputs: Dict[str, Dict[str, object]] = {}

        for group_name, filename in sorted(window["feature_files"].items()):
            entity_type = entity_type_for_feature_file(filename)
            input_path = feature_root / window_name / filename
            output_path = window_dir / filename
            group_outputs[group_name] = write_cleaned_file(
                input_path,
                output_path,
                keep_sets[entity_type],
                group_specs[group_name],
            )

        metadata = {
            "window_name": window_name,
            "split": window["split"],
            "days": list(window["days"]),
            "groups": group_outputs,
        }
        with (window_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        window_outputs.append(metadata)
        print(f"[feature-clean] {window_name}: cleaned feature files written", flush=True)

    manifest = {
        "feature_manifest_path": str(feature_manifest_path),
        "node_retention_manifest_path": str(node_retention_manifest_path),
        "group_specs": group_specs,
        "windows": window_outputs,
        "notes": [
            "Only retained nodes are kept in cleaned feature files.",
            "Column screening and scaler fitting are derived from retained train-window rows only.",
            "Numeric missing values are filled with 0 before standardization.",
            "Fully missing text columns and constant numeric columns are dropped.",
        ],
    }
    with (output_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"[feature-clean] wrote {output_dir / 'feature_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
