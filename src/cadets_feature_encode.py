import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features_cleaned" / "feature_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features_model_ready"

RAW_METADATA_COLUMNS = {"node_uuid", "node_type"}

CATEGORICAL_CANDIDATES = {
    "process_view__file_node": ["file_type"],
    "file_view__file_node": ["file_type"],
    "process_view__network_node": ["local_port_bucket", "remote_port_bucket", "external_remote_ip_flag"],
    "network_view__network_node": ["local_port_bucket", "remote_port_bucket", "external_remote_ip_flag"],
    "file_view__process_node": ["subject_type"],
    "network_view__process_node": ["subject_type"],
    "process_view__process_node": [],
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode low-cardinality text features into graph-ready numeric columns using train-window vocabularies."
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
        help=f"Directory for model-ready feature outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fit_group_vocab(train_path: Path, candidate_columns: List[str]) -> Dict[str, List[str]]:
    vocab: Dict[str, Dict[str, int]] = {column: {} for column in candidate_columns}
    with train_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            for column in candidate_columns:
                value = row.get(column, "")
                if value == "":
                    value = "__MISSING__"
                vocab[column][value] = vocab[column].get(value, 0) + 1

    result: Dict[str, List[str]] = {}
    for column, counts in vocab.items():
        categories = sorted(counts.keys())
        if len(categories) <= 1:
            continue
        result[column] = categories
    return result


def build_encoded_columns(group_meta: Dict[str, object], vocab: Dict[str, List[str]]) -> Dict[str, object]:
    metadata_columns = [column for column in group_meta["kept_text_columns"] if column in RAW_METADATA_COLUMNS]
    numeric_columns = list(group_meta["kept_numeric_columns"])
    encoded_columns: List[str] = []
    for column, categories in vocab.items():
        encoded_columns.extend([f"{column}__{category}" for category in categories])

    return {
        "metadata_columns": metadata_columns,
        "numeric_columns": numeric_columns,
        "categorical_vocab": vocab,
        "encoded_columns": encoded_columns,
        "model_feature_columns": numeric_columns + encoded_columns,
    }


def encode_window_group(
    input_path: Path,
    output_path: Path,
    encoded_spec: Dict[str, object],
) -> Dict[str, object]:
    metadata_columns = list(encoded_spec["metadata_columns"])
    numeric_columns = list(encoded_spec["numeric_columns"])
    categorical_vocab = dict(encoded_spec["categorical_vocab"])
    encoded_columns = list(encoded_spec["encoded_columns"])

    fieldnames = metadata_columns + numeric_columns + encoded_columns
    rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            rows += 1
            encoded_row = {column: row[column] for column in metadata_columns}
            for column in numeric_columns:
                encoded_row[column] = row[column]

            for column, categories in categorical_vocab.items():
                raw_value = row.get(column, "")
                value = raw_value if raw_value != "" else "__MISSING__"
                for category in categories:
                    encoded_row[f"{column}__{category}"] = "1.0" if value == category else "0.0"

            writer.writerow(encoded_row)

    return {
        "rows": rows,
        "metadata_columns": metadata_columns,
        "numeric_columns": numeric_columns,
        "encoded_columns": encoded_columns,
        "model_feature_dim": len(numeric_columns) + len(encoded_columns),
    }


def main() -> None:
    args = parse_args()
    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    manifest = load_json(feature_manifest_path)
    feature_root = feature_manifest_path.parent
    train_window = next(window for window in manifest["windows"] if window["window_name"] == "train")

    group_specs: Dict[str, Dict[str, object]] = {}
    for group_name, group_meta in manifest["group_specs"].items():
        train_path = feature_root / "train" / f"{group_name}.tsv"
        candidate_columns = CATEGORICAL_CANDIDATES.get(group_name, [])
        vocab = fit_group_vocab(train_path, candidate_columns)
        group_specs[group_name] = {
            **build_encoded_columns(group_meta, vocab),
            "dropped_raw_text_columns": [
                column
                for column in group_meta["kept_text_columns"]
                if column not in RAW_METADATA_COLUMNS and column not in vocab
            ],
        }
        print(
            f"[feature-encode] {group_name}: numeric={len(group_specs[group_name]['numeric_columns'])} "
            f"encoded={len(group_specs[group_name]['encoded_columns'])}",
            flush=True,
        )

    window_outputs = []
    for window in manifest["windows"]:
        window_name = window["window_name"]
        window_dir = output_dir / window_name
        ensure_dir(window_dir)
        group_outputs: Dict[str, object] = {}
        for group_name in sorted(window["groups"].keys()):
            input_path = feature_root / window_name / f"{group_name}.tsv"
            output_path = window_dir / f"{group_name}.tsv"
            group_outputs[group_name] = encode_window_group(input_path, output_path, group_specs[group_name])

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
        print(f"[feature-encode] {window_name}: model-ready feature files written", flush=True)

    output_manifest = {
        "cleaned_feature_manifest_path": str(feature_manifest_path),
        "group_specs": group_specs,
        "windows": window_outputs,
        "notes": [
            "Low-cardinality categorical vocabularies are fit on train rows only.",
            "High-cardinality raw text fields are not encoded into model features.",
            "node_uuid and node_type are retained as metadata columns only.",
        ],
    }
    with (output_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)
        handle.write("\n")

    print(f"[feature-encode] wrote {output_dir / 'feature_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
