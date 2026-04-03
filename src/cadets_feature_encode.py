import argparse
import csv
import ipaddress
import json
import os
from pathlib import Path
from typing import Callable, Dict, List


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features_cleaned" / "feature_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "features_model_ready"

RAW_METADATA_COLUMNS = {"node_uuid", "node_type"}

RAW_CATEGORICAL_CANDIDATES = {
    "process_view__file_node": ["file_type", "permission_value"],
    "file_view__file_node": ["file_type", "permission_value"],
    "process_view__network_node": [
        "local_port_bucket",
        "remote_port_bucket",
        "external_remote_ip_flag",
        "ip_protocol",
    ],
    "network_view__network_node": [
        "local_port_bucket",
        "remote_port_bucket",
        "external_remote_ip_flag",
        "ip_protocol",
    ],
    "file_view__process_node": ["subject_type"],
    "network_view__process_node": ["subject_type"],
    "process_view__process_node": ["subject_type"],
}

PROCESS_GROUPS = {
    "process_view__process_node",
    "file_view__process_node",
    "network_view__process_node",
}

FILE_GROUPS = {
    "process_view__file_node",
    "file_view__file_node",
}

NETWORK_GROUPS = {
    "process_view__network_node",
    "network_view__network_node",
}

NETWORK_TOOL_PATTERNS = (
    "curl",
    "wget",
    "nc ",
    "ncat",
    "netcat",
    "ssh",
    "scp",
    "sftp",
    "ftp",
    "telnet",
    "socat",
)

INTERPRETER_PATTERNS = ("python", "perl", "ruby", "php", "node", "java")
PACKAGE_TOOL_PATTERNS = ("apt", "apt-get", "yum", "dnf", "pip", "npm", "dpkg", "rpm")
SYSTEM_TOOL_PATTERNS = ("sudo", "su ", "systemctl", "service ", "mount", "chmod", "chown", "cron")
PIPE_TOKENS = ("|", "&&", ";", "$(", "`")

FeatureBuilder = Callable[[Dict[str, str]], str]


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


def normalize_category(value: str) -> str:
    if value is None:
        return "__MISSING__"
    normalized = str(value).strip()
    return normalized if normalized else "__MISSING__"


def raw_column_builder(column: str) -> FeatureBuilder:
    return lambda row, column=column: normalize_category(row.get(column, ""))


def classify_cmd_semantic(cmd_line: str) -> str:
    text = cmd_line.strip().lower()
    if not text:
        return "unknown"
    if any(token in text for token in NETWORK_TOOL_PATTERNS):
        return "network_tool"
    if any(token in text for token in INTERPRETER_PATTERNS):
        return "interpreter"
    if any(token in text for token in PACKAGE_TOOL_PATTERNS):
        return "package_tool"
    if any(token in text for token in SYSTEM_TOOL_PATTERNS):
        return "system_tool"
    first_token = os.path.basename(text.split()[0])
    if first_token in {"sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh"}:
        return "shell"
    return "other"


def classify_cmd_length(cmd_line: str) -> str:
    token_count = len(cmd_line.split())
    if token_count == 0:
        return "empty"
    if token_count <= 2:
        return "short"
    if token_count <= 6:
        return "medium"
    return "long"


def classify_cmd_pipe_flag(cmd_line: str) -> str:
    text = cmd_line.strip()
    if not text:
        return "unknown"
    return "yes" if any(token in text for token in PIPE_TOKENS) else "no"


def classify_file_path_bucket(file_descriptor: str) -> str:
    text = file_descriptor.strip().lower()
    if not text:
        return "unknown"
    if text.startswith("/proc/") or text == "/proc":
        return "procfs"
    if text.startswith("/dev/") or text == "/dev":
        return "devfs"
    if text.startswith("/tmp/") or text.startswith("/var/tmp/") or text.startswith("/dev/shm/"):
        return "temp"
    if text.startswith("/etc/"):
        return "config"
    if text.startswith("/usr/bin/") or text.startswith("/bin/") or text.startswith("/usr/sbin/") or text.startswith("/sbin/"):
        return "system_bin"
    if text.startswith("/usr/lib/") or text.startswith("/lib/") or text.startswith("/lib64/") or text.startswith("/usr/lib64/"):
        return "system_lib"
    if text.startswith("/var/log/"):
        return "log"
    if text.startswith("/home/") or text.startswith("/root/"):
        return "user_home"
    return "other"


def classify_file_extension_bucket(file_descriptor: str) -> str:
    text = file_descriptor.strip().lower()
    if not text:
        return "unknown"
    basename = os.path.basename(text)
    if not basename or "." not in basename:
        return "none"
    extension = basename.rsplit(".", 1)[-1]
    if extension in {"conf", "config", "cfg", "ini", "yaml", "yml", "json", "xml"}:
        return "config"
    if extension in {"sh", "bash", "zsh", "py", "pl", "rb", "php", "js"}:
        return "script"
    if extension in {"so", "dll", "dylib", "a", "o"}:
        return "library"
    if extension in {"log", "txt", "out"}:
        return "log"
    if extension in {"db", "sqlite", "sqlite3", "ldb"}:
        return "database"
    if extension in {"zip", "tar", "gz", "bz2", "xz", "7z", "rar"}:
        return "archive"
    if extension in {"bin", "exe"}:
        return "binary"
    return "other"


def classify_file_hidden_flag(file_descriptor: str) -> str:
    text = file_descriptor.strip()
    if not text:
        return "unknown"
    basename = os.path.basename(text.rstrip("/"))
    return "yes" if basename.startswith(".") else "no"


def parse_ip_scope(address: str) -> str:
    text = address.strip()
    if not text:
        return "unknown"
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return "invalid"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_private:
        return "private"
    return "public"


def classify_service_bucket(port_value: str) -> str:
    try:
        port = int(float(port_value))
    except (TypeError, ValueError):
        return "unknown"
    if port < 0:
        return "unknown"
    if port in {20, 21}:
        return "ftp"
    if port == 22:
        return "ssh"
    if port == 23:
        return "telnet"
    if port == 53:
        return "dns"
    if port in {80, 443, 8000, 8008, 8080, 8443}:
        return "web"
    if port in {25, 110, 143, 465, 587, 993, 995}:
        return "mail"
    if port in {135, 139, 445}:
        return "smb_rpc"
    if port in {88, 389, 464, 636}:
        return "auth_directory"
    if port in {1433, 1521, 3306, 5432, 6379, 9200, 27017}:
        return "database"
    if port == 3389:
        return "rdp"
    if port == 123:
        return "ntp"
    if port == 4444:
        return "high_risk"
    return "other"


def build_feature_builders(group_name: str) -> Dict[str, FeatureBuilder]:
    builders: Dict[str, FeatureBuilder] = {
        column: raw_column_builder(column) for column in RAW_CATEGORICAL_CANDIDATES.get(group_name, [])
    }

    if group_name in PROCESS_GROUPS:
        builders["cmd_semantic_bucket"] = lambda row: classify_cmd_semantic(row.get("cmd_line", ""))
        builders["cmd_length_bucket"] = lambda row: classify_cmd_length(row.get("cmd_line", ""))
        builders["cmd_pipe_flag"] = lambda row: classify_cmd_pipe_flag(row.get("cmd_line", ""))

    if group_name in FILE_GROUPS:
        builders["file_path_bucket"] = lambda row: classify_file_path_bucket(row.get("file_descriptor", ""))
        builders["file_extension_bucket"] = lambda row: classify_file_extension_bucket(row.get("file_descriptor", ""))
        builders["file_hidden_flag"] = lambda row: classify_file_hidden_flag(row.get("file_descriptor", ""))

    if group_name in NETWORK_GROUPS:
        builders["remote_scope_bucket"] = lambda row: parse_ip_scope(row.get("remote_address", ""))
        builders["remote_service_bucket"] = lambda row: classify_service_bucket(row.get("remote_port", ""))

    return builders


def fit_group_vocab(train_path: Path, feature_builders: Dict[str, FeatureBuilder]) -> Dict[str, List[str]]:
    vocab: Dict[str, Dict[str, int]] = {column: {} for column in feature_builders}
    with train_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            for column, builder in feature_builders.items():
                value = normalize_category(builder(row))
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
    feature_builders: Dict[str, FeatureBuilder],
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
                value = normalize_category(feature_builders[column](row))
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
    group_feature_builders: Dict[str, Dict[str, FeatureBuilder]] = {}
    for group_name, group_meta in manifest["group_specs"].items():
        train_path = feature_root / "train" / f"{group_name}.tsv"
        feature_builders = build_feature_builders(group_name)
        vocab = fit_group_vocab(train_path, feature_builders)
        group_feature_builders[group_name] = feature_builders
        raw_categorical_columns = list(RAW_CATEGORICAL_CANDIDATES.get(group_name, []))
        derived_categorical_columns = [
            column for column in feature_builders if column not in raw_categorical_columns
        ]
        group_specs[group_name] = {
            **build_encoded_columns(group_meta, vocab),
            "raw_categorical_columns": raw_categorical_columns,
            "derived_categorical_columns": derived_categorical_columns,
            "dropped_raw_text_columns": [
                column
                for column in group_meta["kept_text_columns"]
                if column not in RAW_METADATA_COLUMNS and column not in raw_categorical_columns
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
            group_outputs[group_name] = encode_window_group(
                input_path,
                output_path,
                group_specs[group_name],
                group_feature_builders[group_name],
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
        print(f"[feature-encode] {window_name}: model-ready feature files written", flush=True)

    output_manifest = {
        "cleaned_feature_manifest_path": str(feature_manifest_path),
        "group_specs": group_specs,
        "windows": window_outputs,
        "notes": [
            "Low-cardinality categorical vocabularies are fit on train rows only.",
            "Derived semantic categorical buckets are built from cmd_line, file_descriptor, and network endpoint metadata.",
            "High-cardinality raw text fields are not encoded directly into model features.",
            "node_uuid and node_type are retained as metadata columns only.",
        ],
    }
    with (output_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)
        handle.write("\n")

    print(f"[feature-encode] wrote {output_dir / 'feature_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
