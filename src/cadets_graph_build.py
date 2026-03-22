import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


DEFAULT_FEATURE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "features_model_ready" / "feature_manifest.json"
DEFAULT_NODE_RETENTION_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "node_retention" / "retention_manifest.json"
DEFAULT_EDGE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "edges_filtered" / "edge_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "graphs"

WINDOW_TO_SPLIT = {
    "train": "train",
    "val": "val",
    "test_2018-04-12": "test",
    "test_2018-04-13": "test",
}

WINDOW_TO_DAYS = {
    "train": ["2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05", "2018-04-06", "2018-04-07", "2018-04-08", "2018-04-09", "2018-04-10"],
    "val": ["2018-04-11"],
    "test_2018-04-12": ["2018-04-12"],
    "test_2018-04-13": ["2018-04-13"],
}

NODE_TYPE_VOCAB = {
    "process": 0,
    "file": 1,
    "network": 2,
}

FEATURE_GROUP_TO_NODE_TYPE = {
    "process_view__process_node": "process",
    "process_view__file_node": "file",
    "process_view__network_node": "network",
    "file_view__process_node": "process",
    "file_view__file_node": "file",
    "network_view__process_node": "process",
    "network_view__network_node": "network",
}


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph-ready window artifacts from retained nodes, model-ready features, and filtered typed edges."
    )
    parser.add_argument(
        "--feature-manifest",
        type=str,
        default=str(DEFAULT_FEATURE_MANIFEST),
        help=f"Path to model-ready feature_manifest.json. Default: {DEFAULT_FEATURE_MANIFEST}",
    )
    parser.add_argument(
        "--node-retention-manifest",
        type=str,
        default=str(DEFAULT_NODE_RETENTION_MANIFEST),
        help=f"Path to node_retention/retention_manifest.json. Default: {DEFAULT_NODE_RETENTION_MANIFEST}",
    )
    parser.add_argument(
        "--edge-manifest",
        type=str,
        default=str(DEFAULT_EDGE_MANIFEST),
        help=f"Path to edges_filtered/edge_manifest.json. Default: {DEFAULT_EDGE_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for graph artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Optional specific window(s) to build. Defaults to all windows in the feature manifest.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_windows(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {window["window_name"]: window for window in manifest["windows"]}


def read_keep_list(path: Path, node_type: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    gt_column = {
        "file": "is_gt_file",
        "process": "is_gt_process",
        "network": "is_gt_network",
    }[node_type]

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not as_bool(row["keep"]):
                continue
            rows.append(
                {
                    "node_uuid": row["node_uuid"],
                    "node_type": node_type,
                    "is_gt": as_bool(row[gt_column]),
                    "decision_reason": row["decision_reason"],
                }
            )
    return rows


def build_nodes(window_name: str, window_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, int], Dict[str, int]]:
    all_rows: List[Dict[str, object]] = []
    per_type_counts: Dict[str, int] = {}

    for node_type, filename in (
        ("process", "process_keep_list.tsv"),
        ("file", "file_keep_list.tsv"),
        ("network", "network_keep_list.tsv"),
    ):
        path = window_dir / filename
        rows = read_keep_list(path, node_type)
        per_type_counts[node_type] = len(rows)
        all_rows.extend(rows)

    node_id_by_uuid = {row["node_uuid"]: index for index, row in enumerate(all_rows)}

    print(
        f"[graph-build] {window_name}: nodes process={per_type_counts['process']} "
        f"file={per_type_counts['file']} network={per_type_counts['network']} total={len(all_rows)}",
        flush=True,
    )
    return all_rows, node_id_by_uuid, per_type_counts


def write_node_table(window_output_dir: Path, rows: List[Dict[str, object]], node_id_by_uuid: Dict[str, int]) -> None:
    path = window_output_dir / "nodes.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "node_uuid", "node_type", "node_type_id", "is_gt", "decision_reason"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "node_id": node_id_by_uuid[row["node_uuid"]],
                    "node_uuid": row["node_uuid"],
                    "node_type": row["node_type"],
                    "node_type_id": NODE_TYPE_VOCAB[row["node_type"]],
                    "is_gt": "1" if row["is_gt"] else "0",
                    "decision_reason": row["decision_reason"],
                }
            )


def collect_event_type_vocab(edge_windows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    event_types = set()
    for window in edge_windows:
        with Path(window["output_edge_file"]).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                event_types.add(row["event_type"])
    return {event_type: index for index, event_type in enumerate(sorted(event_types))}


def build_edges(
    edge_file: Path,
    edge_row_count: int,
    node_id_by_uuid: Dict[str, int],
    event_type_vocab: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    src = np.empty(edge_row_count, dtype=np.int64)
    dst = np.empty(edge_row_count, dtype=np.int64)
    edge_type = np.empty(edge_row_count, dtype=np.int64)
    event_count = np.empty(edge_row_count, dtype=np.float32)
    first_ts = np.empty(edge_row_count, dtype=np.int64)
    last_ts = np.empty(edge_row_count, dtype=np.int64)

    filled = 0
    with edge_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            src_uuid = row["src_uuid"]
            dst_uuid = row["dst_uuid"]
            if src_uuid not in node_id_by_uuid or dst_uuid not in node_id_by_uuid:
                continue
            src[filled] = node_id_by_uuid[src_uuid]
            dst[filled] = node_id_by_uuid[dst_uuid]
            edge_type[filled] = event_type_vocab[row["event_type"]]
            event_count[filled] = float(row["event_count"])
            first_ts[filled] = int(row["first_timestamp_ns"])
            last_ts[filled] = int(row["last_timestamp_ns"])
            filled += 1

    if filled != edge_row_count:
        src = src[:filled]
        dst = dst[:filled]
        edge_type = edge_type[:filled]
        event_count = event_count[:filled]
        first_ts = first_ts[:filled]
        last_ts = last_ts[:filled]

    return {
        "edge_index": torch.from_numpy(np.stack([src, dst], axis=0)),
        "edge_type": torch.from_numpy(edge_type),
        "edge_event_count": torch.from_numpy(event_count),
        "edge_first_timestamp_ns": torch.from_numpy(first_ts),
        "edge_last_timestamp_ns": torch.from_numpy(last_ts),
    }


def load_group_features(
    group_file: Path,
    row_count: int,
    feature_columns: List[str],
    feature_dim: int,
    node_id_by_uuid: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    rows = int(row_count)
    dim = int(feature_dim)

    node_ids = np.empty(rows, dtype=np.int64)
    x = np.empty((rows, dim), dtype=np.float32)

    filled = 0
    with group_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            node_uuid = row["node_uuid"]
            if node_uuid not in node_id_by_uuid:
                continue
            node_ids[filled] = node_id_by_uuid[node_uuid]
            x[filled] = np.asarray([float(row[column]) for column in feature_columns], dtype=np.float32)
            filled += 1

    if filled != rows:
        node_ids = node_ids[:filled]
        x = x[:filled]

    return {
        "node_ids": torch.from_numpy(node_ids),
        "x": torch.from_numpy(x),
    }


def build_labels(rows: List[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_type = np.empty(len(rows), dtype=np.int64)
    y = np.zeros(len(rows), dtype=np.int64)
    gt_mask = np.zeros(len(rows), dtype=np.bool_)

    for index, row in enumerate(rows):
        node_type[index] = NODE_TYPE_VOCAB[row["node_type"]]
        if row["is_gt"]:
            y[index] = 1
            gt_mask[index] = True

    normal_mask = ~gt_mask
    return torch.from_numpy(node_type), torch.from_numpy(y), torch.from_numpy(normal_mask)


def build_window_graph(
    window_name: str,
    feature_window: Dict[str, object],
    feature_group_specs: Dict[str, object],
    node_window: Dict[str, object],
    edge_window: Dict[str, object],
    event_type_vocab: Dict[str, int],
    output_root: Path,
) -> Dict[str, object]:
    node_window_dir = Path(node_window["summary_path"]).parent if "summary_path" in node_window else Path(
        node_window["dropped_gt_uuid_list_path"]
    ).parent
    window_output_dir = output_root / window_name
    ensure_dir(window_output_dir)

    rows, node_id_by_uuid, per_type_counts = build_nodes(window_name, node_window_dir)
    write_node_table(window_output_dir, rows, node_id_by_uuid)
    node_type, y, normal_mask = build_labels(rows)

    edges = build_edges(
        Path(edge_window["output_edge_file"]),
        int(edge_window["retained_typed_edge_rows"]),
        node_id_by_uuid,
        event_type_vocab,
    )
    print(f"[graph-build] {window_name}: edges typed={edges['edge_type'].shape[0]}", flush=True)

    feature_groups = {}
    for group_name, group_meta in sorted(feature_window["groups"].items()):
        group_file = Path(feature_window["window_dir"]) / f"{group_name}.tsv" if "window_dir" in feature_window else (
            Path(output_root).parents[0] / "features_model_ready" / window_name / f"{group_name}.tsv"
        )
        group_spec = feature_group_specs[group_name]
        group_payload = load_group_features(
            group_file=group_file,
            row_count=int(group_meta["rows"]),
            feature_columns=list(group_spec["model_feature_columns"]),
            feature_dim=int(group_meta["model_feature_dim"]),
            node_id_by_uuid=node_id_by_uuid,
        )
        feature_groups[group_name] = {
            **group_payload,
            "node_type": FEATURE_GROUP_TO_NODE_TYPE[group_name],
            "model_feature_dim": int(group_meta["model_feature_dim"]),
            "model_feature_columns": list(group_spec["model_feature_columns"]),
        }
        print(
            f"[graph-build] {window_name}: {group_name} rows={group_payload['node_ids'].shape[0]} "
            f"dim={group_payload['x'].shape[1]}",
            flush=True,
        )

    graph = {
        "window_name": window_name,
        "split": WINDOW_TO_SPLIT[window_name],
        "days": WINDOW_TO_DAYS[window_name],
        "num_nodes": len(rows),
        "num_edges": int(edges["edge_type"].shape[0]),
        "node_type": node_type,
        "y": y,
        "gt_mask": y.bool(),
        "normal_mask": normal_mask,
        "edge_index": edges["edge_index"],
        "edge_type": edges["edge_type"],
        "edge_event_count": edges["edge_event_count"],
        "edge_first_timestamp_ns": edges["edge_first_timestamp_ns"],
        "edge_last_timestamp_ns": edges["edge_last_timestamp_ns"],
        "feature_groups": feature_groups,
        "node_type_vocab": NODE_TYPE_VOCAB,
        "event_type_vocab": event_type_vocab,
    }

    torch.save(graph, window_output_dir / "graph.pt")

    summary = {
        "window_name": window_name,
        "split": WINDOW_TO_SPLIT[window_name],
        "days": WINDOW_TO_DAYS[window_name],
        "num_nodes": len(rows),
        "num_edges": int(edges["edge_type"].shape[0]),
        "process_node_count": per_type_counts["process"],
        "file_node_count": per_type_counts["file"],
        "network_node_count": per_type_counts["network"],
        "gt_node_count": int(y.sum()),
        "feature_group_rows": {group_name: int(payload["node_ids"].shape[0]) for group_name, payload in feature_groups.items()},
        "feature_group_dims": {group_name: int(payload["x"].shape[1]) for group_name, payload in feature_groups.items()},
        "graph_path": str(window_output_dir / "graph.pt"),
        "node_table_path": str(window_output_dir / "nodes.tsv"),
        "feature_group_columns": {
            group_name: list(payload["model_feature_columns"])
            for group_name, payload in feature_groups.items()
        },
    }
    with (window_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary


def main() -> None:
    args = parse_args()
    feature_manifest_path = Path(args.feature_manifest).expanduser().resolve()
    node_manifest_path = Path(args.node_retention_manifest).expanduser().resolve()
    edge_manifest_path = Path(args.edge_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    feature_manifest = load_json(feature_manifest_path)
    node_manifest = load_json(node_manifest_path)
    edge_manifest = load_json(edge_manifest_path)

    feature_windows = index_windows(feature_manifest)
    node_windows = index_windows(node_manifest)
    edge_windows = index_windows(edge_manifest)

    selected_windows = args.window or [window["window_name"] for window in feature_manifest["windows"]]
    event_type_vocab = collect_event_type_vocab(edge_manifest["windows"])
    print(f"[graph-build] global event types={len(event_type_vocab)}", flush=True)

    summaries = []
    for window_name in selected_windows:
        feature_window = feature_windows[window_name]
        feature_window["window_dir"] = str(feature_manifest_path.parent / window_name)
        node_window = node_windows[window_name]
        node_window["summary_path"] = str(node_manifest_path.parent / window_name / "summary.json")
        edge_window = edge_windows[window_name]
        summaries.append(
            build_window_graph(
                window_name=window_name,
                feature_window=feature_window,
                feature_group_specs=feature_manifest["group_specs"],
                node_window=node_window,
                edge_window=edge_window,
                event_type_vocab=event_type_vocab,
                output_root=output_dir,
            )
        )

    output_manifest = {
        "feature_manifest_path": str(feature_manifest_path),
        "node_retention_manifest_path": str(node_manifest_path),
        "edge_manifest_path": str(edge_manifest_path),
        "node_type_vocab": NODE_TYPE_VOCAB,
        "event_type_vocab": event_type_vocab,
        "windows": summaries,
        "notes": [
            "Each window is built as a separate graph artifact.",
            "Feature tensors are stored per (view, node_type) group rather than force-padding all groups into one matrix.",
            "nodes.tsv stores node_id to uuid/type alignment outside graph.pt.",
        ],
    }
    with (output_dir / "graph_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)
        handle.write("\n")

    print(f"[graph-build] wrote {output_dir / 'graph_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
