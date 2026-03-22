import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


DEFAULT_EDGE_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "edges" / "edge_manifest.json"
DEFAULT_NODE_RETENTION_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "node_retention" / "retention_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "edges_filtered"

EDGE_COLUMNS = [
    "src_uuid",
    "src_type",
    "dst_uuid",
    "dst_type",
    "event_type",
    "event_count",
    "first_timestamp_ns",
    "last_timestamp_ns",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter existing typed-edge TSVs using retained node lists.")
    parser.add_argument(
        "--edge-manifest",
        type=str,
        default=str(DEFAULT_EDGE_MANIFEST),
        help=f"Path to edge_manifest.json. Default: {DEFAULT_EDGE_MANIFEST}",
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
        help=f"Directory for filtered typed edges. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_uuid_set(path: Path) -> Set[str]:
    values: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            values.add(row["node_uuid"])
    return values


def load_window_keep_sets(retention_root: Path, window_name: str) -> Dict[str, Set[str]]:
    window_dir = retention_root / window_name
    return {
        "process": read_uuid_set(window_dir / "process_keep_list.tsv"),
        "file": read_uuid_set(window_dir / "file_keep_list.tsv"),
        "network": read_uuid_set(window_dir / "network_keep_list.tsv"),
    }


def filter_window_edges(
    input_path: Path,
    output_path: Path,
    keep_sets: Dict[str, Set[str]],
) -> Dict[str, object]:
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    kept_event_weight = 0
    dropped_event_weight = 0

    by_dst_type_total: Dict[str, int] = {}
    by_dst_type_kept: Dict[str, int] = {}

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=EDGE_COLUMNS, delimiter="\t")
        writer.writeheader()

        for row in reader:
            total_rows += 1
            src_type = row["src_type"]
            dst_type = row["dst_type"]
            src_uuid = row["src_uuid"]
            dst_uuid = row["dst_uuid"]
            event_weight = int(row["event_count"])

            by_dst_type_total[dst_type] = by_dst_type_total.get(dst_type, 0) + 1

            src_keep = src_uuid in keep_sets.get(src_type, set())
            dst_keep = dst_uuid in keep_sets.get(dst_type, set())

            if src_keep and dst_keep:
                writer.writerow(row)
                kept_rows += 1
                kept_event_weight += event_weight
                by_dst_type_kept[dst_type] = by_dst_type_kept.get(dst_type, 0) + 1
            else:
                dropped_rows += 1
                dropped_event_weight += event_weight

    return {
        "original_typed_edge_rows": total_rows,
        "retained_typed_edge_rows": kept_rows,
        "dropped_typed_edge_rows": dropped_rows,
        "typed_edge_retention_ratio": 0.0 if total_rows == 0 else kept_rows / total_rows,
        "retained_event_weight": kept_event_weight,
        "dropped_event_weight": dropped_event_weight,
        "event_weight_retention_ratio": 0.0
        if (kept_event_weight + dropped_event_weight) == 0
        else kept_event_weight / (kept_event_weight + dropped_event_weight),
        "typed_edges_by_dst_type_before": by_dst_type_total,
        "typed_edges_by_dst_type_after": by_dst_type_kept,
    }


def main() -> None:
    args = parse_args()
    edge_manifest_path = Path(args.edge_manifest).expanduser().resolve()
    node_retention_manifest_path = Path(args.node_retention_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    edge_manifest = load_json(edge_manifest_path)
    node_retention_manifest = load_json(node_retention_manifest_path)

    edge_root = edge_manifest_path.parent
    retention_root = node_retention_manifest_path.parent
    edge_windows = {item["window_name"]: item for item in edge_manifest["windows"]}

    window_summaries: List[Dict[str, object]] = []

    for retention_window in node_retention_manifest["windows"]:
        window_name = retention_window["window_name"]
        edge_window = edge_windows[window_name]
        keep_sets = load_window_keep_sets(retention_root, window_name)

        input_path = edge_root / window_name / "typed_edges.tsv"
        window_dir = output_dir / window_name
        ensure_dir(window_dir)
        output_path = window_dir / "typed_edges.tsv"

        print(f"[edge-filter] {window_name}: filtering existing typed edges", flush=True)
        result = filter_window_edges(input_path, output_path, keep_sets)

        metadata = {
            "window_name": window_name,
            "days": edge_window["days"],
            "input_edge_file": str(input_path),
            "output_edge_file": str(output_path),
            "process_node_count": retention_window["retained_process_node_count"],
            "file_node_count": retention_window["retained_file_node_count"],
            "network_node_count": retention_window["retained_network_node_count"],
            "total_node_count": retention_window["retained_total_node_count"],
            **result,
            "total_gt_dropped_count": retention_window["total_gt_dropped_count"],
        }

        with (window_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        print(
            f"[edge-filter] {window_name}: typed_edges={metadata['retained_typed_edge_rows']:,} "
            f"dropped={metadata['dropped_typed_edge_rows']:,} "
            f"event_weight_retention={metadata['event_weight_retention_ratio']:.4f}",
            flush=True,
        )
        window_summaries.append(metadata)

    manifest = {
        "edge_manifest_path": str(edge_manifest_path),
        "node_retention_manifest_path": str(node_retention_manifest_path),
        "windows": window_summaries,
        "notes": [
            "This stage never revisits events_raw.",
            "Edges are filtered only from existing typed_edges.tsv files.",
            "An edge survives iff both endpoints remain in the retained node lists for the same window.",
        ],
    }

    with (output_dir / "edge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"[edge-filter] wrote {output_dir / 'edge_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
