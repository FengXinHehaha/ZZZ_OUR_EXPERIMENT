import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from graph_loader import RELATION_GROUP_SCHEMES, event_name_to_relation_group, load_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "evaluations" / "rel_grouped_dot_30ep_eval"
DEFAULT_GRAPH_ROOT = REPO_ROOT / "artifacts" / "graphs"
DEFAULT_TARGET_WINDOW = "test_2018-04-13"
DEFAULT_REFERENCE_WINDOW = "test_2018-04-12"
DEFAULT_REFERENCE_TOPK = (1000, 5000, 10000)

SUMMARY_FIELDS = (
    "rank",
    "score",
    "total_degree",
    "in_degree",
    "out_degree",
    "unique_neighbors",
    "unique_process_neighbors",
    "unique_file_neighbors",
    "unique_network_neighbors",
    "incident_process_neighbor_edges",
    "incident_file_neighbor_edges",
    "incident_network_neighbor_edges",
    "incident_group_file_read",
    "incident_group_file_write",
    "incident_group_file_meta",
    "incident_group_process",
    "incident_group_network",
    "incident_group_flow_other",
    "incoming_group_file_read",
    "incoming_group_file_write",
    "incoming_group_file_meta",
    "incoming_group_process",
    "incoming_group_network",
    "incoming_group_flow_other",
    "outgoing_group_file_read",
    "outgoing_group_file_write",
    "outgoing_group_file_meta",
    "outgoing_group_process",
    "outgoing_group_network",
    "outgoing_group_flow_other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze top-ranked file false positives versus GT files for a scored evaluation window, "
            "and compare them against a reference window."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help=f"Evaluation directory containing evaluation_summary.json. Default: {DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--graph-root",
        type=str,
        default=str(DEFAULT_GRAPH_ROOT),
        help=(
            "Fallback root containing <window>/graph.pt if evaluation_summary.json does not expose graph paths. "
            f"Default: {DEFAULT_GRAPH_ROOT}"
        ),
    )
    parser.add_argument(
        "--target-window",
        type=str,
        default=DEFAULT_TARGET_WINDOW,
        help=f"Window to analyze. Default: {DEFAULT_TARGET_WINDOW}",
    )
    parser.add_argument(
        "--reference-window",
        type=str,
        default=DEFAULT_REFERENCE_WINDOW,
        help=f"Reference window used for overlap checks. Default: {DEFAULT_REFERENCE_WINDOW}",
    )
    parser.add_argument(
        "--node-type",
        type=str,
        default="file",
        help="Node type to analyze. Default: file",
    )
    parser.add_argument(
        "--fp-rank-max",
        type=int,
        default=10000,
        help="Only treat false positives within this rank cutoff as the high-score FP pool. Default: 10000",
    )
    parser.add_argument(
        "--fp-limit",
        type=int,
        default=100,
        help="Maximum number of top false positives to export. Default: 100",
    )
    parser.add_argument(
        "--gt-limit",
        type=int,
        default=0,
        help="Optional cap on exported GT nodes, keeping the best-ranked ones. 0 means keep all. Default: 0",
    )
    parser.add_argument(
        "--reference-topk",
        type=int,
        action="append",
        default=[],
        help="Reference top-k cutoffs to report overlap against. Can be passed multiple times.",
    )
    parser.add_argument(
        "--relation-group-scheme",
        type=str,
        default="coarse_v1",
        choices=sorted(RELATION_GROUP_SCHEMES),
        help="Relation grouping used for per-node incident-edge stats. Default: coarse_v1.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional output directory name under eval-dir. Default uses target/reference windows.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "mean": float(sum(ordered) / len(ordered)),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "max": float(ordered[-1]),
    }


def summarize_counter(counter: Counter, limit: int = 10) -> List[Dict[str, object]]:
    return [{"value": value, "count": int(count)} for value, count in counter.most_common(limit)]


def resolve_window_graph_paths(eval_dir: Path, graph_root: Path) -> Dict[str, Path]:
    summary_path = eval_dir / "evaluation_summary.json"
    window_graph_paths: Dict[str, Path] = {}
    if summary_path.exists():
        summary = load_json(summary_path)
        for item in summary.get("graphs", []):
            name = str(item.get("name", ""))
            path = item.get("path")
            if name and path:
                window_graph_paths[name] = Path(str(path)).expanduser().resolve()

    if not window_graph_paths:
        for graph_path in graph_root.glob("*/graph.pt"):
            window_graph_paths[graph_path.parent.name] = graph_path.resolve()

    return window_graph_paths


def load_scored_rows(eval_dir: Path, window_name: str) -> List[Dict[str, object]]:
    scores_path = eval_dir / window_name / "node_scores.tsv"
    rows = read_tsv(scores_path)
    result: List[Dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "rank": to_int(row["rank"]),
                "node_id": to_int(row["node_id"]),
                "node_uuid": row["node_uuid"],
                "node_type": row["node_type"],
                "is_gt": to_int(row["is_gt"]),
                "score": to_float(row["score"]),
            }
        )
    result.sort(key=lambda item: int(item["rank"]))
    return result


def load_nodes_by_id(nodes_path: Path) -> Dict[int, Dict[str, object]]:
    rows = read_tsv(nodes_path)
    result: Dict[int, Dict[str, object]] = {}
    for row in rows:
        node_id = to_int(row["node_id"])
        result[node_id] = {
            "node_id": node_id,
            "node_uuid": row["node_uuid"],
            "node_type": row["node_type"],
            "node_type_id": to_int(row.get("node_type_id")),
            "is_gt": to_int(row["is_gt"]),
            "decision_reason": row.get("decision_reason", ""),
        }
    return result


def relation_group_names_for_graph(graph: Dict[str, object], scheme: str) -> List[str]:
    group_names = sorted(
        {
            event_name_to_relation_group(event_name, scheme=scheme)
            for event_name in graph["event_type_vocab"].keys()
        }
    )
    return group_names


def compute_selected_node_stats(
    graph_path: Path,
    selected_node_ids: Iterable[int],
    relation_group_scheme: str,
) -> Dict[int, Dict[str, object]]:
    selected_ids = sorted(set(int(node_id) for node_id in selected_node_ids))
    if not selected_ids:
        return {}

    graph = load_graph(graph_path)
    nodes_rows = load_nodes_by_id(graph_path.parent / "nodes.tsv")
    node_types_by_id = [nodes_rows[node_id]["node_type"] for node_id in range(int(graph["num_nodes"]))]

    edge_index = graph["edge_index"].to(dtype=torch.long)
    edge_type = graph["edge_type"].to(dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]

    out_degree = torch.bincount(src, minlength=int(graph["num_nodes"])).tolist()
    in_degree = torch.bincount(dst, minlength=int(graph["num_nodes"])).tolist()

    group_by_event_type = [""] * len(graph["event_type_vocab"])
    for event_name, event_id in graph["event_type_vocab"].items():
        group_by_event_type[event_id] = event_name_to_relation_group(event_name, scheme=relation_group_scheme)

    relation_group_names = relation_group_names_for_graph(graph, relation_group_scheme)
    selected_set = set(selected_ids)
    stats: Dict[int, Dict[str, object]] = {}
    unique_neighbor_sets: Dict[int, set[int]] = {}
    unique_neighbor_sets_by_type: Dict[int, Dict[str, set[int]]] = {}

    for node_id in selected_ids:
        stats[node_id] = {
            "in_degree": int(in_degree[node_id]),
            "out_degree": int(out_degree[node_id]),
            "total_degree": int(in_degree[node_id] + out_degree[node_id]),
            "incident_process_neighbor_edges": 0,
            "incident_file_neighbor_edges": 0,
            "incident_network_neighbor_edges": 0,
        }
        for group_name in relation_group_names:
            stats[node_id][f"incident_group_{group_name}"] = 0
            stats[node_id][f"incoming_group_{group_name}"] = 0
            stats[node_id][f"outgoing_group_{group_name}"] = 0
        unique_neighbor_sets[node_id] = set()
        unique_neighbor_sets_by_type[node_id] = {
            "process": set(),
            "file": set(),
            "network": set(),
        }

    src_list = src.tolist()
    dst_list = dst.tolist()
    edge_type_list = edge_type.tolist()

    for src_id, dst_id, event_type_id in zip(src_list, dst_list, edge_type_list):
        group_name = group_by_event_type[event_type_id]

        if src_id in selected_set:
            neighbor_type = node_types_by_id[dst_id]
            stats[src_id][f"incident_group_{group_name}"] += 1
            stats[src_id][f"outgoing_group_{group_name}"] += 1
            stats[src_id][f"incident_{neighbor_type}_neighbor_edges"] += 1
            unique_neighbor_sets[src_id].add(dst_id)
            unique_neighbor_sets_by_type[src_id][neighbor_type].add(dst_id)

        if dst_id in selected_set:
            neighbor_type = node_types_by_id[src_id]
            stats[dst_id][f"incident_group_{group_name}"] += 1
            stats[dst_id][f"incoming_group_{group_name}"] += 1
            stats[dst_id][f"incident_{neighbor_type}_neighbor_edges"] += 1
            unique_neighbor_sets[dst_id].add(src_id)
            unique_neighbor_sets_by_type[dst_id][neighbor_type].add(src_id)

    for node_id in selected_ids:
        stats[node_id]["unique_neighbors"] = len(unique_neighbor_sets[node_id])
        stats[node_id]["unique_process_neighbors"] = len(unique_neighbor_sets_by_type[node_id]["process"])
        stats[node_id]["unique_file_neighbors"] = len(unique_neighbor_sets_by_type[node_id]["file"])
        stats[node_id]["unique_network_neighbors"] = len(unique_neighbor_sets_by_type[node_id]["network"])

    return stats


def select_target_groups(
    rows: List[Dict[str, object]],
    node_type: str,
    fp_rank_max: int,
    fp_limit: int,
    gt_limit: int,
) -> Dict[str, List[Dict[str, object]]]:
    typed_rows = [row for row in rows if row["node_type"] == node_type]
    gt_rows = [row for row in typed_rows if int(row["is_gt"]) == 1]
    fp_pool_rows = [row for row in typed_rows if int(row["is_gt"]) == 0 and int(row["rank"]) <= fp_rank_max]

    gt_rows.sort(key=lambda item: int(item["rank"]))
    fp_pool_rows.sort(key=lambda item: int(item["rank"]))

    if gt_limit > 0:
        gt_rows = gt_rows[:gt_limit]
    if fp_limit > 0:
        fp_rows = fp_pool_rows[:fp_limit]
    else:
        fp_rows = fp_pool_rows

    return {
        "gt_rows": gt_rows,
        "fp_rows": fp_rows,
        "fp_pool_rows": fp_pool_rows,
        "typed_rows": typed_rows,
    }


def build_reference_lookup(
    reference_rows: List[Dict[str, object]],
    reference_nodes_by_id: Dict[int, Dict[str, object]],
    reference_topk: List[int],
) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in reference_rows:
        node_id = int(row["node_id"])
        node_meta = reference_nodes_by_id.get(node_id, {})
        item = {
            "reference_present": 1,
            "reference_rank": int(row["rank"]),
            "reference_score": float(row["score"]),
            "reference_is_gt": int(row["is_gt"]),
            "reference_decision_reason": node_meta.get("decision_reason", ""),
        }
        for cutoff in reference_topk:
            item[f"reference_in_top_{cutoff}"] = int(int(row["rank"]) <= cutoff)
        lookup[str(row["node_uuid"])] = item
    return lookup


def enrich_rows(
    rows: List[Dict[str, object]],
    nodes_by_id: Dict[int, Dict[str, object]],
    node_stats: Dict[int, Dict[str, object]],
    reference_lookup: Dict[str, Dict[str, object]],
    reference_topk: List[int],
    group_label: str,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        node_id = int(row["node_id"])
        uuid = str(row["node_uuid"])
        node_meta = nodes_by_id[node_id]
        stats = node_stats.get(node_id, {})
        ref = reference_lookup.get(uuid, {})

        item: Dict[str, object] = {
            "group_label": group_label,
            "rank": int(row["rank"]),
            "node_id": node_id,
            "node_uuid": uuid,
            "node_type": row["node_type"],
            "is_gt": int(row["is_gt"]),
            "score": float(row["score"]),
            "decision_reason": node_meta.get("decision_reason", ""),
        }
        item.update(stats)

        if ref:
            item.update(ref)
        else:
            item["reference_present"] = 0
            item["reference_rank"] = None
            item["reference_score"] = None
            item["reference_is_gt"] = None
            item["reference_decision_reason"] = None
            for cutoff in reference_topk:
                item[f"reference_in_top_{cutoff}"] = 0

        enriched.append(item)

    return enriched


def summarize_rows(
    rows: List[Dict[str, object]],
    reference_topk: List[int],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "count": len(rows),
        "decision_reasons_top": summarize_counter(Counter(str(row["decision_reason"]) for row in rows)),
    }

    for field in SUMMARY_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[field] = numeric_summary(values)

    summary["reference_overlap"] = {
        "present_in_reference": int(sum(int(row.get("reference_present", 0)) for row in rows)),
        "reference_gt": int(sum(1 for row in rows if row.get("reference_is_gt") == 1)),
    }
    for cutoff in reference_topk:
        summary["reference_overlap"][f"in_top_{cutoff}"] = int(
            sum(int(row.get(f"reference_in_top_{cutoff}", 0)) for row in rows)
        )

    reference_ranks = [float(row["reference_rank"]) for row in rows if row.get("reference_rank") is not None]
    summary["reference_rank"] = numeric_summary(reference_ranks)
    return summary


def build_metric_gap_summary(
    gt_rows: List[Dict[str, object]],
    fp_rows: List[Dict[str, object]],
    limit: int = 10,
) -> List[Dict[str, object]]:
    gt_summary = summarize_rows(gt_rows, [])
    fp_summary = summarize_rows(fp_rows, [])
    gaps: List[Dict[str, object]] = []

    for field in SUMMARY_FIELDS:
        gt_median = float(gt_summary[field]["median"])
        fp_median = float(fp_summary[field]["median"])
        gt_mean = float(gt_summary[field]["mean"])
        fp_mean = float(fp_summary[field]["mean"])
        scale = abs(math.log1p(fp_median) - math.log1p(gt_median))
        gaps.append(
            {
                "field": field,
                "gt_median": gt_median,
                "fp_median": fp_median,
                "median_delta_fp_minus_gt": fp_median - gt_median,
                "gt_mean": gt_mean,
                "fp_mean": fp_mean,
                "mean_delta_fp_minus_gt": fp_mean - gt_mean,
                "log1p_median_gap": scale,
            }
        )

    gaps.sort(key=lambda item: item["log1p_median_gap"], reverse=True)
    return gaps[:limit]


def to_serializable_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serializable: List[Dict[str, object]] = []
    for row in rows:
        item: Dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float):
                item[key] = round(value, 6)
            else:
                item[key] = value
        serializable.append(item)
    return serializable


def print_group_summary(name: str, summary: Dict[str, object], reference_topk: List[int]) -> None:
    rank_summary = summary["rank"]
    score_summary = summary["score"]
    degree_summary = summary["total_degree"]
    unique_summary = summary["unique_neighbors"]
    print(
        f"[{name}] count={summary['count']} "
        f"best_rank={int(rank_summary['min']) if rank_summary['count'] else 0} "
        f"median_rank={rank_summary['median']:.1f} "
        f"median_score={score_summary['median']:.6f} "
        f"median_total_degree={degree_summary['median']:.1f} "
        f"median_unique_neighbors={unique_summary['median']:.1f}",
        flush=True,
    )
    print(
        f"  decision_reasons={summary['decision_reasons_top'][:5]}",
        flush=True,
    )
    overlap = summary["reference_overlap"]
    overlap_parts = [
        f"present={overlap['present_in_reference']}",
        f"reference_gt={overlap['reference_gt']}",
    ]
    for cutoff in reference_topk:
        overlap_parts.append(f"reference_top_{cutoff}={overlap[f'in_top_{cutoff}']}")
    print(f"  reference_overlap={' '.join(overlap_parts)}", flush=True)


def print_preview(name: str, rows: List[Dict[str, object]], limit: int = 10) -> None:
    print(f"[{name}-preview]", flush=True)
    for row in rows[:limit]:
        print(
            "  "
            f"rank={row['rank']} "
            f"score={row['score']:.6f} "
            f"uuid={row['node_uuid']} "
            f"reason={row['decision_reason']} "
            f"deg={row.get('total_degree', 0)} "
            f"uniq={row.get('unique_neighbors', 0)} "
            f"ref_rank={row.get('reference_rank')} "
            f"ref_gt={row.get('reference_is_gt')}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    graph_root = Path(args.graph_root).expanduser().resolve()
    reference_topk = sorted(set(args.reference_topk or list(DEFAULT_REFERENCE_TOPK)))

    window_graph_paths = resolve_window_graph_paths(eval_dir, graph_root)
    if args.target_window not in window_graph_paths:
        raise FileNotFoundError(f"Could not resolve graph.pt for target window: {args.target_window}")
    if args.reference_window not in window_graph_paths:
        raise FileNotFoundError(f"Could not resolve graph.pt for reference window: {args.reference_window}")

    target_graph_path = window_graph_paths[args.target_window]
    reference_graph_path = window_graph_paths[args.reference_window]
    target_nodes_by_id = load_nodes_by_id(target_graph_path.parent / "nodes.tsv")
    reference_nodes_by_id = load_nodes_by_id(reference_graph_path.parent / "nodes.tsv")
    target_rows = load_scored_rows(eval_dir, args.target_window)
    reference_rows = load_scored_rows(eval_dir, args.reference_window)

    selected = select_target_groups(
        rows=target_rows,
        node_type=args.node_type,
        fp_rank_max=args.fp_rank_max,
        fp_limit=args.fp_limit,
        gt_limit=args.gt_limit,
    )

    selected_node_ids = [int(row["node_id"]) for row in selected["gt_rows"]] + [
        int(row["node_id"]) for row in selected["fp_rows"]
    ]
    target_stats = compute_selected_node_stats(
        graph_path=target_graph_path,
        selected_node_ids=selected_node_ids,
        relation_group_scheme=args.relation_group_scheme,
    )
    reference_lookup = build_reference_lookup(
        reference_rows=reference_rows,
        reference_nodes_by_id=reference_nodes_by_id,
        reference_topk=reference_topk,
    )

    enriched_gt_rows = enrich_rows(
        rows=selected["gt_rows"],
        nodes_by_id=target_nodes_by_id,
        node_stats=target_stats,
        reference_lookup=reference_lookup,
        reference_topk=reference_topk,
        group_label="target_gt",
    )
    enriched_fp_rows = enrich_rows(
        rows=selected["fp_rows"],
        nodes_by_id=target_nodes_by_id,
        node_stats=target_stats,
        reference_lookup=reference_lookup,
        reference_topk=reference_topk,
        group_label="target_top_false_positive",
    )

    gt_summary = summarize_rows(enriched_gt_rows, reference_topk)
    fp_summary = summarize_rows(enriched_fp_rows, reference_topk)
    metric_gaps = build_metric_gap_summary(enriched_gt_rows, enriched_fp_rows)

    output_name = args.output_name or (
        f"file_fp_analysis_{args.target_window}_vs_{args.reference_window}_{args.node_type}"
    )
    output_dir = eval_dir / output_name
    ensure_dir(output_dir)

    analysis = {
        "eval_dir": str(eval_dir),
        "target_window": args.target_window,
        "reference_window": args.reference_window,
        "node_type": args.node_type,
        "fp_rank_max": args.fp_rank_max,
        "fp_limit": args.fp_limit,
        "gt_limit": args.gt_limit,
        "relation_group_scheme": args.relation_group_scheme,
        "target_graph_path": str(target_graph_path),
        "reference_graph_path": str(reference_graph_path),
        "target_counts": {
            "typed_nodes": len(selected["typed_rows"]),
            "gt_nodes": len([row for row in selected["typed_rows"] if int(row["is_gt"]) == 1]),
            "false_positive_pool_within_rank_max": len(selected["fp_pool_rows"]),
            "exported_gt_nodes": len(enriched_gt_rows),
            "exported_top_false_positive_nodes": len(enriched_fp_rows),
        },
        "gt_summary": gt_summary,
        "top_false_positive_summary": fp_summary,
        "largest_median_gaps": metric_gaps,
        "top_false_positive_rows": to_serializable_rows(enriched_fp_rows),
        "gt_rows": to_serializable_rows(enriched_gt_rows),
    }

    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    write_tsv(output_dir / "top_false_positive_rows.tsv", to_serializable_rows(enriched_fp_rows))
    write_tsv(output_dir / "gt_rows.tsv", to_serializable_rows(enriched_gt_rows))
    write_tsv(output_dir / "largest_median_gaps.tsv", to_serializable_rows(metric_gaps))

    print(
        f"[analyze-file-fp] eval_dir={eval_dir} target_window={args.target_window} "
        f"reference_window={args.reference_window} node_type={args.node_type}",
        flush=True,
    )
    print(
        f"[analyze-file-fp] fp_pool_within_rank_max={len(selected['fp_pool_rows'])} "
        f"exported_fp={len(enriched_fp_rows)} exported_gt={len(enriched_gt_rows)}",
        flush=True,
    )
    print_group_summary("target-gt", gt_summary, reference_topk)
    print_group_summary("target-top-fp", fp_summary, reference_topk)
    print("[largest-median-gaps]", flush=True)
    for item in metric_gaps[:8]:
        print(
            "  "
            f"{item['field']}: "
            f"gt_median={item['gt_median']:.3f} "
            f"fp_median={item['fp_median']:.3f} "
            f"delta={item['median_delta_fp_minus_gt']:.3f}",
            flush=True,
        )
    print_preview("target-top-fp", enriched_fp_rows)
    print_preview("target-gt", enriched_gt_rows)
    print(f"[analyze-file-fp] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
