import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_loader import (
    VIEW_GROUP_ORDER,
    build_all_view_matrices,
    build_event_type_adjacencies,
    build_normalized_adjacency,
    build_relation_group_adjacencies,
    load_graph,
    RELATION_GROUP_SCHEMES,
    summarize_graph,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover - optional at runtime
    average_precision_score = None
    roc_auc_score = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_GRAPH = REPO_ROOT / "artifacts" / "graphs" / "train" / "graph.pt"
DEFAULT_EVAL_GRAPHS = [
    REPO_ROOT / "artifacts" / "graphs" / "val" / "graph.pt",
    REPO_ROOT / "artifacts" / "graphs" / "test_2018-04-12" / "graph.pt",
    REPO_ROOT / "artifacts" / "graphs" / "test_2018-04-13" / "graph.pt",
]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "training_runs"


VIEW_INPUT_DIMS = {
    "process_view": 62,
    "file_view": 37,
    "network_view": 37,
}

DECODER_TYPES = ("dot", "mlp", "rel_mlp")
MESSAGE_PASSING_TYPES = ("vanilla", "rel_grouped", "rgcn")
LOSS_TYPES = ("bce", "focal")

SELECTION_METRICS = {
    "val_edge_loss": ("val", "edge_loss", "min"),
    "val_roc_auc": ("val", "roc_auc", "max"),
    "val_average_precision": ("val", "average_precision", "max"),
    "test_2018-04-12_roc_auc": ("test_2018-04-12", "roc_auc", "max"),
    "test_2018-04-12_average_precision": ("test_2018-04-12", "average_precision", "max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full-batch multi-view GNN baseline on prebuilt CADETS graph artifacts."
    )
    parser.add_argument(
        "--train-graph",
        type=str,
        default=str(DEFAULT_TRAIN_GRAPH),
        help=f"Path to the training graph. Default: {DEFAULT_TRAIN_GRAPH}",
    )
    parser.add_argument(
        "--eval-graph",
        action="append",
        default=[],
        help="Optional eval graph(s). Defaults to val + both test windows.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device. Default: cuda if available, else cpu.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for each view encoder.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension for each view encoder.")
    parser.add_argument(
        "--message-passing-type",
        type=str,
        default="vanilla",
        choices=sorted(MESSAGE_PASSING_TYPES),
        help="Message passing type. Default: vanilla.",
    )
    parser.add_argument(
        "--relation-group-scheme",
        type=str,
        default="coarse_v1",
        choices=sorted(RELATION_GROUP_SCHEMES),
        help="Relation grouping scheme for --message-passing-type=rel_grouped.",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="dot",
        choices=sorted(DECODER_TYPES),
        help="Edge decoder type. Default: dot.",
    )
    parser.add_argument(
        "--decoder-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for the MLP-family decoder. Ignored when --decoder-type=dot.",
    )
    parser.add_argument(
        "--relation-embedding-dim",
        type=int,
        default=16,
        help="Relation embedding dimension for --decoder-type=rel_mlp.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied after the first graph layer.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="bce",
        choices=sorted(LOSS_TYPES),
        help="Edge reconstruction loss. Default: bce.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for --loss-type=focal. Ignored for BCE. Default: 2.0.",
    )
    parser.add_argument(
        "--pos-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier on the positive-edge loss term. Default: 1.0.",
    )
    parser.add_argument(
        "--neg-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier on the negative-edge loss term. Default: 1.0.",
    )
    parser.add_argument(
        "--negative-samples-per-positive",
        type=float,
        default=1.0,
        help="How many sampled negatives to draw per positive edge. Default: 1.0.",
    )
    parser.add_argument(
        "--train-pos-edge-cap",
        type=int,
        default=0,
        help="Optional cap on positive training edges per epoch. 0 means use all eligible edges.",
    )
    parser.add_argument(
        "--eval-pos-edge-cap",
        type=int,
        default=0,
        help="Optional cap on positive eval edges. 0 means use all eval edges.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for checkpoints and metrics. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_edge_loss",
        choices=sorted(SELECTION_METRICS.keys()),
        help="Metric used to save best_model.pt. Default: val_edge_loss",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_eval_graphs(eval_graph_args: List[str]) -> List[Path]:
    if eval_graph_args:
        return [Path(item).expanduser().resolve() for item in eval_graph_args]
    return [path.resolve() for path in DEFAULT_EVAL_GRAPHS]


def graph_window_name(graph_path: Path) -> str:
    return graph_path.parent.name


def build_train_edge_mask(graph: Dict[str, object]) -> torch.Tensor:
    edge_index = graph["edge_index"]
    normal_mask = graph["normal_mask"].bool()
    src = edge_index[0].long()
    dst = edge_index[1].long()
    return normal_mask[src] & normal_mask[dst]


def maybe_cap_edge_index(edge_index: torch.Tensor, cap: int) -> torch.Tensor:
    if cap <= 0 or edge_index.shape[1] <= cap:
        return edge_index
    perm = torch.randperm(edge_index.shape[1], device=edge_index.device)[:cap]
    return edge_index[:, perm]


def maybe_cap_edges(edge_index: torch.Tensor, edge_type: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cap <= 0 or edge_index.shape[1] <= cap:
        return edge_index, edge_type
    perm = torch.randperm(edge_index.shape[1], device=edge_index.device)[:cap]
    return edge_index[:, perm], edge_type[perm]


def sample_negative_edges(num_nodes: int, num_samples: int, device: torch.device) -> torch.Tensor:
    src = torch.randint(0, num_nodes, (num_samples,), device=device)
    dst = torch.randint(0, num_nodes, (num_samples,), device=device)

    same = src == dst
    while same.any():
        dst[same] = torch.randint(0, num_nodes, (int(same.sum().item()),), device=device)
        same = src == dst

    return torch.stack([src, dst], dim=0)


def sample_negative_edge_types(positive_edge_type: torch.Tensor) -> torch.Tensor:
    if positive_edge_type.numel() == 0:
        return positive_edge_type
    perm = torch.randperm(positive_edge_type.shape[0], device=positive_edge_type.device)
    return positive_edge_type[perm]


def negative_sample_count(num_positive_edges: int, negative_samples_per_positive: float) -> int:
    if num_positive_edges <= 0:
        return 0
    return max(1, int(math.ceil(num_positive_edges * negative_samples_per_positive)))


def edge_loss_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)

    if loss_type != "focal":
        raise ValueError(f"Unsupported loss type: {loss_type}")

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
    focal_weight = torch.pow((1.0 - pt).clamp_min(1e-8), focal_gamma)
    return (focal_weight * bce).mean()


def compute_node_scores(num_nodes: int, edge_index: torch.Tensor, edge_error: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].long()
    dst = edge_index[1].long()

    score_sum = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    degree = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    ones = torch.ones(edge_error.shape[0], device=edge_index.device, dtype=torch.float32)

    score_sum.index_add_(0, src, edge_error)
    score_sum.index_add_(0, dst, edge_error)
    degree.index_add_(0, src, ones)
    degree.index_add_(0, dst, ones)

    return score_sum / degree.clamp_min(1.0)


def compute_binary_metrics(y_true: torch.Tensor, scores: torch.Tensor) -> Dict[str, float | None]:
    if roc_auc_score is None or average_precision_score is None:
        return {"roc_auc": None, "average_precision": None}

    y_np = y_true.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()

    positives = int(y_true.sum().item())
    negatives = int((1 - y_true).sum().item())
    if positives == 0 or negatives == 0:
        return {"roc_auc": None, "average_precision": None}

    return {
        "roc_auc": float(roc_auc_score(y_np, s_np)),
        "average_precision": float(average_precision_score(y_np, s_np)),
    }


def find_eval_metric(eval_metrics: List[Dict[str, object]], name: str) -> Dict[str, object] | None:
    return next((metric for metric in eval_metrics if metric["name"] == name), None)


def selection_score(eval_metrics: List[Dict[str, object]], selection_metric: str) -> tuple[float | None, str]:
    eval_name, field_name, mode = SELECTION_METRICS[selection_metric]
    metric = find_eval_metric(eval_metrics, eval_name)
    if metric is None:
        return None, mode
    value = metric.get(field_name)
    if value is None:
        return None, mode
    return float(value), mode


def infer_view_input_dims_from_graph(graph: Dict[str, object]) -> Dict[str, int]:
    feature_groups = graph["feature_groups"]
    return {
        view_name: sum(int(feature_groups[group_name]["x"].shape[1]) for group_name in group_names)
        for view_name, group_names in VIEW_GROUP_ORDER.items()
    }


def normalize_view_input_dims(view_input_dims: Dict[str, object] | None) -> Dict[str, int]:
    if view_input_dims is None:
        return {key: int(value) for key, value in VIEW_INPUT_DIMS.items()}

    normalized = {}
    for view_name in VIEW_GROUP_ORDER:
        if view_name not in view_input_dims:
            raise ValueError(f"Missing view_input_dims entry for {view_name}")
        normalized[view_name] = int(view_input_dims[view_name])
    return normalized


def validate_matching_view_input_dims(
    reference: Dict[str, object],
    candidate: Dict[str, object],
    context: str,
) -> None:
    normalized_reference = normalize_view_input_dims(reference)
    normalized_candidate = normalize_view_input_dims(candidate)
    if normalized_reference != normalized_candidate:
        raise ValueError(
            f"{context} has mismatched view_input_dims: "
            f"expected={normalized_reference} actual={normalized_candidate}"
        )


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.sparse.mm(adjacency, x)
        return self.linear(propagated)


class RelationGroupedGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.relation_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])

    def forward(self, x: torch.Tensor, relation_adjacencies: List[torch.Tensor]) -> torch.Tensor:
        out = self.self_linear(x)
        for adjacency, linear in zip(relation_adjacencies, self.relation_linears):
            out = out + linear(torch.sparse.mm(adjacency, x))
        return out / float(len(self.relation_linears) + 1)


class VanillaViewEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = SparseGCNLayer(in_dim, hidden_dim)
        self.layer2 = SparseGCNLayer(hidden_dim, latent_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adjacency)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.layer2(h, adjacency)
        return z


class RelationGroupedViewEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float, num_relations: int) -> None:
        super().__init__()
        self.layer1 = RelationGroupedGCNLayer(in_dim, hidden_dim, num_relations)
        self.layer2 = RelationGroupedGCNLayer(hidden_dim, latent_dim, num_relations)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, relation_adjacencies: List[torch.Tensor]) -> torch.Tensor:
        h = self.layer1(x, relation_adjacencies)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.layer2(h, relation_adjacencies)
        return z


class RGCNViewEncoder(RelationGroupedViewEncoder):
    pass


class MultiViewFullBatchGAE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        view_input_dims: Dict[str, int] | None = None,
        decoder_type: str = "dot",
        decoder_hidden_dim: int = 64,
        message_passing_type: str = "vanilla",
        num_relation_groups: int = 0,
        num_relations: int = 0,
        relation_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.message_passing_type = message_passing_type
        self.view_input_dims = normalize_view_input_dims(view_input_dims)
        if message_passing_type == "vanilla":
            encoder_cls = VanillaViewEncoder
            self.process_encoder = encoder_cls(self.view_input_dims["process_view"], hidden_dim, latent_dim, dropout)
            self.file_encoder = encoder_cls(self.view_input_dims["file_view"], hidden_dim, latent_dim, dropout)
            self.network_encoder = encoder_cls(self.view_input_dims["network_view"], hidden_dim, latent_dim, dropout)
        elif message_passing_type in {"rel_grouped", "rgcn"}:
            if num_relation_groups <= 0:
                raise ValueError(
                    "num_relation_groups must be positive when message_passing_type is relation-aware"
                )
            encoder_cls = RelationGroupedViewEncoder if message_passing_type == "rel_grouped" else RGCNViewEncoder
            self.process_encoder = encoder_cls(
                self.view_input_dims["process_view"], hidden_dim, latent_dim, dropout, num_relation_groups
            )
            self.file_encoder = encoder_cls(
                self.view_input_dims["file_view"], hidden_dim, latent_dim, dropout, num_relation_groups
            )
            self.network_encoder = encoder_cls(
                self.view_input_dims["network_view"], hidden_dim, latent_dim, dropout, num_relation_groups
            )
        else:
            raise ValueError(f"Unsupported message passing type: {message_passing_type}")
        self.decoder_type = decoder_type

        self.process_gate = nn.Linear(latent_dim, 1)
        self.file_gate = nn.Linear(latent_dim, 1)
        self.network_gate = nn.Linear(latent_dim, 1)
        if decoder_type == "mlp":
            self.edge_decoder = nn.Sequential(
                nn.Linear(latent_dim * 4, decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(decoder_hidden_dim, 1),
            )
        elif decoder_type == "rel_mlp":
            if num_relations <= 0:
                raise ValueError("num_relations must be positive when decoder_type=rel_mlp")
            self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)
            self.edge_decoder = nn.Sequential(
                nn.Linear(latent_dim * 4 + relation_embedding_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(decoder_hidden_dim, 1),
            )
        elif decoder_type != "dot":
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

    def encode(
        self,
        x_views: Dict[str, torch.Tensor],
        adjacency: torch.Tensor,
        relation_adjacencies: List[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.message_passing_type == "vanilla":
            z_process = self.process_encoder(x_views["process_view"], adjacency)
            z_file = self.file_encoder(x_views["file_view"], adjacency)
            z_network = self.network_encoder(x_views["network_view"], adjacency)
        else:
            if relation_adjacencies is None:
                raise ValueError("relation_adjacencies are required when message_passing_type=rel_grouped")
            z_process = self.process_encoder(x_views["process_view"], relation_adjacencies)
            z_file = self.file_encoder(x_views["file_view"], relation_adjacencies)
            z_network = self.network_encoder(x_views["network_view"], relation_adjacencies)

        gate_scores = torch.cat(
            [
                self.process_gate(z_process),
                self.file_gate(z_file),
                self.network_gate(z_network),
            ],
            dim=1,
        )
        gate_alpha = F.softmax(gate_scores, dim=1)

        z_fused = (
            gate_alpha[:, 0:1] * z_process
            + gate_alpha[:, 1:2] * z_file
            + gate_alpha[:, 2:3] * z_network
        )

        return {
            "process_view": z_process,
            "file_view": z_file,
            "network_view": z_network,
            "gate_alpha": gate_alpha,
            "z_fused": z_fused,
        }

    def decode_edges(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if self.decoder_type == "dot":
            return (z[src] * z[dst]).sum(dim=1)

        edge_feature_parts = [
            z[src],
            z[dst],
            z[src] * z[dst],
            torch.abs(z[src] - z[dst]),
        ]
        if self.decoder_type == "rel_mlp":
            if edge_type is None:
                raise ValueError("edge_type is required when decoder_type=rel_mlp")
            edge_feature_parts.append(self.relation_embedding(edge_type.long()))
        edge_features = torch.cat(edge_feature_parts, dim=1)
        return self.edge_decoder(edge_features).squeeze(-1)


def prepare_graph_payload(
    graph_path: Path,
    device: torch.device,
    restrict_to_normal_edges: bool,
    message_passing_type: str = "vanilla",
    relation_group_scheme: str = "coarse_v1",
) -> Dict[str, object]:
    graph = load_graph(graph_path)
    summary = summarize_graph(graph)
    view_input_dims = infer_view_input_dims_from_graph(graph)
    views = build_all_view_matrices(graph)

    x_views = {
        view_name: payload["x"].to(device=device, dtype=torch.float32)
        for view_name, payload in views.items()
    }

    edge_index = graph["edge_index"].to(device=device, dtype=torch.long)
    edge_type = graph["edge_type"].to(device=device, dtype=torch.long)
    edge_weight = torch.log1p(graph["edge_event_count"].to(device=device, dtype=torch.float32))
    y = graph["y"].to(device=device, dtype=torch.float32)
    gt_mask = graph["gt_mask"].to(device=device, dtype=torch.bool)
    normal_mask = graph["normal_mask"].to(device=device, dtype=torch.bool)

    if restrict_to_normal_edges:
        train_mask = build_train_edge_mask(graph).to(device=device, dtype=torch.bool)
        edge_index = edge_index[:, train_mask]
        edge_type = edge_type[train_mask]
        edge_weight = edge_weight[train_mask]

    adjacency = build_normalized_adjacency(
        edge_index=edge_index,
        num_nodes=int(graph["num_nodes"]),
        edge_weight=edge_weight,
        add_self_loops=True,
        device=device,
    )

    relation_group_names: List[str] = []
    relation_adjacencies: List[torch.Tensor] | None = None
    if message_passing_type == "rel_grouped":
        relation_group_names, relation_adjacencies = build_relation_group_adjacencies(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=int(graph["num_nodes"]),
            event_type_vocab=graph["event_type_vocab"],
            edge_weight=edge_weight,
            scheme=relation_group_scheme,
            device=device,
        )
    elif message_passing_type == "rgcn":
        relation_group_names, relation_adjacencies = build_event_type_adjacencies(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=int(graph["num_nodes"]),
            event_type_vocab=graph["event_type_vocab"],
            edge_weight=edge_weight,
            device=device,
        )

    return {
        "path": str(graph_path),
        "name": graph_window_name(graph_path),
        "summary": summary,
        "view_input_dims": view_input_dims,
        "num_nodes": int(graph["num_nodes"]),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "adjacency": adjacency,
        "relation_group_names": relation_group_names,
        "relation_adjacencies": relation_adjacencies,
        "x_views": x_views,
        "y": y,
        "gt_mask": gt_mask,
        "normal_mask": normal_mask,
    }


def evaluate_graph(
    model: MultiViewFullBatchGAE,
    graph_payload: Dict[str, object],
    edge_cap: int,
    loss_type: str,
    focal_gamma: float,
    pos_loss_weight: float,
    neg_loss_weight: float,
    negative_samples_per_positive: float,
) -> Dict[str, object]:
    validate_matching_view_input_dims(model.view_input_dims, graph_payload["view_input_dims"], graph_payload["name"])
    model.eval()
    with torch.no_grad():
        encoded = model.encode(
            graph_payload["x_views"],
            graph_payload["adjacency"],
            graph_payload.get("relation_adjacencies"),
        )
        z_fused = encoded["z_fused"]

        positive_edges, positive_edge_type = maybe_cap_edges(
            graph_payload["edge_index"],
            graph_payload["edge_type"],
            edge_cap,
        )
        num_negative_edges = negative_sample_count(
            int(positive_edges.shape[1]),
            negative_samples_per_positive,
        )
        negative_edges = sample_negative_edges(graph_payload["num_nodes"], num_negative_edges, z_fused.device)
        if positive_edge_type.numel() == 0:
            negative_edge_type = positive_edge_type
        else:
            sampled_indices = torch.randint(
                0,
                positive_edge_type.shape[0],
                (num_negative_edges,),
                device=positive_edge_type.device,
            )
            negative_edge_type = positive_edge_type[sampled_indices]

        pos_logits = model.decode_edges(z_fused, positive_edges, positive_edge_type)
        neg_logits = model.decode_edges(z_fused, negative_edges, negative_edge_type)

        pos_loss = edge_loss_term(pos_logits, torch.ones_like(pos_logits), loss_type, focal_gamma)
        neg_loss = edge_loss_term(neg_logits, torch.zeros_like(neg_logits), loss_type, focal_gamma)
        edge_loss = float((pos_loss_weight * pos_loss + neg_loss_weight * neg_loss).item())

        edge_error = 1.0 - torch.sigmoid(
            model.decode_edges(z_fused, graph_payload["edge_index"], graph_payload["edge_type"])
        )
        node_scores = compute_node_scores(graph_payload["num_nodes"], graph_payload["edge_index"], edge_error)
        node_metrics = compute_binary_metrics(graph_payload["y"], node_scores)

        gate_alpha = encoded["gate_alpha"]
        gate_means = gate_alpha.mean(dim=0).detach().cpu().tolist()

    return {
        "name": graph_payload["name"],
        "path": graph_payload["path"],
        "edge_loss": edge_loss,
        "roc_auc": node_metrics["roc_auc"],
        "average_precision": node_metrics["average_precision"],
        "gt_nodes": int(graph_payload["y"].sum().item()),
        "num_nodes": graph_payload["num_nodes"],
        "num_edges": int(graph_payload["edge_index"].shape[1]),
        "mean_gate_alpha": {
            "process_view": float(gate_means[0]),
            "file_view": float(gate_means[1]),
            "network_view": float(gate_means[2]),
        },
    }


def train_epoch(
    model: MultiViewFullBatchGAE,
    train_payload: Dict[str, object],
    optimizer: torch.optim.Optimizer,
    train_pos_edge_cap: int,
    loss_type: str,
    focal_gamma: float,
    pos_loss_weight: float,
    neg_loss_weight: float,
    negative_samples_per_positive: float,
) -> Dict[str, float]:
    validate_matching_view_input_dims(model.view_input_dims, train_payload["view_input_dims"], train_payload["name"])
    model.train()
    optimizer.zero_grad(set_to_none=True)

    encoded = model.encode(
        train_payload["x_views"],
        train_payload["adjacency"],
        train_payload.get("relation_adjacencies"),
    )
    z_fused = encoded["z_fused"]

    positive_edges, positive_edge_type = maybe_cap_edges(
        train_payload["edge_index"],
        train_payload["edge_type"],
        train_pos_edge_cap,
    )
    num_negative_edges = negative_sample_count(
        int(positive_edges.shape[1]),
        negative_samples_per_positive,
    )
    negative_edges = sample_negative_edges(train_payload["num_nodes"], num_negative_edges, z_fused.device)
    if positive_edge_type.numel() == 0:
        negative_edge_type = positive_edge_type
    else:
        sampled_indices = torch.randint(
            0,
            positive_edge_type.shape[0],
            (num_negative_edges,),
            device=positive_edge_type.device,
        )
        negative_edge_type = positive_edge_type[sampled_indices]

    pos_logits = model.decode_edges(z_fused, positive_edges, positive_edge_type)
    neg_logits = model.decode_edges(z_fused, negative_edges, negative_edge_type)

    pos_loss = edge_loss_term(pos_logits, torch.ones_like(pos_logits), loss_type, focal_gamma)
    neg_loss = edge_loss_term(neg_logits, torch.zeros_like(neg_logits), loss_type, focal_gamma)
    loss = pos_loss_weight * pos_loss + neg_loss_weight * neg_loss
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        gate_alpha = encoded["gate_alpha"].mean(dim=0).detach().cpu().tolist()

    return {
        "loss": float(loss.item()),
        "pos_loss": float(pos_loss.item()),
        "neg_loss": float(neg_loss.item()),
        "mean_gate_alpha_process": float(gate_alpha[0]),
        "mean_gate_alpha_file": float(gate_alpha[1]),
        "mean_gate_alpha_network": float(gate_alpha[2]),
        "num_train_edges": int(positive_edges.shape[1]),
        "num_negative_edges": int(num_negative_edges),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    train_graph_path = Path(args.train_graph).expanduser().resolve()
    eval_graph_paths = resolve_eval_graphs(args.eval_graph)
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    run_name = args.run_name or f"multiview_gae_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    ensure_dir(run_dir)

    print(f"[train-gnn] device={device}", flush=True)
    print(f"[train-gnn] train_graph={train_graph_path}", flush=True)
    print(f"[train-gnn] eval_graphs={[str(path) for path in eval_graph_paths]}", flush=True)

    train_payload = prepare_graph_payload(
        train_graph_path,
        device=device,
        restrict_to_normal_edges=True,
        message_passing_type=args.message_passing_type,
        relation_group_scheme=args.relation_group_scheme,
    )
    eval_payloads = [
        prepare_graph_payload(
            path,
            device=device,
            restrict_to_normal_edges=False,
            message_passing_type=args.message_passing_type,
            relation_group_scheme=args.relation_group_scheme,
        )
        for path in eval_graph_paths
    ]

    print("[train-gnn] loaded train graph summary:", flush=True)
    print(json.dumps(train_payload["summary"], indent=2), flush=True)
    for payload in eval_payloads:
        validate_matching_view_input_dims(
            train_payload["view_input_dims"],
            payload["view_input_dims"],
            f"eval graph {payload['name']}",
        )

    model = MultiViewFullBatchGAE(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        view_input_dims=train_payload["view_input_dims"],
        decoder_type=args.decoder_type,
        decoder_hidden_dim=args.decoder_hidden_dim,
        message_passing_type=args.message_passing_type,
        num_relation_groups=len(train_payload.get("relation_group_names", [])),
        num_relations=int(train_payload["summary"]["event_type_count"]),
        relation_embedding_dim=args.relation_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, object]] = []
    _, _, selection_mode = SELECTION_METRICS[args.selection_metric]
    best_score = math.inf if selection_mode == "min" else -math.inf
    best_record: Dict[str, object] | None = None
    best_checkpoint = run_dir / "best_model.pt"

    config = {
        "train_graph": str(train_graph_path),
        "eval_graphs": [str(path) for path in eval_graph_paths],
        "device": str(device),
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "view_input_dims": train_payload["view_input_dims"],
        "message_passing_type": args.message_passing_type,
        "relation_group_scheme": args.relation_group_scheme,
        "num_relation_groups": len(train_payload.get("relation_group_names", [])),
        "decoder_type": args.decoder_type,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "relation_embedding_dim": args.relation_embedding_dim,
        "num_relations": int(train_payload["summary"]["event_type_count"]),
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "pos_loss_weight": args.pos_loss_weight,
        "neg_loss_weight": args.neg_loss_weight,
        "negative_samples_per_positive": args.negative_samples_per_positive,
        "train_pos_edge_cap": args.train_pos_edge_cap,
        "eval_pos_edge_cap": args.eval_pos_edge_cap,
        "run_name": run_name,
        "selection_metric": args.selection_metric,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model=model,
            train_payload=train_payload,
            optimizer=optimizer,
            train_pos_edge_cap=args.train_pos_edge_cap,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            pos_loss_weight=args.pos_loss_weight,
            neg_loss_weight=args.neg_loss_weight,
            negative_samples_per_positive=args.negative_samples_per_positive,
        )
        eval_metrics = [
            evaluate_graph(
                model=model,
                graph_payload=payload,
                edge_cap=args.eval_pos_edge_cap,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                pos_loss_weight=args.pos_loss_weight,
                neg_loss_weight=args.neg_loss_weight,
                negative_samples_per_positive=args.negative_samples_per_positive,
            )
            for payload in eval_payloads
        ]

        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(record)

        print(
            f"[train-gnn] epoch={epoch:03d} "
            f"loss={train_metrics['loss']:.6f} "
            f"pos_loss={train_metrics['pos_loss']:.6f} "
            f"neg_loss={train_metrics['neg_loss']:.6f} "
            f"neg_edges={train_metrics['num_negative_edges']} "
            f"seconds={elapsed:.2f}",
            flush=True,
        )
        for metric in eval_metrics:
            print(
                f"  [eval] {metric['name']}: edge_loss={metric['edge_loss']:.6f} "
                f"roc_auc={metric['roc_auc']} ap={metric['average_precision']} "
                f"gt_nodes={metric['gt_nodes']}",
                flush=True,
            )

        score, mode = selection_score(eval_metrics, args.selection_metric)
        if score is not None:
            improved = score < best_score if mode == "min" else score > best_score
            if improved:
                best_score = score
                best_record = {
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_mode": mode,
                    "selection_score": score,
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "history": history,
                        "best_record": best_record,
                    },
                    best_checkpoint,
                )
                with (run_dir / "best_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(best_record, handle, indent=2)
                print(
                    f"[train-gnn] new best checkpoint: epoch={epoch:03d} "
                    f"{args.selection_metric}={score:.6f}",
                    flush=True,
                )

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "history": history,
            "best_record": best_record,
        },
        run_dir / "last_model.pt",
    )

    if best_record is not None:
        print(
            f"[train-gnn] best epoch={best_record['epoch']:03d} "
            f"{best_record['selection_metric']}={best_record['selection_score']:.6f}",
            flush=True,
        )
    print(f"[train-gnn] finished. outputs -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
