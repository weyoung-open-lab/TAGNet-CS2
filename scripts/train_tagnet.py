from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs"
SNAPSHOT_SECONDS = (10, 15, 20)
AREAS = ("A", "B", "Mid", "CT")
ECON_TYPES = ("pistol", "eco", "half_buy", "long_gun")
PREV_ATTACK_SITES = ("A", "B", "Mid", "unknown")
PREV_WINNERS = ("ct", "t", "none")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


@dataclass
class SplitData:
    sample_ids: list[str]
    match_ids: list[str]
    rounds: list[int]
    weak_labels: list[str | None]
    weak_label_mask: list[bool]
    node_features: torch.Tensor
    edge_distances: torch.Tensor
    place_targets: torch.Tensor
    area_targets: torch.Tensor
    distance_targets: torch.Tensor
    context_features: torch.Tensor

    def subset(self, indices: list[int]) -> "SplitData":
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        return SplitData(
            sample_ids=[self.sample_ids[i] for i in indices],
            match_ids=[self.match_ids[i] for i in indices],
            rounds=[self.rounds[i] for i in indices],
            weak_labels=[self.weak_labels[i] for i in indices],
            weak_label_mask=[self.weak_label_mask[i] for i in indices],
            node_features=self.node_features.index_select(0, idx_tensor),
            edge_distances=self.edge_distances.index_select(0, idx_tensor),
            place_targets=self.place_targets.index_select(0, idx_tensor),
            area_targets=self.area_targets.index_select(0, idx_tensor),
            distance_targets=self.distance_targets.index_select(0, idx_tensor),
            context_features=self.context_features.index_select(0, idx_tensor),
        )


def slice_split_time(split: SplitData, snapshot_second: int, snapshot_seconds: tuple[int, ...]) -> SplitData:
    second_to_index = {second: idx for idx, second in enumerate(snapshot_seconds)}
    time_idx = second_to_index[snapshot_second]
    time_slice = slice(time_idx, time_idx + 1)
    return SplitData(
        sample_ids=split.sample_ids,
        match_ids=split.match_ids,
        rounds=split.rounds,
        weak_labels=split.weak_labels,
        weak_label_mask=split.weak_label_mask,
        node_features=split.node_features[:, time_slice, :, :],
        edge_distances=split.edge_distances[:, time_slice, :, :],
        place_targets=split.place_targets[:, time_slice, :],
        area_targets=split.area_targets[:, time_slice, :],
        distance_targets=split.distance_targets[:, time_slice, :],
        context_features=split.context_features[:, time_slice, :],
    )


def normalize_min_max(value: float, min_value: float, max_value: float) -> float:
    denom = max(max_value - min_value, 1e-9)
    out = (value - min_value) / denom
    return float(min(max(out, 0.0), 1.0))


def build_context_feature(
    row: pd.Series,
    score_min: float,
    score_max: float,
    smoke_min: float,
    smoke_max: float,
    inferno_min: float,
    inferno_max: float,
) -> list[float]:
    total_smokes = float(row["ct_smokes_used"])
    total_infernos = float(row["ct_infernos_used"])
    score_diff = float(row["score_diff_ct_minus_t_before"])

    smoke_area_total = max(total_smokes, 1.0)
    inferno_area_total = max(total_infernos, 1.0)

    return [
        float(row["ct_total_equip_norm"]),
        float(row["t_total_equip_norm"]),
        normalize_min_max(score_diff, score_min, score_max),
        normalize_min_max(total_smokes, smoke_min, smoke_max),
        normalize_min_max(total_infernos, inferno_min, inferno_max),
        float(row["ct_smoke_throw_area_A"]) / smoke_area_total,
        float(row["ct_smoke_throw_area_B"]) / smoke_area_total,
        float(row["ct_smoke_throw_area_Mid"]) / smoke_area_total,
        float(row["ct_smoke_throw_area_CT"]) / smoke_area_total,
        float(row["ct_inferno_throw_area_A"]) / inferno_area_total,
        float(row["ct_inferno_throw_area_B"]) / inferno_area_total,
        float(row["ct_inferno_throw_area_Mid"]) / inferno_area_total,
        float(row["ct_inferno_throw_area_CT"]) / inferno_area_total,
        float(bool(row["pistol_round_flag"])),
        *one_hot(str(row["ct_econ_type"]), ECON_TYPES),
        *one_hot(str(row["prev_round_attack_site"]), PREV_ATTACK_SITES),
        *one_hot(
            "none" if pd.isna(row["prev_round_winner_side"]) else str(row["prev_round_winner_side"]),
            PREV_WINNERS,
        ),
    ]


def load_split_data(data_dir: Path) -> tuple[dict[str, SplitData], dict[str, Any]]:
    nodes_df = pd.read_parquet(data_dir / "ct_snapshot_nodes.parquet")
    edges_df = pd.read_parquet(data_dir / "ct_snapshot_edges.parquet")
    context_df = pd.read_parquet(data_dir / "ct_round_context.parquet")
    rounds_df = pd.read_parquet(data_dir / "ct_round_samples.parquet")
    normalization_stats = json.loads((data_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    snapshot_seconds = tuple(int(v) for v in normalization_stats.get("snapshot_seconds", list(SNAPSHOT_SECONDS)))

    with (data_dir / "place_vocab.json").open("r", encoding="utf-8") as fh:
        place_vocab_data = json.load(fh)
    place_vocab = place_vocab_data["place_vocab"]
    place_to_idx = place_vocab_data["place_to_idx"]
    place_dim = len(place_vocab)

    train_context = context_df[context_df["split"] == "train"]
    score_min = float(train_context["score_diff_ct_minus_t_before"].min())
    score_max = float(train_context["score_diff_ct_minus_t_before"].max())
    smoke_min = float(train_context["ct_smokes_used"].min())
    smoke_max = float(train_context["ct_smokes_used"].max())
    inferno_min = float(train_context["ct_infernos_used"].min())
    inferno_max = float(train_context["ct_infernos_used"].max())

    sample_to_meta = {
        row["sample_id"]: {
            "match_id": row["match_id"],
            "round_num": int(row["round_num"]),
            "split": row["split"],
        }
        for _, row in rounds_df.iterrows()
    }

    sample_ids = rounds_df["sample_id"].tolist()
    node_feat_dim = place_dim + 2
    context_dim = None

    per_sample: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        per_sample[sample_id] = {
            "node_features": [],
            "place_targets": [],
            "area_targets": [],
            "distance_targets": [],
            "context_features": [],
            "weak_labels": {},
        }

    grouped_nodes = nodes_df.groupby(["sample_id", "second"], sort=True)
    for (sample_id, second), group in grouped_nodes:
        group = group.sort_values("player_slot")
        features: list[list[float]] = []
        place_hist = [0.0] * place_dim
        area_hist = [0.0] * len(AREAS)
        for row in group.itertuples(index=False):
            place_name = str(row.place)
            place_index = int(place_to_idx[place_name])
            place_one_hot = [0.0] * place_dim
            place_one_hot[place_index] = 1.0
            feature = place_one_hot + [float(row.health_norm), float(row.equip_norm)]
            features.append(feature)
            place_hist[place_index] += 1.0 / 5.0
            if row.area in AREAS:
                area_hist[AREAS.index(row.area)] += 1.0 / 5.0
        per_sample[sample_id]["node_features"].append((second, features))
        per_sample[sample_id]["place_targets"].append((second, place_hist))
        per_sample[sample_id]["area_targets"].append((second, area_hist))

    grouped_edges = edges_df.groupby(["sample_id", "second"], sort=True)
    for (sample_id, second), group in grouped_edges:
        upper_triangle: list[float] = []
        sorted_group = group.sort_values(["src_slot", "dst_slot"])
        for row in sorted_group.itertuples(index=False):
            upper_triangle.append(float(row.distance_xy_norm))
        per_sample[sample_id]["distance_targets"].append((second, upper_triangle))

    grouped_context = context_df.groupby(["sample_id", "second"], sort=True)
    for (sample_id, second), group in grouped_context:
        row = group.iloc[0]
        context_vector = build_context_feature(
            row=row,
            score_min=score_min,
            score_max=score_max,
            smoke_min=smoke_min,
            smoke_max=smoke_max,
            inferno_min=inferno_min,
            inferno_max=inferno_max,
        )
        context_dim = len(context_vector)
        per_sample[sample_id]["context_features"].append((second, context_vector))
        per_sample[sample_id]["weak_labels"][second] = None if pd.isna(row["weak_label"]) else str(row["weak_label"])

    split_buckets: dict[str, dict[str, list[Any]]] = {}
    for split_name in ("train", "val", "test"):
        split_buckets[split_name] = {
            "sample_ids": [],
            "match_ids": [],
            "rounds": [],
            "weak_labels": [],
            "weak_label_mask": [],
            "node_features": [],
            "edge_distances": [],
            "place_targets": [],
            "area_targets": [],
            "distance_targets": [],
            "context_features": [],
        }

    for sample_id in sample_ids:
        meta = sample_to_meta[sample_id]
        split_name = str(meta["split"])
        bucket = split_buckets[split_name]
        node_seq = [features for _, features in sorted(per_sample[sample_id]["node_features"], key=lambda item: item[0])]
        place_seq = [hist for _, hist in sorted(per_sample[sample_id]["place_targets"], key=lambda item: item[0])]
        area_seq = [hist for _, hist in sorted(per_sample[sample_id]["area_targets"], key=lambda item: item[0])]
        dist_seq = [hist for _, hist in sorted(per_sample[sample_id]["distance_targets"], key=lambda item: item[0])]
        context_seq = [feat for _, feat in sorted(per_sample[sample_id]["context_features"], key=lambda item: item[0])]
        weak_label = per_sample[sample_id]["weak_labels"].get(15)

        if (
            len(node_seq) != len(snapshot_seconds)
            or len(place_seq) != len(snapshot_seconds)
            or len(dist_seq) != len(snapshot_seconds)
            or len(context_seq) != len(snapshot_seconds)
        ):
            continue

        bucket["sample_ids"].append(sample_id)
        bucket["match_ids"].append(str(meta["match_id"]))
        bucket["rounds"].append(int(meta["round_num"]))
        bucket["weak_labels"].append(weak_label)
        bucket["weak_label_mask"].append(weak_label is not None)
        bucket["node_features"].append(node_seq)
        bucket["edge_distances"].append(
            [
                [
                    [0.0 for _ in range(5)]
                    for _ in range(5)
                ]
                for _ in range(len(snapshot_seconds))
            ]
        )
        bucket["place_targets"].append(place_seq)
        bucket["area_targets"].append(area_seq)
        bucket["distance_targets"].append(dist_seq)
        bucket["context_features"].append(context_seq)

    for split_name in ("train", "val", "test"):
        bucket = split_buckets[split_name]
        for idx, sample_id in enumerate(bucket["sample_ids"]):
            sample_groups = edges_df[edges_df["sample_id"] == sample_id]
            for t_idx, second in enumerate(snapshot_seconds):
                group = sample_groups[sample_groups["second"] == second].sort_values(["src_slot", "dst_slot"])
                matrix = [[0.0 for _ in range(5)] for _ in range(5)]
                for row in group.itertuples(index=False):
                    src = int(row.src_slot)
                    dst = int(row.dst_slot)
                    value = float(row.distance_xy_norm)
                    matrix[src][dst] = value
                    matrix[dst][src] = value
                bucket["edge_distances"][idx][t_idx] = matrix

    split_data: dict[str, SplitData] = {}
    for split_name in ("train", "val", "test"):
        bucket = split_buckets[split_name]
        if not bucket["sample_ids"]:
            continue
        split_data[split_name] = SplitData(
            sample_ids=bucket["sample_ids"],
            match_ids=bucket["match_ids"],
            rounds=bucket["rounds"],
            weak_labels=bucket["weak_labels"],
            weak_label_mask=bucket["weak_label_mask"],
            node_features=torch.tensor(bucket["node_features"], dtype=torch.float32),
            edge_distances=torch.tensor(bucket["edge_distances"], dtype=torch.float32),
            place_targets=torch.tensor(bucket["place_targets"], dtype=torch.float32),
            area_targets=torch.tensor(bucket["area_targets"], dtype=torch.float32),
            distance_targets=torch.tensor(bucket["distance_targets"], dtype=torch.float32),
            context_features=torch.tensor(bucket["context_features"], dtype=torch.float32),
        )

    metadata = {
        "place_vocab": place_vocab,
        "place_dim": place_dim,
        "node_feat_dim": node_feat_dim,
        "context_dim": context_dim or 0,
        "snapshot_seconds": list(snapshot_seconds),
        "splits": {name: len(data.sample_ids) for name, data in split_data.items()},
    }
    return split_data, metadata


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 1),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = math.sqrt(max(q.size(-1), 1))
        logits = torch.matmul(q, k.transpose(-1, -2)) / scale
        logits = logits + self.edge_bias(edge_dist.unsqueeze(-1)).squeeze(-1)
        attention = torch.softmax(F.leaky_relu(logits, negative_slope=0.2), dim=-1)
        out = torch.matmul(attention, v)
        return self.norm(out)


class SnapshotEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_layers):
            self.layers.append(GraphAttentionLayer(current_dim, hidden_dim))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

    def forward(self, node_features: torch.Tensor, edge_distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = node_features
        for layer, ffn, norm in zip(self.layers, self.ffns, self.norms):
            x = layer(x, edge_distances)
            x = norm(x + ffn(x))
        graph_embedding = x.mean(dim=1)
        return x, graph_embedding


class LegacySharedFFNSnapshotEncoder(nn.Module):
    """Two-layer encoder of the earlier pipeline (paper Table III, "Baseline").

    Both attention layers share one feed-forward block and one LayerNorm. Only
    used to reproduce the reported 2-layer baseline; TAGNet uses ``SnapshotEncoder``.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_dim, hidden_dim)
        self.layer2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(node_features, edge_distances)
        x = self.node_norm(x + self.ffn(x))
        x = self.layer2(x, edge_distances)
        x = self.node_norm(x + self.ffn(x))
        graph_embedding = x.mean(dim=1)
        return x, graph_embedding


class NoGraphSnapshotEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        blocks: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            blocks.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            current_dim = hidden_dim
        self.node_mlp = nn.Sequential(*blocks)

    def forward(self, node_features: torch.Tensor, edge_distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_distances
        x = self.node_mlp(node_features)
        graph_embedding = x.mean(dim=1)
        return x, graph_embedding


class MethodCModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        context_dim: int,
        place_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        gat_layers: int = 2,
        gru_layers: int = 1,
        use_context: bool = True,
        use_graph: bool = True,
        use_temporal: bool = True,
        legacy_baseline_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_context = use_context
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.place_dim = place_dim
        if legacy_baseline_encoder:
            if not use_graph or gat_layers != 2:
                raise ValueError("The legacy baseline encoder is defined only for the 2-layer graph model.")
            self.snapshot_encoder = LegacySharedFFNSnapshotEncoder(node_feature_dim, hidden_dim)
        else:
            self.snapshot_encoder = (
                SnapshotEncoder(node_feature_dim, hidden_dim, num_layers=gat_layers)
                if use_graph
                else NoGraphSnapshotEncoder(node_feature_dim, hidden_dim, num_layers=gat_layers)
            )
        self.context_out_dim = hidden_dim // 2 if use_context else 0
        if use_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )
        else:
            self.context_encoder = None
        gru_input_dim = hidden_dim + self.context_out_dim
        if use_temporal:
            self.temporal_encoder = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=embedding_dim,
                batch_first=True,
                num_layers=gru_layers,
                dropout=0.1 if gru_layers > 1 else 0.0,
            )
            self.snapshot_to_embedding = None
        else:
            self.temporal_encoder = None
            self.snapshot_to_embedding = nn.Sequential(
                nn.Linear(gru_input_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        self.place_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, place_dim),
        )
        self.area_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, len(AREAS)),
        )
        self.distance_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_distances: torch.Tensor,
        context_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, time_steps, node_count, feat_dim = node_features.shape
        snapshot_embeddings: list[torch.Tensor] = []
        place_preds: list[torch.Tensor] = []
        area_preds: list[torch.Tensor] = []
        distance_preds: list[torch.Tensor] = []

        for t in range(time_steps):
            _, graph_embedding = self.snapshot_encoder(node_features[:, t], edge_distances[:, t])
            snapshot_embeddings.append(graph_embedding)
            place_preds.append(torch.softmax(self.place_decoder(graph_embedding), dim=-1))
            area_preds.append(torch.softmax(self.area_decoder(graph_embedding), dim=-1))
            distance_preds.append(self.distance_decoder(graph_embedding))

        snapshot_tensor = torch.stack(snapshot_embeddings, dim=1)
        if self.use_context:
            context_tensor = self.context_encoder(context_features)
            temporal_input = torch.cat([snapshot_tensor, context_tensor], dim=-1)
        else:
            temporal_input = snapshot_tensor

        if self.use_temporal:
            _, hidden = self.temporal_encoder(temporal_input)
            round_embedding = hidden[-1]
        else:
            round_embedding = self.snapshot_to_embedding(temporal_input[:, 0, :])

        return {
            "snapshot_embeddings": snapshot_tensor,
            "round_embedding": round_embedding,
            "place_preds": torch.stack(place_preds, dim=1),
            "area_preds": torch.stack(area_preds, dim=1),
            "distance_preds": torch.stack(distance_preds, dim=1),
        }


def batch_indices(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def move_split_to_device(split: SplitData, device: torch.device) -> SplitData:
    return SplitData(
        sample_ids=split.sample_ids,
        match_ids=split.match_ids,
        rounds=split.rounds,
        weak_labels=split.weak_labels,
        weak_label_mask=split.weak_label_mask,
        node_features=split.node_features.to(device),
        edge_distances=split.edge_distances.to(device),
        place_targets=split.place_targets.to(device),
        area_targets=split.area_targets.to(device),
        distance_targets=split.distance_targets.to(device),
        context_features=split.context_features.to(device),
    )


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: SplitData,
    lambda_place: float,
    lambda_area: float,
    lambda_distance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    place_loss = F.mse_loss(outputs["place_preds"], batch.place_targets)
    area_loss = F.mse_loss(outputs["area_preds"], batch.area_targets)
    distance_loss = F.mse_loss(outputs["distance_preds"], batch.distance_targets)
    total = lambda_place * place_loss + lambda_area * area_loss + lambda_distance * distance_loss
    metrics = {
        "loss": float(total.detach().cpu().item()),
        "place_loss": float(place_loss.detach().cpu().item()),
        "area_loss": float(area_loss.detach().cpu().item()),
        "distance_loss": float(distance_loss.detach().cpu().item()),
    }
    return total, metrics


def evaluate(
    model: MethodCModel,
    split: SplitData,
    device: torch.device,
    batch_size: int,
    lambda_place: float,
    lambda_area: float,
    lambda_distance: float,
) -> dict[str, float]:
    if len(split.sample_ids) == 0:
        return {"loss": float("nan")}

    model.eval()
    losses: list[dict[str, float]] = []
    with torch.no_grad():
        for start, end in batch_indices(len(split.sample_ids), batch_size):
            batch = split.subset(list(range(start, end)))
            batch = move_split_to_device(batch, device)
            outputs = model(batch.node_features, batch.edge_distances, batch.context_features)
            _, metrics = compute_losses(outputs, batch, lambda_place, lambda_area, lambda_distance)
            losses.append(metrics)

    result = {}
    for key in losses[0].keys():
        result[key] = sum(item[key] for item in losses) / len(losses)
    return result


def encode_split(
    model: MethodCModel,
    split: SplitData,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start, end in batch_indices(len(split.sample_ids), batch_size):
            batch = split.subset(list(range(start, end)))
            batch = move_split_to_device(batch, device)
            outputs = model(batch.node_features, batch.edge_distances, batch.context_features)
            embeddings.append(outputs["round_embedding"].detach().cpu())
    return torch.cat(embeddings, dim=0)


def kmeans_torch(
    x: torch.Tensor,
    k: int,
    max_iters: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed + k)
    initial_indices = torch.randperm(x.size(0), generator=generator, device=x.device)[:k]
    centroids = x.index_select(0, initial_indices).clone()

    for _ in range(max_iters):
        distances = torch.cdist(x, centroids, p=2)
        assignments = distances.argmin(dim=1)
        new_centroids = []
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                replacement_idx = torch.randint(0, x.size(0), (1,), generator=generator, device=x.device)
                new_centroids.append(x[replacement_idx].squeeze(0))
        new_centroids = torch.stack(new_centroids, dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = torch.cdist(x, centroids, p=2)
    assignments = distances.argmin(dim=1)
    inertia = float(torch.sum((x - centroids.index_select(0, assignments)) ** 2).item())
    return assignments, centroids, inertia


def silhouette_score_torch(x: torch.Tensor, assignments: torch.Tensor) -> float:
    unique_clusters = torch.unique(assignments)
    if unique_clusters.numel() < 2:
        return float("nan")

    n = x.size(0)
    distances = torch.cdist(x, x, p=2)
    silhouettes: list[torch.Tensor] = []
    for idx in range(n):
        same_cluster = assignments == assignments[idx]
        same_cluster[idx] = False
        a = distances[idx][same_cluster].mean() if same_cluster.any() else torch.tensor(0.0, device=x.device)

        b_values = []
        for cluster_id in unique_clusters:
            if int(cluster_id.item()) == int(assignments[idx].item()):
                continue
            other_mask = assignments == cluster_id
            if other_mask.any():
                b_values.append(distances[idx][other_mask].mean())
        if not b_values:
            silhouettes.append(torch.tensor(0.0, device=x.device))
            continue
        b = torch.stack(b_values).min()
        denom = torch.maximum(a, b)
        silhouettes.append((b - a) / torch.clamp(denom, min=1e-9))
    return float(torch.stack(silhouettes).mean().item())


def weak_label_purity(labels: list[str | None], assignments: torch.Tensor) -> float | None:
    filtered = [(label, int(cluster)) for label, cluster in zip(labels, assignments.tolist()) if label is not None]
    if not filtered:
        return None
    cluster_to_counts: dict[int, dict[str, int]] = {}
    for label, cluster in filtered:
        cluster_to_counts.setdefault(cluster, {})
        cluster_to_counts[cluster][label] = cluster_to_counts[cluster].get(label, 0) + 1
    total = len(filtered)
    correct = sum(max(counts.values()) for counts in cluster_to_counts.values())
    return correct / total


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    split_data, metadata = load_split_data(data_dir)
    snapshot_seconds = tuple(int(v) for v in metadata["snapshot_seconds"])

    if args.single_snapshot is not None:
        if args.single_snapshot not in snapshot_seconds:
            raise ValueError(f"single snapshot {args.single_snapshot} is not present in dataset snapshot seconds {snapshot_seconds}")
        split_data = {
            split_name: slice_split_time(split, args.single_snapshot, snapshot_seconds)
            for split_name, split in split_data.items()
        }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_data = split_data["train"]
    val_data = split_data["val"]
    test_data = split_data["test"]

    model = MethodCModel(
        node_feature_dim=metadata["node_feat_dim"],
        context_dim=metadata["context_dim"],
        place_dim=metadata["place_dim"],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        gat_layers=args.gat_layers,
        gru_layers=args.gru_layers,
        use_context=not args.no_context,
        use_graph=not args.no_graph,
        use_temporal=args.single_snapshot is None,
        legacy_baseline_encoder=getattr(args, "legacy_baseline_encoder", False),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[dict[str, float]] = []
        for start, end in batch_indices(len(train_data.sample_ids), args.batch_size):
            batch = train_data.subset(list(range(start, end)))
            batch = move_split_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.node_features, batch.edge_distances, batch.context_features)
            total_loss, metrics = compute_losses(
                outputs,
                batch,
                lambda_place=args.lambda_place,
                lambda_area=args.lambda_area,
                lambda_distance=args.lambda_distance,
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(metrics)

        train_metrics = {key: sum(item[key] for item in epoch_losses) / len(epoch_losses) for key in epoch_losses[0].keys()}
        val_metrics = evaluate(
            model=model,
            split=val_data,
            device=device,
            batch_size=args.batch_size,
            lambda_place=args.lambda_place,
            lambda_area=args.lambda_area,
            lambda_distance=args.lambda_distance,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = evaluate(model, train_data, device, args.batch_size, args.lambda_place, args.lambda_area, args.lambda_distance)
    val_eval = evaluate(model, val_data, device, args.batch_size, args.lambda_place, args.lambda_area, args.lambda_distance)
    test_eval = evaluate(model, test_data, device, args.batch_size, args.lambda_place, args.lambda_area, args.lambda_distance)

    embeddings_by_split = {
        split_name: encode_split(model, split, device, args.batch_size)
        for split_name, split in split_data.items()
    }
    all_embeddings = torch.cat([embeddings_by_split["train"], embeddings_by_split["val"], embeddings_by_split["test"]], dim=0)
    all_sample_ids = split_data["train"].sample_ids + split_data["val"].sample_ids + split_data["test"].sample_ids
    all_match_ids = split_data["train"].match_ids + split_data["val"].match_ids + split_data["test"].match_ids
    all_rounds = split_data["train"].rounds + split_data["val"].rounds + split_data["test"].rounds
    all_splits = ["train"] * len(split_data["train"].sample_ids) + ["val"] * len(split_data["val"].sample_ids) + ["test"] * len(split_data["test"].sample_ids)
    all_weak_labels = split_data["train"].weak_labels + split_data["val"].weak_labels + split_data["test"].weak_labels

    k_metrics: list[dict[str, Any]] = []
    assignments_by_k: dict[int, torch.Tensor] = {}
    best_silhouette = -1e9
    chosen_k = None
    chosen_assignments = None
    for k in args.k_list:
        assignments, _, inertia = kmeans_torch(all_embeddings, k, max_iters=args.kmeans_iters, seed=args.seed)
        silhouette = silhouette_score_torch(all_embeddings, assignments)
        purity = weak_label_purity(all_weak_labels, assignments)
        cluster_sizes = torch.bincount(assignments, minlength=k).tolist()
        k_metrics.append(
            {
                "k": int(k),
                "inertia": float(inertia),
                "silhouette": float(silhouette),
                "weak_label_purity_at_15s": None if purity is None else float(purity),
                "cluster_sizes": cluster_sizes,
            }
        )
        assignments_by_k[int(k)] = assignments.detach().cpu()
        if math.isfinite(silhouette) and silhouette > best_silhouette:
            best_silhouette = silhouette
            chosen_k = int(k)
            chosen_assignments = assignments.detach().cpu()

    assert chosen_k is not None and chosen_assignments is not None

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "args": vars(args),
        },
        run_dir / "method_c_model.pt",
    )

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    embedding_rows = []
    for idx, sample_id in enumerate(all_sample_ids):
        row = {
            "sample_id": sample_id,
            "match_id": all_match_ids[idx],
            "round_num": all_rounds[idx],
            "split": all_splits[idx],
            "weak_label_15s": all_weak_labels[idx],
            "cluster_best_k": int(chosen_assignments[idx].item()),
        }
        for dim_idx, value in enumerate(all_embeddings[idx].tolist()):
            row[f"emb_{dim_idx:02d}"] = float(value)
        embedding_rows.append(row)
    pd.DataFrame(embedding_rows).to_parquet(run_dir / "round_embeddings.parquet", index=False)

    cluster_rows = []
    for k, assignments in assignments_by_k.items():
        for sample_id, match_id, round_num, split_name, weak_label, cluster_id in zip(
            all_sample_ids, all_match_ids, all_rounds, all_splits, all_weak_labels, assignments.tolist()
        ):
            cluster_rows.append(
                {
                    "k": int(k),
                    "sample_id": sample_id,
                    "match_id": match_id,
                    "round_num": round_num,
                    "split": split_name,
                    "weak_label_15s": weak_label,
                    "cluster_id": int(cluster_id),
                }
            )
    pd.DataFrame(cluster_rows).to_parquet(run_dir / "cluster_assignments.parquet", index=False)

    with (run_dir / "k_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(k_metrics, fh, indent=2, ensure_ascii=False)

    summary = {
        "device": str(device),
        "splits": metadata["splits"],
        "node_feat_dim": metadata["node_feat_dim"],
        "context_dim": metadata["context_dim"],
        "place_dim": metadata["place_dim"],
        "snapshot_seconds": list(snapshot_seconds),
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "gat_layers": args.gat_layers,
        "gru_layers": args.gru_layers,
        "use_graph": not args.no_graph,
        "use_context": not args.no_context,
        "single_snapshot": args.single_snapshot,
        "use_temporal": args.single_snapshot is None,
        "legacy_baseline_encoder": getattr(args, "legacy_baseline_encoder", False),
        "best_k_by_silhouette": int(chosen_k),
        "train_eval": train_eval,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return {"summary": summary, "k_metrics": k_metrics, "run_dir": str(run_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAGNet (graph attention + GRU + KMeans) on prepared Dust2 CT data.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-place", type=float, default=1.0)
    parser.add_argument("--lambda-area", type=float, default=0.5)
    parser.add_argument("--lambda-distance", type=float, default=1.0)
    parser.add_argument("--k-list", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--single-snapshot", type=int, default=None)
    parser.add_argument(
        "--legacy-baseline-encoder",
        action="store_true",
        help="Use the shared-FFN 2-layer encoder of the paper's Table III baseline (requires --gat-layers 2).",
    )
    args = parser.parse_args()
    run_name = args.run_name or f"tagnet_{timestamp_name()}"
    args.run_dir = str(Path(args.run_root) / run_name)
    return args


def main() -> None:
    args = parse_args()
    result = run_training(args)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(json.dumps(result["k_metrics"], indent=2, ensure_ascii=False))
    print(result["run_dir"])


if __name__ == "__main__":
    main()
