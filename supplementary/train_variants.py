"""Train the supplementary-experiment variants of TAGNet (paper Section IV-K).

Each variant reuses the data loader, model and training loop of
``scripts/train_tagnet.py`` and swaps only the component under test: temporal
pooling (GRU / mean), temporal cell (GRU / LSTM), node position encoding
(place one-hot / continuous XY), edge distance column (Euclidean / navmesh
geodesic) and an optional hinge distance-consistency term.

As in the main model, every reconstruction decoder reads the per-snapshot graph
embedding directly; the temporal module only produces the round embedding that
is clustered with K-Means.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import torch
import torch.nn as nn


DEFAULT_TRAINER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tagnet.py"
OPTIONS: dict[str, str] = {"pool_mode": "gru", "temporal_cell": "gru"}


def load_trainer_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tagnet_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_model_class(trainer: ModuleType) -> type[nn.Module]:
    class VariantModel(trainer.MethodCModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pool_mode = "single" if not self.use_temporal else OPTIONS["pool_mode"]
            self.temporal_cell = OPTIONS["temporal_cell"]
            # Graph and no-graph encoders both output hidden_dim.
            temporal_input_dim = self.place_decoder[0].in_features
            temporal_input_dim += self.context_out_dim
            embedding_dim = self.temporal_encoder.hidden_size if self.temporal_encoder is not None else self.snapshot_to_embedding[-1].out_features

            if self.pool_mode == "mean":
                self.temporal_encoder = None
                self.snapshot_to_embedding = nn.Sequential(
                    nn.Linear(temporal_input_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
            elif self.pool_mode == "gru" and self.temporal_cell == "lstm":
                layers = self.temporal_encoder.num_layers
                self.temporal_encoder = nn.LSTM(
                    input_size=temporal_input_dim,
                    hidden_size=embedding_dim,
                    batch_first=True,
                    num_layers=layers,
                    dropout=0.1 if layers > 1 else 0.0,
                )

        def forward(self, node_features, edge_distances, context_features):
            _, time_steps, _, _ = node_features.shape
            snapshot_embeddings, place_preds, area_preds, distance_preds = [], [], [], []
            for time_index in range(time_steps):
                _, graph_embedding = self.snapshot_encoder(node_features[:, time_index], edge_distances[:, time_index])
                snapshot_embeddings.append(graph_embedding)
                # Decoders read the snapshot embedding directly, as in train_tagnet.py.
                place_preds.append(torch.softmax(self.place_decoder(graph_embedding), dim=-1))
                area_preds.append(torch.softmax(self.area_decoder(graph_embedding), dim=-1))
                distance_preds.append(self.distance_decoder(graph_embedding))

            snapshot_tensor = torch.stack(snapshot_embeddings, dim=1)
            if self.use_context:
                context_tensor = self.context_encoder(context_features)
                temporal_input = torch.cat([snapshot_tensor, context_tensor], dim=-1)
            else:
                temporal_input = snapshot_tensor

            if self.pool_mode == "gru":
                _, hidden = self.temporal_encoder(temporal_input)
                round_embedding = hidden[0][-1] if self.temporal_cell == "lstm" else hidden[-1]
            elif self.pool_mode == "mean":
                round_embedding = self.snapshot_to_embedding(temporal_input.mean(dim=1))
            else:
                round_embedding = self.snapshot_to_embedding(temporal_input[:, 0, :])
            return {
                "snapshot_embeddings": snapshot_tensor,
                "round_embedding": round_embedding,
                "place_preds": torch.stack(place_preds, dim=1),
                "area_preds": torch.stack(area_preds, dim=1),
                "distance_preds": torch.stack(distance_preds, dim=1),
            }

    return VariantModel


def replace_coordinates(trainer: ModuleType, split_data, metadata: dict, data_dir: Path, mode: str) -> None:
    if mode == "place":
        return
    nodes = pd.read_parquet(data_dir / "ct_snapshot_nodes.parquet")
    train_nodes = nodes[nodes["split"] == "train"]
    x_mean, x_std = float(train_nodes["X"].mean()), max(float(train_nodes["X"].std(ddof=0)), 1e-9)
    y_mean, y_std = float(train_nodes["Y"].mean()), max(float(train_nodes["Y"].std(ddof=0)), 1e-9)
    coordinates = {}
    for sample_id, group in nodes.groupby("sample_id", sort=False):
        frames = []
        for _, frame in group.groupby("second", sort=True):
            frame = frame.sort_values("player_slot")
            frames.append([[(float(row.X) - x_mean) / x_std, (float(row.Y) - y_mean) / y_std] for row in frame.itertuples(index=False)])
        coordinates[str(sample_id)] = frames
    for split in split_data.values():
        coord_tensor = torch.tensor([coordinates[sample_id] for sample_id in split.sample_ids], dtype=torch.float32)
        state_tensor = split.node_features[..., -2:]
        if mode == "xy":
            split.node_features = torch.cat([state_tensor, coord_tensor], dim=-1)
        elif mode == "place_xy":
            split.node_features = torch.cat([split.node_features, coord_tensor], dim=-1)
        else:
            raise ValueError(f"Unsupported node-position mode: {mode}")
    metadata["node_feat_dim"] = int(next(iter(split_data.values())).node_features.shape[-1])
    metadata["node_position_mode"] = mode


def replace_edge_distances(split_data, data_dir: Path, column: str) -> None:
    if column == "distance_xy_norm":
        return
    edges = pd.read_parquet(data_dir / "ct_snapshot_edges.parquet")
    if column not in edges.columns:
        raise ValueError(f"Edge distance column not found: {column}")
    matrices = {}
    for sample_id, sample_edges in edges.groupby("sample_id", sort=False):
        sequence = []
        for _, frame in sample_edges.groupby("second", sort=True):
            matrix = [[0.0] * 5 for _ in range(5)]
            for row in frame.itertuples(index=False):
                value = float(getattr(row, column))
                matrix[int(row.src_slot)][int(row.dst_slot)] = value
                matrix[int(row.dst_slot)][int(row.src_slot)] = value
            sequence.append(matrix)
        matrices[str(sample_id)] = sequence
    for split in split_data.values():
        split.edge_distances = torch.tensor([matrices[sample_id] for sample_id in split.sample_ids], dtype=torch.float32)


def add_hinge_bounds(split_data, data_dir: Path) -> None:
    """Attach train-split distance intervals per place pair for the optional hinge term."""
    edges = pd.read_parquet(data_dir / "ct_snapshot_edges.parquet")
    required = {"src_nav_place", "dst_nav_place", "distance_xy_norm"}
    if not required.issubset(edges.columns):
        raise ValueError("Hinge consistency requires prepared navigation-place columns.")
    train = edges[edges["split"] == "train"]
    interval = {
        (str(src), str(dst)): (float(values.min()), float(values.max()))
        for (src, dst), values in train.groupby(["src_nav_place", "dst_nav_place"], sort=False)["distance_xy_norm"]
    }
    all_bounds, all_masks = {}, {}
    for sample_id, sample_edges in edges.groupby("sample_id", sort=False):
        sequence_bounds, sequence_masks = [], []
        for _, frame in sample_edges.groupby("second", sort=True):
            bounds, masks = [], []
            for row in frame.sort_values(["src_slot", "dst_slot"]).itertuples(index=False):
                key = (str(row.src_nav_place), str(row.dst_nav_place))
                bounds.append(interval.get(key, (0.0, 0.0)))
                masks.append(float(key in interval))
            sequence_bounds.append(bounds)
            sequence_masks.append(masks)
        all_bounds[str(sample_id)] = sequence_bounds
        all_masks[str(sample_id)] = sequence_masks
    for split in split_data.values():
        split.distance_bounds = torch.tensor([all_bounds[sample_id] for sample_id in split.sample_ids], dtype=torch.float32)
        split.distance_bound_mask = torch.tensor([all_masks[sample_id] for sample_id in split.sample_ids], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-training-graph TAGNet supplementary variants.")
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--gat-layers", type=int, default=3)
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
    parser.add_argument("--pool-mode", choices=["gru", "mean"], default="gru")
    parser.add_argument("--temporal-cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--node-position-mode", choices=["place", "place_xy", "xy"], default="place")
    parser.add_argument("--edge-distance-column", default="distance_xy_norm")
    parser.add_argument("--lambda-hinge", type=float, default=0.0)
    args = parser.parse_args()
    args.run_dir = str(args.run_root / args.run_name)
    return args


def main() -> None:
    args = parse_args()
    if not args.trainer_script.exists():
        raise FileNotFoundError(args.trainer_script)
    if args.pool_mode == "mean" and args.single_snapshot is not None:
        raise ValueError("Mean pooling requires the full multi-snapshot input.")
    OPTIONS.update(pool_mode=args.pool_mode, temporal_cell=args.temporal_cell)
    trainer = load_trainer_module(args.trainer_script)
    original_loader = trainer.load_split_data

    def patched_loader(data_dir: Path):
        split_data, metadata = original_loader(data_dir)
        replace_coordinates(trainer, split_data, metadata, data_dir, args.node_position_mode)
        replace_edge_distances(split_data, data_dir, args.edge_distance_column)
        if args.lambda_hinge > 0:
            add_hinge_bounds(split_data, data_dir)
        return split_data, metadata

    trainer.load_split_data = patched_loader
    trainer.MethodCModel = make_model_class(trainer)
    if args.lambda_hinge > 0:
        original_subset = trainer.SplitData.subset
        original_move = trainer.move_split_to_device
        original_losses = trainer.compute_losses

        def subset_with_hinge(self, indices):
            result = original_subset(self, indices)
            result.distance_bounds = self.distance_bounds.index_select(0, torch.tensor(indices, dtype=torch.long))
            result.distance_bound_mask = self.distance_bound_mask.index_select(0, torch.tensor(indices, dtype=torch.long))
            return result

        def move_with_hinge(split, device):
            result = original_move(split, device)
            result.distance_bounds = split.distance_bounds.to(device)
            result.distance_bound_mask = split.distance_bound_mask.to(device)
            return result

        def losses_with_hinge(outputs, batch, lambda_place, lambda_area, lambda_distance):
            total, metrics = original_losses(outputs, batch, lambda_place, lambda_area, lambda_distance)
            prediction = outputs["distance_preds"]
            lower, upper = batch.distance_bounds[..., 0], batch.distance_bounds[..., 1]
            mask = batch.distance_bound_mask
            violation = torch.relu(lower - prediction) + torch.relu(prediction - upper)
            hinge = (violation * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            total = total + args.lambda_hinge * hinge
            metrics["hinge_loss"] = float(hinge.detach().item())
            metrics["loss"] = float(total.detach().item())
            return total, metrics

        trainer.SplitData.subset = subset_with_hinge
        trainer.move_split_to_device = move_with_hinge
        trainer.compute_losses = losses_with_hinge
    result = trainer.run_training(args)
    summary_path = Path(args.run_dir) / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["protocol"] = "original_direct_snapshot_decoder"
    summary["supplement_variant"] = {
        "pool_mode": args.pool_mode,
        "temporal_cell": args.temporal_cell,
        "node_position_mode": args.node_position_mode,
        "edge_distance_column": args.edge_distance_column,
        "lambda_hinge": args.lambda_hinge,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
