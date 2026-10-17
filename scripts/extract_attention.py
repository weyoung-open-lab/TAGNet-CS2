from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "tagnet_main" / "run"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "results" / "tagnet_main" / "run_analysis"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def finalize(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def compute_attention(layer: torch.nn.Module, x: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
    q = layer.query(x)
    k = layer.key(x)
    scale = np.sqrt(max(q.size(-1), 1))
    logits = torch.matmul(q, k.transpose(-1, -2)) / scale
    logits = logits + layer.edge_bias(edge_dist.unsqueeze(-1)).squeeze(-1)
    return torch.softmax(torch.nn.functional.leaky_relu(logits, negative_slope=0.2), dim=-1)


def forward_snapshot_with_attentions(snapshot_encoder: torch.nn.Module, x: torch.Tensor, edge_dist: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    attentions: list[tuple[str, np.ndarray]] = []
    current = x
    if hasattr(snapshot_encoder, "layers"):
        layers = snapshot_encoder.layers
        ffns = snapshot_encoder.ffns
        norms = snapshot_encoder.norms
        for idx, (layer, ffn, norm) in enumerate(zip(layers, ffns, norms), start=1):
            attn = compute_attention(layer, current, edge_dist).squeeze(0).cpu().numpy()
            attentions.append((f"layer{idx}", attn))
            current = layer(current, edge_dist)
            current = norm(current + ffn(current))
    else:
        raise ValueError("Snapshot encoder does not expose stacked attention layers.")
    return attentions


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TAGNet attention matrices from a saved checkpoint.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_tagnet as train_script

    apply_style()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    analysis_dir = Path(args.analysis_dir)
    figure_dir = analysis_dir / "figures"

    split_data, metadata = train_script.load_split_data(data_dir)
    snapshot_seconds = tuple(int(v) for v in metadata.get("snapshot_seconds", [10, 15, 20]))
    assignments_df = pd.read_parquet(analysis_dir / "analysis_cluster_assignments_bestk.parquet")
    sample_to_cluster = dict(zip(assignments_df["sample_id"].tolist(), assignments_df["cluster_id"].tolist()))

    ckpt = torch.load(run_dir / "method_c_model.pt", map_location="cpu")
    model_args = ckpt["args"]
    model = train_script.MethodCModel(
        node_feature_dim=int(metadata["node_feat_dim"]),
        context_dim=int(metadata["context_dim"]),
        place_dim=int(metadata["place_dim"]),
        hidden_dim=int(model_args["hidden_dim"]),
        embedding_dim=int(model_args["embedding_dim"]),
        gat_layers=int(model_args.get("gat_layers", 2)),
        gru_layers=int(model_args.get("gru_layers", 1)),
        use_context=not bool(model_args.get("no_context", False)),
        use_graph=True,
        use_temporal=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    second_to_index = {second: idx for idx, second in enumerate(snapshot_seconds) if second in {10, 15, 20}}
    if 15 not in second_to_index:
        raise ValueError(f"Dataset snapshot seconds {snapshot_seconds} do not contain the required 15-second view.")

    with torch.no_grad():
        for split_name, split in split_data.items():
            node_features = split.node_features
            edge_distances = split.edge_distances
            for local_idx, sample_id in enumerate(split.sample_ids):
                cluster_id = sample_to_cluster.get(sample_id)
                if cluster_id is None:
                    continue
                sample_nodes = node_features[local_idx : local_idx + 1]
                sample_edges = edge_distances[local_idx : local_idx + 1]
                for second, t_idx in sorted(second_to_index.items()):
                    x0 = sample_nodes[:, t_idx]
                    e0 = sample_edges[:, t_idx]
                    for layer_name, attn in forward_snapshot_with_attentions(model.snapshot_encoder, x0, e0):
                        for src in range(attn.shape[0]):
                            for dst in range(attn.shape[1]):
                                rows.append(
                                    {
                                        "sample_id": sample_id,
                                        "split": split_name,
                                        "cluster_id": int(cluster_id),
                                        "second": int(second),
                                        "layer": layer_name,
                                        "src_slot": int(src),
                                        "dst_slot": int(dst),
                                        "attention": float(attn[src, dst]),
                                    }
                                )

    attention_df = pd.DataFrame(rows)
    attention_df.to_csv(analysis_dir / "attention_matrices.csv", index=False, encoding="utf-8-sig")

    summary_df = (
        attention_df.groupby(["cluster_id", "second", "layer", "src_slot", "dst_slot"])["attention"]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(analysis_dir / "attention_matrices_mean.csv", index=False, encoding="utf-8-sig")

    available_layers = sorted(summary_df["layer"].unique().tolist())
    target_layer = available_layers[-1]
    plot_df = summary_df[(summary_df["layer"] == target_layer) & (summary_df["second"] == 15)].copy()
    clusters = sorted(plot_df["cluster_id"].unique().tolist())
    ncols = 3
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, 3.7 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    im = None
    for ax, cluster_id in zip(axes, clusters):
        matrix = (
            plot_df[plot_df["cluster_id"] == cluster_id]
            .pivot(index="src_slot", columns="dst_slot", values="attention")
            .reindex(index=range(5), columns=range(5), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(plot_df["attention"].max(), 0.4))
        ax.set_title(f"C{cluster_id} {target_layer} @15s")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xlabel("Attend to slot")
        ax.set_ylabel("Source slot")
        for i in range(5):
            for j in range(5):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.24 else "black")
    for ax in axes[len(clusters):]:
        ax.set_axis_off()
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Mean attention")
    fig.suptitle("TAGNet Attention Heatmaps", y=1.01, fontsize=16)
    finalize(fig, figure_dir / f"fig_attention_heatmaps_{target_layer}_15s")

    print(str(analysis_dir / "attention_matrices_mean.csv"))
    print(str(figure_dir / f"fig_attention_heatmaps_{target_layer}_15s.png"))


if __name__ == "__main__":
    main()
