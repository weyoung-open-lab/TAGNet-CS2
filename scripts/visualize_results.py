from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "results" / "tagnet_main" / "run_analysis"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    "#D98F5C",
    "#7A7A7A",
    "#94B447",
    "#7048A3",
]
AREAS = ["A", "B", "Mid", "CT"]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 14
    axes_linewidth: float = 1.8
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def finalize_figure(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_k_selection(k_df: pd.DataFrame, training_best_k: int, analysis_best_k: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("silhouette", "Silhouette", False),
        ("davies_bouldin", "Davies-Bouldin", True),
        ("calinski_harabasz", "Calinski-Harabasz", False),
    ]
    x = k_df["k"].to_numpy()
    for ax, (col, title, lower_is_better) in zip(axes, metrics):
        y = k_df[col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=PALETTE["blue_main"], linewidth=2.2)
        ax.axvline(training_best_k, color=PALETTE["red_strong"], linestyle="--", linewidth=1.8, label="Train-best K")
        ax.axvline(analysis_best_k, color=PALETTE["teal"], linestyle=":", linewidth=2.2, label="Multi-metric K")
        best_idx = int(np.argmin(y) if lower_is_better else np.argmax(y))
        ax.scatter([x[best_idx]], [y[best_idx]], s=70, color=PALETTE["highlight"], edgecolors="black", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="best", fontsize=11)
    fig.suptitle("TAGNet K Selection", y=1.02, fontsize=16)
    finalize_figure(fig, out_dir / "fig_k_selection")


def plot_tsne(tsne_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    clusters = sorted(tsne_df["cluster_id"].unique().tolist())
    for idx, cluster_id in enumerate(clusters):
        group = tsne_df[tsne_df["cluster_id"] == cluster_id]
        ax.scatter(
            group["tsne_x"],
            group["tsne_y"],
            s=28,
            alpha=0.78,
            color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
            label=f"C{cluster_id}",
            edgecolors="white",
            linewidths=0.3,
        )
        ax.text(group["tsne_x"].mean(), group["tsne_y"].mean(), f"C{cluster_id}", fontsize=11, weight="bold")
    ax.set_title("t-SNE of Round Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(ncol=3, fontsize=10, loc="best")
    finalize_figure(fig, out_dir / "fig_tsne_clusters")


def plot_tsne_teamshape(tsne_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    clusters = sorted(tsne_df["cluster_id"].dropna().unique().tolist())
    cluster_color_map = {cluster_id: DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx, cluster_id in enumerate(clusters)}
    marker_pool = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    resolved_df = tsne_df[tsne_df["ct_team_name"].notna()].copy()
    top_teams = (
        resolved_df["ct_team_name"].value_counts().head(min(len(marker_pool), 6)).index.tolist()
        if not resolved_df.empty
        else []
    )
    marker_map = {team_name: marker_pool[idx] for idx, team_name in enumerate(top_teams)}
    default_marker = "o"

    for team_name, team_group in tsne_df.groupby(tsne_df["ct_team_name"].fillna("unknown_team")):
        marker = marker_map.get(team_name, default_marker)
        for cluster_id in clusters:
            group = team_group[team_group["cluster_id"] == cluster_id]
            if group.empty:
                continue
            ax.scatter(
                group["tsne_x"],
                group["tsne_y"],
                s=28,
                alpha=0.75,
                color=cluster_color_map[cluster_id],
                marker=marker,
                edgecolors="white",
                linewidths=0.25,
            )

    ax.set_title("t-SNE of Round Embeddings (Color=Cluster, Shape=Team)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    cluster_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=cluster_color_map[cluster_id], markeredgecolor="none", label=f"C{cluster_id}")
        for cluster_id in clusters
    ]
    team_handles = [
        Line2D([0], [0], marker=marker_map[team_name], linestyle="", markersize=8, color="black", markerfacecolor="white", label=team_name)
        for team_name in top_teams
    ]
    if team_handles:
        team_handles.append(
            Line2D([0], [0], marker=default_marker, linestyle="", markersize=8, color="black", markerfacecolor="white", label="other teams")
        )
    legend1 = ax.legend(handles=cluster_handles, title="Cluster", loc="upper right", fontsize=9)
    ax.add_artist(legend1)
    if team_handles:
        ax.legend(handles=team_handles, title="Team shape", loc="lower left", fontsize=8)
    finalize_figure(fig, out_dir / "fig_tsne_clusters_teamshape")


def plot_team_style_heatmap(team_fp: pd.DataFrame, out_dir: Path) -> None:
    fp = team_fp[team_fp["round_count"] >= 10].copy()
    cluster_cols = [col for col in fp.columns if col.startswith("cluster_")]
    fp = fp.sort_values("round_count", ascending=False).head(12).copy()
    fp["dominant_cluster"] = fp[cluster_cols].to_numpy().argmax(axis=1)
    fp = fp.sort_values(["dominant_cluster", "round_count"], ascending=[True, False])
    matrix = fp[cluster_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(matrix.max(), 0.25))
    ax.set_title("Team Style Fingerprints")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("CT Team")
    ax.set_xticks(range(len(cluster_cols)))
    ax.set_xticklabels([col.replace("cluster_", "C") for col in cluster_cols])
    ax.set_yticks(range(len(fp)))
    ax.set_yticklabels([f"{name} ({rc})" for name, rc in zip(fp["ct_team_name"], fp["round_count"])])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Usage ratio")
    finalize_figure(fig, out_dir / "fig_team_style_heatmap")


def plot_context_trigger_heatmaps(analysis_dir: Path, out_dir: Path) -> None:
    files = [
        ("score_state", "Score State"),
        ("ct_econ_type", "CT Economy"),
        ("prev_round_attack_site", "Prev Attack Site"),
        ("prev_round_winner_side", "Prev Winner"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (stem, title) in zip(axes, files):
        csv_path = analysis_dir / f"context_distribution_{stem}.csv"
        table = pd.read_csv(csv_path, index_col=0)
        matrix = table.to_numpy(dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
        im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels([f"C{col}" for col in table.columns], rotation=45, ha="right")
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index.tolist())
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.45 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Row-normalized ratio")
    fig.suptitle("Context Trigger Distributions", y=1.01, fontsize=16)
    finalize_figure(fig, out_dir / "fig_context_trigger_heatmaps")


def plot_cluster_area_over_time(nodes_with_cluster: pd.DataFrame, out_dir: Path) -> None:
    area_df = (
        nodes_with_cluster.groupby(["cluster_id", "second", "area"]).size().reset_index(name="count")
    )
    total_df = area_df.groupby(["cluster_id", "second"])["count"].transform("sum")
    area_df["ratio"] = area_df["count"] / total_df
    clusters = sorted(nodes_with_cluster["cluster_id"].dropna().unique().tolist())
    seconds = [second for second in [10, 15, 20] if second in sorted(nodes_with_cluster["second"].dropna().unique().tolist())]
    if not seconds:
        seconds = sorted(nodes_with_cluster["second"].dropna().unique().tolist())[:3]

    fig, axes = plt.subplots(1, len(seconds), figsize=(4.8 * len(seconds), 7), sharey=True)
    if len(seconds) == 1:
        axes = [axes]
    for ax, second in zip(axes, seconds):
        matrix_rows = []
        for cluster_id in clusters:
            row = []
            group = area_df[(area_df["cluster_id"] == cluster_id) & (area_df["second"] == second)]
            for area in AREAS:
                value = group.loc[group["area"] == area, "ratio"]
                row.append(float(value.iloc[0]) if not value.empty else 0.0)
            matrix_rows.append(row)
        matrix = np.array(matrix_rows, dtype=float)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(matrix.max(), 0.7))
        ax.set_title(f"{second}s Area Mix")
        ax.set_xticks(range(len(AREAS)))
        ax.set_xticklabels(AREAS)
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([f"C{cluster_id}" for cluster_id in clusters])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black" if value < 0.55 else "white")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Area ratio")
    fig.suptitle("Cluster Area Composition Over Time", y=1.01, fontsize=16)
    finalize_figure(fig, out_dir / "fig_cluster_area_over_time")


def plot_cluster_position_heatmaps(nodes_with_cluster: pd.DataFrame, out_dir: Path) -> None:
    df = nodes_with_cluster[nodes_with_cluster["second"] == 15].copy()
    clusters = sorted(df["cluster_id"].dropna().unique().tolist())
    n_clusters = len(clusters)
    ncols = 3
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.1 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    xmin, xmax = df["X"].min(), df["X"].max()
    ymin, ymax = df["Y"].min(), df["Y"].max()

    for ax, cluster_id in zip(axes, clusters):
        group = df[df["cluster_id"] == cluster_id]
        hist = ax.hist2d(
            group["X"],
            group["Y"],
            bins=28,
            range=[[xmin, xmax], [ymin, ymax]],
            cmap="magma",
            cmin=1,
        )
        top_areas = group["area"].value_counts(normalize=True).head(2)
        top_area_text = ", ".join([f"{idx}:{value:.2f}" for idx, value in top_areas.items()])
        ax.set_title(f"C{cluster_id} @15s\n{top_area_text}", fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    for ax in axes[n_clusters:]:
        ax.set_axis_off()
    cbar = fig.colorbar(hist[3], ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Player density")
    fig.suptitle("Cluster Position Heatmaps at 15 Seconds", y=1.01, fontsize=16)
    finalize_figure(fig, out_dir / "fig_cluster_position_heatmaps_15s")


def plot_score_shift(score_shift_df: pd.DataFrame, out_dir: Path) -> None:
    if score_shift_df.empty:
        return
    top = score_shift_df.sort_values("leading_vs_trailing_cosine_distance", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    values = top["leading_vs_trailing_cosine_distance"].to_numpy(dtype=float)
    labels = top["ct_team_name"].tolist()
    y = np.arange(len(top))
    ax.barh(y, values, color=PALETTE["teal"], edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine distance")
    ax.set_title("Leading vs Trailing Style Shift")
    ax.grid(axis="x", alpha=0.18)
    finalize_figure(fig, out_dir / "fig_score_state_shift")


def plot_attention_overlay(analysis_dir: Path, nodes_with_cluster: pd.DataFrame, out_dir: Path) -> None:
    attention_path = analysis_dir / "attention_matrices_mean.csv"
    if not attention_path.exists():
        return
    attention_df = pd.read_csv(attention_path)
    if attention_df.empty:
        return
    available_layers = sorted(attention_df["layer"].unique().tolist())
    target_layer = available_layers[-1]
    attn_15 = attention_df[(attention_df["layer"] == target_layer) & (attention_df["second"] == 15)].copy()
    pos_15 = (
        nodes_with_cluster[nodes_with_cluster["second"] == 15]
        .groupby(["cluster_id", "player_slot"])[["X", "Y"]]
        .mean()
        .reset_index()
    )
    clusters = sorted(attn_15["cluster_id"].dropna().unique().tolist())
    if not clusters:
        return

    ncols = 3
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    xmin, xmax = pos_15["X"].min(), pos_15["X"].max()
    ymin, ymax = pos_15["Y"].min(), pos_15["Y"].max()

    for ax, cluster_id in zip(axes, clusters):
        cluster_pos = pos_15[pos_15["cluster_id"] == cluster_id].copy()
        cluster_attn = attn_15[attn_15["cluster_id"] == cluster_id].copy()
        matrix = (
            cluster_attn.pivot(index="src_slot", columns="dst_slot", values="attention")
            .reindex(index=range(5), columns=range(5), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        pair_weights = []
        for src in range(5):
            for dst in range(src + 1, 5):
                pair_weights.append(0.5 * (matrix[src, dst] + matrix[dst, src]))
        threshold = float(np.quantile(pair_weights, 0.75)) if pair_weights else 0.0

        ax.scatter(cluster_pos["X"], cluster_pos["Y"], s=90, c=PALETTE["highlight"], edgecolors="black", zorder=3)
        for row in cluster_pos.itertuples(index=False):
            ax.text(row.X, row.Y, f"S{int(row.player_slot)}", fontsize=8, ha="center", va="center", zorder=4)

        for src in range(5):
            for dst in range(src + 1, 5):
                weight = 0.5 * (matrix[src, dst] + matrix[dst, src])
                if weight < threshold:
                    continue
                src_row = cluster_pos[cluster_pos["player_slot"] == src]
                dst_row = cluster_pos[cluster_pos["player_slot"] == dst]
                if src_row.empty or dst_row.empty:
                    continue
                x1, y1 = float(src_row["X"].iloc[0]), float(src_row["Y"].iloc[0])
                x2, y2 = float(dst_row["X"].iloc[0]), float(dst_row["Y"].iloc[0])
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=PALETTE["red_strong"],
                    alpha=min(max(weight / 0.35, 0.15), 0.9),
                    linewidth=1.5 + 5.0 * weight,
                    zorder=2,
                )

        ax.set_title(f"C{cluster_id} attention overlay @15s", fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(xmin - 150, xmax + 150)
        ax.set_ylim(ymin - 150, ymax + 150)
        ax.grid(alpha=0.12)

    for ax in axes[len(clusters):]:
        ax.set_axis_off()
    fig.suptitle("TAGNet Attention Overlay on Map Coordinates", y=1.01, fontsize=16)
    finalize_figure(fig, out_dir / "fig_attention_overlay_15s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-style figures for TAGNet results.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_style(FigureStyle(font_size=13, axes_linewidth=1.7))

    k_df = pd.read_csv(analysis_dir / "k_selection_metrics.csv")
    tsne_df = pd.read_parquet(analysis_dir / "tsne_coordinates.parquet")
    team_fp = pd.read_csv(analysis_dir / "team_style_fingerprints.csv")
    score_shift_df = pd.read_csv(analysis_dir / "score_state_style_shift.csv")
    nodes_df = pd.read_parquet(data_dir / "ct_snapshot_nodes.parquet")
    assignments_df = pd.read_parquet(analysis_dir / "analysis_cluster_assignments_bestk.parquet")
    with (analysis_dir / "analysis_summary.json").open("r", encoding="utf-8") as fh:
        analysis_summary = json.load(fh)

    training_best_k = int(analysis_summary["reference_best_k_from_training"])
    analysis_best_k = int(analysis_summary["recommended_k_from_multi_metric_analysis"])

    best_assignments = assignments_df[["sample_id", "cluster_id"]].drop_duplicates()
    nodes_with_cluster = nodes_df.merge(best_assignments, on="sample_id", how="inner")

    plot_k_selection(k_df, training_best_k, analysis_best_k, figures_dir)
    plot_tsne(tsne_df, figures_dir)
    plot_tsne_teamshape(tsne_df, figures_dir)
    plot_team_style_heatmap(team_fp, figures_dir)
    plot_context_trigger_heatmaps(analysis_dir, figures_dir)
    plot_cluster_area_over_time(nodes_with_cluster, figures_dir)
    plot_cluster_position_heatmaps(nodes_with_cluster, figures_dir)
    plot_score_shift(score_shift_df, figures_dir)
    plot_attention_overlay(analysis_dir, nodes_with_cluster, figures_dir)

    print(str(figures_dir))


if __name__ == "__main__":
    main()
