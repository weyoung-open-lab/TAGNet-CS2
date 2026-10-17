"""Draw the data figures of the paper from the files in results/.

Output (vector PDF and 600 dpi PNG, named by figure number in the paper):

    fig3_convergence        training curves of the baseline and TAGNet
    k_selection_curves      K-selection curves (numbers in Table IV)
    fig4_cluster_heatmaps   CT positions of each cluster at 15 s on the Dust2 base map
    fig5_tsne               t-SNE of the round embeddings
    fig6_area_over_time     area occupancy of each cluster over time
    fig7_team_style         (a) team fingerprints, (b) team style similarity
    fig8_context_tests      (a) significance and effect size, (b) raw vs. Holm p
    fig9_score_state        leading vs. trailing usage of each cluster
    fig10_ablation_radar    ablation radar chart
    fig11_supplementary     (a) rule-based baseline, (b) supplementary variants

Figs. 1 and 2 are diagrams. The base map of Fig. 4 is built by build_map_mask.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
ANALYSIS = RESULTS / "tagnet_main" / "run_analysis"
NODES = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap" / "ct_snapshot_nodes.parquet"
MAP_MASK = REPO_ROOT / "data" / "map" / "dust2_walkable_mask.npz"
COLUMN = 3.5
PAGE = 7.16

CLUSTER_NAMES = [
    "Anchor-A Biased",
    "Default Balanced",
    "B-Biased Double-Point",
    "Mid-Control Balanced",
    "A-Long+Mid",
    "Aggressive A-Heavy Push",
]
CLUSTER_SHORT = [
    "Anchor-A\nBiased",
    "Default\nBalanced",
    "B-Biased\nDouble-Pt.",
    "Mid-Control\nBalanced",
    "A-Long\n+Mid",
    "Aggr. A-Heavy\nPush",
]
CLUSTER_TITLES = [
    "C0 Anchor-A Biased",
    "C1 Default Balanced",
    "C2 B-Biased Double-Pt.",
    "C3 Mid-Control Bal.",
    "C4 A-Long+Mid",
    "C5 Aggr. A-Heavy Push",
]
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
TEAM_NAMES = {
    "natus-vincere": "Natus Vincere",
    "parivision": "PARIVISION",
    "furia": "FURIA",
    "falcons": "Falcons",
    "aurora": "Aurora",
    "mouz": "MOUZ",
    "heroic": "Heroic",
    "monte": "Monte",
    "9z": "9z",
    "fut": "FUT",
    "the-mongolz": "The MongolZ",
    "spirit": "Spirit",
    "vitality": "Vitality",
    "nrg": "NRG",
    "astralis": "Astralis",
    "legacy": "Legacy",
    "b8": "B8",
    "faze": "FaZe",
    "gamerlegion": "GamerLegion",
    "inner-circle": "Inner Circle",
    "liquid": "Liquid",
    "pain": "paiN",
    "semperfi": "SemperFi",
    "virtus-pro": "Virtus.pro",
    "g2": "G2",
    "3dmax": "3DMAX",
}
CONTEXT_LABELS = {
    "ct_econ_type": "CT economy\ntype",
    "prev_round_winner_side": "Prev.-round\nwinner",
    "prev_round_attack_site": "Prev.-round\nattack dir.",
    "score_state": "Current\nscore state",
}
SUPP_VARIANTS = [
    ("full_gat_gru", "Full TAGNet", "#1a4f8b"),
    ("t20_snapshot", "t = 20 s only", "#319795"),
    ("mean_pool", "Mean-pool", "#319795"),
    ("nav_geodesic", "Navmesh geodesic", "#805ad5"),
    ("nav_geodesic_hinge1", "Geodesic + hinge", "#805ad5"),
    ("no_graph_gru", "No-graph MLP+GRU", "#c53030"),
    ("no_graph_lstm", "No-graph MLP+LSTM", "#c53030"),
    ("xy_replace_place", "Continuous XY", "#dd8452"),
]
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"
GREY = "#8a8a8a"
ORANGE = "#dd6b20"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def panel_label(fig: plt.Figure, x: float, y: float, text: str) -> None:
    fig.text(x, y, text, fontsize=8, fontweight="bold", ha="left", va="top")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def significance_color(p: float) -> str:
    if p < 0.001:
        return RED
    if p < 0.05:
        return ORANGE
    return GREY


def context_tests() -> pd.DataFrame:
    tests = read_csv(RESULTS / "supplementary" / "holm" / "context_trigger_tests.csv").set_index("variable")
    return tests.loc[list(CONTEXT_LABELS)]


def team_fingerprints() -> pd.DataFrame:
    fp = read_csv(ANALYSIS / "team_style_fingerprints_by_name.csv")
    return fp.sort_values(["round_count", "ct_team_name"], ascending=[False, True]).reset_index(drop=True)


# Fig. 3 ---------------------------------------------------------------------

def draw_loss_curves(axes) -> None:
    runs = [
        ("Baseline (2 layers, 3 snapshots)", RESULTS / "baseline_2layer_3snap" / "run", BLUE),
        ("TAGNet (3 layers, 5 snapshots)", RESULTS / "tagnet_main" / "run", RED),
    ]
    for ax, (title, run_dir, color) in zip(axes, runs):
        history = read_csv(run_dir / "training_history.csv")
        test_loss = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))["test_eval"]["loss"]
        ax.plot(history["epoch"], history["train_loss"], color=color, label="Train")
        ax.plot(history["epoch"], history["val_loss"], color=color, linestyle="--", label="Validation")
        ax.axhline(test_loss, color=color, linestyle=":", linewidth=1.0, label=f"Test = {test_loss:.4f}")
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, history["epoch"].max())
        ax.legend(loc="upper right", handlelength=1.8)
    axes[0].set_ylabel("Reconstruction loss")


def draw_k_selection(axes) -> None:
    k_df = read_csv(ANALYSIS / "k_selection_metrics.csv")
    k_df = k_df[k_df["k"] <= 8]
    panels = [
        ("silhouette", "Silhouette\n(higher better)", BLUE),
        ("davies_bouldin", "Davies-Bouldin\n(lower better)", RED),
        ("calinski_harabasz", "Calinski-Harabasz\n(higher better)", GREEN),
    ]
    for ax, (col, title, color) in zip(axes, panels):
        ax.axvline(6, color=ORANGE, linestyle="--", linewidth=0.9, zorder=0)
        ax.plot(k_df["k"], k_df[col], color=color, marker="o", markersize=2.8)
        ax.set_title(title, fontsize=6)
        ax.set_xticks(range(2, 9))
        ax.set_xlabel("K")
    axes[0].annotate("K = 6", (6, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points", ha="left", va="top", fontsize=6, color=ORANGE)


def fig3_convergence(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.6), sharey=True)
    draw_loss_curves(list(axes))
    fig.tight_layout(w_pad=0.6)
    save(fig, out_dir, "fig3_convergence")


def fig_k_selection(out_dir: Path) -> None:
    """K-selection curves; the same numbers are listed in Table IV of the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN, 1.45))
    draw_k_selection(list(axes))
    fig.tight_layout(w_pad=0.4)
    save(fig, out_dir, "k_selection_curves")


# Fig. 4 ---------------------------------------------------------------------

def fig4_cluster_heatmaps(out_dir: Path) -> None:
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap
    from scipy import ndimage

    base = np.load(MAP_MASK)
    walkable, map_x, map_y = base["mask"], base["x_edges"], base["y_edges"]
    nodes = pd.read_parquet(NODES, columns=["sample_id", "second", "area", "X", "Y"])
    assignments = pd.read_parquet(ANALYSIS / "analysis_cluster_assignments_bestk.parquet", columns=["sample_id", "cluster_id"])
    nodes = nodes[nodes["second"] == 15].merge(assignments.drop_duplicates(), on="sample_id")

    x0, x1, y0, y1, cell = -2300, 1900, 250, 3200, 40
    bins_x = np.arange(x0, x1 + cell, cell)
    bins_y = np.arange(y0, y1 + cell, cell)
    centre_x = ((bins_x[:-1] + cell / 2 - map_x[0]) // (map_x[1] - map_x[0])).astype(int)
    centre_y = ((bins_y[:-1] + cell / 2 - map_y[0]) // (map_y[1] - map_y[0])).astype(int)
    on_map = walkable[np.ix_(np.clip(centre_x, 0, walkable.shape[0] - 1), np.clip(centre_y, 0, walkable.shape[1] - 1))]

    density = {}
    for cluster in range(6):
        group = nodes[nodes["cluster_id"] == cluster]
        counts, _, _ = np.histogram2d(group["X"], group["Y"], bins=[bins_x, bins_y])
        density[cluster] = ndimage.gaussian_filter(counts, 1.0) / group["sample_id"].nunique() * on_map
    vmax = np.percentile(np.concatenate([d[d > 0] for d in density.values()]), 99.5)

    places = {
        "B site": (-1700, 3060),
        "A site": (1250, 3040),
        "CT": (150, 2640),
        "Mid": (-350, 560),
        "Short": (80, 1950),
        "Long A": (1150, 850),
        "B tun.": (-1880, 760),
    }
    halo = [patheffects.withStroke(linewidth=1.3, foreground="white")]
    fig, axes = plt.subplots(2, 3, figsize=(COLUMN, 2.0), gridspec_kw={"wspace": 0.04, "hspace": 0.3})
    for cluster, ax in enumerate(axes.flat):
        ax.imshow(
            np.where(walkable, 1.0, np.nan).T, origin="lower", extent=[map_x[0], map_x[-1], map_y[0], map_y[-1]],
            cmap=ListedColormap(["#dadada"]), interpolation="nearest", zorder=0,
        )
        ax.imshow(
            np.ma.masked_less(density[cluster], 0.02 * vmax).T, origin="lower",
            extent=[x0, x0 + cell * (len(bins_x) - 1), y0, y0 + cell * (len(bins_y) - 1)],
            cmap="YlOrRd", vmin=0, vmax=vmax, alpha=0.92, interpolation="bilinear", zorder=1,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_color("#8a8a8a")
        share = nodes.loc[nodes["cluster_id"] == cluster, "area"].value_counts(normalize=True)
        ax.set_title(
            f"{CLUSTER_TITLES[cluster]}\nA {share.get('A', 0):.2f}, B {share.get('B', 0):.2f}, Mid {share.get('Mid', 0):.2f}",
            fontsize=5.8, pad=1.5, linespacing=1.1,
        )
        for name, (x, y) in places.items():
            ax.text(x, y, name, fontsize=5.0, color="#222222", ha="center", va="center", zorder=3, path_effects=halo)
    save(fig, out_dir, "fig4_cluster_heatmaps")


# Fig. 5 ---------------------------------------------------------------------

def fig5_tsne(out_dir: Path) -> pd.DataFrame:
    tsne = pd.read_parquet(ANALYSIS / "tsne_coordinates_paper_fig6.parquet")
    markers = {"train": "o", "val": "s", "test": "^"}
    fig, ax = plt.subplots(figsize=(COLUMN, 2.65))
    for split, marker in markers.items():
        part = tsne[tsne["split"] == split]
        ax.scatter(
            part["tsne_x"],
            part["tsne_y"],
            c=[CLUSTER_COLORS[c] for c in part["cluster_id"]],
            marker=marker,
            s=8 if split == "train" else 11,
            linewidths=0.2,
            edgecolors="white",
            alpha=0.9,
        )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    cluster_handles = [Patch(color=CLUSTER_COLORS[i], label=f"C{i} {CLUSTER_NAMES[i]}") for i in range(6)]
    split_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor=GREY, markersize=4.5, label=s)
        for s, m in [("Training", "o"), ("Validation", "s"), ("Test", "^")]
    ]
    ax.legend(handles=split_handles, loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.32)
    fig.legend(handles=cluster_handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2, columnspacing=1.0)
    save(fig, out_dir, "fig5_tsne")
    return tsne


def tsne_inset(tsne: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.scatter(tsne["tsne_x"], tsne["tsne_y"], c=[CLUSTER_COLORS[c] for c in tsne["cluster_id"]], s=4, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#999999")
    fig.savefig(out_dir / "fig1_tsne_inset.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)


# Fig. 6 ---------------------------------------------------------------------

def parse_area_distribution(text: str) -> dict[str, float]:
    values = {}
    for item in text.split(","):
        area, value = item.strip().split(":")
        values[area.strip()] = float(value)
    return values


def fig6_area_over_time(out_dir: Path) -> None:
    profiles = read_csv(ANALYSIS / "cluster_profiles.csv")
    styles = {
        "A": ("A site", RED, "-"),
        "B": ("B site", BLUE, "--"),
        "Mid": ("Mid", GREEN, "-."),
        "CT": ("CT spawn", "#805ad5", ":"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(COLUMN, 2.2), sharex=True, sharey=True)
    for cluster_id, ax in enumerate(axes.flat):
        part = profiles[profiles["cluster_id"] == cluster_id].sort_values("second")
        dist = [parse_area_distribution(s) for s in part["area_distribution"]]
        for area, (label, color, ls) in styles.items():
            ax.plot(part["second"], [d.get(area, 0.0) for d in dist], color=color, linestyle=ls, marker="o", markersize=2, label=label)
        ax.set_title(CLUSTER_TITLES[cluster_id], color=CLUSTER_COLORS[cluster_id], fontsize=6)
        ax.set_xticks([10, 12, 15, 18, 20])
        ax.set_ylim(0, 0.8)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Area share")
    handles = [Line2D([0], [0], color=c, linestyle=ls, marker="o", markersize=2, label=l) for l, c, ls in styles.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, out_dir, "fig6_area_over_time")


# Fig. 7 ---------------------------------------------------------------------

def draw_team_heatmap(fig: plt.Figure, ax) -> None:
    fp = team_fingerprints().head(15)
    values = fp[[f"cluster_{i}" for i in range(6)]].to_numpy()
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6, color="white" if v > 0.3 else "#1a202c")
    ax.set_yticks(range(len(fp)))
    ax.set_yticklabels([f"{TEAM_NAMES.get(n, n)} ({r})" for n, r in zip(fp["ct_team_name"], fp["round_count"])], fontsize=6)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"C{i} {CLUSTER_SHORT[i]}" for i in range(6)], rotation=45, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax.set_ylabel("CT team (rounds)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.06, pad=0.03)
    bar.set_label("Usage proportion")
    bar.outline.set_linewidth(0.5)


def draw_team_similarity(fig: plt.Figure, ax) -> None:
    fp = team_fingerprints().head(12)
    values = fp[[f"cluster_{i}" for i in range(6)]].to_numpy()
    norm = values / np.linalg.norm(values, axis=1, keepdims=True)
    sim = norm @ norm.T
    masked = np.ma.masked_where(np.eye(len(fp), dtype=bool), sim)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0)
    for i in range(len(fp)):
        for j in range(len(fp)):
            if i != j:
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color="white" if sim[i, j] > 0.85 else "#1a202c")
    labels = [TEAM_NAMES.get(n, n) for n in fp["ct_team_name"]]
    ax.set_xticks(range(len(fp)))
    ax.set_yticks(range(len(fp)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Cosine similarity")
    bar.outline.set_linewidth(0.5)


def fig7_team_style(out_dir: Path) -> None:
    fig = plt.figure(figsize=(PAGE, 2.75))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], left=0.11, right=0.97, top=0.97, bottom=0.17, wspace=0.42)
    draw_team_heatmap(fig, fig.add_subplot(grid[0, 0]))
    draw_team_similarity(fig, fig.add_subplot(grid[0, 1]))
    panel_label(fig, 0.0, 1.0, "(a)")
    panel_label(fig, 0.5, 1.0, "(b)")
    save(fig, out_dir, "fig7_team_style")


# Fig. 8 ---------------------------------------------------------------------

def draw_significance(ax) -> None:
    tests = context_tests()
    x = np.arange(len(tests))
    logp = -np.log10(tests["p_value"].to_numpy())
    colors = [significance_color(p) for p in tests["p_value"]]
    ax.scatter(x, logp, s=tests["cramers_v"].to_numpy() * 600, c=colors, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
    for xi, yi, p, ph in zip(x, logp, tests["p_value"], tests["p_holm"]):
        mark = stars(p) + ("$^{\\dagger}$" if p < 0.05 <= ph else "")
        ax.annotate(mark, (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ax.axhline(-np.log10(0.05), color=ORANGE, linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(0.001), color=RED, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CONTEXT_LABELS[v]}\n$\\chi^2$={c:.2f}\nV={vv:.3f}" for v, c, vv in zip(tests.index, tests["chi2"], tests["cramers_v"])],
        fontsize=5.6,
    )
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, logp.max() * 1.3)
    ax.set_ylabel(r"$-\log_{10}(p)$, raw")
    ax.grid(axis="x", visible=False)


def draw_raw_vs_holm(ax) -> None:
    tests = context_tests()
    x = np.arange(len(tests))
    ax.bar(x - 0.19, -np.log10(tests["p_value"].to_numpy()), 0.36, color="#3182ce", label="Raw p")
    ax.bar(x + 0.19, -np.log10(tests["p_holm"].to_numpy()), 0.36, color=RED, label="Holm p")
    ax.axhline(-np.log10(0.05), color=ORANGE, linestyle="--", linewidth=0.8, label="p = 0.05")
    ax.axhline(-np.log10(0.001), color="#9b2c2c", linestyle=":", linewidth=0.9, label="p = 0.001")
    ax.set_xticks(x)
    ax.set_xticklabels([CONTEXT_LABELS[v] for v in tests.index], fontsize=6)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)


def fig8_context_tests(out_dir: Path) -> None:
    fig = plt.figure(figsize=(COLUMN, 3.0))
    grid = fig.add_gridspec(2, 1, left=0.13, right=0.99, top=0.97, bottom=0.08, hspace=0.75)
    draw_significance(fig.add_subplot(grid[0, 0]))
    draw_raw_vs_holm(fig.add_subplot(grid[1, 0]))
    panel_label(fig, 0.0, 1.0, "(a)")
    panel_label(fig, 0.0, 0.49, "(b)")
    save(fig, out_dir, "fig8_context_tests")


# Fig. 9 ---------------------------------------------------------------------

def fig9_score_state(out_dir: Path) -> None:
    pairs = read_csv(ANALYSIS / "h2_score_state_team_pairs.csv")
    tests = read_csv(RESULTS / "supplementary" / "holm" / "h2_score_state_paired_ttest.csv").set_index("cluster_id")
    n = len(pairs)
    t_crit = stats.t.ppf(0.975, n - 1)
    lead = [pairs[f"cluster_{i}_leading"] for i in range(6)]
    trail = [pairs[f"cluster_{i}_trailing"] for i in range(6)]
    lead_mean = np.array([v.mean() for v in lead])
    trail_mean = np.array([v.mean() for v in trail])
    lead_ci = np.array([t_crit * v.std(ddof=1) / np.sqrt(n) for v in lead])
    trail_ci = np.array([t_crit * v.std(ddof=1) / np.sqrt(n) for v in trail])
    x = np.arange(6)
    w = 0.38
    fig, ax = plt.subplots(figsize=(COLUMN, 1.7))
    err = {"elinewidth": 0.7, "capthick": 0.7}
    ax.bar(x - w / 2, lead_mean, w, yerr=lead_ci, capsize=2, color=BLUE, label="Leading", error_kw=err)
    ax.bar(x + w / 2, trail_mean, w, yerr=trail_ci, capsize=2, color=RED, label="Trailing", error_kw=err)
    top = np.maximum(lead_mean + lead_ci, trail_mean + trail_ci)
    for xi, yi, cid in zip(x, top, range(6)):
        p = tests.loc[cid, "p_value"]
        ax.text(xi, yi + 0.012, stars(p), ha="center", va="bottom", fontsize=6, color=RED if p < 0.05 else GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}\n{CLUSTER_SHORT[i]}" for i in range(6)], fontsize=5.6)
    ax.set_ylabel("Mean usage proportion")
    ax.set_ylim(0, max(top) + 0.06)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out_dir, "fig9_score_state")


# Fig. 10 --------------------------------------------------------------------

ABLATION_ROWS = {
    "TAGNet (full)": (0.0086, 0.1788, 1.7094, 104.67, 10.0),
    "w/o Context Feat.": (0.0086, 0.1857, 1.6716, 100.44, 6.0),
    "w/o Temporal Mod. (GRU)": (0.0090, 0.2074, 1.6684, 108.95, 6.0),
    "w/o Graph Structure": (0.0143, 0.1716, 1.7579, 165.90, 2.0),
}


def fig10_ablation_radar(out_dir: Path) -> None:
    names = list(ABLATION_ROWS)
    loss, sc, dbi, ch, semantic = (np.array([ABLATION_ROWS[n][i] for n in names]) for i in range(5))
    scores = np.column_stack([10 * loss.min() / loss, 10 * sc / sc.max(), 10 * dbi.min() / dbi, 10 * ch / ch.max(), semantic])
    axes_labels = ["Test loss\n(inverse)", "Silhouette", "DBI\n(inverse)", "CH", "Semantic\nvalidity"]
    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    colors = [BLUE, ORANGE, GREEN, RED]
    fig = plt.figure(figsize=(COLUMN, 2.35))
    ax = fig.add_subplot(111, polar=True)
    for name, row, color in zip(names, scores, colors):
        values = np.concatenate([row, row[:1]])
        ax.plot(angles, values, color=color, linewidth=1.8 if name == "TAGNet (full)" else 1.0, label=name)
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=6.3)
    ax.set_ylim(0, 10.5)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=5.5, color=GREY)
    ax.spines["polar"].set_linewidth(0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out_dir, "fig10_ablation_radar")


# Fig. 11 --------------------------------------------------------------------

def draw_rule_baseline(axes) -> None:
    metrics = json.loads((RESULTS / "supplementary" / "rule_baseline" / "baseline_metrics.json").read_text(encoding="utf-8"))
    methods = [("Count-seq.\nK-Means", "count_kmeans", "#319795"), ("Fixed\nrule", "rule_labels", "#dd8452"), ("TAGNet", "tagnet_kmeans", "#1a4f8b")]
    panels = [("silhouette", "Silhouette (higher better)"), ("davies_bouldin", "Davies-Bouldin (lower better)"), ("calinski_harabasz", "Calinski-Harabasz (higher better)")]
    for ax, (key, title) in zip(axes[:3], panels):
        vals = [metrics[m][key] for _, m, _ in methods]
        ax.bar(range(3), vals, color=[c for _, _, c in methods], width=0.65)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for n, _, _ in methods], fontsize=6)
        ax.set_title(title, fontsize=6.5)
        ax.grid(axis="x", visible=False)
    agree = metrics["tagnet_agreement"]
    ax = axes[3]
    x = np.arange(2)
    ax.bar(x - 0.18, [agree["count_kmeans_ari"], agree["rule_ari"]], 0.34, color="#3182ce", label="ARI")
    ax.bar(x + 0.18, [agree["count_kmeans_nmi"], agree["rule_nmi"]], 0.34, color="#ecc94b", label="NMI")
    ax.set_xticks(x)
    ax.set_xticklabels(["Count-seq.\nK-Means", "Fixed\nrule"], fontsize=6)
    ax.set_ylim(0, 0.6)
    ax.set_title("Agreement with TAGNet", fontsize=6.5)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)


def draw_supplementary_variants(ax_loss, ax_ari) -> None:
    summary = read_csv(RESULTS / "supplementary" / "summary" / "original_protocol_summary.csv").set_index("variant")
    keys = [k for k, _, _ in SUPP_VARIANTS]
    labels = [l for _, l, _ in SUPP_VARIANTS]
    colors = [c for _, _, c in SUPP_VARIANTS]
    loss = summary.loc[keys, "test_total_loss_mean"].to_numpy()
    loss_sd = summary.loc[keys, "test_total_loss_std"].to_numpy()
    ari = summary.loc[keys, "mean_pairwise_ari_all697"].to_numpy()
    x = np.arange(len(keys))
    ax_loss.bar(x, loss, yerr=loss_sd, color=colors, width=0.7, capsize=1.5, error_kw={"elinewidth": 0.6})
    ax_loss.axhline(loss[0], color=GREY, linestyle="--", linewidth=0.8)
    ax_loss.set_ylabel("Test loss")
    ax_loss.set_title("Reconstruction loss (lower better)", fontsize=6.5)
    ax_ari.bar(x, ari, color=colors, width=0.7)
    ax_ari.axhline(ari[0], color=GREY, linestyle="--", linewidth=0.8)
    ax_ari.set_ylabel("Cross-seed ARI")
    ax_ari.set_title("Cross-seed cluster agreement (higher better)", fontsize=6.5)
    for ax in (ax_loss, ax_ari):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
        ax.grid(axis="x", visible=False)


def fig11_supplementary(out_dir: Path) -> None:
    fig = plt.figure(figsize=(PAGE, 3.0))
    top = fig.add_gridspec(1, 4, left=0.05, right=0.99, top=0.93, bottom=0.63, wspace=0.32)
    bottom = fig.add_gridspec(1, 2, left=0.05, right=0.99, top=0.47, bottom=0.17, wspace=0.16)
    draw_rule_baseline([fig.add_subplot(top[0, i]) for i in range(4)])
    draw_supplementary_variants(fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1]))
    panel_label(fig, 0.0, 1.0, "(a)")
    panel_label(fig, 0.0, 0.535, "(b)")
    save(fig, out_dir, "fig11_supplementary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the data figures of the paper from results/.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "paper_figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig3_convergence(args.output_dir)
    fig_k_selection(args.output_dir)
    fig4_cluster_heatmaps(args.output_dir)
    tsne = fig5_tsne(args.output_dir)
    tsne_inset(tsne, args.output_dir)
    fig6_area_over_time(args.output_dir)
    fig7_team_style(args.output_dir)
    fig8_context_tests(args.output_dir)
    fig9_score_state(args.output_dir)
    fig10_ablation_radar(args.output_dir)
    fig11_supplementary(args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
