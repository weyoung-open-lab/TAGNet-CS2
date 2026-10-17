from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "tagnet_main" / "run"


def weak_label_purity(labels: list[str | None], clusters: np.ndarray) -> float | None:
    filtered = [(label, int(cluster)) for label, cluster in zip(labels, clusters.tolist()) if label is not None and str(label) != "nan"]
    if not filtered:
        return None
    counts: dict[int, dict[str, int]] = {}
    for label, cluster in filtered:
        counts.setdefault(cluster, {})
        counts[cluster][label] = counts[cluster].get(label, 0) + 1
    correct = sum(max(v.values()) for v in counts.values())
    return correct / len(filtered)


def get_labels(method: str, x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    if method == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed)
        return model.fit_predict(x)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
    if method == "spectral":
        neighbors = max(10, min(40, x.shape[0] // 20))
        return SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(x)
    raise ValueError(f"unsupported method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clusterers on TAGNet embeddings.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(f"{run_dir}_clusterer_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_df = pd.read_parquet(run_dir / "round_embeddings.parquet")
    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    x = embeddings_df[emb_cols].to_numpy(dtype=float)
    weak_labels = [None if pd.isna(v) else str(v) for v in embeddings_df["weak_label_15s"].tolist()]

    methods = ["kmeans", "gmm", "agglomerative", "spectral"]
    rows: list[dict[str, object]] = []
    assignments_store: dict[tuple[str, int], np.ndarray] = {}

    for method in methods:
        for k in range(args.min_k, args.max_k + 1):
            labels = get_labels(method, x, k, args.seed)
            if len(np.unique(labels)) < 2:
                continue
            silhouette = silhouette_score(x, labels)
            dbi = davies_bouldin_score(x, labels)
            ch = calinski_harabasz_score(x, labels)
            purity = weak_label_purity(weak_labels, labels)
            sizes = np.bincount(labels, minlength=k)
            rows.append(
                {
                    "method": method,
                    "k": int(k),
                    "silhouette": float(silhouette),
                    "davies_bouldin": float(dbi),
                    "calinski_harabasz": float(ch),
                    "weak_label_purity_at_15s": None if purity is None else float(purity),
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                    "cluster_sizes": ",".join(map(str, sizes.tolist())),
                }
            )
            assignments_store[(method, k)] = labels

    results_df = pd.DataFrame(rows)
    results_df["rank_silhouette"] = results_df["silhouette"].rank(ascending=False, method="dense")
    results_df["rank_davies_bouldin"] = results_df["davies_bouldin"].rank(ascending=True, method="dense")
    results_df["rank_calinski_harabasz"] = results_df["calinski_harabasz"].rank(ascending=False, method="dense")
    results_df["avg_rank"] = results_df[["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]].mean(axis=1)
    results_df = results_df.sort_values(["avg_rank", "rank_silhouette", "method", "k"]).reset_index(drop=True)
    results_df.to_csv(out_dir / "clusterer_comparison.csv", index=False, encoding="utf-8-sig")

    best_row = results_df.iloc[0]
    best_method = str(best_row["method"])
    best_k = int(best_row["k"])
    best_labels = assignments_store[(best_method, best_k)]
    best_assignments = embeddings_df[["sample_id", "match_id", "round_num", "split", "weak_label_15s"]].copy()
    best_assignments["cluster_id"] = best_labels
    best_assignments.to_parquet(out_dir / "best_cluster_assignments.parquet", index=False)

    summary = {
        "best_method": best_method,
        "best_k": best_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_davies_bouldin": float(best_row["davies_bouldin"]),
        "best_calinski_harabasz": float(best_row["calinski_harabasz"]),
    }
    (out_dir / "clusterer_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "clusterer_comparison_summary.md").write_text(
        "\n".join(
            [
                "# Clusterer Comparison",
                "",
                f"- Best method: `{best_method}`",
                f"- Best K: `{best_k}`",
                "",
                "```text",
                results_df.to_string(index=False),
                "```",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(str(out_dir))


if __name__ == "__main__":
    main()
