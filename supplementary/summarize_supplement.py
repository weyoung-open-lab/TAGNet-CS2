"""Summarize the five-seed supplementary runs (paper Tables VIII-X and XII)."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score


VARIANTS = (
    "full_gat_gru", "t20_snapshot", "mean_pool", "no_graph_gru", "no_graph_lstm",
    "nav_geodesic", "nav_geodesic_hinge1", "xy_replace_place",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize original-protocol supplementary runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows, summary_rows, assignment_rows = [], [], []
    for variant in VARIANTS:
        pattern = re.compile(rf"^{re.escape(variant)}_seed(42|137|271|503|911)$")
        run_dirs = sorted(path for path in args.run_root.iterdir() if path.is_dir() and pattern.fullmatch(path.name) and (path / "run_summary.json").exists())
        if len(run_dirs) != 5:
            raise RuntimeError(f"{variant}: expected 5 completed seeds, found {len(run_dirs)}")
        assignments = []
        for run_dir in run_dirs:
            seed = int(run_dir.name.rsplit("seed", 1)[1])
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            embedding = pd.read_parquet(run_dir / "round_embeddings.parquet")
            embedding = embedding.sort_values("sample_id").reset_index(drop=True)
            columns = [column for column in embedding.columns if column.startswith("emb_")]
            matrix = embedding[columns].to_numpy(dtype=float)
            labels = KMeans(n_clusters=args.k, n_init=20, random_state=args.kmeans_seed).fit_predict(matrix)
            assignments.append(pd.Series(labels, index=embedding["sample_id"].astype(str)))
            for sample_id, split, cluster_id in zip(embedding["sample_id"], embedding["split"], labels):
                assignment_rows.append({"variant": variant, "seed": seed, "sample_id": sample_id, "split": split, "cluster_id": int(cluster_id)})
            test = summary["test_eval"]
            reconstruction = float(test["place_loss"]) + 0.5 * float(test["area_loss"]) + float(test["distance_loss"])
            detail_rows.append({
                "variant": variant, "seed": seed, "test_total_loss": float(test["loss"]),
                "test_reconstruction_loss": reconstruction, "test_hinge_loss": float(test.get("hinge_loss", 0.0)),
                "silhouette_k6_all697": float(silhouette_score(matrix, labels)),
                "dbi_k6_all697": float(davies_bouldin_score(matrix, labels)),
                "ch_k6_all697": float(calinski_harabasz_score(matrix, labels)),
                "min_cluster_size": int(pd.Series(labels).value_counts().min()),
                "max_cluster_size": int(pd.Series(labels).value_counts().max()),
                "train_loss": float(summary["train_eval"]["loss"]),
                "val_loss": float(summary["val_eval"]["loss"]),
            })
        frame = pd.DataFrame([row for row in detail_rows if row["variant"] == variant])
        stability = [adjusted_rand_score(left.loc[right.index], right) for left, right in itertools.combinations(assignments, 2)]
        metrics = ["test_total_loss", "test_reconstruction_loss", "test_hinge_loss", "silhouette_k6_all697", "dbi_k6_all697", "ch_k6_all697", "min_cluster_size", "max_cluster_size", "train_loss", "val_loss"]
        summary_rows.append({
            "variant": variant, "seed_runs": len(frame),
            **{f"{metric}_mean": float(frame[metric].mean()) for metric in metrics},
            **{f"{metric}_std": float(frame[metric].std(ddof=1)) for metric in metrics},
            "mean_pairwise_ari_all697": float(sum(stability) / len(stability)),
        })
    detail = pd.DataFrame(detail_rows).sort_values(["variant", "seed"])
    summary = pd.DataFrame(summary_rows)
    assignments = pd.DataFrame(assignment_rows)
    detail.to_csv(args.output_dir / "original_protocol_seed_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "original_protocol_summary.csv", index=False, encoding="utf-8-sig")
    assignments.to_parquet(args.output_dir / "original_protocol_sklearn_k6_assignments.parquet", index=False)
    markdown = (
        "# Supplementary experiments: summary over five seeds\n\n"
        "Clustering metrics follow the paper: sklearn KMeans (K=6, n_init=20, random_state=42) is refit on all 697 round embeddings. "
        "Test loss is computed on the 116 rounds of the match-level test split.\n\n"
        "```text\n" + summary.to_string(index=False) + "\n```\n"
    )
    (args.output_dir / "original_protocol_summary.md").write_text(markdown, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
