"""Domain rule-based positional baseline vs. TAGNet (paper Section IV-K-1, Table VII)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score


AREAS = ("A", "B", "Mid", "CT", "other")
SECONDS = (10, 12, 15, 18, 20)


def rule_label(counts: pd.Series) -> str:
    a, b, mid = int(counts["A"]), int(counts["B"]), int(counts["Mid"])
    if mid >= 2:
        return "mid_control"
    if a >= 4:
        return "a_heavy"
    if b >= 3:
        return "b_heavy"
    if a >= b + 2:
        return "a_leaning"
    if b >= a + 2:
        return "b_leaning"
    return "balanced"


def cluster_metrics(features, labels) -> dict[str, float | int]:
    return {
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "cluster_count": int(len(set(labels))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Original-protocol rule baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tagnet-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()
    nodes = pd.read_parquet(args.data_dir / "ct_snapshot_nodes.parquet")
    rounds = pd.read_parquet(args.data_dir / "ct_round_samples.parquet")
    counts = nodes.groupby(["sample_id", "second", "area"]).size().unstack(fill_value=0).reindex(columns=AREAS, fill_value=0)
    counts = counts.reindex(pd.MultiIndex.from_product([rounds["sample_id"], SECONDS], names=["sample_id", "second"]), fill_value=0)
    if not (counts.sum(axis=1) == 5).all():
        raise ValueError("Each included snapshot must contain five CT players.")
    features = counts.unstack("second").reindex(columns=pd.MultiIndex.from_product([AREAS, SECONDS]), fill_value=0)
    features.columns = [f"count_{area}_{second}" for area, second in features.columns]
    frame = rounds[["sample_id", "match_id", "round_num", "split"]].merge(features.reset_index(), on="sample_id", how="inner", validate="one_to_one")
    final_counts = counts.xs(20, level="second").reindex(frame["sample_id"])
    frame["rule_label"] = final_counts.apply(rule_label, axis=1).to_numpy()
    rule_codes, rule_names = pd.factorize(frame["rule_label"], sort=True)
    frame["rule_cluster"] = rule_codes
    feature_matrix = frame[features.columns].to_numpy(dtype=float)
    count_labels = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed).fit_predict(feature_matrix)
    frame["count_kmeans_cluster"] = count_labels

    embeddings = pd.read_parquet(args.tagnet_run_dir / "round_embeddings.parquet")
    emb_cols = [column for column in embeddings.columns if column.startswith("emb_")]
    tagnet_labels = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed).fit_predict(embeddings[emb_cols].to_numpy(dtype=float))
    tagnet = embeddings[["sample_id"]].copy()
    tagnet["tagnet_cluster"] = tagnet_labels
    frame = frame.merge(tagnet, on="sample_id", how="inner", validate="one_to_one")
    metrics = {
        "protocol": "original_all_697_sklearn_kmeans",
        "sample_count": int(len(frame)),
        "rule_label_counts": frame["rule_label"].value_counts().to_dict(),
        "count_kmeans": cluster_metrics(feature_matrix, count_labels),
        "rule_labels": cluster_metrics(feature_matrix, rule_codes),
        "tagnet_kmeans": cluster_metrics(embeddings[emb_cols].to_numpy(dtype=float), tagnet_labels),
        "tagnet_agreement": {
            "count_kmeans_ari": float(adjusted_rand_score(frame["tagnet_cluster"], frame["count_kmeans_cluster"])),
            "count_kmeans_nmi": float(normalized_mutual_info_score(frame["tagnet_cluster"], frame["count_kmeans_cluster"])),
            "rule_ari": float(adjusted_rand_score(frame["tagnet_cluster"], frame["rule_cluster"])),
            "rule_nmi": float(normalized_mutual_info_score(frame["tagnet_cluster"], frame["rule_cluster"])),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "baseline_assignments.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(args.output_dir / "baseline_assignments.parquet", index=False)
    final_counts.assign(rule_label=frame["rule_label"].to_numpy()).value_counts().rename("round_count").reset_index().to_csv(args.output_dir / "final_area_config_counts.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "baseline_summary.md").write_text("# Rule-based positional baseline (paper Table VII)\n\n```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
