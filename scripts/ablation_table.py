"""Rebuild the ablation table (paper Table VI) from trained runs.

For each run, the partition is the training-stage K-Means stored in
``cluster_assignments.parquet`` at that run's ``best_k_by_silhouette``
(written by ``train_tagnet.py``). SC, DBI and CH are then computed on the
run's round embeddings with scikit-learn.

Note: each variant is evaluated at its own best K (full TAGNet: K=8), not at
the K=6 partition used for the tactical analysis in Sections IV-C to IV-I.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "TAGNet (full)": REPO_ROOT / "results" / "tagnet_main" / "run",
    "w/o Context Feat.": REPO_ROOT / "results" / "ablations" / "no_context",
    "w/o Temporal Mod. (GRU)": REPO_ROOT / "results" / "ablations" / "single_snapshot_15s",
    "w/o Graph Structure": REPO_ROOT / "results" / "ablations" / "no_graph",
}


def evaluate_run(run_dir: Path) -> dict[str, float | int]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    best_k = int(summary["best_k_by_silhouette"])
    embeddings = pd.read_parquet(run_dir / "round_embeddings.parquet")
    assignments = pd.read_parquet(run_dir / "cluster_assignments.parquet")
    labels = assignments[assignments["k"] == best_k].set_index("sample_id").loc[embeddings["sample_id"], "cluster_id"].to_numpy()
    matrix = embeddings[[c for c in embeddings.columns if c.startswith("emb_")]].to_numpy(dtype=float)
    return {
        "test_loss": float(summary["test_eval"]["loss"]),
        "best_k": best_k,
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the ablation table from trained runs.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "ablations" / "ablation_table.csv")
    args = parser.parse_args()
    rows = [{"variant": name, **evaluate_run(run_dir)} for name, run_dir in DEFAULT_RUNS.items()]
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
