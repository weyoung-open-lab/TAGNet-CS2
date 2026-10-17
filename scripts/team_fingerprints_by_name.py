"""Team tactical fingerprints and style similarity by team name (paper Figs. 8 and 9).

``analyze_results.py`` writes one fingerprint per CT line-up (``ct_team_key``).
This script merges line-ups that share a team name, weighting by round count,
and computes the pairwise cosine distance between team fingerprints. The
similarity shown in Fig. 9 is ``1 - cosine_distance``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import cosine_distance  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "results" / "tagnet_main" / "run_analysis"


def fingerprints_by_name(fp_df: pd.DataFrame) -> pd.DataFrame:
    cluster_cols = [col for col in fp_df.columns if col.startswith("cluster_")]
    rows: list[dict[str, float | str | int]] = []
    for team_name, group in fp_df.groupby("ct_team_name", dropna=False):
        total_rounds = int(group["round_count"].sum())
        row: dict[str, float | str | int] = {"ct_team_name": team_name, "round_count": total_rounds}
        for col in cluster_cols:
            row[col] = float((group[col] * group["round_count"]).sum() / total_rounds)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["round_count", "ct_team_name"], ascending=[False, True]).reset_index(drop=True)


def similarity_pairs(team_fp: pd.DataFrame) -> pd.DataFrame:
    cluster_cols = [col for col in team_fp.columns if col.startswith("cluster_")]
    values = team_fp.set_index("ct_team_name")
    rows: list[dict[str, float | str]] = []
    for idx_a, team_a in enumerate(values.index):
        for idx_b, team_b in enumerate(values.index):
            if idx_b <= idx_a:
                continue
            vec_a = values.loc[team_a, cluster_cols].to_numpy(dtype=float)
            vec_b = values.loc[team_b, cluster_cols].to_numpy(dtype=float)
            rows.append({"team_a": team_a, "team_b": team_b, "cosine_distance": cosine_distance(vec_a, vec_b)})
    return pd.DataFrame(rows).sort_values("cosine_distance", ascending=True).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate team fingerprints by team name and compute style similarity.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    fp_df = pd.read_csv(args.analysis_dir / "team_style_fingerprints.csv", encoding="utf-8-sig")
    team_fp = fingerprints_by_name(fp_df)
    pairs = similarity_pairs(team_fp)
    team_fp.to_csv(args.analysis_dir / "team_style_fingerprints_by_name.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.analysis_dir / "team_style_similarity_by_name_pairs.csv", index=False, encoding="utf-8-sig")
    print(team_fp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
