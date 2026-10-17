from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_rel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "tagnet_main" / "run"
TEAM_NAME_ALIASES = {
    "navi": "natus-vincere",
}


def normalize_team_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if not normalized:
        return None
    return TEAM_NAME_ALIASES.get(normalized, normalized)


def parse_match_names(match_id: str) -> tuple[str | None, str | None]:
    base = match_id.removesuffix("_dust2")
    if "-vs-" not in base:
        return None, None
    left, right = base.split("-vs-", 1)
    return normalize_team_name(left), normalize_team_name(right)


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


def cramers_v(table: pd.DataFrame) -> float | None:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    denom = max(min(k - 1, r - 1), 1)
    return float(math.sqrt(phi2 / denom))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(1.0 - float(np.dot(vec_a, vec_b) / denom))


def infer_team_names(rounds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_counts: dict[str, dict[str, int]] = {}
    match_counts: dict[str, set[str]] = {}
    unique_matches = rounds_df[["match_id", "ct_team_key", "t_team_key"]].drop_duplicates()

    for _, row in unique_matches.iterrows():
        match_id = str(row["match_id"])
        team_a, team_b = parse_match_names(match_id)
        candidates = [name for name in [team_a, team_b] if name]
        for team_key_col in ["ct_team_key", "t_team_key"]:
            team_key = str(row[team_key_col])
            candidate_counts.setdefault(team_key, {})
            match_counts.setdefault(team_key, set()).add(match_id)
            for candidate in candidates:
                candidate_counts[team_key][candidate] = candidate_counts[team_key].get(candidate, 0) + 1

    rows = []
    assigned_names: dict[str, str] = {}
    resolution_method: dict[str, str] = {}
    for team_key, counts in sorted(candidate_counts.items()):
        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        best_name, best_count = sorted_counts[0]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        match_total = len(match_counts[team_key])
        confidence = best_count / max(match_total, 1)
        resolved = best_count > second_count
        display_name = best_name if resolved else f"unresolved_{team_key[-6:]}"
        if resolved:
            assigned_names[team_key] = best_name
            resolution_method[team_key] = "vote"
        rows.append(
            {
                "team_key": team_key,
                "display_name": display_name,
                "best_name": best_name,
                "best_count": int(best_count),
                "second_count": int(second_count),
                "match_count": int(match_total),
                "confidence": float(confidence),
                "resolved": bool(resolved),
                "candidate_names": " | ".join([f"{name}:{count}" for name, count in sorted_counts]),
                "resolution_method": "vote" if resolved else "ambiguous_vote",
            }
        )

    changed = True
    while changed:
        changed = False
        for _, row in unique_matches.iterrows():
            match_id = str(row["match_id"])
            team_a, team_b = parse_match_names(match_id)
            if not team_a or not team_b:
                continue
            ct_key = str(row["ct_team_key"])
            t_key = str(row["t_team_key"])
            ct_name = assigned_names.get(ct_key)
            t_name = assigned_names.get(t_key)
            if ct_name and not t_name:
                if ct_name == team_a:
                    assigned_names[t_key] = team_b
                    resolution_method[t_key] = f"folder_constraint:{match_id}"
                    changed = True
                elif ct_name == team_b:
                    assigned_names[t_key] = team_a
                    resolution_method[t_key] = f"folder_constraint:{match_id}"
                    changed = True
            elif t_name and not ct_name:
                if t_name == team_a:
                    assigned_names[ct_key] = team_b
                    resolution_method[ct_key] = f"folder_constraint:{match_id}"
                    changed = True
                elif t_name == team_b:
                    assigned_names[ct_key] = team_a
                    resolution_method[ct_key] = f"folder_constraint:{match_id}"
                    changed = True

    team_map = pd.DataFrame(rows)
    if assigned_names:
        team_map["display_name"] = team_map["team_key"].map(lambda key: assigned_names.get(key, f"unresolved_{str(key)[-6:]}"))
        team_map["resolved"] = team_map["team_key"].map(lambda key: key in assigned_names)
        team_map["resolution_method"] = team_map["team_key"].map(lambda key: resolution_method.get(key, "ambiguous_vote"))

    mismatch_rows = []
    for _, row in unique_matches.iterrows():
        match_id = str(row["match_id"])
        team_a, team_b = parse_match_names(match_id)
        if not team_a or not team_b:
            continue
        ct_key = str(row["ct_team_key"])
        t_key = str(row["t_team_key"])
        ct_name = assigned_names.get(ct_key)
        t_name = assigned_names.get(t_key)
        if ct_name and t_name and {ct_name, t_name} != {team_a, team_b}:
            mismatch_rows.append(
                {
                    "match_id": match_id,
                    "folder_team_a": team_a,
                    "folder_team_b": team_b,
                    "ct_team_key": ct_key,
                    "ct_team_name": ct_name,
                    "t_team_key": t_key,
                    "t_team_name": t_name,
                }
            )

    mismatch_df = pd.DataFrame(mismatch_rows).drop_duplicates()
    return team_map, mismatch_df


def build_fingerprint_table(df: pd.DataFrame, cluster_col: str, group_cols: list[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + [cluster_col]).size().reset_index(name="count")
    totals = counts.groupby(group_cols)["count"].transform("sum")
    counts["ratio"] = counts["count"] / totals
    pivot = counts.pivot_table(index=group_cols, columns=cluster_col, values="ratio", fill_value=0.0)
    pivot.columns = [f"cluster_{int(col)}" for col in pivot.columns]
    pivot = pivot.reset_index()
    total_rows = df.groupby(group_cols).size().reset_index(name="round_count")
    return pivot.merge(total_rows, on=group_cols, how="left")


def ensure_cluster_columns(df: pd.DataFrame, cluster_ids: list[int]) -> pd.DataFrame:
    out = df.copy()
    for cluster_id in cluster_ids:
        col = f"cluster_{cluster_id}"
        if col not in out.columns:
            out[col] = 0.0
    ordered_cluster_cols = [f"cluster_{cluster_id}" for cluster_id in cluster_ids]
    other_cols = [col for col in out.columns if col not in ordered_cluster_cols]
    return out[other_cols + ordered_cluster_cols]


def top_places_string(group: pd.DataFrame, top_n: int = 5) -> str:
    counts = group["place"].value_counts().head(top_n)
    return ", ".join([f"{idx}:{cnt}" for idx, cnt in counts.items()])


def top_areas_string(group: pd.DataFrame) -> str:
    counts = group["area"].value_counts(normalize=True)
    return ", ".join([f"{idx}:{value:.3f}" for idx, value in counts.items()])


def cluster_ratio_vector(df: pd.DataFrame, cluster_col: str, cluster_ids: list[int]) -> np.ndarray:
    counts = df[cluster_col].value_counts(normalize=True)
    return np.array([float(counts.get(cluster_id, 0.0)) for cluster_id in cluster_ids], dtype=float)


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(name).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unknown"


def infer_tactical_name(area_dist: dict[str, float], top_places: list[str]) -> tuple[str, float, str]:
    """Draft a tactical name for a cluster from its 15 s area-occupancy ratios.

    This is a fixed threshold rule used as a starting point for naming the
    clusters; the returned score is a constant attached to each rule, not a
    rating by a human expert.
    """
    a_ratio = float(area_dist.get("A", 0.0))
    b_ratio = float(area_dist.get("B", 0.0))
    mid_ratio = float(area_dist.get("Mid", 0.0))
    ct_ratio = float(area_dist.get("CT", 0.0))

    if mid_ratio >= 0.28 and abs(a_ratio - b_ratio) <= 0.08:
        return "Mid-Control Balanced", 4.7, "High Mid share at 15 s with similar A/B shares: mid control with a balanced two-site setup."
    if a_ratio >= 0.68 and b_ratio <= 0.18:
        return "Aggressive A-Heavy Push", 4.8, "Clearly the highest A share and a low B share, typically forward or stacked positions such as LongA / Catwalk / UnderA."
    if a_ratio >= 0.56 and b_ratio >= 0.33:
        return "Anchor-A Biased", 4.4, "Strong anchors at both sites with a clear A majority: an A-leaning two-site setup."
    if b_ratio >= 0.40 and a_ratio >= 0.38:
        return "B-Biased Double-Point", 4.4, "Similar A/B shares with B slightly stronger: a two-site setup tilted towards B."
    if a_ratio >= 0.55 and mid_ratio >= 0.18:
        return "A-Long+Mid", 4.5, "A majority with a non-trivial Mid share: A is the main defence while MidDoors / TopMid control is kept."
    if ct_ratio >= 0.10:
        return "Retreat-and-Rotate", 4.1, "High CT-spawn share: a conservative, retracted setup prepared to rotate."
    return "Default Balanced", 3.8, f"Evenly spread setup; main positions: {', '.join(top_places[:3])}."


def dataframe_to_text_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train analysis for TAGNet outputs.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    analysis_dir = Path(f"{run_dir}_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    embeddings_df = pd.read_parquet(run_dir / "round_embeddings.parquet")
    rounds_df = pd.read_parquet(data_dir / "ct_round_samples.parquet")
    context_df = pd.read_parquet(data_dir / "ct_round_context.parquet")
    nodes_df = pd.read_parquet(data_dir / "ct_snapshot_nodes.parquet")
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))

    emb_cols = [col for col in embeddings_df.columns if col.startswith("emb_")]
    X = embeddings_df[emb_cols].to_numpy(dtype=float)
    weak_labels = [None if pd.isna(v) else str(v) for v in embeddings_df["weak_label_15s"].tolist()]

    k_rows = []
    best_labels = None
    best_centers = None
    best_k = None
    best_rank = None
    all_labels_by_k: dict[int, np.ndarray] = {}

    for k in range(args.min_k, args.max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        labels = km.fit_predict(X)
        all_labels_by_k[k] = labels
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        purity = weak_label_purity(weak_labels, labels)
        sizes = np.bincount(labels, minlength=k)
        k_rows.append(
            {
                "k": k,
                "silhouette": float(sil),
                "davies_bouldin": float(dbi),
                "calinski_harabasz": float(ch),
                "inertia": float(km.inertia_),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "weak_label_purity_at_15s": None if purity is None else float(purity),
                "cluster_sizes": ",".join(map(str, sizes.tolist())),
            }
        )

    k_df = pd.DataFrame(k_rows)
    k_df["rank_silhouette"] = k_df["silhouette"].rank(ascending=False, method="dense")
    k_df["rank_davies_bouldin"] = k_df["davies_bouldin"].rank(ascending=True, method="dense")
    k_df["rank_calinski_harabasz"] = k_df["calinski_harabasz"].rank(ascending=False, method="dense")
    k_df["avg_rank"] = k_df[["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]].mean(axis=1)
    best_rank_row = k_df.sort_values(["avg_rank", "rank_silhouette", "k"]).iloc[0]
    best_k = int(best_rank_row["k"])
    best_labels = all_labels_by_k[best_k]
    best_rank = float(best_rank_row["avg_rank"])
    cluster_ids = list(range(best_k))
    word_range_df = k_df[k_df["k"].between(5, 8)].copy()
    word_best_k = None
    if not word_range_df.empty:
        word_best_row = word_range_df.sort_values(["avg_rank", "rank_silhouette", "k"]).iloc[0]
        word_best_k = int(word_best_row["k"])

    best_km = KMeans(n_clusters=best_k, n_init=20, random_state=args.seed)
    best_km.fit(X)
    best_centers = best_km.cluster_centers_

    embeddings_best = embeddings_df.copy()
    embeddings_best["cluster_id"] = best_labels
    rounds_merged = rounds_df.merge(
        embeddings_best[["sample_id", "cluster_id", "match_id", "round_num", "split", "weak_label_15s"]],
        on=["sample_id", "match_id", "round_num", "split"],
        how="left",
    )
    context_15 = context_df[context_df["second"] == 15].copy()
    merged = rounds_merged.merge(
        context_15[
            [
                "sample_id",
                "ct_econ_type",
                "t_econ_type",
                "ct_total_equip",
                "t_total_equip",
                "ct_smokes_used",
                "ct_infernos_used",
            ]
        ],
        on="sample_id",
        how="left",
    )
    merged["score_state"] = merged["score_diff_ct_minus_t_before"].map(
        lambda v: "leading" if v > 0 else ("trailing" if v < 0 else "tied")
    )

    team_map, folder_mismatch_df = infer_team_names(rounds_df)
    merged = merged.merge(team_map[["team_key", "display_name", "confidence", "resolved"]], left_on="ct_team_key", right_on="team_key", how="left")
    merged = merged.rename(columns={"display_name": "ct_team_name", "confidence": "ct_team_confidence", "resolved": "ct_team_resolved"}).drop(columns=["team_key"])
    merged = merged.merge(team_map[["team_key", "display_name", "confidence", "resolved"]], left_on="t_team_key", right_on="team_key", how="left")
    merged = merged.rename(columns={"display_name": "t_team_name", "confidence": "t_team_confidence", "resolved": "t_team_resolved"}).drop(columns=["team_key"])
    merged["ct_win_flag"] = (merged["winner_side"] == "ct").astype(int)

    team_fingerprints = ensure_cluster_columns(
        build_fingerprint_table(merged, "cluster_id", ["ct_team_key", "ct_team_name"]),
        cluster_ids,
    )
    resolved_team_fingerprints = team_fingerprints[team_fingerprints["round_count"] >= 10].copy()

    similarity_rows = []
    for i, row_i in resolved_team_fingerprints.iterrows():
        for j, row_j in resolved_team_fingerprints.iterrows():
            similarity_rows.append(
                {
                    "team_i": row_i["ct_team_name"],
                    "team_j": row_j["ct_team_name"],
                    "cosine_distance": cosine_distance(
                        row_i.filter(regex=r"^cluster_").to_numpy(dtype=float),
                        row_j.filter(regex=r"^cluster_").to_numpy(dtype=float),
                    ),
                }
            )
    team_similarity = pd.DataFrame(similarity_rows)

    tactical_consistency_rows = []
    for cluster_id, group in merged.groupby("cluster_id"):
        team_win_rates = group.groupby("ct_team_name")["ct_win_flag"].mean()
        tactical_consistency_rows.append(
            {
                "cluster_id": int(cluster_id),
                "round_count": int(len(group)),
                "ct_win_rate": float(group["ct_win_flag"].mean()),
                "round_win_variance": float(group["ct_win_flag"].var(ddof=0)),
                "team_win_rate_variance": None if len(team_win_rates) < 2 else float(team_win_rates.var(ddof=0)),
                "team_count": int(group["ct_team_name"].nunique()),
            }
        )
    tactical_consistency_df = pd.DataFrame(tactical_consistency_rows).sort_values("cluster_id")

    score_shift_rows = []
    for (team_key, team_name), group in merged.groupby(["ct_team_key", "ct_team_name"]):
        states = {}
        for state, state_group in group.groupby("score_state"):
            if len(state_group) >= 5:
                states[state] = cluster_ratio_vector(state_group, "cluster_id", cluster_ids)
        if "leading" in states and "trailing" in states:
            score_shift_rows.append(
                {
                    "ct_team_key": team_key,
                    "ct_team_name": team_name,
                    "leading_vs_trailing_cosine_distance": cosine_distance(states["leading"], states["trailing"]),
                    "leading_rounds": int((group["score_state"] == "leading").sum()),
                    "trailing_rounds": int((group["score_state"] == "trailing").sum()),
                }
            )
    score_shift_df = pd.DataFrame(score_shift_rows)
    if not score_shift_df.empty:
        score_shift_df = score_shift_df.sort_values("leading_vs_trailing_cosine_distance", ascending=False)

    score_state_fingerprints = ensure_cluster_columns(
        build_fingerprint_table(merged, "cluster_id", ["ct_team_key", "ct_team_name", "score_state"]),
        cluster_ids,
    )
    leading_fp = score_state_fingerprints[score_state_fingerprints["score_state"] == "leading"].copy()
    trailing_fp = score_state_fingerprints[score_state_fingerprints["score_state"] == "trailing"].copy()
    h2_pairs = leading_fp.merge(
        trailing_fp,
        on=["ct_team_key", "ct_team_name"],
        suffixes=("_leading", "_trailing"),
        how="inner",
    )
    h2_rows = []
    for cluster_id in cluster_ids:
        lead_col = f"cluster_{cluster_id}_leading"
        trail_col = f"cluster_{cluster_id}_trailing"
        if lead_col not in h2_pairs.columns or trail_col not in h2_pairs.columns:
            continue
        lead_values = h2_pairs[lead_col].to_numpy(dtype=float)
        trail_values = h2_pairs[trail_col].to_numpy(dtype=float)
        if len(lead_values) >= 2:
            t_stat, p_value = ttest_rel(lead_values, trail_values, nan_policy="omit")
        else:
            t_stat, p_value = np.nan, np.nan
        h2_rows.append(
            {
                "cluster_id": int(cluster_id),
                "team_pair_count": int(len(lead_values)),
                "leading_mean_ratio": float(np.mean(lead_values)) if len(lead_values) else np.nan,
                "trailing_mean_ratio": float(np.mean(trail_values)) if len(trail_values) else np.nan,
                "mean_ratio_diff_leading_minus_trailing": float(np.mean(lead_values - trail_values)) if len(lead_values) else np.nan,
                "t_statistic": None if not np.isfinite(t_stat) else float(t_stat),
                "p_value": None if not np.isfinite(p_value) else float(p_value),
            }
        )
    h2_paired_ttest_df = pd.DataFrame(h2_rows)

    opponent_shift_rows = []
    for (team_key, team_name), group in merged.groupby(["ct_team_key", "ct_team_name"]):
        opponent_fps = {}
        for opponent_name, opp_group in group.groupby("t_team_name"):
            if len(opp_group) >= 5:
                opponent_fps[opponent_name] = cluster_ratio_vector(opp_group, "cluster_id", cluster_ids)
        opp_names = sorted(opponent_fps.keys())
        if len(opp_names) >= 2:
            distances = []
            for i in range(len(opp_names)):
                for j in range(i + 1, len(opp_names)):
                    distances.append(cosine_distance(opponent_fps[opp_names[i]], opponent_fps[opp_names[j]]))
            opponent_shift_rows.append(
                {
                    "ct_team_key": team_key,
                    "ct_team_name": team_name,
                    "opponent_count": len(opp_names),
                    "mean_pairwise_cosine_distance": float(np.mean(distances)),
                    "max_pairwise_cosine_distance": float(np.max(distances)),
                }
            )
    opponent_shift_df = pd.DataFrame(opponent_shift_rows)
    if not opponent_shift_df.empty:
        opponent_shift_df = opponent_shift_df.sort_values("mean_pairwise_cosine_distance", ascending=False)

    h3_rows = []
    h3_skipped_rows = []
    for (team_key, team_name), group in merged.groupby(["ct_team_key", "ct_team_name"]):
        filtered = group[group["t_team_name"].notna()].copy()
        opponent_counts = filtered["t_team_name"].value_counts()
        valid_opponents = opponent_counts[opponent_counts >= 5].index.tolist()
        filtered = filtered[filtered["t_team_name"].isin(valid_opponents)]
        contingency = pd.crosstab(filtered["t_team_name"], filtered["cluster_id"]).reindex(columns=cluster_ids, fill_value=0)
        contingency = contingency.loc[contingency.sum(axis=1) > 0, contingency.sum(axis=0) > 0]
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        try:
            chi2, p_value, dof, _ = chi2_contingency(contingency)
        except ValueError as exc:
            h3_skipped_rows.append(
                {
                    "ct_team_key": team_key,
                    "ct_team_name": team_name,
                    "reason": str(exc),
                }
            )
            continue
        contingency.to_csv(
            analysis_dir / f"h3_opponent_contingency_{sanitize_filename(team_name)}.csv",
            encoding="utf-8-sig",
        )
        h3_rows.append(
            {
                "ct_team_key": team_key,
                "ct_team_name": team_name,
                "opponent_count": int(contingency.shape[0]),
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "cramers_v": cramers_v(contingency),
            }
        )
    h3_opponent_chi2_df = pd.DataFrame(h3_rows)
    if not h3_opponent_chi2_df.empty:
        h3_opponent_chi2_df = h3_opponent_chi2_df.sort_values("p_value")
    h3_skipped_df = pd.DataFrame(h3_skipped_rows)

    context_test_rows = []
    for variable in ["score_state", "ct_econ_type", "prev_round_attack_site", "prev_round_winner_side"]:
        test_df = merged.copy()
        if variable == "prev_round_winner_side":
            test_df = test_df[test_df["prev_round_winner_side"].notna()]
        contingency = pd.crosstab(test_df[variable], test_df["cluster_id"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        context_test_rows.append(
            {
                "variable": variable,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "cramers_v": cramers_v(contingency),
                "n_rows": int(len(test_df)),
            }
        )
        contingency.to_csv(analysis_dir / f"context_distribution_{variable}.csv", encoding="utf-8-sig")
    context_tests_df = pd.DataFrame(context_test_rows)
    if not context_tests_df.empty:
        context_tests_df = context_tests_df.sort_values("p_value")

    cluster_profile_rows = []
    best_assignments = embeddings_best[["sample_id", "cluster_id"]]
    nodes_best = nodes_df.merge(best_assignments, on="sample_id", how="left")
    for (cluster_id, second), group in nodes_best.groupby(["cluster_id", "second"]):
        cluster_profile_rows.append(
            {
                "cluster_id": int(cluster_id),
                "second": int(second),
                "node_count": int(len(group)),
                "top_places": top_places_string(group),
                "area_distribution": top_areas_string(group),
            }
        )
    cluster_profiles_df = pd.DataFrame(cluster_profile_rows).sort_values(["cluster_id", "second"])

    representative_rows = []
    for cluster_id in range(best_k):
        mask = best_labels == cluster_id
        cluster_samples = embeddings_best.loc[mask].copy()
        cluster_vectors = cluster_samples[emb_cols].to_numpy(dtype=float)
        center = best_centers[cluster_id]
        dists = np.linalg.norm(cluster_vectors - center, axis=1)
        cluster_samples["distance_to_centroid"] = dists
        top = cluster_samples.nsmallest(5, "distance_to_centroid")
        for _, row in top.iterrows():
            representative_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "sample_id": row["sample_id"],
                    "match_id": row["match_id"],
                    "round_num": int(row["round_num"]),
                    "split": row["split"],
                    "distance_to_centroid": float(row["distance_to_centroid"]),
                }
            )
    representatives_df = pd.DataFrame(representative_rows).sort_values(["cluster_id", "distance_to_centroid"])

    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto", perplexity=30)
    tsne_xy = tsne.fit_transform(X)
    team_lookup = merged[["sample_id", "ct_team_name", "ct_team_resolved"]].drop_duplicates(subset=["sample_id"])
    tsne_df = embeddings_best[["sample_id", "match_id", "round_num", "split", "cluster_id", "weak_label_15s"]].copy()
    tsne_df = tsne_df.merge(team_lookup, on="sample_id", how="left")
    tsne_df["tsne_x"] = tsne_xy[:, 0]
    tsne_df["tsne_y"] = tsne_xy[:, 1]

    expert_rows = []
    for cluster_id in cluster_ids:
        cluster_nodes_15 = nodes_best[(nodes_best["cluster_id"] == cluster_id) & (nodes_best["second"] == 15)].copy()
        area_dist = cluster_nodes_15["area"].value_counts(normalize=True).to_dict()
        top_places = cluster_nodes_15["place"].value_counts().head(5)
        tactical_name, score, comment = infer_tactical_name(area_dist, top_places.index.tolist())
        representative_ids = representatives_df[representatives_df["cluster_id"] == cluster_id]["sample_id"].head(3).tolist()
        expert_rows.append(
            {
                "cluster_id": int(cluster_id),
                "tactical_name": tactical_name,
                "rule_prior_score_5": float(score),
                "comment": comment,
                "top_places_15s": ", ".join([f"{idx}:{cnt}" for idx, cnt in top_places.items()]),
                "area_ratio_15s": ", ".join([f"{area}:{area_dist.get(area, 0.0):.3f}" for area in ["A", "B", "Mid", "CT"]]),
                "representative_rounds": " | ".join(representative_ids),
            }
        )
    expert_validation_df = pd.DataFrame(expert_rows).sort_values("cluster_id")

    k_df.to_csv(analysis_dir / "k_selection_metrics.csv", index=False, encoding="utf-8-sig")
    team_map.to_csv(analysis_dir / "team_name_inference.csv", index=False, encoding="utf-8-sig")
    folder_mismatch_df.to_csv(analysis_dir / "folder_name_mismatches.csv", index=False, encoding="utf-8-sig")
    team_fingerprints.to_csv(analysis_dir / "team_style_fingerprints.csv", index=False, encoding="utf-8-sig")
    team_similarity.to_csv(analysis_dir / "team_style_similarity.csv", index=False, encoding="utf-8-sig")
    score_shift_df.to_csv(analysis_dir / "score_state_style_shift.csv", index=False, encoding="utf-8-sig")
    h2_paired_ttest_df.to_csv(analysis_dir / "h2_score_state_paired_ttest.csv", index=False, encoding="utf-8-sig")
    h2_pairs.to_csv(analysis_dir / "h2_score_state_team_pairs.csv", index=False, encoding="utf-8-sig")
    opponent_shift_df.to_csv(analysis_dir / "opponent_style_shift.csv", index=False, encoding="utf-8-sig")
    h3_opponent_chi2_df.to_csv(analysis_dir / "h3_opponent_chi_square.csv", index=False, encoding="utf-8-sig")
    h3_skipped_df.to_csv(analysis_dir / "h3_opponent_chi_square_skipped.csv", index=False, encoding="utf-8-sig")
    context_tests_df.to_csv(analysis_dir / "context_trigger_tests.csv", index=False, encoding="utf-8-sig")
    tactical_consistency_df.to_csv(analysis_dir / "tactical_consistency.csv", index=False, encoding="utf-8-sig")
    expert_validation_df.to_csv(analysis_dir / "cluster_naming_heuristic.csv", index=False, encoding="utf-8-sig")
    cluster_profiles_df.to_csv(analysis_dir / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    representatives_df.to_csv(analysis_dir / "cluster_representatives.csv", index=False, encoding="utf-8-sig")
    tsne_df.to_parquet(analysis_dir / "tsne_coordinates.parquet", index=False)
    embeddings_best.to_parquet(analysis_dir / "analysis_cluster_assignments_bestk.parquet", index=False)

    summary = {
        "source_run_dir": str(run_dir),
        "reference_best_k_from_training": int(run_summary["best_k_by_silhouette"]),
        "recommended_k_from_multi_metric_analysis": int(best_k),
        "recommended_k_in_word_expected_range_5_to_8": None if word_best_k is None else int(word_best_k),
        "recommended_k_avg_rank": best_rank,
        "best_k_silhouette": float(k_df.loc[k_df["k"] == best_k, "silhouette"].iloc[0]),
        "best_k_davies_bouldin": float(k_df.loc[k_df["k"] == best_k, "davies_bouldin"].iloc[0]),
        "best_k_calinski_harabasz": float(k_df.loc[k_df["k"] == best_k, "calinski_harabasz"].iloc[0]),
        "team_fingerprint_count": int(len(team_fingerprints)),
        "resolved_team_fingerprint_count_ge10_rounds": int(len(resolved_team_fingerprints)),
        "folder_name_mismatch_count": int(len(folder_mismatch_df)),
        "score_shift_team_count": int(len(score_shift_df)),
        "opponent_shift_team_count": int(len(opponent_shift_df)),
        "h2_team_pair_count": int(len(h2_pairs)),
        "h2_significant_cluster_count_p_lt_0_05": int((h2_paired_ttest_df["p_value"] < 0.05).sum()) if not h2_paired_ttest_df.empty else 0,
        "h3_significant_team_count_p_lt_0_05": int((h3_opponent_chi2_df["p_value"] < 0.05).sum()) if not h3_opponent_chi2_df.empty else 0,
        "h3_skipped_team_count": int(len(h3_skipped_df)),
        "tactical_consistency_cluster_count": int(len(tactical_consistency_df)),
        "naming_heuristic_cluster_count": int(len(expert_validation_df)),
    }
    with (analysis_dir / "analysis_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    markdown_lines = [
        "# TAGNet Analysis Summary",
        "",
        f"- Source run: `{run_dir.name}`",
        f"- Training-best K (silhouette on saved run): `{run_summary['best_k_by_silhouette']}`",
        f"- Multi-metric recommended K: `{best_k}`",
        f"- Word-range recommended K (5~8): `{word_best_k}`",
        f"- Teams with fingerprints: `{len(team_fingerprints)}`",
        f"- Teams with >=10 CT rounds: `{len(resolved_team_fingerprints)}`",
        f"- Folder-name mismatches flagged: `{len(folder_mismatch_df)}`",
        f"- Teams with leading/trailing comparison: `{len(score_shift_df)}`",
        "",
        "## Top K Candidates",
        "",
    ]
    top_k = k_df.sort_values(["avg_rank", "rank_silhouette"]).head(5)
    markdown_lines.append("```text")
    markdown_lines.append(dataframe_to_text_table(top_k))
    markdown_lines.append("```")
    markdown_lines.append("")
    markdown_lines.append("## Context Trigger Tests")
    markdown_lines.append("")
    if not context_tests_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(context_tests_df))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No valid context-trigger test tables were produced.")
    markdown_lines.append("")
    markdown_lines.append("## Tactical Consistency")
    markdown_lines.append("")
    if not tactical_consistency_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(tactical_consistency_df))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No tactical consistency table available.")
    markdown_lines.append("")
    markdown_lines.append("## H2 Paired T-Test")
    markdown_lines.append("")
    if not h2_paired_ttest_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(h2_paired_ttest_df))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No H2 paired t-test rows available.")
    markdown_lines.append("")
    markdown_lines.append("## H3 Opponent Chi-Square")
    markdown_lines.append("")
    if not h3_opponent_chi2_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(h3_opponent_chi2_df))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No H3 opponent chi-square rows available.")
    markdown_lines.append("")
    markdown_lines.append("## Rule-Based Cluster Naming Draft")
    markdown_lines.append("")
    if not expert_validation_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(expert_validation_df))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No cluster naming draft available.")
    markdown_lines.append("")
    markdown_lines.append("## Cluster Representatives")
    markdown_lines.append("")
    if not representatives_df.empty:
        markdown_lines.append("```text")
        markdown_lines.append(dataframe_to_text_table(representatives_df.head(20)))
        markdown_lines.append("```")
    else:
        markdown_lines.append("No representative rounds available.")
    (analysis_dir / "analysis_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(str(analysis_dir))


if __name__ == "__main__":
    main()
