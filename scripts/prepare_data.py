from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "parsed"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
TICK_RATE = 64
DEFAULT_SNAPSHOT_SECONDS = (10, 15, 20)
SNAPSHOT_SECONDS = DEFAULT_SNAPSHOT_SECONDS
RANDOM_SEED = 42
COACH_MARKER = "Coach|"


PLACE_CANONICAL_MAP = {
    "TopofMid": "TopMid",
}

AREA_TO_PLACES = {
    "A": {
        "BombsiteA",
        "ARamp",
        "LongA",
        "LongDoors",
        "ExtendedA",
        "Catwalk",
        "ShortStairs",
        "Pit",
        "OutsideLong",
        "UnderA",
    },
    "B": {
        "BombsiteB",
        "BDoors",
        "LowerTunnel",
        "UpperTunnel",
        "OutsideTunnel",
        "TunnelStairs",
    },
    "Mid": {
        "MidDoors",
        "Middle",
        "TopMid",
        "Suicide",
        "Xbox",
        "Hole",
    },
    "CT": {
        "CTSpawn",
        "Side",
    },
}

AREA_ORDER = {
    "A": 0,
    "B": 1,
    "Mid": 2,
    "CT": 3,
    "other": 4,
}

VALID_DUST2_MAP = "de_dust2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TAGNet training data from parsed match folders.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--snapshot-seconds", nargs="+", type=int, default=list(DEFAULT_SNAPSHOT_SECONDS))
    return parser.parse_args()


def stable_hash(values: list[str], prefix: str) -> str:
    joined = "|".join(sorted(values))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def canonical_place(place: Any) -> str:
    if place is None or (isinstance(place, float) and math.isnan(place)):
        return "UNKNOWN"
    place_str = str(place).strip()
    if not place_str:
        return "UNKNOWN"
    return PLACE_CANONICAL_MAP.get(place_str, place_str)


def map_area(place: Any) -> str:
    place_name = canonical_place(place)
    for area, places in AREA_TO_PLACES.items():
        if place_name in places:
            return area
    return "other"


def parse_folder_teams(folder_name: str) -> tuple[str | None, str | None]:
    base = folder_name
    if base.endswith("_dust2"):
        base = base[: -len("_dust2")]
    if "-vs-" not in base:
        return None, None
    left, right = base.split("-vs-", 1)
    return left or None, right or None


def economy_bucket(total_equip: float, pistol_round: bool) -> str:
    if pistol_round:
        return "pistol"
    if total_equip < 10000:
        return "eco"
    if total_equip < 25000:
        return "half_buy"
    return "long_gun"


def one_hot(place: str, vocab: list[str], index_map: dict[str, int]) -> list[int]:
    vec = [0] * len(vocab)
    idx = index_map[place]
    vec[idx] = 1
    return vec


def assign_match_splits(match_ids: list[str]) -> dict[str, str]:
    rng = random.Random(RANDOM_SEED)
    ordered = sorted(match_ids)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    split_map: dict[str, str] = {}
    for idx, match_id in enumerate(ordered):
        if idx < n_train:
            split_map[match_id] = "train"
        elif idx < n_train + n_val:
            split_map[match_id] = "val"
        else:
            split_map[match_id] = "test"
    return split_map


def anonymize_players(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace SteamIDs and in-game names by stable ids (player_001, ...) ordered by SteamID."""
    steamids = sorted(
        set(nodes_df["steamid"].astype("int64"))
        | set(edges_df["src_steamid"].astype("int64"))
        | set(edges_df["dst_steamid"].astype("int64"))
    )
    mapping = {sid: f"player_{i:03d}" for i, sid in enumerate(steamids, start=1)}
    nodes = nodes_df.copy()
    nodes.insert(nodes.columns.get_loc("steamid"), "player_id", nodes["steamid"].astype("int64").map(mapping))
    nodes = nodes.drop(columns=["steamid", "name"])
    edges = edges_df.copy()
    position = edges.columns.get_loc("src_steamid")
    edges.insert(position, "src_player_id", edges["src_steamid"].astype("int64").map(mapping))
    edges.insert(position + 1, "dst_player_id", edges["dst_steamid"].astype("int64").map(mapping))
    edges = edges.drop(columns=["src_steamid", "dst_steamid", "src_name", "dst_name"])
    return nodes, edges


def prepare_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_header(match_dir: Path) -> dict[str, Any]:
    with (match_dir / "header.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_parquet(match_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(match_dir / name, columns=columns).to_pandas()


def build_match_manifest(match_dirs: list[Path]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for match_dir in sorted(match_dirs):
        header = load_header(match_dir)
        folder_team_a, folder_team_b = parse_folder_teams(match_dir.name)
        row = {
            "match_id": match_dir.name,
            "folder_name": match_dir.name,
            "folder_team_a": folder_team_a,
            "folder_team_b": folder_team_b,
            "map_name": header.get("map_name"),
            "patch_version": str(header.get("patch_version")),
            "server_name": header.get("server_name"),
            "include_in_model": header.get("map_name") == VALID_DUST2_MAP,
            "exclusion_reason": None if header.get("map_name") == VALID_DUST2_MAP else "non_dust2_map",
        }
        if (match_dir / "rounds.parquet").exists():
            row["round_count"] = pq.ParquetFile(match_dir / "rounds.parquet").metadata.num_rows
        else:
            row["round_count"] = 0
            row["include_in_model"] = False
            row["exclusion_reason"] = "missing_rounds"
        manifest_rows.append(row)
    return manifest_rows


def prepare_match(
    match_dir: Path,
    weak_label_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    header = load_header(match_dir)
    rounds = read_parquet(match_dir, "rounds.parquet")

    roster_ticks = read_parquet(match_dir, "ticks.parquet", columns=["round_num", "side", "steamid", "name"])
    roster_ticks = roster_ticks[~roster_ticks["name"].astype(str).str.contains(COACH_MARKER, regex=False, na=False)].copy()
    roster_ticks = roster_ticks[roster_ticks["side"].isin(["ct", "t"])].copy()
    roster_ticks = roster_ticks.drop_duplicates(subset=["round_num", "side", "steamid", "name"])

    lineup_by_round_side: dict[tuple[int, str], dict[str, Any]] = {}
    grouped = roster_ticks.groupby(["round_num", "side"], sort=True)
    for (round_num, side), df in grouped:
        steamids = sorted(str(int(v)) for v in df["steamid"].dropna().astype("int64").tolist())
        names = sorted(str(v) for v in df["name"].dropna().tolist())
        lineup_by_round_side[(int(round_num), side)] = {
            "steamids": steamids,
            "names": names,
            "team_key": stable_hash(steamids, "team"),
        }

    round_target_ticks: dict[int, dict[int, int]] = {}
    candidate_ticks: set[int] = set()
    for row in rounds.itertuples(index=False):
        round_num = int(row.round_num)
        if pd.isna(row.freeze_end):
            # Some demos start recording during the first freeze time; such rounds have no anchor tick.
            continue
        tick_map: dict[int, int] = {}
        for second in SNAPSHOT_SECONDS:
            target_tick = int(row.freeze_end) + second * TICK_RATE
            tick_map[second] = target_tick
            candidate_ticks.add(target_tick)
        round_target_ticks[round_num] = tick_map

    snapshot_ticks = read_parquet(
        match_dir,
        "ticks.parquet",
        columns=[
            "round_num",
            "tick",
            "side",
            "steamid",
            "name",
            "place",
            "X",
            "Y",
            "Z",
            "health",
            "armor",
            "current_equip_value",
            "has_defuser",
            "has_helmet",
        ],
    )
    snapshot_ticks = snapshot_ticks[~snapshot_ticks["name"].astype(str).str.contains(COACH_MARKER, regex=False, na=False)].copy()
    snapshot_ticks = snapshot_ticks[snapshot_ticks["tick"].isin(candidate_ticks)].copy()
    snapshot_ticks = snapshot_ticks[snapshot_ticks["side"].isin(["ct", "t"])].copy()
    snapshot_ticks["place_canonical"] = snapshot_ticks["place"].map(canonical_place)
    snapshot_ticks["area"] = snapshot_ticks["place_canonical"].map(map_area)
    snapshot_ticks["steamid"] = snapshot_ticks["steamid"].astype("int64")

    smokes = read_parquet(
        match_dir,
        "smokes.parquet",
        columns=["round_num", "start_tick", "thrower_side", "thrower_place", "entity_id"],
    )
    smokes["throw_area"] = smokes["thrower_place"].map(map_area)

    infernos = read_parquet(
        match_dir,
        "infernos.parquet",
        columns=["round_num", "start_tick", "thrower_side", "thrower_place", "entity_id"],
    )
    infernos["throw_area"] = infernos["thrower_place"].map(map_area)

    t_20_area_counts: dict[int, dict[str, int]] = {}
    valid_rounds: list[int] = []

    for row in rounds.itertuples(index=False):
        round_num = int(row.round_num)
        if header.get("map_name") != VALID_DUST2_MAP:
            continue
        if (round_num, "ct") not in lineup_by_round_side or (round_num, "t") not in lineup_by_round_side:
            continue
        if round_num not in round_target_ticks:
            continue
        usable = True
        for second in SNAPSHOT_SECONDS:
            target_tick = round_target_ticks[round_num][second]
            if target_tick > int(row.end):
                usable = False
                break
            ct_rows = snapshot_ticks[
                (snapshot_ticks["round_num"] == round_num)
                & (snapshot_ticks["tick"] == target_tick)
                & (snapshot_ticks["side"] == "ct")
            ]
            if ct_rows["steamid"].nunique() != 5:
                usable = False
                break
        if not usable:
            continue
        valid_rounds.append(round_num)
        t_target = round_target_ticks[round_num][20]
        t_rows = snapshot_ticks[
            (snapshot_ticks["round_num"] == round_num)
            & (snapshot_ticks["tick"] == t_target)
            & (snapshot_ticks["side"] == "t")
        ]
        counts = t_rows["area"].value_counts().to_dict()
        t_20_area_counts[round_num] = {area: int(counts.get(area, 0)) for area in ["A", "B", "Mid", "CT", "other"]}

    round_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []

    score_by_team: defaultdict[str, int] = defaultdict(int)

    for row in rounds.sort_values("round_num").itertuples(index=False):
        round_num = int(row.round_num)
        ct_lineup = lineup_by_round_side.get((round_num, "ct"))
        t_lineup = lineup_by_round_side.get((round_num, "t"))
        if not ct_lineup or not t_lineup:
            continue

        ct_team_key = ct_lineup["team_key"]
        t_team_key = t_lineup["team_key"]
        ct_team_score_before = int(score_by_team[ct_team_key])
        t_team_score_before = int(score_by_team[t_team_key])

        if round_num in valid_rounds:
            pistol_round = round_num in {1, 13}
            prev_round_num = round_num - 1
            prev_round_winner_side = None
            prev_round_attack_site = "unknown"
            if prev_round_num >= 1:
                prev_round_row = rounds[rounds["round_num"] == prev_round_num]
                if not prev_round_row.empty:
                    prev_round = prev_round_row.iloc[0]
                    prev_round_winner_side = prev_round["winner"]
                    bomb_site = str(prev_round["bomb_site"]).lower()
                    if bomb_site == "bombsite_a":
                        prev_round_attack_site = "A"
                    elif bomb_site == "bombsite_b":
                        prev_round_attack_site = "B"
                    else:
                        counts = t_20_area_counts.get(prev_round_num, {})
                        a_count = counts.get("A", 0)
                        b_count = counts.get("B", 0)
                        if a_count > b_count and a_count > 0:
                            prev_round_attack_site = "A"
                        elif b_count > a_count and b_count > 0:
                            prev_round_attack_site = "B"
                        elif counts.get("Mid", 0) > 0:
                            prev_round_attack_site = "Mid"
                        else:
                            prev_round_attack_site = "unknown"

            sample_id = f"{match_dir.name}__r{round_num:02d}"
            round_rows.append(
                {
                    "sample_id": sample_id,
                    "match_id": match_dir.name,
                    "round_num": round_num,
                    "start_tick": int(row.start),
                    "freeze_end_tick": int(row.freeze_end),
                    "end_tick": int(row.end),
                    "official_end_tick": int(row.official_end),
                    **{f"snapshot_tick_{second}": round_target_ticks[round_num][second] for second in SNAPSHOT_SECONDS},
                    "winner_side": row.winner,
                    "round_end_reason": row.reason,
                    "bomb_site": row.bomb_site,
                    "ct_team_key": ct_team_key,
                    "t_team_key": t_team_key,
                    "ct_team_score_before": ct_team_score_before,
                    "t_team_score_before": t_team_score_before,
                    "score_diff_ct_minus_t_before": ct_team_score_before - t_team_score_before,
                    "pistol_round_flag": pistol_round,
                    "prev_round_winner_side": prev_round_winner_side,
                    "prev_round_attack_site": prev_round_attack_site,
                }
            )

            for second in SNAPSHOT_SECONDS:
                target_tick = round_target_ticks[round_num][second]
                tick_rows = snapshot_ticks[
                    (snapshot_ticks["round_num"] == round_num) & (snapshot_ticks["tick"] == target_tick)
                ].copy()
                ct_rows = tick_rows[tick_rows["side"] == "ct"].copy()
                t_rows = tick_rows[tick_rows["side"] == "t"].copy()

                ct_rows["area_sort"] = ct_rows["area"].map(lambda x: AREA_ORDER.get(x, AREA_ORDER["other"]))
                ct_rows = ct_rows.sort_values(
                    by=["area_sort", "place_canonical", "Y", "X", "steamid"],
                    ascending=[True, True, True, True, True],
                ).reset_index(drop=True)
                ct_rows["player_slot"] = np.arange(len(ct_rows), dtype=int)

                ct_total_equip = float(ct_rows["current_equip_value"].fillna(0).sum())
                t_total_equip = float(t_rows["current_equip_value"].fillna(0).sum())

                smoke_slice = smokes[
                    (smokes["round_num"] == round_num)
                    & (smokes["thrower_side"] == "ct")
                    & (smokes["start_tick"] <= target_tick)
                ]
                inferno_slice = infernos[
                    (infernos["round_num"] == round_num)
                    & (infernos["thrower_side"] == "ct")
                    & (infernos["start_tick"] <= target_tick)
                ]

                smoke_counts = smoke_slice["throw_area"].value_counts().to_dict()
                inferno_counts = inferno_slice["throw_area"].value_counts().to_dict()
                weak_label = weak_label_cache.get((match_dir.name, round_num, second), {})

                context_rows.append(
                    {
                        "sample_id": sample_id,
                        "match_id": match_dir.name,
                        "round_num": round_num,
                        "second": second,
                        "tick": target_tick,
                        "ct_team_key": ct_team_key,
                        "t_team_key": t_team_key,
                        "ct_team_score_before": ct_team_score_before,
                        "t_team_score_before": t_team_score_before,
                        "score_diff_ct_minus_t_before": ct_team_score_before - t_team_score_before,
                        "pistol_round_flag": pistol_round,
                        "prev_round_winner_side": prev_round_winner_side,
                        "prev_round_attack_site": prev_round_attack_site,
                        "ct_total_equip": ct_total_equip,
                        "t_total_equip": t_total_equip,
                        "ct_econ_type": economy_bucket(ct_total_equip, pistol_round),
                        "t_econ_type": economy_bucket(t_total_equip, pistol_round),
                        "ct_smokes_used": int(smoke_slice.shape[0]),
                        "ct_infernos_used": int(inferno_slice.shape[0]),
                        "ct_smoke_throw_area_A": int(smoke_counts.get("A", 0)),
                        "ct_smoke_throw_area_B": int(smoke_counts.get("B", 0)),
                        "ct_smoke_throw_area_Mid": int(smoke_counts.get("Mid", 0)),
                        "ct_smoke_throw_area_CT": int(smoke_counts.get("CT", 0)),
                        "ct_inferno_throw_area_A": int(inferno_counts.get("A", 0)),
                        "ct_inferno_throw_area_B": int(inferno_counts.get("B", 0)),
                        "ct_inferno_throw_area_Mid": int(inferno_counts.get("Mid", 0)),
                        "ct_inferno_throw_area_CT": int(inferno_counts.get("CT", 0)),
                        "weak_label": weak_label.get("label"),
                        "weak_players_A": weak_label.get("players_A"),
                        "weak_players_B": weak_label.get("players_B"),
                        "weak_players_Mid": weak_label.get("players_mid"),
                        "weak_equip_value": weak_label.get("equip_value"),
                    }
                )

                for ct_row in ct_rows.itertuples(index=False):
                    node_rows.append(
                        {
                            "sample_id": sample_id,
                            "match_id": match_dir.name,
                            "round_num": round_num,
                            "second": second,
                            "tick": target_tick,
                            "player_slot": int(ct_row.player_slot),
                            "steamid": int(ct_row.steamid),
                            "name": ct_row.name,
                            "place": ct_row.place_canonical,
                            "area": ct_row.area,
                            "X": float(ct_row.X),
                            "Y": float(ct_row.Y),
                            "Z": float(ct_row.Z),
                            "health": int(ct_row.health),
                            "armor": int(ct_row.armor),
                            "current_equip_value": int(ct_row.current_equip_value),
                            "has_defuser": bool(ct_row.has_defuser),
                            "has_helmet": bool(ct_row.has_helmet),
                            "ct_team_key": ct_team_key,
                        }
                    )

                ct_subset = ct_rows[["player_slot", "X", "Y", "steamid", "name"]].copy()
                for i in range(len(ct_subset)):
                    for j in range(i + 1, len(ct_subset)):
                        row_i = ct_subset.iloc[i]
                        row_j = ct_subset.iloc[j]
                        dist_xy = math.dist((float(row_i["X"]), float(row_i["Y"])), (float(row_j["X"]), float(row_j["Y"])))
                        edge_rows.append(
                            {
                                "sample_id": sample_id,
                                "match_id": match_dir.name,
                                "round_num": round_num,
                                "second": second,
                                "tick": target_tick,
                                "src_slot": int(row_i["player_slot"]),
                                "dst_slot": int(row_j["player_slot"]),
                                "src_steamid": int(row_i["steamid"]),
                                "dst_steamid": int(row_j["steamid"]),
                                "src_name": row_i["name"],
                                "dst_name": row_j["name"],
                                "distance_xy": dist_xy,
                            }
                        )

        winner_side = str(row.winner)
        if winner_side == "ct":
            score_by_team[ct_team_key] += 1
        elif winner_side == "t":
            score_by_team[t_team_key] += 1

    match_summary = {
        "match_id": match_dir.name,
        "map_name": header.get("map_name"),
        "round_count_total": int(rounds.shape[0]),
        "round_count_valid": int(len(valid_rounds)),
    }
    return round_rows, node_rows, edge_rows, context_rows, match_summary


def main() -> None:
    global DATASET_ROOT, OUTPUT_ROOT, SNAPSHOT_SECONDS
    args = parse_args()
    DATASET_ROOT = Path(args.dataset_root)
    OUTPUT_ROOT = Path(args.output_root)
    SNAPSHOT_SECONDS = tuple(sorted(dict.fromkeys(int(v) for v in args.snapshot_seconds)))
    if 15 not in SNAPSHOT_SECONDS:
        raise ValueError("snapshot seconds must include 15 so weak-label alignment remains available.")

    prepare_output_dirs()

    match_dirs = [p for p in DATASET_ROOT.iterdir() if p.is_dir()]
    manifest_rows = build_match_manifest(match_dirs)
    manifest_df = pd.DataFrame(manifest_rows).sort_values("match_id").reset_index(drop=True)

    weak_label_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    for match_dir in match_dirs:
        label_path = match_dir / "defense_labels.parquet"
        if label_path.exists():
            label_rows = pq.ParquetFile(label_path).metadata.num_rows
            if label_rows > 0:
                labels = read_parquet(match_dir, "defense_labels.parquet")
                for row in labels.itertuples(index=False):
                    weak_label_cache[(match_dir.name, int(row.round_num), int(row.second))] = {
                        "label": row.label,
                        "players_A": int(row.players_A),
                        "players_B": int(row.players_B),
                        "players_mid": int(row.players_mid),
                        "equip_value": int(row.equip_value),
                    }

    included_match_dirs = [
        p for p in match_dirs if manifest_df.loc[manifest_df["match_id"] == p.name, "include_in_model"].iloc[0]
    ]
    split_map = assign_match_splits([p.name for p in included_match_dirs])
    manifest_df["split"] = manifest_df["match_id"].map(split_map)

    all_round_rows: list[dict[str, Any]] = []
    all_node_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_context_rows: list[dict[str, Any]] = []
    match_summaries: list[dict[str, Any]] = []

    for match_dir in included_match_dirs:
        round_rows, node_rows, edge_rows, context_rows, match_summary = prepare_match(match_dir, weak_label_cache)
        all_round_rows.extend(round_rows)
        all_node_rows.extend(node_rows)
        all_edge_rows.extend(edge_rows)
        all_context_rows.extend(context_rows)
        match_summaries.append(match_summary)

    rounds_df = pd.DataFrame(all_round_rows).sort_values(["match_id", "round_num"]).reset_index(drop=True)
    nodes_df = pd.DataFrame(all_node_rows).sort_values(["match_id", "round_num", "second", "player_slot"]).reset_index(drop=True)
    edges_df = pd.DataFrame(all_edge_rows).sort_values(["match_id", "round_num", "second", "src_slot", "dst_slot"]).reset_index(drop=True)
    context_df = pd.DataFrame(all_context_rows).sort_values(["match_id", "round_num", "second"]).reset_index(drop=True)
    match_summary_df = pd.DataFrame(match_summaries).sort_values("match_id").reset_index(drop=True)

    for df in [rounds_df, nodes_df, edges_df, context_df]:
        df["split"] = df["match_id"].map(split_map)

    place_vocab = sorted(nodes_df["place"].dropna().unique().tolist(), key=lambda place: (AREA_ORDER.get(map_area(place), 99), place))
    place_to_idx = {place: idx for idx, place in enumerate(place_vocab)}
    nodes_df["place_idx"] = nodes_df["place"].map(place_to_idx)
    nodes_df["health_norm"] = (nodes_df["health"].astype(float) / 100.0).clip(0.0, 1.0)

    train_nodes = nodes_df[nodes_df["split"] == "train"].copy()
    train_context = context_df[context_df["split"] == "train"].copy()
    train_edges = edges_df[edges_df["split"] == "train"].copy()

    train_node_equip_log = np.log1p(train_nodes["current_equip_value"].astype(float))
    node_equip_min = float(train_node_equip_log.min())
    node_equip_max = float(train_node_equip_log.max())
    node_equip_den = max(node_equip_max - node_equip_min, 1e-9)

    nodes_df["equip_log1p"] = np.log1p(nodes_df["current_equip_value"].astype(float))
    nodes_df["equip_norm"] = ((nodes_df["equip_log1p"] - node_equip_min) / node_equip_den).clip(0.0, 1.0)

    train_team_equip_log = np.log1p(
        pd.concat(
            [train_context["ct_total_equip"].astype(float), train_context["t_total_equip"].astype(float)],
            ignore_index=True,
        )
    )
    team_equip_min = float(train_team_equip_log.min())
    team_equip_max = float(train_team_equip_log.max())
    team_equip_den = max(team_equip_max - team_equip_min, 1e-9)

    context_df["ct_total_equip_log1p"] = np.log1p(context_df["ct_total_equip"].astype(float))
    context_df["t_total_equip_log1p"] = np.log1p(context_df["t_total_equip"].astype(float))
    context_df["ct_total_equip_norm"] = (
        (context_df["ct_total_equip_log1p"] - team_equip_min) / team_equip_den
    ).clip(0.0, 1.0)
    context_df["t_total_equip_norm"] = (
        (context_df["t_total_equip_log1p"] - team_equip_min) / team_equip_den
    ).clip(0.0, 1.0)

    edge_dist_max = float(train_edges["distance_xy"].max())
    edge_dist_den = max(edge_dist_max, 1e-9)
    edges_df["distance_xy_norm"] = (edges_df["distance_xy"].astype(float) / edge_dist_den).clip(0.0, 1.0)

    nodes_df["place_onehot"] = nodes_df["place"].map(lambda place: one_hot(place, place_vocab, place_to_idx))
    nodes_df["node_feature_vector"] = nodes_df.apply(
        lambda row: row["place_onehot"] + [float(row["health_norm"]), float(row["equip_norm"])],
        axis=1,
    )

    round_vector_rows: list[dict[str, Any]] = []
    for sample_id, sample_nodes in nodes_df.groupby("sample_id", sort=True):
        sample_meta = sample_nodes.iloc[0]
        row_out: dict[str, Any] = {
            "sample_id": sample_id,
            "match_id": sample_meta["match_id"],
            "round_num": int(sample_meta["round_num"]),
            "split": sample_meta["split"],
        }
        sequence_vector: list[int] = []
        for second in SNAPSHOT_SECONDS:
            sec_nodes = sample_nodes[sample_nodes["second"] == second].sort_values("player_slot")
            snapshot_vector: list[int] = []
            for node in sec_nodes.itertuples(index=False):
                snapshot_vector.extend(node.place_onehot)
            row_out[f"snapshot_{second}_place_vector"] = snapshot_vector
            sequence_vector.extend(snapshot_vector)
        row_out["sequence_place_vector"] = sequence_vector
        round_vector_rows.append(row_out)

    round_vectors_df = pd.DataFrame(round_vector_rows).sort_values(["match_id", "round_num"]).reset_index(drop=True)

    manifest_df = manifest_df.merge(match_summary_df, on=["match_id", "map_name"], how="left")

    normalization_stats = {
        "tick_rate": TICK_RATE,
        "snapshot_seconds": list(SNAPSHOT_SECONDS),
        "node_health": {"method": "min_max_fixed", "min": 0.0, "max": 100.0},
        "node_equip": {
            "method": "log1p_then_min_max_train_only",
            "log_min_train": node_equip_min,
            "log_max_train": node_equip_max,
        },
        "team_total_equip": {
            "method": "log1p_then_min_max_train_only",
            "log_min_train": team_equip_min,
            "log_max_train": team_equip_max,
        },
        "edge_distance_xy": {
            "method": "min_max_train_only",
            "min_train": 0.0,
            "max_train": edge_dist_max,
        },
        "place_vocab_size": len(place_vocab),
        "place_vocab": place_vocab,
    }

    with (OUTPUT_ROOT / "normalization_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(normalization_stats, fh, indent=2, ensure_ascii=False)

    with (OUTPUT_ROOT / "place_vocab.json").open("w", encoding="utf-8") as fh:
        json.dump({"place_vocab": place_vocab, "place_to_idx": place_to_idx}, fh, indent=2, ensure_ascii=False)

    split_export = defaultdict(list)
    for match_id, split in split_map.items():
        split_export[split].append(match_id)
    with (OUTPUT_ROOT / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(split_export, fh, indent=2, ensure_ascii=False)

    manifest_df.to_csv(OUTPUT_ROOT / "match_manifest.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_parquet(OUTPUT_ROOT / "match_manifest.parquet", index=False)

    rounds_df.to_csv(OUTPUT_ROOT / "ct_round_samples.csv", index=False, encoding="utf-8-sig")
    rounds_df.to_parquet(OUTPUT_ROOT / "ct_round_samples.parquet", index=False)

    context_df.to_csv(OUTPUT_ROOT / "ct_round_context.csv", index=False, encoding="utf-8-sig")
    context_df.to_parquet(OUTPUT_ROOT / "ct_round_context.parquet", index=False)

    nodes_df, edges_df = anonymize_players(nodes_df, edges_df)
    nodes_df.to_parquet(OUTPUT_ROOT / "ct_snapshot_nodes.parquet", index=False)
    edges_df.to_parquet(OUTPUT_ROOT / "ct_snapshot_edges.parquet", index=False)
    round_vectors_df.to_parquet(OUTPUT_ROOT / "ct_round_place_vectors.parquet", index=False)

    summary = {
        "included_match_count": int(manifest_df["include_in_model"].sum()),
        "excluded_match_count": int((~manifest_df["include_in_model"]).sum()),
        "prepared_round_samples": int(rounds_df.shape[0]),
        "prepared_node_rows": int(nodes_df.shape[0]),
        "prepared_edge_rows": int(edges_df.shape[0]),
        "prepared_context_rows": int(context_df.shape[0]),
        "snapshot_seconds": list(SNAPSHOT_SECONDS),
    }
    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
