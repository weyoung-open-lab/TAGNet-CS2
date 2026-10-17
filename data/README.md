# Data

**English** | [简体中文](README_zh-CN.md)

## Contents

| Path | Description |
|---|---|
| `match_list.csv` | The 34 parsed Dust2 match folders: inclusion flag, exclusion reason, split, round counts, demo patch and server. 33 matches are used; `legacy-vs-aurora_dust2` is excluded because its demo header reports a map other than `de_dust2`. |
| `processed/dust2_ct_5snap/` | Model-ready dataset used for TAGNet and all supplementary experiments (snapshots at 10/12/15/18/20 s). |
| `processed/dust2_ct_3snap/` | The same 697 rounds sampled at 10/15/20 s. Used only for the 2-layer baseline of Table III. |
| `map/dust2_walkable_mask.npz` | Base map of Fig. 3: Dust2 walkable area on a 16-unit grid, estimated from about 5 million player positions in three parsed matches by `scripts/build_map_mask.py`. |
| `parsed/` | Not included. Output location of `scripts/parse_demos.py`, i.e. one folder of raw Awpy tables per match. |

The raw `.dem` files are not redistributed. The dataset was derived from
publicly available professional match demos on [HLTV](https://www.hltv.org/);
the matches are listed in `match_list.csv` (`<team-a>-vs-<team-b>_dust2`).
Demos can be parsed with `scripts/parse_demos.py` and converted with
`scripts/prepare_data.py` using the same procedure as in the paper.

Every table is provided as Parquet (read by the code) and as CSV (for
inspection). In the CSV copies of the node table, the list-valued columns
`place_onehot` and `node_feature_vector` are written as space-separated values.

## Processed dataset (`processed/dust2_ct_5snap/`)

697 CT-side rounds from 33 matches and 132 players (26 team names; 28 distinct CT line-ups). The split is by
match (23 / 5 / 5 matches, 486 / 95 / 116 rounds); see `splits.json`.

A round is kept when the demo map is `de_dust2`, both line-ups are known, all
five snapshots fall before the round ends, and five distinct CT players are
present in the tick data at every snapshot. Snapshot ticks are
`freeze_end + s × 64` for `s ∈ {10, 12, 15, 18, 20}`. Players who have already
died at a snapshot are kept, with `health = 0` and the position recorded in the
tick data (408 of 17,425 node rows, in 108 rounds).

| File | Rows | Content |
|---|---:|---|
| `ct_round_samples.parquet` / `.csv` | 697 | One row per round: ids, split, snapshot ticks, winner, end reason, bomb site, team keys, score before the round, pistol flag, previous-round winner and attack site. |
| `ct_snapshot_nodes.parquet` / `.csv` | 17,425 | One row per (round, snapshot, CT player): anonymous `player_id`, named place, macro area (A/B/Mid/CT/other), X/Y/Z, health, armor, equipment value, normalized node features. |
| `ct_snapshot_edges.parquet` / `.csv` | 34,850 | One row per (round, snapshot, player pair): anonymous player ids, planar Euclidean distance `distance_xy` and its train-set min-max normalization `distance_xy_norm`. |
| `ct_round_context.parquet` / `.csv` | 3,485 | One row per (round, snapshot): equipment values, economy type, utility counts and throw areas up to that snapshot, score state, previous-round information. |
| `ct_round_place_vectors.parquet` | 697 | Flattened per-round place vectors (not used by the model). |
| `normalization_stats.json` | | Train-split statistics used for normalization (equipment log-min/max, maximum pairwise distance 4215.87 units). |
| `place_vocab.json` | | The 20 named places used for the one-hot node feature. |
| `splits.json` | | Match ids per split. |
| `match_manifest.*` | | Same information as `match_list.csv`, as written by `prepare_data.py`. |
| `summary.json` | | Row counts. |

The `weak_label` columns in the context table are empty for this release; they
are only used by an optional diagnostic (`weak_label_purity_at_15s`).

## Player identifiers

Player in-game names and SteamIDs are not included. Each player is identified
by a stable anonymous id (`player_001` to `player_132`, ordered by SteamID),
which is consistent across rounds and across the two processed datasets.
`prepare_data.py` applies the same anonymization to its output. Team names are
kept, since they appear in the public match ids; `ct_team_key` and
`t_team_key` are hashes of the line-up and do not reveal individual players.
