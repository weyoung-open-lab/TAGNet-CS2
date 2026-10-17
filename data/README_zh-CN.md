# 数据说明

[English](README.md) | **简体中文**

## 内容

| 路径 | 说明 |
|---|---|
| `match_list.csv` | 34 个已解析的 Dust2 比赛文件夹：是否使用、排除原因、所属划分、回合数、demo 版本号和服务器。实际使用 33 场；`legacy-vs-aurora_dust2` 因 demo 头信息显示地图不是 `de_dust2` 而被排除。 |
| `processed/dust2_ct_5snap/` | TAGNet 和全部补充实验使用的数据集（快照时刻 10/12/15/18/20 秒）。 |
| `processed/dust2_ct_3snap/` | 同样的 697 个回合，快照时刻为 10/15/20 秒，只用于表 III 的两层基线。 |
| `map/dust2_walkable_mask.npz` | 图 4 的底图：16 单位网格上的 Dust2 可行走区域，由 `scripts/build_map_mask.py` 根据三场已解析比赛中约 500 万个选手位置估计得到。 |
| `parsed/` | 未包含。这是 `scripts/parse_demos.py` 的输出位置，每场比赛一个文件夹，存放 Awpy 解析出的原始数据表。 |

原始 `.dem` 文件不在仓库中重新分发。数据集来自 [HLTV](https://www.hltv.org/) 上公开的职业比赛 demo，比赛列表见 `match_list.csv`（文件夹名格式为 `<队伍A>-vs-<队伍B>_dust2`）。用 `scripts/parse_demos.py` 解析 demo、再用 `scripts/prepare_data.py` 转换，即可按论文中相同的流程得到数据集。

每张表都同时提供 Parquet 格式（供代码读取）和 CSV 格式（方便查看）。节点表的 CSV 版本中，列表型的两列 `place_onehot` 和 `node_feature_vector` 以空格分隔的形式写出。

## 处理后的数据集（`processed/dust2_ct_5snap/`）

来自 33 场比赛、132 名选手（26 支队伍，28 套不同的 CT 阵容）的 697 个 CT 防守回合。按比赛划分为 23 / 5 / 5 场，对应 486 / 95 / 116 个回合，详见 `splits.json`。

保留一个回合需要同时满足：
- demo 地图是 `de_dust2`；
- 双方阵容都已知；
- 5 个快照都在回合结束之前；
- 每个快照时刻，tick 数据里都有 5 名不同的 CT 选手。

快照 tick 为 `freeze_end + s × 64`，其中 `s ∈ {10, 12, 15, 18, 20}`。在某个快照时刻已经阵亡的选手会被保留，血量记为 0，位置取 tick 数据中记录的值（共 408 / 17,425 行节点，涉及 108 个回合）。

| 文件 | 行数 | 内容 |
|---|---:|---|
| `ct_round_samples.parquet` / `.csv` | 697 | 每个回合一行：编号、所属划分、快照 tick、胜方、结束原因、炸弹点、队伍标识、回合前比分、是否手枪局、上一回合胜方和进攻点。 |
| `ct_snapshot_nodes.parquet` / `.csv` | 17,425 | 每个（回合，快照，CT 选手）一行：匿名编号 `player_id`、命名区域、大区（A/B/Mid/CT/other）、X/Y/Z、血量、护甲、装备价值、归一化后的节点特征。 |
| `ct_snapshot_edges.parquet` / `.csv` | 34,850 | 每个（回合，快照，选手对）一行：两名选手的匿名编号、平面欧氏距离 `distance_xy` 及其按训练集做 min-max 归一化后的值 `distance_xy_norm`。 |
| `ct_round_context.parquet` / `.csv` | 3,485 | 每个（回合，快照）一行：装备价值、经济类型、截至该快照的道具数量和投掷区域、比分状态、上一回合信息。 |
| `ct_round_place_vectors.parquet` | 697 | 每个回合展平后的区域向量（模型未使用）。 |
| `normalization_stats.json` | | 归一化用到的训练集统计量（装备价值取对数后的最小/最大值，两两距离最大值 4215.87 个游戏单位）。 |
| `place_vocab.json` | | 节点 one-hot 特征使用的 20 个命名区域。 |
| `splits.json` | | 各划分包含的比赛编号。 |
| `match_manifest.*` | | 与 `match_list.csv` 内容相同，由 `prepare_data.py` 写出。 |
| `summary.json` | | 各表行数。 |

情境表中的 `weak_label` 相关列在本版本中为空，只被一个可选的诊断指标（`weak_label_purity_at_15s`）使用。

## 选手标识

数据中不包含选手的游戏内名字和 SteamID。每名选手用一个固定的匿名编号表示（`player_001` 到 `player_132`，按 SteamID 排序），同一名选手在所有回合、以及两个处理后数据集中的编号都相同。`prepare_data.py` 输出数据时也会做同样的匿名化。队名保留，因为它们本来就出现在公开的比赛编号中；`ct_team_key` 和 `t_team_key` 是阵容的哈希值，不会暴露具体选手。
