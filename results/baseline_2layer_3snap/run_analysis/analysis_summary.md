# TAGNet Analysis Summary

- Source run: `run`
- Training-best K (silhouette on saved run): `7`
- Multi-metric recommended K: `9`
- Word-range recommended K (5~8): `8`
- Teams with fingerprints: `28`
- Teams with >=10 CT rounds: `26`
- Folder-name mismatches flagged: `2`
- Teams with leading/trailing comparison: `9`

## Top K Candidates

```text
 k  silhouette  davies_bouldin  calinski_harabasz     inertia  min_cluster_size  max_cluster_size  weak_label_purity_at_15s                  cluster_sizes  rank_silhouette  rank_davies_bouldin  rank_calinski_harabasz  avg_rank
 9    0.197874        1.533997         104.618082 1994.838500                44               109                  0.844444    67,84,109,98,91,81,69,44,54              1.0                  2.0                     8.0  3.666667
 8    0.194951        1.491668         108.969395 2098.408683                48               148                  0.855556     54,82,148,100,70,109,48,86              3.0                  1.0                     7.0  3.666667
 7    0.183409        1.575345         112.479252 2235.267148                57               150                  0.844444        114,150,57,91,103,83,99              5.0                  3.0                     6.0  4.666667
10    0.195941        1.615961          99.464604 1919.878913                39               107                  0.844444 47,57,79,90,66,87,52,73,39,107              2.0                  4.0                     9.0  5.000000
 2    0.185331        1.995871         168.406179 3559.123337               347               350                  0.844444                        350,347              4.0                 14.0                     1.0  6.333333
```

## Context Trigger Tests

```text
              variable       chi2      p_value  dof  cramers_v  n_rows
          ct_econ_type 156.072705 2.445194e-21   24   0.273204     697
prev_round_attack_site  63.873986 1.777787e-05   24   0.174777     697
           score_state  40.731602 6.082300e-04   16   0.170936     697
prev_round_winner_side  21.870270 5.161945e-03    8   0.181486     664
```

## Tactical Consistency

```text
 cluster_id  round_count  ct_win_rate  round_win_variance  team_win_rate_variance  team_count
          0           67     0.552239            0.247271                0.135125          21
          1           84     0.583333            0.243056                0.095149          21
          2          109     0.550459            0.247454                0.115992          24
          3           98     0.500000            0.250000                0.075810          22
          4           91     0.604396            0.239102                0.119089          20
          5           81     0.370370            0.233196                0.101298          25
          6           69     0.492754            0.249947                0.149615          23
          7           44     0.318182            0.216942                0.131632          17
          8           54     0.333333            0.222222                0.056997          19
```

## H2 Paired T-Test

```text
 cluster_id  team_pair_count  leading_mean_ratio  trailing_mean_ratio  mean_ratio_diff_leading_minus_trailing  t_statistic  p_value
          0               22            0.134787             0.094758                                0.040028     0.725434 0.476193
          1               22            0.115025             0.131294                               -0.016269    -0.313505 0.756990
          2               22            0.166769             0.166351                                0.000418     0.007316 0.994232
          3               22            0.188331             0.139923                                0.048408     0.854037 0.402722
          4               22            0.090810             0.129312                               -0.038502    -1.407116 0.174022
          5               22            0.095813             0.141447                               -0.045634    -1.306959 0.205358
          6               22            0.120898             0.080057                                0.040841     1.250479 0.224879
          7               22            0.033636             0.030924                                0.002712     0.149105 0.882893
          8               22            0.053932             0.085933                               -0.032001    -1.544430 0.137421
```

## H3 Opponent Chi-Square

```text
      ct_team_key  ct_team_name  opponent_count      chi2  p_value  dof  cramers_v
team_28ddfbbe505c        aurora               4 34.982569 0.068660   24   0.503485
team_2cafdd88a2b3           nrg               2 13.269841 0.102896    8   0.835711
team_93e508613892         monte               2 11.263877 0.127516    7   0.593293
team_ab866f0ad3c4            9z               2  8.142857 0.227823    6   0.582482
team_decca57bd08a       falcons               3 19.638889 0.236927   16   0.572114
team_673aa049ddc8    parivision               6 44.801467 0.277382   40   0.357777
team_0445d85984a4           fut               2  8.200000 0.315289    7   0.584523
team_32ca7afd3148      vitality               2  6.680159 0.351441    6   0.592948
team_0ac8ac222de3        legacy               2  6.563889 0.363065    6   0.621378
team_8cfda985b903        heroic               2  6.404444 0.379440    6   0.539547
team_c6e3c1179741        spirit               3 14.285714 0.428649   14   0.597614
team_61eb6d2a16cc          mouz               3 14.066667 0.444756   14   0.442007
team_c02184ca70de      astralis               2  4.650000 0.460068    5   0.508265
team_7e72ea96dd75         furia               4 23.400000 0.496300   24   0.380058
team_5f41706eee17 natus-vincere               7 47.125848 0.508598   48   0.295415
team_fb66b63dfb7b       falcons               2  4.875000 0.675214    7   0.433013
team_2e4e9fb62afd   the-mongolz               2  3.496465 0.835600    7   0.389898
```

## Rule-Based Cluster Naming Draft

```text
 cluster_id           tactical_name  rule_prior_score_5                                                                                                                 comment                                                      top_places_15s                        area_ratio_15s                                                                                       representative_rounds
          0        Default Balanced                 3.8                                                          Evenly spread setup; main positions: BombsiteB, LongA, UnderA.        BombsiteB:106, LongA:69, UnderA:34, LongDoors:26, CTSpawn:19 A:0.496, B:0.376, Mid:0.066, CT:0.063                mouz-vs-heroic_dust2__r09 | gamerlegion-vs-vitality_dust2__r16 | falcons-vs-furia_dust2__r26
          1              A-Long+Mid                 4.5                 A majority with a non-trivial Mid share: A is the main defence while MidDoors / TopMid control is kept.       LongA:135, MidDoors:80, BombsiteB:78, LongDoors:38, UnderA:27 A:0.583, B:0.200, Mid:0.202, CT:0.014              the-mongolz-vs-aurora_dust2__r19 | aurora-vs-furia_dust2__r07 | natus-vincere-vs-b8_dust2__r05
          2         Anchor-A Biased                 4.4                                      Strong anchors at both sites with a clear A majority: an A-leaning two-site setup.   BombsiteB:191, LongA:162, UnderA:41, LongDoors:40, ShortStairs:28 A:0.585, B:0.383, Mid:0.024, CT:0.007       parivision-vs-monte_dust2__r25 | falcons-vs-parivision_dust2__r22 | natus-vincere-vs-3dmax_dust2__r17
          3        Default Balanced                 3.8                                                        Evenly spread setup; main positions: MidDoors, LongA, BombsiteB.       MidDoors:147, LongA:99, BombsiteB:77, UnderA:35, LongDoors:27 A:0.408, B:0.216, Mid:0.327, CT:0.049                  liquid-vs-falcons_dust2__r12 | natus-vincere-vs-monte_dust2__r15 | fut-vs-furia_dust2__r08
          4   B-Biased Double-Point                 4.4                                         Similar A/B shares with B slightly stronger: a two-site setup tilted towards B. BombsiteB:171, LongA:101, MidDoors:69, LongDoors:30, ShortStairs:17 A:0.396, B:0.413, Mid:0.156, CT:0.035            parivision-vs-monte_dust2__r23 | monte-vs-natus-vincere_dust2__r06 | falcons-vs-furia_dust2__r06
          5        Default Balanced                 3.8                                                      Evenly spread setup; main positions: Catwalk, MidDoors, BombsiteB.          Catwalk:99, MidDoors:65, BombsiteB:52, LongA:37, UnderA:20 A:0.499, B:0.222, Mid:0.242, CT:0.037                     liquid-vs-falcons_dust2__r17 | natus-vincere-vs-g2_dust2__r14 | fut-vs-furia_dust2__r21
          6        Default Balanced                 3.8                                                     Evenly spread setup; main positions: ShortStairs, BombsiteB, LongA.      ShortStairs:83, BombsiteB:63, LongA:48, MidDoors:41, UnderA:40 A:0.583, B:0.255, Mid:0.128, CT:0.035 falcons-vs-parivision_dust2__r08 | natus-vincere-vs-legacy_dust2__r06 | falcons-vs-natus-vincere_dust2__r17
          7        Default Balanced                 3.8                                                  Evenly spread setup; main positions: BombsiteB, MidDoors, LowerTunnel.      BombsiteB:75, MidDoors:32, LowerTunnel:27, LongA:15, BDoors:12 A:0.191, B:0.591, Mid:0.186, CT:0.032                     heroic-vs-furia_dust2__r01 | nrg-vs-furia_dust2__r13 | falcons-vs-parivision_dust2__r15
          8 Aggressive A-Heavy Push                 4.8 Clearly the highest A share and a low B share, typically forward or stacked positions such as LongA / Catwalk / UnderA.         LongA:95, BombsiteB:32, UnderA:31, LongDoors:23, CTSpawn:16 A:0.737, B:0.159, Mid:0.044, CT:0.059                            parivision-vs-monte_dust2__r02 | mouz-vs-fut_dust2__r18 | mouz-vs-fut_dust2__r16
```

## Cluster Representatives

```text
 cluster_id                          sample_id                      match_id  round_num split  distance_to_centroid
          0          mouz-vs-heroic_dust2__r09          mouz-vs-heroic_dust2          9 train              1.077780
          0 gamerlegion-vs-vitality_dust2__r16 gamerlegion-vs-vitality_dust2         16  test              1.096151
          0        falcons-vs-furia_dust2__r26        falcons-vs-furia_dust2         26 train              1.173927
          0     natus-vincere-vs-b8_dust2__r08     natus-vincere-vs-b8_dust2          8 train              1.190900
          0        falcons-vs-furia_dust2__r22        falcons-vs-furia_dust2         22 train              1.274047
          1   the-mongolz-vs-aurora_dust2__r19   the-mongolz-vs-aurora_dust2         19 train              0.813022
          1         aurora-vs-furia_dust2__r07         aurora-vs-furia_dust2          7  test              0.828287
          1     natus-vincere-vs-b8_dust2__r05     natus-vincere-vs-b8_dust2          5 train              0.847665
          1              mouz-vs-9z_dust2__r11              mouz-vs-9z_dust2         11 train              0.864230
          1  virtus-pro-vs-vitality_dust2__r08  virtus-pro-vs-vitality_dust2          8  test              0.874603
          2     parivision-vs-monte_dust2__r25     parivision-vs-monte_dust2         25 train              0.865328
          2   falcons-vs-parivision_dust2__r22   falcons-vs-parivision_dust2         22 train              0.906205
          2  natus-vincere-vs-3dmax_dust2__r17  natus-vincere-vs-3dmax_dust2         17 train              0.944379
          2   the-mongolz-vs-aurora_dust2__r20   the-mongolz-vs-aurora_dust2         20 train              0.973247
          2          faze-vs-aurora_dust2__r06          faze-vs-aurora_dust2          6 train              1.009294
          3       liquid-vs-falcons_dust2__r12       liquid-vs-falcons_dust2         12 train              1.038937
          3  natus-vincere-vs-monte_dust2__r15  natus-vincere-vs-monte_dust2         15   val              1.059424
          3            fut-vs-furia_dust2__r08            fut-vs-furia_dust2          8  test              1.079388
          3 gamerlegion-vs-vitality_dust2__r06 gamerlegion-vs-vitality_dust2          6  test              1.122796
          3          navi-vs-aurora_dust2__r08          navi-vs-aurora_dust2          8 train              1.150868
```