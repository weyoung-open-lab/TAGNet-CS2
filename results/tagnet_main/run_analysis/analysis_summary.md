# TAGNet Analysis Summary

- Source run: `run`
- Training-best K (silhouette on saved run): `8`
- Multi-metric recommended K: `6`
- Word-range recommended K (5~8): `6`
- Teams with fingerprints: `28`
- Teams with >=10 CT rounds: `26`
- Folder-name mismatches flagged: `2`
- Teams with leading/trailing comparison: `9`

## Top K Candidates

```text
 k  silhouette  davies_bouldin  calinski_harabasz     inertia  min_cluster_size  max_cluster_size  weak_label_purity_at_15s                  cluster_sizes  rank_silhouette  rank_davies_bouldin  rank_calinski_harabasz  avg_rank
 6    0.183548        1.745832         116.089521 3299.589333                62               151                  0.844444         124,109,115,151,136,62              2.0                  6.0                     5.0  4.333333
 2    0.211831        1.800564         207.323149 4676.307017               340               357                  0.844444                        340,357              1.0                 12.0                     1.0  4.666667
 8    0.178258        1.709396         104.669161 2942.364136                56               117                  0.844444     64,106,56,86,106,84,78,117              5.0                  3.0                     7.0  5.000000
 7    0.182500        1.748375         110.121998 3101.417379                58               135                  0.844444       123,104,135,113,80,58,84              3.0                  7.0                     6.0  5.333333
10    0.175134        1.706331          93.249102 2732.836642                39               111                  0.844444 73,81,70,111,39,75,72,73,60,43              6.0                  2.0                     9.0  5.666667
```

## Context Trigger Tests

```text
              variable      chi2      p_value  dof  cramers_v  n_rows
          ct_econ_type 96.252629 6.665500e-14   15   0.214550     697
prev_round_winner_side 26.759550 6.353311e-05    5   0.200750     664
prev_round_attack_site 27.468835 2.514061e-02   15   0.114615     697
           score_state 12.624737 2.454163e-01   10   0.095166     697
```

## Tactical Consistency

```text
 cluster_id  round_count  ct_win_rate  round_win_variance  team_win_rate_variance  team_count
          0          124     0.572581            0.244732                0.113022          25
          1          109     0.495413            0.249979                0.117193          23
          2          115     0.591304            0.241664                0.139925          23
          3          151     0.443709            0.246831                0.042917          25
          4          136     0.522059            0.249513                0.068655          24
          5           62     0.241935            0.183403                0.094168          21
```

## H2 Paired T-Test

```text
 cluster_id  team_pair_count  leading_mean_ratio  trailing_mean_ratio  mean_ratio_diff_leading_minus_trailing  t_statistic  p_value
          0               22            0.181285             0.186984                               -0.005698    -0.100126 0.921194
          1               22            0.195801             0.184850                                0.010951     0.206768 0.838182
          2               22            0.128821             0.137223                               -0.008402    -0.262863 0.795218
          3               22            0.258978             0.180665                                0.078313     1.219277 0.236258
          4               22            0.204071             0.194442                                0.009629     0.194599 0.847577
          5               22            0.031044             0.115837                               -0.084793    -3.535497 0.001961
```

## H3 Opponent Chi-Square

```text
      ct_team_key  ct_team_name  opponent_count      chi2  p_value  dof  cramers_v
team_28ddfbbe505c        aurora               4 25.675291 0.011928   12   0.431338
team_61eb6d2a16cc          mouz               3 17.821429 0.058052   10   0.497514
team_ab866f0ad3c4            9z               2  9.952381 0.076596    5   0.643958
team_93e508613892         monte               2  9.909785 0.077832    5   0.556490
team_673aa049ddc8    parivision               6 29.615195 0.239021   25   0.290886
team_2cafdd88a2b3           nrg               2  5.569940 0.350338    5   0.541438
team_7e72ea96dd75         furia               4 15.310909 0.429262   15   0.307428
team_fb66b63dfb7b       falcons               2  4.405556 0.492618    5   0.411636
team_2e4e9fb62afd   the-mongolz               2  3.162500 0.531009    4   0.370810
team_c02184ca70de      astralis               2  3.750000 0.585941    5   0.456435
team_32ca7afd3148      vitality               2  1.809524 0.612865    3   0.308607
team_c6e3c1179741        spirit               3  4.888889 0.769383    8   0.349603
team_0ac8ac222de3        legacy               2  1.586667 0.811187    4   0.305505
team_5f41706eee17 natus-vincere               7 21.536842 0.870354   30   0.218768
team_0445d85984a4           fut               2  1.809524 0.874823    5   0.274585
team_decca57bd08a       falcons               3  3.815476 0.955287   10   0.252173
team_8cfda985b903        heroic               2  0.220000 0.994376    4   0.100000
```

## Rule-Based Cluster Naming Draft

```text
 cluster_id           tactical_name  rule_prior_score_5                                                                                                                 comment                                                      top_places_15s                        area_ratio_15s                                                                                      representative_rounds
          0         Anchor-A Biased                 4.4                                      Strong anchors at both sites with a clear A majority: an A-leaning two-site setup.   BombsiteB:209, LongA:173, LongDoors:46, UnderA:44, ShortStairs:43 A:0.590, B:0.373, Mid:0.031, CT:0.006 natus-vincere-vs-monte_dust2__r04 | inner-circle-vs-spirit_dust2__r12 | natus-vincere-vs-aurora_dust2__r06
          1        Default Balanced                 3.8                                                          Evenly spread setup; main positions: LongA, BombsiteB, UnderA.      LongA:136, BombsiteB:120, UnderA:69, MidDoors:48, LongDoors:47 A:0.534, B:0.284, Mid:0.095, CT:0.086    the-mongolz-vs-aurora_dust2__r17 | gamerlegion-vs-vitality_dust2__r17 | the-mongolz-vs-3dmax_dust2__r08
          2   B-Biased Double-Point                 4.4                                         Similar A/B shares with B slightly stronger: a two-site setup tilted towards B. BombsiteB:211, LongA:114, MidDoors:77, LongDoors:37, ShortStairs:32 A:0.403, B:0.417, Mid:0.141, CT:0.038                parivision-vs-legacy_dust2__r11 | faze-vs-aurora_dust2__r22 | parivision-vs-pain_dust2__r22
          3    Mid-Control Balanced                 4.7                             High Mid share at 15 s with similar A/B shares: mid control with a balanced two-site setup.   MidDoors:190, BombsiteB:154, LongA:97, Catwalk:54, LowerTunnel:45 A:0.335, B:0.327, Mid:0.310, CT:0.028          natus-vincere-vs-aurora_dust2__r14 | gamerlegion-vs-vitality_dust2__r02 | fut-vs-furia_dust2__r08
          4              A-Long+Mid                 4.5                 A majority with a non-trivial Mid share: A is the main defence while MidDoors / TopMid control is kept.   LongA:175, BombsiteB:119, MidDoors:118, ShortStairs:64, UnderA:48 A:0.587, B:0.213, Mid:0.190, CT:0.010             gamerlegion-vs-vitality_dust2__r18 | natus-vincere-vs-monte_dust2__r07 | mouz-vs-9z_dust2__r12
          5 Aggressive A-Heavy Push                 4.8 Clearly the highest A share and a low B share, typically forward or stacked positions such as LongA / Catwalk / UnderA.       LongA:66, Catwalk:38, UnderA:33, BombsiteB:32, ShortStairs:28 A:0.687, B:0.148, Mid:0.100, CT:0.065       natus-vincere-vs-monte_dust2__r10 | heroic-vs-furia_dust2__r02 | falcons-vs-natus-vincere_dust2__r11
```

## Cluster Representatives

```text
 cluster_id                          sample_id                      match_id  round_num split  distance_to_centroid
          0  natus-vincere-vs-monte_dust2__r04  natus-vincere-vs-monte_dust2          4   val              1.124395
          0  inner-circle-vs-spirit_dust2__r12  inner-circle-vs-spirit_dust2         12 train              1.234761
          0 natus-vincere-vs-aurora_dust2__r06 natus-vincere-vs-aurora_dust2          6 train              1.235792
          0   falcons-vs-parivision_dust2__r22   falcons-vs-parivision_dust2         22 train              1.236781
          0   the-mongolz-vs-aurora_dust2__r20   the-mongolz-vs-aurora_dust2         20 train              1.243007
          1   the-mongolz-vs-aurora_dust2__r17   the-mongolz-vs-aurora_dust2         17 train              1.443270
          1 gamerlegion-vs-vitality_dust2__r17 gamerlegion-vs-vitality_dust2         17  test              1.478104
          1    the-mongolz-vs-3dmax_dust2__r08    the-mongolz-vs-3dmax_dust2          8   val              1.550507
          1          faze-vs-aurora_dust2__r19          faze-vs-aurora_dust2         19 train              1.599006
          1          falcons-vs-nrg_dust2__r09          falcons-vs-nrg_dust2          9   val              1.619913
          2    parivision-vs-legacy_dust2__r11    parivision-vs-legacy_dust2         11 train              1.149952
          2          faze-vs-aurora_dust2__r22          faze-vs-aurora_dust2         22 train              1.158178
          2      parivision-vs-pain_dust2__r22      parivision-vs-pain_dust2         22   val              1.181248
          2     parivision-vs-monte_dust2__r23     parivision-vs-monte_dust2         23 train              1.192063
          2  parivision-vs-astralis_dust2__r11  parivision-vs-astralis_dust2         11 train              1.216301
          3 natus-vincere-vs-aurora_dust2__r14 natus-vincere-vs-aurora_dust2         14 train              1.240166
          3 gamerlegion-vs-vitality_dust2__r02 gamerlegion-vs-vitality_dust2          2  test              1.438006
          3            fut-vs-furia_dust2__r08            fut-vs-furia_dust2          8  test              1.529489
          3         heroic-vs-furia_dust2__r13         heroic-vs-furia_dust2         13 train              1.557611
          3  monte-vs-natus-vincere_dust2__r03  monte-vs-natus-vincere_dust2          3 train              1.647775
```