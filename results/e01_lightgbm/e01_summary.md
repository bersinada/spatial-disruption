# E01 Continuous Spatial Baseline — Results

- **Model**: LightGBM GBDT (Classifier + Huber Regressor)
- **Runtime**: 1.13 seconds
- **Features**: 36 continuous spatial, intact routing, hazard, and density attributes
- **DEC-015 Leakage Contract**: 100% verified; zero target or post-disruption leakage

## Primary Task: Post-Disruption Reachability (Binary Classification)

| Split                                |   ROC-AUC |   PR-AUC |   Accuracy |   Balanced Acc |     F1 |   Brier Score |
|:-------------------------------------|----------:|---------:|-----------:|---------------:|-------:|--------------:|
| Train (Seattle In-Sample)            |    1      |   1      |      1     |         1      | 1      |        0      |
| Val (Seattle Selection)              |    0.9998 |   0.9999 |      0.984 |         0.9896 | 0.9895 |        0.0103 |
| Test (Seattle Tier 1 In-City)        |    0.9945 |   0.9991 |      0.952 |         0.8261 | 0.9729 |        0.0456 |
| Transfer (Portland Tier 2 Zero-Shot) |    0.9794 |   0.9952 |      0.95  |         0.888  | 0.9697 |        0.0447 |

## Secondary Task: Relative Detour (Conditional Regression on Reachable OD Pairs)

| Split                                |    MAE |   Median AE |   RMSE |   Pearson r |   Spearman rho |
|:-------------------------------------|-------:|------------:|-------:|------------:|---------------:|
| Train (Seattle In-Sample)            | 0.0083 |      0.001  | 0.023  |      0.8946 |         0.826  |
| Val (Seattle Selection)              | 0.0131 |      0.0012 | 0.0285 |      0.6028 |         0.7201 |
| Test (Seattle Tier 1 In-City)        | 0.0847 |      0.0042 | 0.2327 |      0.4672 |         0.7639 |
| Transfer (Portland Tier 2 Zero-Shot) | 0.0243 |      0.002  | 0.0513 |      0.4656 |         0.7506 |

## Top 10 Features by Predictive Gain

| feature                      |   split_importance |   gain_importance |
|:-----------------------------|-------------------:|------------------:|
| min_dist_od_to_epicenter_m   |                404 |          15802.2  |
| dist_origin_to_epicenter_m   |                413 |           1491.12 |
| origin_in_hazard_zone        |                117 |            668.05 |
| origin_y                     |                363 |            603.53 |
| hazard_intersects_od_segment |                 19 |            346.2  |
| dist_dest_to_epicenter_m     |                 76 |            345.36 |
| h3_origin_node_density_res9  |                368 |            293.54 |
| disruption_radius_m          |                161 |            265.99 |
| orig_avg_edge_length         |                166 |             93.05 |
| origin_x                     |                178 |             78.56 |

## Key Scientific Takeaways

1. **In-City Disruption Generalization (Tier 1)**: LightGBM achieves ROC-AUC 0.9945 and Balanced Accuracy 0.8261 on unseen disruption events within Seattle.
2. **Zero-Shot Cross-City Transfer (Tier 2)**: On Portland, reachability ROC-AUC is 0.9794 (PR-AUC: 0.9952), establishing the strong empirical non-graph baseline benchmark.
3. **Graph Baseline Target for E02/E03**: For GraphSAGE (E02) and CompGCN (E03) to demonstrate topological reasoning superiority, they must outperform these tabular spatial generalization frontiers under Tier 1 and Tier 2 transfer.
