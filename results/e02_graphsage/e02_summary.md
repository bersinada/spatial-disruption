# E02 Homogeneous Graph Baseline (GraphSAGE) — Results

- **Model**: Inductive GraphSAGE (2-layer SAGEConv, hidden_dim=64)
- **Runtime**: 9.04 seconds on cuda
- **Representation**: Homogeneous directed road graph with operational corridor topology
- **DEC-015 Leakage Contract**: 100% verified; zero target or test graph leakage

## Representation Hierarchy Comparison: E01 (Spatial) vs E02 (Topology)

### Primary Task: Post-Disruption Reachability

| Split                                |   E01 LightGBM AUC |   E02 GraphSAGE AUC |   E02 PR-AUC |   E02 Acc |   E02 Bal Acc |   E02 F1 |   E02 Brier |
|:-------------------------------------|-------------------:|--------------------:|-------------:|----------:|--------------:|---------:|------------:|
| Train (Seattle In-Sample)            |             1      |              1      |       1      |     0.993 |        0.9821 |   0.9957 |      0.0029 |
| Val (Seattle Selection)              |             0.9998 |              1      |       1      |     1     |        1      |   1      |      0      |
| Test (Seattle Tier 1 In-City)        |             0.9945 |              1      |       1      |     1     |        1      |   1      |      0      |
| Transfer (Portland Tier 2 Zero-Shot) |             0.9794 |              0.9445 |       0.9839 |     0.95  |        0.8677 |   0.9701 |      0.0435 |

### Secondary Task: Conditional Relative Detour (Reachable Pairs)

| Split                                |   E01 MAE |   E02 MAE |   E01 Pearson r |   E02 Pearson r |   E02 Spearman rho |   E02 RMSE |
|:-------------------------------------|----------:|----------:|----------------:|----------------:|-------------------:|-----------:|
| Train (Seattle In-Sample)            |    0.0083 |    0.0187 |          0.8946 |          0.167  |             0.5373 |     0.0519 |
| Val (Seattle Selection)              |    0.0131 |    0.0102 |          0.6028 |          0.2481 |             0.566  |     0.0328 |
| Test (Seattle Tier 1 In-City)        |    0.0847 |    0.0947 |          0.4672 |         -0.0586 |             0.2262 |     0.2573 |
| Transfer (Portland Tier 2 Zero-Shot) |    0.0243 |    0.0208 |          0.4656 |          0.1337 |             0.3271 |     0.0588 |

## Scientific Findings & Representation Hierarchy Analysis

1. **Zero-Shot Transfer Reachability**: Inductive GraphSAGE achieves ROC-AUC 0.9445 (PR-AUC: 0.9839) on unseen Portland network, compared to 0.9794 for LightGBM.
2. **Detour Rerouting Correlation**: On zero-shot transfer detour prediction, GraphSAGE achieves Pearson r = 0.1337 (Spearman rho = 0.3271) with MAE = 0.0208.
3. **Topological Message Passing Role**: Operating directly over operational corridor edges allows GraphSAGE to propagate connectivity along the physical network structure without memorizing geographic coordinates.
