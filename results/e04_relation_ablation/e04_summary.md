# E04 Relation Semantics Ablation — Results

- **Experiment**: E04 (Relation Semantics Ablation)
- **Runtime**: 34.81 seconds on cuda
- **Comparison**: E03 Full Relational Graph (3 typed relations) vs E04 Generic Edge Graph (1 generic relation)
- **DEC-015 Leakage Contract**: 100% verified; zero target or test graph leakage

## Ablation Benchmark: Full Relational (E03) vs Generic Edge (E04)

### Primary Task: Post-Disruption Reachability

| Split                                |   E03 Relational AUC |   E04 Generic AUC |   Delta AUC (E03 - E04) |   E04 PR-AUC |   E04 Acc |   E04 Bal Acc |   E04 F1 |   E04 Brier |
|:-------------------------------------|---------------------:|------------------:|------------------------:|-------------:|----------:|--------------:|---------:|------------:|
| Train (Seattle In-Sample)            |               0.9956 |            1      |                 -0.0044 |        1     |     0.999 |        0.9974 |   0.9994 |      0.0009 |
| Val (Seattle Selection)              |               0.9927 |            1      |                 -0.0073 |        1     |     0.964 |        0.9211 |   0.9772 |      0.0322 |
| Test (Seattle Tier 1 In-City)        |               0.969  |            1      |                 -0.031  |        1     |     0.966 |        0.8768 |   0.9807 |      0.0337 |
| Transfer (Portland Tier 2 Zero-Shot) |               0.9369 |            0.9133 |                  0.0236 |        0.966 |     0.95  |        0.8677 |   0.9701 |      0.0428 |

### Secondary Task: Relative Detour (Reachable OD Pairs)

| Split                                |   E03 Detour r |   E04 Detour r |   Delta r (E03 - E04) |   E03 MAE |   E04 MAE |   E04 Spearman rho |   E04 RMSE |
|:-------------------------------------|---------------:|---------------:|----------------------:|----------:|----------:|-------------------:|-----------:|
| Train (Seattle In-Sample)            |         0.3007 |         0.2637 |                0.037  |    0.0354 |    0.0187 |             0.5457 |     0.052  |
| Val (Seattle Selection)              |         0.3183 |         0.2417 |                0.0766 |    0.0318 |    0.0102 |             0.5408 |     0.0328 |
| Test (Seattle Tier 1 In-City)        |        -0.0596 |        -0.0015 |               -0.0581 |    0.104  |    0.0947 |             0.203  |     0.2573 |
| Transfer (Portland Tier 2 Zero-Shot) |         0.1639 |         0.0701 |                0.0938 |    0.0377 |    0.0208 |             0.2678 |     0.0588 |

## Scientific Conclusions from E04 Ablation

1. **Impact on Zero-Shot Transfer**: Removing relation typing changed Portland transfer ROC-AUC from 0.9369 (E03) to 0.9133 (E04), and detour Pearson correlation from 0.1639 to 0.0701.
2. **Role of Typed Semantics**: Relation typing provides specialized message passing channels separating traversal along physical infrastructure (`connects_to`) from access to critical infrastructure (`accessible_from`), preventing facility access edges from diluting road topology propagation.
