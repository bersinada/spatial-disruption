# Current Sprint — Week 1

## P0 — Research-Critical

### Research and Scope

- [x] Freeze research thesis
- [x] Define exact prediction target (Primary: reachable, Secondary: relative_detour)
- [x] Define graph schema (Canonical directed DiGraph, Option A)
- [x] Define disruption generation protocol (Localized spatial hazard, 100-250m)
- [x] Define train/validation/test protocol (Scenario-based split)
- [x] Select City A and City B based on explicit data criteria (Seattle and Portland)

### Infrastructure

- [x] Initialize git repository
- [x] Bootstrap Python environment with uv
- [x] Pin dependencies (osmnx, geopandas, networkx, scipy, pyarrow)
- [x] Add reproducible configuration
- [x] Add experiment logging
- [x] Add deterministic random seeds

### Data

- [x] Download road networks for City A and City B with 1 km buffer
- [x] Download critical facility POIs (hospitals and clinics)
- [x] Clean and normalize road network
- [x] Construct directed canonical road graph
- [x] Map facilities to valid surface access nodes
- [x] Validate graph connectivity and boundary containment
- [x] Persist processed graph artifacts

### Counterfactual Dataset

- [x] Implement localized disruption generator
- [x] Define disruption severity distribution (100m to 250m radii)
- [x] Generate disrupted graph states (40 scenarios across Seattle and Portland)
- [x] Generate exact reachability labels via Dijkstra
- [x] Generate shortest-path relative detour labels
- [x] Implement pilot dataset generation and automated 15-check validation
- [x] Generate full 4,000-sample benchmark dataset
- [x] Validate generated samples and confirm deterministic regeneration

### Baseline

- [x] Lock feature leakage contract (DEC-015)
- [x] Implement LightGBM baseline
- [x] Implement spatial feature extraction
- [x] Implement leakage-safe preprocessing (frozen City A normalizers)
- [x] Establish baseline metrics (ROC-AUC 0.9945 in-city, 0.9794 zero-shot transfer)

### Graph Models

- [x] Implement GraphSAGE baseline
- [x] Implement relational GNN (CompGCN / RGCN)
- [x] Establish identical evaluation harness
- [x] Run in-city validation (Seattle Test: ROC-AUC 1.0000, Acc 100%)
- [x] Run cross-city transfer (Portland Transfer: ROC-AUC 0.9445, PR-AUC 0.9839)

### Ablations

- [ ] Relation semantics ablation
- [ ] Geometry/coordinate ablation
- [ ] Message-passing depth ablation
- [ ] Disruption severity evaluation (E07)
- [ ] Secondary geographic spatial split evaluation

### Benchmark

- [ ] Benchmark learned inference latency
- [ ] Benchmark exact Dijkstra
- [ ] Record throughput
- [ ] Record memory/VRAM usage
- [ ] Record prediction error

### Analysis

- [ ] Compare representation levels
- [ ] Analyze cross-city degradation
- [ ] Analyze failure cases
- [ ] Determine whether relational semantics provide additional signal
- [ ] Determine whether topology generalizes without coordinates

### Publication

- [ ] Clean experiment outputs
- [ ] Generate final tables
- [ ] Generate final plots
- [ ] Write technical report
- [ ] Write architecture README
- [ ] Document reproduction procedure
- [ ] Publish repository

---

## P1 — Only If P0 Is Complete

- [ ] TransE sanity check
- [ ] RotatE sanity check
- [ ] Bidirectional City A ↔ City B transfer
- [ ] Additional disruption types
- [ ] Additional facility classes (fire stations, police)
- [ ] Natural-prevalence test benchmark using unbiased OD sampling

---

## Explicitly Out of Scope

- [ ] 3D Gaussian Splatting
- [ ] NeRF
- [ ] GraphRAG
- [ ] LLM agents
- [ ] full 3D scene graphs
- [ ] production API
- [ ] interactive frontend
- [ ] multi-city foundation model
- [ ] custom CUDA kernels

---

## Definition of Done

The sprint is complete when:

1. A real urban graph can be constructed reproducibly.
2. Counterfactual disruptions can be generated reproducibly.
3. Exact algorithms can generate ground truth.
4. A strong tabular baseline exists.
5. A homogeneous GNN baseline exists.
6. A relational GNN exists.
7. At least one geographic transfer experiment has been completed.
8. At least two structural ablations have been completed.
9. All results are reproducible from scripts/configuration.
10. The findings are documented publicly.
