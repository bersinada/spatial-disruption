# Inductive Spatial Reasoning Under Network Disruption

An empirical benchmark and research framework evaluating whether relational graph representations can learn transferable structural patterns of counterfactual urban accessibility under physical road network disruptions.

---

## Research Objective

Can inductive relational graph neural networks learn transferable spatial reasoning of post-disruption urban accessibility—moving beyond continuous spatial coordinates and homogeneous graph message passing?

### Representation Evaluation Hierarchy

```text
Spatial Features (Tabular coordinates, Euclidean distance, local density)
      ↓
Physical Topology (Homogeneous road network DiGraph, message passing)
      ↓
Relational Topology (Planned: typed corridor relations and road-facility edges)
      ↓
Geographic Transfer (Zero-shot evaluation on an unseen metropolitan network)
```

Ground-truth accessibility and detours are generated using exact graph algorithms (Dijkstra). Learned models are evaluated as approximate predictive systems, not as replacements for exact shortest-path ground truth.

---

## Project Status: Benchmark v1.0 Generated

- **Status**: Benchmark dataset generation complete; 16/16 automated protocol checks passed.
- **Model Training**: No ML models have been trained yet.
- **Next Experiment**: E01 (LightGBM tabular spatial baseline).

### Benchmark Dataset Overview

| Metric | Seattle (City A) | Portland (City B) | Benchmark Total |
| :--- | :--- | :--- | :--- |
| **OSM Relation** | Boundary 237385 | Boundary 186579 | Official Municipalities |
| **Extraction Buffer** | 1,000 m external | 1,000 m external | Prevents artificial edge clipping |
| **Physical Nodes** | 2,252 | 2,682 | Surface junction nodes |
| **Canonical Edges** | 5,878 | 6,853 | Parallel corridors collapsed |
| **Critical Facilities** | 27 (Hospitals/Clinics) | 15 (Hospitals/Clinics) | Snapped to surface streets |
| **Disruption Scenarios** | 30 (20 Train, 5 Val, 5 Test) | 10 (Zero-Shot Transfer) | 40 Scenarios |
| **Balanced Samples** | 3,000 (100 per scenario) | 1,000 (100 per scenario) | 4,000 Samples |
| **Active / Control Mix** | 50 Active / 50 Control | 50 Active / 50 Control | Balanced per scenario |
| **Validation Status** | 100% Passed (16/16) | 100% Passed (16/16) | Deterministic & mask-verified |

*Note on representation: The current physical graph is a canonical directed DiGraph where intersections are nodes and road corridors are edges. Critical facilities are currently snapped to road junction nodes. Heterogeneous road-facility relations (e.g. `accessible_from`) will be constructed at the model evaluation stage (E03), not in the raw physical graph.*

*Note on sampling distribution: The 50/50 Active Core / Control Context ratio is an intentional experimental sampling design to prevent trivial >95% unaffected majority-class prediction. It is not intended to represent natural urban origin-destination prevalence. Natural-prevalence testing is planned as a P1 extension.*

---

## Prediction Targets

For every origin-destination query under a localized disruption scenario:

1. **Primary Target — Reachability (`reachable`)**:
   Binary classification: `1` if the destination facility remains reachable post-disruption; `0` if severed.
2. **Secondary Target — Relative Detour (`relative_detour`)**:
   Conditional regression on reachable pairs: `(disrupted_distance - original_distance) / original_distance`. Stored as `null/NaN` for severed pairs.

---

## Feature Leakage Contract (DEC-015)

To guarantee scientific validity and prevent data leakage, all models must strictly respect the input space contract:

### Strictly Forbidden Model Inputs
The following columns contain ground-truth outcomes, scenario identifiers, or sampling indicators and **must never** be provided to models during training or inference:
- `reachable` (Primary target)
- `relative_detour` (Secondary target)
- `disrupted_distance_m` (Trivially encodes reachability and detour)
- `disrupt_path_hops` (Directly encodes reachability)
- `sample_type` (Leaked indicator: reveals whether intact path intersected disruption)
- `split` (Partition identifier)
- `scenario_id` (Scenario memorization risk)
- `seed` (RNG seed)
- `disrupted_edge_ids` (Raw target list; models must inspect graph structure or operational masks)
- Any field directly derived from post-disruption routing.

### Permitted Model Inputs
- **Tabular / Spatial Baseline (E01)**: Origin coordinates (`origin_x`, `origin_y`), destination coordinates (`destination_x`, `destination_y`), facility attributes (`facility_amenity`, `facility_snap_distance_m`), intact network distance (`original_distance_m`), intact path hops (`orig_path_hops`), and disruption hazard geometry (epicenter `x`, `y`, radius in meters, Euclidean distances to epicenter).
- **Graph Models (E02, E03)**: Intact network topology, node/corridor spatial attributes, edge operational binary mask (`edge_disruption_mask`), and query node indicators.
- **Normalization Policy**: All scalers and feature transformers must be fit exclusively on the Seattle Train split and applied frozen to validation, test, and transfer splits.

---

## Leakage Safeguards & Split Semantics

- **Scenario-Level Leakage Prevention**: In the primary benchmark, exact disruption scenarios (epicenter and disabled edge set) appear in exactly one split (Seattle Train, Seattle Validation, Seattle Test, or Portland Transfer).
- **Geographic Scope Limitation**: Scenario splitting does **not** provide geographic independence within the same city. Origins, destination facilities, and intact road corridors may recur across scenarios.
- **Geographic Generalization Testing**: True spatial transfer is evaluated through zero-shot cross-city transfer to Portland (10 scenarios / 1,000 samples). A secondary buffered spatial split within Seattle (1,500 m canal buffer) is defined as an isolated topology transfer experiment.

---

## Disruption Representation & Persistence (DEC-013)

Disruption represents physical corridor closure. Every scenario is persisted using two complementary representations:

1. **Explicit Edge IDs**: List of disabled directed canonical edge pairs `[[u1, v1], [u2, v2], ...]` stored in `scenarios_metadata.json`.
2. **Edge-Level Binary Mask**:
   - `edge_disruption_mask_binary`: Binary string (`1` = operational, `0` = disrupted) aligned with the deterministic canonical edge order, stored in `benchmark_scenarios_metadata.json`.
   - `benchmark_edge_masks.npz`: Compressed NumPy boolean/int8 array keyed by `scenario_id`.
   - `{city}_canonical_edges.json`: Sorted canonical edge index `[[u, v], ...]`.

Parallel directed OSM edges are collapsed into single corridors with aggregated lane-count attributes. Dynamic traffic capacity is not modeled in Week 1.

---

## Repository Structure

```text
spatial-disruption/
├── data/
│   ├── raw/                           # Cached OSM boundaries, road networks, facilities
│   │   ├── seattle/
│   │   └── portland/
│   └── processed/                     # Processed benchmark artifacts (Parquet, JSON, NPZ)
│       ├── benchmark_samples.parquet
│       ├── benchmark_samples.json
│       ├── benchmark_scenarios_metadata.json
│       ├── benchmark_edge_masks.npz
│       ├── seattle_canonical_edges.json
│       ├── portland_canonical_edges.json
│       └── seattle_spatial_split_metadata.json
├── src/
│   ├── data/
│   │   └── osm_loader.py              # Cached OSM boundaries, graphs, and facilities
│   ├── graph/
│   │   ├── physical_graph.py          # Canonical DiGraph builder & surface facility snapping
│   │   └── diagnostics.py             # Graph connectivity and structural diagnostics
│   ├── disruption/
│   │   └── generator.py               # Deterministic localized spatial disruptions (100–250 m)
│   ├── ground_truth/
│   │   └── reachability.py            # Exact Dijkstra ground-truth reachability & detour engine
│   └── dataset/
│       ├── generator.py               # Deterministic benchmark generator (pilot + full)
│       └── validator.py               # Automated 16-check integrity, leakage & repro audit
├── AGENTS.md                          # Research execution protocol & agent constraints
├── DECISIONS.md                       # Log of locked decisions (DEC-001 to DEC-015)
├── EXPERIMENTS.md                     # Formal experiment designs (E01 to E07)
├── PROJECTS.md                        # Research thesis, scope limits, and contracts
└── TASKS.md                           # Sprint tracking and definition of done
```

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Setup virtual environment with Python 3.12+
uv venv
source .venv/bin/activate
uv pip install osmnx geopandas networkx scipy pyarrow
```

### 2. Run Pilot Generation & Validation (500 Samples)

```bash
# Generate 5 pilot scenarios
python -m src.dataset.generator --mode pilot

# Run 16-check automated validation suite
python -m src.dataset.validator --pilot
```

### 3. Generate Full 4,000-Sample Benchmark

```bash
# Generate all 40 scenarios with edge masks and canonical edge lists
python -m src.dataset.generator --mode full --with-spatial-split

# Run complete 16-check validation suite
python -m src.dataset.validator
```

---

## Research Documentation

- [PROJECTS.md](PROJECTS.md): Problem definition, research questions, and feature contracts.
- [DECISIONS.md](DECISIONS.md): Architectural decisions DEC-001 through DEC-015.
- [EXPERIMENTS.md](EXPERIMENTS.md): Formal experimental protocols for E01–E07.
- [TASKS.md](TASKS.md): Sprint task tracking and definition of done.
