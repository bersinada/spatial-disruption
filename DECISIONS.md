# Decisions

## DEC-001 — Research Problem

### Decision

The project focuses on counterfactual spatial reasoning under network
disruption rather than generic spatial node classification.

### Rationale

Generic graph-versus-tabular comparison is insufficiently specific.
Counterfactual network disruption directly tests whether graph structure
supports reasoning about changes in physical connectivity.

---

## DEC-002 — Primary Spatial Representation

### Decision

The environment will be represented as a physical urban network graph
containing intersections, road segments, and critical facilities.

### Rationale

The research question concerns accessibility and network disruption.
Physical connectivity is therefore part of the underlying problem rather
than an engineered approximation.

---

## DEC-003 — Disruption Representation

### Decision

Network disruption will primarily be represented as an edge/node state
or attribute indicating whether infrastructure remains operational.

We will not model `severed_by` as an independent physical relation unless
a later experiment demonstrates that such a relation has a clear semantic
interpretation.

### Rationale

A road being closed is a state change to physical infrastructure, not
necessarily a new semantic relationship between two spatial entities.

---

## DEC-004 — Ground Truth

### Decision

Exact graph algorithms will generate ground-truth reachability and
shortest-path changes under synthetic disruptions.

### Rationale

The learned model must be evaluated against a deterministic reference.

The GNN is an approximation layer, not the definition of physical
reachability.

---

## DEC-005 — Primary Generalization Protocol

### Decision

Geographic transfer will be a primary evaluation protocol.

Training and testing cities must have disjoint spatial graphs and zero
entity overlap.

### Rationale

Random splits can exploit spatial autocorrelation and graph connectivity,
making them insufficient to establish geographic generalization.

---

## DEC-006 — Primary Representation Comparison

### Decision

The main comparison hierarchy is:

1. engineered spatial/tabular representation
2. homogeneous graph representation
3. multi-relational graph representation

### Rationale

This isolates the incremental value of:

geometry → topology → relational semantics.

---

## DEC-007 — KG Embeddings

### Decision

TransE and RotatE are not primary competing models.

They may be included as sanity-check baselines for demonstrating the
limitations of transductive KG embeddings under geographic transfer.

### Rationale

A direct TransE-versus-RotatE benchmark would mostly reproduce known
properties of the respective embedding operators.

---

## DEC-008 — One-Week Scope

### Decision

The first sprint will focus on a reproducible two-city urban disruption
benchmark.

The sprint will not attempt to build:

- a production platform
- a 3D scene graph
- GraphRAG
- an LLM spatial agent
- a full world model
- a real-time municipal deployment

### Rationale

The objective is to validate the core representation/reasoning hypothesis before expanding system scope.
---

## DEC-009 — Geographic Boundaries and Buffer Protocol

### Decision

The study areas for City A (Seattle) and City B (Portland) are defined by their
official municipal administrative boundary relations in OpenStreetMap:
- Seattle: OSM administrative boundary relation 237385
- Portland: OSM administrative boundary relation 186579

Persisted locally as:
`data/raw/<city>/boundary.geojson`

The physical road network is extracted using a 1,000 m external buffer around
the official municipal boundary.

However, all evaluated sample origin coordinates, destination facilities, and
disruption epicenters must be strictly located inside the official municipal
boundary.

### Rationale

Clipping the road network strictly at municipal boundaries (0 m buffer) artificially
distorts peripheral routing, severing border arterials and freeway interchanges.
Empirical validation demonstrated that a 0 m clip creates artificial routing
detours for 0.9% of border trips with an average penalty of 214.3 m (up to 396.9 m).
A 1,000 m buffer ensures complete route continuity while preserving municipal
study fidelity.

---

## DEC-010 — Primary Scenario-Based Train/Validation/Test Split

### Decision

The primary benchmark split is partitioned strictly by disruption scenario.
A specific physical disruption scenario (its disabled edge set and epicenter)
appears in exactly one split:
- Seattle Train: 20 scenarios
- Seattle Validation: 5 scenarios
- Seattle In-City Test: 5 scenarios
- Portland Zero-Shot Transfer Test: 10 scenarios

Normalization parameters and feature scalers are fitted exclusively on City A
training data and applied frozen to validation, test, and transfer splits.

### Rationale

This directly tests whether a model can generalize to an unseen disruption
event on a known network (in-city test) or an unseen network (geographic transfer).

Important Methodological Note:
Scenario splitting does NOT guarantee geographic independence within the same city.
The same origin node, destination facility, or intact road corridor may appear
under different disruption scenarios.

---

## DEC-011 — Benchmark Scale and Balanced Sampling Protocol

### Decision

The Week-1 benchmark dataset is fixed at 40 disruption scenarios and 4,000 samples:
- Seattle: 30 scenarios (2,000 train, 500 validation, 500 in-city test)
- Portland: 10 scenarios (1,000 zero-shot transfer test)
- Total: 4,000 samples

Each scenario contains exactly 100 OD evaluation samples:
- 50 Active Core samples: intact shortest path traverses at least one disrupted
  canonical edge.
- 50 Control Context samples: intact shortest path does NOT traverse any disrupted
  edge, stratified across short, medium, and long distance tiers.

### Rationale

In real urban networks, over 95% of random OD pairs are completely unaffected by
a localized disruption. A uniform random sample would produce a trivial benchmark
where predicting "unaffected" yields >95% accuracy. Balanced 50/50 sampling
forces models to learn structural rerouting and disconnection while maintaining
false-positive resistance on control pairs.

---

## DEC-012 — Secondary Buffered Geographic Split

### Decision

A secondary within-city spatial generalization experiment is defined for Seattle,
partitioning the city into:
- Region 1: South/Central Seattle (Train)
- Region 2: North Seattle (Spatial Test)
- Exclusion Buffer: 1,500 m along the Lake Washington Ship Canal corridor

### Rationale

Tests whether the model can generalize to unseen geographic coordinates and
localized network topology within the same metropolitan area without cross-city
domain shift. The 1,500 m exclusion buffer prevents spatial autocorrelation
and multi-hop message-passing contamination across the boundary.

---

## DEC-013 — Disruption Representation and Semantics

### Decision

Disruption is defined physically as "a road corridor becomes unavailable for
routing."

Each scenario is represented using both:
1. An explicit list of disabled canonical directed edge IDs.
2. An edge-level binary disruption mask (1 = operational, 0 = disrupted).

We do not invent a "severed_by" relation. "Severed" is an outcome label,
not a graph relation.

### Rationale

Maintains physical fidelity with real-world infrastructure damage. Disabling
edges removes traversal capacity without altering the underlying relational
schema.

---

## DEC-014 — Facility Scope and Surface-Road Snapping

### Decision

Week-1 critical facilities are restricted to emergency medical infrastructure:
hospitals and clinics (`amenity in ['hospital', 'clinic']`).

Facility coordinates are snapped to the nearest road network node incident to
surface streets, strictly filtering out elevated freeways and viaducts
(`motorway`, `trunk`, `motorway_link`, `trunk_link`).

### Rationale

Emergency medical access is a foundational accessibility domain with uniform
OSM tagging. Filtering out freeway-only nodes prevents snapping errors where a
surface clinic is incorrectly mapped to an elevated highway passing overhead.

---

## DEC-015 — Feature Leakage Contract and Permitted Model Inputs

### Decision

Before any model training (starting with E01 LightGBM baseline), a strict feature contract is enforced on the benchmark dataset.

The following fields are evaluation targets, ground-truth derivations, or scenario metadata and MUST NOT be used as model input features:
- `reachable` (Primary evaluation target)
- `relative_detour` (Secondary evaluation target)
- `disrupted_distance_m` (Directly leaks reachability and relative detour)
- `disrupt_path_hops` (Directly leaks reachability)
- `sample_type` (Stratification metadata; indicates whether intact shortest path traversed a disrupted edge)
- `split` (Split identifier)
- `scenario_id` (Scenario identifier; memorization hazard)
- `seed` (RNG seed)
- `disrupted_edge_ids` (Ground-truth list of disrupted corridors; models must consume graph structure or operational edge masks, not sample-level ground-truth lists)
- Any field directly derived from post-disruption routing or Dijkstra ground truth.

Permitted model inputs for tabular spatial baselines (E01):
- Origin node coordinates: `origin_x`, `origin_y`
- Destination facility coordinates: `destination_x`, `destination_y`
- Facility metadata: `facility_amenity`, `facility_snap_distance_m`
- Pre-disruption intact network features: Euclidean distance, intact network shortest-path distance (`original_distance_m`), intact path hops (`orig_path_hops`), local street density
- Disruption scenario hazard geometry: disruption epicenter coordinates (`x`, `y`), disruption radius (`disruption_radius_m`), Euclidean distance from origin/destination to epicenter.

Permitted model inputs for graph models (E02, E03):
- Intact graph topology with node/edge spatial attributes
- Edge operational state mask (`edge_disruption_mask`) indicating which corridors are closed
- Origin and destination node indicators/queries.

### Rationale

Using `sample_type` as a feature would trivialize reachability prediction because Active Core samples have an intact path that hit the disruption, whereas Control Context samples do not. Similarly, post-disruption distances directly encode the prediction targets. Enforcing this contract prevents data leakage.
