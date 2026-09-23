# Projects

## Project: Inductive Spatial Reasoning Under Network Disruption

### Status

Active — Week 1 Research Engineering Sprint

### Mission

Build a reproducible Spatial Intelligence benchmark that tests whether
relational graph representations can learn transferable structural patterns
of urban accessibility under counterfactual network disruptions.

### Core Research Question

Can an inductive relational graph representation learn transferable
structural patterns of counterfactual urban accessibility under network
disruptions, beyond what continuous spatial features and homogeneous graph
message passing provide?

### What This Project Is NOT

This project does not attempt to demonstrate that:

- graphs are generally better than tabular models;
- GNNs are universally superior to classical ML;
- RotatE is better than TransE;
- graph neural networks replace shortest-path algorithms;
- graph representations are inherently more intelligent than geometric
  representations.

These questions are either too broad, already substantially studied, or
not sufficiently meaningful for the objective of this project.

### Core Idea

Represent an urban environment as a physical spatial graph.

Then introduce controlled counterfactual disruptions:

    intact network
          ↓
    localized edge ablation
          ↓
    disrupted network
          ↓
    accessibility change

The system must reason about whether an origin remains connected to a
critical facility and, optionally, how network travel distance changes.

Ground-truth accessibility changes are generated using exact graph
algorithms such as Dijkstra. Learned models are evaluated as approximate
predictive systems rather than as replacements for the definition of
ground truth.

### Representation Layers

The project explicitly separates:

1. Continuous spatial information
   - coordinates
   - distances
   - road lengths
   - local density

2. Physical topology
   - intersections
   - road connectivity
   - network paths

3. Relational semantics
   - connects_to
   - accessible_from
   - disruption state

The central experimental question is whether adding these representational
layers provides transferable information.

### Primary Hypothesis

Inductive relational graph models can learn structural patterns of
counterfactual urban accessibility that transfer across geographically
disjoint urban environments more effectively than models relying primarily
on continuous spatial features or homogeneous graph message passing.

### Secondary Questions

1. How much predictive information is contained in continuous spatial
   features alone?

2. How much additional information is provided by physical network
   topology?

3. Do typed spatial relations provide information beyond homogeneous
   connectivity?

4. How dependent is the learned model on explicit geographic coordinates?

5. Does the learned representation transfer to a geographically distinct
   urban network?

6. What is the accuracy/latency trade-off between learned prediction and
   exact graph search?

### Long-Term Direction

This project is a technical seed for a broader Spatial Intelligence system
capable of representing physical environments, simulating counterfactual
changes, and reasoning about their consequences.

Potential future domains include:

- infrastructure resilience
- climate disruption analysis
- logistics and accessibility
- autonomous mobility
- industrial digital twins
- robotic spatial reasoning

The one-week project does not attempt to build these products.

Its purpose is to validate the underlying representation and reasoning
capability.
---

## Experimental Benchmark Design (Week 1 Locked)

### Study Area & Cities
- City A (Training & In-City Validation): Seattle, WA (OSM relation 237385)
- City B (Zero-Shot Transfer Test): Portland, OR (OSM relation 186579)
- Road network extracted using a 1,000 m external buffer around official municipal boundaries.
- All evaluated origins, destinations, facilities, and disruption epicenters are strictly located inside the official municipal boundary.

### Target Definitions
- Primary Target: Post-disruption reachability (binary: 1 = reachable, 0 = unreachable).
- Secondary Target: Conditional relative detour (disrupted_distance - original_distance) / original_distance for reachable pairs (null if unreachable).
- Ground Truth: Deterministic Dijkstra shortest paths on canonical physical road edge lengths.

### Benchmark Scale & Scenarios
- Total Scenarios: 40 disruption scenarios.
  - Seattle: 20 train, 5 validation, 5 in-city test (30 total Seattle).
  - Portland: 10 zero-shot transfer test.
- Samples per Scenario: Exactly 100 OD pairs:
  - 50 Active Core samples (intact path traverses at least one disrupted canonical edge).
  - 50 Control Context samples (intact path does not traverse any disrupted edge, stratified across distance terciles).
- Benchmark Total: 4,000 samples across 40 scenarios.

### Evaluation Tiers
- Tier 1: In-City Unseen Disruption Generalization (Seattle Test, 5 scenarios / 500 samples).
- Tier 2: Zero-Shot Geographic Transfer (Portland Transfer, 10 scenarios / 1,000 samples).
- Tier 3: Secondary Within-City Geographic Generalization (Seattle South/Central vs North with 1,500 m exclusion buffer).

### Methodological Safeguards & Limitations
- Scenario splitting does NOT guarantee geographic independence within the same city; the same origin node or intact corridor may appear across scenarios.
- True geographic generalization is rigorously evaluated via Tier 2 (City B transfer) and Tier 3 (buffered within-city spatial split).
- Normalizers and preprocessing scalers are fitted exclusively on City A training data and applied frozen to all evaluation splits.

### Feature Leakage Contract (Pre-E01 Safety Constraint)
To prevent subtle information leakage across all predictive models (DEC-015):
- **Strictly Forbidden Inputs**: `reachable`, `relative_detour`, `disrupted_distance_m`, `disrupt_path_hops`, `sample_type`, `split`, `scenario_id`, `seed`, `disrupted_edge_ids`, and any post-disruption ground truth.
- **Permitted Inputs**: Pre-disruption graph topology, origin/destination coordinates and facility attributes, intact network shortest-path distance (`original_distance_m`), intact hops (`orig_path_hops`), and disruption hazard geometry (epicenter coordinates and radius) or binary edge operational masks (`edge_disruption_mask`).
- **Feature Scaling**: All normalizers must be fit exclusively on the Seattle Train split and applied frozen to validation, in-city test, and Portland transfer splits.
