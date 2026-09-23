# Experiments

## Experimental Principle

Every experiment must answer a distinct question about spatial
representation or reasoning.

Models must not be added merely because they are popular.

The experimental hierarchy is:

    Geometry
       ↓
    Topology
       ↓
    Relational topology
       ↓
    Geographic transfer
       ↓
    Counterfactual robustness

---

## E01 — Continuous Spatial Baseline

### Question

How much of counterfactual accessibility can be predicted without explicit
graph message passing?

### Model

LightGBM

### Features

- origin coordinates
- destination coordinates
- Euclidean distance
- pre-disruption network distance
- local road density
- disruption proximity
- H3-based spatial density features

### Target

Primary:
- binary post-disruption reachability

Secondary:
- change in network distance, where computationally feasible

### Purpose

Establish the strongest practical non-graph baseline.

### Interpretation

If this baseline performs extremely well, the graph representation must
demonstrate additional value under geographic transfer or structural
perturbation rather than merely matching it in-domain.

---

## E02 — Homogeneous Graph Baseline

### Question

Does physical network topology provide predictive information beyond
engineered spatial features?

### Model

Inductive GraphSAGE

### Representation

- intersection nodes
- road connectivity edges
- continuous node/edge attributes
- disruption state

All relation types are collapsed into a homogeneous graph.

### Purpose

Separate the value of graph topology from the value of semantic relation
types.

---

## E03 — Relational Graph Model

### Question

Do typed spatial relations provide predictive information beyond
homogeneous graph connectivity?

### Model

CompGCN or another inductive relational GNN selected during implementation.

### Relations

- connects_to
- accessible_from

Disruption is represented primarily as an edge state/attribute rather than
inventing a physically unnatural relation.

### Purpose

Test whether relational semantics contribute additional transferable
information.

---

## E04 — Relation Semantics Ablation

### Question

What happens when heterogeneous relation types are removed?

### Comparison

Full relational graph:

    connects_to
    accessible_from

versus:

    generic edge

### Interpretation

The difference estimates the contribution of relation typing under an
otherwise comparable architecture.

---

## E05 — Geometry Ablation

### Question

Is the model learning transferable topology or primarily interpolating
geographic coordinates?

### Full representation

- topology
- road attributes
- coordinates
- facility attributes

### Ablated representation

- topology
- structural attributes

Coordinates are removed.

### Interpretation

A substantial performance collapse indicates strong dependence on
continuous geographic information.

Stable performance would provide evidence that structural information is
being learned.

---

## E06 — Geographic Transfer

### Question

Does the learned representation generalize to an unseen urban network?

### Protocol

Train:

    City A

Test:

    City B

There must be:

- no node overlap
- no graph overlap
- no training observations from City B
- no global preprocessing leakage

### Primary Metrics

- ROC-AUC
- PR-AUC
- reachability calibration

### Purpose

This is the primary generalization experiment.

---

## E06b — Secondary Within-City Geographic Generalization

### Question

Does the learned representation generalize to an unseen geographic region
within the same metropolitan network without cross-city domain shift?

### Protocol

Train:
    Seattle Region 1 (South/Central Seattle)

Test:
    Seattle Region 2 (North Seattle)

Exclusion:
    1,500 m buffer along the Lake Washington Ship Canal corridor.
    Origins and facilities in the buffer are strictly excluded from both splits.

### Purpose

Isolate topological transfer from city-to-city tagging or scale variance.

---

## E07 — Disruption Severity Generalization

### Question

Does the learned representation generalize across disruption magnitudes?

### Protocol

Train on a controlled range of edge-ablation severities.

Evaluate on both:

- interpolation within the training range
- extrapolation toward stronger disruptions

### Purpose

Test whether the model has learned structural consequences rather than
memorizing a fixed disruption distribution.

---

## E08 — Exact Algorithm Comparison

### Question

What is the accuracy/latency trade-off between learned prediction and
exact graph search?

### Exact baseline

Dijkstra / appropriate exact shortest-path implementation.

### Learned model

Best-performing inductive relational model.

### Measure

- inference latency
- throughput
- VRAM
- CPU/GPU utilization
- prediction error
- reachability error

### Important Constraint

Dijkstra remains the source of ground truth.

The experiment does not claim that a GNN mathematically replaces shortest
path algorithms.

It tests whether learned inference can act as a useful approximate
surrogate for repeated large-scale scenario evaluation.

---

## E09 — Message-Passing Depth Ablation

### Question

Is the model actually benefiting from multi-hop structure?

### Compare

- 1 layer
- 2 layers
- 3 layers
- optionally deeper configurations if computationally justified

### Purpose

Determine whether performance improvements arise from meaningful
multi-hop structure or local smoothing.

Monitor:

- performance
- inference cost
- representation similarity
- oversmoothing indicators

---

# TransE / RotatE

Classical KG embeddings are NOT part of the primary model competition.

They may be implemented as a P1 sanity check.

Their role is to demonstrate the limitation of transductive entity-ID-based
KGE under unseen geographic entities or regions.

A TransE/RotatE comparison is not considered a research contribution by
itself.