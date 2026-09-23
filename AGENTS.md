# Research Execution Protocol

This repository implements an empirical Spatial Intelligence research
project.

The agent is an implementation agent.

The agent does not use LaTeX ($ or $$) for pipelines, diagrams, or math notation; use plain Markdown lists, Mermaid diagrams, or unicode arrows (→) instead.

The agent must not silently change the research question, experimental
protocol, graph schema, evaluation split, or target definition.

If implementation constraints require a change to the research design:

1. identify the conflict;
2. explain the technical consequence;
3. propose the smallest change;
4. record the decision in DECISIONS.md only after approval.

## Research Priority

Optimize for information gained per unit of engineering effort.

A working experiment is more valuable than an elaborate architecture that
has not been tested.

Do not add models merely because they are popular.

Every model must answer a distinct experimental question.

## Representation Hierarchy

The core comparison is:

    Spatial features
          ↓
    Physical topology
          ↓
    Relational topology
          ↓
    Geographic transfer

The purpose is to identify what information each representation adds.

## Reproducibility

Every experiment must record:

- dataset version
- configuration
- random seed
- model configuration
- spatial split
- metrics
- runtime
- hardware
- commit hash where practical

## Leakage Prevention

Never:

- randomly split spatial observations without justification;
- allow message passing across train/test boundaries;
- compute global spatial statistics before partitioning;
- fit normalization/scaling parameters using test data;
- use test-city graph information during training.

## Ground Truth

Exact graph algorithms may be used to generate labels.

Learned models must not be evaluated against labels generated using leaked
test information.

The learned model is an approximation to the ground-truth computation.

## Code Before Abstraction

Implement the smallest end-to-end experiment first.

Do not create a large framework before a single real sample can pass
through:

    raw data
      ↓
    graph
      ↓
    disruption
      ↓
    ground truth
      ↓
    model
      ↓
    metric

## Agent Behavior

When blocked, prefer:

1. simplify;
2. reduce dataset size;
3. reduce model complexity;
4. test on a small real subset;
5. measure the failure;
6. then scale.

Do not compensate for an unclear research question with additional code.