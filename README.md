# ASRI

**Adaptive Sparse Recurrent Intelligence**

ASRI is a research project for building small-active-compute intelligence systems that allocate computation according to task need rather than spending roughly uniform compute on every problem.

The long-term architecture combines:

```text
small shared core
+ recurrent reasoning
+ sparse specialists
+ memory
+ tools
+ verification
+ adaptive compute control
```

The project begins much more narrowly. The first empirical question is:

> **Can a small pretrained model use weight-shared recurrent computation to improve useful-task quality per unit of inference compute, and can adaptive depth outperform uniform compute allocation at matched average cost?**

ASRI starts at research readiness **R0**. The architecture is a hypothesis; no efficiency, quality, or product claims are implied by the design.

## Research objective

ASRI aims to maximize useful task success under explicit compute and hardware budgets. A useful abstraction is:

```text
maximize expected task quality
subject to latency, FLOPs, memory, energy, and monetary cost
```

The project treats reasoning steps, specialist activation, memory access, retrieval, tool calls, verification, and candidate search as potentially allocatable computational resources.

## Current phase

**Phase 1: Recurrent Depth Frontier**

The first program isolates recurrent reasoning from every other major ASRI idea. It compares an unchanged pretrained baseline against weight-shared recurrent depth at fixed iteration counts, then—only if the frontier is promising—tests adaptive stopping at matched average compute.

No sparse expert bank, retrieval system, persistent memory, verifier, tool controller, expert offloading, or attention surgery is required to establish the first result.

## Documentation

- [`docs/FOUNDATION.md`](docs/FOUNDATION.md) — research thesis, boundaries, and north-star architecture.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — staged research path and promotion gates.
- [`docs/EVIDENCE_DOCTRINE.md`](docs/EVIDENCE_DOCTRINE.md) — measurement, claims, readiness, and anti-overclaim rules.
- [`docs/experiments/ASRI-RDF-v1.md`](docs/experiments/ASRI-RDF-v1.md) — frozen outline for the first recurrent-depth experiment.

## Relationship to ExpertOS

ASRI is a separate project. It inherits methodological lessons from ExpertOS—especially mechanism-versus-policy separation, real-runtime measurement, negative-result closure, and hardware-aware resource accounting—but it does not inherit ExpertOS readiness or evidence.

The projects may converge later: ASRI can expose controllable computational resources, while ExpertOS can eventually provide hardware-aware policy and runtime control for those resources.

## Project rule

> **Architecture must not outrun executable evidence.**

A new major primitive is introduced only when the current mechanism has produced interpretable real-model evidence or has been explicitly closed as a negative result.
