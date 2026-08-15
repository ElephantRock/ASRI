# ASRI Research Foundation

```text
Document: canonical project foundation
Status: initial research charter
Project state: R0 — hypothesis only
Inception: 2026-08-15
```

## 1. Purpose

ASRI investigates whether useful intelligence can be produced more efficiently by dynamically assembling computation from specialized primitives instead of applying roughly uniform neural computation to every task.

The fixed research goal is:

> **Build small-active-compute intelligence systems that spend additional computation only when the expected task benefit justifies the cost.**

The project is not defined by a particular parameter count, expert count, tokenizer, attention mechanism, or base model. Those are experimental choices.

## 2. Central hypothesis

A conventional model tends to couple representation, knowledge, reasoning, memory, calculation, and verification inside one large repeated neural stack. ASRI tests whether these functions can be factored into resources that are invoked conditionally.

North-star abstraction:

```text
representation
+ recurrent reasoning
+ sparse specialization
+ memory
+ retrieval
+ tools
+ search
+ verification
+ learned compute allocation
```

The controller problem can be expressed conceptually as:

```text
choose actions that maximize expected task quality
minus the measured cost of the computation used
```

Actual ASRI experiments must use measured hardware/runtime costs rather than arbitrary virtual prices whenever a cost claim is made.

## 3. North-star architecture

The eventual architecture may contain:

### Shared core
A compact pretrained language/reasoning substrate responsible for general representation and communication.

### Recurrent reasoning core
A weight-shared transformation reusable for variable effective depth. Additional reasoning should increase computation without requiring proportional parameter growth.

### Sparse specialist bank
Conditionally activated neural capacity that may specialize by task, representation, operation, or other learned structure.

### Compute controller
A policy that decides whether the current state warrants additional recurrence, specialist activation, retrieval, tools, verification, search, or termination.

### External memory
Inspectible persistent and working memory that can be retrieved conditionally rather than repeatedly inserted into every context.

### Tools
Deterministic or external computation used when it is cheaper, more reliable, or more exact than neural approximation.

### Verification
Conditional evaluation of candidate outputs, tool results, proofs, tests, or alternative trajectories.

This is a destination architecture, not the v1 implementation plan.

## 4. Research questions

ASRI is organized around progressively harder questions:

1. **Recurrent depth:** Can weight sharing across reasoning depth improve quality per unit of compute?
2. **Adaptive depth:** Can a controller allocate recurrent steps better than uniform compute at matched average cost?
3. **Sparse specialization:** Can specialist capacity improve quality per active FLOP without making total deployment economics worse?
4. **Resource allocation:** Can recurrence, experts, memory, tools, and verification be selected through one cost-aware policy?
5. **Policy learning:** Can failure-directed distillation teach not only answers but efficient execution strategies?
6. **Deployment-aware structure:** Can training shape routing, reuse, locality, predictability, and representation so the model is natively cheaper to serve?
7. **System competitiveness:** On chosen workloads, can the composed system match much larger models at materially lower measured inference cost?

Each question requires independent evidence. A positive answer to an earlier question does not imply later questions are solved.

## 5. Starting constraints

The first phase deliberately excludes:

```text
large sparse expert banks
expert offload or residency policies
persistent neural memory
RAG as a capability crutch
multi-candidate search
verifier ensembles
tool-use optimization
attention replacement
KV-cache redesign
byte-level modeling
from-scratch pretraining
```

These may become later programs only after the preceding mechanism produces interpretable evidence.

## 6. Starting substrate

The initial substrate should be a strong permissively usable pretrained model in approximately the 4–8B class.

The exact model, revision, tokenizer, dtype, backend, and license must be frozen in the first executable experiment before results are interpreted.

ASRI begins from pretrained capability because the first research question concerns computation structure, not whether the project can afford to relearn language and world knowledge from random initialization.

## 7. Relationship to ExpertOS

ASRI and ExpertOS are sibling research projects with different directions of attack.

```text
ExpertOS:
existing MoE -> measure -> manipulate -> reshape

ASRI:
desired compute behavior -> design -> train -> measure -> adapt
```

ASRI inherits several methodological lessons:

- mechanism correctness is not policy usefulness;
- sparse activation is not automatically cheap deployment;
- memory savings and runtime savings are separate claims;
- controller overhead must be measured;
- routing frequency is not equivalent to importance;
- locality, predictability, and prefetchability are different properties;
- negative results should close unproductive branches;
- infrastructure completion is not empirical progress.

ASRI does **not** inherit ExpertOS readiness, validated support envelopes, or product claims.

## 8. Project principles

```text
Architecture must not outrun executable evidence.
One major unknown should be isolated whenever practical.
Claims must match what was directly measured.
Matched-compute comparisons are preferred over unbounded benchmark comparisons.
Negative results are first-class outcomes.
Hardware economics are part of the architecture, not a postscript.
No optimization is a valid result when no intervention earns promotion.
```

## 9. Long-term success criterion

The project should ultimately be judged on useful task success under measured resource constraints, not parameter count alone.

Primary families of metrics include:

```text
quality / task success
quality per inference FLOP
quality per joule
quality per second
quality per dollar
peak memory
memory bandwidth / transfer cost
active parameters
latency distribution
```

A smaller model that requires expensive search, repeated retries, or excessive memory movement is not efficient merely because its active parameter count is low.

## 10. Current authorization

The only major architectural program authorized by this foundation is:

> **ASRI Recurrent Depth Frontier v1**

Its contract lives in `docs/experiments/ASRI-RDF-v1.md`.

Sparse specialists and the broader compute controller remain roadmap items until recurrent-depth evidence is available or the recurrent hypothesis is explicitly closed.
