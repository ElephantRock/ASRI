# ASRI Research Roadmap

```text
Document: staged research path
Status: initial roadmap
Rule: later phases are conditional, not promised implementation work
```

## 1. Roadmap principle

ASRI is intentionally sequenced so that each major architectural primitive earns the right to complicate the system.

The project should not build the full north-star stack before validating the foundational mechanisms. Every phase must produce one of three outcomes:

```text
PROMOTE — evidence justifies the next layer of complexity
REFINE — bounded evidence identifies a correctable mechanism issue
CLOSE — the hypothesis is not useful enough to remain on the critical path
```

A phase may produce valuable negative evidence and still be complete.

## 2. Phase 0 — Foundation and baseline

### Objective
Create the minimum apparatus required to make the first real-model comparison credible.

### Required outputs

```text
frozen base-model identity and revision
license/provenance record
tokenizer and generation settings
hardware/software environment manifest
baseline evaluation corpus
profile / held-out / cross-domain / OOD splits where applicable
baseline quality results
baseline latency and memory results
reproducible evaluation command
result schema
```

### Exit gate
The unchanged baseline must run reproducibly before recurrent modifications are interpreted.

---

## 3. Phase 1 — Recurrent Depth Frontier

### Question
Can shared recurrent computation add useful effective depth at favorable quality/compute economics?

### Mechanism
Add one weight-shared recurrent reasoning transformation while holding every other major ASRI idea out of scope.

### Initial sweep

```text
baseline unchanged
recurrent depth 1
recurrent depth 2
recurrent depth 4
recurrent depth 8
recurrent depth 16, if prior points justify it
```

### Primary comparisons

- quality versus inference FLOPs;
- quality versus wall-clock latency;
- quality versus peak memory;
- training/recovery cost;
- degradation on protected baseline slices.

### Promotion condition
At least one recurrent configuration must establish a reproducible improvement in the quality/compute frontier or comparable measured evidence that justifies adaptive-depth research.

### Kill condition
If recovery is successful but recurrent depth provides no useful quality/compute frontier across the bounded sweep, close recurrence as the foundational ASRI mechanism and reconsider the architecture before adding experts or controllers.

---

## 4. Phase 2 — Adaptive Depth

### Question
Can learned allocation beat uniform allocation at matched average compute?

### Mechanism
Introduce a controller that selects or stops recurrent iterations based only on information available at runtime.

### Required baselines

```text
fixed-depth models at comparable mean compute
random or heuristic allocation control
maximum-depth upper bound
minimum-depth fast path
```

### Decisive test

> **Adaptive depth must outperform fixed/uniform allocation at matched average inference cost.**

A result that spends more compute and gains more quality is not sufficient evidence for adaptive allocation.

### Measurements

```text
depth distribution by workload
task difficulty versus allocated depth
overthinking / unnecessary-depth rate
premature-stop failures
controller overhead
quality per FLOP
quality per second
```

---

## 5. Phase 3 — Sparse Specialist Capacity

### Question
Can sparse specialist capacity improve capability per active compute without destroying deployment economics?

### Initial constraint
Keep the expert bank resident during the first sparse-specialist experiment. Do not mix expert placement/offload economics with the capability question.

### Measure separately

```text
quality gain
active FLOPs
total parameters
routing concentration
routing stability
expert importance
expert contribution
expert burst size
load balance
```

### Promotion condition
Sparse specialists must improve a measured capability/active-compute frontier relative to dense/recurrent controls.

### Second step
Only after this capability result exists should ASRI evaluate expert locality, placement, movement, and offload.

---

## 6. Phase 4 — Unified Compute Controller

### Question
Can heterogeneous computational actions be allocated through a common policy?

Candidate actions may include:

```text
another recurrent step
activate additional specialist capacity
retrieve external memory
call a deterministic tool
run verification
produce another candidate
terminate
```

### Cost model
Costs must be measured and state-dependent where material. The controller must not rely on arbitrary fixed prices for claim-bearing conclusions.

Conceptual objective:

```text
maximize expected task quality
- latency penalty
- compute penalty
- memory penalty
- energy penalty
- external-service cost
```

### Required ablations
Each action class must be evaluated independently before the composed controller receives credit for the combined result.

---

## 7. Phase 5 — Failure-Directed Policy Distillation

### Question
Can expensive teacher supervision be concentrated around the student's capability and efficiency frontier?

### Candidate trigger signals

```text
wrong answer
verifier disagreement
calibrated uncertainty
wrong tool choice
excessive computation
premature stopping
routing instability
protected-slice regression
```

### Training target
Distill not only final answers but efficient execution strategies:

```text
problem
student trajectory
failure classification
teacher strategy
tool decisions
minimal successful compute budget
verification outcome
alternative valid trajectory
```

### Promotion condition
Failure-directed sampling must beat a comparable generic-distillation budget on held-out task success and/or compute efficiency.

---

## 8. Phase 6 — Memory and External Computation

### Order of introduction

1. external inspectable memory;
2. retrieval policy;
3. deterministic tools;
4. conditional verification/search;
5. only later, learned persistent neural memory if evidence justifies it.

### Principle
Memory and tools should reduce the work demanded of model weights, not merely add system complexity.

---

## 9. Phase 7 — Sequence and Serving Efficiency

Potential research includes:

```text
hybrid attention
compressed KV representations
sparse/global attention scheduling
context condensation
serving-aware batching
```

These are deferred because they can materially improve serving economics while remaining largely orthogonal to the first ASRI intelligence hypothesis.

---

## 10. Phase 8 — Deployment-Aware Model/Runtime Co-Design

This is where ASRI may converge with ExpertOS-style resource control.

Training objectives may encourage, subject to protected quality:

```text
stable useful specialization
layer-local expert locality
temporal reuse
predictable routing
bounded expert bursts
hardware-friendly representation
low controller overhead
cheap fallback behavior
```

The goal is to stop treating deployability as an after-the-fact runtime problem and instead train structures the runtime can exploit safely.

---

## 11. Phase 9 — Native ASRI Model

Only after the prior mechanisms earn evidence should the project consider a native model trained or heavily reshaped around ASRI objectives.

Candidate endpoint:

```text
small shared core
+ adaptive recurrent depth
+ sparse specialist bank
+ cost-aware controller
+ external memory and tools
+ conditional verification
+ hardware-aware runtime
```

A from-scratch model is not required for ASRI success. Distillation or architectural conversion of a pretrained substrate may remain the better engineering path.

## 12. Evaluation trajectory

Early phases optimize narrow mechanism questions. Later phases must broaden toward useful workloads such as:

```text
coding and software agents
mathematical / structured reasoning
technical research
infrastructure troubleshooting
document intelligence
tool-mediated enterprise tasks
```

The long-term comparison target is not “win every benchmark.” It is:

> **Match or exceed larger-model task success on selected useful workloads at materially lower measured inference cost.**

## 13. Current critical path

```text
1. Freeze baseline substrate and evaluation contract.
2. Implement Recurrent Depth Frontier v1.
3. Run the real-model fixed-depth sweep.
4. Decide PROMOTE / REFINE / CLOSE from measured evidence.
5. Only if promoted, build Adaptive Depth v1.
```
