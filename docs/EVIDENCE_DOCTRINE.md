# ASRI Evidence Doctrine

```text
Document: canonical evidence and claim rules
Status: initial doctrine
Applies to: experiments, reports, roadmap promotion, and public claims
```

## 1. Purpose

ASRI contains several attractive architectural hypotheses. The evidence doctrine exists to prevent those hypotheses from turning into conclusions before the system has earned them.

The governing rule is:

> **A mechanism, architecture, or controller receives credit only for effects measured on the path that actually produced them.**

## 2. Progress categories

Every status report should separate:

```text
empirical_progress:
  new facts learned from real model execution

mechanism_progress:
  intervention behavior demonstrated on the real model/runtime

infrastructure_progress:
  code, schemas, evaluators, harnesses, CI, documentation
```

Infrastructure progress must not be described as capability or efficiency progress.

## 3. Claim taxonomy

| Claim | Meaning |
|---|---|
| `measured_quality_claim` | A defined task-quality effect was directly measured against a baseline. |
| `measured_runtime_claim` | Latency/throughput changed under rigorous real-runtime measurement. |
| `measured_compute_claim` | FLOPs, executed operations, active parameters, or another explicit compute quantity changed and was measured or deterministically accounted. |
| `measured_memory_claim` | Peak/resident memory or transfer volume changed and was directly measured. |
| `measured_efficiency_claim` | A quality-versus-cost frontier improved under matched or explicitly normalized resource conditions. |
| `representation_claim` | Stored representation or theoretical byte accounting changed without necessarily proving runtime benefit. |
| `oracle_claim` | Result depends on information unavailable to a real runtime. |
| `surrogate_claim` | Result uses a proxy/surrogate rather than the intended native path. |
| `negative_result` | A mechanism or policy failed its preregistered usefulness gate. |
| `not_yet_claimable` | Implementation exists, but evidence needed for the intended claim does not. |

## 4. Anti-overclaim rules

```text
Do not claim efficiency from parameter count alone.
Do not claim speedup from FLOP reduction alone.
Do not claim lower compute from fewer generated reasoning tokens alone.
Do not claim memory benefit from sparse activation alone.
Do not claim sparse deployment benefit when inactive parameters remain operationally expensive.
Do not claim adaptive allocation unless matched-average-compute controls are included.
Do not claim routing specialization from frequency alone.
Do not claim expert redundancy from similarity alone.
Do not claim controller value without including controller overhead.
Do not claim system value from an oracle policy.
Do not use component evidence as a composed-system claim.
Do not generalize beyond the evaluated model, workload, hardware, and software envelope.
```

## 5. Runtime measurement rigor

A claim-bearing runtime comparison should include:

```text
synchronized device execution
identical model/checkpoint conditions except the intervention
identical workload and generation settings
warm-up policy
multiple trials
trial-level results
latency distribution, not a single mean
uncertainty / confidence interval
recorded hardware and software environment
quality checks for the compared conditions
```

Where practical, use paired comparisons inside the same process or tightly matched fresh-process trials.

A latency improvement that is not distinguishable from measurement noise remains `not_yet_claimable` as a speedup.

## 6. Efficiency comparisons

ASRI's central claims require stronger controls than raw benchmark scores.

Preferred comparisons include:

```text
quality at matched average FLOPs
quality at matched average latency
compute required to reach a fixed quality target
latency required to reach a fixed quality target
quality per joule
quality per dollar
```

For adaptive computation, report the entire allocation distribution rather than only the mean.

A controller that improves quality only by spending materially more average compute has not yet demonstrated efficient allocation.

## 7. Quality protection

Every experiment should distinguish:

```text
target workloads
protected baseline capabilities
held-out evaluation
cross-domain evaluation
OOD / negative-control evaluation where relevant
```

When an intervention benefits one slice and harms another, report both. Aggregate score improvements must not hide protected-slice regressions.

## 8. Readiness ladder

Readiness measures empirical maturity, not implementation volume.

| Level | Meaning |
|---|---|
| R0 | Hypothesis/design only. |
| R1 | Synthetic or unit-level mechanism evidence. |
| R2 | Real pretrained model, offline or non-claim-bearing analysis. |
| R3 | Isolated real-model/runtime intervention measured against a baseline. |
| R4 | Reproducible validator-gated mechanism with bounded support envelope. |
| R4.5 | Composed real-runtime path measured end-to-end with quality and cost evaluation. |
| R5 | Controller/policy may select the mechanism within its validated envelope. |
| R6 | Stable externally usable research/product surface with explicit support boundaries. |

ASRI begins at **R0**.

No phase may inherit readiness from ExpertOS, a teacher model, a base model, or a related paper.

## 9. Promotion invariant

A major mechanism advances only when its real execution evidence supports the next research question.

Examples:

```text
recurrent block implemented + unit tests pass
  != recurrent depth is useful

adaptive controller predicts depth
  != adaptive compute is efficient

sparse router activates few experts
  != sparse deployment is cheaper

memory system retrieves relevant items
  != memory improves task economics
```

## 10. Apparatus-to-evidence rule

ASRI adopts a maximum default apparatus-to-evidence ratio of **1:1**:

> After a real execution attempt, complete at most one major governance/infrastructure milestone before the next real execution attempt, unless a concrete execution blocker requires otherwise.

Examples of apparatus:

```text
schemas
validators
documentation
CI
new abstractions
benchmark plumbing
controller interfaces
```

Examples of evidence:

```text
real baseline run
real intervention run
failed recovery run
negative benchmark result
hardware/runtime characterization
```

Failed real executions count as evidence when the failure is recorded precisely.

## 11. Required experiment record

Every claim-bearing experiment should record:

```text
experiment ID
source revision
base-model ID and revision
model modifications
training data/provenance
training configuration
runtime backend
hardware/software environment
workload and split identity
baseline result
intervention result
resource measurements
quality measurements
validator/evaluator result
limitations
claim classification
next allowed action
```

## 12. Negative results

A negative result is complete when it identifies a useful boundary and prevents repeated work.

Valid closure examples:

```text
recurrent depth recovers quality but does not improve the compute frontier
adaptive stopping does not beat fixed depth at matched average compute
sparse experts improve active FLOPs but total serving cost regresses
retrieval improves factuality but latency/cost exceeds the declared budget
```

The project may then change architecture instead of tuning indefinitely.

## 13. Baseline-first rule

No architectural intervention is interpreted until the unchanged substrate has a frozen reproducible baseline on the same evaluation path.

The baseline is a first-class experimental condition, not a historical number copied from a model card or external benchmark.

## 14. Public-facing rule

ASRI documentation may discuss hypotheses, targets, and research directions freely, but must label them as such.

Public claims about superiority, efficiency, cost, memory, reasoning quality, or frontier competitiveness require corresponding measured evidence in the repository.
