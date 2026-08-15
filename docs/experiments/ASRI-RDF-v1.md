# ASRI Recurrent Depth Frontier v1

```text
Experiment ID: ASRI-RDF-v1
Status: AUTHORIZED FOR IMPLEMENTATION
Claim state before execution: not_yet_claimable
Readiness: R0 -> potential R3
Primary question: Does weight-shared recurrent reasoning improve the quality/compute frontier of a small pretrained model?
```

## 1. Objective

Test the smallest load-bearing ASRI hypothesis before introducing sparse specialists, memory, tools, verification, or a general compute controller.

The experiment asks two sequential questions:

1. **Fixed recurrence:** Does additional weight-shared recurrent depth produce useful capability gains at favorable measured compute cost?
2. **Adaptive recurrence:** Only if question 1 is positive, can runtime allocation of recurrent depth outperform uniform allocation at matched average compute?

Question 2 belongs to the follow-on ASRI Adaptive Depth program. This v1 experiment should collect the evidence needed to authorize or reject that follow-on.

## 2. Hypotheses

### H1 — recurrent scaling
At least part of the evaluated workload should improve as recurrent depth increases after recovery/adaptation training.

### H2 — compute frontier
At least one recurrent configuration should improve quality at matched or normalized inference compute relative to the unchanged substrate strongly enough to justify adaptive-depth research.

### H3 — bounded degradation
Recurrent conversion should not require unacceptable regressions on protected baseline capabilities to obtain the target gains.

H1 alone is insufficient. A model that becomes better only by spending proportionally more compute may be scientifically interesting but does not yet establish ASRI's efficiency thesis.

## 3. Explicit non-goals

This experiment does not test:

```text
sparse MoE experts
expert specialization
expert residency/offload
RAG or external memory
tool use
verification/search
latent-only reasoning claims
KV-cache compression
hybrid attention
frontier-model competitiveness
from-scratch training
```

No conclusion about those mechanisms may be drawn from RDF-v1.

## 4. Base-model selection contract

Before implementation results are interpreted, freeze:

```text
model family
exact checkpoint/revision
license
parameter count
tokenizer identity
context configuration
dtype/quantization
attention/backend implementation
training-framework revision
runtime-framework revision
```

Initial target class:

```text
strong pretrained/instruction model
approximately 4–8B parameters
small enough for repeated local experimentation
strong enough that the experiment tests architecture rather than basic language acquisition
```

Model selection should favor reproducibility, modifiability, permissive use, and compatibility with activation/gradient instrumentation.

## 5. Baseline contract

The unchanged model must be evaluated first through the exact harness intended for recurrent variants.

Record:

```text
quality by workload/slice
prompt/token counts
generation length
prefill latency
decode latency or task latency
peak allocated/reserved memory
inference FLOPs or deterministic operation accounting where practical
throughput
hardware/software environment
```

External model-card benchmark numbers are context only and cannot serve as the experiment baseline.

## 6. Recurrent mechanism

The preferred first mechanism is deliberately simple:

```text
input hidden state
      ↓
shared recurrent reasoning transformation R
      ↓
R(h), R²(h), R⁴(h), ...
      ↓
normal decoding/output path
```

The exact insertion point is an implementation decision that must be frozen before the claim-bearing sweep.

Required properties:

```text
weights shared across recurrence
recurrence count explicit and logged
same recurrent weights used at every iteration
no hidden teacher/oracle signal at inference
deterministic depth when fixed-depth condition is selected
ability to disable recurrence and reproduce the baseline path or a separately defined conversion control
```

## 7. Conversion and recovery training

Architectural surgery may damage pretrained capability. Separate recovery from the recurrent-depth claim.

Recommended sequence:

```text
A. unchanged baseline
B. converted architecture at minimal recurrence
C. recovery/adaptation training
D. frozen checkpoint for depth sweep
```

Record recovery curves and training cost.

Do not hide a large recovery budget when reporting inference efficiency. Training cost is a separate metric from inference cost, but it remains part of system economics.

## 8. Fixed-depth sweep

Initial requested depths:

```text
0 / unchanged baseline where architecturally comparable
1
2
4
8
16 only if the preceding frontier warrants it
```

If architecture conversion prevents exact depth-0 equivalence, maintain two controls:

```text
original unchanged model
converted minimal-depth control
```

This distinguishes conversion damage from recurrence effects.

## 9. Workload structure

The first suite should contain several complementary workload families without becoming a broad benchmark project.

Preferred initial families:

```text
structured reasoning / mathematics
code reasoning or debugging
instruction following / general capability protection
short-form factual or language controls
```

Where feasible maintain:

```text
profile/development split
held-out evaluation split
cross-domain slice
OOD or negative-control slice
```

The development split may guide implementation. Claim-bearing selection must use held-out data.

## 10. Measurements

### Quality

```text
exact-match / task-specific success
pass@1 where appropriate
protected-slice regression
failure categories
```

### Compute

```text
recurrent iterations
active parameter count by condition
estimated or measured FLOPs
executed operation counts where available
generated tokens
```

### Runtime

```text
prefill latency
per-token/decode latency where applicable
end-to-end task latency
throughput
controller overhead = zero in fixed-depth v1 by design
```

### Memory

```text
peak allocated VRAM
peak reserved VRAM
model resident bytes
activation-memory impact by depth where measurable
```

### Training economics

```text
recovery steps/tokens
training FLOPs or GPU-hours
checkpoint storage
```

## 11. Primary analysis

Plot and report frontiers rather than a single preferred depth:

```text
quality vs recurrent depth
quality vs inference FLOPs
quality vs latency
quality vs peak memory
marginal quality gain per additional recurrent step
```

Look specifically for:

```text
diminishing returns
saturation
regression at excess depth
domain-specific depth sensitivity
latency overhead not explained by FLOPs
memory growth
```

## 12. Comparators

Minimum:

```text
unchanged base model
converted minimal-depth control
fixed recurrent depths
```

If feasible, include a parameter-matched or compute-matched non-recurrent control to distinguish recurrence from simply adding capacity or repeated computation.

## 13. Promotion decision

The closure must issue one of:

### `RECURRENT_FRONTIER_PROMOTE`
Use when recurrent depth produces reproducible evidence of a useful quality/compute frontier that justifies adaptive-depth allocation research.

### `RECURRENT_FRONTIER_REFINE`
Use when the mechanism shows credible benefit but conversion/recovery/runtime overhead prevents a clean conclusion and a bounded corrective experiment is identifiable.

### `RECURRENT_FRONTIER_CLOSE`
Use when recovery is adequate but the recurrent mechanism fails to produce useful quality/compute economics across the bounded sweep.

### `EXPERIMENT_INVALID`
Use when provenance, baseline, measurement, recovery, or implementation integrity prevents interpretation.

## 14. Promotion requirements

`RECURRENT_FRONTIER_PROMOTE` requires all of:

```text
real pretrained model executed
recurrent intervention proven active
held-out quality result
measured or defensible compute accounting
real runtime measurements
protected-slice evaluation
repeated trials for latency claims
reproducible source/environment record
no material unexplained baseline mismatch
```

In addition, the result must show more than “more compute gives more quality.” It must provide credible evidence that weight-shared recurrent depth is a useful substrate for later **adaptive** compute allocation.

Exact numerical thresholds may be frozen after baseline characterization but **before held-out recurrent results are examined**.

## 15. Kill criteria

Close or hold the mechanism when:

```text
baseline cannot be reproduced
converted control loses capability that recovery cannot restore
recurrence causes unstable training or inference without a bounded fix
quality is flat or degrades across useful depths
latency overhead dominates theoretical compute benefit
protected capabilities regress materially
reported gains exist only on development prompts
measurement path cannot distinguish recurrence from another intervention
```

## 16. Evidence artifacts

The implementation should eventually produce a tracked structure resembling:

```text
evidence/asri_rdf_v1/
  environment.json
  model_manifest.json
  dataset_manifest.json
  training_manifest.json
  baseline_results.json
  depth_1_results.json
  depth_2_results.json
  depth_4_results.json
  depth_8_results.json
  depth_16_results.json        # optional
  frontier_analysis.json
  validator_result.json
  closure.md
```

Exact schema names may evolve before execution but must freeze before claim-bearing interpretation.

## 17. Definition of done

RDF-v1 is complete when it can answer:

```text
1. Did the recurrent transformation preserve/recover the base model sufficiently?
2. How does task quality change with recurrent depth?
3. How does actual inference cost change with recurrent depth?
4. Is there a better quality/compute frontier than the unchanged/converted controls?
5. Which workloads benefit, saturate, or regress?
6. Is adaptive-depth research authorized next?
```

No sparse-specialist implementation is required to complete this program.
