# ASRI Phase 0 — Substrate and Baseline Contract v1

```text
Program: ASRI-P0-v1
Status: AUTHORIZED FOR EXECUTION
Project readiness before execution: R0
Purpose: freeze the first model substrate, evaluation surface, environment record, and unchanged baseline before recurrent modification
Successor: ASRI-RDF-v1
```

## 1. Objective

Phase 0 exists to make the first recurrent-depth result interpretable.

It must answer four questions before architectural surgery begins:

```text
1. What exact pretrained model is ASRI modifying?
2. What exact workloads and scorers define the initial quality surface?
3. What hardware/software path defines inference cost?
4. Can the unchanged model be reproduced reliably through that path?
```

Phase 0 does not claim that the selected substrate is globally best. It selects a practical, modifiable, reproducible starting point for the first ASRI mechanism test.

## 2. Primary substrate decision

### Selected substrate

```text
model_id: Qwen/Qwen3-4B
revision: 1cfa9a7208912126459214e8b04321603b3df60c
model_type: dense causal language model
parameter_class: 4B
license: Apache-2.0
initial runtime: Hugging Face Transformers / PyTorch
thinking_mode_for_ASRI_baseline: disabled
quantization_for_claim-bearing baseline: none unless hardware forces a separately declared fallback condition
```

Primary sources:

- https://huggingface.co/Qwen/Qwen3-4B
- https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c

### Why this substrate

Qwen3-4B is selected because it combines:

```text
small enough size for repeated experimental iteration
strong post-trained capability rather than a weak architecture-only toy
Apache-2.0 licensing
native Qwen3 support in modern Transformers
standard causal-LM structure suitable for intervention/instrumentation
an explicit hard switch between textual thinking and non-thinking modes
```

The ASRI baseline must use `enable_thinking=False` in the chat template. This prevents Qwen3's variable textual chain-of-thought path from becoming an uncontrolled test-time-compute dimension while recurrent depth is being studied.

The reasoning-capable thinking mode may be measured later as a contextual comparator, but it is not the Phase-0 baseline and must not be mixed into the primary recurrent-depth claim.

## 3. Secondary substrate register

Secondary candidates are recorded now so a primary-substrate failure can be diagnosed without post-hoc model shopping.

| Candidate | Role | Advantages | Main concern | Status |
|---|---|---|---|---|
| `microsoft/Phi-4-mini-instruct` | first alternate | ~3.8B, MIT license, reasoning-oriented post-training | model card currently uses a custom-code loading path, increasing surgery/instrumentation surface | HOLD |
| `Qwen/Qwen3-8B` | scale control | same family, Apache-2.0, native Transformers support | roughly doubles model size and slows experimental iteration | HOLD |
| `google/gemma-3-4b-it` | architecture diversity control | 4B-class strong open-weight model | Gemma terms/gated access and multimodal model surface add nonessential Phase-0 complications | HOLD |

Primary sources:

- https://huggingface.co/microsoft/Phi-4-mini-instruct
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/google/gemma-3-4b-it

No alternate may replace the primary substrate after recurrent held-out results are visible unless the Qwen3-4B program is first closed as invalid or unsuitable with a recorded reason.

## 4. Model-freeze requirements

Before baseline results are claim-bearing, persist a machine-readable model manifest containing at minimum:

```text
model_id
model_revision
license identifier
config hash
tokenizer file hashes
generation-config hash
safetensor index hash
weight shard hashes or an immutable artifact digest
parameter count
layer count
hidden size
attention configuration
torch dtype
chat-template hash
thinking-mode setting
context cap used by the experiment
```

The experiment should not rely on the moving `main` revision after this manifest is produced.

## 5. Phase-0 inference mode

The baseline is deliberately conservative.

```text
backend: PyTorch + Transformers
precision: BF16 when supported by the selected hardware; otherwise record the native safe dtype explicitly
generation: model post-training path with thinking disabled
context cap: 4096 tokens for the first benchmark pass unless a workload requires less
max output: workload-specific, frozen in the dataset manifest
batching: batch 1 for latency characterization; optional throughput runs reported separately
KV cache: enabled
external tools: disabled
retrieval: disabled
verification: disabled
speculative decoding: disabled
quantization: disabled for the primary mechanistic baseline
compile/fused experimental transforms: disabled unless part of the frozen unchanged runtime
```

The objective is not maximum serving performance. It is a stable reference path that can survive architectural intervention with minimal hidden differences.

## 6. Hardware contract

The exact hardware is not assumed by the documentation. The first real run must freeze the machine actually used.

Required environment fields:

```text
GPU model
physical VRAM
CPU model
system RAM
OS/kernel
CUDA runtime
GPU driver
PyTorch version
Transformers version
attention implementation
Python version
relevant compiler/runtime versions
power-management state where observable
```

### Hardware rule

Do not change GPU class between the unchanged baseline and the claim-bearing recurrent sweep.

If training/recovery requires a larger training machine, that is allowed, but inference comparisons must return to the frozen evaluation hardware and the training environment must be recorded separately.

## 7. Evaluation surface

The initial suite is intentionally compact. It should expose reasoning sensitivity while preserving general behavior without turning Phase 0 into a leaderboard project.

### A. Mathematical reasoning — MATH-500

Purpose:

```text
high-signal structured reasoning
short enough for repeated depth sweeps
automatically scoreable final answers
```

Reference implementation/source:

- https://github.com/openai/simple-evals

Use the MATH-500 task definition as a reference, but freeze ASRI's exact prompt, answer extraction, and scorer locally. External leaderboard numbers are context only.

### B. Code correctness — EvalPlus HumanEval+

Purpose:

```text
reasoning plus executable correctness
stronger test coverage than original HumanEval
compact enough for repeated pass@1 evaluation
```

Primary source:

- https://github.com/evalplus/evalplus

Initial Phase-0 code metric:

```text
HumanEval+ pass@1
one generated candidate per task
sandboxed execution
fixed timeout and Python environment
```

MBPP+ is a later expansion if the first suite lacks enough code variance.

### C. Instruction-following protection — IFEval

Purpose:

```text
protect post-trained instruction adherence
catch recurrence-induced format/instruction regressions
use deterministic strict/loose constraint scorers
```

Primary source:

- https://github.com/google-research/google-research/tree/master/instruction_following_eval

Because public benchmark issues can evolve, pin the exact evaluator commit and input-data hash in the dataset manifest rather than relying on repository `master`.

### D. Broad reasoning/control — MMLU-Pro bounded subset

Purpose:

```text
cross-domain capability protection
multiple-choice control with low generation cost
category-level regression visibility
```

Primary source:

- https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

MMLU-Pro currently contains validation and test splits. ASRI must freeze a deterministic bounded subset for Phase 0 rather than using the entire benchmark on every recurrent iteration.

## 8. Dataset partitioning

The claim-bearing corpus must not double as the tuning set.

Use four roles:

```text
development:
  small visible subset for harness debugging and architecture debugging

baseline-characterization:
  fixed subset used to understand score variance and freeze promotion thresholds

held-out:
  untouched until recurrent mechanism and thresholds are frozen

negative/control:
  tasks expected not to benefit materially from additional recurrent reasoning, used to detect unnecessary degradation
```

Public benchmark provenance remains visible, but ASRI's internal assignment of benchmark items to these roles must be deterministic and hash-persisted.

No recovery/adaptation training may consume held-out evaluation items or their labels.

## 9. Generation and scoring policy

Phase 0 must separate deterministic scoring from sampling variance.

### Primary quality pass

Use a single frozen generation configuration per workload. The selected configuration must be established from the unchanged model before recurrent held-out evaluation.

For Qwen3 non-thinking mode, the model card recommends a sampling configuration rather than greedy decoding. ASRI should therefore record the exact seed and sampling parameters and repeat any condition where stochasticity materially affects the conclusion.

### Secondary deterministic diagnostic

Where useful, maintain a greedy or temperature-zero diagnostic only as an implementation consistency control. It is not automatically the quality baseline if it conflicts with the model's documented operating mode.

## 10. Baseline measurements

Every benchmark condition must emit:

### Quality

```text
task success / score
per-item verdict
workload/category breakdown
generated-token count
failure classification where practical
```

### Runtime

```text
prefill latency
decode or generation latency
end-to-end latency
throughput where relevant
trial count
mean/median
p90/p95 when sample count supports it
confidence interval or dispersion statistic
```

### Memory

```text
model resident bytes
peak allocated VRAM
peak reserved VRAM
host RSS if practical
```

### Compute

```text
input tokens
output tokens
model forward count
FLOP estimate or deterministic operation-accounting method
active parameter count
```

## 11. Timing rigor

For CUDA timing:

```text
warm up before claim-bearing trials
synchronize GPU work at timing boundaries
repeat conditions
record individual trial values
keep baseline and intervention settings identical except for the declared mechanism
avoid mixing model-loading time with steady inference unless startup latency is explicitly being measured
```

Wall-clock dispatch time without completion synchronization is not an ASRI runtime claim.

## 12. Phase-0 baseline acceptance gates

The baseline is accepted only when all of the following hold:

```text
MODEL_FREEZE_PASS
  exact model/tokenizer/config provenance persisted

HARNESS_PASS
  every selected workload executes end to end

SCORER_PASS
  known-answer or fixture checks pass for each scorer

REPRODUCIBILITY_PASS
  rerunning a bounded fixed subset reproduces verdicts under the frozen seed/configuration

RUNTIME_MEASUREMENT_PASS
  synchronized repeated latency collection works and emits trial-level values

MEMORY_MEASUREMENT_PASS
  peak memory fields are captured on the frozen hardware

EVIDENCE_WRITE_PASS
  manifests and results are git-trackable and schema-valid
```

Phase 0 does not require a particular benchmark score. It requires a trustworthy reference state.

## 13. Phase-0 outputs

Expected tracked structure:

```text
evidence/asri_p0_v1/
  model_manifest.json
  environment.json
  dataset_manifest.json
  generation_manifest.json
  scorer_manifest.json
  baseline_quality.json
  baseline_runtime.json
  baseline_memory.json
  reproducibility.json
  validator_result.json
  closure.md
```

Implementation may refine schemas before the first real run, but schema work should remain the minimum required to execute and judge the baseline.

## 14. Exit verdicts

Phase 0 must close with exactly one of:

### `BASELINE_READY`

The Qwen3-4B substrate and evaluation path are sufficiently reproducible to authorize ASRI-RDF-v1 architectural surgery.

### `BASELINE_REFINE`

A bounded technical defect prevents trustworthy comparison and the smallest corrective action is known.

### `SUBSTRATE_REJECTED`

The selected substrate is unsuitable for the recurrent experiment for a concrete reason such as architecture incompatibility, unrecoverable runtime constraints, licensing/provenance failure, or inability to create a stable baseline.

### `PHASE0_INVALID`

The execution cannot be interpreted because provenance or measurement integrity failed.

## 15. Next authorized action

The next implementation work is intentionally narrow:

```text
1. create the environment/model manifest machinery
2. load the frozen Qwen3-4B revision
3. prove thinking-disabled generation and tokenizer determinism
4. implement the compact evaluation harness
5. run the unchanged baseline
6. freeze RDF-v1 numerical promotion thresholds from baseline characterization
7. only then modify the model architecture
```

No recurrent mechanism should be interpreted before `BASELINE_READY`.
