# ASRI Phase-0 Closure Record

```text
Program: ASRI-P0-v1
Machine verdict: BASELINE_READY (all seven §12 gates PASS; proposed closure
pending project review)
Run window: 2026-08-15 .. 2026-08-16 (local)
Hardware: AMD Ryzen 5 5600G, 31.9 GB RAM, RTX 3080 Ti 12 GiB, driver 610.47
Software: Python 3.12.10, torch 2.13.0+cu130, transformers 5.15.0
Substrate: Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
           (bf16, thinking disabled, remote code disabled, no quantization)
```

## Unchanged-baseline quality (executable roles, frozen sampling config)

| Workload | n | score | dominant failures |
|---|---|---|---|
| MATH-500 (ASRI frozen prompt/scorer) | 385 | 0.6156 | no_boxed 98, undecidable 33, mismatch 17 |
| HumanEval+ pass@1 (evalplus backend) | 127 | 0.5276 | base_fail 53, plus_fail 7 |
| IFEval strict prompt-level (pinned official evaluator) | 442 | 0.7760 | instruction_violated 99 |
| MMLU-Pro bounded slice (240 frozen) | 185 | 0.5730 | mismatch 72, no_choice 7 |

Notes: 33 MATH items scored "undecidable" by the guarded sympy equivalence
check and are counted as incorrect (conservative). Role partition (hash-based,
salt `asri-p0-v1`): held-out never executed (MATH-500 115, HumanEval+ 37,
IFEval 99, MMLU-Pro 55); MMLU-Pro negative-control = 84 items in knowledge/
recall-leaning categories.

## Runtime and memory

Synchronized batch-1 greedy latency protocol (6 items x 5 trials per workload,
post-warm-up), mean end-to-end per item: HumanEval+ 19.9 s, IFEval 27.0 s,
MATH-500 38.3 s, MMLU-Pro 17.0 s at ~11 tokens/s decode. Peak allocated VRAM
8.63 GiB, peak reserved 9.61 GiB, max host RSS 1.89 GiB. Trial-level values
in `baseline_runtime.json`.

## Gates (validator_result.json)

```text
MODEL_FREEZE_PASS            true  (resolved revision == pinned revision)
HARNESS_PASS                 true  (all four workloads executed end to end)
SCORER_PASS                  true  (all fixture suites pass incl. canonical/
                                  broken HumanEval discrimination)
REPRODUCIBILITY_PASS         true  (sampled text, greedy text, and verdicts
                                  reproduce exactly under frozen seeds)
RUNTIME_MEASUREMENT_PASS     true  (30 trial-level values per workload)
MEMORY_MEASUREMENT_PASS      true  (peak fields captured on frozen hardware)
EVIDENCE_WRITE_PASS          true  (schema-valid, git-trackable, path-free)
```

## Claim boundary

This closure establishes only that the unchanged frozen substrate executes
reproducibly through the ASRI measurement path with trustworthy provenance.
It does **not** establish any recurrent-depth benefit, quality improvement,
efficiency improvement, or leaderboard-comparable score. Scores above use
ASRI's frozen prompts/scorers and are not directly comparable to external
leaderboard numbers. The next authorized action per ASRI-P0-v1 §15 is
freezing RDF-v1 numerical promotion thresholds from baseline characterization
before any architectural modification.
