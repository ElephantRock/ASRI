# ASRI Phase-0 Closure Record

```text
Program: ASRI-P0-v1
Machine verdict: BASELINE_READY (all seven §12 gates PASS; proposed closure
pending project review)
Run window: 2026-08-15 .. 2026-08-16 (local)
Hardware: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 31.9 GB RAM, NVIDIA GeForce RTX 3080 Ti (driver 610.47, P8)
Software: Python 3.12.10, torch 2.13.0+cu130, transformers 5.15.0, attention: sdpa
Substrate: Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
           (bf16, thinking disabled, remote code disabled, no quantization)
```

## Unchanged-baseline quality (executable roles, frozen sampling config)

| Workload | n | score | dominant failures |
|---|---|---|---|
| math500 | 385 | 0.6156 | mismatch 17, no_boxed 98, undecidable 33 |
| humanevalplus | 127 | 0.7323 | base_fail 26, plus_fail 8 |
| ifeval | 442 | 0.7760 | instruction_violated 99 |
| mmlupro | 185 | 0.5730 | mismatch 72, no_choice 7 |

Role partition (hash-based, salt `asri-p0-v1`): held-out never executed
(MATH-500 115, HumanEval+ 37, IFEval 99, MMLU-Pro 55); MMLU-Pro
negative-control = 84 items in knowledge/recall-leaning categories.

## Runtime and memory

Synchronized batch-1 greedy latency protocol (6 items x 5 trials per workload,
post-warm-up), mean end-to-end per item: HumanEval+ 19.9 s, IFEval 27.0 s,
MATH-500 38.3 s, MMLU-Pro 17.0 s at ~11 tokens/s decode. Model resident
7.55 GiB
(post-load); peak allocated VRAM 8.03 GiB,
peak reserved 8.95 GiB, max host RSS
1.76 GiB. Trial-level values in
`baseline_runtime.json`.

## Gates (validator_result.json)

```text
MODEL_FREEZE_PASS            true
HARNESS_PASS                 true
SCORER_PASS                  true
REPRODUCIBILITY_PASS         true
RUNTIME_MEASUREMENT_PASS     true
MEMORY_MEASUREMENT_PASS      true
EVIDENCE_WRITE_PASS          true
```

Windows evalplus compatibility preflight: platform `win32`,
hook active: `true` (child-process
resource/signal shims verified before HumanEval+ execution).


## Scoring correction

HumanEval+ verdicts were re-scored post-run in a light process after the strengthened SCORER_PASS fixture exposed an execution-context defect: 26 of 127 verdicts flipped (0.5276 -> 0.7323). Model responses are unchanged (no regeneration); the original in-run verdicts are preserved in quality_run_original.jsonl and the audit in rescore_report.json. Root cause and fix are recorded in the scorer manifest execution policy.

## Claim boundary

This closure establishes only that the unchanged frozen substrate executes
reproducibly through the ASRI measurement path with trustworthy provenance.
It does **not** establish any recurrent-depth benefit, quality improvement,
efficiency improvement, or leaderboard-comparable score. Scores above use
ASRI's frozen prompts/scorers and are not directly comparable to external
leaderboard numbers. The next authorized action per ASRI-P0-v1 §15 is
freezing RDF-v1 numerical promotion thresholds from baseline characterization
before any architectural modification.
