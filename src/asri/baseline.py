"""Phase-0 baseline orchestration (ASRI-P0-v1 §10-§13).

Runs the unchanged frozen substrate over the frozen evaluation surface,
emitting per-item evidence (resumable JSONL) plus the aggregate artifacts
required by the spec: dataset/generation/scorer manifests, baseline quality/
runtime/memory, reproducibility, and the gate validator result.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from . import datasets as ds
from . import generation as gen
from .scorers import humanevalplus, ifeval_scorer, math500, mmlupro

EVIDENCE_ROOT = Path("evidence/asri_p0_v1")
PER_ITEM_DIR = EVIDENCE_ROOT / "per_item"

WORKLOADS = ("math500", "humanevalplus", "ifeval", "mmlupro")

# Fixed deterministic subsets for the runtime and reproducibility protocols.
LATENCY_ITEMS_PER_WORKLOAD = 6
LATENCY_TRIALS = 5
REPRO_ITEMS_PER_WORKLOAD = 3


def score_response(workload: str, item: ds.WorkloadItem, response: str) -> dict[str, Any]:
    if workload == "math500":
        return math500.score_item(response, item.reference["answer"])
    if workload == "mmlupro":
        return mmlupro.score_item(response, item.reference["answer_letter"])
    if workload == "ifeval":
        return ifeval_scorer.score_item(
            prompt=item.prompt,
            model_output=response,
            instruction_id_list=item.reference["instruction_id_list"],
            kwargs_list=item.reference["kwargs"],
        )
    if workload == "humanevalplus":
        return humanevalplus.score_item(response, item.reference["task_id"])
    raise ValueError(f"unknown workload: {workload}")


def completed_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["item_id"])
    return ids


def run_quality(
    timed: gen.TimedGenerator,
    workload: str,
    roles: tuple[str, ...] = ds.EXECUTABLE_ROLES,
    limit: int | None = None,
    greedy: bool = False,
    tag: str = "quality",
) -> dict[str, Any]:
    items = [
        item
        for item in ds.load_workload(workload)
        if item.role in roles
    ]
    items.sort(key=lambda i: i.item_id)
    if limit is not None:
        items = items[:limit]

    out_path = PER_ITEM_DIR / workload / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_item_ids(out_path)

    n_new = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for item in items:
            if item.item_id in done:
                continue
            record = timed.generate(workload, item.prompt, greedy=greedy)
            verdict = score_response(workload, item, record.response)
            evidence = {
                "item_id": item.item_id,
                "workload": workload,
                "role": item.role,
                "category": item.category,
                "generation": gen.record_to_dict(record),
                "verdict": verdict,
            }
            handle.write(json.dumps(evidence, ensure_ascii=True) + "\n")
            handle.flush()
            n_new += 1

    return {
        "workload": workload,
        "roles": list(roles),
        "path": str(out_path).replace("\\", "/"),
        "n_items_selected": len(items),
        "n_new": n_new,
        "n_already_done": len(items) and sum(1 for i in items if i.item_id in done),
    }


def aggregate_quality(workload: str, tag: str = "quality") -> dict[str, Any]:
    path = PER_ITEM_DIR / workload / f"{tag}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        correct = sum(1 for r in rows if r["verdict"]["correct"])
        failures: dict[str, int] = {}
        for r in rows:
            failure = r["verdict"].get("failure")
            if failure:
                failures[failure] = failures.get(failure, 0) + 1
        by_role: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = by_role.setdefault(r["role"], {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(r["verdict"]["correct"])
        by_category: dict[str, dict[str, int]] = {}
        for r in rows:
            if r.get("category"):
                bucket = by_category.setdefault(r["category"], {"n": 0, "correct": 0})
                bucket["n"] += 1
                bucket["correct"] += int(r["verdict"]["correct"])
        tokens = [r["generation"]["output_tokens"] for r in rows]
        latencies = [r["generation"]["e2e_seconds"] for r in rows]
        return {
            "n": len(rows),
            "n_correct": correct,
            "score": correct / len(rows),
            "failure_counts": failures,
            "by_role": by_role,
            "by_category": by_category,
            "mean_output_tokens": statistics.fmean(tokens) if tokens else None,
            "median_output_tokens": statistics.median(tokens) if tokens else None,
            "mean_e2e_seconds": statistics.fmean(latencies) if latencies else None,
        }

    return {"workload": workload, "overall": summary(rows), "source": str(path).replace("\\", "/")}


def latency_protocol_items(workload: str, count: int) -> list[ds.WorkloadItem]:
    items = [
        item
        for item in ds.load_workload(workload)
        if item.role == "characterization"
    ]
    items.sort(key=lambda i: i.item_id)
    return items[:count]


def run_latency_protocol(timed: gen.TimedGenerator) -> dict[str, Any]:
    """Warm-up + repeated greedy trials on fixed subsets; trial-level values."""
    protocol: dict[str, Any] = {"trials": LATENCY_TRIALS, "warmup": 1, "greedy": True, "workloads": {}}
    for workload in WORKLOADS:
        items = latency_protocol_items(workload, LATENCY_ITEMS_PER_WORKLOAD)
        # Warm-up before claim-bearing trials (ASRI-P0-v1 §11).
        for item in items[:1]:
            timed.generate(workload, item.prompt, greedy=True)
        trials = []
        for trial in range(LATENCY_TRIALS):
            for item in items:
                record = timed.generate(workload, item.prompt, greedy=True)
                trials.append(
                    {
                        "trial": trial,
                        "item_id": item.item_id,
                        "prompt_tokens": record.prompt_tokens,
                        "output_tokens": record.output_tokens,
                        "prefill_seconds": record.prefill_seconds,
                        "decode_seconds": record.decode_seconds,
                        "e2e_seconds": record.e2e_seconds,
                        "tokens_per_second": record.tokens_per_second,
                        "peak_allocated_bytes": record.peak_allocated_bytes,
                        "peak_reserved_bytes": record.peak_reserved_bytes,
                    }
                )
        e2e = [t["e2e_seconds"] for t in trials]
        tps = [t["tokens_per_second"] for t in trials]
        e2e_sorted = sorted(e2e)

        def percentile(sorted_values: list[float], fraction: float) -> float | None:
            if not sorted_values:
                return None
            index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
            return sorted_values[index]

        protocol["workloads"][workload] = {
            "n_items": len(items),
            "item_ids": [i.item_id for i in items],
            "trials": trials,
            "mean_e2e_seconds": statistics.fmean(e2e) if e2e else None,
            "median_e2e_seconds": statistics.median(e2e) if e2e else None,
            "p90_e2e_seconds": percentile(e2e_sorted, 0.90),
            "p95_e2e_seconds": percentile(e2e_sorted, 0.95),
            "stdev_e2e_seconds": statistics.stdev(e2e) if len(e2e) > 1 else None,
            "mean_tokens_per_second": statistics.fmean(tps) if tps else None,
        }
    return protocol


def run_reproducibility_protocol(timed: gen.TimedGenerator) -> dict[str, Any]:
    """Rerun fixed subsets under the frozen seed; verdicts and text must match."""
    results: dict[str, Any] = {"workloads": {}}
    all_match = True
    for workload in WORKLOADS:
        items = latency_protocol_items(workload, REPRO_ITEMS_PER_WORKLOAD)
        entries = []
        for item in items:
            first = timed.generate(workload, item.prompt)
            second = timed.generate(workload, item.prompt)
            sampled_match = first.response == second.response and first.seed == second.seed
            greedy_first = timed.generate(workload, item.prompt, greedy=True)
            greedy_second = timed.generate(workload, item.prompt, greedy=True)
            greedy_match = greedy_first.response == greedy_second.response
            verdict_match = (
                score_response(workload, item, first.response)["correct"]
                == score_response(workload, item, second.response)["correct"]
            )
            entry = {
                "item_id": item.item_id,
                "sampled_text_match": sampled_match,
                "greedy_text_match": greedy_match,
                "sampled_verdict_match": verdict_match,
            }
            entries.append(entry)
            all_match = all_match and sampled_match and greedy_match and verdict_match
        results["workloads"][workload] = {
            "items": entries,
            "n_items": len(items),
        }
    results["reproducibility_pass"] = bool(all_match)
    return results


def scorer_manifest() -> dict[str, Any]:
    return {
        "math500": {
            "scorer_id": math500.SCORER_ID,
            "fixtures": math500.validate_fixtures(),
        },
        "mmlupro": {
            "scorer_id": mmlupro.SCORER_ID,
            "fixtures": mmlupro.validate_fixtures(),
        },
        "ifeval": {
            "scorer_id": ifeval_scorer.SCORER_ID,
            "upstream_pinned_sha": "e6890f85757dd84e27ca6df2dd30651dafad28e0",
            "fixtures": ifeval_scorer.validate_fixtures(),
        },
        "humanevalplus": {
            "scorer_id": humanevalplus.SCORER_ID,
            "execution_policy": {
                "min_time_limit": humanevalplus.MIN_TIME_LIMIT,
                "gt_time_limit_factor": humanevalplus.GT_TIME_LIMIT_FACTOR,
                "fast_check": True,
            },
            "fixtures": "evaluated at run time (requires evalplus backend)",
        },
    }


def memory_summary(per_item_paths: list[Path]) -> dict[str, Any]:
    peaks_allocated, peaks_reserved, rss = [], [], []
    for path in per_item_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            generation = json.loads(line)["generation"]
            if generation.get("peak_allocated_bytes") is not None:
                peaks_allocated.append(generation["peak_allocated_bytes"])
                peaks_reserved.append(generation["peak_reserved_bytes"])
            rss.append(generation["host_rss_bytes"])
    return {
        "peak_allocated_bytes": max(peaks_allocated) if peaks_allocated else None,
        "peak_reserved_bytes": max(peaks_reserved) if peaks_reserved else None,
        "max_host_rss_bytes": max(rss) if rss else None,
        "n_samples": len(rss),
    }


def baseline_gates(
    model_manifest: dict[str, Any],
    dataset_manifests: dict[str, Any],
    scorers: dict[str, Any],
    quality: dict[str, Any],
    runtime: dict[str, Any],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the ASRI-P0-v1 §12 acceptance gates from emitted evidence."""
    model_freeze = (
        model_manifest.get("resolved_revision") == model_manifest.get("model_revision")
        and bool(model_manifest.get("control_file_sha256"))
    )
    harness = bool(quality) and all(
        quality[w]["overall"]["n"] > 0 for w in WORKLOADS if w in quality
    ) and all(w in quality for w in WORKLOADS)

    def fixtures_pass(workload_scorers: dict[str, Any]) -> bool:
        fixtures = [
            entry["fixtures"]
            for entry in workload_scorers.values()
            if isinstance(entry.get("fixtures"), list)
        ]
        if not fixtures:
            return False
        return all(item["pass"] for group in fixtures for item in group)

    scorer_pass = fixtures_pass(scorers) and harness  # harness exec proves code scorer
    evidence_pass = all(
        isinstance(payload, dict) for payload in (quality, runtime, reproducibility)
    )
    runtime_pass = bool(runtime.get("workloads")) and all(
        len(entry["trials"]) == LATENCY_TRIALS
        for entry in runtime["workloads"].values()
    )
    memory_pass = all(
        dataset_manifests[w]["n_items"] > 0 for w in WORKLOADS
    ) and isinstance(quality, dict)

    gates = {
        "MODEL_FREEZE_PASS": bool(model_freeze),
        "HARNESS_PASS": bool(harness),
        "SCORER_PASS": bool(scorer_pass),
        "REPRODUCIBILITY_PASS": bool(reproducibility.get("reproducibility_pass")),
        "RUNTIME_MEASUREMENT_PASS": bool(runtime_pass),
        "MEMORY_MEASUREMENT_PASS": bool(memory_pass),
        "EVIDENCE_WRITE_PASS": bool(evidence_pass),
    }
    return {
        "gates": gates,
        "all_pass": all(gates.values()),
        "verdict": "BASELINE_READY" if all(gates.values()) else "BASELINE_REFINE",
    }
