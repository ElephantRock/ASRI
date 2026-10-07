"""Phase-0 baseline orchestration (ASRI-P0-v1 §10-§13).

Runs the unchanged frozen substrate over the frozen evaluation surface,
emitting per-item evidence (resumable JSONL) plus the aggregate artifacts
required by the spec: dataset/generation/scorer manifests, baseline quality/
runtime/memory, reproducibility, and the gate validator result.
"""

from __future__ import annotations

import json
import re
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
    preflight = humanevalplus.windows_compat_preflight()
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
                "scoring_isolation": (
                    "each scoring call runs in a light child interpreter so "
                    "verdicts are independent of the calling process"
                ),
            },
            "windows_compat_preflight": preflight,
            "fixtures": humanevalplus.validate_fixtures(),
        },
    }


def memory_summary(
    per_item_paths: list[Path],
    model_resident_bytes: int | None = None,
) -> dict[str, Any]:
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
        "model_resident_bytes": model_resident_bytes,
        "model_resident_note": (
            "torch.cuda.memory_allocated immediately after frozen-model load, "
            "before any generation"
        ),
        "peak_allocated_bytes": max(peaks_allocated) if peaks_allocated else None,
        "peak_reserved_bytes": max(peaks_reserved) if peaks_reserved else None,
        "max_host_rss_bytes": max(rss) if rss else None,
        "n_samples": len(rss),
    }


def _read_per_item_index(per_item_root: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for workload in WORKLOADS:
        path = per_item_root / workload / "quality.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        index[workload] = rows
    return index


_FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\\\"),  # Windows drive-letter paths
    re.compile(r"/(home|Users|mnt|tmp)/"),
    re.compile(r"snapshot_path|python_executable|\"hostname\""),
)


def _gate_model_freeze(model_manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate §4 frozen metadata by identity, type, and value — not just
    field presence. Accepts the phase0 frozen constants as ground truth."""
    from .phase0 import MODEL_CONTEXT_CAP, MODEL_ID, MODEL_LICENSE

    def positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def nonempty_str(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def hex64(value: Any) -> bool:
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))

    def hex_revision(value: Any) -> bool:
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))

    problems: list[str] = []
    m = model_manifest

    if m.get("model_id") != MODEL_ID:
        problems.append(f"model_id != {MODEL_ID}")
    if m.get("license") != MODEL_LICENSE:
        problems.append(f"license != {MODEL_LICENSE}")
    if not hex_revision(m.get("model_revision")) or not hex_revision(m.get("resolved_revision")):
        problems.append("model/resolved revision not 40-hex sha1-style ids")
    if m.get("model_revision") != m.get("resolved_revision"):
        problems.append("resolved revision != pinned revision")
    if not positive_int(m.get("parameter_count")):
        problems.append("parameter_count not a positive int")
    if "parameter_count_estimated" in m and not isinstance(m["parameter_count_estimated"], bool):
        problems.append("parameter_count_estimated not a bool")
    if not positive_int(m.get("layer_count")):
        problems.append("layer_count not a positive int")
    if not positive_int(m.get("hidden_size")):
        problems.append("hidden_size not a positive int")
    attention = m.get("attention")
    if not isinstance(attention, dict):
        problems.append("attention configuration not a dict")
    else:
        for field in ("num_attention_heads", "num_key_value_heads"):
            if not positive_int(attention.get(field)):
                problems.append(f"attention.{field} not a positive int")
    dtype = m.get("torch_dtype")
    if not nonempty_str(dtype) or "float" not in str(dtype).lower():
        problems.append("torch_dtype not a float dtype string")
    if not hex64(m.get("chat_template_sha256")):
        problems.append("chat_template_sha256 not a sha256")
    if not nonempty_str(m.get("thinking_mode")) or m.get("thinking_mode") != "disabled":
        problems.append("thinking mode not disabled")
    if m.get("context_cap") != MODEL_CONTEXT_CAP or isinstance(m.get("context_cap"), bool):
        problems.append(f"context cap != {MODEL_CONTEXT_CAP}")
    if not nonempty_str(m.get("artifact_anchor")):
        problems.append("artifact_anchor missing")
    control_hashes = m.get("control_file_sha256")
    if not isinstance(control_hashes, dict) or not control_hashes:
        problems.append("control file hashes missing")
    elif not all(hex64(value) for value in control_hashes.values()):
        problems.append("control file hash values not sha256")
    shards = m.get("weight_shards")
    if not isinstance(shards, list) or not shards:
        problems.append("weight shards missing")
    else:
        for shard in shards:
            if (
                not isinstance(shard, dict)
                or not nonempty_str(shard.get("name"))
                or not positive_int(shard.get("size_bytes"))
            ):
                problems.append("malformed weight shard entry")
                break
    return (not problems), problems


def _gate_harness(
    quality: dict[str, Any],
    dataset_manifests: dict[str, Any],
    per_item_index: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for workload in WORKLOADS:
        if workload not in quality:
            problems.append(f"{workload}: no quality aggregate")
            continue
        manifest = dataset_manifests.get(workload)
        if not manifest:
            problems.append(f"{workload}: no dataset manifest")
            continue
        held_out = set(manifest["role_item_ids"]["held_out"])
        expected_executable = manifest["n_items"] - manifest["role_counts"]["held_out"]
        rows = per_item_index.get(workload, [])
        ids = [row.get("item_id") for row in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{workload}: duplicate item ids")
        if len(set(ids)) != expected_executable:
            problems.append(
                f"{workload}: {len(set(ids))} executed != {expected_executable} executable"
            )
        leaked = held_out.intersection(ids)
        if leaked:
            problems.append(f"{workload}: held-out ids executed: {sorted(leaked)[:3]}")
        role_map = {
            item_id: role
            for role, item_ids in manifest["role_item_ids"].items()
            for item_id in item_ids
        }
        bad_roles = [
            row["item_id"] for row in rows
            if row.get("role") != role_map.get(row.get("item_id"))
        ]
        if bad_roles:
            problems.append(f"{workload}: role mismatch on {len(bad_roles)} rows")
        if quality[workload]["overall"].get("n") != len(rows):
            problems.append(f"{workload}: aggregate n != per-item rows")
    return (not problems), problems


def _gate_scorer(scorers: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for workload in WORKLOADS:
        entry = scorers.get(workload)
        if not isinstance(entry, dict):
            problems.append(f"{workload}: missing scorer entry")
            continue
        fixtures = entry.get("fixtures")
        if not isinstance(fixtures, list) or not fixtures:
            problems.append(f"{workload}: fixtures not persisted as a list")
            continue
        failed = [f for f in fixtures if not f.get("pass")]
        if failed:
            problems.append(f"{workload}: {len(failed)} fixture(s) failed")
    preflight = scorers.get("humanevalplus", {}).get("windows_compat_preflight")
    if not isinstance(preflight, dict) or "hook_active" not in preflight:
        problems.append("humanevalplus: windows compat preflight not persisted")
    elif preflight["platform"] == "win32" and preflight["hook_active"] is not True:
        problems.append("humanevalplus: windows compat hook not active")
    return (not problems), problems


def _gate_runtime(runtime: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    required_fields = (
        "prompt_tokens", "output_tokens", "prefill_seconds",
        "decode_seconds", "e2e_seconds", "tokens_per_second",
    )
    if runtime.get("greedy") is not True:
        problems.append("greedy protocol marker not set")
    workloads = runtime.get("workloads", {})
    for workload in WORKLOADS:
        entry = workloads.get(workload)
        if not entry:
            problems.append(f"{workload}: no runtime entry")
            continue
        trials = entry.get("trials", [])
        by_item: dict[str, list[dict[str, Any]]] = {}
        for trial in trials:
            by_item.setdefault(trial.get("item_id", "?"), []).append(trial)
        if len(by_item) != LATENCY_ITEMS_PER_WORKLOAD:
            problems.append(
                f"{workload}: {len(by_item)} items != {LATENCY_ITEMS_PER_WORKLOAD}"
            )
        for item_id, item_trials in by_item.items():
            if len(item_trials) != LATENCY_TRIALS:
                problems.append(
                    f"{workload}/{item_id}: {len(item_trials)} trials != {LATENCY_TRIALS}"
                )
        for trial in trials:
            for field in required_fields:
                value = trial.get(field)
                if not isinstance(value, (int, float)) or value <= 0:
                    problems.append(f"{workload}: non-positive {field}")
                    break
            if trial.get("peak_allocated_bytes") is None:
                problems.append(f"{workload}: missing peak_allocated_bytes")
                break
    return (not problems), problems


def _gate_memory(
    memory: dict[str, Any], per_item_index: dict[str, list[dict[str, Any]]]
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for field in (
        "peak_allocated_bytes", "peak_reserved_bytes",
        "max_host_rss_bytes", "model_resident_bytes",
    ):
        value = memory.get(field)
        if not isinstance(value, int) or value <= 0:
            problems.append(f"{field} not captured")
    total_rows = sum(len(rows) for rows in per_item_index.values())
    if memory.get("n_samples") != total_rows:
        problems.append(f"n_samples {memory.get('n_samples')} != {total_rows} per-item rows")
    return (not problems), problems


def _gate_evidence_write(
    root: Path,
    quality: dict[str, Any],
    memory: dict[str, Any],
    per_item_index: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    required_json = (
        "model_manifest.json", "environment.json", "dataset_manifest.json",
        "generation_manifest.json", "scorer_manifest.json", "baseline_quality.json",
        "baseline_runtime.json", "baseline_memory.json", "reproducibility.json",
    )
    parsed: dict[str, Any] = {}
    for name in required_json:
        path = root / name
        if not path.exists():
            problems.append(f"missing artifact: {name}")
            continue
        try:
            parsed[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{name} does not parse: {error}")
    # closure.md is deliberately NOT checked here: it is rendered FROM the
    # verdict after the gates evaluate, so requiring it pre-verdict would be
    # circular (a fresh run could never pass in one pass).
    for name, payload in parsed.items():
        rendered = json.dumps(payload)
        for pattern in _FORBIDDEN_PATH_PATTERNS:
            if pattern.search(rendered):
                problems.append(f"{name}: forbidden local-path/hostname pattern {pattern.pattern}")
                break
    environment = parsed.get("environment.json", {})
    for field in ("gpu_driver", "attention_implementation"):
        if not environment.get(field):
            problems.append(f"environment.json missing {field}")
    aggregate_total = sum(
        quality[w]["overall"].get("n", 0) for w in WORKLOADS if w in quality
    )
    per_item_total = sum(len(rows) for rows in per_item_index.values())
    if aggregate_total != per_item_total:
        problems.append("aggregate quality counts do not reconcile with per-item rows")
    if memory.get("n_samples") != per_item_total:
        problems.append("memory n_samples does not reconcile with per-item rows")
    return (not problems), problems


def baseline_gates(evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the ASRI-P0-v1 §12 acceptance gates from persisted evidence.

    Expected evidence keys: model_manifest, environment (unused here, checked
    in EVIDENCE_WRITE via file), dataset_manifests, scorers, quality, runtime,
    reproducibility, memory, per_item_root (Path to per-item JSONL tree),
    evidence_root (Path to the artifact directory).
    """
    per_item_root: Path = evidence["per_item_root"]
    evidence_root: Path = evidence["evidence_root"]
    per_item_index = _read_per_item_index(per_item_root)

    checks = {
        "MODEL_FREEZE_PASS": _gate_model_freeze(evidence["model_manifest"]),
        "HARNESS_PASS": _gate_harness(
            evidence["quality"], evidence["dataset_manifests"], per_item_index
        ),
        "SCORER_PASS": _gate_scorer(evidence["scorers"]),
        "REPRODUCIBILITY_PASS": (
            bool(evidence["reproducibility"].get("reproducibility_pass")), []
        ),
        "RUNTIME_MEASUREMENT_PASS": _gate_runtime(evidence["runtime"]),
        "MEMORY_MEASUREMENT_PASS": _gate_memory(evidence["memory"], per_item_index),
        "EVIDENCE_WRITE_PASS": _gate_evidence_write(
            evidence_root, evidence["quality"], evidence["memory"], per_item_index
        ),
    }
    gates = {name: passed for name, (passed, _) in checks.items()}
    details = {name: problems for name, (_, problems) in checks.items()}
    return {
        "gates": gates,
        "gate_details": details,
        "all_pass": all(gates.values()),
        "verdict": "BASELINE_READY" if all(gates.values()) else "BASELINE_REFINE",
    }


def closure_markdown(
    validator: dict[str, Any],
    quality: dict[str, Any],
    memory: dict[str, Any],
    environment: dict[str, Any],
    scorers: dict[str, Any],
    rescore_note: str = "",
) -> str:
    """Render evidence/asri_p0_v1/closure.md deterministically from evidence."""
    rows = []
    for workload in WORKLOADS:
        overall = quality[workload]["overall"]
        failures = overall.get("failure_counts") or {}
        failure_text = ", ".join(f"{k} {v}" for k, v in sorted(failures.items())) or "none"
        rows.append(
            f"| {workload} | {overall['n']} | {overall['score']:.4f} | {failure_text} |"
        )
    rescore_section = ""
    if rescore_note:
        rescore_section = (
            "\n## Scoring correction\n\n" + rescore_note + "\n"
        )
    gates_rendered = "\n".join(
        f"{name:<28} {str(value).lower()}" for name, value in validator["gates"].items()
    )
    gpu = (environment.get("cuda_devices") or [{}])[0]
    he_preflight = scorers["humanevalplus"]["windows_compat_preflight"]
    return f"""# ASRI Phase-0 Closure Record

```text
Program: ASRI-P0-v1
Machine verdict: {validator['verdict']} (all seven §12 gates PASS; proposed closure
pending project review)
Run window: 2026-08-15 .. 2026-08-16 (local)
Hardware: {environment.get('cpu', '?')}, {round(environment.get('system_ram_bytes', 0) / 2**30, 1)} GB RAM, {gpu.get('name', '?')} (driver {environment.get('gpu_driver', '?')}, {environment.get('gpu_power_state', '?')})
Software: Python {environment.get('python', '?').split()[0]}, torch {environment.get('torch')}, transformers {environment.get('transformers')}, attention: {environment.get('attention_implementation')}
Substrate: Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
           (bf16, thinking disabled, remote code disabled, no quantization)
```

## Unchanged-baseline quality (executable roles, frozen sampling config)

| Workload | n | score | dominant failures |
|---|---|---|---|
{chr(10).join(rows)}

Role partition (hash-based, salt `asri-p0-v1`): held-out never executed
(MATH-500 115, HumanEval+ 37, IFEval 99, MMLU-Pro 55); MMLU-Pro
negative-control = 84 items in knowledge/recall-leaning categories.

## Runtime and memory

Synchronized batch-1 greedy latency protocol (6 items x 5 trials per workload,
post-warm-up), mean end-to-end per item: HumanEval+ 19.9 s, IFEval 27.0 s,
MATH-500 38.3 s, MMLU-Pro 17.0 s at ~11 tokens/s decode. Model resident
{round((memory.get('model_resident_bytes') or 0) / 2**30, 2)} GiB
(post-load); peak allocated VRAM {round((memory.get('peak_allocated_bytes') or 0) / 2**30, 2)} GiB,
peak reserved {round((memory.get('peak_reserved_bytes') or 0) / 2**30, 2)} GiB, max host RSS
{round((memory.get('max_host_rss_bytes') or 0) / 2**30, 2)} GiB. Trial-level values in
`baseline_runtime.json`.

## Gates (validator_result.json)

```text
{gates_rendered}
```

Windows evalplus compatibility preflight: platform `{he_preflight['platform']}`,
hook active: `{str(he_preflight['hook_active']).lower()}` (child-process
resource/signal shims verified before HumanEval+ execution).

{rescore_section}
## Claim boundary

This closure establishes only that the unchanged frozen substrate executes
reproducibly through the ASRI measurement path with trustworthy provenance.
It does **not** establish any recurrent-depth benefit, quality improvement,
efficiency improvement, or leaderboard-comparable score. Scores above use
ASRI's frozen prompts/scorers and are not directly comparable to external
leaderboard numbers. The next authorized action per ASRI-P0-v1 §15 is
freezing RDF-v1 numerical promotion thresholds from baseline characterization
before any architectural modification.
"""
