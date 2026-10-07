#!/usr/bin/env python3
"""Execute the ASRI Phase-0 unchanged baseline (ASRI-P0-v1 §10-§13).

Modes:
  pilot     -- dev-role subset per workload (harness debugging; not claim-bearing)
  run       -- full executable roles across all workloads, then protocols + gates
  validate  -- recompute manifests/aggregates/gates from existing evidence
               (loads the frozen model for static metadata only; never
               regenerates quality items)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from asri import baseline, datasets, generation, phase0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "run", "validate"])
    parser.add_argument("--workloads", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap items per workload")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=baseline.EVIDENCE_ROOT,
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def finalize_evidence(
    root: Path,
    workloads: list[str],
    tokenizer,
    model,
    device: torch.device,
    model_resident_bytes: int | None,
    quality: dict | None = None,
    runtime: dict | None = None,
    repro: dict | None = None,
) -> dict:
    """Write/refresh static manifests and gate artifacts; keep quality evidence
    immutable when passed as None (loaded from disk instead)."""
    snapshot = phase0.freeze_model_snapshot()
    dataset_manifests = {
        w: datasets.dataset_manifest(datasets.load_workload(w)) for w in workloads
    }
    runtime_extras = {
        "attention_implementation": getattr(model.config, "_attn_implementation", "unknown"),
    }
    write_json(root / "model_manifest.json", phase0.model_manifest(snapshot, tokenizer=tokenizer, model=model))
    write_json(root / "environment.json", phase0.environment_manifest(runtime=runtime_extras))
    write_json(root / "generation_manifest.json", generation.generation_manifest(tokenizer, model))
    scorers = baseline.scorer_manifest()
    write_json(root / "scorer_manifest.json", scorers)

    if quality is None:
        quality = {w: baseline.aggregate_quality(w) for w in workloads}
    if runtime is None:
        runtime = json.loads((root / "baseline_runtime.json").read_text(encoding="utf-8"))
    if repro is None:
        repro = json.loads((root / "reproducibility.json").read_text(encoding="utf-8"))

    per_item_paths = [baseline.PER_ITEM_DIR / w / "quality.jsonl" for w in workloads]
    write_json(
        root / "baseline_quality.json",
        quality,
    )
    write_json(
        root / "baseline_memory.json",
        baseline.memory_summary(per_item_paths, model_resident_bytes=model_resident_bytes),
    )

    memory = json.loads((root / "baseline_memory.json").read_text(encoding="utf-8"))
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    validator = baseline.baseline_gates(
        {
            "model_manifest": json.loads((root / "model_manifest.json").read_text(encoding="utf-8")),
            "dataset_manifests": dataset_manifests,
            "scorers": scorers,
            "quality": quality,
            "runtime": runtime,
            "reproducibility": repro,
            "memory": memory,
            "per_item_root": baseline.PER_ITEM_DIR,
            "evidence_root": root,
        }
    )
    write_json(root / "validator_result.json", validator)
    rescore_report = root / "per_item" / "humanevalplus" / "rescore_report.json"
    rescore_note = ""
    if rescore_report.exists():
        report = json.loads(rescore_report.read_text(encoding="utf-8"))
        rescore_note = (
            "HumanEval+ verdicts were re-scored post-run in a light process after "
            "the strengthened SCORER_PASS fixture exposed an execution-context "
            f"defect: {report['n_flips']} of {report['n']} verdicts flipped "
            f"({report['old_score']:.4f} -> {report['new_score']:.4f}). Model "
            "responses are unchanged (no regeneration); the original in-run "
            "verdicts are preserved in quality_run_original.jsonl and the audit "
            "in rescore_report.json. Root cause and fix are recorded in the "
            "scorer manifest execution policy."
        )
    (root / "closure.md").write_text(
        baseline.closure_markdown(
            validator, quality, memory, environment, scorers, rescore_note=rescore_note
        ),
        encoding="utf-8",
    )
    # Post-write integrity: reported, never fed back into the verdict.
    closure_text = (root / "closure.md").read_text(encoding="utf-8")
    integrity_ok = bool(closure_text.strip()) and validator["verdict"] in closure_text
    print(f"closure.md post-write integrity: {'ok' if integrity_ok else 'FAILED'}")
    return validator


def main() -> int:
    args = parse_args()
    workloads = args.workloads or list(baseline.WORKLOADS)
    root = args.output_root

    # Frozen substrate (unchanged from the smoke path).
    snapshot = phase0.freeze_model_snapshot()
    tokenizer, model, device = phase0.load_frozen_model(snapshot)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        model_resident_bytes = int(torch.cuda.memory_allocated(device))
    else:
        model_resident_bytes = None
    timed = generation.TimedGenerator(tokenizer, model, device)

    if args.mode == "pilot":
        for workload in workloads:
            started = time.time()
            stats = baseline.run_quality(
                timed, workload, roles=("dev",), limit=args.limit, tag="pilot"
            )
            aggregate = baseline.aggregate_quality(workload, tag="pilot")
            print(
                f"pilot {workload}: new={stats['n_new']} done={stats['n_already_done']} "
                f"score={aggregate['overall'].get('score')} ({time.time() - started:.0f}s)"
            )
        return 0

    if args.mode == "run":
        quality: dict[str, object] = {}
        for workload in workloads:
            stats = baseline.run_quality(timed, workload)
            quality[workload] = baseline.aggregate_quality(workload)
            print(
                f"run {workload}: n={quality[workload]['overall']['n']} "
                f"score={quality[workload]['overall'].get('score')}"
            )

        print("latency protocol...")
        runtime = baseline.run_latency_protocol(timed)
        print("reproducibility protocol...")
        repro = baseline.run_reproducibility_protocol(timed)
        write_json(root / "baseline_runtime.json", runtime)
        write_json(root / "reproducibility.json", repro)

        validator = finalize_evidence(
            root, workloads, tokenizer, model, device, model_resident_bytes,
            quality=quality, runtime=runtime, repro=repro,
        )
        print(json.dumps(validator, indent=2))
        return 0 if validator["all_pass"] else 1

    if args.mode == "validate":
        validator = finalize_evidence(
            root, workloads, tokenizer, model, device, model_resident_bytes
        )
        print(json.dumps(validator, indent=2))
        return 0 if validator["all_pass"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
