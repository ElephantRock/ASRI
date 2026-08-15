#!/usr/bin/env python3
"""Execute the ASRI Phase-0 unchanged baseline (ASRI-P0-v1 §10-§13).

Modes:
  pilot     -- dev-role subset per workload (harness debugging; not claim-bearing)
  run       -- full executable roles across all workloads, then protocols + gates
  validate  -- recompute manifests/aggregates/gates from existing evidence
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


def main() -> int:
    args = parse_args()
    workloads = args.workloads or list(baseline.WORKLOADS)
    root = args.output_root

    # Frozen substrate (unchanged from the smoke path).
    snapshot = phase0.freeze_model_snapshot()
    tokenizer, model, device = phase0.load_frozen_model(snapshot)
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

        per_item_paths = [
            baseline.PER_ITEM_DIR / w / "quality.jsonl" for w in workloads
        ]
        write_json(root / "dataset_manifest.json", {
            w: datasets.dataset_manifest(datasets.load_workload(w))
            for w in workloads
        })
        write_json(root / "generation_manifest.json", generation.generation_manifest(tokenizer, model))
        write_json(root / "environment.json", phase0.environment_manifest())
        write_json(root / "model_manifest.json", phase0.model_manifest(snapshot))
        scorers = baseline.scorer_manifest()
        write_json(root / "scorer_manifest.json", scorers)
        write_json(root / "baseline_quality.json", quality)
        write_json(root / "baseline_runtime.json", runtime)
        write_json(root / "baseline_memory.json", baseline.memory_summary(per_item_paths))
        write_json(root / "reproducibility.json", repro)

        model_manifest = phase0.model_manifest(snapshot)
        dataset_manifests = {
            w: datasets.dataset_manifest(datasets.load_workload(w)) for w in workloads
        }
        gates = baseline.baseline_gates(
            model_manifest, dataset_manifests, scorers, quality, runtime, repro
        )
        write_json(root / "validator_result.json", gates)
        print(json.dumps(gates, indent=2))
        return 0 if gates["all_pass"] else 1

    if args.mode == "validate":
        quality = {
            w: baseline.aggregate_quality(w) for w in workloads
        }
        runtime = json.loads((root / "baseline_runtime.json").read_text(encoding="utf-8"))
        repro = json.loads((root / "reproducibility.json").read_text(encoding="utf-8"))
        scorers = json.loads((root / "scorer_manifest.json").read_text(encoding="utf-8"))
        model_manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
        dataset_manifests = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
        gates = baseline.baseline_gates(
            model_manifest, dataset_manifests, scorers, quality, runtime, repro
        )
        write_json(root / "validator_result.json", gates)
        print(json.dumps(gates, indent=2))
        return 0 if gates["all_pass"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
