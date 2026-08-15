#!/usr/bin/env python3
"""Execute the first ASRI Phase-0 real-model smoke run.

This command is diagnostic infrastructure, not a benchmark. A PASS only proves
that the frozen substrate can be loaded and exercised through the intended
thinking-disabled path while emitting provenance/runtime evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asri.phase0 import (
    freeze_model_snapshot,
    load_frozen_model,
    resolved_revision_from_snapshot,
    run_smoke,
    smoke_validator_checks,
    write_phase0_smoke_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evidence/asri_p0_v1/smoke"),
        help="Directory for git-trackable smoke evidence.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Return exactly the word READY.",
        help="Diagnostic prompt; not a quality benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = freeze_model_snapshot(cache_dir=args.cache_dir)
    tokenizer, model, device = load_frozen_model(snapshot)
    result = run_smoke(tokenizer, model, device, prompt=args.prompt)
    write_phase0_smoke_evidence(args.output_dir, snapshot, result)

    checks = smoke_validator_checks(result, resolved_revision_from_snapshot(snapshot))
    verdict = checks["verdict"]
    print(f"ASRI-P0 smoke: {verdict}")
    print(f"device={result.device} dtype={result.dtype}")
    print(f"latency_s={result.elapsed_seconds:.6f}")
    print(f"input_tokens={result.input_tokens} output_tokens={result.output_tokens}")
    print(f"response={result.response!r}")
    print(f"evidence={args.output_dir}")
    if verdict != "PASS":
        print(f"failed_checks={sorted(k for k, v in checks.items() if v is False)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
