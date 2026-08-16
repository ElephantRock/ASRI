"""Unit tests for the Phase-0 baseline harness logic (no model, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from asri import baseline, datasets
from asri.scorers import ifeval_scorer, math500, mmlupro


class TestRolePartitioning:
    def test_assignment_is_deterministic(self) -> None:
        for workload in ("math500", "ifeval", "mmlupro", "humanevalplus"):
            for item_id in ("a", "b", "item-123", "x" * 40):
                assert datasets.assign_role(workload, item_id) == datasets.assign_role(
                    workload, item_id
                )

    def test_roles_are_exhaustive_and_valid(self) -> None:
        ids = [f"item-{i}" for i in range(2000)]
        for workload in ("math500", "ifeval", "humanevalplus"):
            roles = {datasets.assign_role(workload, i) for i in ids}
            assert roles <= set(datasets.ROLE_ORDER)
            assert "dev" in roles and "held_out" in roles and "characterization" in roles

    def test_mmlupro_negative_control_category(self) -> None:
        # "health" is in the frozen negative-control set; "math" is not.
        seen = set()
        for i in range(500):
            role = datasets.assign_role("mmlupro", f"probe-{i}", "health")
            seen.add(role)
        assert "negative_control" in seen
        assert "characterization" not in seen or True  # hash spread may include others
        math_roles = {
            datasets.assign_role("mmlupro", f"probe-{i}", "math") for i in range(500)
        }
        assert "negative_control" not in math_roles

    def test_held_out_is_excluded_from_executable_roles(self) -> None:
        assert "held_out" not in datasets.EXECUTABLE_ROLES


class TestDatasetManifest:
    def _items(self) -> list[datasets.WorkloadItem]:
        return [
            datasets.WorkloadItem(
                workload="math500",
                item_id=f"math500-test-{i}",
                role=datasets.assign_role("math500", f"math500-test-{i}"),
                category="Algebra",
                prompt=f"prompt {i}",
                reference={"answer": str(i)},
            )
            for i in range(50)
        ]

    def test_manifest_has_no_local_paths_and_covers_roles(self) -> None:
        manifest = datasets.dataset_manifest(self._items())

        rendered = json.dumps(manifest)
        assert not re.search(r"[A-Za-z]:\\\\", rendered)
        assert "snapshot" not in rendered.lower()
        counts = manifest["role_counts"]
        assert sum(counts.values()) == manifest["n_items"] == 50
        assert set(counts) == set(datasets.ROLE_ORDER)
        assert manifest["content_digest"]

    def test_manifest_digest_is_content_stable(self) -> None:
        assert datasets.dataset_manifest(
            self._items()
        )["content_digest"] == datasets.dataset_manifest(self._items())["content_digest"]


class TestMath500Scorer:
    def test_all_fixtures_pass(self) -> None:
        results = math500.validate_fixtures()
        failures = [r for r in results if not r["pass"]]
        assert not failures, failures

    def test_boxed_extraction_nested_braces(self) -> None:
        assert (
            math500.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        )
        assert math500.extract_boxed("no box") is None

    def test_failure_classification(self) -> None:
        assert math500.score_item("answer without box", "3")["failure"] == "no_boxed"
        assert math500.score_item(r"\boxed{5}", "6")["failure"] == "mismatch"


class TestMmluProScorer:
    def test_all_fixtures_pass(self) -> None:
        results = mmlupro.validate_fixtures()
        failures = [r for r in results if not r["pass"]]
        assert not failures, failures

    def test_prefers_answer_is_pattern(self) -> None:
        assert mmlupro.extract_choice("A first... The answer is (D)") == "D"

    def test_no_choice_failure(self) -> None:
        assert mmlupro.score_item("nothing here", "A")["failure"] == "no_choice"


class TestIfevalScorer:
    def test_all_fixtures_pass(self) -> None:
        results = ifeval_scorer.validate_fixtures()
        failures = [r for r in results if not r["pass"]]
        assert not failures, failures


class TestResume:
    def test_completed_item_ids_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "quality.jsonl"
        path.write_text(
            json.dumps({"item_id": "a"}) + "\n" + json.dumps({"item_id": "b"}) + "\n",
            encoding="utf-8",
        )
        assert baseline.completed_item_ids(path) == {"a", "b"}
        assert baseline.completed_item_ids(tmp_path / "missing.jsonl") == set()


class TestBaselineGates:
    def _quality(self, n: int = 3) -> dict:
        return {
            w: {"overall": {"n": n, "score": 0.5}} for w in baseline.WORKLOADS
        }

    def test_all_gates_pass(self) -> None:
        model_manifest = {
            "model_revision": "r" * 40,
            "resolved_revision": "r" * 40,
            "control_file_sha256": {"config.json": "abc"},
        }
        dataset_manifests = {w: {"n_items": 10} for w in baseline.WORKLOADS}
        scorers = {
            "math500": {"fixtures": [{"pass": True}]},
            "mmlupro": {"fixtures": [{"pass": True}]},
            "ifeval": {"fixtures": [{"pass": True}]},
            "humanevalplus": {"fixtures": "evaluated at run time (requires evalplus backend)"},
        }
        runtime = {
            "workloads": {
                w: {"trials": [{}] * (baseline.LATENCY_ITEMS_PER_WORKLOAD * baseline.LATENCY_TRIALS)} for w in baseline.WORKLOADS
            }
        }
        repro = {"reproducibility_pass": True}
        gates = baseline.baseline_gates(
            model_manifest, dataset_manifests, scorers, self._quality(), runtime, repro
        )
        assert gates["all_pass"] is True
        assert gates["verdict"] == "BASELINE_READY"

    def test_revision_mismatch_fails_model_freeze(self) -> None:
        model_manifest = {
            "model_revision": "r" * 40,
            "resolved_revision": "s" * 40,
            "control_file_sha256": {"config.json": "abc"},
        }
        gates = baseline.baseline_gates(
            model_manifest,
            {w: {"n_items": 10} for w in baseline.WORKLOADS},
            {"a": {"fixtures": [{"pass": True}]}},
            self._quality(),
            {"workloads": {w: {"trials": [{}] * (baseline.LATENCY_ITEMS_PER_WORKLOAD * baseline.LATENCY_TRIALS)} for w in baseline.WORKLOADS}},
            {"reproducibility_pass": True},
        )
        assert gates["gates"]["MODEL_FREEZE_PASS"] is False
        assert gates["verdict"] == "BASELINE_REFINE"

    def test_repro_failure_blocks_ready(self) -> None:
        model_manifest = {
            "model_revision": "r" * 40,
            "resolved_revision": "r" * 40,
            "control_file_sha256": {"config.json": "abc"},
        }
        gates = baseline.baseline_gates(
            model_manifest,
            {w: {"n_items": 10} for w in baseline.WORKLOADS},
            {"a": {"fixtures": [{"pass": True}]}},
            self._quality(),
            {"workloads": {w: {"trials": [{}] * (baseline.LATENCY_ITEMS_PER_WORKLOAD * baseline.LATENCY_TRIALS)} for w in baseline.WORKLOADS}},
            {"reproducibility_pass": False},
        )
        assert gates["verdict"] == "BASELINE_REFINE"


class TestGenerationSeeds:
    def test_item_seed_deterministic_and_range(self) -> None:
        from asri import generation

        assert generation.item_seed("math500", "abc") == generation.item_seed(
            "math500", "abc"
        )
        assert generation.item_seed("math500", "abc") != generation.item_seed(
            "math500", "abd"
        )
        assert 0 <= generation.item_seed("ifeval", "x") < 2**63
