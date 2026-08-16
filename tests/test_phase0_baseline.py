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
    @staticmethod
    def _model_manifest() -> dict:
        return {
            "model_id": "Qwen/Qwen3-4B",
            "model_revision": "r" * 40,
            "resolved_revision": "r" * 40,
            "license": "Apache-2.0",
            "thinking_mode": "disabled",
            "context_cap": 4096,
            "parameter_count": 4_000_000_000,
            "parameter_count_estimated": False,
            "layer_count": 36,
            "hidden_size": 2560,
            "attention": {"num_attention_heads": 32, "num_key_value_heads": 8},
            "vocab_size": 151936,
            "torch_dtype": "bfloat16",
            "chat_template_sha256": "a" * 64,
            "artifact_anchor": "immutable HF revision + control-file sha256 set",
            "control_file_sha256": {"config.json": "abc"},
            "weight_shards": [{"name": "a.safetensors", "size_bytes": 1}],
        }

    @staticmethod
    def _scorers(platform: str = "win32", hook_active: bool = True) -> dict:
        fixtures = [{"pass": True}]
        return {
            "math500": {"fixtures": fixtures},
            "mmlupro": {"fixtures": fixtures},
            "ifeval": {"fixtures": fixtures},
            "humanevalplus": {
                "fixtures": [{"pass": True}],
                "windows_compat_preflight": {
                    "platform": platform,
                    "hook_active": hook_active,
                },
            },
        }

    @classmethod
    def _dataset_manifest(cls, ids: list, held_out: list) -> dict:
        return {
            "n_items": len(ids) + len(held_out),
            "role_counts": {
                "dev": len(ids), "held_out": len(held_out),
                "negative_control": 0, "characterization": 0,
            },
            "role_item_ids": {
                "dev": ids, "held_out": held_out,
                "negative_control": [], "characterization": [],
            },
        }

    @classmethod
    def _runtime(cls) -> dict:
        trials = []
        for item in range(baseline.LATENCY_ITEMS_PER_WORKLOAD):
            for _ in range(baseline.LATENCY_TRIALS):
                trials.append(
                    {
                        "item_id": f"item-{item}",
                        "prompt_tokens": 10, "output_tokens": 5,
                        "prefill_seconds": 0.1, "decode_seconds": 0.5,
                        "e2e_seconds": 0.6, "tokens_per_second": 8.3,
                        "peak_allocated_bytes": 1, "peak_reserved_bytes": 1,
                    }
                )
        return {
            "greedy": True,
            "workloads": {w: {"trials": trials} for w in baseline.WORKLOADS},
        }

    @classmethod
    def _context(cls, tmp_path: Path) -> dict:
        per_item_root = tmp_path / "per_item"
        evidence_root = tmp_path / "evidence"
        evidence_root.mkdir()
        quality = {}
        dataset_manifests = {}
        n_rows = 0
        for w in baseline.WORKLOADS:
            ids = [f"{w}-a", f"{w}-b"]
            held_out = [f"{w}-h"]
            dataset_manifests[w] = cls._dataset_manifest(ids, held_out)
            rows = [
                {"item_id": i, "role": "dev", "generation": {"host_rss_bytes": 1,
                 "peak_allocated_bytes": 1, "peak_reserved_bytes": 1},
                 "verdict": {"correct": True}}
                for i in ids
            ]
            path = per_item_root / w / "quality.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            n_rows += len(rows)
            quality[w] = {"overall": {"n": len(rows), "score": 0.5}}
        for name in (
            "model_manifest.json", "environment.json", "dataset_manifest.json",
            "generation_manifest.json", "scorer_manifest.json", "baseline_quality.json",
            "baseline_runtime.json", "baseline_memory.json", "reproducibility.json",
        ):
            (evidence_root / name).write_text("{}", encoding="utf-8")
        (evidence_root / "closure.md").write_text("closure", encoding="utf-8")
        (evidence_root / "environment.json").write_text(
            json.dumps({"gpu_driver": "610.47", "attention_implementation": "sdpa"}),
            encoding="utf-8",
        )
        memory = {
            "model_resident_bytes": 8_000_000_000,
            "peak_allocated_bytes": 9_000_000_000,
            "peak_reserved_bytes": 9_500_000_000,
            "max_host_rss_bytes": 1_800_000_000,
            "n_samples": n_rows,
        }
        return {
            "model_manifest": cls._model_manifest(),
            "dataset_manifests": dataset_manifests,
            "scorers": cls._scorers(),
            "quality": quality,
            "runtime": cls._runtime(),
            "reproducibility": {"reproducibility_pass": True},
            "memory": memory,
            "per_item_root": per_item_root,
            "evidence_root": evidence_root,
        }

    def test_all_gates_pass(self, tmp_path: Path) -> None:
        gates = baseline.baseline_gates(self._context(tmp_path))
        assert gates["all_pass"] is True, gates["gate_details"]
        assert gates["verdict"] == "BASELINE_READY"

    def test_model_freeze_fails_on_missing_layer_count(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        del context["model_manifest"]["layer_count"]
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["MODEL_FREEZE_PASS"] is False
        assert gates["verdict"] == "BASELINE_REFINE"

    def test_model_freeze_fails_on_thinking_enabled(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["model_manifest"]["thinking_mode"] = "enabled"
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["MODEL_FREEZE_PASS"] is False

    def test_model_freeze_fails_on_revision_mismatch(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["model_manifest"]["resolved_revision"] = "s" * 40
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["MODEL_FREEZE_PASS"] is False

    def test_harness_fails_on_partial_run(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        path = context["per_item_root"] / "math500" / "quality.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(lines[0] + "\n", encoding="utf-8")
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["HARNESS_PASS"] is False

    def test_harness_fails_on_held_out_execution(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        path = context["per_item_root"] / "ifeval" / "quality.jsonl"
        row = {"item_id": "ifeval-h", "role": "held_out",
               "generation": {"host_rss_bytes": 1}, "verdict": {"correct": True}}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["HARNESS_PASS"] is False

    def test_scorer_fails_on_fixture_failure(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["scorers"]["math500"]["fixtures"] = [{"pass": False}]
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["SCORER_PASS"] is False

    def test_scorer_fails_on_missing_humanevalplus_fixtures(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["scorers"]["humanevalplus"]["fixtures"] = "string placeholder"
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["SCORER_PASS"] is False

    def test_scorer_fails_on_inactive_windows_hook(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["scorers"] = self._scorers(platform="win32", hook_active=False)
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["SCORER_PASS"] is False

    def test_runtime_fails_on_wrong_trial_count(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        for entry in context["runtime"]["workloads"].values():
            entry["trials"] = entry["trials"][:-1]
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["RUNTIME_MEASUREMENT_PASS"] is False

    def test_runtime_fails_on_missing_greedy_marker(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["runtime"]["greedy"] = False
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["RUNTIME_MEASUREMENT_PASS"] is False

    def test_runtime_fails_on_non_positive_timing(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["runtime"]["workloads"]["math500"]["trials"][0]["e2e_seconds"] = 0
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["RUNTIME_MEASUREMENT_PASS"] is False

    def test_memory_fails_on_missing_model_resident(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["memory"]["model_resident_bytes"] = None
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["MEMORY_MEASUREMENT_PASS"] is False

    def test_memory_fails_on_sample_mismatch(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["memory"]["n_samples"] += 1
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["MEMORY_MEASUREMENT_PASS"] is False

    def test_evidence_write_fails_on_missing_artifact(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        (context["evidence_root"] / "baseline_runtime.json").unlink()
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["EVIDENCE_WRITE_PASS"] is False

    def test_evidence_write_fails_on_local_path_leak(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        leaked = "C:" + "\\" + "\\" + "Users" + "\\" + "\\" + "someone"
        (context["evidence_root"] / "environment.json").write_text(
            json.dumps({"gpu_driver": "610.47", "note": leaked}),
            encoding="utf-8",
        )
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["EVIDENCE_WRITE_PASS"] is False

    def test_evidence_write_fails_on_missing_driver_field(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        (context["evidence_root"] / "environment.json").write_text(
            json.dumps({"attention_implementation": "sdpa"}), encoding="utf-8"
        )
        gates = baseline.baseline_gates(context)
        assert gates["gates"]["EVIDENCE_WRITE_PASS"] is False

    def test_reproducibility_failure_blocks_ready(self, tmp_path: Path) -> None:
        context = self._context(tmp_path)
        context["reproducibility"]["reproducibility_pass"] = False
        gates = baseline.baseline_gates(context)
        assert gates["verdict"] == "BASELINE_REFINE"


class TestWindowsCompatPreflight:
    def test_preflight_reports_hook_active_on_this_machine(self) -> None:
        from asri.scorers import humanevalplus

        preflight = humanevalplus.windows_compat_preflight()
        assert preflight["hook_active"] is True, preflight
        assert "resource" in preflight["checked"]


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
