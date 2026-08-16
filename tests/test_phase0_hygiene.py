"""Hygiene and validation tests for Phase-0 evidence logic (no model required)."""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import pytest

from asri.phase0 import (
    EXPECTED_SMOKE_RESPONSE,
    MODEL_REVISION,
    SmokeResult,
    environment_manifest,
    model_manifest,
    resolved_revision_from_snapshot,
    smoke_validator_checks,
)


def make_result(
    response: str = EXPECTED_SMOKE_RESPONSE,
    output_tokens: int = 2,
    think_tag_present: bool = False,
) -> SmokeResult:
    return SmokeResult(
        prompt="Return exactly the word READY.",
        rendered_prompt="<|im_start|>user\nReturn exactly the word READY.<|im_end|>",
        response=response,
        input_tokens=18,
        output_tokens=output_tokens,
        elapsed_seconds=1.0,
        device="cuda:0",
        dtype="torch.bfloat16",
        thinking_disabled=True,
        think_tag_present=think_tag_present,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
    )


def snapshot_dir(tmp_path: Path, revision: str = MODEL_REVISION) -> Path:
    return tmp_path / "models--Qwen--Qwen3-4B" / "snapshots" / revision


def iter_strings(payload: object):
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for key, value in payload.items():
            yield from iter_strings(key)
            yield from iter_strings(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from iter_strings(item)


class TestResolvedRevision:
    def test_extracts_revision_from_snapshot_layout(self, tmp_path: Path) -> None:
        assert resolved_revision_from_snapshot(snapshot_dir(tmp_path)) == MODEL_REVISION

    def test_rejects_path_without_snapshots_component(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            resolved_revision_from_snapshot(tmp_path / "not-a-snapshot")

    def test_rejects_non_commit_component(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            resolved_revision_from_snapshot(snapshot_dir(tmp_path, revision="main"))


class TestModelManifestPrivacy:
    @staticmethod
    def _fake_snapshot(tmp_path: Path) -> Path:
        snapshot = snapshot_dir(tmp_path)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            '{"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 2560,'
            ' "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 151936,'
            ' "torch_dtype": "bfloat16"}',
            encoding="utf-8",
        )
        (snapshot / "tokenizer_config.json").write_text(
            '{"chat_template": "<|im_start|>user"}', encoding="utf-8"
        )
        (snapshot / "model.safetensors.index.json").write_text(
            '{"metadata": {"total_size": 8044992000}}', encoding="utf-8"
        )
        (snapshot / "model-00001-of-00003.safetensors").write_bytes(b"0" * 16)
        return snapshot

    def test_has_resolved_revision_and_no_local_paths(self, tmp_path: Path) -> None:
        snapshot = self._fake_snapshot(tmp_path)

        manifest = model_manifest(snapshot)

        assert "snapshot_path" not in manifest
        assert manifest["resolved_revision"] == MODEL_REVISION
        rendered = json.dumps(manifest)
        assert str(tmp_path) not in rendered
        assert not re.search(r"[A-Za-z]:\\\\", rendered)

    def test_records_shards_and_control_hashes(self, tmp_path: Path) -> None:
        snapshot = self._fake_snapshot(tmp_path)

        manifest = model_manifest(snapshot)

        assert manifest["weight_shards"] == [
            {"name": "model-00001-of-00003.safetensors", "size_bytes": 16}
        ]
        assert set(manifest["control_file_sha256"]) == {
            "config.json", "tokenizer_config.json", "model.safetensors.index.json"
        }


class TestEnvironmentManifestPrivacy:
    def test_omits_hostname_and_local_paths(self) -> None:
        manifest = environment_manifest()

        assert "hostname" not in manifest
        assert "python_executable" not in manifest
        hostname = socket.gethostname()
        for value in iter_strings(manifest):
            assert hostname not in value
            assert not re.search(r"[A-Za-z]:\\\\", value)

    def test_environment_variables_recorded_as_presence_only(self) -> None:
        manifest = environment_manifest()

        env_block = manifest["environment_variables_set"]
        assert env_block and all(isinstance(v, bool) for v in env_block.values())
        assert set(env_block) == {
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
        }

    def test_preserves_reproducibility_fields(self) -> None:
        manifest = environment_manifest()

        for key in (
            "platform",
            "python",
            "cpu",
            "logical_cpu_count",
            "physical_cpu_count",
            "system_ram_bytes",
            "torch",
            "transformers",
            "huggingface_hub",
            "cuda_runtime",
            "cuda_available",
            "cuda_devices",
        ):
            assert key in manifest


class TestSmokeValidator:
    def test_accepts_known_good_result(self) -> None:
        checks = smoke_validator_checks(make_result(), MODEL_REVISION)

        assert checks["verdict"] == "PASS"
        assert all(value is True for key, value in checks.items() if key != "verdict")

    def test_rejects_wrong_revision(self) -> None:
        wrong = "0" * 40

        checks = smoke_validator_checks(make_result(), wrong)

        assert checks["model_revision_match"] is False
        assert checks["verdict"] == "FAIL"

    def test_rejects_wrong_response(self) -> None:
        checks = smoke_validator_checks(make_result(response="NOT READY"), MODEL_REVISION)

        assert checks["expected_response_match"] is False
        assert checks["verdict"] == "FAIL"

    def test_rejects_empty_generation(self) -> None:
        checks = smoke_validator_checks(
            make_result(response="", output_tokens=0), MODEL_REVISION
        )

        assert checks["generation_nonempty"] is False
        assert checks["verdict"] == "FAIL"

    def test_rejects_think_tag_leakage(self) -> None:
        checks = smoke_validator_checks(
            make_result(response="READY <think>x</think>", think_tag_present=True),
            MODEL_REVISION,
        )

        assert checks["think_tag_absent"] is False
        assert checks["verdict"] == "FAIL"
