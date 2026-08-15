"""Phase-0 primitives for freezing and smoke-testing the ASRI substrate.

This module intentionally does not implement recurrent computation. Its job is to
establish a trustworthy unchanged-model reference before ASRI-RDF-v1 begins.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_SMOKE_RESPONSE = "READY"

# Environment variables whose presence (never value) is recorded in evidence,
# so machine-local cache locations stay out of public artifacts.
_TRACKED_ENV_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_device_manifest() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    return devices


def environment_manifest() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_bytes": vm.total,
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "huggingface_hub": _version("huggingface-hub"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": _cuda_device_manifest(),
        "environment_variables_set": {
            key: os.environ.get(key) is not None
            for key in _TRACKED_ENV_VARIABLES
        },
    }


def freeze_model_snapshot(cache_dir: str | None = None) -> Path:
    """Resolve the immutable model revision into the local HF cache."""
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
    )
    return Path(snapshot)


def resolved_revision_from_snapshot(snapshot: Path) -> str:
    """Extract the resolved revision recorded in the HF snapshot layout.

    snapshot_download() returns ``.../snapshots/<resolved-revision>``; the
    component after the ``snapshots`` directory is the immutable commit the
    download actually resolved to, independent of the requested pin.
    """
    parts = snapshot.resolve().parts
    for index, part in enumerate(parts):
        if part == "snapshots" and index + 1 < len(parts):
            revision = parts[index + 1]
            if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower()):
                return revision
    raise ValueError(f"cannot extract resolved revision from snapshot path: {snapshot}")


def model_manifest(snapshot: Path) -> dict[str, Any]:
    """Record model identity and lightweight local artifact integrity.

    Full weight hashing is deliberately deferred because the immutable HF revision
    already anchors the payload and hashing ~8 GB on every smoke run adds needless
    apparatus cost. The manifest records weight shard names/sizes and hashes the
    small control files that determine model/tokenizer behavior.
    """
    control_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]

    hashes: dict[str, str] = {}
    for name in control_files:
        path = snapshot / name
        if path.exists():
            hashes[name] = sha256_file(path)

    shards = []
    for path in sorted(snapshot.glob("*.safetensors")):
        shards.append({"name": path.name, "size_bytes": path.stat().st_size})

    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_revision": resolved_revision_from_snapshot(snapshot),
        "control_file_sha256": hashes,
        "weight_shards": shards,
    }


def load_frozen_model(snapshot: Path) -> tuple[Any, Any, torch.device]:
    """Load the frozen substrate without remote code or quantization."""
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


@dataclass(frozen=True)
class SmokeResult:
    prompt: str
    rendered_prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    device: str
    dtype: str
    thinking_disabled: bool
    think_tag_present: bool
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_smoke(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    prompt: str = "Return exactly the word READY.",
    max_new_tokens: int = 16,
) -> SmokeResult:
    """Run one diagnostic generation with Qwen textual thinking hard-disabled."""
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    _sync(device)
    elapsed = time.perf_counter() - start

    input_tokens = encoded["input_ids"].shape[-1]
    output_ids = generated[0, input_tokens:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    parameter = next(model.parameters())
    return SmokeResult(
        prompt=prompt,
        rendered_prompt=rendered,
        response=response,
        input_tokens=int(input_tokens),
        output_tokens=int(output_ids.numel()),
        elapsed_seconds=elapsed,
        device=str(device),
        dtype=str(parameter.dtype),
        thinking_disabled=True,
        think_tag_present=("<think>" in response or "</think>" in response),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def smoke_validator_checks(result: SmokeResult, resolved_revision: str) -> dict[str, Any]:
    """Independently re-check the frozen-substrate contract on a smoke result."""
    checks = {
        "model_id_match": MODEL_ID == "Qwen/Qwen3-4B",
        "model_revision_match": resolved_revision == MODEL_REVISION,
        "thinking_disabled": result.thinking_disabled,
        "think_tag_absent": not result.think_tag_present,
        "generation_nonempty": result.output_tokens > 0,
        "expected_response_match": result.response == EXPECTED_SMOKE_RESPONSE,
    }
    checks["verdict"] = "PASS" if all(value is True for value in checks.values()) else "FAIL"
    return checks


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_phase0_smoke_evidence(
    output_dir: Path,
    snapshot: Path,
    result: SmokeResult,
) -> None:
    resolved_revision = resolved_revision_from_snapshot(snapshot)
    write_json(output_dir / "environment.json", environment_manifest())
    write_json(output_dir / "model_manifest.json", model_manifest(snapshot))
    write_json(output_dir / "smoke_result.json", asdict(result))
    write_json(
        output_dir / "smoke_validator.json",
        smoke_validator_checks(result, resolved_revision),
    )
