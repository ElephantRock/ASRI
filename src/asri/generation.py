"""Frozen generation policy and instrumented inference for Phase 0.

Per ASRI-P0-v1 §5/§9/§11: batch 1, KV cache enabled, thinking disabled,
context cap 4096, one frozen sampling configuration (Qwen3 non-thinking
recommended operating mode) with per-item deterministic seeds, plus a greedy
secondary diagnostic. Timing is CUDA-synchronized; prefill is measured with a
dedicated forward pass and decode latency is derived by subtraction (method
recorded in the generation manifest).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch

GLOBAL_SEED = 20260815

# Qwen3 non-thinking recommended sampling: temperature 0.6, top-p 0.95,
# top-k 20. Workload-specific output budgets frozen here.
QUALITY_CONFIGS = {
    "math500": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 768},
    "humanevalplus": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1024},
    "ifeval": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 640},
    "mmlupro": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 512},
}


@dataclass(frozen=True)
class GenerationRecord:
    response: str
    prompt_tokens: int
    output_tokens: int
    e2e_seconds: float
    prefill_seconds: float
    decode_seconds: float
    tokens_per_second: float
    forward_count: int
    flop_estimate: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    host_rss_bytes: int
    seed: int
    config: dict[str, Any]


def item_seed(workload: str, item_id: str) -> int:
    digest = hashlib.sha256(f"gen:{GLOBAL_SEED}:{workload}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


class TimedGenerator:
    """Wraps the frozen model with synchronized, instrumented generation."""

    def __init__(self, tokenizer: Any, model: Any, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.context_cap = 4096
        self.non_embedding_parameters = self._non_embedding_parameters()

    def _non_embedding_parameters(self) -> int:
        total = 0
        for name, tensor in self.model.named_parameters():
            if "embed" in name:
                continue
            total += tensor.numel()
        return int(total)

    def render(self, user_content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(self, item_workload: str, user_content: str, *, greedy: bool = False) -> GenerationRecord:
        config = dict(QUALITY_CONFIGS[item_workload])
        seed = item_seed(item_workload, user_content_hash(user_content))
        rendered = self.render(user_content)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] + config["max_new_tokens"] > self.context_cap:
            config["max_new_tokens"] = max(
                64, self.context_cap - input_ids.shape[-1]
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(self.device)

        # Prefill measured with a dedicated forward pass over the same inputs.
        prefill_start = time.perf_counter()
        with torch.inference_mode():
            self.model(**encoded)
        if is_cuda:
            torch.cuda.synchronize(self.device)
        prefill_seconds = time.perf_counter() - prefill_start

        gen_kwargs = {
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": not greedy,
            "use_cache": True,
        }
        if not greedy:
            # transformers 5.x removed the per-call generator argument; the
            # device RNG is seeded immediately before each sampled call so
            # per-item sampling remains deterministic under the frozen seed.
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            gen_kwargs.update(
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
            )

        e2e_start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**encoded, **gen_kwargs)
        if is_cuda:
            torch.cuda.synchronize(self.device)
        e2e_seconds = time.perf_counter() - e2e_start

        prompt_tokens = int(input_ids.shape[-1])
        output_ids = generated[0, prompt_tokens:]
        output_tokens = int(output_ids.numel())
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        peak_allocated = peak_reserved = None
        if is_cuda:
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))

        # FLOP accounting: ~2 * non-embedding active params per processed token;
        # prefill covers prompt tokens, each decode step covers one token.
        flops = 2 * self.non_embedding_parameters * (prompt_tokens + output_tokens)

        record = GenerationRecord(
            response=response,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            e2e_seconds=e2e_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=max(0.0, e2e_seconds - prefill_seconds),
            tokens_per_second=(output_tokens / e2e_seconds) if e2e_seconds > 0 else 0.0,
            forward_count=1 + max(0, output_tokens - 1),
            flop_estimate=int(flops),
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            host_rss_bytes=int(psutil.Process().memory_info().rss),
            seed=seed,
            config={**config, "greedy": greedy, "context_cap": self.context_cap},
        )
        return record


def user_content_hash(user_content: str) -> str:
    # The per-item seed must key on the rendered task, not a surrogate id.
    return hashlib.sha256(user_content.encode()).hexdigest()[:16]


def generation_manifest(tokenizer: Any, model: Any) -> dict[str, Any]:
    parameter = next(model.parameters())
    return {
        "global_seed": GLOBAL_SEED,
        "quality_configs": QUALITY_CONFIGS,
        "secondary_diagnostic": "greedy (temperature 0) consistency control only",
        "sampling_reference": "Qwen3 model card non-thinking recommended settings",
        "seed_mechanism": (
            "per-item seed applied via torch.cuda.manual_seed immediately "
            "before each sampled call (transformers 5.x removed per-call "
            "generator argument)"
        ),
        "chat_template_sha256": hashlib.sha256(
            getattr(tokenizer, "chat_template", "").encode()
        ).hexdigest(),
        "dtype": str(parameter.dtype),
        "total_parameters": int(model.num_parameters()),
        "non_embedding_parameters": int(
            sum(t.numel() for n, t in model.named_parameters() if "embed" not in n)
        ),
        "flop_formula": "2 * non_embedding_parameters * (prompt_tokens + output_tokens)",
        "prefill_decode_method": (
            "prefill measured via dedicated synchronized forward pass; "
            "decode derived as e2e minus prefill"
        ),
        "context_cap": 4096,
        "batching": 1,
        "kv_cache": True,
    }


def record_to_dict(record: GenerationRecord) -> dict[str, Any]:
    return asdict(record)
