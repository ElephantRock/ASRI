"""Frozen dataset acquisition, role partitioning, and manifests for Phase 0.

Per ASRI-P0-v1 §7/§8: every workload is pinned to an immutable upstream
revision, ASRI's prompts are frozen locally, and benchmark items are assigned
to roles (development / baseline-characterization / held-out / negative-control)
by a deterministic hash so the assignment can be persisted and audited.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

CONTEXT_CAP_TOKENS = 4096

DATASET_REVISIONS = {
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "split": "test",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "split": "train",
    },
    "mmlupro": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
    },
    "humanevalplus": {
        "repo": "evalplus/humanevalplus",
        "package": "evalplus",
        "loader": "get_human_eval_plus",
    },
}

# Deterministic role partition (ASRI-P0-v1 §8). Assignment order:
# dev -> held-out -> negative-control -> baseline-characterization.
ROLE_POLICY = {
    "math500": {"dev_fraction": 0.032, "heldout_fraction": 0.20},
    "humanevalplus": {"dev_fraction": 0.10, "heldout_fraction": 0.20},
    "ifeval": {"dev_fraction": 0.03, "heldout_fraction": 0.20},
    "mmlupro": {"dev_fraction": 0.07, "heldout_fraction": 0.20},
}

# Knowledge/recall-leaning MMLU-Pro categories: negative-control role, items
# not expected to benefit materially from additional recurrent reasoning.
MMLU_PRO_NEGATIVE_CONTROL_CATEGORIES = frozenset(
    {"business", "economics", "health", "history", "law", "psychology", "other"}
)

MMLU_PRO_SLICE_SIZE = 240

PARTITION_SALT = "asri-p0-v1"

ROLE_ORDER = ("dev", "held_out", "negative_control", "characterization")
EXECUTABLE_ROLES = ("dev", "negative_control", "characterization")

MATH500_PROMPT = (
    "Solve the following math problem. Show your reasoning concisely, then "
    "give the final answer in \\boxed{}.\n\n{problem}"
)

HUMANEVALPLUS_PROMPT = (
    "Complete the following Python function. Reply with a single Python code "
    "block containing the complete function definition.\n\n"
    "```python\n{prompt}```"
)

IFEVAL_PROMPT = "{prompt}"

MMLUPRO_PROMPT = (
    "Answer the following multiple-choice question. Think briefly, then end "
    "your response with the final answer in the exact format "
    "'The answer is (X)' where X is one of A-J.\n\n{question}\n\n{options}"
)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


@dataclass(frozen=True)
class WorkloadItem:
    workload: str
    item_id: str
    role: str
    category: str | None
    prompt: str
    reference: dict[str, Any]


def _render(template: str, **fields: str) -> str:
    """Token replacement immune to literal braces (e.g. \\boxed{})."""
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _hash_rank(workload: str, item_id: str) -> float:
    digest = hashlib.sha256(f"{PARTITION_SALT}:{workload}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_role(workload: str, item_id: str, category: str | None = None) -> str:
    """Deterministically map an item to one of the four ASRI roles."""
    policy = ROLE_POLICY[workload]
    rank = _hash_rank(workload, item_id)
    if rank < policy["dev_fraction"]:
        return "dev"
    if rank < policy["dev_fraction"] + policy["heldout_fraction"]:
        return "held_out"
    if (
        workload == "mmlupro"
        and category is not None
        and category.lower() in MMLU_PRO_NEGATIVE_CONTROL_CATEGORIES
    ):
        return "negative_control"
    return "characterization"


def _canonical_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def load_math500() -> list[WorkloadItem]:
    from datasets import load_dataset

    pinned = DATASET_REVISIONS["math500"]
    dataset = load_dataset(pinned["repo"], split=pinned["split"], revision=pinned["revision"])
    items = []
    for row in dataset:
        item_id = f"math500-{row['unique_id']}"
        items.append(
            WorkloadItem(
                workload="math500",
                item_id=item_id,
                role=assign_role("math500", item_id),
                category=row.get("subject"),
                prompt=_render(MATH500_PROMPT, problem=row["problem"]),
                reference={
                    "answer": str(row["answer"]),
                    "level": row.get("level"),
                },
            )
        )
    return items


def load_ifeval() -> list[WorkloadItem]:
    from datasets import load_dataset

    pinned = DATASET_REVISIONS["ifeval"]
    dataset = load_dataset(pinned["repo"], split=pinned["split"], revision=pinned["revision"])
    items = []
    for row in dataset:
        item_id = f"ifeval-{row['key']}"
        items.append(
            WorkloadItem(
                workload="ifeval",
                item_id=item_id,
                role=assign_role("ifeval", item_id),
                category=None,
                prompt=_render(IFEVAL_PROMPT, prompt=row["prompt"]),
                reference={
                    "instruction_id_list": list(row["instruction_id_list"]),
                    "kwargs": [
                        {k: v for k, v in kwarg.items()} for kwarg in row["kwargs"]
                    ],
                },
            )
        )
    return items


def load_mmlu_pro() -> list[WorkloadItem]:
    """Load the frozen bounded MMLU-Pro slice (hash-rank smallest N of test)."""
    from datasets import load_dataset

    pinned = DATASET_REVISIONS["mmlupro"]
    dataset = load_dataset(pinned["repo"], split=pinned["split"], revision=pinned["revision"])
    ranked = sorted(
        (_hash_rank("mmlupro-slice", str(index)), index, row)
        for index, row in enumerate(dataset)
    )
    items = []
    for _, index, row in ranked[:MMLU_PRO_SLICE_SIZE]:
        options = row["options"]
        option_lines = "\n".join(
            f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options)
        )
        item_id = f"mmlupro-test-{index}"
        category = row["category"]
        items.append(
            WorkloadItem(
                workload="mmlupro",
                item_id=item_id,
                role=assign_role("mmlupro", item_id, category),
                category=category,
                prompt=_render(MMLUPRO_PROMPT,
                    question=row["question"], options=option_lines
                ),
                reference={
                    "answer_index": int(row["answer_index"]),
                    "answer_letter": OPTION_LETTERS[int(row["answer_index"])],
                    "subcategory": row.get("subcategory"),
                },
            )
        )
    items.sort(key=lambda item: item.item_id)
    return items


def load_humaneval_plus() -> list[WorkloadItem]:
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    items = []
    for task_id in sorted(problems):
        problem = problems[task_id]
        item_id = f"humanevalplus-{task_id.replace('/', '__')}"
        items.append(
            WorkloadItem(
                workload="humanevalplus",
                item_id=item_id,
                role=assign_role("humanevalplus", item_id),
                category=None,
                prompt=_render(HUMANEVALPLUS_PROMPT, prompt=problem["prompt"]),
                reference={"task_id": task_id},
            )
        )
    return items


def load_workload(workload: str) -> list[WorkloadItem]:
    loaders = {
        "math500": load_math500,
        "ifeval": load_ifeval,
        "mmlupro": load_mmlu_pro,
        "humanevalplus": load_humaneval_plus,
    }
    if workload not in loaders:
        raise ValueError(f"unknown workload: {workload}")
    return loaders[workload]()


def dataset_manifest(items: list[WorkloadItem]) -> dict[str, Any]:
    """Build the manifest entry for one workload's loaded items.

    Records provenance, role assignment, and content digests. Deliberately
    free of local filesystem paths.
    """
    workload = items[0].workload if items else "unknown"
    roles: dict[str, list[str]] = {role: [] for role in ROLE_ORDER}
    for item in items:
        roles[item.role].append(item.item_id)
    scoring_payload = [
        {"id": item.item_id, "prompt": item.prompt, "reference": item.reference}
        for item in sorted(items, key=lambda i: i.item_id)
    ]
    manifest = {
        "source": DATASET_REVISIONS[workload],
        "n_items": len(items),
        "role_counts": {role: len(ids) for role, ids in roles.items()},
        "role_item_ids": roles,
        "partition": {
            "salt": PARTITION_SALT,
            "policy": ROLE_POLICY[workload],
            "negative_control_categories": (
                sorted(MMLU_PRO_NEGATIVE_CONTROL_CATEGORIES)
                if workload == "mmlupro"
                else []
            ),
            "slice_size": MMLU_PRO_SLICE_SIZE if workload == "mmlupro" else None,
        },
        "content_digest": _canonical_digest(scoring_payload),
    }
    return manifest
