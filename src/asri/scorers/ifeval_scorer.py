"""IFEval strict/loose scorer wrapper around the vendored official evaluator.

The evaluator is pinned by commit (see ifeval/PROVENANCE.md). This wrapper
adapts the official (InputExample, prompt->response) API to per-item scoring
without touching its logic; loose evaluation follows the official eight
response variants.
"""

from __future__ import annotations

from typing import Any

from .ifeval import evaluation_lib

SCORER_ID = "ifeval-official-e6890f8-v1"


def score_item(
    prompt: str,
    model_output: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> dict[str, Any]:
    inp = evaluation_lib.InputExample(
        key=0,
        instruction_id_list=list(instruction_id_list),
        prompt=prompt,
        kwargs=[
            {k: v for k, v in kwargs.items()} if kwargs else {}
            for kwargs in kwargs_list
        ],
    )
    prompt_to_response = {prompt: model_output}
    strict = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return {
        "scorer": SCORER_ID,
        "strict_followed": bool(strict.follow_all_instructions),
        "loose_followed": bool(loose.follow_all_instructions),
        "instruction_strict": [bool(v) for v in strict.follow_instruction_list],
        "instruction_loose": [bool(v) for v in loose.follow_instruction_list],
        "correct": bool(strict.follow_all_instructions),
        "failure": None if strict.follow_all_instructions else "instruction_violated",
    }


FIXTURES = [
    {
        "prompt": "Write an essay about the sea.",
        "output": "Tangible waves crash. " * 40,
        "instruction_id_list": ["keywords:existence"],
        "kwargs_list": [{"keywords": ["tangible"]}],
        "expected_strict": True,
    },
    {
        "prompt": "Write an essay about the sea.",
        "output": "This response omits the required keyword entirely.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs_list": [{"keywords": ["tangible"]}],
        "expected_strict": False,
    },
    {
        "prompt": "Write a story.",
        "output": "first\n***\nsecond\n***\nthird\n***\nfourth",
        "instruction_id_list": ["length_constraints:number_paragraphs"],
        "kwargs_list": [{"num_paragraphs": 4}],
        "expected_strict": True,
    },
]


def validate_fixtures() -> list[dict[str, Any]]:
    results = []
    for index, fixture in enumerate(FIXTURES):
        verdict = score_item(
            fixture["prompt"],
            fixture["output"],
            fixture["instruction_id_list"],
            fixture["kwargs_list"],
        )
        expected = fixture["expected_strict"]
        results.append(
            {
                "fixture": index,
                "expected_strict": expected,
                "actual_strict": verdict["strict_followed"],
                "pass": bool(verdict["strict_followed"]) == bool(expected),
            }
        )
    return results
