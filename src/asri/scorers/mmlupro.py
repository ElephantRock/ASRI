"""Frozen MMLU-Pro multiple-choice extraction scorer.

Extraction order (frozen): last "answer is (X)" pattern, then last bare
"(X)" parenthetical, then last standalone option-letter token.
"""

from __future__ import annotations

import re
from typing import Any

SCORER_ID = "mmlupro-letter-v1"

_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*\(?([A-J])\)?", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\(([A-J])\)")
_STANDALONE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])")


def extract_choice(text: str) -> str | None:
    matches = _ANSWER_IS.findall(text)
    if matches:
        return matches[-1].upper()
    matches = _PARENTHETICAL.findall(text)
    if matches:
        return matches[-1].upper()
    matches = _STANDALONE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


def score_item(model_output: str, gold_letter: str) -> dict[str, Any]:
    extracted = extract_choice(model_output)
    if extracted is None:
        return {
            "scorer": SCORER_ID,
            "extracted": None,
            "correct": False,
            "failure": "no_choice",
        }
    return {
        "scorer": SCORER_ID,
        "extracted": extracted,
        "correct": extracted == gold_letter.upper(),
        "failure": None if extracted == gold_letter.upper() else "mismatch",
    }


FIXTURES = [
    {"output": "Thinking... The answer is (C)", "gold": "C", "expected_correct": True},
    {"output": "the answer is: d", "gold": "D", "expected_correct": True},
    {"output": "Option (B) looks right", "gold": "B", "expected_correct": True},
    {"output": "I would pick A.", "gold": "A", "expected_correct": True},
    {"output": "The answer is (E)", "gold": "C", "expected_correct": False},
    {"output": "Cannot determine", "gold": "F", "expected_correct": False},
]


def validate_fixtures() -> list[dict[str, Any]]:
    results = []
    for index, fixture in enumerate(FIXTURES):
        verdict = score_item(fixture["output"], fixture["gold"])
        results.append(
            {
                "fixture": index,
                "expected_correct": fixture["expected_correct"],
                "actual_correct": verdict["correct"],
                "pass": verdict["correct"] == fixture["expected_correct"],
            }
        )
    return results
