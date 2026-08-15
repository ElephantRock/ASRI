"""Frozen MATH-500 answer extraction and equivalence scorer.

Per ASRI-P0-v1 §7A the prompt, extraction, and scorer are frozen locally
(reference: openai/simple-evals MATH-500 task definition). The scorer extracts
the last \\boxed{...}, normalizes Minerva-style, and falls back to a guarded
sympy equivalence check.
"""

from __future__ import annotations

import re
import string
import threading
from typing import Any

# Scorer identity recorded in the scorer manifest.
SCORER_ID = "math500-boxed-minerva-v1"

_FIXTURES: list[dict[str, Any]] = []


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...} with brace matching."""
    marker = "\\boxed"
    start = text.rfind(marker)
    while start != -1:
        cursor = start + len(marker)
        if cursor < len(text) and text[cursor] == "{":
            depth = 0
            for index in range(cursor, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[cursor + 1 : index]
        start = text.rfind(marker, 0, start)
    return None


_STRIPPPE_PREFIXES = ("\\displaystyle", "\\text", "\\textrm", "\\mathrm")


def _strip_outer_braces(value: str) -> str:
    """Remove wrapping braces ({answer}) without touching \\frac{a}{b}."""
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        matched = True
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0 and index < len(value) - 1:
                matched = False
                break
        if not matched:
            break
        value = value[1:-1].strip()
    return value


def normalize_answer(answer: str) -> str:
    lowered = answer.strip().lower()
    lowered = lowered.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    lowered = lowered.replace("\\left", "").replace("\\right", "")
    lowered = lowered.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    lowered = lowered.replace("\\%", "").replace("%", "")
    lowered = lowered.replace("$", "").replace("\\$", "")
    lowered = lowered.replace("^{\\circ}", "").replace("^\\circ", "")
    lowered = lowered.replace("\\degree", "")
    for prefix in _STRIPPPE_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
    lowered = _strip_outer_braces(lowered.strip())
    # Drop trailing sentence punctuation but keep decimals/negatives intact.
    lowered = lowered.rstrip(".")
    # Remove commas that group digits (1,000) but not coordinate tuples.
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", lowered):
        lowered = lowered.replace(",", "")
    lowered = " ".join(lowered.split())
    return lowered


def _latex_to_pythonish(expr: str) -> str:
    converted = expr
    converted = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", converted)
    converted = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", converted)
    converted = converted.replace("\\pi", "pi").replace("\\infty", "oo")
    converted = converted.replace("^", "**").replace("\\cdot", "*").replace("\\times", "*")
    converted = re.sub(r"[{}\[\]]", "", converted)
    converted = converted.translate(str.maketrans("", "", string.whitespace))
    return converted


def _sympy_equal(candidate: str, reference: str, timeout_seconds: float = 5.0) -> bool | None:
    """Guarded symbolic equivalence; None means 'undecidable within timeout'."""
    import sympy

    outcome: dict[str, Any] = {"equal": None}

    def _work() -> None:
        try:
            left = sympy.sympify(_latex_to_pythonish(candidate))
            right = sympy.sympify(_latex_to_pythonish(reference))
            outcome["equal"] = sympy.simplify(left - right) == 0
        except Exception:
            outcome["equal"] = None

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    return outcome["equal"]


def score_item(model_output: str, gold_answer: str) -> dict[str, Any]:
    extracted = extract_boxed(model_output)
    if extracted is None:
        return {
            "scorer": SCORER_ID,
            "extracted": None,
            "correct": False,
            "failure": "no_boxed",
        }
    normalized = normalize_answer(extracted)
    normalized_gold = normalize_answer(gold_answer)
    if normalized == normalized_gold:
        return {
            "scorer": SCORER_ID,
            "extracted": extracted,
            "correct": True,
            "failure": None,
        }
    symbolic = _sympy_equal(normalized, normalized_gold)
    if symbolic is True:
        return {
            "scorer": SCORER_ID,
            "extracted": extracted,
            "correct": True,
            "failure": None,
        }
    return {
        "scorer": SCORER_ID,
        "extracted": extracted,
        "correct": False,
        "failure": "mismatch" if symbolic is False else "undecidable",
    }


_FIXTURES.extend(
    [
        {
            "output": "Blah. The final answer is $\\boxed{\\frac{1}{2}}$.",
            "gold": "\\frac{1}{2}",
            "expected_correct": True,
        },
        {
            "output": "So \\boxed{0.5} done",
            "gold": "\\frac{1}{2}",
            "expected_correct": True,
        },
        {
            "output": "answer \\boxed{x^2 + 1}",
            "gold": "1 + x^2",
            "expected_correct": True,
        },
        {
            "output": "no box here, just text",
            "gold": "3",
            "expected_correct": False,
        },
        {
            "output": "\\boxed{\\dfrac{3}{4}}",
            "gold": "0.75",
            "expected_correct": True,
        },
        {
            "output": "first \\boxed{7} then \\boxed{11}",
            "gold": "11",
            "expected_correct": True,
        },
        {
            "output": "\\boxed{2\\pi}",
            "gold": "2\\pi",
            "expected_correct": True,
        },
        {
            "output": "\\boxed{5}",
            "gold": "6",
            "expected_correct": False,
        },
    ]
)


def validate_fixtures() -> list[dict[str, Any]]:
    results = []
    for index, fixture in enumerate(_FIXTURES):
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
