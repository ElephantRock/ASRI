"""HumanEval+ scorer wrapper around the installed evalplus backend.

Uses evalplus's own dataset, ground truth, and sandboxed execution
(check_correctness with fixed timeout policy) rather than reimplementing
code execution. pass@1 uses the HumanEval+ criterion: base AND plus tests
both pass for the single generated candidate.
"""

from __future__ import annotations

import re
import sys
import types
from typing import Any

SCORER_ID = "humanevalplus-evalplus-v1"

# Execution policy frozen per ASRI-P0-v1 §7B.
MIN_TIME_LIMIT = 1.0
GT_TIME_LIMIT_FACTOR = 4.0

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _install_windows_resource_shim() -> None:
    """Compat shim: evalplus's reliability_guard imports the Unix-only
    ``resource`` module when a memory limit is requested. RLIMITs do not
    exist on Windows; a no-op module keeps the guard's remaining protections
    active without changing test semantics."""
    if sys.platform != "win32" or "resource" in sys.modules:
        return
    stub = types.ModuleType("resource")
    stub.RLIMIT_AS = stub.RLIMIT_DATA = stub.RLIMIT_STACK = 0
    stub.setrlimit = lambda *args, **kwargs: None
    sys.modules["resource"] = stub


def extract_code(model_output: str) -> str:
    blocks = _CODE_BLOCK.findall(model_output)
    code = blocks[0] if blocks else model_output
    return code.strip()


def _backend() -> tuple[Any, Any, Any, str]:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import PASS, get_groundtruth

    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(problems, dataset_hash, [])
    return problems, expected_output, dataset_hash, PASS


def score_item(model_output: str, task_id: str) -> dict[str, Any]:
    from evalplus.evaluate import check_correctness

    _install_windows_resource_shim()
    problems, expected_output, dataset_hash, pass_token = _backend()
    if task_id not in problems:
        raise KeyError(f"task {task_id} not in HumanEval+ dataset")
    solution = extract_code(model_output)
    verdict = check_correctness(
        dataset="humaneval",
        completion_id=0,
        problem=problems[task_id],
        solution=solution,
        expected_output=expected_output[task_id],
        base_only=False,
        fast_check=True,
        identifier=task_id,
        min_time_limit=MIN_TIME_LIMIT,
        gt_time_limit_factor=GT_TIME_LIMIT_FACTOR,
    )
    base_status = verdict["base"][0]
    plus_status = verdict["plus"][0]
    base_pass = base_status == pass_token
    plus_pass = plus_status == pass_token
    return {
        "scorer": SCORER_ID,
        "dataset_hash": dataset_hash,
        "base_pass": base_pass,
        "plus_pass": plus_pass,
        "correct": bool(base_pass and plus_pass),
        "failure": None if (base_pass and plus_pass) else (
            "base_fail" if not base_pass else "plus_fail"
        ),
        "solution_chars": len(solution),
    }


def known_good_solution(task_id: str) -> str:
    problems, _, _, _ = _backend()
    return problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]


FIXTURES = [
    {
        "task_id": "HumanEval/11",
        "mode": "canonical",
        "expected_correct": True,
    },
    {
        "task_id": "HumanEval/11",
        "mode": "broken",
        "expected_correct": False,
    },
]


def validate_fixtures() -> list[dict[str, Any]]:
    """Canonical solution must pass; a redefined broken one must fail."""
    results = []
    for index, fixture in enumerate(FIXTURES):
        if fixture["mode"] == "canonical":
            output = f"```python\n{known_good_solution(fixture['task_id'])}\n```"
        else:
            problems, _, _, _ = _backend()
            entry_point = problems[fixture["task_id"]]["entry_point"]
            output = (
                f"```python\n{known_good_solution(fixture['task_id'])}\n\n"
                f"def {entry_point}(*args, **kwargs):\n"
                "    return 'asri-mutated-wrong'\n```"
            )
        verdict = score_item(output, fixture["task_id"])
        results.append(
            {
                "fixture": index,
                "task_id": fixture["task_id"],
                "mode": fixture["mode"],
                "expected_correct": fixture["expected_correct"],
                "actual_correct": verdict["correct"],
                "pass": verdict["correct"] == fixture["expected_correct"],
            }
        )
    return results
