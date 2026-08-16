"""HumanEval+ scorer wrapper around the installed evalplus backend.

Uses evalplus's own dataset, ground truth, and sandboxed execution
(check_correctness with fixed timeout policy) rather than reimplementing
code execution. pass@1 uses the HumanEval+ criterion: base AND plus tests
both pass for the single generated candidate.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from typing import Any

SCORER_ID = "humanevalplus-evalplus-v1"

# Execution policy frozen per ASRI-P0-v1 §7B.
MIN_TIME_LIMIT = 1.0
GT_TIME_LIMIT_FACTOR = 4.0

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

_CHILD_CHECK = (
    "import resource, signal\n"
    "assert hasattr(signal, 'setitimer') and hasattr(signal, 'SIGALRM')\n"
    "print('ok')\n"
)

_PREFLIGHT_CACHE: dict[str, Any] | None = None


def windows_compat_preflight() -> dict[str, Any]:
    """Verify a fresh child interpreter can import the Unix-only facilities
    evalplus needs (resource module, setitimer/SIGALRM). On Windows these only
    exist when the ``zz_asri_windows_compat.pth`` hook is registered in the
    venv; a fresh environment without it silently fails every execution."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_CHECK],
            capture_output=True,
            text=True,
            timeout=60,
        )
        hook_active = result.returncode == 0 and result.stdout.strip() == "ok"
        detail = result.stderr.strip()[:200] if result.returncode != 0 else None
    except (OSError, subprocess.TimeoutExpired) as error:
        hook_active = False
        detail = repr(error)[:200]
    return {
        "platform": sys.platform,
        "hook_active": bool(hook_active),
        "checked": ["resource", "signal.setitimer", "signal.SIGALRM"],
        "failure_detail": detail,
    }


def ensure_windows_compat() -> dict[str, Any]:
    """Run (once per process) and enforce the compat preflight."""
    global _PREFLIGHT_CACHE
    if _PREFLIGHT_CACHE is None:
        _PREFLIGHT_CACHE = windows_compat_preflight()
    if not _PREFLIGHT_CACHE["hook_active"]:
        raise RuntimeError(
            "HumanEval+ execution is not viable in this environment: evalplus "
            "requires Unix-only resource/signal facilities in its child "
            "processes and the Windows compatibility hook is not active. "
            "Register it with:\n"
            "  echo \"import asri._windows_evalplus_compat  # noqa: F401\" > "
            ".venv/Lib/site-packages/zz_asri_windows_compat.pth\n"
            "Without it every candidate silently reports as failed."
        )
    return _PREFLIGHT_CACHE


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


_CHILD_MARKER = "ASRI_HE_RESULT:"
_CHILD_ENV = "ASRI_HE_PLUS_CHILD"

_CHILD_RUNNER = (
    "import json, os, sys\n"
    f"os.environ['{_CHILD_ENV}'] = '1'\n"
    "payload = json.loads(sys.stdin.read())\n"
    "from asri.scorers.humanevalplus import _score_direct\n"
    "verdict = _score_direct(payload['output'], payload['task_id'])\n"
    f"sys.stdout.write('{_CHILD_MARKER}' + json.dumps(verdict) + chr(10))\n"
    "sys.stdout.flush()\n"
)


def _score_direct(model_output: str, task_id: str) -> dict[str, Any]:
    """Score inside this (child) process using evalplus's own executor."""
    from evalplus.evaluate import check_correctness

    ensure_windows_compat()
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


def score_item(model_output: str, task_id: str) -> dict[str, Any]:
    """Score one candidate, isolated in a light child interpreter.

    evalplus executes candidates in spawned grandchildren that re-import the
    calling interpreter's site. When the caller is heavy (model loaded), that
    import cost can exceed evalplus's outer per-task wall clock and mark
    correct solutions failed. Every scoring call therefore runs in a minimal
    ``python -c`` child whose grandchildren spawn quickly; results depend only
    on (response text, task_id), never on the calling context. Note
    ``EVALPLUS_TIMEOUT_PER_TASK`` cannot be used to widen the budget: evalplus
    compares the env string against a float (upstream bug).
    """
    import os

    ensure_windows_compat()
    if os.environ.get(_CHILD_ENV) == "1":
        return _score_direct(model_output, task_id)
    payload = json.dumps({"output": model_output, "task_id": task_id})
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_RUNNER],
        input=payload,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in result.stdout.splitlines():
        if line.startswith(_CHILD_MARKER):
            return json.loads(line[len(_CHILD_MARKER):])
    raise RuntimeError(
        f"isolated HumanEval+ scoring failed: rc={result.returncode} "
        f"stderr_tail={result.stderr[-400:]!r}"
    )


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
