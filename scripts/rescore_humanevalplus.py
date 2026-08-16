"""Re-score persisted HumanEval+ responses in a light process.

Deterministic: verdicts depend only on (response text, task_id). This does
NOT regenerate any model output; it re-runs the scorer over the persisted
per-item evidence to detect verdict flips caused by execution-context
differences (heavy parent __main__ inflating evalplus child spawn cost).
Writes a comparison report; leaves the original file untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from asri.scorers import humanevalplus

SOURCE = Path("evidence/asri_p0_v1/per_item/humanevalplus/quality.jsonl")
REPORT = Path("evidence/asri_p0_v1/per_item/humanevalplus/rescore_report.json")
RESCORED = Path("evidence/asri_p0_v1/per_item/humanevalplus/quality_rescored.jsonl")


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    flips = []
    out_rows = []
    for row in rows:
        task_id = row["item_id"].replace("humanevalplus-", "").replace("__", "/")
        verdict = humanevalplus.score_item(row["generation"]["response"], task_id)
        new_row = dict(row)
        new_row["verdict"] = verdict
        new_row["rescored"] = True
        out_rows.append(new_row)
        if verdict["correct"] != row["verdict"]["correct"]:
            flips.append(
                {
                    "item_id": row["item_id"],
                    "old_failure": row["verdict"].get("failure"),
                    "new_failure": verdict.get("failure"),
                    "old_correct": row["verdict"]["correct"],
                    "new_correct": verdict["correct"],
                }
            )
    RESCORED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in out_rows) + "\n",
        encoding="utf-8",
    )
    old_score = sum(1 for r in rows if r["verdict"]["correct"]) / len(rows)
    new_score = sum(1 for r in out_rows if r["verdict"]["correct"]) / len(out_rows)
    report = {
        "n": len(rows),
        "old_score": old_score,
        "new_score": new_score,
        "flips": flips,
        "n_flips": len(flips),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "flips"}))
    for flip in flips[:10]:
        print(flip, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
