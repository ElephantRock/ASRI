# Vendored IFEval evaluator — provenance

Vendored verbatim from the official implementation, with one mechanical
modification: absolute imports (`from instruction_following_eval import ...`)
were rewritten as relative imports so the module lives inside `asri.scorers`.

```text
upstream:    https://github.com/google-research/google-research/tree/master/instruction_following_eval
pinned sha:  e6890f85757dd84e27ca6df2dd30651dafad28e0
commit date: 2026-07-22T20:03:19Z
commit msg:  "Adding type suppressions for pyrefly"
license:     Apache-2.0 (see LICENSE)
files:       instructions.py, instructions_registry.py, instructions_util.py
modifications: import paths only
```

Per ASRI-P0-v1 §7C, the evaluator is pinned by commit rather than tracked
against a moving master. The pinned sha is recorded in `scorer_manifest.json`
at run time.
