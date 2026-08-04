# AGENTS.md

## Testing preference

When running ad-hoc Python experiments or one-off verification scripts, do NOT type long inline `python -c "..."` commands into the terminal (they are error-prone). Instead:

1. Create a temporary script file in the repo root named with a leading underscore, e.g. `_test.py` or `_test_llm_expansion.py`.
2. Run it with the appropriate interpreter, e.g. `/Users/ivanlee/miniconda3/envs/ocr/bin/python _test.py`.
3. Delete the temp file afterward.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/` (one feature per directory, one ticket per file). See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map to the default label strings: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.