## What and why

<!-- What does this change, and why? Link the issue it closes (e.g. "Closes #12"). -->

## Type of change

- [ ] Bug fix
- [ ] New benchmark
- [ ] Change to an existing benchmark's prompts / grading / subsets (affects comparability)
- [ ] Core (CLI, runners, models, sandbox, store)
- [ ] Dashboard
- [ ] Docs

## How I tested it

<!-- Commands you ran and what they printed. For grading changes, include before/after scores. -->

- [ ] `uv run pytest`
- [ ] `uv run bench validate <slugs>` (oracle ≈ 1.0, null ≈ 0.0)
- [ ] Smoke run on a real model: `uv run bench run -m <model> -b <slugs> -t smoke`
- [ ] Dashboard: `cd web && npm run typecheck` and `uv run bench serve --build`

## Checklist

- [ ] No dataset files, results, `models.yaml`, `.env` or secrets are included
- [ ] Docs are updated (README, `docs/NEW_BENCHMARK.md`, `CHANGELOG.md`) where behaviour changed
- [ ] `benchmark.yaml` changes: I ran `uv run bench catalog` and committed `docs/BENCHMARKS.md`
- [ ] New benchmark: subsets are frozen and committed, sources are pinned, and every deviation from the official harness has a badge and a note
- [ ] Judge prompt changes: I bumped the version in `PROMPT_VERSIONS`
- [ ] Dashboard changes: I committed the rebuilt `bench/server/static/`
