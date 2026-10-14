# Changelog

All notable changes are documented here. This project follows [Semantic Versioning](https://semver.org).
Changes that alter how a benchmark is prompted or graded are always listed, because they affect
whether scores are comparable.

## [0.1.0] - 2026-10-14

The first public release.

### Added

- **The `bench` CLI**, one master script: `list`, `models`, `doctor`, `fetch`, `subsets`, `run`,
  `worker`, `status`, `results`, `regrade`, `cancel`, `validate`, `serve`, `new` and `catalog`.
- **42 benchmarks across 13 categories:**
  - Coding
  - Math
  - Instruction following
  - Knowledge
  - Long context
  - Tool use
  - Search agents
  - General agents
  - Spreadsheets
  - Presentations
  - Office documents
  - Document / chart vision
  - Text-to-SQL

  Adapted, substitute, partial and judge-graded benchmarks carry badges. See `docs/BENCHMARKS.md`.
- **Three runners:**
  - direct OpenAI-compatible / Anthropic chat
  - sandboxed `opencode run` agents with a keyless, cached web-search MCP server
  - official harnesses in their own virtual environments (τ-bench, BFCL v4, WTM-Bench, Stirrup for GDPval-AA)
- **Model kinds:** external servers, managed servers (ollama, llama.cpp, mlx_lm, mlx_vlm, custom
  command) with a memory check and automatic stop, cloud APIs, and opencode-only models, including
  opencode's free tier (`opencode_stock_prompt`).
- **Docker-free isolation** with macOS `sandbox-exec`: workspace-only writes, unreadable secret stores,
  network access per task, and resource limits for model-written code.
- **Frozen, seeded, stratified subsets** (`smoke` / `quick` / `standard` / `full`), committed for every
  benchmark whose data is public.
- **A SQLite results store.** Runs resume, judge-pending samples can be regraded, and Wilson or bootstrap
  95% confidence intervals are computed.
- **A fixed cloud judge** with versioned prompts. The judge id and prompt version are stored with every
  judged score.
- **A live dashboard** (FastAPI + React):
  - leaderboard heat-matrices
  - benchmark pages with confidence-interval bars and score vs tokens / time
  - model profiles
  - a run explorer with transcripts and workspace files
  - live progress over SSE
  - light and dark themes
- **`docs/NEW_BENCHMARK.md`**, a step-by-step guide to adding a benchmark that a person or an agent can
  follow, plus `bench new` scaffolding.
- **`scripts/demo_results.py`**, synthetic demo data for working on the dashboard.

[0.1.0]: https://github.com/shaansundar/local-bench/releases/tag/v0.1.0
