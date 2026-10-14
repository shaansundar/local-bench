# Contributing to local-bench

Thanks for helping. Contributions of every size are welcome, from fixing a typo, to fixing a grader, to
adding a whole new benchmark. This guide covers:

1. [Ground rules](#ground-rules)
2. [Development setup](#development-setup)
3. [Making a code change](#making-a-code-change)
4. [Adding a new benchmark](#adding-a-new-benchmark)
5. [Changing an existing benchmark or grader](#changing-an-existing-benchmark-or-grader)
6. [Working on the dashboard](#working-on-the-dashboard)
7. [Reporting bugs and asking questions](#reporting-bugs-and-asking-questions)

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security
issues privately, as described in [SECURITY.md](SECURITY.md), not in public issues.

## Ground rules

These keep results trustworthy and the suite runnable on an ordinary Mac:

- **No Docker, VMs or Windows.** Everything has to run natively on macOS. Use `sandbox-exec` (through
  `bench.sandbox`) to isolate model-written code and agents. A benchmark that truly needs a container
  belongs on the excluded list in `docs/BENCHMARKS.md`, along with the reason.
- **Never commit data, results or secrets.** `data/`, `vendor/`, `results/`, `.env` and `models.yaml` are
  gitignored on purpose. Datasets are downloaded from their original source at run time and are never
  redistributed. A pull request that adds dataset files will be asked to remove them.
- **Be honest about differences from the official setup.** If your harness or grader differs from the
  official one, add the matching badge (`adapted`, `substitute` or `partial`) and explain the difference
  in `notes`. A number that looks official but isn't does more harm than a missing benchmark.
- **Comparability comes first.** Don't change the frozen subsets, prompts or grading of an existing
  benchmark without bumping its version (see [below](#changing-an-existing-benchmark-or-grader)). People
  compare results across machines and over time.
- **Treat downloaded data and model output as untrusted.** Never execute them in place, and run them
  only inside the sandbox.

## Development setup

You need the [minimum system requirements](README.md#system-requirements), `uv`, `git`, and Node.js 18
or newer if you'll touch the dashboard.

```bash
git clone https://github.com/<you>/local-bench && cd local-bench
uv sync                              # creates .venv with the core and dev dependencies (pytest)
cp models.example.yaml models.yaml   # your local models; this file is gitignored
uv run pytest                        # unit tests, including a sandbox escape test
uv run bench doctor                  # optional prerequisites and what each one unlocks
```

For an end-to-end check that doesn't need a real model, use the built-in mock models:

```bash
uv run bench validate math500,ifeval              # oracle should score ≈1.0, null ≈0.0
uv run bench run -m tiny-qwen-mlx -b math500 -t smoke   # real 0.5B model; needs `uv tool install mlx-lm`
```

Repository map:

| Path | What lives there |
|---|---|
| `bench/cli.py` | The `bench` command (Typer) |
| `bench/config.py` | `config.yaml` / `models.yaml` loading and secret resolution |
| `bench/models/` | Model clients (OpenAI-compatible, Anthropic, opencode) and the managed server lifecycle |
| `bench/benchmark.py` | The benchmark contract: `BenchmarkSpec`, `Sample`, `ChatRequest`, `AgentTask`, `Grade`, … |
| `bench/runners/` | The `chat` and `oc` (opencode) runners |
| `bench/sandbox/` | The `sandbox-exec` wrapper and sandboxed Python/SQL executors |
| `bench/worker.py`, `bench/store.py` | The job queue, worker, and SQLite results store |
| `bench/judge.py` | The LLM judge: versioned prompts, cache |
| `bench/server/` | FastAPI API, plus the built dashboard in `static/` |
| `benchmarks/<slug>/` | One folder per benchmark |
| `web/` | Dashboard source (Vite + React + TypeScript) |
| `scripts/` | Developer helpers, e.g. `demo_results.py` |

## Making a code change

1. **Open an issue first** for anything bigger than a small fix, so we can agree on the approach before
   you invest time. Small fixes can go straight to a pull request.
2. **Fork, then branch** from `main`, with a name like `fix/scicode-timeout` or `feat/vllm-launcher`.
3. **Match the code around you.** Python 3.12 type hints, `from __future__ import annotations`, small
   focused functions, and comments that explain *why* rather than *what*. There's no auto-formatter yet,
   so keep lines under about 120 columns and follow the file you're editing.
4. **Add or update tests** in `tests/` for any core logic: sampling, metrics, extraction, the sandbox,
   the store. Grader changes are tested with `bench validate` (see below).
5. **Run the checks** that apply to your change:

   | You changed | Run |
   |---|---|
   | Anything in `bench/` | `uv run pytest` |
   | A benchmark adapter or grader | `uv run bench validate <slug>`, plus a smoke run on a real model |
   | Any `benchmark.yaml` | `uv run bench catalog`, and commit the regenerated `docs/BENCHMARKS.md` |
   | Anything in `web/` | `cd web && npm run typecheck`, then `uv run bench serve --build`, and commit `bench/server/static/` |
   | Model lifecycle / launchers | `uv run bench run -m tiny-qwen-mlx -b math500 -t smoke`, and check the server stops afterwards |

6. **Write clear commits.** Use the imperative mood ("Fix SciCode timeout handling"). One logical change
   per commit is ideal.
7. **Open the pull request** using the template. Say what changed, why, and how you tested it. Include
   before and after scores if the change affects grading.

## Adding a new benchmark

New benchmarks are the most valuable contribution. The full walkthrough is
**[docs/NEW_BENCHMARK.md](docs/NEW_BENCHMARK.md)**. It is written so that a person or a coding agent can
follow it without reading the rest of the codebase. Here's the short version.

### Before you start: does it fit?

Open a **"New benchmark proposal"** issue and check it against these requirements:

| Requirement | Why |
|---|---|
| Data is publicly downloadable, or gated behind accepting terms on Hugging Face | Users fetch it themselves; we don't redistribute it |
| Its license allows evaluation use | Results may be published |
| It runs on macOS without Docker, VMs, Windows COM or paid services (other than the configured judge) | Keeps the suite runnable on one Mac |
| Grading is deterministic, executable, or uses the fixed cloud judge with a versioned prompt | Scores must be reproducible |
| It adds signal the suite doesn't already have, or replaces an existing benchmark with a better one | Keeps run time reasonable |

### Building it

```bash
uv run bench new my_bench --runner chat --category Knowledge --title "My Bench"
```

Choose a runner:

- **chat**: one prompt, one answer, or a scripted dialogue.
- **oc**: the model needs files, a shell or the web, so it runs as an opencode agent in a sandbox.
- **native**: an official harness that already drives the model.

Then follow the steps in the guide:

1. Fill in `benchmark.yaml`: title, category, runner, metric, `requires`, badges, and pinned `sources`.
2. Implement `fetch()`, `load()`, `build()` and `grade()` in `adapter.py`. Reuse the helpers in
   `bench/common.py`, `bench/families.py`, `bench/office.py` and `bench/sandbox/`.
3. Freeze the subsets with `uv run bench subsets my_bench`, and commit `benchmarks/my_bench/subsets/`.
4. If you need a judge, add a judge prompt with its own version.
5. Self-test:
   - `uv run bench validate my_bench`: the oracle must score ≈1.0 and the null model ≈0.0.
   - Do a smoke run on a real model and inspect the samples in the dashboard's run explorer.
6. Run `uv run bench catalog`, and commit `docs/BENCHMARKS.md`.

### What the pull request should contain

- `benchmarks/<slug>/benchmark.yaml`, `adapter.py`, any helper files, and the frozen `subsets/*.json`.
- The regenerated `docs/BENCHMARKS.md`.
- No data files, no vendored upstream code (fetch it at a pinned commit in `fetch()`), and no results.
- In the description:
  - the `bench validate` output
  - the smoke-run score of at least one real model
  - how long a smoke run took
  - every deviation from the official harness, and the badge you chose for it

## Changing an existing benchmark or grader

Results are compared across machines and over time, so changes to what is measured have to be visible:

- **Prompts, extraction or grading logic:** describe the effect in the pull request with before and after
  scores on the same stored outputs. `bench regrade --run <id>` re-grades without generating again. Add
  an entry to `CHANGELOG.md`.
- **Judge prompts:** bump the prompt's entry in `PROMPT_VERSIONS` (`bench/judge.py`). Judged scores record
  the version, so old and new results stay distinguishable.
- **Subsets:** don't regenerate the frozen subsets of an existing benchmark. If the upstream dataset
  changes, treat it as a new benchmark version (for example a new slug or a version suffix in the title).
- **Badges:** if your change brings the harness closer to the official one, it's fine to remove
  `adapted` or `partial`. Say so in the pull request.

## Working on the dashboard

The dashboard is a read-only React app served by `bench serve`. To develop it without running any models:

```bash
uv run python scripts/demo_results.py /tmp/bench-demo     # synthetic results for 5 fictional models
BENCH_MODELS=/tmp/bench-demo/models.yaml BENCH_RESULTS_DIR=/tmp/bench-demo/results uv run bench serve --dev
cd web && npm ci && npm run dev                             # Vite dev server, proxies /api
```

Follow the existing design tokens in `web/src/styles.css`. Both light and dark themes are defined
explicitly, not by flipping colours. Charts use a single axis, thin marks and direct labels, and every
mark has a tooltip. When you're done, run `npm run typecheck` and rebuild with
`uv run bench serve --build`. Commit the updated `bench/server/static/`, so users without Node get your
change.

## Reporting bugs and asking questions

- **Bugs:** use the bug report template. Include:
  - your macOS version and chip
  - the output of `uv run bench doctor`
  - the model entry from `models.yaml`, with keys removed
  - the command you ran
  - the relevant part of the worker log or the dashboard's error

  A run id and sample id help a lot.
- **Questions and ideas:** open a discussion or an issue. Questions are welcome; if something is
  confusing, that's a documentation bug.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE),
the same license that covers the project.
