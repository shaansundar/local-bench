# Creating a new benchmark from scratch

This guide is written so that a person **or a coding agent** can add a benchmark end to end without
reading the rest of the codebase. Follow the steps in order; each ends with a check you can run.

A benchmark is one folder, `benchmarks/<slug>/`, containing:

| File | Purpose |
|---|---|
| `benchmark.yaml` | Metadata: title, category, runner type, metric, requirements, sources (validated by `BenchmarkSpec` in `bench/benchmark.py`) |
| `adapter.py` | A `Benchmark` subclass exported as `BENCHMARK` — how to download, load, prompt and grade |
| `subsets/<tier>.json` | Frozen sample-id lists per tier (written automatically, then committed) |
| any helper files | e.g. a script that runs inside a sandbox or a vendored harness venv |

Nothing else needs registering: the registry discovers folders that contain a `benchmark.yaml`, and
the dashboard, scheduler, `bench list`, `bench run` and `bench validate` pick it up automatically.

---

## 1. Decide what you are building

Write down, before any code:

1. **Unit of evaluation** — what is one sample? (a question, a conversation, a repo task, a workbook…)
2. **Gold signal** — exact answer, executable tests, reference file, rubric, or human-expert deliverable?
3. **Grader** — prefer, in order: deterministic check → executed tests → official grader code → LLM judge.
4. **Interaction** — one prompt/one answer, a scripted multi-turn dialogue, a code-execution loop, or a
   free-form agent with tools and files?
5. **Data source and licence** — where it lives, whether it is gated, the exact revision to pin.

## 2. Pick the runner type

```
Does the model need tools, files on disk, a shell, or the web to do the task?
├─ no  ──► runner: chat     (direct API call; optional multi-turn / code-exec loop)
└─ yes ──► Is there an official harness that already drives the model (and you want its exact loop)?
           ├─ no  ──► runner: oc       (opencode agent in a sandboxed per-sample workspace)
           └─ yes ──► runner: native   (run the official harness per sample from execute())
```

| runner | You implement | The suite provides |
|---|---|---|
| `chat` | `build()` → `ChatRequest` | model client (OpenAI-compatible / Anthropic / opencode-plain), retries, streaming, token accounting, `followups` (scripted turns), `observe` (exec loops), `tools` + `tool_handler` |
| `oc` | `build()` → `AgentTask` | fresh workspace per sample with your files copied in, opencode with a `bench-agent` (no questions, auto-approve), macOS `sandbox-exec` (writes only to the workspace; secrets unreadable; localhost-only network for local models), keyless `bench_web_search`/`bench_fetch_url` MCP tools when `network: true`, transcripts in the dashboard, the suite venv as the agent's `python3` |
| `native` | `execute()` → `ModelOutput` | `ctx.model`, `ctx.base_url` (where the model is served), `ctx.judge`, `ctx.run_dir`, `bench.sandbox.run()` |

A `chat` benchmark may also override `execute()` for a custom multi-call loop that still calls
`ctx.client` (see `benchmarks/scicode/adapter.py`); it keeps the "direct" harness label.

## 3. Scaffold

```bash
uv run bench new my_bench --runner chat --category Knowledge --title "My Bench"
```

Categories: `Coding, Math, Instruction following, Knowledge, Long context, Tool use, Search agent,
General agent, Spreadsheet, Presentation, Office docs, Doc / vision, Text-to-SQL`
(add a new one to `CATEGORIES` in `bench/benchmark.py` only if none fits).

### `benchmark.yaml` reference

| Field | Default | Meaning |
|---|---|---|
| `slug` | — | must equal the folder name |
| `title`, `category`, `description` | — | shown in the dashboard |
| `runner` | `chat` | `chat` \| `oc` \| `native` |
| `metric` | `accuracy` | label for the primary score |
| `score_range` | `[0, 1]` | per-sample score range (normalised for category averages) |
| `judge` | `false` | some samples need the cloud judge (they stay `judge_pending` until one is configured) |
| `repeats` | `1` | generations per sample; the score averages them (avg@k) |
| `tiers` | from `config.yaml` | per-tier sample counts, e.g. `{smoke: 5, quick: null, standard: null, full: null}` for a 30-item set |
| `stratify` | `true` | stratify subsets by `Sample.stratum` |
| `max_tokens` | `config.defaults.max_tokens` | output cap (a model's `max_output_tokens` can lower it) |
| `sample_timeout_s` | `config.defaults.agent_timeout_s` | wall clock for oc/native samples |
| `requires.vision` / `min_context` / `tool_calling` / `network` / `judge_runtime` | false/0 | incompatible models are skipped and shown as n/a |
| `badges` | `[]` | `adapted` (harness differs from official), `substitute`, `partial`, `vision`, `judge` |
| `options` | `{}` | free-form settings, read as `self.spec.options` |
| `oracle_min` / `null_max` | `0.95` / `0.05` | `bench validate` thresholds (lower them only with a documented reason) |
| `sources` | `[]` | `{url, revision, note}` — pin revisions |
| `gated` | `false` | HF terms must be accepted first |
| `notes` | — | caveats shown on the benchmark page (be honest about deviations) |

## 4. Implement the adapter

The contract (from `bench/benchmark.py`):

```python
class MyBench(Benchmark):
    def fetch(self) -> None                       # download to self.data_dir; call self.mark_fetched()
    def load(self) -> list[Sample]                # Sample(id: str, data: dict, gold, stratum)
    def build(self, sample, ctx) -> ChatRequest | AgentTask        # chat / oc
    def execute(self, sample, ctx) -> ModelOutput                  # native (or custom chat loop)
    def grade(self, sample, output, ctx) -> Grade                  # Grade(score, extracted, details, judge)
    def oracle(self, sample) -> str | None        # a response that must score ~1.0 (or override oracle_output)
    def aggregate(self, rows) -> dict             # optional breakdowns, e.g. {"by_category": {...}}
```

`RunContext` gives you `ctx.model` (the `ModelSpec`), `ctx.client` (chat client), `ctx.base_url`,
`ctx.run_dir` (per-run directory for files), `ctx.judge`, `ctx.require_judge()`, `ctx.tier`, `ctx.log()`.

### Helpers you should reuse (`bench/common.py`, `bench/families.py`, `bench/office.py`, `bench/sandbox/`)

| Need | Use |
|---|---|
| HF datasets / files | `hf_rows(repo, dest, config=, split=)`, `hf_file(...)`, `hf_snapshot(..., allow=[...])` |
| other downloads | `download(url, dest)`, `gdrive_download(id, dest)`, `safe_extract(archive, dest)` |
| upstream code at a pinned commit | `vendor_repo(url, sha, name, sparse=[...])`, `vendor_raw_files(repo, sha, paths, dest)`, `load_module(name, path)` |
| pickled upstream data | `safe_unpickle(bytes)` (refuses any class/function) |
| answer extraction | `extract_boxed`, `extract_mc_letter`, `short_answer`, `parse_last_json` (`bench/judge.py`) |
| math grading | `math_equal(gold, response)` (math-verify) |
| doc/chart QA metrics | `anls`, `relaxed_match` |
| long contexts | `est_tokens`, `truncate_middle`, `context_budget(ctx, max_tokens)` |
| ready-made shapes | `MathBenchmark`, `MultipleChoiceBenchmark` (`bench/families.py`) |
| run model-written code | `run_tests(code, tests, mode="stdio"|"call")`, `run_python(code, cwd=)`, `sql_match(db, pred, gold)` — all sandboxed |
| arbitrary sandboxed command | `bench.sandbox.run(cmd, cwd=, timeout=, network="none"|"localhost"|"all", write_paths=[...])` |
| office files | `document_text(path)`, `workbook_text`, `recalc_if_needed` (LibreOffice), `render_slides`, `xlsx_drop_sheet` |
| judge | `ctx.require_judge().equivalence(question, response, gold)`, `.score_json(prompt, kind, system=)` |
| web search (chat loops) | `bench.tools.search.TOOL_SCHEMAS` + `tool_handler` |

### Rules

- **Data is untrusted.** Keep it under `self.data_dir`; never import or execute files from it.
  Extract archives with `safe_extract`. Never `eval`/`exec`/`pickle.load` dataset contents — and
  never call upstream helpers that do (some parse model output with `eval`; parse safely instead).
- **Model output is untrusted.** Anything that executes it (tests, SQL, scripts, agents) goes through
  the sandbox with the narrowest network setting that works.
- **Pin everything** (dataset revision, upstream commit). Put the pin in `sources` too.
- **Stable sample ids** — they key frozen subsets and stored results; never derive them from row order
  if the upstream order can change (prefer the dataset's own id).
- **Prefer official prompts and graders**; when you must deviate, add the `adapted` badge and explain in `notes`.
- **Graders must not raise for a bad answer** — return `Grade(score=0.0, details={...why...})`.
  Raise only for infrastructure problems. Raise `JudgeUnavailable` (via `ctx.require_judge()`) to mark a
  sample judge-pending.
- Put everything a reviewer needs to understand a score into `Grade.details` / `Grade.extracted`
  (the dashboard shows both).

## 5. Fetch and freeze subsets

```bash
uv run bench fetch my_bench
uv run bench subsets my_bench          # writes benchmarks/my_bench/subsets/{smoke,quick,standard,full}.json
```

Subsets are seeded, stratified by `Sample.stratum`, and nested (smoke ⊂ quick ⊂ standard ⊂ full).
Commit the subset files: every model must see identical items. If the dataset changes, delete the
subset files deliberately and redraw.

## 6. Add a judge prompt (only if needed)

Use the generic `equivalence()` judge for "is this final answer equivalent to the gold answer". For a
custom rubric, keep the prompt text in your adapter (verbatim from upstream when one exists), call
`judge.score_json(prompt, kind, system=...)`, and if you introduce a new kind, add it to
`PROMPT_VERSIONS` in `bench/judge.py`. **Bump the version whenever the prompt text changes** — the
version is stored with every score so old and new judgements are never silently mixed.

## 7. Self-test

```bash
uv run bench validate my_bench --tier quick
```

- `oracle` must score ≥ `oracle_min` (gold answers pass the grader).
- `null` (empty response) must score ≤ `null_max` (the grader can fail).
- Without a usable oracle (code generation, open-ended tasks), add a manual positive test: one known-good
  and one known-bad output through `grade()` (see how SciCode and the IF benchmarks were checked).

Then run a real smoke pass against a model and look at it in the dashboard:

```bash
uv run bench run -m <model> -b my_bench -t smoke
uv run bench serve        # open http://127.0.0.1:8765 → Leaderboard → the cell → inspect samples
```

## 8. Checklist before you call it done

- [ ] `benchmark.yaml`: honest `description`, pinned `sources`, correct `requires`, badges + `notes` for any deviation
- [ ] `fetch()` is idempotent and ends with `mark_fetched()`; gated data sets `gated: true`
- [ ] `load()` ids are stable; `stratum` set when the data has meaningful groups
- [ ] Untrusted data never executed; model output only executed in the sandbox
- [ ] `bench validate` passes (or the exception is documented via `oracle_min`/`null_max` + `notes`)
- [ ] smoke run on a real model completes; samples look right in the run explorer
- [ ] subset files committed; `docs/BENCHMARKS.md` updated
