"""`bench` — the master script.

  bench list | models | doctor
  bench fetch <benchmarks>            download data
  bench subsets <benchmarks>          freeze seeded subsets for every tier
  bench run -m <models> -b <benchmarks> -t <tier>   queue jobs and run the worker
  bench worker [--watch]              drain the queue (daemon with --watch)
  bench status | results              queue and leaderboard in the terminal
  bench regrade --pending | --run ID  re-grade (e.g. after configuring the judge)
  bench validate <benchmarks>         oracle ≈100% / null ≈0% self-test
  bench serve                         live dashboard
  bench new <slug>                    scaffold a new benchmark
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bench.config import ROOT, load_config, load_models, resolve_secret

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
con = Console()

for noisy in ("httpx", "mcp", "uvicorn.access", "datasets", "huggingface_hub", "ddgs", "primp", "trafilatura"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------------------------
@app.command("list")
def list_benchmarks(category: str = typer.Option(None, "--category", "-c")):
    """List benchmarks with category, runner, badges and data status."""
    from bench.registry import get_benchmark, load_specs

    t = Table(title="Benchmarks")
    for col in ("slug", "title", "category", "runner", "metric", "badges", "data"):
        t.add_column(col)
    for slug, sp in sorted(load_specs().items(), key=lambda kv: (kv[1].category, kv[0])):
        if category and sp.category.lower() != category.lower():
            continue
        b = get_benchmark(slug)
        data = "[green]fetched[/]" if b.is_fetched() else ("[yellow]gated[/]" if sp.gated else "-")
        badges = ",".join(sp.badges + (["judge"] if sp.judge and "judge" not in sp.badges else []))
        t.add_row(slug, sp.title, sp.category, sp.runner, sp.metric, badges, data)
    con.print(t)


@app.command()
def models():
    """List models from models.yaml and whether they are reachable."""
    from bench.models.server import health

    t = Table(title="Models")
    for col in ("key", "kind", "provider", "endpoint / launcher", "ctx", "vision", "conc", "status"):
        t.add_column(col)
    for key, m in load_models().items():
        if m.kind == "mock":
            status = "built-in"
        elif m.kind == "external":
            status = "[green]up[/]" if m.base_url and health(m.base_url, resolve_secret(m.api_key)) else "[red]down[/]"
        elif m.kind == "managed":
            status = "managed"
        elif m.provider == "opencode":
            status = "via opencode" + (" · key set" if os.environ.get("OPENCODE_API_KEY") else " · no OPENCODE_API_KEY")
        else:
            status = "key set" if resolve_secret(m.api_key) else ("opencode" if m.opencode_model else "[red]no key[/]")
        where = m.base_url or (f"{m.launcher}:{m.model}" if m.launcher else (m.model_id or m.opencode_model or ""))
        t.add_row(key, m.kind, m.provider, where, str(m.context), "yes" if m.vision else "", str(m.max_concurrency), status)
    con.print(t)


@app.command()
def fetch(benchmarks: str = typer.Argument("all"), force: bool = typer.Option(False, "--force")):
    """Download benchmark data into data/<slug>/."""
    from bench.registry import get_benchmark, resolve_benchmarks

    for slug in resolve_benchmarks(benchmarks):
        b = get_benchmark(slug)
        if b.is_fetched() and not force:
            con.print(f"[dim]{slug}: already fetched[/]")
            continue
        con.print(f"[bold]{slug}[/]: fetching…")
        try:
            b.fetch()
            n = len(b.load())
            con.print(f"  [green]ok[/] ({n} samples)")
        except Exception as e:
            con.print(f"  [red]failed[/]: {type(e).__name__}: {e}")


@app.command()
def subsets(benchmarks: str = typer.Argument("all"), tiers: str = typer.Option("all", "--tiers")):
    """Freeze seeded, stratified subsets (benchmarks/<slug>/subsets/<tier>.json)."""
    from bench.registry import get_benchmark, resolve_benchmarks
    from bench.sampling import get_subset

    cfg = load_config()
    tier_list = list(cfg.tiers) if tiers == "all" else _split(tiers)
    for slug in resolve_benchmarks(benchmarks):
        b = get_benchmark(slug)
        if not b.is_fetched():
            con.print(f"[yellow]{slug}: not fetched, skipping[/]")
            continue
        samples = b.load()
        sizes = []
        for tier in tier_list:
            sizes.append(f"{tier}={len(get_subset(slug, tier, samples))}")
        con.print(f"{slug}: " + " ".join(sizes))


# ---------------------------------------------------------------------------------------------
@app.command()
def run(
    models_: str = typer.Option(..., "--models", "-m", help="comma list of models.yaml keys"),
    benchmarks: str = typer.Option("all", "--benchmarks", "-b", help="'all', slugs, or category names"),
    tier: str = typer.Option(None, "--tier", "-t"),
    new: bool = typer.Option(False, "--new", help="start fresh instead of resuming an unfinished run"),
    rerun: bool = typer.Option(False, "--rerun", help="queue even if a finished run exists"),
    no_wait: bool = typer.Option(False, "--no-wait", help="only queue; don't start a worker"),
    force: bool = typer.Option(False, "--force", help="skip the free-memory check for managed models"),
    priority: int = typer.Option(0, "--priority"),
):
    """Queue (model × benchmark × tier) jobs, then run the worker in the foreground."""
    from bench.registry import load_specs, resolve_benchmarks
    from bench.store import Store

    cfg = load_config()
    tier = tier or cfg.default_tier
    if tier not in cfg.tiers:
        raise typer.BadParameter(f"unknown tier {tier}")
    known = load_models()
    model_keys = _split(models_)
    for m in model_keys:
        if m not in known:
            raise typer.BadParameter(f"unknown model {m}")
    store = Store()
    specs = load_specs()
    ids = []
    for m in model_keys:
        for slug in resolve_benchmarks(benchmarks):
            existing = store.find_resumable(m, slug, tier)
            if existing and not new:
                if existing["status"] != "queued":
                    store.update_run(existing["id"], status="queued")
                ids.append(existing["id"])
                continue
            if not rerun and not new:
                done = store.q1("SELECT id FROM runs WHERE model=? AND benchmark=? AND tier=? AND status='done' "
                                "ORDER BY finished_at DESC LIMIT 1", (m, slug, tier))
                if done:
                    con.print(f"[dim]{m} × {slug}/{tier}: already done ({done['id']}); --rerun to repeat[/]")
                    continue
            ids.append(store.create_run(m, slug, tier, priority=priority, badges=specs[slug].badges))
    con.print(f"queued {len(ids)} job(s)")
    if no_wait or not ids:
        return
    _run_worker(only=set(ids), force=force)


@app.command()
def worker(watch: bool = typer.Option(False, "--watch", help="keep polling for new jobs"),
           force: bool = typer.Option(False, "--force")):
    """Drain the job queue (all queued runs)."""
    while True:
        _run_worker(only=None, force=force)
        if not watch:
            return
        time.sleep(5)


def _run_worker(only: set[str] | None, force: bool) -> None:
    from bench.worker import Worker, WorkerLock

    lock = WorkerLock()
    if not lock.acquire():
        con.print(f"[yellow]a worker is already running (pid {lock.holder()}); your jobs are queued and it will pick "
                  f"them up after its current work (watch with `bench status` or the dashboard)[/]")
        return
    w = Worker(log=lambda m: con.print(m, markup=False, highlight=False), force_memory=force)
    w.install_signals()
    w.run_queue(only=None)  # drain everything queued, including jobs added by other `bench run` calls meanwhile
    _print_results(only)


# ---------------------------------------------------------------------------------------------
@app.command()
def status(limit: int = typer.Option(30, "--limit", "-n")):
    """Show queued/running/recent runs."""
    from bench.store import Store

    store = Store()
    ws = store.worker_state()
    if ws and time.time() - (ws["state"].get("heartbeat") or 0) < 30:
        st = ws["state"]
        con.print(f"[bold]worker[/] pid {ws['pid']}: {st.get('phase')} "
                  f"{st.get('model', '')} {st.get('benchmark', '')} · {st.get('tokens_per_s_60s', 0):.1f} tok/s")
    else:
        con.print("[dim]no worker running[/]")
    rows = store.q("SELECT * FROM runs ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?", (limit,))
    t = Table()
    for col in ("id", "model", "benchmark", "tier", "status", "progress", "score", "note"):
        t.add_column(col)
    for r in rows:
        prog = f"{r['n_graded']}/{r['n_total']}" + (f" ({r['n_error']} err)" if r["n_error"] else "")
        sc = f"{r['score']:.3f}" if r["score"] is not None else ""
        t.add_row(r["id"], r["model"], r["benchmark"], r["tier"], r["status"], prog, sc, (r["error"] or "")[:60])
    con.print(t)


@app.command()
def results(tier: str = typer.Option(None, "--tier", "-t")):
    """Leaderboard in the terminal (latest finished run per model × benchmark)."""
    _print_results(None, tier)


def _print_results(only: set[str] | None, tier: str | None = None) -> None:
    from bench.store import Store

    store = Store()
    if only:
        ph = ",".join("?" * len(only))
        rows = store.q(f"SELECT * FROM runs WHERE id IN ({ph})", list(only))
    else:
        rows = store.q("SELECT * FROM runs WHERE status='done'" + (" AND tier=?" if tier else "") +
                       " ORDER BY finished_at", [tier] if tier else [])
    if not rows:
        return
    latest = {}
    for r in rows:
        latest[(r["model"], r["benchmark"], r["tier"])] = r
    t = Table(title="Results")
    for col in ("model", "benchmark", "tier", "status", "score", "95% CI", "n", "errors", "pending"):
        t.add_column(col)
    for (m, b, tr), r in sorted(latest.items()):
        sc = f"{r['score']:.3f}" if r["score"] is not None else "-"
        ci = f"{r['ci_low']:.3f}–{r['ci_high']:.3f}" if r["ci_low"] is not None else ""
        t.add_row(m, b, tr, r["status"], sc, ci, str(r["n_graded"]), str(r["n_error"]), str(r["n_pending"]))
    con.print(t)


@app.command()
def regrade(
    run_id: str = typer.Option(None, "--run"),
    pending: bool = typer.Option(False, "--pending", help="all judge_pending samples"),
    errors: bool = typer.Option(False, "--errors", help="also samples whose grading failed"),
    benchmark: str = typer.Option(None, "--benchmark", "-b"),
):
    """Re-grade stored outputs (no regeneration)."""
    from bench.benchmark import RunContext
    from bench.judge import Judge
    from bench.registry import get_benchmark
    from bench.store import Store
    from bench.worker import finalize_run, output_from_row

    store = Store()
    cfg = load_config()
    where, args = [], []
    if run_id:
        where.append("s.run_id=?")
        args.append(run_id)
    if benchmark:
        where.append("r.benchmark=?")
        args.append(benchmark)
    statuses = ["judge_pending"] if pending else ["graded", "judge_pending"]
    if errors:
        statuses.append("error")
    where.append(f"s.status IN ({','.join('?' * len(statuses))})")
    args += statuses
    if errors:
        where.append("(s.status != 'error' OR s.output IS NOT NULL)")
    rows = store.q(f"SELECT s.*, r.benchmark, r.model, r.tier FROM samples s JOIN runs r ON r.id=s.run_id "
                   f"WHERE {' AND '.join(where)}", args)
    con.print(f"regrading {len(rows)} sample(s)")
    judge = Judge()
    models_ = load_models()
    touched: dict[str, str] = {}
    loaded: dict[str, dict] = {}
    for row in rows:
        b = get_benchmark(row["benchmark"])
        if row["benchmark"] not in loaded:
            loaded[row["benchmark"]] = {s.id: s for s in b.load()}
        s = loaded[row["benchmark"]].get(row["sample_id"])
        if not s:
            continue
        ctx = RunContext(model=models_.get(row["model"]) or models_["null"], client=None,
                         run_dir=cfg.results_dir / "runs" / row["run_id"], judge=judge, tier=row["tier"])
        from bench.benchmark import Grade, JudgeUnavailable

        try:
            g = b.grade(s, output_from_row(row), ctx)
        except JudgeUnavailable as e:
            g = Grade(score=None, pending=True, details={"pending": str(e)})
        status_ = "judge_pending" if (g.pending or g.score is None) else "graded"
        store.grade_sample(row["run_id"], s.id, row["rep"], status=status_, score=g.score, extracted=g.extracted,
                           gold=s.gold, details=g.details, judge=g.judge)
        touched[row["run_id"]] = row["benchmark"]
    for rid, slug in touched.items():
        finalize_run(store, rid, get_benchmark(slug))
    _print_results(set(touched) or None)


@app.command()
def cancel(run_id: str = typer.Argument(None), all_queued: bool = typer.Option(False, "--all-queued")):
    """Remove queued jobs (finished samples are kept)."""
    from bench.store import Store

    store = Store()
    if all_queued:
        n = len(store.queued_runs())
        store.q("UPDATE runs SET status='paused' WHERE status='queued'")
        con.print(f"paused {n} queued job(s)")
    elif run_id:
        store.update_run(run_id, status="paused")
        con.print(f"paused {run_id}")


# ---------------------------------------------------------------------------------------------
@app.command()
def validate(benchmarks: str = typer.Argument("all"), tier: str = typer.Option("smoke", "--tier", "-t")):
    """Grader self-test: the oracle model should score ≈1.0 and the null model ≈0.0."""
    from bench.benchmark import JudgeUnavailable, RunContext
    from bench.judge import Judge
    from bench.registry import get_benchmark, resolve_benchmarks
    from bench.sampling import get_subset

    cfg = load_config()
    judge = Judge()
    models_ = load_models()
    t = Table(title=f"validate ({tier})")
    for col in ("benchmark", "n", "oracle", "null", "verdict"):
        t.add_column(col)
    for slug in resolve_benchmarks(benchmarks):
        b = get_benchmark(slug)
        try:
            if not b.is_fetched():
                b.fetch()
            samples = b.load()
            ids = get_subset(slug, tier, samples)
        except Exception as e:
            t.add_row(slug, "-", "-", "-", f"[red]data: {str(e)[:60]}[/]")
            continue
        by_id = {s.id: s for s in samples}
        scores = {}
        errors = {"oracle": 0, "null": 0}
        first_err = ""
        needs_judge = False
        for mk in ("oracle", "null"):
            if needs_judge:
                break
            ctx = RunContext(model=models_[mk], client=None, run_dir=cfg.results_dir / "validate" / slug / mk,
                             judge=judge, tier=tier)
            ctx.run_dir.mkdir(parents=True, exist_ok=True)
            vals = []
            for sid in ids:
                s = by_id[sid]
                from bench.benchmark import ModelOutput

                out = (b.oracle_output(s, ctx) if mk == "oracle" else None)
                if mk == "oracle" and out is None:
                    continue
                out = out or ModelOutput(text="")
                try:
                    g = b.grade(s, out, ctx)
                    if g.score is not None:
                        vals.append(g.score)
                except JudgeUnavailable:
                    needs_judge = True
                    break
                except Exception as e:
                    errors[mk] += 1
                    first_err = first_err or f"{type(e).__name__}: {e}"[:80]
            scores[mk] = (sum(vals) / len(vals)) if vals else None
        if needs_judge:
            t.add_row(slug, str(len(ids)), "-", "-", "[yellow]needs judge[/] [dim](set judge.model)[/]")
            continue
        o, n = scores["oracle"], scores["null"]
        ok_o = o is None or o >= b.spec.oracle_min
        ok_n = n is None or n <= b.spec.null_max
        verdict = "[green]ok[/]" if ok_o and ok_n else "[red]CHECK[/]"
        if o is None:
            verdict += " [dim](no oracle)[/]"
        fmt = lambda v, k: ("-" if v is None else f"{v:.3f}") + (f" ({errors[k]} err)" if errors[k] else "")
        if errors["oracle"] or errors["null"]:
            verdict = "[red]ERRORS[/] " + first_err.replace("[", "(")
        t.add_row(slug, str(len(ids)), fmt(o, "oracle"), fmt(n, "null"), verdict)
    con.print(t)


# ---------------------------------------------------------------------------------------------
@app.command()
def serve(port: int = typer.Option(None, "--port", "-p"), dev: bool = typer.Option(False, "--dev"),
          build: bool = typer.Option(False, "--build", help="(re)build the dashboard first")):
    """Run the live dashboard (http://127.0.0.1:8765)."""
    import uvicorn

    cfg = load_config()
    static = ROOT / "bench" / "server" / "static"
    if build or (not dev and not (static / "index.html").exists()):
        _build_web()
    host, port = cfg.server.host, port or cfg.server.port
    con.print(f"dashboard: http://{host}:{port}")
    if dev:
        con.print("dev mode: run `npm run dev` in web/ (proxies /api to this server)")
    uvicorn.run("bench.server.app:app", host=host, port=port, log_level="warning")


def _build_web() -> None:
    web = ROOT / "web"
    npm = shutil.which("npm")
    if not npm:
        raise typer.Exit("npm not found; install Node to build the dashboard")
    if not (web / "node_modules").exists():
        subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=web, check=True)
    subprocess.run([npm, "run", "build"], cwd=web, check=True)


@app.command()
def doctor():
    """Check prerequisites: tools, credentials, gated datasets, disk."""
    from bench.registry import load_specs

    def row(name, ok, note=""):
        t.add_row(name, "[green]ok[/]" if ok else "[red]missing[/]", note)

    t = Table(title="doctor")
    for col in ("check", "status", "note"):
        t.add_column(col)
    row("sandbox-exec", bool(shutil.which("sandbox-exec")), "Docker-free isolation for code & agents")
    row("opencode", bool(shutil.which("opencode")), "agentic runner")
    lo = shutil.which("soffice") or Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").exists()
    row("LibreOffice", bool(lo), "spreadsheet recalculation, PPTX rendering (brew install --cask libreoffice)")
    row("node/npm", bool(shutil.which("npm")), "builds the dashboard")
    try:
        from huggingface_hub import whoami

        who = whoami().get("name")
        row("Hugging Face login", True, f"as {who}")
    except Exception:
        row("Hugging Face login", False, "`hf auth login`, then accept terms for gated datasets")
    gated = [s for s, sp in load_specs().items() if sp.gated]
    t.add_row("gated datasets", "[yellow]info[/]", ", ".join(gated))
    cfg = load_config()
    jm = cfg.judge.model
    jspec = load_models().get(jm) if jm else None
    judge_ok = bool(jspec and (resolve_secret(jspec.api_key) or jspec.provider == "opencode"))
    row("judge", judge_ok, f"config.yaml judge.model = {jm}" if jm else "set judge.model (e.g. claude-opus-5-5)")
    free = shutil.disk_usage(ROOT).free / 1e9
    row("disk", free > 60, f"{free:.0f} GB free (all datasets + harnesses take ~35 GB; keep ~60 GB)")
    con.print(t)


EXCLUDED = [
    ("SWE-bench Verified, SWE-bench Pro, Terminal-Bench v2.1", "Need Docker (excluded by choice for this suite)"),
    ("LCB-Pro 25Q2 (Easy/Medium)", "Judge server (go-judge/LightCPVerifier) needs Docker + Linux cgroups"),
    ("OJBench", "DMOJ judge is Linux-only"),
    ("WildClawBench, QwenClawBench", "One Docker container per task"),
    ("OdysseyBench / OfficeBench", "OfficeBench Docker testbed"),
    ("OSWorld, WindowsAgentArena", "Full x86 / Windows virtual machines"),
    ("TSBench (Talk-to-Your-Slides)", "Drives PowerPoint through Windows COM"),
    ("SheetQA (google/sheetqa)", "Not publicly available (Hugging Face returns 401)"),
    ('"Search Agent", "General Agent"', "Category headings, not benchmarks (their members are included)"),
]


@app.command()
def catalog(out: Path = typer.Option(ROOT / "docs" / "BENCHMARKS.md", "--out")):
    """Regenerate docs/BENCHMARKS.md from the benchmark.yaml files."""
    from bench.benchmark import CATEGORIES
    from bench.registry import load_specs

    specs = load_specs()
    lines = ["# Benchmark catalog", "",
             "Generated by `bench catalog` from each `benchmarks/<slug>/benchmark.yaml` — edit those, not this file.", "",
             "Runner: **chat** = direct API call · **oc** = opencode agent in a sandboxed workspace · **native** = official harness.",
             "Badges: **adapted** = harness differs from the official one · **substitute** = stand-in for an unavailable "
             "benchmark · **partial** = some subsets not run · **judge** = graded (partly) by the cloud judge.", "",
             f"## Included ({len(specs)})", ""]
    for cat in CATEGORIES:
        rows = sorted((sp for sp in specs.values() if sp.category == cat), key=lambda sp: sp.title)
        if not rows:
            continue
        lines += [f"### {cat}", "", "| Benchmark | Runner | Metric | Needs | Badges | Source |", "|---|---|---|---|---|---|"]
        for sp in rows:
            needs = [n for n, on in (("vision", sp.requires.vision), ("tools", sp.requires.tool_calling),
                                     ("web", sp.requires.network), ("gated data", sp.gated)) if on]
            if sp.requires.min_context:
                needs.append(f"≥{sp.requires.min_context // 1000}k ctx")
            badges = list(sp.badges) + (["judge"] if sp.judge and "judge" not in sp.badges else [])
            src = sp.sources[0].url if sp.sources else ""
            lines.append(f"| **{sp.title}** (`{sp.slug}`) | {sp.runner} | {sp.metric} | {', '.join(needs)} | "
                         f"{', '.join(badges)} | {src} |")
        lines.append("")
        for sp in rows:
            if sp.notes:
                lines.append(f"- *{sp.title}* — {sp.notes.strip()}")
        if any(sp.notes for sp in rows):
            lines.append("")
    lines += ["## Excluded", "", "| Benchmark | Why |", "|---|---|"] + [f"| {n} | {why} |" for n, why in EXCLUDED]
    out.write_text("\n".join(lines) + "\n")
    con.print(f"wrote {out} ({len(specs)} benchmarks)")


@app.command()
def new(slug: str, runner: str = typer.Option("chat", "--runner", "-r", help="chat | oc | native"),
        category: str = typer.Option("Knowledge", "--category", "-c"), title: str = typer.Option(None, "--title")):
    """Scaffold benchmarks/<slug>/ from the template (see docs/NEW_BENCHMARK.md)."""
    from bench.benchmark import CATEGORIES

    if category not in CATEGORIES:
        raise typer.BadParameter(f"category must be one of: {', '.join(CATEGORIES)}")
    if runner not in ("chat", "oc", "native"):
        raise typer.BadParameter("runner must be chat, oc or native")
    dest = ROOT / "benchmarks" / slug
    if dest.exists():
        raise typer.BadParameter(f"{dest} already exists")
    tpl = ROOT / "benchmarks" / "_template"
    shutil.copytree(tpl, dest, ignore=shutil.ignore_patterns("__pycache__"))
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".yaml", ".md"):
            cls = "".join(w.capitalize() for w in slug.replace("-", "_").split("_")) or "NewBenchmark"
            f.write_text(f.read_text().replace("__SLUG__", slug).replace("__TITLE__", title or slug)
                         .replace("__CATEGORY__", category).replace("__RUNNER__", runner).replace("__CLASS__", cls))
    con.print(f"created {dest.relative_to(ROOT)} — next: implement adapter.py, then `bench validate {slug}`")


if __name__ == "__main__":
    app()
