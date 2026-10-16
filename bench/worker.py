"""The orchestrator: drains the run queue, grouped by model.

For each model: bring its server up (managed) or check it (external) -> run each queued benchmark
(generation at the model's max_concurrency) -> stop the server. Grading happens on a separate
pool, so judge calls (cloud) overlap with the next model's generation. Every sample is persisted
as soon as it finishes, so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import threading
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from bench import metrics
from bench.benchmark import AgentTask, Benchmark, ChatRequest, Grade, JudgeUnavailable, ModelOutput, RunContext, Sample
from bench.config import ModelSpec, load_config, load_models
from bench.judge import Judge
from bench.models.client import make_client
from bench.models.server import ServerError, ServerManager
from bench.registry import get_benchmark, load_specs
from bench.sampling import get_subset, subset_hash
from bench.store import Store

RUNNER_ORDER = {"chat": 0, "native": 1, "oc": 2}


def compat_problem(spec: ModelSpec, bench: Benchmark, judge_available: bool = True) -> str | None:
    req = bench.spec.requires
    if spec.kind == "mock":
        return None
    if bench.spec.runner == "native" and spec.provider == "opencode":
        return "official harness needs a direct API endpoint; this model is only reachable through opencode"
    if req.judge_runtime and not judge_available:
        return "needs the judge model at run time (e.g. as user simulator) — set judge.model in config.yaml"
    if req.vision and not spec.vision:
        return "needs a vision model"
    if req.min_context and spec.context < req.min_context:
        return f"needs >= {req.min_context} context (model has {spec.context})"
    if req.tool_calling and not spec.tool_calling:
        return "needs tool calling"
    if bench.spec.runner == "oc" and not (spec.opencode_model or spec.is_local):
        return "agentic benchmark needs opencode_model for cloud models"
    return None


def truncate_prompt(messages: list[dict], limit: int = 12000) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [p if p.get("type") == "text" else {"type": p.get("type"), "note": "[binary omitted]"} for p in c]
            c = [{**p, "text": _cut(p["text"], limit)} if p.get("type") == "text" else p for p in c]
        elif isinstance(c, str):
            c = _cut(c, limit)
        out.append({**m, "content": c})
    return out


def _cut(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = int(limit * 0.7)
    return s[:head] + f"\n\n[... {len(s) - limit} chars omitted ...]\n\n" + s[-(limit - head):]


class Worker:
    def __init__(self, store: Store | None = None, log: Callable[[str], None] = print, *, force_memory: bool = False):
        self.cfg = load_config()
        self.store = store or Store()
        self._log = log
        self.models = load_models()
        self.servers = ServerManager()
        self.judge = Judge()
        self.grade_pool = ThreadPoolExecutor(self.cfg.defaults.grade_workers, thread_name_prefix="grade")
        self.stop_event = threading.Event()
        self.force_memory = force_memory
        self._pending: dict[str, int] = {}
        self._gen_done: set[str] = set()
        self._plock = threading.Lock()
        self._recent: deque = deque(maxlen=40)
        self._tok_window: deque = deque(maxlen=2000)
        self._stream_samples: deque = deque()
        # live state per model being processed (thread-local key), so parallel models don't clobber each other
        self._tl = threading.local()
        self._states: dict[str, dict] = {"_": {"phase": "idle"}}
        self._last_state_write = 0.0
        self._mcp_url: str | None = None

    # ---- logging / live state ----------------------------------------------------------------
    @property
    def _state(self) -> dict:
        return self._states.setdefault(getattr(self._tl, "model", None) or "_", {})

    @_state.setter
    def _state(self, value: dict) -> None:
        self._states[getattr(self._tl, "model", None) or "_"] = value

    def _combined_state(self) -> dict:
        active = [dict(v) for k, v in list(self._states.items()) if k != "_"]
        if not active:
            return dict(self._states.get("_", {}))
        return {**active[0], "parallel": active if len(active) > 1 else None}

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self._recent.append(line)
        self._log(msg)
        self._write_state(force=True)

    def _write_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_state_write < 1.0:
            return
        self._last_state_write = now
        from bench.models.client import STREAMED

        self._stream_samples.append((now, STREAMED[0]))
        while len(self._stream_samples) > 1 and self._stream_samples[0][0] < now - 60:
            self._stream_samples.popleft()
        t0, c0 = self._stream_samples[0]
        streamed_rate = (STREAMED[0] - c0) / (now - t0) if now - t0 >= 5 else 0.0
        cutoff = now - 60
        completed_rate = sum(n for t, n in self._tok_window if t >= cutoff) / 60.0
        # live streaming rate when available (local/OpenAI-compatible models), else completed-sample rate
        state = {**self._combined_state(), "recent": list(self._recent), "tokens_per_s_60s": streamed_rate or completed_rate,
                 "pid": os.getpid(), "heartbeat": now, "started_at": getattr(self, "_started_at", now)}
        try:
            self.store.set_worker_state(os.getpid(), state)
        except Exception:
            pass

    # ---- queue ---------------------------------------------------------------------------------
    def _heartbeat(self, done: threading.Event) -> None:
        while not done.wait(3.0):
            self._write_state(force=True)

    def run_queue(self, only: set[str] | None = None, parallel_models: int | None = None) -> None:
        limit = max(1, parallel_models or self.cfg.defaults.parallel_models)
        self._state = {"phase": "starting", "started_at": time.time()}
        self._started_at = time.time()
        hb_done = threading.Event()
        threading.Thread(target=self._heartbeat, args=(hb_done,), daemon=True, name="heartbeat").start()
        active: dict[str, threading.Thread] = {}
        try:
            while not self.stop_event.is_set():
                for key, t in list(active.items()):
                    if not t.is_alive():
                        del active[key]
                queued = [r for r in self.store.queued_runs() if not only or r["id"] in only]
                if not queued and not active:
                    break
                for model_key in dict.fromkeys(r["model"] for r in queued):  # highest-priority model first
                    if model_key in active or len(active) >= limit:
                        continue
                    self.models = {**self.models, **load_models()}  # pick up models.yaml edits without a restart
                    spec = self.models.get(model_key)
                    if spec and spec.kind == "managed" and any(
                            getattr(self.models.get(k), "kind", None) == "managed" for k in active):
                        continue  # one locally launched model in RAM at a time
                    runs = [r for r in queued if r["model"] == model_key]
                    specs = load_specs()
                    runs.sort(key=lambda r: (-r["priority"], RUNNER_ORDER.get(specs[r["benchmark"]].runner, 9)
                                             if r["benchmark"] in specs else 9, r["created_at"]))
                    if limit > 1:
                        self.log(f"[{model_key}] starting ({len(active) + 1} model(s) running in parallel)")
                    t = threading.Thread(target=self._model_thread, args=(model_key, runs), daemon=True,
                                         name=f"model-{model_key}")
                    active[model_key] = t
                    t.start()
                self.stop_event.wait(2.0)
            for t in active.values():  # after Ctrl-C: let in-flight samples of every model finish
                t.join()
            self._state = {"phase": "finishing grading"}
            self.log("waiting for grading to finish")
            self.grade_pool.shutdown(wait=True)
        finally:
            hb_done.set()
            self._state = {"phase": "idle", "stopped": self.stop_event.is_set()}
            self._write_state(force=True)

    def _model_thread(self, model_key: str, runs: list[dict]) -> None:
        self._tl.model = model_key
        try:
            self.process_model(model_key, runs)
        except Exception as e:  # never let one model's crash take the others down silently
            self.log(f"[{model_key}] worker error: {type(e).__name__}: {e}")
        finally:
            self._states.pop(model_key, None)
            self._tl.model = None

    def process_model(self, model_key: str, runs: list[dict]) -> None:
        spec = self.models.get(model_key)
        if spec is None:
            for r in runs:
                self.store.update_run(r["id"], status="failed", error=f"unknown model {model_key}")
            return
        base_url = None
        server_ready = spec.kind in ("cloud", "mock")
        try:
            for run in runs:
                if self.stop_event.is_set():
                    break
                fresh = self.store.get_run(run["id"])
                if not fresh or fresh["status"] != "queued":
                    continue
                try:
                    bench = get_benchmark(run["benchmark"])
                except Exception as e:
                    self.store.update_run(run["id"], status="failed", error=f"cannot load benchmark: {e}")
                    continue
                problem = compat_problem(spec, bench, self.judge.available)
                if problem:
                    self.store.update_run(run["id"], status="skipped", error=problem, finished_at=time.time())
                    self.log(f"[{model_key}] skip {bench.spec.slug}: {problem}")
                    continue
                if not server_ready:
                    self._state = {"phase": "starting model", "model": model_key}
                    self.log(f"[{model_key}] bringing model up")
                    try:
                        base_url = self.servers.ensure_up(spec, force=self.force_memory, log=self.log)
                        server_ready = True
                        if spec.request_overrides and spec.provider == "openai" and base_url:
                            from bench.models.proxy import override_proxy

                            base_url = override_proxy(base_url, spec.request_overrides).url
                            self.log(f"[{model_key}] forcing {spec.request_overrides} on every request via {base_url}")
                    except ServerError as e:
                        self.log(f"[{model_key}] {e}")
                        for r in runs:
                            if (self.store.get_run(r["id"]) or {}).get("status") == "queued":
                                self.store.update_run(r["id"], status="failed", error=str(e))
                        return
                self.process_run(run["id"], bench, spec, base_url)
        finally:
            # Grading only needs the (cloud) judge, so the local model is released right away.
            if spec.kind == "managed" and server_ready:
                self.servers.stop(spec, log=self.log)

    # ---- one run --------------------------------------------------------------------------------
    def process_run(self, rid: str, bench: Benchmark, spec: ModelSpec, base_url: str | None) -> None:
        slug = bench.spec.slug
        run = self.store.get_run(rid)
        tier = run["tier"]
        run_dir = self.cfg.results_dir / "runs" / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        harness = {"chat": "direct" if spec.provider != "opencode" else "opencode-plain",
                   "oc": "opencode", "native": "native"}[bench.spec.runner]
        if spec.kind == "mock":
            harness = "mock"
        self._state = {"phase": "preparing", "model": spec.key, "benchmark": slug, "tier": tier, "run": rid}
        self.store.update_run(rid, status="running", started_at=run["started_at"] or time.time(), harness=harness,
                              badges=bench.spec.badges, error=None)
        try:
            if not bench.is_fetched():
                self.log(f"[{slug}] fetching data")
                bench.fetch()
            samples = bench.load()
        except Exception as e:
            self.store.update_run(rid, status="failed", error=f"data: {type(e).__name__}: {e}")
            self.log(f"[{slug}] data error: {e}")
            return
        by_id = {s.id: s for s in samples}
        try:
            ids = get_subset(slug, tier, samples)
        except Exception as e:
            self.store.update_run(rid, status="failed", error=str(e))
            return
        reps = max(1, bench.spec.repeats)
        n_total = len(ids) * reps
        self.store.update_run(rid, n_total=n_total, subset_hash=subset_hash(ids))
        done = self.store.done_keys(rid)
        client = None
        if spec.kind != "mock" and bench.spec.runner in ("chat", "native"):
            try:
                client = make_client(spec, base_url=base_url)
            except Exception as e:
                if bench.spec.runner == "chat":
                    self.store.update_run(rid, status="failed", error=f"client: {e}")
                    return
        if bench.spec.requires.network:
            from bench.tools.search import start_mcp_server

            with self._plock:
                self._mcp_url = self._mcp_url or start_mcp_server()
        ctx = RunContext(model=spec, client=client, run_dir=run_dir, judge=self.judge, tier=tier,
                         log=lambda m: self.log(f"[{slug}] {m}"), base_url=base_url, mcp_url=self._mcp_url)
        with self._plock:
            self._pending[rid] = 0
            self._gen_done.discard(rid)
        # regrade samples generated earlier but never graded (crash between generate and grade)
        for (sid, rep), status in done.items():
            if status == "generated" and sid in by_id:
                row = self.store.q1("SELECT * FROM samples WHERE run_id=? AND sample_id=? AND rep=?", (rid, sid, rep))
                self._submit_grade(bench, by_id[sid], rep, output_from_row(row), ctx, rid)
        todo = [(sid, rep) for sid in ids for rep in range(reps)
                if done.get((sid, rep)) not in ("generated", "graded", "judge_pending")]
        self.log(f"[{spec.key}] {slug}/{tier}: {len(todo)} to generate ({n_total - len(todo)} already done)")
        self._state.update({"phase": "generating", "todo": len(todo), "n_total": n_total})
        conc = max(1, spec.max_concurrency if spec.kind != "mock" else 8)
        stopped = False
        with ThreadPoolExecutor(conc, thread_name_prefix="gen") as pool:
            inflight: set[Future] = set()
            it = iter(todo)
            while True:
                while len(inflight) < conc and not self.stop_event.is_set():
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    sid, rep = nxt
                    inflight.add(pool.submit(self._generate_one, bench, by_id[sid], rep, ctx, rid))
                if not inflight:
                    break
                finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result()
                self.store.refresh_counts(rid)
                self._write_state()
            stopped = self.stop_event.is_set() and next(it, None) is not None
        if stopped:
            # not added to _gen_done: in-flight grades still land, but nothing finalizes a paused run
            self.store.update_run(rid, status="paused")
            self.log(f"[{slug}] paused (rerun the same command to resume)")
            return
        # Status must be "grading" before generation is published as done; otherwise the last grade
        # thread could finalize ("done") and this thread would then overwrite it with "grading".
        self.store.update_run(rid, status="grading")
        with self._plock:
            self._gen_done.add(rid)
            pending = self._pending.get(rid, 0)
        if pending == 0:
            self.finalize(rid, bench)

    def _generate_one(self, bench: Benchmark, sample: Sample, rep: int, ctx: RunContext, rid: str) -> None:
        key = f"{_safe(sample.id)}__{rep}"
        prompt = None
        try:
            if ctx.model.kind == "mock":
                out = (bench.oracle_output(sample, ctx) if ctx.model.key == "oracle" else None) or ModelOutput(text="")
            elif bench.spec.runner == "native" or type(bench).execute is not Benchmark.execute:
                out = bench.execute(sample, ctx)  # native harness, or a custom multi-call chat loop
            else:
                req = bench.build(sample, ctx)
                if isinstance(req, ChatRequest):
                    from bench.runners.chat import run_chat

                    prompt = truncate_prompt(req.messages)
                    out = run_chat(bench, req, ctx)
                elif isinstance(req, AgentTask):
                    from bench.runners.opencode import run_agent

                    prompt = [{"role": "user", "content": _cut(req.prompt, 12000)}]
                    out = run_agent(bench, req, ctx, sample_key=key, base_url=ctx.base_url, mcp_url=ctx.mcp_url)
                else:
                    raise TypeError(f"build() returned {type(req).__name__}")
        except Exception as e:
            self.store.put_sample(rid, sample.id, rep, status="error", prompt=prompt,
                                  error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            self.log(f"[{bench.spec.slug}] {sample.id}: {type(e).__name__}: {str(e)[:200]}")
            return
        if out.usage.get("output_tokens"):
            self._tok_window.append((time.time(), int(out.usage["output_tokens"])))
        self.store.put_sample(
            rid, sample.id, rep, status="generated", prompt=prompt, output=out.text, reasoning=out.reasoning,
            usage=out.usage, latency=out.latency, finish_reason=out.finish_reason, artifacts=out.artifacts,
            extra={**out.extra, **({"turns": truncate_prompt(out.turns, 6000)} if out.turns else {}),
                   **({"workspace": str(out.workspace)} if out.workspace else {})},
        )
        self._submit_grade(bench, sample, rep, out, ctx, rid)

    # ---- grading --------------------------------------------------------------------------------
    def _submit_grade(self, bench, sample, rep, out, ctx, rid) -> None:
        with self._plock:
            self._pending[rid] = self._pending.get(rid, 0) + 1
        self.grade_pool.submit(self._grade_one, bench, sample, rep, out, ctx, rid)

    def _grade_one(self, bench: Benchmark, sample: Sample, rep: int, out: ModelOutput, ctx: RunContext, rid: str) -> None:
        try:
            try:
                g = bench.grade(sample, out, ctx)
            except JudgeUnavailable as e:
                g = Grade(score=None, pending=True, details={"pending": str(e)})
            status = "judge_pending" if (g.pending or g.score is None) else "graded"
            self.store.grade_sample(rid, sample.id, rep, status=status, score=g.score, extracted=g.extracted,
                                    gold=sample.gold, details=g.details, judge=g.judge)
        except Exception as e:
            self.store.grade_sample(rid, sample.id, rep, status="error",
                                    details={"grading_error": f"{type(e).__name__}: {e}",
                                             "trace": traceback.format_exc()[-1500:]})
            self.log(f"[{bench.spec.slug}] grading {sample.id} failed: {e}")
        finally:
            with self._plock:
                self._pending[rid] -= 1
                ready = self._pending[rid] == 0 and rid in self._gen_done
            if ready:
                self.finalize(rid, bench)

    def finalize(self, rid: str, bench: Benchmark) -> None:
        finalize_run(self.store, rid, bench)
        run = self.store.get_run(rid)
        sc = f"{run['score']:.3f}" if run["score"] is not None else "n/a"
        self.log(f"[{run['model']}] {run['benchmark']}/{run['tier']} done: score={sc} "
                 f"(n={run['n_graded']}, errors={run['n_error']}, judge_pending={run['n_pending']})")

    # ---- signals --------------------------------------------------------------------------------
    def install_signals(self) -> None:
        def handler(signum, frame):
            if self.stop_event.is_set():
                self._log("second interrupt: exiting now")
                os._exit(130)
            self._log("stopping after in-flight samples finish (Ctrl-C again to force)")
            self.stop_event.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def finalize_run(store: Store, rid: str, bench: Benchmark) -> None:
    counts = store.refresh_counts(rid)
    rows = store.q("SELECT sample_id, rep, status, score, details, judge, usage FROM samples WHERE run_id=?", (rid,))
    for r in rows:
        for k in ("details", "judge", "usage"):
            r[k] = json.loads(r[k]) if r[k] else None
    graded = [r for r in rows if r["status"] == "graded"]
    summary = metrics.summarize(graded)
    try:
        extra = bench.aggregate(graded) or {}
    except Exception as e:
        extra = {"aggregate_error": str(e)}
    judge_ids = sorted({(r["judge"] or {}).get("judge_id") for r in graded if r["judge"]} - {None})
    run = store.get_run(rid)
    store.update_run(
        rid, status="done", finished_at=time.time(), score=summary["score"], ci_low=summary["ci_low"],
        ci_high=summary["ci_high"], judge_id=",".join(judge_ids) or None,
        metrics={**extra, "n_scored": summary["n_scored"], "complete": counts["n_graded"] >= (run["n_total"] or 0)},
    )


def output_from_row(row: dict) -> ModelOutput:
    extra = json.loads(row["extra"]) if row.get("extra") else {}
    ws = extra.pop("workspace", None)
    turns = extra.pop("turns", [])
    return ModelOutput(text=row.get("output") or "", reasoning=row.get("reasoning"),
                       usage=json.loads(row["usage"]) if row.get("usage") else {},
                       latency=row.get("latency") or 0.0, finish_reason=row.get("finish_reason"),
                       turns=turns, workspace=Path(ws) if ws else None,
                       artifacts=json.loads(row["artifacts"]) if row.get("artifacts") else {}, extra=extra)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


class WorkerLock:
    def __init__(self):
        self.path = load_config().results_dir / "worker.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = None

    def acquire(self) -> bool:
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return True

    def holder(self) -> str:
        try:
            return self.path.read_text().strip()
        except OSError:
            return "?"
