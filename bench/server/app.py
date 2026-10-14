"""Read-only dashboard API + static frontend. Bound to 127.0.0.1 by `bench serve`."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bench.benchmark import CATEGORIES
from bench.config import ROOT, load_config, load_models
from bench.registry import load_specs
from bench.store import Store

app = FastAPI(title="local-bench", docs_url="/api/docs", openapi_url="/api/openapi.json")
_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _j(v):
    if v is None or not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _run_public(r: dict) -> dict:
    r = dict(r)
    for k in ("metrics", "badges"):
        r[k] = _j(r.get(k))
    return r


@app.get("/api/overview")
def overview():
    cfg = load_config()
    specs = load_specs()
    models = load_models()
    rows = store().q("SELECT * FROM runs WHERE status IN ('done','running','grading','queued','failed','skipped','paused') "
                     "ORDER BY COALESCE(finished_at, updated_at, created_at)")
    # latest finished run per (model, benchmark, tier); in-flight status alongside
    cells: dict[tuple, dict] = {}
    live: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["benchmark"], r["tier"])
        if r["status"] == "done":
            cells[key] = _run_public(r)
        else:
            live[key] = {"run_id": r["id"], "status": r["status"], "n_graded": r["n_graded"], "n_total": r["n_total"],
                         "error": r["error"]}
    out_cells = []
    mock = {k for k, m in models.items() if m.kind == "mock"}
    for key in (set(cells) | set(live)):
        if key[0] in mock:
            continue
        c = cells.get(key) or {"model": key[0], "benchmark": key[1], "tier": key[2]}
        out_cells.append({**c, "live": live.get(key)})
    used_models = {c["model"] for c in out_cells}
    return {
        "categories": CATEGORIES,
        "tiers": list(cfg.tiers),
        "default_tier": cfg.default_tier,
        "judge_model": cfg.judge.model,
        "models": [
            {"key": k, "name": m.name, "kind": m.kind, "provider": m.provider, "context": m.context,
             "vision": m.vision, "notes": m.notes, "has_results": k in used_models}
            for k, m in models.items() if m.kind != "mock"  # oracle/null are grader self-tests, not contenders
        ],
        "benchmarks": [
            {**sp.model_dump(mode="json"), "fetched": (cfg.data_dir / slug / ".fetched").exists()}
            for slug, sp in specs.items()
        ],
        "cells": out_cells,
    }


@app.get("/api/runs")
def runs(model: str | None = None, benchmark: str | None = None, tier: str | None = None,
         status: str | None = None, limit: int = 200):
    where, args = [], []
    for col, val in (("model", model), ("benchmark", benchmark), ("tier", tier), ("status", status)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    sql = "SELECT * FROM runs" + (f" WHERE {' AND '.join(where)}" if where else "") + \
          " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?"
    return [_run_public(r) for r in store().q(sql, [*args, limit])]


@app.get("/api/runs/{rid}")
def run_detail(rid: str):
    r = store().get_run(rid)
    if not r:
        raise HTTPException(404, "run not found")
    spec = load_specs().get(r["benchmark"])
    return {**_run_public(r), "spec": spec.model_dump(mode="json") if spec else None}


SAMPLE_LIST_COLS = "sample_id, rep, status, score, extracted, latency, usage, finish_reason, error, artifacts"


@app.get("/api/runs/{rid}/samples")
def run_samples(rid: str, offset: int = 0, limit: int = Query(100, le=1000), filter: str | None = None,
                q: str | None = None):
    where, args = ["run_id=?"], [rid]
    if filter == "wrong":
        where.append("status='graded' AND score < 1")
    elif filter == "right":
        where.append("status='graded' AND score >= 1")
    elif filter == "error":
        where.append("status='error'")
    elif filter == "pending":
        where.append("status IN ('judge_pending','generated')")
    if q:
        where.append("(sample_id LIKE ? OR output LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    w = " AND ".join(where)
    total = store().q1(f"SELECT COUNT(*) AS n FROM samples WHERE {w}", args)["n"]
    rows = store().q(f"SELECT {SAMPLE_LIST_COLS} FROM samples WHERE {w} ORDER BY sample_id, rep LIMIT ? OFFSET ?",
                     [*args, limit, offset])
    for r in rows:
        r["usage"] = _j(r["usage"])
        r["artifacts"] = _j(r["artifacts"])
        ex = _j(r["extracted"])
        r["extracted"] = (ex[:200] if isinstance(ex, str) else ex)
        if r["error"]:
            r["error"] = r["error"].splitlines()[0][:300]
    return {"total": total, "rows": rows}


@app.get("/api/runs/{rid}/samples/{sid:path}")
def sample_detail(rid: str, sid: str, rep: int = 0):
    r = store().q1("SELECT * FROM samples WHERE run_id=? AND sample_id=? AND rep=?", (rid, sid, rep))
    if not r:
        raise HTTPException(404, "sample not found")
    for k in ("prompt", "extracted", "gold", "details", "judge", "usage", "artifacts", "extra"):
        r[k] = _j(r.get(k))
    return r


def _run_dir(rid: str) -> Path:
    return (load_config().results_dir / "runs" / rid).resolve()


@app.get("/api/runs/{rid}/files/{path:path}")
def run_file(rid: str, path: str):
    base = _run_dir(rid)
    target = (base / path).resolve()
    if not str(target).startswith(str(base) + "/") or not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target)


@app.get("/api/runs/{rid}/tree/{path:path}")
def run_tree(rid: str, path: str):
    base = _run_dir(rid)
    target = (base / path).resolve()
    if not (str(target) == str(base) or str(target).startswith(str(base) + "/")) or not target.is_dir():
        raise HTTPException(404, "directory not found")
    files = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and ".opencode" not in p.parts and len(files) < 500:
            files.append({"path": str(p.relative_to(base)), "size": p.stat().st_size})
    return files


@app.get("/api/runs/{rid}/transcript/{key}")
def transcript(rid: str, key: str):
    f = _run_dir(rid) / "transcripts" / f"{key}.jsonl"
    if not f.is_file():
        raise HTTPException(404, "no transcript")
    events = []
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = e.get("part") or {}
        t = e.get("type")
        if t == "text":
            events.append({"type": "text", "text": p.get("text", "")})
        elif t == "tool_use":
            st = p.get("state") or {}
            out = st.get("output")
            events.append({"type": "tool", "tool": p.get("tool"), "status": st.get("status"),
                           "input": st.get("input"), "output": out[:6000] if isinstance(out, str) else out})
        elif t == "step_finish":
            events.append({"type": "step", "reason": p.get("reason"), "tokens": p.get("tokens")})
        elif t == "error":
            events.append({"type": "error", "error": e.get("error") or e})
    return events


def _live_snapshot() -> dict:
    s = store()
    ws = s.worker_state()
    alive = bool(ws and time.time() - (ws["state"].get("heartbeat") or 0) < 30)
    active = s.q("SELECT id, model, benchmark, tier, status, n_total, n_done, n_graded, n_error, n_pending, "
                 "tokens_out, gen_seconds, started_at, updated_at, error FROM runs "
                 "WHERE status IN ('running','grading','queued') ORDER BY status DESC, priority DESC, created_at")
    recent = s.q("SELECT id, model, benchmark, tier, status, score, n_graded, n_total, n_error, finished_at, error "
                 "FROM runs WHERE status IN ('done','failed','skipped','paused') "
                 "ORDER BY COALESCE(finished_at, updated_at) DESC LIMIT 15")
    return {"now": time.time(), "worker": ws["state"] if alive else None, "worker_alive": alive,
            "active": active, "recent": recent}


@app.get("/api/live/snapshot")
def live_snapshot():
    return _live_snapshot()


@app.get("/api/live")
async def live():
    async def gen():
        while True:
            data = await asyncio.to_thread(_live_snapshot)
            yield f"data: {json.dumps(data, default=str)}\n\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/servers/{key}/log")
def server_log(key: str, lines: int = 200):
    f = load_config().results_dir / "servers" / f"{Path(key).name}.log"
    if not f.is_file():
        return JSONResponse({"lines": []})
    return {"lines": f.read_text(errors="replace").splitlines()[-lines:]}


STATIC = ROOT / "bench" / "server" / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
