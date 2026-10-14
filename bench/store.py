"""SQLite results store. WAL mode lets the dashboard read while the worker writes.

runs     one row per (model, benchmark, tier) attempt; doubles as the job queue
samples  one row per (run, sample, repeat): prompt, output, grade, usage
worker   live worker state for the dashboard
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from bench.config import load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  benchmark TEXT NOT NULL,
  tier TEXT NOT NULL,
  status TEXT NOT NULL,            -- queued|running|grading|done|failed|paused|skipped
  priority INTEGER DEFAULT 0,
  subset_hash TEXT,
  harness TEXT,
  judge_id TEXT,
  badges TEXT,
  n_total INTEGER DEFAULT 0,
  n_done INTEGER DEFAULT 0,
  n_graded INTEGER DEFAULT 0,
  n_error INTEGER DEFAULT 0,
  n_pending INTEGER DEFAULT 0,
  score REAL, ci_low REAL, ci_high REAL,
  metrics TEXT,
  error TEXT,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  gen_seconds REAL DEFAULT 0,
  created_at REAL, started_at REAL, finished_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS runs_key ON runs(model, benchmark, tier);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS samples (
  run_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  rep INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,            -- generated|graded|error|judge_pending
  score REAL,
  prompt TEXT,
  output TEXT,
  reasoning TEXT,
  extracted TEXT,
  gold TEXT,
  details TEXT,
  judge TEXT,
  error TEXT,
  usage TEXT,
  artifacts TEXT,
  extra TEXT,
  finish_reason TEXT,
  latency REAL,
  created_at REAL,
  graded_at REAL,
  PRIMARY KEY (run_id, sample_id, rep)
);
CREATE TABLE IF NOT EXISTS worker (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pid INTEGER,
  state TEXT,
  updated_at REAL
);
"""

RUN_COLUMNS = (
    "id model benchmark tier status priority subset_hash harness judge_id badges n_total n_done n_graded "
    "n_error n_pending score ci_low ci_high metrics error tokens_in tokens_out gen_seconds created_at "
    "started_at finished_at updated_at"
).split()


def _dump(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return json.dumps(v, default=str, ensure_ascii=False)


class Store:
    def __init__(self, path: Path | None = None):
        cfg = load_config()
        self.path = path or cfg.results_dir / "results.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def q(self, sql: str, args: tuple | list = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def q1(self, sql: str, args: tuple | list = ()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    # ---- runs ----------------------------------------------------------------------------
    def find_resumable(self, model: str, benchmark: str, tier: str) -> dict | None:
        return self.q1(
            "SELECT * FROM runs WHERE model=? AND benchmark=? AND tier=? "
            "AND status IN ('queued','running','paused','failed','grading') ORDER BY created_at DESC LIMIT 1",
            (model, benchmark, tier),
        )

    def create_run(self, model: str, benchmark: str, tier: str, *, priority: int = 0, badges=None) -> str:
        rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        with self.tx() as c:
            c.execute(
                "INSERT INTO runs (id, model, benchmark, tier, status, priority, badges, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, model, benchmark, tier, "queued", priority, _dump(badges or []), now, now),
            )
        return rid

    def update_run(self, rid: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [_dump(v) if k in ("metrics", "badges") else v for k, v in fields.items()]
        with self.tx() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id=?", (*vals, rid))

    def get_run(self, rid: str) -> dict | None:
        return self.q1("SELECT * FROM runs WHERE id=?", (rid,))

    def queued_runs(self) -> list[dict]:
        return self.q("SELECT * FROM runs WHERE status='queued' ORDER BY priority DESC, created_at")

    # ---- samples -------------------------------------------------------------------------
    def done_keys(self, rid: str) -> dict[tuple[str, int], str]:
        rows = self.q("SELECT sample_id, rep, status FROM samples WHERE run_id=?", (rid,))
        return {(r["sample_id"], r["rep"]): r["status"] for r in rows}

    def put_sample(self, rid: str, sample_id: str, rep: int, **fields: Any) -> None:
        fields = {k: (_dump(v) if k in ("prompt", "extracted", "gold", "details", "judge", "usage", "artifacts", "extra") else v)
                  for k, v in fields.items()}
        fields.setdefault("created_at", time.time())
        cols = ["run_id", "sample_id", "rep", *fields]
        ph = ",".join("?" * len(cols))
        upd = ", ".join(f"{k}=excluded.{k}" for k in fields)
        with self.tx() as c:
            c.execute(
                f"INSERT INTO samples ({','.join(cols)}) VALUES ({ph}) "
                f"ON CONFLICT(run_id, sample_id, rep) DO UPDATE SET {upd}",
                (rid, sample_id, rep, *fields.values()),
            )

    def grade_sample(self, rid: str, sample_id: str, rep: int, **fields: Any) -> None:
        fields = {k: (_dump(v) if k in ("extracted", "gold", "details", "judge") else v) for k, v in fields.items()}
        fields["graded_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE samples SET {cols} WHERE run_id=? AND sample_id=? AND rep=?",
                      (*fields.values(), rid, sample_id, rep))

    def sample_rows(self, rid: str, columns: str = "sample_id, rep, status, score, details, usage, latency") -> list[dict]:
        return self.q(f"SELECT {columns} FROM samples WHERE run_id=?", (rid,))

    def refresh_counts(self, rid: str) -> dict:
        row = self.q1(
            "SELECT COUNT(*) AS n_done, "
            "SUM(CASE WHEN status='graded' THEN 1 ELSE 0 END) AS n_graded, "
            "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS n_error, "
            "SUM(CASE WHEN status='judge_pending' THEN 1 ELSE 0 END) AS n_pending, "
            "SUM(COALESCE(json_extract(usage,'$.input_tokens'),0)) AS tokens_in, "
            "SUM(COALESCE(json_extract(usage,'$.output_tokens'),0)) AS tokens_out, "
            "SUM(COALESCE(latency,0)) AS gen_seconds FROM samples WHERE run_id=?",
            (rid,),
        ) or {}
        row = {k: (v or 0) for k, v in row.items()}
        self.update_run(rid, **row)
        return row

    # ---- worker state --------------------------------------------------------------------
    def set_worker_state(self, pid: int, state: dict) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO worker (id, pid, state, updated_at) VALUES (1,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET pid=excluded.pid, state=excluded.state, updated_at=excluded.updated_at",
                (pid, json.dumps(state, default=str), time.time()),
            )

    def worker_state(self) -> dict | None:
        row = self.q1("SELECT * FROM worker WHERE id=1")
        if not row:
            return None
        row["state"] = json.loads(row["state"] or "{}")
        return row
