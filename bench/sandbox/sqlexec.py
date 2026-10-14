"""Runs inside the sandbox: execute gold and predicted SQL on a read-only SQLite DB and compare.

stdin: {"db": path, "pred": sql, "gold": sql, "timeout": seconds, "mode": "set"|"ordered_multiset"}
stdout: {"match": bool, "error": str|None, "pred_rows": int, "gold_rows": int}
"""

import json
import sqlite3
import sys
import time
from collections import Counter


def run(db: str, sql: str, timeout: float):
    conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def norm(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def main():
    job = json.loads(sys.stdin.read())
    out = {"match": False, "error": None, "pred_rows": 0, "gold_rows": 0}
    try:
        gold = run(job["db"], job["gold"], job["timeout"])
    except Exception as e:
        out["error"] = f"gold failed: {e}"
        print(json.dumps(out))
        return
    out["gold_rows"] = len(gold)
    try:
        pred = run(job["db"], job["pred"], job["timeout"])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out))
        return
    out["pred_rows"] = len(pred)
    g = [tuple(norm(v) for v in r) for r in gold]
    p = [tuple(norm(v) for v in r) for r in pred]
    if job.get("mode") == "set":
        out["match"] = set(g) == set(p)
    elif "order by" in job["gold"].lower():
        out["match"] = g == p
    else:
        out["match"] = Counter(g) == Counter(p)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
