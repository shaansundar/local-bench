"""Helpers to run model-written code under the sandbox."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from bench import sandbox
from bench.config import load_config

PYEXEC = Path(__file__).with_name("pyexec.py")


def scratch_dir(prefix: str = "exec-") -> Path:
    base = load_config().results_dir / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def extract_code(text: str, lang: str = "python") -> str | None:
    """Last fenced code block (preferring the requested language)."""
    blocks = re.findall(r"```([\w+-]*)[^\n]*\n(.*?)```", text or "", re.S)
    if not blocks:
        return None
    same = [b for tag, b in blocks if tag.lower() in (lang, lang[:2], "py" if lang == "python" else lang)]
    return (same or [b for _, b in blocks])[-1]


def run_tests(code: str, tests: list[dict], *, mode: str = "stdio", func_name: str | None = None,
              per_test_timeout: float = 6.0, total_timeout: float = 300.0, fail_fast: bool = True) -> dict:
    job = {"code": code, "mode": mode, "func_name": func_name, "tests": tests,
           "timeout": per_test_timeout, "fail_fast": fail_fast}
    work = scratch_dir()
    try:
        r = sandbox.run([sys.executable, "-I", str(PYEXEC)], cwd=work, timeout=total_timeout,
                        input=json.dumps(job), network="none")
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    if r.timed_out:
        return {"passed": 0, "total": len(tests), "results": [{"ok": False, "status": "global_timeout"}]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"passed": 0, "total": len(tests), "results": [{"ok": False, "status": "harness_error",
                                                               "stderr": r.stderr[-1000:]}]}


def run_python(code: str, *, cwd: Path, timeout: float = 120.0, network: str = "none",
               write_paths: list[Path] | None = None) -> sandbox.SandboxResult:
    """Run a standalone script in `cwd` (e.g. a spreadsheet-editing program)."""
    script = cwd / f"_model_code_{abs(hash(code)) % 10**8}.py"
    script.write_text(code)
    try:
        return sandbox.run([sys.executable, str(script)], cwd=cwd, timeout=timeout, network=network,
                           write_paths=write_paths)
    finally:
        script.unlink(missing_ok=True)


SQLEXEC = Path(__file__).with_name("sqlexec.py")


def sql_match(db: Path, pred: str, gold: str, *, mode: str = "set", timeout: float = 30.0) -> dict:
    """Execution-accuracy check for text-to-SQL, isolated in the sandbox (no network, no writes)."""
    work = scratch_dir("sql-")
    try:
        r = sandbox.run([sys.executable, "-I", str(SQLEXEC)], cwd=work, timeout=timeout * 2 + 15, network="none",
                        input=json.dumps({"db": str(db), "pred": pred, "gold": gold, "timeout": timeout, "mode": mode}))
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    if r.timed_out:
        return {"match": False, "error": "timeout"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"match": False, "error": f"harness: {r.stderr[-300:]}"}


def sqlite_schema(db: Path, sample_rows: int = 0) -> str:
    """CREATE TABLE statements (+ optional sample rows as comments) for a text-to-SQL prompt."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        parts = []
        for name, ddl in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
                                      "AND name NOT LIKE 'sqlite_%'"):
            block = ddl.strip()
            if sample_rows:
                try:
                    cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}')
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    block += f"\n/* {len(rows)} example rows:\n" + "\t".join(cols) + "\n" + \
                             "\n".join("\t".join(str(v)[:60] for v in r) for r in rows) + "\n*/"
                except sqlite3.Error:
                    pass
            parts.append(block)
        return "\n\n".join(parts)
    finally:
        conn.close()
