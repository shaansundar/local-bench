"""Test harness for model-written Python. Runs *inside* the sandbox; reads a job on stdin:

  {"code": str, "mode": "stdio"|"call", "func_name": str|None,
   "tests": [{"input": str, "output": str}], "timeout": float, "fail_fast": bool}

Each test runs in its own python process (inherits the sandbox). Prints a JSON result:
  {"passed": int, "total": int, "results": [{"ok": bool, "status": str, ...}]}

Comparison semantics follow LiveCodeBench: stdio compares stripped lines, falling back to
exact Decimal comparison of whitespace tokens; call-based compares JSON values (tuples → lists).
"""

import json
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

IMPORT_HEADER = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""

CALL_WRAPPER = """
import json as _json, sys as _sys
_lines = _sys.stdin.read().split("\\n")
_args = [_json.loads(l) for l in _lines if l.strip()]
_obj = Solution() if "Solution" in globals() else None
_fn = getattr(_obj, {func!r}) if _obj is not None and hasattr(_obj, {func!r}) else globals()[{func!r}]
_res = _fn(*_args)
def _norm(x):
    if isinstance(x, tuple):
        return [_norm(i) for i in x]
    if isinstance(x, list):
        return [_norm(i) for i in x]
    return x
_sys.stdout.write("\\n@@RESULT@@" + _json.dumps(_norm(_res)))
"""


def _tokens_equal(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if Decimal(x) != Decimal(y):
                return False
        except InvalidOperation:
            return False
    return True


def stdio_equal(actual: str, expected: str) -> bool:
    al = [l.strip() for l in actual.strip().splitlines()]
    el = [l.strip() for l in expected.strip().splitlines()]
    al = [l for l in al if l] if len(al) != len(el) else al
    el = [l for l in el if l] if len(al) != len(el) else el
    if len(al) != len(el):
        return False
    return all(x == y or _tokens_equal(x, y) for x, y in zip(al, el))


def call_equal(actual, expected) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return False


def main() -> None:
    job = json.loads(sys.stdin.read())
    mode = job.get("mode", "stdio")
    timeout = float(job.get("timeout", 6))
    fail_fast = job.get("fail_fast", True)
    workdir = Path(tempfile.mkdtemp(prefix="run-"))
    prog = workdir / "prog.py"
    body = IMPORT_HEADER + "\n" + job["code"]
    if mode == "call":
        body += "\n" + CALL_WRAPPER.format(func=job["func_name"])
    prog.write_text(body)
    results = []
    passed = 0
    for t in job["tests"]:
        try:
            p = subprocess.run([sys.executable, str(prog)], input=t["input"], capture_output=True, text=True,
                               timeout=timeout, cwd=workdir, errors="replace")
        except subprocess.TimeoutExpired:
            results.append({"ok": False, "status": "timeout"})
            if fail_fast:
                break
            continue
        if p.returncode != 0:
            results.append({"ok": False, "status": "runtime_error", "stderr": p.stderr[-800:]})
            if fail_fast:
                break
            continue
        if mode == "call":
            out = p.stdout.rpartition("@@RESULT@@")[2]
            try:
                actual = json.loads(out)
                ok = call_equal(actual, json.loads(t["output"]))
            except json.JSONDecodeError:
                ok = False
        else:
            ok = stdio_equal(p.stdout, t["output"])
        results.append({"ok": ok, "status": "pass" if ok else "wrong_answer",
                        **({} if ok else {"got": p.stdout[-300:], "want": t["output"][-300:]})})
        passed += ok
        if not ok and fail_fast:
            break
    print(json.dumps({"passed": passed, "total": len(job["tests"]), "results": results}))


if __name__ == "__main__":
    main()
