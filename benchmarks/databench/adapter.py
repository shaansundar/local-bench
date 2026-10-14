import json
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from bench import sandbox
from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_file, hf_rows
from bench.sandbox.exec import extract_code, scratch_dir

REPO = "cardiffnlp/databench"

PROMPT = """You are given a pandas DataFrame `df` loaded from the "{dataset}" dataset.

Columns and dtypes:
{schema}

First rows:
{head}

Question: {question}

Write a Python function `answer(df: pd.DataFrame)` that computes the answer from the FULL DataFrame and returns it as one of: a boolean, a single value (category or number), or a list of values. Return the value only — no explanations. Put the function in a single ```python code block (pandas is imported as pd, numpy as np)."""

RUNNER = r'''
import json, sys
import numpy as np
import pandas as pd
df = pd.read_parquet(sys.argv[1])
ns = {"pd": pd, "np": np}
exec(open(sys.argv[2]).read(), ns)
res = ns["answer"](df.copy())
def conv(x):
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple, set)):
        return [conv(i) for i in list(x)]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return x
print("@@RESULT@@" + json.dumps(conv(res), default=str))
'''


def default_compare(value, truth, semantic):
    """Verbatim logic of databench_eval Evaluator.default_compare (jorses/databench_eval)."""
    STRIP_CHARS = "[]'\" "
    semantic = semantic.strip()
    valid_null_set = [None, "nan", "", " ", np.nan, "np.nan", "None"]
    if str(value).strip(STRIP_CHARS) in valid_null_set and str(truth).strip(STRIP_CHARS) in valid_null_set:
        return True
    if str(value).strip(STRIP_CHARS) in valid_null_set or str(truth).strip(STRIP_CHARS) in valid_null_set:
        return False
    if semantic == "boolean":
        t, f = ["true", "yes", "y"], ["false", "no", "n"]
        v, g = str(value).strip(STRIP_CHARS).lower(), str(truth).strip(STRIP_CHARS).lower()
        return (v in t and g in t) or (v in f and g in f)
    if semantic == "category":
        v, g = str(value).strip(STRIP_CHARS), str(truth).strip(STRIP_CHARS)
        if v == g:
            return True
        try:
            return pd.to_datetime(v).date() == pd.to_datetime(g).date()
        except (ValueError, TypeError):
            return v == g
    if semantic == "number":
        try:
            vc = "".join(c for c in str(value) if c.isdigit() or c in [".", "-"])
            gc = "".join(c for c in str(truth) if c.isdigit() or c in [".", "-"])
            return math.trunc(float(vc) * 100) == math.trunc(float(gc) * 100)
        except Exception:
            return False
    if semantic == "list[category]":
        try:
            vl = [i.strip(STRIP_CHARS) for i in str(value).strip("[]").split(",")]
            gl = [i.strip(STRIP_CHARS) for i in str(truth).strip("[]").split(",")]
            vl = [v if v not in valid_null_set else "" for v in vl]
            gl = [g if g not in valid_null_set else "" for g in gl]
            if len(vl) != len(gl):
                return False
            try:
                return set(pd.to_datetime(i).date() for i in vl) == set(pd.to_datetime(i).date() for i in gl)
            except (ValueError, TypeError):
                return set(vl) == set(gl)
        except Exception:
            return False
    if semantic == "list[number]":
        try:
            vl = sorted(float("".join(c for c in v.strip() if c.isdigit() or c in [".", "-"])) for v in str(value).strip("[]").split(",") if v.strip())
            gl = sorted(float("".join(c for c in t.strip() if c.isdigit() or c in [".", "-"])) for t in str(truth).strip("[]").split(",") if t.strip())
            vl = [int(v * 100) / 100 for v in vl]
            gl = [int(t * 100) / 100 for t in gl]
            return len(vl) == len(gl) and set(vl) == set(gl)
        except Exception:
            return False
    raise ValueError(f"Semantic not supported: {semantic}")


def _fmt(v):
    """Render a returned value the way submissions were formatted (lists as Python list strings)."""
    return str(v)


class DataBench(Benchmark):
    def fetch(self):
        rows = hf_rows(REPO, self.data_dir, config="semeval", split="test")
        for ds in sorted({r["dataset"] for r in rows}):
            hf_file(REPO, f"data/{ds}/all.parquet", self.data_dir)
        self.mark_fetched()

    def load(self):
        rows = hf_rows(REPO, self.data_dir, config="semeval", split="test")
        return [Sample(id=str(i), data=r, gold=r["answer"], stratum=r["type"]) for i, r in enumerate(rows)]

    def _parquet(self, ds):
        return self.data_dir / "data" / ds / "all.parquet"

    def build(self, sample, ctx):
        d = sample.data
        df = pd.read_parquet(self._parquet(d["dataset"]))
        schema = "\n".join(f"- {c}: {t}" for c, t in df.dtypes.astype(str).items())
        head = df.head(5).to_string(max_colwidth=40)
        return ChatRequest(messages=[{"role": "user", "content": PROMPT.format(dataset=d["dataset"], schema=schema, head=head,
                                                                               question=d["question"])}])

    def _run(self, code, ds):
        work = scratch_dir("databench-")
        (work / "fn.py").write_text(code)
        (work / "runner.py").write_text(RUNNER)
        r = sandbox.run([sys.executable, str(work / "runner.py"), str(self._parquet(ds)), str(work / "fn.py")],
                        cwd=work, timeout=120, network="none")
        out = r.stdout.rpartition("@@RESULT@@")[2].strip()
        if r.returncode != 0 or not out:
            return None, r.stderr[-500:] or "no result"
        return json.loads(out), None

    def grade(self, sample, output, ctx):
        code = extract_code(output.text)
        if not code:
            return Grade(score=0.0, details={"error": "no code", "type": sample.data["type"]})
        value, err = self._run(code, sample.data["dataset"])
        if err:
            return Grade(score=0.0, extracted=None, details={"error": err, "type": sample.data["type"]})
        ok = default_compare(_fmt(value), sample.gold, sample.data["type"])
        return Grade(score=float(ok), extracted=_fmt(value)[:300], details={"type": sample.data["type"]})

    def oracle(self, sample):
        return f"```python\ndef answer(df):\n    return {sample.gold!r}\n```"

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("type")].append(r["score"])
        return {"by_type": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = DataBench
