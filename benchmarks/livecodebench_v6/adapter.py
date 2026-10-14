import base64
import json
import zlib
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_file, safe_unpickle
from bench.sandbox.exec import extract_code, run_tests

REPO = "livecodebench/code_generation_lite"
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
          "generate a correct Python program that matches the specification and passes all tests.")
WITH_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your "
                "code within delimiters.")
WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
                   "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
                   "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")


def decode_private(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return json.loads(safe_unpickle(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


class LiveCodeBench(Benchmark):
    def fetch(self):
        for f in self.spec.options.get("files", ["test6.jsonl"]):
            hf_file(REPO, f, self.data_dir)
        self.mark_fetched()

    def load(self):
        o = self.spec.options
        lo, hi = o.get("date_from", "0000"), o.get("date_to", "9999")
        out = []
        for f in o.get("files", ["test6.jsonl"]):
            with open(self.data_dir / f) as fh:
                for line in fh:
                    r = json.loads(line)
                    date = r["contest_date"][:10]
                    if not (lo <= date <= hi):
                        continue
                    meta = json.loads(r.get("metadata") or "{}")
                    out.append(Sample(
                        id=r["question_id"],
                        data={"title": r["question_title"], "content": r["question_content"],
                              "starter_code": r.get("starter_code") or "", "platform": r["platform"],
                              "difficulty": r["difficulty"], "date": date, "func_name": meta.get("func_name"),
                              "public": r["public_test_cases"], "private": r["private_test_cases"]},
                        stratum=r["difficulty"]))
        return out

    def build(self, sample, ctx):
        d = sample.data
        p = f"### Question:\n{d['content']}\n\n"
        if d["starter_code"]:
            p += f"### Format: {WITH_STARTER}\n```python\n{d['starter_code']}\n```\n\n"
        else:
            p += f"### Format: {WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
        p += "### Answer: (use the provided format with backticks)\n\n"
        return ChatRequest(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}])

    def grade(self, sample, output, ctx):
        d = sample.data
        code = extract_code(output.text)
        if not code:
            return Grade(score=0.0, details={"status": "no_code", "difficulty": d["difficulty"]})
        tests = json.loads(d["public"]) + decode_private(d["private"])
        call = bool(d["starter_code"] and d["func_name"])
        res = run_tests(code, [{"input": t["input"], "output": t["output"]} for t in tests],
                        mode="call" if call else "stdio", func_name=d["func_name"],
                        per_test_timeout=float(self.spec.options.get("per_test_timeout_s", 6)),
                        total_timeout=600)
        ok = res["passed"] == res["total"] and res["total"] > 0
        first_fail = next((r for r in res["results"] if not r["ok"]), None)
        return Grade(score=float(ok), extracted=code[-4000:],
                     details={"passed": res["passed"], "total": res["total"], "difficulty": d["difficulty"],
                              "platform": d["platform"], "first_failure": first_fail})

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("difficulty")].append(r["score"])
        return {"by_difficulty": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = LiveCodeBench
