import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from bench import sandbox
from bench.benchmark import Benchmark, Grade, ModelOutput, Sample
from bench.common import read_jsonl, vendor_repo, write_jsonl
from bench.config import resolve_secret

UPSTREAM = "https://github.com/ShishirPatil/gorilla"
REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
HELPER = Path(__file__).with_name("bfcl_helper.py")
GROUP = {"simple": "non-live", "multiple": "non-live", "parallel": "non-live", "irrelevance": "non-live",
         "live": "live", "multi": "multi-turn", "web": "agentic"}


class BFCL(Benchmark):
    @property
    def root(self):
        return vendor_repo(UPSTREAM, REVISION, "gorilla", sparse=["/berkeley-function-call-leaderboard/"]) / \
            "berkeley-function-call-leaderboard"

    @property
    def py(self):
        return self.root / ".bench_venv" / "bin" / "python"

    def fetch(self):
        import subprocess

        if not self.py.exists():
            subprocess.run(["uv", "venv", "-q", "--python", "3.11", str(self.root / ".bench_venv")], check=True)
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(self.py), "-e", str(self.root), "ddgs", "soundfile"], check=True)
        work = self.data_dir / "list"
        work.mkdir(exist_ok=True)
        p = subprocess.run([str(self.py), str(HELPER), "list"], capture_output=True, text=True,
                           env={**os.environ, "BFCL_PROJECT_ROOT": str(work)}, timeout=600)
        rows = json.loads([l for l in p.stdout.splitlines() if l.startswith("[")][-1])
        write_jsonl(self.data_dir / "entries.jsonl", rows)
        self.mark_fetched()

    def load(self):
        return [Sample(id=f"{r['category']}/{r['id']}", data=r, stratum=r["category"]) for r in read_jsonl(self.data_dir / "entries.jsonl")]

    def execute(self, sample, ctx):
        spec = ctx.model
        base = ctx.base_url or spec.base_url
        if spec.provider != "openai" or not base:
            raise RuntimeError("BFCL here needs an OpenAI-compatible endpoint (local model, OpenAI, OpenRouter)")
        work = ctx.run_dir / "bfcl" / sample.id.replace("/", "__")
        work.mkdir(parents=True, exist_ok=True)
        env = {"BFCL_PROJECT_ROOT": str(work), "OPENAI_BASE_URL": base, "OPENAI_API_KEY": resolve_secret(spec.api_key) or "local",
               "BENCH_MODEL_ID": spec.model_id or spec.model or spec.key, "TOKENIZERS_PARALLELISM": "false",
               "HF_HOME": str(self.data_dir / "hf")}
        r = sandbox.run([str(self.py), str(HELPER), "run", sample.data["category"], sample.data["id"]], cwd=work,
                        timeout=self.spec.sample_timeout_s or 1800, network="all", env=env,
                        write_paths=[self.data_dir / "hf"])
        lines = [l for l in r.stdout.splitlines() if l.startswith('{"correct"')]
        if not lines:
            raise RuntimeError(f"bfcl helper failed (rc={r.returncode}): {r.stderr[-1500:]}")
        res = json.loads(lines[-1])
        return ModelOutput(text=res.get("result_tail") or "", latency=r.seconds, finish_reason="stop",
                           extra={"correct": res.get("correct"), "failure": res.get("failure")})

    def grade(self, sample, output, ctx):
        cat = sample.data["category"]
        return Grade(score=float(bool(output.extra.get("correct"))), details={"category": cat, "failure": output.extra.get("failure")})

    def aggregate(self, rows):
        by_cat, by_group = defaultdict(list), defaultdict(list)
        for r in rows:
            c = (r["details"] or {}).get("category", "")
            by_cat[c].append(r["score"])
            by_group[GROUP.get(c.split("_")[0], "other")].append(r["score"])
        avg = lambda v: sum(v) / len(v)
        return {"by_group": {k: avg(v) for k, v in by_group.items()}, "by_category": {k: avg(v) for k, v in sorted(by_cat.items())}}


BENCHMARK = BFCL
