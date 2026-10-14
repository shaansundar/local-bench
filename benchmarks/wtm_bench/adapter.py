import ast
import json
import subprocess

from bench import sandbox
from bench.benchmark import Benchmark, Grade, ModelOutput, Sample
from bench.common import vendor_repo
from bench.config import resolve_secret

UPSTREAM = "https://github.com/microsoft/WTM-Bench"
REVISION = "83d45ae601f3241ac7246613c67b765e48f93c06"


def _info(v):
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(v)


class WTMBench(Benchmark):
    @property
    def repo(self):
        return vendor_repo(UPSTREAM, REVISION, "WTM-Bench")

    @property
    def py(self):
        return self.repo / ".bench_venv" / "bin" / "python"

    def fetch(self):
        if not self.py.exists():
            subprocess.run(["uv", "venv", "-q", "--python", "3.11", str(self.repo / ".bench_venv")], check=True)
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(self.py), "-r", str(self.repo / "requirements.txt")],
                           check=True)
        self.mark_fetched()

    def _table(self):
        import pyarrow.parquet as pq

        return pq.read_table(self.repo / self.spec.options.get("test_file", "data/Backward_EnronFuse_v3/enronfuse_150_test.parquet"))

    def load(self):
        t = self._table()
        out = []
        for i, r in enumerate(t.to_pylist()):
            info = _info(r["extra_info"])
            name = str(info.get("file_path", "")).rsplit("/", 1)[-1].removesuffix(".xlsx") or str(i)
            out.append(Sample(id=f"{i:03d}-{name}", data={"row": i, "file": info.get("file_path")}))
        return out

    def execute(self, sample, ctx):
        import pyarrow.parquet as pq

        spec = ctx.model
        base = ctx.base_url or spec.base_url
        if spec.provider != "openai" or not base:
            raise RuntimeError("WTM-Bench needs an OpenAI-compatible endpoint")
        work = ctx.run_dir / "wtm" / sample.id
        work.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._table().slice(sample.data["row"], 1), work / "task.parquet")
        cmd = [str(self.py), str(self.repo / "benchmark.py"), "--models", spec.model_id or spec.model or spec.key,
               "--base-url", base, "--api-key", resolve_secret(spec.api_key) or "EMPTY", "--test-file", str(work / "task.parquet"),
               "--output-dir", str(work / "out"), "--cache-dir", str(work / "cache"),
               "--max-turns", str(self.spec.options.get("max_turns", 10))]
        if spec.max_output_tokens:
            cmd += ["--max-tokens", str(spec.max_output_tokens)]
        r = sandbox.run(cmd, cwd=self.repo, timeout=self.spec.sample_timeout_s or 3600,
                        network="localhost" if spec.is_local else "all", write_paths=[work])
        rollouts = sorted((work / "out").glob("*_rollouts.jsonl"))
        if not rollouts:
            raise RuntimeError(f"WTM-Bench produced no rollout (rc={r.returncode}): {r.stderr[-1500:]}")
        res = json.loads(rollouts[0].read_text().splitlines()[-1])
        conv = res.get("messages") or res.get("conversation") or []
        details = res.get("grading_details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                details = [details]
        summary = {"score": res.get("score"), "cell_grade": res.get("cell_grade"), "step_reward": res.get("step_reward"),
                   "num_turns": res.get("num_turns"), "grading_details": (details or [])[:25]}
        final = next((m.get("content") for m in reversed(conv) if isinstance(m, dict) and m.get("role") == "assistant"), "")
        return ModelOutput(text=str(final or "")[:8000], latency=r.seconds, finish_reason="stop",
                           turns=[{"role": m.get("role"), "content": str(m.get("content"))[:4000]} for m in conv[-40:]
                                  if isinstance(m, dict)],
                           extra={"score": res.get("score"), "detail": summary})

    def grade(self, sample, output, ctx):
        s = output.extra.get("score")
        return Grade(score=float(s or 0.0), details={"wtm": output.extra.get("detail")})


BENCHMARK = WTMBench
