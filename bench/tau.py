"""Shared runner for tau2-bench domains (τ²-Telecom, τ³-Banking).

Each sample = one task, run through tau2's own CLI (its own venv, installed from its uv.lock):
the model under test is the agent, the suite's judge model is the user simulator (a fixed cloud
model, so all agents face the same user), reward comes from tau2's evaluator.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from bench.benchmark import Benchmark, Grade, JudgeUnavailable, ModelOutput, Sample
from bench.common import vendor_repo
from bench.config import ModelSpec, resolve_secret

UPSTREAM = "https://github.com/sierra-research/tau2-bench"
REVISION = "fc0055dc4e0a316c3f83133267fbd6faaa770992"  # v1.0.1 (banking_knowledge grading fix)


def litellm_ref(spec: ModelSpec, base_url: str | None) -> tuple[str, dict]:
    """LiteLLM model string + extra args for tau2's --*-llm / --*-llm-args."""
    if spec.provider == "anthropic":
        args = {}
        if spec.api_key and resolve_secret(spec.api_key):
            args["api_key"] = resolve_secret(spec.api_key)
        return f"anthropic/{spec.model_id or spec.key}", args
    url = base_url or spec.base_url
    if not url:
        raise RuntimeError(f"model {spec.key}: tau2 needs an OpenAI-compatible endpoint or an Anthropic model")
    args = {"api_base": url, "api_key": resolve_secret(spec.api_key) or "local"}
    args.update({k: v for k, v in spec.params.items() if k in ("temperature", "top_p", "reasoning_effort")})
    return f"openai/{spec.model_id or spec.model or spec.key}", args


class TauBenchmark(Benchmark):
    domain = ""
    split: str | None = None
    extra_args: list[str] = []

    @property
    def repo(self) -> Path:
        return vendor_repo(UPSTREAM, REVISION, "tau2-bench")

    @property
    def cli(self) -> Path:
        return self.repo / ".venv" / "bin" / "tau2"

    def fetch(self):
        repo = self.repo
        if not self.cli.exists():
            subprocess.run(["uv", "sync", "--frozen", "--extra", "knowledge"], cwd=repo, check=True)
        self.mark_fetched()

    def _tasks(self) -> list[dict]:
        base = self.repo / "data" / "tau2" / "domains" / self.domain
        tasks = json.loads((base / "tasks.json").read_text())
        if self.split:
            ids = set(json.loads((base / "split_tasks.json").read_text())[self.split])
            tasks = [t for t in tasks if t["id"] in ids]
        return tasks

    def load(self):
        return [Sample(id=str(t["id"]), data={"description": (t.get("description") or {}).get("purpose") if isinstance(t.get("description"), dict) else t.get("description")})
                for t in self._tasks()]

    def execute(self, sample, ctx):
        judge = ctx.judge
        if judge is None or not judge.available:
            raise JudgeUnavailable("tau2 needs the judge model as its user simulator (config.yaml judge.model)")
        agent, agent_args = litellm_ref(ctx.model, ctx.base_url)
        user, user_args = litellm_ref(judge.spec, None)
        out_dir = ctx.run_dir / "tau2"
        out_dir.mkdir(parents=True, exist_ok=True)
        save = out_dir / f"{sample.id.replace('/', '_')}_{int(time.time() * 1000)}.json"
        cmd = [str(self.cli), "run", "--domain", self.domain, "--task-ids", sample.id, "--num-trials", "1",
               "--agent-llm", agent, "--agent-llm-args", json.dumps(agent_args),
               "--user-llm", user, "--user-llm-args", json.dumps(user_args),
               "--save-to", str(save), "--max-concurrency", "1", *self.extra_args]
        if self.split:
            cmd += ["--task-split-name", self.split]
        env = {**os.environ, "TAU2_DATA_DIR": str(self.repo / "data")}
        t0 = time.time()
        p = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, env=env,
                           timeout=self.spec.sample_timeout_s or 3600)
        found = [save] if save.exists() else sorted(out_dir.glob(save.stem + "*"))
        if not found:
            raise RuntimeError(f"tau2 produced no results (rc={p.returncode}): {p.stderr[-1500:]}")
        data = json.loads(found[0].read_text())
        sims = data.get("simulations") or []
        if not sims:
            raise RuntimeError(f"tau2 results contain no simulation: {p.stderr[-800:]}")
        sim = sims[0]
        reward = (sim.get("reward_info") or {}).get("reward")
        msgs = sim.get("messages") or []
        usage = {"input_tokens": 0, "output_tokens": 0}
        for m in msgs:
            u = m.get("usage") or {}
            if m.get("role") == "assistant":
                usage["input_tokens"] += u.get("prompt_tokens") or 0
                usage["output_tokens"] += u.get("completion_tokens") or 0
        transcript = [{"role": m.get("role"), "content": (m.get("content") or "")[:4000],
                       **({"tool_calls": m.get("tool_calls")} if m.get("tool_calls") else {})} for m in msgs]
        return ModelOutput(text=f"reward={reward} termination={sim.get('termination_reason')}", usage=usage,
                           latency=time.time() - t0, finish_reason=str(sim.get("termination_reason")), turns=transcript,
                           artifacts={"tau2_results": str(found[0].relative_to(ctx.run_dir))},
                           extra={"reward": reward, "reward_info": sim.get("reward_info"),
                                  "user_simulator": judge.model_key})

    def grade(self, sample, output, ctx):
        r = output.extra.get("reward")
        if r is None:
            return Grade(score=0.0, details={"error": "no reward"})
        return Grade(score=float(r), details={"reward_info": output.extra.get("reward_info"),
                                              "user_simulator": output.extra.get("user_simulator")})
