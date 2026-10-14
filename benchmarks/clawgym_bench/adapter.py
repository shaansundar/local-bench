import json
import shutil
import sys
from collections import defaultdict
from types import SimpleNamespace

from bench import sandbox
from bench.benchmark import AgentTask, Benchmark, Grade, JudgeUnavailable, Sample
from bench.common import read_jsonl, vendor_repo

UPSTREAM = "https://github.com/ClawGym/ClawGym-Bench"
REVISION = "c8cbd1d0df033af24fc14064d40d6964ca8f611f"
CODE_W, RUBRIC_W = 0.7, 0.3


def _ib():
    root = vendor_repo(UPSTREAM, REVISION, "ClawGym-Bench")
    pkg = root / "evaluation" / "localclawbench"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from internalbench import judge, reward_with_grade, workspace

    return root, workspace, judge, reward_with_grade


class ClawGym(Benchmark):
    def fetch(self):
        _ib()
        self.mark_fetched()

    def load(self):
        root = _ib()[0]
        rows = read_jsonl(root / "data" / "benchmark_data.jsonl")
        return [Sample(id=f"{i:03d}", data={**r, "ordinal": i}, stratum=r.get("macro_category")) for i, r in enumerate(rows)]

    def _record(self, sample):
        return SimpleNamespace(prompt=sample.data["prompt"], rules=sample.data.get("rules") or [],
                               input_files=sample.data["input_files"], raw=sample.data)

    def build(self, sample, ctx):
        _, ws_mod, _, _ = _ib()
        staging = ctx.run_dir / "staging" / sample.id
        if staging.exists():
            shutil.rmtree(staging)
        ws_mod.stage_payload_files(self._record(sample), staging, include_reward=False)  # hidden verifier excluded
        files = {str(p.relative_to(staging)): p for p in staging.rglob("*") if p.is_file()}
        return AgentTask(prompt=sample.data["prompt"], files=files)

    def grade(self, sample, output, ctx):
        _, ws_mod, judge_mod, rw = _ib()
        ws = output.workspace
        if ws is None or not ws.exists():
            return Grade(score=0.0, details={"error": "no workspace"})
        ws_mod.stage_reward_file(self._record(sample), ws)  # inject the hidden test only now
        r = sandbox.run([sys.executable, "reward/test.py", str(ws)], cwd=ws, timeout=600, network="none")
        code_score, metrics = rw._parse_reward_output(r.stdout, r.stderr)
        code_score = float(code_score) if code_score is not None else 0.0
        rules = sample.data.get("rules") or []
        details = {"category": sample.stratum, "code_score": code_score, "metrics": metrics, "hybrid": bool(rules)}
        if not rules:
            return Grade(score=code_score, details=details)
        try:
            judge = ctx.require_judge()
        except JudgeUnavailable:
            return Grade(score=None, pending=True, details={**details, "pending": "rubric needs the judge"})
        targets = {}
        for rule in rules:
            fp = str(rule.get("file_path") or "")
            if not fp or fp in targets:
                continue
            p = ws / fp
            targets[fp] = {"exists": p.is_file(), "content": p.read_text(errors="replace")[:20000] if p.is_file() else ""}
        prompt = judge_mod.build_rubric_prompt(self._record(sample), targets)
        parsed, info = judge.score_json(prompt, "rubric", system=judge_mod.JUDGE_SYSTEM_PROMPT)
        rubric, norm = judge_mod.score_judge_scores(parsed, rules)
        rubric = float(rubric or 0.0)
        return Grade(score=CODE_W * code_score + RUBRIC_W * rubric, extracted=f"code {code_score:.2f} / rubric {rubric:.2f}",
                     judge={**info, "normalization": norm}, details={**details, "rubric_score": rubric})

    def aggregate(self, rows):
        by = defaultdict(list)
        code = [(r["details"] or {}).get("code_score", 0) for r in rows]
        for r in rows:
            by[(r["details"] or {}).get("category")].append(r["score"])
        return {"code_only_mean": sum(code) / len(code) if code else None,
                "by_category": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = ClawGym
