import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from bench import sandbox
from bench.benchmark import Benchmark, Grade, JudgeUnavailable, ModelOutput, Sample
from bench.common import hf_rows, hf_snapshot
from bench.config import ROOT, resolve_secret
from bench.office import document_text

REPO = "openai/gdpval"
VENV = ROOT / "vendor" / "stirrup" / ".venv"
RUNNER = Path(__file__).with_name("stirrup_runner.py")
JUDGE = """You are an expert evaluator in the occupation "{occupation}". Two deliverables were produced for the same work task. Compare them as a demanding client would: correctness, completeness against every requirement in the task, professional quality and formatting, and whether the files are usable as delivered.

## Task
{task}

## Deliverable A
{a}

## Deliverable B
{b}

Which deliverable is better overall? Reply with brief reasoning, then a JSON object: {{"winner": "A" | "B" | "tie", "reason": "..."}}"""


def _listify(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        import ast

        return ast.literal_eval(v) if v else []


class GDPval(Benchmark):
    def fetch(self):
        if not (VENV / "bin" / "python").exists():
            subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(VENV)], check=True)
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(VENV / "bin" / "python"), "stirrup==0.2.0", "ddgs",
                            "trafilatura", "python-docx", "openpyxl", "python-pptx", "pandas", "matplotlib", "reportlab"],
                           check=True)
        hf_rows(REPO, self.data_dir, split="train")
        hf_snapshot(REPO, self.data_dir / "files", allow=["reference_files/**", "deliverable_files/**"])
        self.mark_fetched()

    def load(self):
        rows = hf_rows(REPO, self.data_dir, split="train")
        return [Sample(id=r["task_id"], data={"prompt": r["prompt"], "sector": r["sector"], "occupation": r["occupation"],
                                              "refs": _listify(r["reference_files"]), "gold": _listify(r["deliverable_files"])},
                       stratum=r["sector"]) for r in rows]

    def execute(self, sample, ctx):
        spec = ctx.model
        base = ctx.base_url or spec.base_url
        if spec.provider != "openai" or not base:
            raise RuntimeError("GDPval-AA (Stirrup) needs an OpenAI-compatible endpoint")
        work = ctx.run_dir / "gdpval" / sample.id
        out = work / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        (work / "exec").mkdir(exist_ok=True)
        job = {"base_url": base, "api_key": resolve_secret(spec.api_key) or "local", "model": spec.model_id or spec.model or spec.key,
               "max_tokens": spec.max_output_tokens or 32768, "context": spec.context, "prompt": sample.data["prompt"],
               "input_files": [str(self.data_dir / "files" / f) for f in sample.data["refs"]], "output_dir": str(out),
               "exec_dir": str(work / "exec"), "max_turns": int(self.spec.options.get("max_turns", 60)),
               "result_path": str(work / "result.json")}
        (work / "job.json").write_text(json.dumps(job))
        r = sandbox.run([str(VENV / "bin" / "python"), str(RUNNER), str(work / "job.json")], cwd=work,
                        timeout=self.spec.sample_timeout_s or 5400, network="all", write_paths=[work],
                        env={"PATH": f"{VENV / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin", "MPLCONFIGDIR": str(work / "mpl")})
        res = json.loads((work / "result.json").read_text()) if (work / "result.json").exists() else {}
        if not res:
            raise RuntimeError(f"Stirrup run failed (rc={r.returncode}): {r.stderr[-1500:]}")
        return ModelOutput(text=res.get("finish", ""), latency=r.seconds, finish_reason="stop", workspace=work,
                           artifacts={"workspace": str(out.relative_to(ctx.run_dir))}, extra={"files": res.get("files", [])})

    def _render(self, paths, limit):
        blocks = []
        for p in paths:
            try:
                text = document_text(p)
            except Exception:
                text = f"(binary file, {p.stat().st_size} bytes)"
            blocks.append(f"### File: {p.name}\n{text[:limit]}")
        return "\n\n".join(blocks) or "(no files delivered)"

    def grade(self, sample, output, ctx):
        judge = ctx.require_judge()
        limit = int(self.spec.options.get("max_chars_per_file", 30000))
        produced = [output.workspace / "outputs" / f for f in output.extra.get("files", [])] if output.workspace else []
        expert = [self.data_dir / "files" / f for f in sample.data["gold"]]
        model_txt = self._render(produced, limit) if produced else "(no files delivered)\n\nFinal message: " + (output.text or "")[:4000]
        expert_txt = self._render(expert, limit)
        model_is_a = random.Random(sample.id).random() < 0.5
        a, b = (model_txt, expert_txt) if model_is_a else (expert_txt, model_txt)
        obj, info = judge.score_json(JUDGE.format(occupation=sample.data["occupation"], task=sample.data["prompt"], a=a, b=b), "pairwise")
        w = str(obj.get("winner", "")).strip().upper()
        if w == "TIE":
            score, outcome = 0.5, "tie"
        elif (w == "A") == model_is_a and w in ("A", "B"):
            score, outcome = 1.0, "win"
        else:
            score, outcome = 0.0, "loss"
        return Grade(score=score, extracted=outcome, judge={**info, "model_was": "A" if model_is_a else "B"},
                     details={"sector": sample.stratum, "outcome": outcome, "files": output.extra.get("files", [])})

    def aggregate(self, rows):
        outcomes = [(r["details"] or {}).get("outcome") for r in rows]
        n = len(outcomes) or 1
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("sector")].append(r["score"])
        return {"win_or_tie_rate": sum(o in ("win", "tie") for o in outcomes) / n, "win_rate": outcomes.count("win") / n,
                "by_sector": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = GDPval
