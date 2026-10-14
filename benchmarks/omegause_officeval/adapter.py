import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

from bench import sandbox
from bench.benchmark import AgentTask, Benchmark, Grade, Sample
from bench.common import hf_snapshot, vendor_repo

REPO = "baidu-frontier-research/OmegaUse-OfficeVal"
UPSTREAM = "https://github.com/baidu-frontier-research/OmegaUse-OfficeVal"
REVISION = "5f3da616ac6a0bc30ca35e9678f8f1eb255bd91a"
COM_REQUIRED = {"001", "008", "019", "022", "023", "030", "039", "074", "081"}  # core/config.py

PROMPT = """You are completing a professional office task.

Input files are in ./input/ ({files}).
Save every final deliverable into ./output/ — only the deliverable files, nothing else (no scripts or scratch files).
Use Python (python-docx, openpyxl, python-pptx, pdfplumber, pypdfium2 and Pillow are installed) or shell tools.

Task:
{instruction}"""

RUN_VERIFIER = r'''
import importlib.util, json, sys, os
spec = importlib.util.spec_from_file_location("verifier", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.modules["verifier"] = mod  # dataclasses in verifiers need their module registered
os.environ["OFFICEVAL_COM_ENABLED"] = "0"
spec.loader.exec_module(mod)
res = mod.evaluate(sys.argv[2])
print("@@RESULT@@" + json.dumps(res, default=str, ensure_ascii=False))
'''


class OmegaUse(Benchmark):
    @property
    def repo(self):
        return vendor_repo(UPSTREAM, REVISION, "OmegaUse-OfficeVal")

    def fetch(self):
        _ = self.repo
        hf_snapshot(REPO, self.data_dir, allow=["task-en/*", "tasks/*", "task_files/**"])
        self.mark_fetched()

    def load(self):
        lang_dir = "task-en" if self.spec.options.get("language", "en") == "en" else "tasks"
        out = []
        for f in sorted((self.data_dir / lang_dir).glob("officeval_*.json")):
            t = json.loads(f.read_text())
            num = t["id"].split("_")[-1]
            if num in COM_REQUIRED:
                continue
            out.append(Sample(id=t["id"], data={"instruction": t["instruction"], "domain": t.get("domain"),
                                                "intent": t.get("operation_intent"),
                                                "human_minutes": t.get("human_labor_time")},
                              stratum=t.get("operation_intent")))
        return out

    def _inputs(self, sid):
        d = self.data_dir / "task_files" / sid
        return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []

    def build(self, sample, ctx):
        inputs = self._inputs(sample.id)
        files = {f"input/{p.relative_to(self.data_dir / 'task_files' / sample.id)}": p for p in inputs}
        files["output/.keep"] = self.data_dir / ".fetched"
        names = ", ".join(p.name for p in inputs) or "none"
        return AgentTask(prompt=PROMPT.format(files=names, instruction=sample.data["instruction"]), files=files)

    def grade(self, sample, output, ctx):
        if output.workspace is None:
            return Grade(score=0.0, details={"error": "no workspace", "intent": sample.data["intent"]})
        out_dir = output.workspace / "output"
        (out_dir / ".keep").unlink(missing_ok=True)
        if not out_dir.exists() or not any(out_dir.iterdir()):
            return Grade(score=0.0, details={"error": "no deliverables in ./output", "intent": sample.data["intent"]})
        num = sample.id.split("_")[-1]
        verifier = self.repo / "verifiers" / f"officeval_{num}_verifier.py"
        runner = out_dir.parent / "_run_verifier.py"
        runner.write_text(RUN_VERIFIER)
        r = sandbox.run([sys.executable, str(runner), str(verifier), str(out_dir)], cwd=out_dir.parent, timeout=900,
                        network="none", env={"PYTHONPATH": f"{self.repo}:{self.repo / 'verifiers'}"})
        runner.unlink(missing_ok=True)
        raw = r.stdout.rpartition("@@RESULT@@")[2].strip()
        if not raw:
            return Grade(score=0.0, details={"error": f"verifier crashed: {r.stderr[-600:]}", "intent": sample.data["intent"]})
        res = json.loads(raw)
        total, maximum = res.get("total_score"), res.get("max_score")
        if res.get("status") != "ok" or res.get("dim1_pass") is False or not maximum:
            score = 0.0
        else:
            score = max(0.0, float(total)) / float(maximum)
        return Grade(score=score, extracted=f"{total}/{maximum}",
                     details={"status": res.get("status"), "dim1_pass": res.get("dim1_pass"),
                              "dim1_reason": str(res.get("dim1_reason"))[:300], "intent": sample.data["intent"],
                              "items": res.get("dim2_items")})

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("intent")].append(r["score"])
        return {"by_intent": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = OmegaUse
