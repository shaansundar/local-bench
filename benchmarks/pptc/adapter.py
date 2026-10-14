import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from bench import sandbox
from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import vendor_repo
from bench.sandbox.exec import scratch_dir

UPSTREAM = "https://github.com/gydpku/PPTC"
REVISION = "277f56ffc548ce51024d5cc7c0607aeb49b584b8"
HELPER = Path(__file__).with_name("pptc_helper.py")
DEPS = ["python-pptx==0.6.21", "openai==0.28.1", "backoff", "tiktoken", "pandas", "jsonlines", "sacremoses",
        "mosestokenizer", "matplotlib", "plotly", "scipy", "scikit-learn", "tqdm", "tenacity"]
SETS = {"short": "Create_new_slides", "long": "Edit_PPT_template"}


class PPTC(Benchmark):
    @property
    def repo(self):
        return vendor_repo(UPSTREAM, REVISION, "PPTC")

    @property
    def py(self):
        return self.repo / ".bench_venv" / "bin" / "python"

    def _env(self):
        return {"TIKTOKEN_CACHE_DIR": str(self.data_dir / "tiktoken"), "PYTHONDONTWRITEBYTECODE": "1",
                "MPLCONFIGDIR": str(self.data_dir / "mpl")}

    def _helper(self, *args, sandboxed=False, cwd_write=None, timeout=300):
        cmd = [str(self.py), str(HELPER), *map(str, args)]
        if sandboxed:
            r = sandbox.run(cmd, cwd=self.repo, timeout=timeout, network="none", env=self._env(),
                            write_paths=[cwd_write] if cwd_write else None)
            out, err, rc = r.stdout, r.stderr, r.returncode
        else:
            p = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, **self._env()})
            out, err, rc = p.stdout, p.stderr, p.returncode
        lines = [l for l in out.strip().splitlines() if l.startswith("{")]
        if rc != 0 or not lines:
            raise RuntimeError(f"pptc helper {args[0]} failed (rc={rc}): {err[-800:]}")
        return json.loads(lines[-1])

    def fetch(self):
        repo = self.repo
        if not self.py.exists():
            subprocess.run(["uv", "venv", "-q", "--python", "3.11", str(repo / ".bench_venv")], check=True)
            subprocess.run(["uv", "pip", "install", "-q", "--python", str(self.py), *DEPS], check=True)
        (self.data_dir / "tiktoken").mkdir(exist_ok=True)
        subprocess.run([str(self.py), "-c", "import tiktoken; tiktoken.get_encoding('cl100k_base')"], check=True,
                       env={**os.environ, **self._env()})
        for ds in SETS:
            self._helper("prepare", ds, timeout=3600)
        self.mark_fetched()

    def load(self):
        out = []
        for ds, name in SETS.items():
            folder = self.repo / "PPT_test_input" / name
            files = sorted(os.listdir(folder), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for fname in files:
                with open(folder / fname) as f:
                    turns = [json.loads(l) for l in f if l.strip()]
                for t in turns:
                    out.append(Sample(id=f"{ds}/{fname.split('.')[0]}/{t['Turn']}",
                                      data={"set": ds, "file": fname, "turn": t["Turn"], "instruction": t["User instruction"],
                                            "label_api": t["Feasible API sequence"]}, stratum=ds))
        return out

    def build(self, sample, ctx):
        d = sample.data
        prompt = self._helper("prompt", d["set"], d["file"], d["turn"])["prompt"]
        return ChatRequest(messages=[{"role": "user", "content": prompt}])

    def grade(self, sample, output, ctx):
        d = sample.data
        work = scratch_dir("pptc-")
        reply = work / "reply.txt"
        reply.write_text(output.text or "")
        res = self._helper("grade", d["set"], d["file"], d["turn"], reply, work / "pred.pptx", sandboxed=True,
                           cwd_write=work, timeout=300)
        return Grade(score=float(res["str_correct"]), details={**res, "set": d["set"]})

    def oracle(self, sample):
        apis = [a for a in sample.data["label_api"].strip(";").split(";") if a.strip()]
        return "<code>\n" + ";\n".join(apis) + ";\n</code>"

    def aggregate(self, rows):
        by = defaultdict(list)
        pos_t = pos_c = 0
        for r in rows:
            d = r["details"] or {}
            by[d.get("set")].append(r["score"])
            pos_t += d.get("pos_total", 0)
            pos_c += d.get("pos_correct", 0)
        return {"by_task": {("create new slides" if k == "short" else "edit template"): sum(v) / len(v)
                            for k, v in by.items() if k},
                "position_accuracy": pos_c / pos_t if pos_t else None}


BENCHMARK = PPTC
