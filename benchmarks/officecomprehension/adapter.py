import json
import re
from collections import defaultdict

import pandas as pd

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import context_budget, download, hf_snapshot, read_jsonl, truncate_middle, vendor_raw_files, VENDOR, write_jsonl
from bench.judge import parse_last_json
from bench.office import document_text
from bench.runners.chat import resolve_max_tokens

REPO = "microsoft/OfficeComprehensionBenchmark"
UPSTREAM = "microsoft/OfficeComprehensionBench"
REVISION = "ce90d36916025d5a4a9c3794e422d10a30c0ae80"
import os

# sec.gov rejects requests without a declared contact (e.g. "Your Name your@email.com"); set it in .env
UA = os.environ.get("BENCH_DOWNLOAD_USER_AGENT") or "Mozilla/5.0 (research benchmark download; local-bench)"


def _judge_prompt():
    vendor_raw_files(UPSTREAM, REVISION, ["eval_prompt.md"], VENDOR / "officecomprehension")
    return (VENDOR / "officecomprehension" / "eval_prompt.md").read_text()


class OCB(Benchmark):
    def fetch(self):
        _judge_prompt()
        snap = hf_snapshot(REPO, self.data_dir / "hf")
        urls = pd.read_parquet(snap / "data" / "ocb_source_urls.parquet")
        q = pd.read_parquet(snap / "data" / "ocb_qna_data.parquet")
        text_dir = self.data_dir / "text"
        text_dir.mkdir(exist_ok=True)
        have = {p.name: p for p in (snap / "reference_files").glob("*")}
        missing = []
        for r in urls.to_dict("records"):
            name = r["filename"]
            out = text_dir / f"{name}.txt"
            if out.exists():
                continue
            src = have.get(name)
            if src is None:
                ext = str(r.get("original_format") or name.rsplit(".", 1)[-1]).lower()
                dest = self.data_dir / "orig" / f"{name.rsplit('.', 1)[0]}.{ext}"
                try:
                    import httpx

                    if not dest.exists():
                        dest.parent.mkdir(exist_ok=True)
                        resp = httpx.get(r["url"], headers={"User-Agent": UA}, follow_redirects=True, timeout=120)
                        resp.raise_for_status()
                        dest.write_bytes(resp.content)
                    src = dest
                except Exception as e:
                    missing.append((name, str(e)[:120]))
                    continue
            try:
                out.write_text(document_text(src))
            except Exception as e:
                missing.append((name, f"extract: {e}"[:120]))
        rows = []
        for i, r in enumerate(q.to_dict("records")):
            files = list(r["reference_files"])
            if not all((text_dir / f"{f}.txt").exists() for f in files):
                continue
            rows.append({"id": str(i), "question": r["question"], "files": files, "assertions": list(r["expected_assertions"]),
                         "weights": [float(w) for w in (r["weights"] if r["weights"] is not None else [])],
                         "track": r["track"], "domain": r["domain"], "app": r["app_type"]})
        write_jsonl(self.data_dir / "meta.jsonl", rows)
        (self.data_dir / "missing_files.json").write_text(json.dumps(missing, indent=1))
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["assertions"], stratum=r["track"]) for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def build(self, sample, ctx):
        d = sample.data
        budget = context_budget(ctx, resolve_max_tokens(None, self, ctx)) - 800
        per = max(2000, budget // max(1, len(d["files"])))
        docs = "\n\n".join(f'<document name="{f}">\n{truncate_middle((self.data_dir / "text" / f"{f}.txt").read_text(), per)}\n</document>'
                           for f in d["files"])
        prompt = f"{docs}\n\nUsing the document(s) above, answer the user's request.\n\nRequest: {d['question']}"
        return ChatRequest(messages=[{"role": "user", "content": prompt}])

    def grade(self, sample, output, ctx):
        judge = ctx.require_judge()
        tmpl = _judge_prompt()
        d = sample.data
        weights = d["weights"] if len(d["weights"]) == len(d["assertions"]) else [1.0] * len(d["assertions"])
        results, got, total = [], 0.0, 0.0
        for a, w in zip(d["assertions"], weights):
            prompt = tmpl.replace("{question}", d["question"]).replace("{ai_response}", output.text or "").replace("{assertion}", a)
            obj, info = judge.score_json(prompt, "assertions")
            passed = int(obj.get("score", 0) or 0) == 1 if obj else bool(re.search(r"assertion is TRUE", info["raw"], re.I))
            results.append({"assertion": a[:200], "weight": w, "pass": passed})
            got += w * passed
            total += w
        return Grade(score=got / total if total else 0.0, extracted=f"{sum(r['pass'] for r in results)}/{len(results)} assertions",
                     judge={"judge_id": info["judge_id"], "assertions": results}, details={"track": d["track"], "app": d["app"]})

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("track")].append(r["score"])
        return {"by_track": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = OCB
