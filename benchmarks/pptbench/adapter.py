import ast
import json
import sys
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import VENDOR, hf_snapshot, read_jsonl, vendor_repo, write_jsonl
from bench.judge import parse_last_json
from bench.models.client import image_part

UPSTREAM = "https://github.com/Gastronomicluna/PPTBench-Eval"
REVISION = "03c9706f4dc8d4eac6852511fa02764b9675b03e"
SUBSETS = {"understanding": "tyrionhuu/PPTBench-Understanding", "detection": "tyrionhuu/PPTBench-Detection"}


def _official():
    root = vendor_repo(UPSTREAM, REVISION, "PPTBench-Eval")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.detection import judge as det_judge
    from src.detection import prompts as det_prompts
    from src.understanding import prompts as und_prompts

    return und_prompts, det_prompts, det_judge


def safe_answer(text: str):
    """The official parser's accepted formats (JSON, escaped JSON, Python literal) without eval()."""
    t = (text or "").strip()
    for cand in (t, t.removeprefix("```json").removesuffix("```").strip()):
        try:
            obj = json.loads(cand)
            break
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(cand)
                break
            except (ValueError, SyntaxError):
                obj = None
    if obj is None:
        obj = parse_last_json(t)
    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"]
    return obj


class PPTBench(Benchmark):
    def fetch(self):
        _official()
        import pyarrow.parquet as pq

        rows = []
        for subset, repo in SUBSETS.items():
            snap = hf_snapshot(repo, self.data_dir / "raw" / subset, allow=["*.parquet", "data/*.parquet"])
            img_dir = self.data_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(snap.rglob("*.parquet")):
                for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                    for r in batch.to_pylist():
                        sid = f"{subset}/{r['hash']}"
                        img = r.get("image")
                        if isinstance(img, dict) and img.get("bytes"):
                            (img_dir / f"{r['hash']}.png").write_bytes(img["bytes"])
                        rows.append({"id": sid, "subset": subset, "task": r.get("task"), "subcategory": r.get("subcategory"),
                                     "question": r.get("question") or r.get("description"), "options": r.get("options"),
                                     "ground_truth": r.get("ground_truth"), "hash": r["hash"],
                                     "slide_json": r.get("json_content") or r.get("json_data")})
        write_jsonl(self.data_dir / "meta.jsonl", rows)
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["ground_truth"], stratum=f"{r['subset']}/{r.get('task')}")
                for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def build(self, sample, ctx):
        und, det, _ = _official()
        d = sample.data
        slide = json.loads(d["slide_json"]) if isinstance(d["slide_json"], str) else d["slide_json"]
        if d["subset"] == "understanding":
            text = und.build_prompt(question=d["question"], options=d["options"], slide_json=slide)
            img = self.data_dir / "images" / f"{d['hash']}.png"
            content = [image_part(img), {"type": "text", "text": text}] if (img.exists() and ctx.model.vision) else text
        else:
            content = det.build_prompt(query=d["question"], slide_json=slide, subcategory=d["subcategory"])
        return ChatRequest(messages=[{"role": "user", "content": content}])

    def grade(self, sample, output, ctx):
        _, _, det_judge = _official()
        d = sample.data
        ans = safe_answer(output.text)
        if d["subset"] == "understanding":
            ok = ans is not None and str(d["ground_truth"]).strip().lower() == str(ans).strip().lower()
        else:
            if d["subcategory"] == "layout detection":
                ans = json.dumps(ans) if isinstance(ans, (dict, list)) else (None if ans is None else str(ans))
            try:
                ok = bool(det_judge.judge_answer(d["subcategory"], d["ground_truth"], ans))
            except Exception:
                ok = False
        return Grade(score=float(ok), extracted=None if ans is None else str(ans)[:300],
                     details={"subset": d["subset"], "task": d.get("task")})

    def oracle(self, sample):
        gt = sample.data["ground_truth"]
        try:
            gt = json.loads(gt) if isinstance(gt, str) and gt.strip()[:1] in "{[" else gt
        except json.JSONDecodeError:
            pass
        if sample.data.get("subcategory") == "layout detection" and isinstance(gt, dict):
            return json.dumps({k: gt[k] for k in ("left", "top", "width", "height")})  # the LAYOUT_TEMPLATE keys
        return json.dumps({"answer": gt})

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            by[f"{d.get('subset')}/{d.get('task')}"].append(r["score"])
        return {"by_task": {k: sum(v) / len(v) for k, v in sorted(by.items())}}


BENCHMARK = PPTBench
