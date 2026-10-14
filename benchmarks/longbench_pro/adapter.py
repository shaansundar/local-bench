import json
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import (VENDOR, context_budget, hf_file, load_module, read_jsonl, truncate_middle, vendor_raw_files,
                          write_jsonl)
from bench.runners.chat import resolve_max_tokens

REPO = "caskcsg/LongBench-Pro"
UPSTREAM = "caskcsg/longcontext"
REVISION = "49ee3192bd2f9176515444370348aa55db5348ce"
DEST = VENDOR / "longbench_pro"
METRIC = {
    "T1.1": "NDCG", "T1.2": "NDCG", "T2.1": "Pairwise_Accuracy", "T2.2": "Pairwise_Accuracy", "T3.1": "Accuracy",
    "T3.2": "Accuracy", "T4.1": "Summary", "T4.2": "Summary", "T5.1": "F1_Score", "T5.2": "F1_Score", "T6.1": "SubEM",
    "T6.2": "F1_Score", "T6.3": "Pairwise_Accuracy", "T7.1": "F1_Score", "T7.2": "F1_Score", "T7.3": "F1_Score",
    "T8.1": "SubEM", "T8.2": "SubEM", "T8.3": "SubEM", "T9.1": "F1_Score", "T9.2": "F1_Score", "T10.1": "SubEM",
    "T10.2": "SubEM", "T11.1": "Accuracy", "T11.2": "Accuracy",
}


def _utils():
    vendor_raw_files(UPSTREAM, REVISION, ["LongBench-Pro/modules/utils.py"], DEST)
    return load_module("longbench_pro_utils", DEST / "LongBench-Pro" / "modules" / "utils.py")


class LongBenchPro(Benchmark):
    def fetch(self):
        _utils()
        src = hf_file(REPO, "longbench_pro.json", self.data_dir)
        raw = src.read_text()
        data = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
        ctxdir = self.data_dir / "contexts"
        ctxdir.mkdir(exist_ok=True)
        meta = []
        for r in data:
            (ctxdir / f"{r['id']}.txt").write_text(r["context"])
            meta.append({k: v for k, v in r.items() if k != "context"})
        write_jsonl(self.data_dir / "meta.jsonl", meta)
        src.unlink()
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["answer"], stratum=r["token_length"])
                for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def build(self, sample, ctx):
        r = sample.data
        q = r["question_thinking"] if self.spec.options.get("question", "thinking") == "thinking" else r["question_nonthinking"]
        context = (self.data_dir / "contexts" / f"{r['id']}.txt").read_text()
        budget = context_budget(ctx, resolve_max_tokens(None, self, ctx)) - (len(q) // 3 + 200)
        return ChatRequest(messages=[{"role": "user", "content": f"{truncate_middle(context, budget)}\n\n{q}"}])

    def grade(self, sample, output, ctx):
        u = _utils()
        r = sample.data
        task = r["secondary_task"].split()[0]
        metric = METRIC.get(task, "Accuracy")
        answers = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
        pred = output.text or ""
        if not pred:
            val = 0.0
        elif metric == "Summary":
            val = u.Summary_Max_Rouge_L(answers, pred, r["language"] == "Chinese")
        else:
            val = getattr(u, metric)(answers, pred)
        return Grade(score=float(val), extracted=pred[-300:],
                     details={"task": r["primary_task"], "metric": metric, "length": r["token_length"],
                              "language": r["language"]})

    def oracle(self, sample):
        ans = sample.gold if isinstance(sample.gold, list) else [sample.gold]
        return "[Answer]\n" + "\n".join(str(a) for a in ans)

    def aggregate(self, rows):
        by_task, by_len = defaultdict(list), defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            by_task[d.get("task")].append(r["score"])
            by_len[d.get("length")].append(r["score"])
        avg = lambda v: sum(v) / len(v)
        return {"by_task": {k: avg(v) for k, v in sorted(by_task.items()) if k},
                "by_length": {k: avg(v) for k, v in by_len.items() if k}}


BENCHMARK = LongBenchPro
