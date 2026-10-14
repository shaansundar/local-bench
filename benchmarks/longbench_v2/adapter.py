import json
import re
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import context_budget, extract_mc_letter, hf_file, read_jsonl, truncate_middle, write_jsonl
from bench.runners.chat import resolve_max_tokens

REPO = "zai-org/LongBench-v2"
TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Format your response as follows: "The correct answer is (insert answer here)"."""


class LongBenchV2(Benchmark):
    def fetch(self):
        src = hf_file(REPO, "data.json", self.data_dir)
        data = json.loads(src.read_text())
        ctxdir = self.data_dir / "contexts"
        ctxdir.mkdir(exist_ok=True)
        meta = []
        for r in data:
            (ctxdir / f"{r['_id']}.txt").write_text(r["context"])
            meta.append({k: v for k, v in r.items() if k != "context"})
        write_jsonl(self.data_dir / "meta.jsonl", meta)
        src.unlink()  # the per-sample files replace the 465 MB blob
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["_id"], data=r, gold=r["answer"], stratum=f"{r['difficulty']}/{r['length']}")
                for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def build(self, sample, ctx):
        r = sample.data
        context = (self.data_dir / "contexts" / f"{r['_id']}.txt").read_text()
        budget = context_budget(ctx, resolve_max_tokens(None, self, ctx)) - 600
        prompt = TEMPLATE.format(context=truncate_middle(context, budget), question=r["question"].strip(),
                                 A=r["choice_A"], B=r["choice_B"], C=r["choice_C"], D=r["choice_D"])
        return ChatRequest(messages=[{"role": "user", "content": prompt}])

    def grade(self, sample, output, ctx):
        text = (output.text or "").replace("*", "")
        m = re.findall(r"The correct answer is \(?([A-D])\)?", text)
        letter = m[-1] if m else extract_mc_letter(text, "ABCD")
        return Grade(score=float(letter == sample.gold), extracted=letter,
                     details={"difficulty": sample.data["difficulty"], "length": sample.data["length"]})

    def oracle(self, sample):
        return f"The correct answer is ({sample.gold})"

    def aggregate(self, rows):
        by_len, by_diff = defaultdict(list), defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            by_len[d.get("length")].append(r["score"])
            by_diff[d.get("difficulty")].append(r["score"])
        avg = lambda v: sum(v) / len(v)
        return {"by_length": {k: avg(v) for k, v in by_len.items() if k},
                "by_difficulty": {k: avg(v) for k, v in by_diff.items() if k}}


BENCHMARK = LongBenchV2
