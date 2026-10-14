from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_rows, read_jsonl, write_jsonl

REPO = "cais/hle"
SYSTEM_EXACT = ("Your response should be in the following format:\nExplanation: {your explanation for your final answer}\n"
                "Exact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}")
SYSTEM_MC = ("Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}")


class HLE(Benchmark):
    def fetch(self):
        rows = hf_rows(REPO, self.data_dir / "raw", split="test")
        keep = ("id", "question", "image", "answer", "answer_type", "raw_subject", "category")
        write_jsonl(self.data_dir / "hle.jsonl", [{k: r.get(k) for k in keep} for r in rows])
        self.mark_fetched()

    def load(self):
        text_only = self.spec.options.get("text_only", True)
        return [Sample(id=r["id"], data=r, gold=r["answer"], stratum=r["category"])
                for r in read_jsonl(self.data_dir / "hle.jsonl") if not (text_only and r.get("image"))]

    def build(self, sample, ctx):
        d = sample.data
        system = SYSTEM_MC if d["answer_type"] == "multipleChoice" else SYSTEM_EXACT
        content = d["question"]
        if d.get("image"):
            content = [{"type": "text", "text": d["question"]}, {"type": "image_url", "image_url": {"url": d["image"]}}]
        return ChatRequest(messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])

    def grade(self, sample, output, ctx):
        ok, info = ctx.require_judge().equivalence(sample.data["question"], output.text, str(sample.gold))
        return Grade(score=float(ok), extracted=info.get("extracted"), judge=info,
                     details={"category": sample.data["category"]})

    def oracle(self, sample):
        return f"Explanation: omitted\nExact Answer: {sample.gold}\nConfidence: 100%"

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("category")].append(r["score"])
        return {"by_category": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = HLE
