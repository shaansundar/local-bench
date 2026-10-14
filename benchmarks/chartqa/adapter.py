from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_snapshot, parquet_to_samples, read_jsonl, relaxed_match, short_answer, write_jsonl
from bench.models.client import image_part

REPO = "HuggingFaceM4/ChartQA"


class ChartQA(Benchmark):
    def fetch(self):
        snap = hf_snapshot(REPO, self.data_dir / "raw", allow=["data/test-*.parquet"])
        files = sorted((snap / "data").glob("test-*.parquet"))
        counter = iter(range(10**9))
        rows = parquet_to_samples(files, self.data_dir, image_col="image", id_fn=lambda r: next(counter),
                                  keep=["query", "label", "human_or_machine"])
        write_jsonl(self.data_dir / "meta.jsonl", rows)
        self.mark_fetched()

    def load(self):
        out = []
        for r in read_jsonl(self.data_dir / "meta.jsonl"):
            kind = "human" if r["human_or_machine"] in (0, "human") else "augmented"
            gold = r["label"][0] if isinstance(r["label"], list) else r["label"]
            out.append(Sample(id=r["id"], data={**r, "kind": kind}, gold=gold, stratum=kind))
        return out

    def build(self, sample, ctx):
        img = self.data_dir / "images" / f"{sample.id}.png"
        return ChatRequest(messages=[{"role": "user", "content": [
            image_part(img), {"type": "text", "text": f"{sample.data['query']}\nAnswer the question with a single word or number."}]}])

    def grade(self, sample, output, ctx):
        ans = short_answer(output.text)
        return Grade(score=float(relaxed_match(ans, sample.gold)), extracted=ans, details={"kind": sample.data["kind"]})

    def oracle(self, sample):
        return str(sample.gold)

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("kind")].append(r["score"])
        return {"by_split": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = ChartQA
