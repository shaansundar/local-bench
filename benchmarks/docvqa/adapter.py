from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import anls, hf_snapshot, parquet_to_samples, read_jsonl, short_answer, write_jsonl
from bench.models.client import image_part

REPO = "lmms-lab/DocVQA"


class DocVQA(Benchmark):
    def fetch(self):
        snap = hf_snapshot(REPO, self.data_dir / "raw", allow=["DocVQA/validation-*.parquet"])
        files = sorted((snap / "DocVQA").glob("validation-*.parquet"))
        rows = parquet_to_samples(files, self.data_dir, image_col="image", id_fn=lambda r: r["questionId"],
                                  keep=["question", "answers", "question_types", "docId"])
        write_jsonl(self.data_dir / "meta.jsonl", rows)
        for f in files:
            f.unlink()
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["answers"], stratum=(r.get("question_types") or ["other"])[0])
                for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def build(self, sample, ctx):
        img = self.data_dir / "images" / f"{sample.id}.png"
        return ChatRequest(messages=[{"role": "user", "content": [
            image_part(img), {"type": "text", "text": f"{sample.data['question']}\nAnswer the question using a single word or phrase."}]}])

    def grade(self, sample, output, ctx):
        ans = short_answer(output.text)
        return Grade(score=anls(ans, sample.gold), extracted=ans)

    def oracle(self, sample):
        return sample.gold[0]


BENCHMARK = DocVQA
