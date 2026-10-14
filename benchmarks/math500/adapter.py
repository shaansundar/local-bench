from bench.benchmark import Sample
from bench.common import hf_rows
from bench.families import MathBenchmark


class Math500(MathBenchmark):
    def fetch(self):
        hf_rows("HuggingFaceH4/MATH-500", self.data_dir, split="test")
        self.mark_fetched()

    def load(self):
        rows = hf_rows("HuggingFaceH4/MATH-500", self.data_dir, split="test")
        return [Sample(id=r["unique_id"], data=r, gold=r["answer"], stratum=r["subject"]) for r in rows]


BENCHMARK = Math500
