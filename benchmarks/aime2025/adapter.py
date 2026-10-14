from bench.benchmark import Sample
from bench.common import hf_rows
from bench.families import MathBenchmark

REPO = "MathArena/aime_2025"


class MathArenaSet(MathBenchmark):
    def fetch(self):
        hf_rows(REPO, self.data_dir, split="train")
        self.mark_fetched()

    def load(self):
        rows = hf_rows(REPO, self.data_dir, split="train")
        return [Sample(id=str(r["problem_idx"]), data=r, gold=str(r["answer"])) for r in rows]


BENCHMARK = MathArenaSet
