"""__TITLE__ — scaffolded by `bench new __SLUG__`. See docs/NEW_BENCHMARK.md.

Fill in fetch / load / build / grade. Delete what you don't use. Keep downloaded data under
self.data_dir (never execute anything from it) and pin every upstream revision.
"""

from bench.benchmark import AgentTask, Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_rows  # also: hf_file, hf_snapshot, download, vendor_repo, extract_mc_letter, math_equal, ...


class __CLASS__(Benchmark):
    def fetch(self):
        # Download into self.data_dir (idempotent), then mark it fetched.
        hf_rows("ORG/DATASET", self.data_dir, split="test")
        self.mark_fetched()

    def load(self) -> list[Sample]:
        # One Sample per scored item. ids must be stable strings (they key frozen subsets and results).
        rows = hf_rows("ORG/DATASET", self.data_dir, split="test")
        return [Sample(id=str(r["id"]), data=r, gold=r["answer"], stratum=r.get("category")) for r in rows]

    def build(self, sample, ctx):
        # runner: chat -> return a ChatRequest; runner: oc -> return an AgentTask.
        # (runner: native -> implement execute(sample, ctx) -> ModelOutput instead.)
        if self.spec.runner == "oc":
            return AgentTask(prompt=sample.data["question"])
        return ChatRequest(messages=[{"role": "user", "content": sample.data["question"]}])

    def grade(self, sample, output, ctx) -> Grade:
        # Deterministic grading preferred. For judge grading: ctx.require_judge().equivalence(...)
        answer = (output.text or "").strip()
        return Grade(score=float(answer == str(sample.gold)), extracted=answer)

    def oracle(self, sample):
        # A response that must score ~1.0; used by `bench validate`. Return None if impossible.
        return str(sample.gold)


BENCHMARK = __CLASS__
