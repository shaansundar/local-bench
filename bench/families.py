"""Reusable benchmark base classes for common shapes (math with boxed answers, multiple choice)."""

from __future__ import annotations

from bench.benchmark import Benchmark, ChatRequest, Grade, ModelOutput, RunContext, Sample
from bench.common import MATH_PROMPT, extract_mc_letter, math_equal, mc_prompt


class MathBenchmark(Benchmark):
    """Problem -> reasoning -> \\boxed{answer}; graded with math-verify against data gold."""

    prompt_template = MATH_PROMPT

    def build(self, sample: Sample, ctx: RunContext) -> ChatRequest:
        return ChatRequest(messages=[{"role": "user", "content": self.prompt_template.format(problem=sample.data["problem"])}])

    def grade(self, sample: Sample, output: ModelOutput, ctx: RunContext) -> Grade:
        ok, extracted = math_equal(str(sample.gold), output.text)
        return Grade(score=float(ok), extracted=extracted)

    def oracle(self, sample: Sample) -> str:
        return f"The answer is \\boxed{{{sample.gold}}}."


class MultipleChoiceBenchmark(Benchmark):
    """sample.data['choices'] (list[str]) + sample.gold (letter). Override prompt() if needed."""

    letters = "ABCDEFGHIJKLMNOP"

    def prompt(self, sample: Sample) -> str:
        return mc_prompt(sample.data["question"], sample.data["choices"], self.letters)

    def build(self, sample: Sample, ctx: RunContext) -> ChatRequest:
        return ChatRequest(messages=[{"role": "user", "content": self.prompt(sample)}])

    def grade(self, sample: Sample, output: ModelOutput, ctx: RunContext) -> Grade:
        n = len(sample.data["choices"])
        letter = extract_mc_letter(output.text, self.letters[:n])
        return Grade(score=float(letter == sample.gold), extracted=letter)

    def oracle(self, sample: Sample) -> str:
        return f"Reasoning omitted.\nAnswer: {sample.gold}"

    def aggregate(self, rows: list[dict]) -> dict:
        return {}


def if_aggregate(rows: list[dict]) -> dict:
    """IFEval-family summary: prompt/instruction-level, strict/loose."""
    d = [r["details"] or {} for r in rows if r.get("details")]
    if not d or "strict_all" not in d[0]:
        return {}
    inst_s = [x for r in d for x in r.get("strict_each", [])]
    inst_l = [x for r in d for x in r.get("loose_each", [])]
    return {
        "prompt_strict": sum(r["strict_all"] for r in d) / len(d),
        "prompt_loose": sum(r["loose_all"] for r in d) / len(d),
        "instruction_strict": sum(inst_s) / max(1, len(inst_s)),
        "instruction_loose": sum(inst_l) / max(1, len(inst_l)),
    }
