from collections import defaultdict

from bench.benchmark import Grade, Sample
from bench.common import extract_mc_letter, hf_rows
from bench.families import MultipleChoiceBenchmark

REPO = "m-a-p/SuperGPQA"


class SuperGPQA(MultipleChoiceBenchmark):
    letters = "ABCDEFGHIJ"

    def fetch(self):
        hf_rows(REPO, self.data_dir, split="train")
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["uuid"], data={"question": r["question"], "choices": r["options"], "discipline": r["discipline"],
                                           "difficulty": r["difficulty"]},
                       gold=r["answer_letter"], stratum=r["discipline"])
                for r in hf_rows(REPO, self.data_dir, split="train")]

    def grade(self, sample, output, ctx):
        n = len(sample.data["choices"])
        letter = extract_mc_letter(output.text, self.letters[:n])
        return Grade(score=float(letter == sample.gold), extracted=letter,
                     details={"discipline": sample.data["discipline"], "difficulty": sample.data["difficulty"]})

    def aggregate(self, rows):
        by_d, by_diff = defaultdict(list), defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            by_d[d.get("discipline")].append(r["score"])
            by_diff[d.get("difficulty")].append(r["score"])
        avg = lambda v: sum(v) / len(v)
        return {"by_discipline": {k: avg(v) for k, v in by_d.items() if k},
                "by_difficulty": {k: avg(v) for k, v in by_diff.items() if k}}


BENCHMARK = SuperGPQA
