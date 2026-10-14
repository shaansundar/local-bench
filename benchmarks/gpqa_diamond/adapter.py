import random

from bench.benchmark import Sample
from bench.common import hf_rows
from bench.families import MultipleChoiceBenchmark

REPO = "Idavidrein/gpqa"


class GPQADiamond(MultipleChoiceBenchmark):
    def fetch(self):
        hf_rows(REPO, self.data_dir, config="gpqa_diamond", split="train")
        self.mark_fetched()

    def load(self):
        out = []
        for r in hf_rows(REPO, self.data_dir, config="gpqa_diamond", split="train"):
            correct = r["Correct Answer"].strip()
            choices = [correct] + [r[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
            random.Random(f"gpqa:{r['Record ID']}").shuffle(choices)
            gold = "ABCD"[choices.index(correct)]
            out.append(Sample(id=r["Record ID"], data={"question": r["Question"].strip(), "choices": choices,
                                                        "domain": r.get("High-level domain")},
                              gold=gold, stratum=r.get("High-level domain")))
        return out


BENCHMARK = GPQADiamond
