import re
from collections import defaultdict

from bench.benchmark import Grade, Sample
from bench.common import extract_mc_letter, hf_rows
from bench.families import MultipleChoiceBenchmark

REPO = "TIGER-Lab/MMLU-Pro"
LETTERS = "ABCDEFGHIJ"


class MMLUPro(MultipleChoiceBenchmark):
    letters = LETTERS

    def fetch(self):
        hf_rows(REPO, self.data_dir, split="test")
        self.mark_fetched()

    def load(self):
        return [
            Sample(id=str(r["question_id"]), data={"question": r["question"], "choices": r["options"],
                                                   "category": r["category"]},
                   gold=r["answer"], stratum=r["category"])
            for r in hf_rows(REPO, self.data_dir, split="test")
        ]

    def prompt(self, sample):
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(sample.data["choices"]))
        return (
            f"The following is a multiple choice question about {sample.data['category']}. Think step by step and "
            'then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
            f"Question: {sample.data['question']}\nOptions:\n{opts}\nAnswer: Let's think step by step."
        )

    def grade(self, sample, output, ctx):
        n = len(sample.data["choices"])
        text = output.text or ""
        m = re.findall(r"answer is \(?([A-J])\)?", text)
        letter = m[-1] if m else extract_mc_letter(text, LETTERS[:n])
        return Grade(score=float(letter == sample.gold), extracted=letter,
                     details={"category": sample.data["category"]})

    def oracle(self, sample):
        return f"... so the answer is ({sample.gold})"

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("category")].append(r["score"])
        return {"by_category": {k: sum(v) / len(v) for k, v in sorted(by.items()) if k}}


BENCHMARK = MMLUPro
